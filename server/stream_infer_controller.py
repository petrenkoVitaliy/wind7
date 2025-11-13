from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, cast

import orjson
from aiortc import MediaStreamTrack, RTCDataChannel, RTCPeerConnection, VideoStreamTrack
from av import VideoFrame

from server.configs.predictions import PredictionsConfig
from server.model_adapter import ModelAdapter
from server.utils import L, tprint


@dataclass
class PeerConnectionState:
    pc: RTCPeerConnection
    active_channel: RTCDataChannel | None = None
    video_track: StreamInferController | None = None
    consume_task: asyncio.Task[None] | None = None

    def __hash__(self) -> int:
        return id(self)


class StreamInferController(VideoStreamTrack):
    track: MediaStreamTrack
    state: PeerConnectionState
    is_processing: bool
    predictions_config: PredictionsConfig
    model_adapter: ModelAdapter | None
    _frames_received: int
    _frames_dropped: int
    _total_inference_ms: float
    _inference_count: int
    _process_task: asyncio.Task[None] | None

    def __init__(
        self,
        track: MediaStreamTrack,
        state: PeerConnectionState,
        default_model_name: str = "",
    ) -> None:
        super().__init__()
        self.track: MediaStreamTrack = track
        self.state: PeerConnectionState = state

        self.is_processing: bool = False

        self.predictions_config: PredictionsConfig = PredictionsConfig(
            model_name=default_model_name,
            task="track",
            conf=0.4,
            retina_masks=False,
            tracker="bytetrack",
        )

        self.model_adapter: ModelAdapter | None = (
            ModelAdapter(self.predictions_config) if default_model_name else None
        )

        self._frames_received: int = 0
        self._frames_dropped: int = 0
        self._total_inference_ms: float = 0.0
        self._inference_count: int = 0
        self._process_task: asyncio.Task[None] | None = None

        tprint(L.INIT_VIDEO_TRANSFORM_TRACK)

    async def process_frame(self, frame: VideoFrame) -> None:
        if self.is_processing:
            self._frames_dropped += 1
            return

        self.is_processing = True
        t0 = time.perf_counter()
        try:
            channel = self.state.active_channel

            def _inference_task() -> dict[str, Any] | None:
                img = frame.to_ndarray(format="bgr24")
                if self.model_adapter is None:
                    return None
                return self.model_adapter.get_predictions(img)

            if (
                channel
                and channel.readyState == "open"
                and self.model_adapter is not None
            ):
                loop = asyncio.get_running_loop()
                results = await loop.run_in_executor(None, _inference_task)

                if results is not None:
                    channel.send(orjson.dumps(results).decode())

        except Exception as e:
            tprint(L.ERROR_PROCESS_FRAME, err=e, exc_info=True)
        finally:
            self.is_processing = False
            elapsed = (time.perf_counter() - t0) * 1000
            self._total_inference_ms += elapsed
            self._inference_count += 1
            self._frames_received += 1

    async def update_predictions_config(self, data: Any) -> None:
        updated_predictions_config = PredictionsConfig(
            model_name=data.get("model", self.predictions_config.model_name),
            task=data.get("task", self.predictions_config.task),
            conf=float(data.get("conf", self.predictions_config.conf)),
            retina_masks=data.get("retina_masks", self.predictions_config.retina_masks),
            tracker=data.get("tracker", self.predictions_config.tracker),
        )

        if updated_predictions_config != self.predictions_config:
            self.predictions_config = updated_predictions_config

            tprint(L.RELOAD_CONFIG_NEW, config=self.predictions_config)

            if self.model_adapter is None:
                self.model_adapter = ModelAdapter(self.predictions_config)
            else:
                await self.model_adapter.reload_model(self.predictions_config)
        else:
            tprint(L.RELOAD_SAME_CONFIG)

    async def recv(self) -> VideoFrame:
        frame = cast(VideoFrame, await self.track.recv())
        if not self.is_processing:
            self._process_task = asyncio.create_task(self.process_frame(frame))
        return frame

    async def close(self) -> None:
        if self._process_task is not None:
            self._process_task.cancel()
            self._process_task = None

    def get_stats(self) -> dict[str, Any]:
        avg_ms = (
            self._total_inference_ms / self._inference_count
            if self._inference_count
            else 0
        )
        return {
            "frames_received": self._frames_received,
            "frames_dropped": self._frames_dropped,
            "avg_inference_ms": round(avg_ms, 1),
        }
