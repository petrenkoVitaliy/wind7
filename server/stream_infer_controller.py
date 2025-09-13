import asyncio
import json
import concurrent
from aiortc import VideoStreamTrack


from server.model_adapter import ModelAdapter
from server.predictions_config import PredictionsConfig
from server.model_configs import ModelsConfig
from server.utils import tprint


class StreamInferController(VideoStreamTrack):
    def __init__(self, track, pc):
        super().__init__()
        self.track = track
        self.pc = pc

        self.is_processing = False

        self.predictions_config = PredictionsConfig(
            model_name=ModelsConfig.S26_OPENVINO_800.value.name,
            task="track",
            conf=0.4,
            retina_masks=False
        )

        self.model_adapter = ModelAdapter(self.predictions_config)

        self.inference_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1
        )

        tprint("INIT: VideoTransformTrack initialized")

    async def process_frame(self, frame):
        if self.is_processing:
            return

        self.is_processing = True
        try:
            channel = getattr(self.pc, "active_channel", None)

            def _inference_task():
                img = frame.to_ndarray(format="bgr24")
                return self.model_adapter.get_predictions(img)

            if channel and channel.readyState == "open":
                loop = asyncio.get_running_loop()

                results = await loop.run_in_executor(
                    self.inference_executor,
                    _inference_task
                )

                if results is not None:
                    channel.send(json.dumps(results))

        except Exception as e:
            tprint(f"ERROR: process_frame: {e}")
        finally:
            self.is_processing = False

    async def update_predictions_config(self, data: any):
        updated_predictions_config = PredictionsConfig(
            model_name=data.get(
                "model", self.predictions_config.model_name
            ),
            task=data.get(
                "task", self.predictions_config.task
            ),
            conf=float(
                data.get(
                    "conf", self.predictions_config.conf)
            ),
            retina_masks=data.get(
                "retina_masks", self.predictions_config.retina_masks)
        )

        if updated_predictions_config != self.predictions_config:
            self.predictions_config = updated_predictions_config

            tprint(f"RELOAD: Config is new: {self.predictions_config}")

            await self.model_adapter.reload_model(self.predictions_config)
        else:
            tprint("RELOAD: Same config, no reload needed.")

    async def recv(self):
        frame = await self.track.recv()

        asyncio.create_task(self.process_frame(frame))

        return frame
