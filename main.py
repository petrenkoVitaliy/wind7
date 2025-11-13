from __future__ import annotations

import asyncio
import contextlib
import json
import multiprocessing
import os
from typing import Any

import psutil
import torch
from aiortc import (
    RTCConfiguration,
    RTCDataChannel,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)
from aiortc.contrib.media import MediaRelay
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from pynvml import (
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetUtilizationRates,
    nvmlInit,
)
from twilio.rest import Client

from server.configs.models import ModelsConfig
from server.stream_infer_controller import PeerConnectionState, StreamInferController
from server.utils.formatter import L, tprint

TWILIO_SID: str | None = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_TOKEN: str | None = os.environ.get("TWILIO_AUTH_TOKEN")
NUM_THREADS: str = os.environ.get("NUM_THREADS", "1")

DEV_ENV: bool = os.environ.get("DEV_ENV", "false").lower() == "true"
if DEV_ENV:
    tprint(L.INIT_DEV_ENV)

IS_VAST: bool = os.environ.get("VAST_CONTAINERLABEL") is not None

num_threads: str = (
    str(multiprocessing.cpu_count()) if (DEV_ENV or IS_VAST) else NUM_THREADS
)

tprint(L.INIT_NUM_THREADS, num=num_threads)

os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OMP_NUM_THREADS"] = num_threads
os.environ["MKL_NUM_THREADS"] = num_threads
os.environ["OPENBLAS_NUM_THREADS"] = num_threads


rtc_config: RTCConfiguration = RTCConfiguration(
    iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])]
)

MODELS_LIST: list[dict[str, str]] = [
    {"name": m.value.name, "description": m.value.description} for m in ModelsConfig
]


app: FastAPI = FastAPI()
relay: MediaRelay = MediaRelay()

peer_connections: set[PeerConnectionState] = set()


with contextlib.suppress(Exception):
    nvmlInit()


def get_system_stats() -> dict[str, str]:
    stats: dict[str, str] = {
        "cpu": f"{psutil.cpu_percent(interval=None)}%",
        "ram": f"{psutil.virtual_memory().percent}%",
        "gpu": "None",
        "gpu_load": "0%",
        "vram": "0%",
    }
    if torch.cuda.is_available():
        stats["gpu"] = torch.cuda.get_device_name(0)
        try:
            h = nvmlDeviceGetHandleByIndex(0)
            res = nvmlDeviceGetUtilizationRates(h)
            mem = nvmlDeviceGetMemoryInfo(h)
            stats["gpu_load"] = f"{res.gpu}%"
            stats["vram"] = (
                f"{int(int(mem.used) / 1024**2)}MB / {int(int(mem.total) / 1024**2)}MB"
            )
        except Exception:
            pass
    return stats


@app.get("/stats")
async def get_stats() -> dict[str, dict[str, str]]:
    return {"stats": get_system_stats()}


@app.get("/models")
async def list_models() -> dict[str, list[dict[str, str]] | dict[str, str]]:
    return {"models": MODELS_LIST, "stats": get_system_stats()}


@app.get("/ice-config")
async def get_ice_config() -> dict[str, list[dict[str, str | list[str]]]]:
    try:
        if DEV_ENV:
            tprint(L.INIT_DEV_ICE)
            return {"iceServers": [{"urls": "stun:stun.l.google.com:19302"}]}

        if TWILIO_SID is None or TWILIO_TOKEN is None:
            tprint(L.ERROR_TWILIO_CREDS)
            return {"iceServers": [{"urls": "stun:stun.l.google.com:19302"}]}

        client = Client(TWILIO_SID, TWILIO_TOKEN)
        token = client.tokens.create()

        ice_servers: Any = token.ice_servers
        return {
            "iceServers": ice_servers,
        }
    except Exception as e:
        tprint(L.ERROR_TWILIO, err=e)
        return {"iceServers": [{"urls": "stun:stun.l.google.com:19302"}]}


@app.post("/offer")
async def offer(request: Request) -> dict[str, str]:
    params: Any = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection(configuration=rtc_config)
    connection_state = PeerConnectionState(pc=pc)
    peer_connections.add(connection_state)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange() -> None:
        tprint(L.INIT_CONN_STATE, state=pc.connectionState)
        if pc.connectionState in ["failed", "closed"]:
            if connection_state.video_track is not None:
                await connection_state.video_track.close()
            peer_connections.discard(connection_state)

    @pc.on("datachannel")
    def on_datachannel(channel: RTCDataChannel) -> None:
        connection_state.active_channel = channel
        tprint(L.INIT_DATACHANNEL)

        @channel.on("message")
        async def on_message(message: str) -> None:
            try:
                data: Any = json.loads(message)

                if data.get("type") == "ping":
                    channel.send(
                        json.dumps({"type": "pong", "timestamp": data.get("timestamp")})
                    )
                    return

                if data.get("type") == "config":
                    if not connection_state.video_track:
                        tprint(L.ERROR_NO_VIDEO_TRACK)
                        return

                    tprint(L.SEP)
                    tprint(L.RELOAD_CONFIG, data=data)

                    await connection_state.video_track.update_predictions_config(data)

            except Exception as e:
                tprint(L.ERROR_PARSE_MSG, err=e)

    @pc.on("track")
    def on_track(track: Any) -> None:
        tprint(L.INIT_TRACK_RECEIVED, kind=track.kind)

        if track.kind == "video":
            controller = StreamInferController(relay.subscribe(track), connection_state)
            connection_state.video_track = controller

            async def force_consume() -> None:
                try:
                    while True:
                        await controller.recv()

                except Exception as e:
                    tprint(L.ERROR_CONSUMPTION, err=e)

            connection_state._consume_task = asyncio.create_task(force_consume())

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


@app.get("/")
async def index() -> FileResponse:
    return FileResponse("client/index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
