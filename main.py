import asyncio
import json
import multiprocessing
import os
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from aiortc import RTCIceServer, RTCPeerConnection, RTCSessionDescription, RTCConfiguration
from aiortc.contrib.media import MediaRelay
from twilio.rest import Client
import psutil
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates, nvmlDeviceGetMemoryInfo, nvmlShutdown
import torch

from server.model_configs import ModelsConfig
from server.stream_infer_controller import StreamInferController
from server.utils import tprint


TWILIO_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TWILIO_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
NUM_THREADS = os.environ.get("NUM_THREADS", "1")


DEV_ENV = os.environ.get("DEV_ENV", "false").lower() == "true"
if DEV_ENV:
    tprint("INIT: using dev env")

num_threads = str(multiprocessing.cpu_count()) if DEV_ENV else NUM_THREADS

os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OMP_NUM_THREADS"] = num_threads
os.environ["MKL_NUM_THREADS"] = num_threads
os.environ["OPENBLAS_NUM_THREADS"] = num_threads


rtc_config = RTCConfiguration(
    iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])]
)

MODELS_LIST = [m.value.name for m in ModelsConfig]


app = FastAPI()
relay = MediaRelay()

peer_connections = set()


def get_system_stats():
    stats = {
        "cpu": f"{psutil.cpu_percent(interval=None)}%",
        "ram": f"{psutil.virtual_memory().percent}%",
        "gpu": "None", "gpu_load": "0%", "vram": "0%"
    }
    if torch.cuda.is_available():
        stats["gpu"] = torch.cuda.get_device_name(0)
        try:
            nvmlInit()
            h = nvmlDeviceGetHandleByIndex(0)
            res = nvmlDeviceGetUtilizationRates(h)
            mem = nvmlDeviceGetMemoryInfo(h)
            stats["gpu_load"] = f"{res.gpu}%"
            stats["vram"] = f"{int(mem.used / 1024**2)}MB / {int(mem.total / 1024**2)}MB"
            nvmlShutdown()
        except:
            pass
    return stats


@app.get("/stats")
async def get_stats():
    return {"stats": get_system_stats()}


@app.get("/models")
async def list_models():
    return {"models": MODELS_LIST, "stats": get_system_stats()}


@app.get("/ice-config")
async def get_ice_config():
    try:
        if DEV_ENV:
            tprint("INIT: Using dev ICE config with public STUN server")
            return {"iceServers": [{"urls": "stun:stun.l.google.com:19302"}]}

        client = Client(TWILIO_SID, TWILIO_TOKEN)
        token = client.tokens.create()

        return {
            "iceServers": token.ice_servers,
        }
    except Exception as e:
        tprint(f"ERROR: Twilio: {e}")
        return {"iceServers": [{"urls": "stun:stun.l.google.com:19302"}]}


@app.post("/offer")
async def offer(request: Request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection(configuration=rtc_config)
    pc.active_channel = None
    pc.video_track = None
    peer_connections.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        tprint(f"Connection state is {pc.connectionState}")
        if pc.connectionState in ["failed", "closed"]:
            peer_connections.discard(pc)

    @pc.on("datachannel")
    def on_datachannel(channel):
        pc.active_channel = channel
        tprint("INIT: Data channel opened!")

        @channel.on("message")
        async def on_message(message):
            # tprint(f"PROCESS: Message from client: {message}")

            try:
                data = json.loads(message)

                if data.get("type") == "ping":
                    channel.send(
                        json.dumps(
                            {
                                "type": "pong",
                                "timestamp": data.get("timestamp")
                            }
                        )
                    )
                    return

                if data.get("type") == "config":
                    if not pc.video_track:
                        tprint("ERROR: No video track available yet.")

                        return

                    tprint(f"RELOAD: Received config: {data}")

                    await pc.video_track.update_predictions_config(data)

            except Exception as e:
                tprint(f"ERROR: Error parsing message: {e}")

    @pc.on("track")
    def on_track(track):
        tprint(f"INIT: Track received: {track.kind}")

        if track.kind == "video":
            pc.video_track = StreamInferController(relay.subscribe(track), pc)

            async def force_consume():
                try:
                    while True:
                        await pc.video_track.recv()

                except Exception as e:
                    tprint(f"ERROR: Consumption stopped: {e}")

            asyncio.create_task(force_consume())

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


@app.get("/")
async def index():
    return FileResponse('client/index.html')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
