import asyncio
import json
import os
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from aiortc import RTCIceServer, RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, RTCConfiguration
from aiortc.contrib.media import MediaRelay
from twilio.rest import Client

from server.model_controller import ModelController
from server.predictions_config import PredictionsConfig
from server.model_configs import ModelsConfig
from server.utils import tprint

os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"

DEV_ENV = os.environ.get("DEV_ENV", "false").lower() == "true"
if DEV_ENV:
    tprint(f"INIT: using dev env")


TWILIO_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")

rtc_config = RTCConfiguration(
    iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])]
)

MODELS_LIST = [m.value.name for m in ModelsConfig]


app = FastAPI()
relay = MediaRelay()


class VideoTransformTrack(VideoStreamTrack):
    def __init__(self, track, pc):
        super().__init__()
        self.track = track
        self.pc = pc

        self.is_processing = False

        self.predictions_config = PredictionsConfig(
            model_name=ModelsConfig.S26_OPENVINO_800.value.name,
            task="track",
            conf=0.4
        )

        self.model_controller = ModelController(self.predictions_config)

        tprint("INIT: VideoTransformTrack initialized")

    async def update_predictions_config(self, data: any):
        updated_predictions_config = PredictionsConfig(
            model_name=data.get(
                "model", video_track.predictions_config.model_name
            ),
            task=data.get(
                "task", video_track.predictions_config.task
            ),
            conf=float(
                data.get(
                    "conf", video_track.predictions_config.conf)
            )
        )

        if updated_predictions_config != self.predictions_config:
            self.predictions_config = updated_predictions_config

            await self.model_controller.reload_model(self.predictions_config)

            tprint(f"RELOAD: Received new config: {self.predictions_config}")
        else:
            tprint("RELOAD: Received same config, no reload needed.")

    async def recv(self):
        frame = await self.track.recv()

        if self.is_processing:
            return frame

        self.is_processing = True

        img = frame.to_ndarray(format="bgr24")
        asyncio.create_task(self.process_frame(img))

        return frame

    async def process_frame(self, img):
        try:
            loop = asyncio.get_event_loop()

            channel = getattr(self.pc, "active_channel", None)

            if channel and channel.readyState == "open":
                results = await loop.run_in_executor(
                    None,
                    lambda: self.model_controller.get_predictions(img)
                )

                channel.send(json.dumps(results))
        except Exception as e:
            tprint(f"ERROR: process_frame: {e}")
        finally:
            self.is_processing = False


video_track = None


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


@app.get("/models")
async def list_models():
    return {"models": MODELS_LIST}


@app.post("/offer")
async def offer(request: Request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection(configuration=rtc_config)
    pc.active_channel = None

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
                    if not video_track:
                        tprint("ERROR: No video track available yet.")

                        return

                    tprint(f"RELOAD: Received config: {data}")

                    await video_track.update_predictions_config(data)

            except Exception as e:
                tprint(f"ERROR: Error parsing message: {e}")

    @pc.on("track")
    def on_track(track):
        global video_track

        tprint(f"INIT: Track received: {track.kind}")

        if track.kind == "video":
            video_track = VideoTransformTrack(relay.subscribe(track), pc)

            async def force_consume():
                try:
                    while True:
                        await video_track.recv()

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
    uvicorn.run(app, host="0.0.0.0", port=7860)
