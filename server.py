import asyncio
import json
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, RTCConfiguration, RTCIceServer
from aiortc.contrib.media import MediaRelay
from ultralytics import YOLO
from enum import Enum
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    size: int
    name: str
    path: Path


MODEL_BASE_PATH = Path("models")


class ModelsConfig(Enum):
    S26_OPENVINO_640 = ModelConfig(
        size=640, name="S26_OPENVINO_640", path=MODEL_BASE_PATH / "best_openvino_model_640_s")
    S26_OPENVINO_800 = ModelConfig(
        size=800, name="S26_OPENVINO_800", path=MODEL_BASE_PATH / "best_openvino_model_800_s")
    N26_OPENVINO_800 = ModelConfig(
        size=800, name="N26_OPENVINO_800", path=MODEL_BASE_PATH / "best_openvino_model_800_n")
    N26_OPENVINO_640 = ModelConfig(
        size=640, name="N26_OPENVINO_640", path=MODEL_BASE_PATH / "best_openvino_model_640_n")


MODELS_LIST = [m.value.name for m in ModelsConfig]

current_config = {
    "model_name": ModelsConfig.S26_OPENVINO_800.value.name,
    "task": "track",
    "conf": 0.4
}

model_config = ModelsConfig[current_config['model_name']].value
model = YOLO(model_config.path, task='segment')

rtc_config = RTCConfiguration(
    iceServers=[
        RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
        RTCIceServer(
            urls=["turn:openrelay.metered.ca:80", "turn:openrelay.metered.ca:443",
                  "turn:openrelay.metered.ca:443?transport=tcp"],
            username="openrelayproject",
            credential="openrelayproject"
        )
    ]
)

app = FastAPI()
relay = MediaRelay()


class VideoTransformTrack(VideoStreamTrack):
    def __init__(self, track, pc):
        super().__init__()
        self.track = track
        self.pc = pc
        self.is_processing = False

    async def recv(self):
        frame = await self.track.recv()
        if self.is_processing:
            return frame

        self.is_processing = True
        img = frame.to_ndarray(format="bgr24")
        asyncio.create_task(self.process_frame(img))
        return frame

    def get_results(self, img):
        global model, model_config

        if current_config["task"] == "track":
            return model.track(
                img,
                task="segment",
                imgsz=model_config.size,
                verbose=False,
                conf=current_config["conf"],
                iou=0.5,
                persist=True,
                tracker="botsort.yaml",
                half=True
            )[0]

        return model.predict(
            img,
            task="segment",
            imgsz=model_config.size,
            verbose=False,
            conf=current_config["conf"],
            iou=0.5,
            half=True
        )[0]

    async def process_frame(self, img):
        try:
            h, w = img.shape[:2]
            loop = asyncio.get_event_loop()

            results = await loop.run_in_executor(
                None,
                lambda: self.get_results(img)
            )

            channel = getattr(self.pc, "active_channel", None)
            if channel and channel.readyState == "open":
                predictions = []
                if results.boxes:
                    for i, box in enumerate(results.boxes):
                        if current_config["task"] == "predict":
                            t_id = -1
                        else:
                            t_id = int(box.id[0]) if box.id is not None else -1

                        coords = box.xyxy[0].tolist()

                        norm_box = [
                            coords[0] / w, coords[1] / h,
                            coords[2] / w, coords[3] / h
                        ]

                        norm_mask = []
                        if results.masks:
                            norm_mask = [
                                [p[0]/w, p[1]/h]
                                for p in results.masks.xy[i]]

                        predictions.append({
                            "box": norm_box,
                            "mask": norm_mask,
                            "label": results.names[int(box.cls[0])],
                            "id": t_id,
                            "conf": round(float(box.conf[0]), 2)
                        })

                payload = {
                    "data": predictions,
                    "metrics": {
                        "pre": round(results.speed['preprocess'], 1),
                        "inf": round(results.speed['inference'], 1),
                        "post": round(results.speed['postprocess'], 1),
                        "total": round(sum(results.speed.values()), 1)
                    }
                }
                channel.send(json.dumps(payload))
        except Exception as e:
            print(f"Error in process_frame: {e}")
        finally:
            self.is_processing = False


async def async_reload_model(target_model_name):
    global model, model_config
    try:
        loop = asyncio.get_event_loop()
        new_config = ModelsConfig[target_model_name].value

        new_model = await loop.run_in_executor(
            None,
            lambda: YOLO(new_config.path, task='segment')
        )

        model_config = new_config
        model = new_model
        print(f"Model reloaded successfully: {target_model_name}")
    except Exception as e:
        print(f"Failed to reload model: {e}")


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
        print("Data channel opened!")

        @channel.on("message")
        def on_message(message):
            try:
                data = json.loads(message)
                if data.get("type") == "config":
                    updated_model = data.get(
                        "model", current_config["model_name"])

                    if updated_model != current_config["model_name"]:
                        current_config["model_name"] = updated_model
                        loop = asyncio.get_event_loop()
                        loop.create_task(async_reload_model(updated_model))

                    current_config["task"] = data.get(
                        "task", current_config["task"])

                    current_config["conf"] = float(
                        data.get("conf", current_config["conf"]))

                    print(f"Updated config from client: {current_config}")

                if data.get("type") == "ping":
                    channel.send(json.dumps(
                        {"type": "pong", "timestamp": data.get("timestamp")}))
            except Exception as e:
                print(f"Error parsing client message: {e}")

    @pc.on("track")
    def on_track(track):
        if track.kind == "video":
            local_video = VideoTransformTrack(relay.subscribe(track), pc)
            pc.addTrack(local_video)

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
