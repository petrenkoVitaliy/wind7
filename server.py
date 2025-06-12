import os
import json
import asyncio
import cv2
import numpy as np
import time
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, RTCIceServer, RTCConfiguration
from aiortc.rtcrtpsender import RTCRtpSender
from ultralytics import YOLO

# Обмежуємо кількість потоків, щоб не вбити CPU на Hugging Face
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

app = FastAPI()

# --- Конфігурація моделі ---
MODEL_PATH = "S26_OPENVINO_640"  # Твоя папка з OpenVINO моделлю
try:
    model = YOLO(MODEL_PATH, task='segment')
    # Прогрів моделі (Warmup)
    model.predict(np.zeros((320, 320, 3), dtype=np.uint8),
                  imgsz=320, verbose=False)
    print("✅ Model loaded and warmed up!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# --- Налаштування WebRTC (твоя розширена конфігурація) ---
TURN_USERNAME = os.environ.get("TURN_USERNAME", "fake")
TURN_CREDENTIAL = os.environ.get("TURN_CREDENTIAL", "fake")

ICE_SERVERS = [
    RTCIceServer(urls=["stun:stun.relay.metered.ca:80"]),
    RTCIceServer(
        urls=[
            "turn:global.relay.metered.ca:80",
            "turn:global.relay.metered.ca:443",
            "turn:global.relay.metered.ca:80?transport=tcp",
            "turns:global.relay.metered.ca:443?transport=tcp"
        ],
        username=TURN_USERNAME,
        credential=TURN_CREDENTIAL
    )
]
RTC_CONFIG = RTCConfiguration(iceServers=ICE_SERVERS)

# Глобальні налаштування для керування з клієнта
current_config = {
    "model_name": "S26_OPENVINO_640",
    "conf": 0.4,
    "task": "track"
}


class VideoTransformTrack(VideoStreamTrack):
    def __init__(self, track, pc):
        super().__init__()
        self.track = track
        self.pc = pc
        self.data_channel = None
        self.is_processing = False  # Прапорець, щоб не забивати чергу

    async def recv(self):
        frame = await self.track.recv()

        # Якщо AI вже зайнятий або Data Channel не готовий - просто пропускаємо кадр на вихід
        if self.is_processing or not self.data_channel or self.data_channel.readyState != "open":
            return frame

        # Починаємо обробку
        self.is_processing = True
        img = frame.to_ndarray(format="bgr24")

        # Запускаємо важкі обчислення в окремому потоці, щоб не блокувати Event Loop
        asyncio.get_event_loop().run_in_executor(None, self.process_ai, img)

        return frame

    def process_ai(self, img):
        try:
            # Отримуємо результати (зменшуємо imgsz для швидкості на HF)
            results = model.track(
                img,
                persist=True,
                conf=current_config["conf"],
                imgsz=320,
                verbose=False
            )[0]

            predictions = []
            if results.boxes:
                for i, box in enumerate(results.boxes):
                    coords = box.xyxy[0].tolist()
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    obj_id = int(box.id[0]) if box.id is not None else i

                    predictions.append({
                        "id": obj_id,
                        "bbox": coords,
                        "class": results.names[cls_id],
                        "conf": conf
                    })

            # Відправляємо результати назад клієнту через Data Channel
            if self.data_channel and self.data_channel.readyState == "open":
                payload = json.dumps({"data": predictions})
                self.data_channel.send(payload)

        except Exception as e:
            print(f"AI Error: {e}")
        finally:
            self.is_processing = False


@app.post("/offer")
async def offer(request: Request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection(configuration=RTC_CONFIG)

    @pc.on("datachannel")
    def on_datachannel(channel):
        print("✅ Data channel opened!")

        @channel.on("message")
        def on_message(message):
            global current_config
            try:
                msg_data = json.loads(message)
                current_config.update(msg_data)
                print(f"⚙️ Config updated: {current_config}")
            except:
                pass

    @pc.on("track")
    def on_track(track):
        if track.kind == "video":
            print("🎥 Video track received!")
            local_video = VideoTransformTrack(track, pc)
            pc.addTrack(local_video)

            # Прив'язуємо канал до треку для відправки результатів
            @pc.on("datachannel")
            def set_dc(channel):
                local_video.data_channel = channel

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        print(f"❄️ ICE Connection State: {pc.iceConnectionState}")
        if pc.iceConnectionState == "failed":
            await pc.close()

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return JSONResponse({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    })


@app.get("/ice-config")
async def get_ice_config():
    # Повертаємо ту саму структуру, що хоче клієнт
    return {
        "iceServers": [
            {"urls": "stun:stun.relay.metered.ca:80"},
            {
                "urls": [
                    "turn:global.relay.metered.ca:80",
                    "turn:global.relay.metered.ca:443",
                    "turn:global.relay.metered.ca:80?transport=tcp",
                    "turns:global.relay.metered.ca:443?transport=tcp"
                ],
                "username": TURN_USERNAME,
                "credential": TURN_CREDENTIAL
            }
        ]
    }


@app.get("/")
async def index():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

if __name__ == "__main__":
    import uvicorn
    # Порт 7860 обов'язковий для Hugging Face
    uvicorn.run(app, host="0.0.0.0", port=7860, loop="asyncio")
