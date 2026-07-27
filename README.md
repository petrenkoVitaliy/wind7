---
title: Wind7
emoji: 📸
colorFrom: green
colorTo: purple
sdk: docker
pinned: false
---

# 📸 Wind7 - AI Client-Server Segmentation: Real-Time WebRTC Inference Engine

High-performance, low latency real-time object detection, segmentation and tracking system. Built with WebRTC for seamless video streaming and bidirectional data exchange.

https://github.com/user-attachments/assets/b4bffb0f-4549-483b-afb8-f1a5a187c3e8

## 📌 About the Project

**Wind7** is a client-server application that streams video from a client's camera to a server via WebRTC, asynchronously processes frames using ML models (YOLO, ONNX, TensorRT, OpenVINO), and instantly returns inference results (bounding boxes, segmentation polygons, tracking IDs) via WebRTC DataChannels.

Designed for GPU instances (including `vast.ai`), the project supports **hot-swapping of models and trackers** without interrupting the video stream.

## 🚀 Key Features

- **Low Latency:** Utilizes WebRTC (`aiortc`) instead of HTTP/WebSocket for video streaming and detection metadata transfer.
- **Render Synchronization (Sync Mode):** The client-side frame buffer synchronizes server results with the corresponding historical frame, eliminating visual tearing.
- **Zero-Downtime Hot Swapping:** Change configurations (YOLO ↔ ONNX), trackers, or confidence levels on the fly. Models load in separate threads without blocking the Event Loop.
- **Custom Tracking Algorithms:** Custom implementations of **ByteTrack** and **BoT-SORT** (featuring Global Motion Compensation via OpenCV optical flow and Kalman Filters).
- **Resource Monitoring:** Real-time integration with `pynvml` and `psutil` to display CPU, RAM, GPU, and VRAM metrics on the client.

## 🛠 Technology Stack

**Backend & ML:**

- **Core:** Python 3.10+, FastAPI, Asyncio, Multiprocessing
- **WebRTC:** `aiortc`, `av` (PyAV)
- **Computer Vision:** OpenCV, Ultralytics (YOLO)
- **Inference Engines:** PyTorch, ONNX Runtime (CUDA / OpenVINO providers), TensorRT
- **Math & Tracking:** NumPy, SciPy (Linear Sum Assignment), Kalman Filters

**Frontend:**

- HTML5 Canvas API, WebRTC MediaStreams
- Tailwind CSS, Alpine.js (zero-build reactive UI)

**DevOps & Infrastructure:**

- Docker (including a specialized image for `vast.ai` + `ngrok`)
- Linter / Formatter: Ruff, Pyright (Strict Typing)

## ⚙️ Installation & Setup

### 1. Local Run (via Docker)

```bash
# Clone the repository
git clone [https://github.com/your-username/wind7.git](https://github.com/your-username/wind7.git)
cd wind7

# Build and run the container
docker build -t wind7 .
docker run --gpus all -p 8000:7860 -it wind7
```

## 💻 Local Setup (Without Docker)

**Prerequisites:**

- Python 3.10+
- _Optional but recommended:_ NVIDIA GPU with installed CUDA Toolkit and cuDNN for hardware-accelerated inference.

**1. Clone the repository:**

```bash
git clone [https://github.com/your-username/wind7.git](https://github.com/your-username/wind7.git)
cd wind7
```

**2. Create and activate a virtual environment:**

```bash
python -m venv venv

# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

**3. Install dependencies:**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**4. Set up environment variables:**

Create a .env file in the root directory. If running locally on the same network, Twilio configuration can be omitted by setting DEV_ENV.

```bash
DEV_ENV=true
# Optional: STUN/TURN servers for external networks
# TWILIO_ACCOUNT_SID=your_sid
# TWILIO_AUTH_TOKEN=your_token
```

**5. Start the server:**

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open your browser and navigate to http://localhost:8000 or http://127.0.0.1:8000. Make sure to grant camera and microphone permissions when prompted.

## 📁 Detailed Project Structure

```text
ai-vision-pro/
├── main.py                     # Entry point: FastAPI app, CORS setup, and WebRTC signaling (SDP offer/answer).
├── requirements.txt            # Project dependencies (FastAPI, aiortc, ultralytics, onnxruntime, etc.).
├── .env                        # Environment variables configuration (Twilio STUN/TURN credentials).
├── server/                     # Backend and ML core.
│   ├── stream_infer_controller.py # Manages WebRTC MediaStreamTrack, frame queues, and DataChannel messaging.
│   ├── model_handler/          # Inference logic utilizing the Strategy pattern.
│   │   ├── base.py             # Abstract base interface for model adapters.
│   │   ├── yolo_handler.py     # Implementation for PyTorch-based Ultralytics models.
│   │   └── onnx_handler.py     # High-performance implementation for ONNX Runtime (CUDA/OpenVINO).
│   ├── tracker/                # Custom Object Tracking implementations.
│   │   ├── bytetrack.py        # ByteTrack algorithm implementation.
│   │   ├── botsort.py          # BoT-SORT algorithm with Global Motion Compensation (GMC).
│   │   ├── kalman_filter.py    # Standard and Extended Kalman Filters for state estimation.
│   │   └── matching.py         # Hungarian algorithm and IoU/DIoU distance calculations.
│   └── utils/                  # Shared utilities.
│       ├── draw.py             # Rendering logic (bounding boxes, masks, FPS counter).
│       └── hardware.py         # pynvml and psutil wrappers for real-time resource monitoring.
└── client/                     # Frontend UI.
    └── index.html              # Zero-build SPA (Tailwind CSS + Alpine.js). Contains WebRTC client logic.
```
