from datetime import datetime
from enum import Enum


class Stage(str, Enum):
    INIT = "INIT"
    RELOAD = "RELOAD"
    ERROR = "ERROR"
    NONE = ""


class L(Enum):
    INIT_DEV_ENV = (Stage.INIT, "using dev env")
    INIT_NUM_THREADS = (Stage.INIT, "Setting number of threads to {num}")
    INIT_DEV_ICE = (Stage.INIT, "Using dev ICE config with public STUN server")
    ERROR_TWILIO_CREDS = (Stage.ERROR, "Twilio credentials not configured")
    ERROR_TWILIO = (Stage.ERROR, "Twilio: {err}")
    INIT_CONN_STATE = (Stage.INIT, "Connection state is {state}")
    INIT_DATACHANNEL = (Stage.INIT, "Data channel opened!")
    ERROR_NO_VIDEO_TRACK = (Stage.ERROR, "No video track available yet.")
    SEP = (Stage.NONE, "--------------------")
    RELOAD_CONFIG = (Stage.RELOAD, "Received config: {data}")
    ERROR_PARSE_MSG = (Stage.ERROR, "Error parsing message: {err}")
    INIT_TRACK_RECEIVED = (Stage.INIT, "Track received: {kind}")
    ERROR_CONSUMPTION = (Stage.ERROR, "Consumption stopped: {err}")
    INIT_VIDEO_TRANSFORM_TRACK = (Stage.INIT, "VideoTransformTrack initialized")
    ERROR_PROCESS_FRAME = (Stage.ERROR, "process_frame: {err}")
    RELOAD_CONFIG_NEW = (Stage.RELOAD, "Config is new: {config}")
    RELOAD_SAME_CONFIG = (Stage.RELOAD, "Same config, no reload needed.")
    RELOAD_SAME_MODEL_TYPE = (Stage.RELOAD, "Same model type: {mtype}")
    RELOAD_NEW_MODEL_TYPE = (Stage.RELOAD, "New model type: {mtype}")
    INIT_ONNX_PROVIDERS = (Stage.INIT, "ONNX providers: {providers}")
    RELOAD_ONNX_TRACKER = (Stage.RELOAD, "ONNX Switched tracker to {tracker}")
    RELOAD_ONNX_OK = (Stage.RELOAD, "ONNX successfully: {name}")
    ERROR_ONNX_RELOAD = (Stage.ERROR, "ONNX Failed to reload model: {err}")
    INIT_BYTETRACK = (Stage.INIT, "ByteTrack initialized")
    INIT_BOTSORT = (Stage.INIT, "BoTSORT initialized")
    RELOAD_YOLO_LOADING = (Stage.RELOAD, "YOLO Loading model {name}")
    RELOAD_YOLO_WARMUP = (Stage.RELOAD, "YOLO Warming up...")
    ERROR_YOLO_RELOAD = (Stage.ERROR, "YOLO reload failed: {err}")
    RELOAD_YOLO_OK = (Stage.RELOAD, "YOLO successfully: {name}")
    ERROR_YOLO_FAIL = (Stage.ERROR, "YOLO Failed to reload model: {err}")

    def __init__(self, stage: Stage, template: str) -> None:
        self._stage = stage
        self._template = template

    @property
    def stage(self) -> Stage:
        return self._stage

    @property
    def template(self) -> str:
        return self._template

    def format(self, **kw: object) -> str:
        msg = self._template.format(**kw) if kw else self._template
        if self._stage is Stage.NONE:
            return msg
        return f"{self._stage.value}: {msg}"


def tprint(event: L, /, **kw: object) -> None:
    ts = datetime.now().strftime("[%M:%S.%03f]")
    print(f"{ts} {event.format(**kw)}")
