from datetime import datetime
from enum import Enum


class Stage(str, Enum):
    INIT = "INIT"
    RELOAD = "RELOAD"
    ERROR = "ERROR"
    NONE = ""


class Source(str, Enum):
    MAIN = "MAIN"
    YOLO_HANDLER = "YOLO_HANDLER"
    ONNX_HANDLER = "ONNX_HANDLER"
    MODEL_ADAPTER = "MODEL_ADAPTER"
    INFER_CONTROLLER = "INFER_CONTROLLER"


class L(Enum):
    # SEP
    SEP = (Stage.NONE, Source.MAIN, "-" * 30)

    # INIT :: MAIN
    INIT_DEV_ENV = (Stage.INIT, Source.MAIN, "Using dev env")
    INIT_NUM_THREADS = (Stage.INIT, Source.MAIN, "Setting number of threads to {num}")
    INIT_DEV_ICE = (
        Stage.INIT,
        Source.MAIN,
        "Using dev ICE config",
    )
    INIT_CONN_STATE = (Stage.INIT, Source.MAIN, "Connection state is {state}")
    INIT_DATACHANNEL = (Stage.INIT, Source.MAIN, "Data channel opened")
    INIT_TRACK_RECEIVED = (Stage.INIT, Source.MAIN, "Track received: {kind}")

    # INIT :: INFER_CONTROLLER
    INIT_VIDEO_TRANSFORM_TRACK = (
        Stage.INIT,
        Source.INFER_CONTROLLER,
        "StreamInferController initialized",
    )

    # INIT :: ONNX_HANDLER
    INIT_ONNX_PROVIDERS = (Stage.INIT, Source.ONNX_HANDLER, "Providers: {providers}")
    INIT_BYTETRACK = (Stage.INIT, Source.ONNX_HANDLER, "ByteTrack initialized")
    INIT_BOTSORT = (Stage.INIT, Source.ONNX_HANDLER, "BoTSORT initialized")

    # RELOAD :: MAIN
    RELOAD_CONFIG = (Stage.RELOAD, Source.MAIN, "Received config: {data}")

    # RELOAD :: INFER_CONTROLLER
    RELOAD_CONFIG_NEW = (
        Stage.RELOAD,
        Source.INFER_CONTROLLER,
        "Config is new: {config}",
    )
    RELOAD_SAME_CONFIG = (
        Stage.RELOAD,
        Source.INFER_CONTROLLER,
        "Same config",
    )

    # RELOAD :: MODEL_ADAPTER
    RELOAD_SAME_MODEL_TYPE = (
        Stage.RELOAD,
        Source.MODEL_ADAPTER,
        "Same model type: {mtype}",
    )
    RELOAD_NEW_MODEL_TYPE = (
        Stage.RELOAD,
        Source.MODEL_ADAPTER,
        "New model type: {mtype}",
    )

    # RELOAD :: ONNX_HANDLER
    RELOAD_ONNX_TRACKER = (
        Stage.RELOAD,
        Source.ONNX_HANDLER,
        "Switched tracker to {tracker}",
    )
    RELOAD_ONNX_OK = (
        Stage.RELOAD,
        Source.ONNX_HANDLER,
        "Successfully reloaded: {name}",
    )

    # RELOAD :: YOLO_HANDLER
    RELOAD_YOLO_LOADING = (Stage.RELOAD, Source.YOLO_HANDLER, "Loading model {name}")
    RELOAD_YOLO_OK = (
        Stage.RELOAD,
        Source.YOLO_HANDLER,
        "Successfully reloaded: {name}",
    )

    # ERROR :: MAIN
    ERROR_TWILIO_CREDS = (Stage.ERROR, Source.MAIN, "Twilio credentials not configured")
    ERROR_TWILIO = (Stage.ERROR, Source.MAIN, "Twilio: {err}")
    ERROR_NO_VIDEO_TRACK = (Stage.ERROR, Source.MAIN, "No video track available yet")
    ERROR_PARSE_MSG = (Stage.ERROR, Source.MAIN, "Error parsing message: {err}")
    ERROR_CONSUMPTION = (Stage.ERROR, Source.MAIN, "Consumption stopped: {err}")

    # ERROR :: INFER_CONTROLLER
    ERROR_PROCESS_FRAME = (
        Stage.ERROR,
        Source.INFER_CONTROLLER,
        "Failed to process frame: {err}",
    )

    # ERROR :: ONNX_HANDLER
    ERROR_ONNX_RELOAD = (
        Stage.ERROR,
        Source.ONNX_HANDLER,
        "Failed to reload model: {err}",
    )

    # ERROR :: YOLO_HANDLER
    ERROR_YOLO_RELOAD = (
        Stage.ERROR,
        Source.YOLO_HANDLER,
        "Failed to reload model in task: {err}",
    )
    ERROR_YOLO_FAIL = (
        Stage.ERROR,
        Source.YOLO_HANDLER,
        "Failed to reload model: {err}",
    )

    def __init__(self, stage: Stage, source: Source, template: str) -> None:
        self._stage = stage
        self._source = source
        self._template = template

    @property
    def stage(self) -> Stage:
        return self._stage

    @property
    def source(self) -> Source:
        return self._source

    @property
    def template(self) -> str:
        return self._template

    def format(self, **kw: object) -> str:
        msg = self._template.format(**kw) if kw else self._template
        if self._stage is Stage.NONE:
            return msg
        return f"{self._stage.value} :: {self._source.value} :: {msg}"


def tprint(event: L, /, exc_info: bool = False, **kw: object) -> None:
    ts = datetime.now()
    base = ts.strftime("%H:%M:%S")
    cs = f"{min(round(ts.microsecond / 10000), 99):02d}"
    if exc_info:
        import traceback
        traceback.print_exc()
    print(f"[{base}:{cs}] {event.format(**kw)}", flush=True)
