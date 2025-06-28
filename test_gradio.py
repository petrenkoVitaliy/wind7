import gradio as gr
import cv2
import spaces
from ultralytics import YOLO

js_and_css = """
<style>
    video, .input-image img { transform: scaleX(1) !important; }
    #output_image { border: 2px solid #00ff00; border-radius: 15px; overflow: hidden; }
</style>
<script>
    const originalGetUserMedia = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
    navigator.mediaDevices.getUserMedia = (constraints) => {
        if (constraints && constraints.video) {
            constraints.video.facingMode = { exact: "environment" };
            
            if (!constraints.video.facingMode) {
                constraints.video.facingMode = "environment";
            }
        }
        return originalGetUserMedia(constraints);
    };
</script>
"""

model = YOLO('models/best_openvino_model_640_n', task='segment')


@spaces.GPU
def predict_video(frame, conf, quality):
    if frame is None:
        return None

    img_size = 640
    if quality == "Performance (320p)":
        img_size = 320
    elif quality == "Balanced (640p)":
        img_size = 640
    elif quality == "High Accuracy (800p)":
        img_size = 800

    results = model.track(
        frame,
        persist=True,
        conf=conf,
        imgsz=img_size,
        tracker="botsort.yaml",
        # half=True,
        verbose=False
    )[0]

    annotated_frame = results.plot(labels=False, boxes=True, masks=False)
    return annotated_frame


with gr.Blocks(head=js_and_css, css="#input_container { margin-bottom: 20px; }") as demo:
    gr.Markdown("## AI Vision Pro ")

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Accordion("Settings", open=True):
                quality_dropdown = gr.Dropdown(
                    choices=[
                        "Performance (320p)", "Balanced (640p)", "High Accuracy (800p)"],
                    value="Balanced (640p)",
                    label="Resolution"
                )

                conf_slider = gr.Slider(
                    0.1, 1.0, value=0.35, label="Confidence")

                input_video = gr.Image(
                    sources=["webcam"],
                    streaming=True,
                    elem_id="input_container",
                    mirror_webcam=False
                )

        with gr.Column(scale=2):
            output_image = gr.Image(label="AI Prediction", interactive=False)

    input_video.stream(
        fn=predict_video,
        inputs=[input_video, conf_slider, quality_dropdown],
        outputs=[output_image],
        time_limit=120
    )

if __name__ == "__main__":
    demo.launch()


# requirements
# gradio
# fastrtc
# ultralytics
# numpy
# twilio
# lap
# spaces
# torch
# torchvision
# opencv-python-headless
# opencv-contrib-python-headless
