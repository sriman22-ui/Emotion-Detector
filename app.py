import cv2
import gradio as gr
from inference import load_assets, annotate_frame

model, face_cascade = load_assets()  # load once at startup, not per frame


def process(frame):
    if frame is None:
        return None
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # gr.Image gives RGB; annotate_frame expects BGR
    annotated = annotate_frame(bgr, model, face_cascade)
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)


with gr.Blocks(title="Emotion Detector") as demo:
    gr.Markdown("# Live Emotion Detector\nAllow webcam access, then watch your emotion get classified in real time.")
    with gr.Row():
        webcam = gr.Image(sources=["webcam"], streaming=True, label="Your webcam")
        output = gr.Image(label="Annotated output")
    webcam.stream(fn=process, inputs=webcam, outputs=output, stream_every=0.2)

if __name__ == "__main__":
    demo.launch()
