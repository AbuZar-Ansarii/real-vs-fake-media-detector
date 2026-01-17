import streamlit as st
import os
import cv2
import torch
import uuid
import shutil
import requests
from PIL import Image
from transformers import pipeline

# ----------------------------
# CONFIG
# ----------------------------



st.set_page_config(page_title="AI Generated vs Real Media Detector", layout="wide")
ai_image = "banner image..jpg"
st.image(ai_image, width="stretch")

IMAGE_DIR = "feed_images"
FRAME_DIR = "temp_frames"

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(FRAME_DIR, exist_ok=True)

# ----------------------------
# LOAD MODEL (CACHED)
# ----------------------------
@st.cache_resource
def load_hf_model():
    return pipeline(
        task="image-classification",
        model="Hemg/AI-VS-REAL-IMAGE-DETECTION",
        device=0 if torch.cuda.is_available() else -1
    )

pipe = load_hf_model()

# ----------------------------
# IMAGE PREDICTION
# ----------------------------
def predict_image(img: Image.Image):
    img = img.convert("RGB")
    with torch.no_grad():
        results = pipe(img)
    top = max(results, key=lambda x: x["score"])
    return top["label"], float(top["score"])

# ----------------------------
# VIDEO FRAME EXTRACTION
# ----------------------------
def extract_frames(video_path, fps=1):
    shutil.rmtree(FRAME_DIR, ignore_errors=True)
    os.makedirs(FRAME_DIR, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(video_fps // fps) if video_fps > 0 else 1

    count, saved = 0, 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            frame_path = os.path.join(FRAME_DIR, f"frame_{saved}.jpg")
            cv2.imwrite(frame_path, frame)
            saved += 1
        count += 1

    cap.release()
    return saved

# ----------------------------
# VIDEO ANALYSIS
# ----------------------------
def analyze_video(video_path, threshold=0.4):
    extract_frames(video_path)

    fake_frames = 0
    total_frames = 0

    for f in os.listdir(FRAME_DIR):
        if f.endswith(".jpg"):
            img = Image.open(os.path.join(FRAME_DIR, f))
            label, _ = predict_image(img)
            total_frames += 1
            if label.upper() == "FAKE":
                fake_frames += 1

    fake_ratio = fake_frames / total_frames if total_frames > 0 else 0
    final_label = "FAKE" if fake_ratio >= threshold else "REAL"

    return final_label, fake_ratio, total_frames

# ----------------------------
# SIDEBAR
# ----------------------------
st.sidebar.title("📤 Upload Media")

uploaded_images = st.sidebar.file_uploader(
    "Upload Image(s)",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True
)

uploaded_video = st.sidebar.file_uploader(
    "Upload Short Video",
    type=["mp4"]
)

# Save uploaded images
if uploaded_images:
    for file in uploaded_images:
        img = Image.open(file)
        img.save(os.path.join(IMAGE_DIR, f"{uuid.uuid4()}.jpg"))
    st.sidebar.success("Images uploaded")

# ----------------------------
# MAIN UI
# ----------------------------
st.markdown(
    """
    <h1 style="text-align:center;">🧠 AI Real vs Fake Media Detector</h1>
    <p style="text-align:center;color:gray;">
    Image & short-video authenticity analysis using Hugging Face
    </p>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# IMAGE FEED
# ----------------------------
st.subheader("🖼️ Image Feed")

image_files = [
    os.path.join(IMAGE_DIR, f)
    for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".png", ".jpeg", ".webp"))
]

cols = st.columns(4)

for idx, path in enumerate(image_files):
    with cols[idx % 4]:
        img = Image.open(path)
        label, score = predict_image(img)
        st.image(img, use_container_width=True)

        color = "green" if label.upper() == "REAL" else "red"
        icon = "✅" if label.upper() == "REAL" else "❌"

        st.markdown(
            f"<p style='color:{color}; font-weight:bold;'>"
            f"{icon} {label} ({score:.2%})</p>",
            unsafe_allow_html=True
        )

# ----------------------------
# VIDEO RESULT
# ----------------------------
if uploaded_video:
    st.subheader("🎞️ Video Analysis")

    video_path = f"temp_{uuid.uuid4()}.mp4"
    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())

    st.video(video_path)

    with st.spinner("Analyzing video frames..."):
        label, fake_ratio, total_frames = analyze_video(video_path)

    st.success(f"Final Result: {label}")
    st.info(f"Fake Frame Ratio: {fake_ratio:.2%}")
    st.info(f"Frames Analyzed: {total_frames}")

    os.remove(video_path)
