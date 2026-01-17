import os
import numpy as np
from langchain_huggingface import HuggingFacePipeline
import requests
from dotenv import load_dotenv
load_dotenv()
import torch
from transformers import pipeline
from PIL import Image
import requests
import torch

pipe = pipeline(
    task="image-classification", 
    model="Hemg/AI-VS-REAL-IMAGE-DETECTION"
)

def predict_image(image_path_or_url):
    if image_path_or_url.startswith("http"):
        img = Image.open(requests.get(image_path_or_url, stream=True).raw)
    else:
        img = Image.open(image_path_or_url)

    img = img.convert("RGB")

    with torch.no_grad():
        results = pipe(img)

    return results

def get_final_prediction(results):
    top = max(results, key=lambda x: x["score"])
    return top["label"], top["score"]



def predict_video_frames_hf(frames_dir, threshold=0.4):
    fake_frames = 0
    total_frames = 0

    for file in os.listdir(frames_dir):
        if file.lower().endswith(".jpg"):
            image_path = os.path.join(frames_dir, file)

            results = predict_image(image_path)
            label, score = get_final_prediction(results)

            total_frames += 1
            if label.upper() == "FAKE":
                fake_frames += 1

    fake_ratio = fake_frames / total_frames
    final_label = "FAKE" if fake_ratio >= threshold else "REAL"

    return {
        "video_label": final_label,
        "fake_ratio": fake_ratio,
        "total_frames": total_frames
    }


print(predict_video_frames_hf(r"E:\GEN AI PROJECT\fake image detector\Reality Media\ex_frames"))
