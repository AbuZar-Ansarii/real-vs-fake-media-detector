import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model(r"E:\GEN AI PROJECT\fake image detector\Reality Media\vg16_ai_model.h5")
CLASS_NAMES = ["Fake", "Real"]

def preprocess_image(image_path):
    img = cv2.imread(image_path) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    img = cv2.resize(img, (256, 256))  
    img = img / 255.0  
    img = np.expand_dims(img, axis=0)  
    return img

def predict_video_frames(frames_dir):
    fake_frames = 0
    total_frames = 0
    confidences = []

    for file in os.listdir(frames_dir):
        if file.endswith(".jpg"):
            image_path = os.path.join(frames_dir, file)

            processed = preprocess_image(image_path)
            prediction = model.predict(processed, verbose=0)

            label_index = prediction.argmax()
            confidence = float(prediction.max())
            label = CLASS_NAMES[label_index]

            total_frames += 1
            confidences.append(confidence)

            if label == "Fake":
                fake_frames += 1

    return {'fake frame': fake_frames,'totoal frame': total_frames, 'confidence':  confidences}



print(predict_video_frames(r"E:\GEN AI PROJECT\fake image detector\Reality Media\ex_frames"))
