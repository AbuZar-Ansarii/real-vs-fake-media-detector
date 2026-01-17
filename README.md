# 🧠 AI Real vs Fake Media Detector
![banner image](https://github.com/user-attachments/assets/39d39184-0681-4d11-ae3f-d323907a2a7d)

An AI-powered system that detects whether **images or short videos (reels)** are real or AI-generated.  
Built using **Streamlit** and a **Hugging Face vision model**, this project demonstrates a production-style media authenticity pipeline.

---
## sample(video analysis) 
<img width="1920" height="1080" alt="Screenshot (104)" src="https://github.com/user-attachments/assets/348e4811-6740-43fd-abdf-f97295dc6b6a" />
<img width="1920" height="1080" alt="Screenshot (105)" src="https://github.com/user-attachments/assets/b59c29a8-352c-4f5a-9b03-26d5ad602d94" />

## 🚀 Features

- 📷 Real vs Fake detection for images
- 🎞️ Short video / reel analysis using frame sampling
- 🧠 Hugging Face pretrained AI model
- 📊 Confidence-based predictions
- 🖼️ Pinterest-style image feed
- 📤 Sidebar upload for images & videos

---

## 🧠 Model Used

**Hugging Face Model:**  
`Hemg/AI-VS-REAL-IMAGE-DETECTION`

- Task: Image Classification
- Output: REAL / FAKE + confidence
- Backend: PyTorch

---

## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/ai-real-vs-fake-media-detector.git
cd ai-real-vs-fake-media-detector

 ** Install dependencies**
pip install -r requirements.txt

**Run the app**
streamlit run app.py

