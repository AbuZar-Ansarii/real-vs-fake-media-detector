import cv2
import os

def extract_frames(video_path, output_dir, fps=3):
    """
    Extract frames from video at given FPS.
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    frame_interval = int(video_fps // fps)
    frame_count = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{saved_count}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    return saved_count

print(extract_frames(r"E:\GEN AI PROJECT\fake image detector\Reality Media\short_vids\VID_20260115_214359190.mp4", r"E:\GEN AI PROJECT\fake image detector\Reality Media\ex_frames"))