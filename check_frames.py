import cv2, os, numpy as np

for ep in ["episode_0000", "episode_0001"]:
    mp4_dir = f"/Users/Balu/Documents/NEED/episodes/{ep}"
    mp4_file = [f for f in os.listdir(mp4_dir) if f.endswith(".mp4")][0]
    mp4 = os.path.join(mp4_dir, mp4_file)
    cap = cv2.VideoCapture(mp4)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()
    print(f"{ep}: {len(frames)} frames at {frames[0].shape if frames else 'N/A'}")
    if frames:
        print(f"  Center pixel: {frames[len(frames)//2][500, 960]}")
        print(f"  Frame mean: {np.mean(frames[len(frames)//2]):.1f}")
