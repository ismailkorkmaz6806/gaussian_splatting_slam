import cv2
import os

input_file = "raw_short.mp4.webm"
output_file = "tunel_ucus_temiz.mp4"

cap = cv2.VideoCapture(input_file)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Giris Videosu: {w}x{h}, {total_frames} kare, {fps:.1f} FPS")

# Alttaki kumanda ve el kismini kes (Ustteki %55'lik saf dron ucusunu al)
crop_h = int(h * 0.55) # 1056 piksel
crop_w = w             # 1080 piksel

# Standart 16:9 veya temiz 1080x720 / 1080x1056 video olustur
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, fps, (crop_w, crop_h))

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cropped = frame[0:crop_h, 0:crop_w]
    out.write(cropped)
    count += 1
    if count % 60 == 0:
        print(f" -> Islenen Kare: {count}/{total_frames}")

cap.release()
out.release()
print(f"\n[OK] Video basariyla kesildi ve kaydedildi: {output_file} ({count} Kare, {crop_w}x{crop_h})")
