import cv2
import glob

vfiles = glob.glob('raw_short_3.*')
input_file = vfiles[0]
output_file = "tunel_ucus_3.mp4"

cap = cv2.VideoCapture(input_file)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Giris: {w}x{h}, {total_frames} kare, {fps:.1f} FPS")

# Alttaki kumanda kismini kes (Ustteki %58'lik saf ucusu al)
crop_h = int(h * 0.58)
crop_w = w

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
    if count % 50 == 0:
        print(f" -> Islenen: {count}/{total_frames}")

cap.release()
out.release()
print(f"\n[OK] Video basariyla kesildi: {output_file} ({count} Kare, {crop_w}x{crop_h})")
