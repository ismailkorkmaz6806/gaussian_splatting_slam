"""
======================================================================================
 🚁 SANAL DRON CANLI VİDEO YAYIN SUNUCUSU (simulated_drone_streamer.py)
======================================================================================
Bu script, havada uçan gerçek bir dronun Wi-Fi/Fiber üzerinden yer bilgisayarına
canlı HD video yayını yapmasını %100 simüle eder.

Yayın Adresi: http://127.0.0.1:8554/drone_stream
"""

import time
import os
import sys
import threading
import cv2
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_CANDIDATES = [
    os.path.join(CURR_DIR, "tunel_ucus_temiz.mp4"),
    os.path.join(CURR_DIR, "tunel_ucus_2.mp4"),
    os.path.join(CURR_DIR, "tunel_ucus_3.mp4"),
    os.path.join(CURR_DIR, "ofisvideo.mp4")
]

selected_video = None
for v in VIDEO_CANDIDATES:
    if os.path.exists(v):
        selected_video = v
        break

if not selected_video:
    print("❌ HATA: Yayınlanacak video dosyası bulunamadı!")
    sys.exit(1)

# Global Frame Buffer
current_frame_jpeg = None
lock = threading.Lock()

def video_stream_producer():
    global current_frame_jpeg
    print(f" 🎥 Sanal Dron Kamerası Başlatıldı: {os.path.basename(selected_video)}")
    
    while True:
        cap = cv2.VideoCapture(selected_video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_delay = 1.0 / fps

        t0 = time.time()
        frame_idx = 0

        while cap.isOpened():
            t_loop = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            elapsed = time.time() - t0

            # Ham temiz kamera karesini JPEG olarak sıkıştır (Yapay zeka için pikseller tertemiz olmalı)
            ret_enc, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

            if ret_enc:
                with lock:
                    current_frame_jpeg = jpeg.tobytes()

            sleep_time = frame_delay - (time.time() - t_loop)
            if sleep_time > 0:
                time.sleep(sleep_time)

        cap.release()
        time.sleep(0.5)

class DroneStreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/drone_stream':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=--jpgboundary')
            self.end_headers()
            while True:
                with lock:
                    if current_frame_jpeg is not None:
                        frame_bytes = current_frame_jpeg
                    else:
                        frame_bytes = None

                if frame_bytes:
                    try:
                        self.wfile.write(b"--jpgboundary\r\n")
                        self.send_header('Content-type', 'image/jpeg')
                        self.send_header('Content-length', str(len(frame_bytes)))
                        self.end_headers()
                        self.wfile.write(frame_bytes)
                        self.wfile.write(b"\r\n")
                        time.sleep(0.03)
                    except Exception:
                        break
        else:
            self.send_response(404)
            self.end_headers()

def start_server(port=8554):
    server = HTTPServer(('0.0.0.0', port), DroneStreamHandler)
    print(f"\n" + "="*65)
    print(f" 🚀 SANAL DRON CANLI YAYINI AKTİF!")
    print(f" 📡 Canlı Akış Linki: http://127.0.0.1:{port}/drone_stream")
    print(f" 💡 BASLAT.bat ➔ [1] ile bu yayına anında bağlanabilirsiniz.")
    print("="*65 + "\n")
    server.serve_forever()

if __name__ == "__main__":
    t = threading.Thread(target=video_stream_producer, daemon=True)
    t.start()
    start_server(8554)
