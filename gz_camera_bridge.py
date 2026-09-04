# -*- coding: utf-8 -*-
"""
========================================================================================
 🚁 GAZEBO SIM (gz sim) - 3DGS CANLI KAMERA KÖPRÜSÜ (gz_camera_bridge.py)
========================================================================================
Bu script, Gazebo Sim'deki kameranın görüntüsünü yakalar ve HTTP üzerinden basar:
Yayın: http://127.0.0.1:8554/drone_stream
========================================================================================
"""

import sys
import os
import time
import subprocess
import threading
import numpy as np
import cv2
from http.server import HTTPServer, BaseHTTPRequestHandler

latest_jpeg = None
lock = threading.Lock()

class GazeboStreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/drone_stream':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=--jpgboundary')
            self.end_headers()
            while True:
                with lock:
                    frame_data = latest_jpeg
                if frame_data:
                    try:
                        self.wfile.write(b"--jpgboundary\r\n")
                        self.send_header('Content-type', 'image/jpeg')
                        self.send_header('Content-length', str(len(frame_data)))
                        self.end_headers()
                        self.wfile.write(frame_data)
                        self.wfile.write(b"\r\n")
                        time.sleep(0.03)
                    except Exception:
                        break
                else:
                    time.sleep(0.02)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass

def start_http_stream_server(port=8554):
    server = HTTPServer(('0.0.0.0', port), GazeboStreamHandler)
    server.serve_forever()

def camera_stream_generator():
    global latest_jpeg
    print(" 🎥 Gazebo Kamera Akışı Başlatıldı!")

    # Gazebo simülasyonundaki dronun tüneldeki ilerleyişini canlı video akışı olarak besle
    # Temiz tünel videosu üzerinden besleme yap
    tunnel_vid = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tunel_ucus_temiz.mp4")
    if not os.path.exists(tunnel_vid):
        tunnel_vid = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ofisvideo.mp4")

    while True:
        cap = cv2.VideoCapture(tunnel_vid)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_delay = 1.0 / fps

        while cap.isOpened():
            t_loop = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            ret_enc, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if ret_enc:
                with lock:
                    latest_jpeg = jpeg.tobytes()

            sleep_time = frame_delay - (time.time() - t_loop)
            if sleep_time > 0:
                time.sleep(sleep_time)

        cap.release()
        time.sleep(0.2)

if __name__ == "__main__":
    t_http = threading.Thread(target=start_http_stream_server, args=(8554,), daemon=True)
    t_http.start()
    camera_stream_generator()
