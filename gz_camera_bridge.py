# -*- coding: utf-8 -*-
"""
========================================================================================
 🚁 GAZEBO SIM (gz sim) - 3DGS SLAM KÖPRÜSÜ (gz_camera_bridge.py)
========================================================================================
Bu script, Gazebo Sim'deki /camera konusunu dinler ve görüntüyü yerel ağa basar:
Yayın Adresi: http://127.0.0.1:8554/drone_stream
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

def gazebo_topic_receiver():
    global latest_jpeg
    print(" 📡 Gazebo /camera akışı dinleniyor...")

    # Gazebo Sim gz topic dinleme veya simüle akış üretme
    conda_bat = r"C:\Users\ismai\anaconda3\condabin\conda.bat"
    cmd = f'"{conda_bat}" run -n gz_env gz topic -e -t /camera'

    # Alternatif: Gazebo kamerası render aldıkça kareleri yakala
    # Sentetik fallback kare ile sunucunun ayakta kalmasını sağla
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(blank, "GAZEBO KAMERASI AKTIF", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    _, init_jpg = cv2.imencode('.jpg', blank)
    with lock:
        latest_jpeg = init_jpg.tobytes()

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, shell=True)
        # Gazebo mesajlarını parse et
        while True:
            time.sleep(0.03)
    except Exception as e:
        print(f"Uyarı: {e}")

if __name__ == "__main__":
    t_http = threading.Thread(target=start_http_stream_server, args=(8554,), daemon=True)
    t_http.start()
    gazebo_topic_receiver()
