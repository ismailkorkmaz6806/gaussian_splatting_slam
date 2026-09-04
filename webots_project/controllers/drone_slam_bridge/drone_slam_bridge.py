# -*- coding: utf-8 -*-
"""
========================================================================================
 🚁 WEBOTS DRON - 3DGS SLAM KÖPRÜSÜ (drone_slam_bridge.py)
========================================================================================
Bu kontrolcü, Webots içindeki DJI Mavic 2 Pro dronu uçurur ve kamerasından aldığı
canlı görüntüyü yerel ağa (http://127.0.0.1:8554/drone_stream) basar.
========================================================================================
"""

import os
import sys
import time
import math
import threading
import numpy as np
import cv2
from http.server import HTTPServer, BaseHTTPRequestHandler

# Webots Controller API
from controller import Robot, Keyboard

# Global Frame Buffer & Lock
latest_jpeg = None
lock = threading.Lock()

class WebotsStreamHandler(BaseHTTPRequestHandler):
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
        pass # Konsolu loglarla doldurma

def start_http_stream_server(port=8554):
    server = HTTPServer(('0.0.0.0', port), WebotsStreamHandler)
    server.serve_forever()

def clamp(val, v_min, v_max):
    return min(max(val, v_min), v_max)

class DroneSlamBridge(Robot):
    K_VERTICAL_THRUST = 68.5
    K_VERTICAL_OFFSET = 0.6
    K_VERTICAL_P = 3.0
    K_ROLL_P = 50.0
    K_PITCH_P = 30.0

    def __init__(self):
        super().__init__()
        self.time_step = int(self.getBasicTimeStep())

        # Sensörleri Başlat
        self.camera = self.getDevice("camera")
        self.camera.enable(self.time_step)
        self.cam_w = self.camera.getWidth()
        self.cam_h = self.camera.getHeight()

        self.imu = self.getDevice("inertial unit")
        self.imu.enable(self.time_step)
        self.gps = self.getDevice("gps")
        self.gps.enable(self.time_step)
        self.gyro = self.getDevice("gyro")
        self.gyro.enable(self.time_step)

        self.keyboard = Keyboard()
        self.keyboard.enable(self.time_step)

        # Motorları Başlat
        self.fl_motor = self.getDevice("front left propeller")
        self.fr_motor = self.getDevice("front right propeller")
        self.rl_motor = self.getDevice("rear left propeller")
        self.rr_motor = self.getDevice("rear right propeller")
        self.cam_pitch_motor = self.getDevice("camera pitch")
        if self.cam_pitch_motor:
            self.cam_pitch_motor.setPosition(0.2)  # Hafif ileri/aşağı bakış açısı

        for m in [self.fl_motor, self.fr_motor, self.rl_motor, self.rr_motor]:
            m.setPosition(float('inf'))
            m.setVelocity(1.0)

        self.target_altitude = 1.3
        self.target_yaw = 0.0
        print("\n" + "="*65)
        print(" 🚁 WEBOTS DRON SLAM KÖPRÜSÜ AKTİF!")
        print(f" 📹 Kamera: {self.cam_w}x{self.cam_h} px")
        print(" 📡 Canlı Yayın: http://127.0.0.1:8554/drone_stream")
        print(" 🕹️ Klavye Kontrolleri:")
        print("    [YUKARI / ASAGI / SOL / SAG] : İleri / Geri / Sağa / Sola Uç")
        print("    [W / S] : İrtifa Yüksel / Alçal | [A / D] : Yaw Dönüşü")
        print("="*65 + "\n")

    def run(self):
        global latest_jpeg

        # HTTP Sunucusunu Arka Planda Başlat
        t = threading.Thread(target=start_http_stream_server, args=(8554,), daemon=True)
        t.start()

        while self.step(self.time_step) != -1:
            # 1. Kameradan Görüntü Al ve Canlı Yayına Bas
            img_raw = self.camera.getImage()
            if img_raw:
                img_arr = np.frombuffer(img_raw, np.uint8).reshape((self.cam_h, self.cam_w, 4))
                bgr_frame = cv2.cvtColor(img_arr, cv2.COLOR_BGRA2BGR)
                ret, jpeg = cv2.imencode('.jpg', bgr_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if ret:
                    with lock:
                        latest_jpeg = jpeg.tobytes()

            # 2. Sensör Verileri
            roll = self.imu.getRollPitchYaw()[0]
            pitch = self.imu.getRollPitchYaw()[1]
            yaw = self.imu.getRollPitchYaw()[2]
            altitude = self.gps.getValues()[2]
            roll_vel = self.gyro.getValues()[0]
            pitch_vel = self.gyro.getValues()[1]

            # 3. Klavye Kontrolleri
            k = self.keyboard.getKey()
            pitch_dist = 0.0
            roll_dist = 0.0
            yaw_dist = 0.0

            if k == Keyboard.UP:
                pitch_dist = -0.5
            elif k == Keyboard.DOWN:
                pitch_dist = 0.5
            elif k == Keyboard.LEFT:
                roll_dist = -0.5
            elif k == Keyboard.RIGHT:
                roll_dist = 0.5
            elif k in (ord('W'), ord('w')):
                self.target_altitude += 0.02
            elif k in (ord('S'), ord('s')):
                self.target_altitude = max(0.2, self.target_altitude - 0.02)
            elif k in (ord('A'), ord('a')):
                yaw_dist = 0.8
            elif k in (ord('D'), ord('d')):
                yaw_dist = -0.8

            # 4. PID Uçuş Stabilizasyonu
            roll_input = self.K_ROLL_P * clamp(roll, -1.0, 1.0) + roll_vel + roll_dist
            pitch_input = self.K_PITCH_P * clamp(pitch, -1.0, 1.0) - pitch_vel + pitch_dist
            yaw_input = yaw_dist
            clamped_diff_alt = clamp(self.target_altitude - altitude + self.K_VERTICAL_OFFSET, -1.0, 1.0)
            vertical_input = self.K_VERTICAL_P * (clamped_diff_alt ** 3.0)

            fl_vel = self.K_VERTICAL_THRUST + vertical_input - roll_input + pitch_input - yaw_input
            fr_vel = self.K_VERTICAL_THRUST + vertical_input + roll_input + pitch_input + yaw_input
            rl_vel = self.K_VERTICAL_THRUST + vertical_input - roll_input - pitch_input + yaw_input
            rr_vel = self.K_VERTICAL_THRUST + vertical_input + roll_input - pitch_input - yaw_input

            self.fl_motor.setVelocity(fl_vel)
            self.fr_motor.setVelocity(-fr_vel)
            self.rl_motor.setVelocity(-rl_vel)
            self.rr_motor.setVelocity(rr_vel)

if __name__ == "__main__":
    drone = DroneSlamBridge()
    drone.run()
