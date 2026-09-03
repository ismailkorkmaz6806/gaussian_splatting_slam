"""
======================================================================================
 🎮 GAZEBO 3B DRON KLAVYE UÇUŞ İSTASYONU (WSL2 / PYTHON)
======================================================================================
W/A/S/D/SPACE tuşlarıyla Gazebo'daki dronu gerçek zamanlı uçurun!
"""

import time
import os
import sys
import subprocess
import math
import numpy as np
import cv2

class GazeboDroneController:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.8
        self.yaw = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.vyaw = 0.0
        self.speed = 1.2
        self.auto_mode = False
        self.auto_t = 0.0

    def update_physics(self, dt=0.04):
        if self.auto_mode:
            self.auto_t += dt
            self.x += 0.6 * dt
            self.y = 0.7 * math.sin(self.auto_t * 0.8)
            self.z = 1.2 + 0.15 * math.sin(self.auto_t * 1.5)
            self.yaw = 0.25 * math.cos(self.auto_t * 0.8)
            if self.x > 25.0:
                self.x = 0.0
                self.auto_t = 0.0
        else:
            # Sürtünme ve hareket
            self.x += (self.vx * math.cos(self.yaw) - self.vy * math.sin(self.yaw)) * dt
            self.y += (self.vx * math.sin(self.yaw) + self.vy * math.cos(self.yaw)) * dt
            self.z = max(0.2, min(2.8, self.z + self.vz * dt))
            self.yaw += self.vyaw * dt

            # Sönümleme
            self.vx *= 0.85
            self.vy *= 0.85
            self.vz *= 0.85
            self.vyaw *= 0.80

    def send_to_gazebo(self):
        cmd = f"wsl -d Ubuntu-22.04 -u root -- gz model -m mapping_drone -x {self.x:.2f} -y {self.y:.2f} -z {self.z:.2f} -Y {self.yaw:.2f}"
        subprocess.Popen(cmd, shell=True)

def run_gazebo_teleop():
    controller = GazeboDroneController()
    cv2.namedWindow("Gazebo Dron Ucus Kontrol Paneli", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Gazebo Dron Ucus Kontrol Paneli", 720, 480)

    print("\n" + "="*60)
    print(" 🎮 GAZEBO DRON KLAVYE KONTROLLERİ:")
    print("   [W / S]       : İleri / Geri Uç")
    print("   [A / D]       : Sola / Sağa Kay")
    print("   [SPACE / C]   : Yukarı Yüksel / Aşağı İn")
    print("   [Q / E]       : Sola / Sağa Dön (Yaw)")
    print("   [P]           : Otonom Tünel Turunu Başlat / Durdur")
    print("   [ESC]         : Çıkış")
    print("="*60 + "\n")

    running = True
    last_send = time.time()

    while running:
        # GUI HUD Çizimi
        hud = np.zeros((480, 720, 3), dtype=np.uint8)
        hud[:] = (20, 24, 30)

        # Üst Başlık
        cv2.rectangle(hud, (0, 0), (720, 60), (35, 40, 50), -1)
        cv2.putText(hud, "GAZEBO DRON UCUS PANELI (WSL2)", (20, 38), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 200), 2)

        # Durum Modu
        mode_str = "OTONOM TUNEL TURU (AI)" if controller.auto_mode else "MANUEL KLAVYE MODU"
        mode_col = (0, 180, 255) if controller.auto_mode else (0, 255, 120)
        cv2.putText(hud, f"MOD: {mode_str}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.65, mode_col, 2)

        # Telemetri Bilgileri
        cv2.putText(hud, f"X Konumu (Ileri)   : {controller.x:.2f} m", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
        cv2.putText(hud, f"Y Konumu (Yanal)   : {controller.y:.2f} m", (20, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
        cv2.putText(hud, f"Z Irtifasi (Yukseklik): {controller.z:.2f} m", (20, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
        cv2.putText(hud, f"Bas Acisi (Yaw)    : {math.degrees(controller.yaw):.1f} deg", (20, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)

        # Kısayol Rehberi
        cv2.rectangle(hud, (20, 310), (700, 450), (28, 34, 42), -1)
        cv2.putText(hud, "TUSLAR: [W/S]: Ileri/Geri | [A/D]: Sol/Sag | [SPACE/C]: Yukari/Asagi", (35, 345), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 200, 220), 1)
        cv2.putText(hud, "        [Q/E]: Saga/Sola Don (Yaw) | [P]: Otonom Tur | [ESC]: Cikis", (35, 375), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 200, 220), 1)
        cv2.putText(hud, "* Bu pencere seciliyken tuslara basiniz.", (35, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 160, 255), 1)

        cv2.imshow("Gazebo Dron Ucus Kontrol Paneli", hud)

        # Tuş Okuma
        key = cv2.waitKey(30) & 0xFF

        if key in (ord('w'), ord('W')): controller.vx = controller.speed
        elif key in (ord('s'), ord('S')): controller.vx = -controller.speed

        if key in (ord('a'), ord('A')): controller.vy = controller.speed
        elif key in (ord('d'), ord('D')): controller.vy = -controller.speed

        if key == 32: controller.vz = controller.speed # SPACE
        elif key in (ord('c'), ord('C')): controller.vz = -controller.speed

        if key in (ord('q'), ord('Q')): controller.vyaw = 0.5
        elif key in (ord('e'), ord('E')): controller.vyaw = -0.5

        if key in (ord('p'), ord('P')):
            controller.auto_mode = not controller.auto_mode

        if key == 27: # ESC
            running = False

        controller.update_physics(0.03)

        if time.time() - last_send > 0.08:
            controller.send_to_gazebo()
            last_send = time.time()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_gazebo_teleop()
