# -*- coding: utf-8 -*-
"""
========================================================================================
 🎮 GAZEBO DRON OTONOM VE MANUEL PİLOT KONTROLCÜSÜ (gz_drone_pilot.py)
========================================================================================
Bu script, Gazebo Sim'deki /cmd_vel konusuna hız komutları basarak dronu tünelde
otonom olarak ileri uçurur, tünel sonunda döner ve devriye atar.
========================================================================================
"""

import sys
import os
import time
import subprocess

print("=" * 60)
print(" 🚁 GAZEBO DRON OTOMATİK PİLOT AKTİF!")
print(" -> Dron tünel içinde 1.3m irtifada ileri doğru uçuyor...")
print("=" * 60)

conda_bat = r"C:\Users\ismai\anaconda3\condabin\conda.bat"

def send_vel(vx, vy, vz, wz):
    # gz topic üzerinden Twist mesajı bas
    msg = f'linear: {{x: {vx:.2f}, y: {vy:.2f}, z: {vz:.2f}}}, angular: {{z: {wz:.2f}}}'
    cmd = f'"{conda_bat}" run -n gz_env gz topic -t "/cmd_vel" -m gz.msgs.Twist -p "{msg}"'
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)

# Otonom Devriye Döngüsü
step = 0
while True:
    step += 1
    # 25 saniye ileri git, 5 saniye 180 derece dön, 25 saniye geri gel
    phase = step % 60
    if phase < 25:
        # İleri Uç
        send_vel(1.2, 0.0, 0.0, 0.0)
    elif phase < 30:
        # Sağa Dön (180 derece)
        send_vel(0.1, 0.0, 0.0, 0.6)
    elif phase < 55:
        # Geri Uç
        send_vel(1.2, 0.0, 0.0, 0.0)
    else:
        # Tekrar Başa Dön
        send_vel(0.1, 0.0, 0.0, -0.6)

    time.sleep(0.5)
