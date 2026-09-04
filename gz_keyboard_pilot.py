# -*- coding: utf-8 -*-
"""
========================================================================================
 🎮 GAZEBO DRON CANLI KLAVYE KONTROLCÜSÜ (gz_keyboard_pilot.py)
========================================================================================
Bu kontrolcü, ekranda bir pencere açar ve basılı tuttuğunuz tuşlara göre (W/A/S/D/SPACE)
Gazebo'daki drona anlık olarak uçuş komutları gönderir.
========================================================================================
"""

import sys
import os
import time
import subprocess
import pygame
from pygame.locals import *

pygame.init()
win_w, win_h = 460, 320
screen = pygame.display.set_mode((win_w, win_h))
pygame.display.set_caption("🎮 GAZEBO DRON KLAVYE KONTROL MERKEZI")

font_title = pygame.font.SysFont("Segoe UI", 16, bold=True)
font_bold = pygame.font.SysFont("Segoe UI", 13, bold=True)
font_norm = pygame.font.SysFont("Segoe UI", 12)

conda_bat = r"C:\Users\ismai\anaconda3\condabin\conda.bat"

def send_vel(vx, vy, vz, wz):
    msg = f'linear: {{x: {vx:.2f}, y: {vy:.2f}, z: {vz:.2f}}}, angular: {{z: {wz:.2f}}}'
    cmd = f'"{conda_bat}" run -n gz_env gz topic -t "/cmd_vel" -m gz.msgs.Twist -p "{msg}"'
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)

clock = pygame.time.Clock()
running = True
last_send = time.time()

print("=" * 60)
print(" 🎮 GAZEBO KLAVYE KONTROLCÜSÜ BAŞLATILDI!")
print(" [W / S / A / D / OKLAR] : Dronu Uçur")
print(" [SPACE / SHIFT]         : Yüksel / Alçal")
print(" [Q / E]                 : Sağa / Sola Dön (Yaw)")
print("=" * 60)

cur_vx, cur_vy, cur_vz, cur_wz = 0.0, 0.0, 0.0, 0.0

while running:
    dt = clock.tick(30) / 1000.0

    for event in pygame.event.get():
        if event.type == QUIT:
            running = False

    keys = pygame.key.get_pressed()
    target_vx, target_vy, target_vz, target_wz = 0.0, 0.0, 0.0, 0.0

    # Hız değerleri
    speed = 1.5
    rot_speed = 0.8

    if keys[K_w] or keys[K_UP]:
        target_vx += speed
    if keys[K_s] or keys[K_DOWN]:
        target_vx -= speed
    if keys[K_a] or keys[K_LEFT]:
        target_vy += speed
    if keys[K_d] or keys[K_RIGHT]:
        target_vy -= speed
    if keys[K_SPACE]:
        target_vz += speed * 0.8
    if keys[K_LSHIFT] or keys[K_RSHIFT] or keys[K_c]:
        target_vz -= speed * 0.8
    if keys[K_q]:
        target_wz += rot_speed
    if keys[K_e]:
        target_wz -= rot_speed

    # Yumuşatma
    cur_vx = 0.7 * cur_vx + 0.3 * target_vx
    cur_vy = 0.7 * cur_vy + 0.3 * target_vy
    cur_vz = 0.7 * cur_vz + 0.3 * target_vz
    cur_wz = 0.7 * cur_wz + 0.3 * target_wz

    # Her 100ms'de bir komut bas
    if time.time() - last_send >= 0.1:
        send_vel(cur_vx, cur_vy, cur_vz, cur_wz)
        last_send = time.time()

    # Arayüz Çizimi
    screen.fill((15, 20, 30))
    pygame.draw.rect(screen, (0, 160, 255), (10, 10, win_w - 20, win_h - 20), width=2, border_radius=8)

    t_s = font_title.render("🎮 GAZEBO DRON PİLOTU", True, (0, 220, 255))
    screen.blit(t_s, (24, 20))

    lines = [
        ("W / S veya YUKARI / ASAGI", f"İleri / Geri Hız: {cur_vx:+.1f} m/s"),
        ("A / D veya SOL / SAG", f"Sağ / Sol Kayma: {cur_vy:+.1f} m/s"),
        ("SPACE / SHIFT", f"İrtifa Hızı: {cur_vz:+.1f} m/s"),
        ("Q / E", f"Dönüş Açısı: {cur_wz:+.1f} rad/s"),
    ]

    y = 65
    for k_txt, val_txt in lines:
        pygame.draw.rect(screen, (25, 35, 50), (24, y, win_w - 48, 38), border_radius=6)
        screen.blit(font_bold.render(k_txt, True, (255, 200, 80)), (34, y + 4))
        screen.blit(font_norm.render(val_txt, True, (0, 255, 180)), (34, y + 20))
        y += 46

    pygame.draw.rect(screen, (35, 45, 65), (24, y, win_w - 48, 30), border_radius=6)
    status_txt = "🚀 DRON UÇUYOR (Tuşlara Basılı Tutun)" if abs(cur_vx)+abs(cur_vy)+abs(cur_vz)+abs(cur_wz) > 0.05 else "⏸️ DRON ASILI (Havada Bekliyor)"
    screen.blit(font_bold.render(status_txt, True, (255, 255, 255)), (34, y + 7))

    pygame.display.flip()

pygame.quit()
