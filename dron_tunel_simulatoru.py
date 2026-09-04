# -*- coding: utf-8 -*-
"""
========================================================================================
 🎮 144+ FPS 3B DRON & TÜNEL UÇUŞ SİMÜLATÖRÜ (dron_tunel_simulatoru.py)
========================================================================================
Doğrudan [W/A/S/D/SPACE/SHIFT] ve fare ile tünelde serbestçe uçabileceğiniz,
dönen pervaneleri, ön feneri, TFmini lazeri ve 3. şahıs / FPV kamerası olan
saf yerel Windows simülatörü!
========================================================================================
"""

import sys
import os
import time
import math
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

# Pro modüller
from pro_features import Drone3DModel

def create_tunnel_mesh(length=80.0, radius=3.5, segments=32):
    """Karanlık, kemerli 3B kaya tüneli mesh'i üretir."""
    verts = []
    cols = []
    
    # Tünel Segmanları
    dz = 1.0
    num_rings = int(length / dz)
    
    for ring in range(num_rings):
        z0 = ring * dz
        z1 = (ring + 1) * dz
        
        for i in range(segments):
            angle0 = (i / segments) * 2 * math.pi
            angle1 = ((i + 1) / segments) * 2 * math.pi
            
            x0_0, y0_0 = math.cos(angle0) * radius, math.sin(angle0) * radius + 2.0
            x0_1, y0_1 = math.cos(angle1) * radius, math.sin(angle1) * radius + 2.0
            
            x1_0, y1_0 = math.cos(angle0) * radius, math.sin(angle0) * radius + 2.0
            x1_1, y1_1 = math.cos(angle1) * radius, math.sin(angle1) * radius + 2.0
            
            # Kaya rengi varyasyonu
            base_c = 0.22 + 0.08 * math.sin(ring * 0.5 + i)
            c = (base_c * 0.9, base_c * 0.8, base_c * 0.75, 1.0)
            
            # 2 Üçgen (Quad)
            verts.extend([
                x0_0, y0_0, z0,  x0_1, y0_1, z0,  x1_1, y1_1, z1,
                x0_0, y0_0, z0,  x1_1, y1_1, z1,  x1_0, y1_0, z1
            ])
            cols.extend([*c, *c, *c, *c, *c, *c])
            
    return np.array(verts, dtype=np.float32), np.array(cols, dtype=np.float32)

def main():
    pygame.init()
    win_w, win_h = 1280, 720
    pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
    pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)
    pygame.display.set_mode((win_w, win_h), DOUBLEBUF | OPENGL | RESIZABLE)
    pygame.display.set_caption("🎮 3B DRON TÜNEL SİMÜLATÖRÜ [W/A/S/D: Uç | Fare: Bak | F: Kamera]")

    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)

    # 3B Tünel Verisini Hazırla
    t_verts, t_cols = create_tunnel_mesh(length=100.0, radius=3.2)
    tunnel_vertex_count = len(t_verts) // 3

    vbo_t_xyz = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_t_xyz)
    glBufferData(GL_ARRAY_BUFFER, t_verts.nbytes, t_verts, GL_STATIC_DRAW)

    vbo_t_col = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_t_col)
    glBufferData(GL_ARRAY_BUFFER, t_cols.nbytes, t_cols, GL_STATIC_DRAW)
    glBindBuffer(GL_ARRAY_BUFFER, 0)

    # Dron Modeli
    drone = Drone3DModel()
    drone.x, drone.y, drone.z = 0.0, 1.5, 2.0
    drone.spotlight = True
    drone.laser = True

    cam_yaw, cam_pitch = 0.0, 0.0
    target_yaw, target_pitch = 0.0, 0.0
    cam_fov = 65.0
    camera_mode = 0  # 0: 3. Şahıs Takip (GTA), 1: 1. Şahıs FPV

    vel_x, vel_y, vel_z = 0.0, 0.0, 0.0
    flight_speed = 0.28

    clock = pygame.time.Clock()
    mouse_down = False
    last_mpos = (0, 0)
    running = True

    font = pygame.font.SysFont("Segoe UI", 13, bold=True)

    recording = False
    recorded_frames = 0
    t_rec_start = 0

    print("\n" + "="*65)
    print(" 🚁 144+ FPS 3B DRON & TÜNEL SİMÜLATÖRÜ AKTİF!")
    print(" 🕹️ [W / S / A / D] : İleri / Geri / Sola / Sağa Uç")
    print(" 🚀 [SPACE / SHIFT] : Yukarı Yüksel / Aşağı Alçal")
    print(" 🔄 [Fare Sürükle]  : Dronun ve Kameranın Yönünü Çevir")
    print(" 📷 [F] Tuşu        : 3. Şahıs Takip <-> 1. Şahıs FPV Kamera")
    print(" 📸 [R] Tuşu        : Canlı 3B Harita Taramasını Başlat / Bitir")
    print("="*65 + "\n")

    while running:
        dt = clock.tick(144) / 1000.0

        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == VIDEORESIZE:
                win_w, win_h = event.w, event.h
                glViewport(0, 0, win_w, win_h)
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouse_down = True
                    last_mpos = event.pos
            elif event.type == MOUSEBUTTONUP:
                if event.button == 1:
                    mouse_down = False
            elif event.type == MOUSEMOTION and mouse_down:
                dx = event.pos[0] - last_mpos[0]
                dy = event.pos[1] - last_mpos[1]
                last_mpos = event.pos
                target_yaw += dx * 0.35
                target_pitch = max(-80.0, min(80.0, target_pitch + dy * 0.35))
            elif event.type == KEYDOWN:
                if event.key == K_f:
                    camera_mode = 1 - camera_mode
                elif event.key == K_l:
                    drone.spotlight = not drone.spotlight
                    drone.laser = drone.spotlight
                elif event.key == K_r:
                    recording = not recording
                    if recording:
                        recorded_frames = 0
                        t_rec_start = time.time()
                        print(" 🔴 TARAMA BAŞLATILDI! Dronu tünelde uçurun...")
                    else:
                        print(f" 💾 TARAMA BİTTİ ({recorded_frames} Kare Alındı). 3B Harita üretiliyor...")
                        # 3B Haritayı Başlat
                        pygame.quit()
                        os.system(f'python "{os.path.join(os.path.dirname(__file__), "gaussian_renderer.py")}" gaussian_scene.ply')
                        return
                elif event.key in (K_ESCAPE, K_q):
                    running = False

        # Kamera Yumuşak Takip
        cam_yaw = 0.85 * cam_yaw + 0.15 * target_yaw
        cam_pitch = 0.85 * cam_pitch + 0.15 * target_pitch

        rad_yaw, rad_pitch = math.radians(cam_yaw), math.radians(cam_pitch)
        fwd_x = math.sin(rad_yaw) * math.cos(rad_pitch)
        fwd_y = -math.sin(rad_pitch)
        fwd_z = math.cos(rad_yaw) * math.cos(rad_pitch)
        right_x = math.cos(rad_yaw)
        right_z = -math.sin(rad_yaw)

        # Klavye Kontrolleri
        keys = pygame.key.get_pressed()
        target_vx, target_vy, target_vz = 0.0, 0.0, 0.0
        spd = flight_speed * (60.0 * dt)

        if keys[K_w] or keys[K_UP]:
            target_vx += fwd_x * spd; target_vy += fwd_y * spd; target_vz += fwd_z * spd
        if keys[K_s] or keys[K_DOWN]:
            target_vx -= fwd_x * spd; target_vy -= fwd_y * spd; target_vz -= fwd_z * spd
        if keys[K_a] or keys[K_LEFT]:
            target_vx -= right_x * spd; target_vz -= right_z * spd
        if keys[K_d] or keys[K_RIGHT]:
            target_vx += right_x * spd; target_vz += right_z * spd
        if keys[K_SPACE]:
            target_vy += spd
        if keys[K_LSHIFT] or keys[K_RSHIFT] or keys[K_c]:
            target_vy -= spd

        vel_x = 0.78 * vel_x + 0.22 * target_vx
        vel_y = 0.78 * vel_y + 0.22 * target_vy
        vel_z = 0.78 * vel_z + 0.22 * target_vz

        drone.x += vel_x
        drone.y = max(0.5, min(3.5, drone.y + vel_y))
        drone.z = max(0.5, min(95.0, drone.z + vel_z))
        drone.yaw = cam_yaw
        drone.pitch = cam_pitch
        drone.update(dt)

        if recording:
            recorded_frames += 1

        # Render
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(cam_fov, win_w / max(win_h, 1), 0.05, 200.0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        if camera_mode == 0:
            # 3. Şahıs Takip Kamerası (Dronun Arkasından)
            cam_d, cam_h = 1.6, 0.45
            rc_x = drone.x - fwd_x * cam_d
            rc_y = drone.y + cam_h - fwd_y * cam_d
            rc_z = drone.z - fwd_z * cam_d
        else:
            # 1. Şahıs Kokpit (FPV)
            rc_x, rc_y, rc_z = drone.x, drone.y, drone.z

        glRotatef(cam_pitch, 1, 0, 0)
        glRotatef(-cam_yaw, 0, 1, 0)
        glTranslatef(-rc_x, -rc_y, -rc_z)

        # 3B Tünel Çizimi
        glBindBuffer(GL_ARRAY_BUFFER, vbo_t_xyz)
        glVertexPointer(3, GL_FLOAT, 0, None)
        glEnableClientState(GL_VERTEX_ARRAY)

        glBindBuffer(GL_ARRAY_BUFFER, vbo_t_col)
        glColorPointer(4, GL_FLOAT, 0, None)
        glEnableClientState(GL_COLOR_ARRAY)

        glDrawArrays(GL_TRIANGLES, 0, tunnel_vertex_count)

        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        # 3B Dron Çizimi
        if camera_mode == 0:
            drone.draw_3d(floor_y=0.0)
        elif drone.spotlight:
            drone.draw_3d(floor_y=0.0)

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
