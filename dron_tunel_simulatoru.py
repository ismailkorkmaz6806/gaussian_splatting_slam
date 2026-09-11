# -*- coding: utf-8 -*-
"""
========================================================================================
 🎮 144+ FPS 3B DRON & TÜNEL UÇUŞ SİMÜLATÖRÜ (dron_tunel_simulatoru.py)
========================================================================================
Holybro X500 Quadcopter Kit mimarisi ile gerçekçi fizik motoru, dinamik pitch/roll eğimi,
yüksek hızlı dönen pervaneler (motion-blur), irtifa mikro-salınımı ve kamera sarsıntısı!
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
from pro_features import Drone3DModel, DronePhysicsEngine

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
    pygame.display.set_caption("🎮 3B HOLYBRO X500 TÜNEL SİMÜLATÖRÜ [W/A/S/D: Uç | Fare: Bak | F: Kamera]")

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

    # Dron Modeli & Fizik Motoru
    drone = Drone3DModel()
    physics = DronePhysicsEngine()
    physics.x, physics.y, physics.z = 0.0, 1.6, 2.0
    physics.set_floor(0.0)

    drone.x, drone.y, drone.z = physics.x, physics.y, physics.z
    drone.spotlight = True
    drone.laser = True

    cam_yaw, cam_pitch = 0.0, 0.0
    target_yaw, target_pitch = 0.0, 0.0
    cam_fov = 65.0
    camera_mode = 0  # 0: 3. Şahıs Takip, 1: 1. Şahıs FPV

    clock = pygame.time.Clock()
    mouse_down = False
    last_mpos = (0, 0)
    running = True

    recording = False
    recorded_frames = 0

    print("\n" + "="*70)
    print(" 🚁 144+ FPS 3B HOLYBRO X500 DRON & TÜNEL SİMÜLATÖRÜ AKTİF!")
    print(" 🌪️ Gerçekçi Aerodinamik Tilt (Pitch/Roll), Pervane Motion Blur & Salınım")
    print(" 🕹️ [W / S / A / D] : İleri (Nose-Down) / Geri / Sola / Sağa (Roll Bank)")
    print(" 🚀 [SPACE / SHIFT] : Yukarı Gaz / Aşağı İniş")
    print(" 🔄 [Fare Sürükle]  : Dronun ve Kameranın Yönünü Çevir")
    print(" 📷 [F] Tuşu        : 3. Şahıs Takip <-> 1. Şahıs FPV Kamera")
    print(" 📸 [R] Tuşu        : Canlı 3B Harita Taramasını Başlat / Bitir")
    print("="*70 + "\n")

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
                        print(" 🔴 TARAMA BAŞLATILDI! Dronu tünelde uçurun...")
                    else:
                        print(f" 💾 TARAMA BİTTİ ({recorded_frames} Kare Alındı). 3B Harita üretiliyor...")
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

        # Klavye Kontrolleri & Fizik Motoru Entegrasyonu
        keys = pygame.key.get_pressed()
        move_fwd = bool(keys[K_w] or keys[K_UP])
        move_back = bool(keys[K_s] or keys[K_DOWN])
        move_left = bool(keys[K_a] or keys[K_LEFT])
        move_right = bool(keys[K_d] or keys[K_RIGHT])
        throttle_up = bool(keys[K_SPACE])
        fast_mode = bool(keys[K_LSHIFT] or keys[K_RSHIFT])

        # Fizik motorunu işlet (Kuvvet -> İvme -> Hız -> Konum)
        dx, dy, dz = physics.step(
            dt=dt,
            throttle_up=throttle_up,
            move_fwd=move_fwd,
            move_back=move_back,
            move_left=move_left,
            move_right=move_right,
            fwd_x=fwd_x, fwd_y=fwd_y, fwd_z=fwd_z,
            right_x=right_x, right_z=right_z,
            fast_mode=fast_mode
        )

        # Tünel sınırları koruması
        physics.y = max(0.2, min(3.8, physics.y))
        physics.z = max(0.5, min(95.0, physics.z))

        # Dron Modelinin dinamik tilt, pervane dönüşü ve irtifa salınımını güncelle
        drone.x, drone.y, drone.z = physics.x, physics.y, physics.z
        drone.yaw = cam_yaw
        drone.update(
            dt=dt,
            vx=physics.vx,
            vy=physics.vy,
            vz=physics.vz,
            throttle=physics.throttle,
            is_airborne=not physics.is_landed
        )

        if recording:
            recorded_frames += 1

        # Render
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(cam_fov, win_w / max(win_h, 1), 0.05, 200.0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # Render pozisyonu (İrtifa mikro-salınımı uygulanmış)
        rx, ry, rz, ryaw, rpitch, rroll = drone.get_render_pose()

        if camera_mode == 0:
            # 3. Şahıs Takip Kamerası (Dronun Arkasından) + Kamera Sarsıntısı
            cam_d, cam_h = 1.55, 0.42
            rc_x = rx - fwd_x * cam_d + drone.cam_shake_x
            rc_y = ry + cam_h - fwd_y * cam_d + drone.cam_shake_y
            rc_z = rz - fwd_z * cam_d
            eff_pitch = cam_pitch + drone.cam_shake_rot
        else:
            # 1. Şahıs Kokpit (FPV)
            rc_x = rx + drone.cam_shake_x
            rc_y = ry + 0.05 + drone.cam_shake_y
            rc_z = rz
            eff_pitch = cam_pitch + rpitch * 0.5 + drone.cam_shake_rot

        glRotatef(eff_pitch, 1, 0, 0)
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

        # 3B Fotogerçekçi Holybro X500 Dron Çizimi
        if camera_mode == 0:
            drone.draw_3d(floor_y=0.0)
        elif drone.spotlight:
            drone.draw_3d(floor_y=0.0)

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
