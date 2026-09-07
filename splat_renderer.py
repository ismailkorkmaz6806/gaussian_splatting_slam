"""
MASt3R-SLAM: 144+ FPS Dedicated NVIDIA RTX 3D Mesh Viewer
----------------------------------------------------------
1. Donanımsal NVIDIA GeForce RTX 4060 GPU Hızlandırması
2. 🏠 Tavan Gizleme / Kesme Sistemi ([G] Tuşu & Kuşbakışında Otomatik)
3. ✂️ Tavan Kesme Yüksekliği Ayarı ([7] / [8] Tuşları)
4. 🗺️ 2B Kat Planı & m² Alan Hesabı ([T] Kuşbakışı)
5. 📏 3B Metrik Lazer Cetvel & Mesafe Ölçüm Aracı ([E] Tuşu)
6. 🎬 60 FPS Pürüzsüz MP4 Video Kaydedici ([V] Tuşu)
7. 🌐 Tek Tıkla Web / HTML 3B Model Çıktısı ([K] Tuşu)
8. 🔍 Büyütme/Küçültme Zoom ([+/- / Tekerlek]), Sağ/Sol Ayna ([Y/A])
9. 🎮 Şık Yarı Saydam Kontrol Paneli ([H] veya [TAB])
"""

import sys
import os
import math
import time
import ctypes
import numpy as np

# Windows Optimus / Hybrid Laptop için NVIDIA GPU Zorlaması
os.environ["SHIM_MCCOMPAT"] = "0x000000001"
os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

from pro_features import (
    CeilingCuller, LaserRuler, VideoRecorder,
    FloorplanEstimator, export_to_standalone_html
)


def draw_frustum(pos, size=0.15, color=(1.0, 0.25, 0.25)):
    """Kamera konisi/piramidi çizer."""
    x, y, z = pos
    s = size
    glColor3f(*color)
    glBegin(GL_LINES)
    glVertex3f(x, y, z); glVertex3f(x - s, y + s*0.6, z + s*1.5)
    glVertex3f(x, y, z); glVertex3f(x + s, y + s*0.6, z + s*1.5)
    glVertex3f(x, y, z); glVertex3f(x - s, y - s*0.6, z + s*1.5)
    glVertex3f(x, y, z); glVertex3f(x + s, y - s*0.6, z + s*1.5)
    glVertex3f(x - s, y + s*0.6, z + s*1.5); glVertex3f(x + s, y + s*0.6, z + s*1.5)
    glVertex3f(x + s, y + s*0.6, z + s*1.5); glVertex3f(x + s, y - s*0.6, z + s*1.5)
    glVertex3f(x + s, y - s*0.6, z + s*1.5); glVertex3f(x - s, y - s*0.6, z + s*1.5)
    glVertex3f(x - s, y - s*0.6, z + s*1.5); glVertex3f(x - s, y + s*0.6, z + s*1.5)
    glEnd()


class HUDControllerSLAMPro:
    """MASt3R-SLAM için 2B Yarı Saydam Kontrol Paneli, Cetvel ve Analiz Kartı."""

    def __init__(self):
        pygame.font.init()
        try:
            self.font_title = pygame.font.SysFont("Segoe UI", 16, bold=True)
            self.font_bold = pygame.font.SysFont("Segoe UI", 13, bold=True)
            self.font_norm = pygame.font.SysFont("Segoe UI", 12)
            self.font_small = pygame.font.SysFont("Segoe UI", 11)
        except Exception:
            self.font_title = pygame.font.Font(None, 19)
            self.font_bold = pygame.font.Font(None, 15)
            self.font_norm = pygame.font.Font(None, 14)
            self.font_small = pygame.font.Font(None, 13)

        self.tex_id = None
        self.show_panel = True
        self.toast_msg = ""
        self.toast_timer = 0

    def set_toast(self, msg, duration=2.8):
        self.toast_msg = msg
        self.toast_timer = time.time() + duration

    def build_surface(self, win_w, win_h, fps, num_verts, num_faces, cam_fov,
                      render_mode, is_flipped, ruler, recorder, room_bounds, top_down_view, culler):
        surf = pygame.Surface((win_w, win_h), pygame.SRCALPHA)
        now = time.time()

        # 1. Bildirim Mesajı (Toast Notification)
        if now < self.toast_timer and self.toast_msg:
            tw, th = 520, 42
            tx = (win_w - tw) // 2
            ty = 16
            pygame.draw.rect(surf, (18, 24, 38, 235), (tx, ty, tw, th), border_radius=8)
            pygame.draw.rect(surf, (0, 180, 255, 200), (tx, ty, tw, th), width=2, border_radius=8)
            t_txt = self.font_bold.render(self.toast_msg, True, (255, 255, 255))
            surf.blit(t_txt, (tx + (tw - t_txt.get_width()) // 2, ty + 11))

        # 2. Üst Durum Çubuğu
        bar_w, bar_h = 670, 36
        bx, by = 16, 16
        pygame.draw.rect(surf, (12, 18, 28, 210), (bx, by, bar_w, bar_h), border_radius=6)
        pygame.draw.rect(surf, (40, 60, 85, 180), (bx, by, bar_w, bar_h), width=1, border_radius=6)

        tavan_durum = f"🏠 Tavan: {'KAPALI (%{:.0f})'.format(culler.cut_ratio*100) if culler.enabled else 'AÇIK'}"
        rec_icon = "🔴 KAYITTA" if recorder.recording else ""
        stat_text = f"⚡ {fps} FPS  |  🏗️ {num_faces:,} Üçgen  |  🔍 Zoom: {cam_fov:.0f}°  |  {tavan_durum}  |  🪞 Ayna: {'DÜZELTİLDİ' if is_flipped else 'ORİJİNAL'} {rec_icon}"
        stat_surf = self.font_bold.render(stat_text, True, (0, 230, 190))
        surf.blit(stat_surf, (bx + 12, by + 9))

        # Kontrol Paneli Butonu
        btn_x = bx + bar_w + 12
        btn_w, btn_h = 175, 36
        pygame.draw.rect(surf, (22, 32, 48, 220), (btn_x, by, btn_w, btn_h), border_radius=6)
        pygame.draw.rect(surf, (0, 150, 255, 180), (btn_x, by, btn_w, btn_h), width=1, border_radius=6)
        h_txt = self.font_bold.render("[H] Kontrol Paneli", True, (255, 255, 255))
        surf.blit(h_txt, (btn_x + 12, by + 9))

        # 3. Cetvel Ölçüm Bilgi Kartı (Aktifken)
        if ruler.active:
            rw, rh = 340, 95
            rx = win_w - rw - 16
            ry = 16
            pygame.draw.rect(surf, (20, 28, 44, 230), (rx, ry, rw, rh), border_radius=8)
            pygame.draw.rect(surf, (255, 70, 100, 220), (rx, ry, rw, rh), width=2, border_radius=8)

            r_title = self.font_bold.render("📏 3B LAZER CETVEL (ÖLÇÜM)", True, (255, 100, 130))
            surf.blit(r_title, (rx + 12, ry + 10))

            if ruler.last_distance is not None:
                d_str = f"📐 Toplam Mesafe : {ruler.last_distance:.2f} Metre"
                d_surf = self.font_bold.render(d_str, True, (0, 255, 180))
                surf.blit(d_surf, (rx + 12, ry + 36))

                dx, dy, dz = abs(ruler.last_delta[0]), abs(ruler.last_delta[1]), abs(ruler.last_delta[2])
                dim_str = f"↔ X: {dx:.2f}m  |  ↕ Y: {dy:.2f}m  |  ↗ Z: {dz:.2f}m"
                dim_surf = self.font_small.render(dim_str, True, (200, 215, 235))
                surf.blit(dim_surf, (rx + 12, ry + 62))
            else:
                pts_cnt = len(ruler.points)
                hint = f"🖱️ {2 - pts_cnt} Noktaya Tıklayın (Sol Tık)" if pts_cnt < 2 else "Hesaplanıyor..."
                h_s = self.font_norm.render(hint, True, (255, 220, 100))
                surf.blit(h_s, (rx + 12, ry + 42))

        # 4. Kuşbakışı Kat Planı & Alan Hesabı Kartı (Top-Down aktifken)
        if top_down_view and room_bounds is not None:
            fw, fh = 370, 115
            fx = win_w - fw - 16
            fy = (120 if ruler.active else 16)
            pygame.draw.rect(surf, (15, 24, 38, 230), (fx, fy, fw, fh), border_radius=8)
            pygame.draw.rect(surf, (0, 200, 255, 200), (fx, fy, fw, fh), width=2, border_radius=8)

            f_title = self.font_bold.render("🗺️ OFİS MİMARİ KAT PLANI VE ALAN", True, (0, 220, 255))
            surf.blit(f_title, (fx + 12, fy + 10))

            dim_txt = f"📐 Boyutlar: {room_bounds['width']:.1f}m (Genişlik) x {room_bounds['length']:.1f}m (Derinlik)"
            surf.blit(self.font_norm.render(dim_txt, True, (220, 235, 255)), (fx + 12, fy + 35))

            area_txt = f"🏢 Tahmini Net Alan: ~{room_bounds['area_m2']:.1f} m²  |  Tavan: {room_bounds['height']:.1f}m"
            surf.blit(self.font_bold.render(area_txt, True, (0, 255, 160)), (fx + 12, fy + 58))

            tavan_hint = f"🏠 Tavan Gizlendi: %{culler.cut_ratio*100:.0f} Seviye ([7]/[8] Ayarla)"
            surf.blit(self.font_small.render(tavan_hint, True, (255, 200, 80)), (fx + 12, fy + 84))

        # 5. Tam Kontrol Paneli Kartı (Açıkken)
        if self.show_panel:
            pw, ph = 430, 500
            px, py = 16, 62

            pygame.draw.rect(surf, (10, 15, 24, 235), (px, py, pw, ph), border_radius=10)
            pygame.draw.rect(surf, (0, 160, 255, 160), (px, py, pw, ph), width=2, border_radius=10)

            title_surf = self.font_title.render("🎮 MASt3R-SLAM KONTROL MERKEZİ", True, (0, 215, 255))
            surf.blit(title_surf, (px + 16, py + 12))

            sub_surf = self.font_small.render("[H] / [TAB] ile paneli gizle veya göster", True, (150, 170, 195))
            surf.blit(sub_surf, (px + 16, py + 34))

            pygame.draw.line(surf, (35, 50, 75), (px + 14, py + 52), (px + pw - 14, py + 52), 1)

            controls = [
                ("🕹️ HAREKET & UÇUŞ", ""),
                ("[W / A / S / D]", "İleri / Sol / Geri / Sağ Serbest Uçuş"),
                ("[SPACE / SHIFT]", "Yukarı Yüksel / Aşağı Alçal"),
                ("[Fare Sol Sürükle]", "360° Serbest Kamera Açısı"),
                ("", ""),
                ("🔍 BÜYÜTME & KÜÇÜLTME (ZOOM)", ""),
                ("[+] / [-]", "🔍 Kamera Yakınlaş (Büyüt) / Uzaklaş (Küçült)"),
                ("[Fare Tekerleği]", "🔍 Hızlı Kamera Büyüt / Küçült"),
                ("[0] (Sıfır)", "Standart Zoom Açısına Sıfırla (60°)"),
                ("", ""),
                ("🏠 MİMARİ & TAVAN KESME ARAÇLARI", ""),
                ("[G] Tuşu", "🏠 Tavanı Gizle / Aç (İç Mekanı Kuşbakışı Gör)"),
                ("[7] / [8]", "✂️ Tavan Kesme Yüksekliğini Alçalt / Yükselt"),
                ("[T] Tuşu", "🗺️ 90° Kuşbakışı Kat Planı ve m² Alan Hesabı"),
                ("[E] Tuşu", "📏 3B Lazer Cetvel (İki Nokta Arası Metre Ölçümü)"),
                ("", ""),
                ("🚀 MEŞ & PRO ÇIKTI", ""),
                ("[M] Tuşu", f"🎨 Görünüm Değiştir ({render_mode.upper()})"),
                ("[L] Tuşu", "💡 Dinamik Işıklandırma Aç / Kapat"),
                ("[V] Tuşu", "🎬 60 FPS MP4 Video Kaydını Başlat / Durdur"),
                ("[K] Tuşu", "🌐 Bağımsız Web / HTML 3B Modelini Dışa Aktar"),
                ("[Y] / [A]", "🪞 Sağ / Sol Yönünü Tersine Çevir (Ayna)"),
                ("[P] Tuşu", "🎥 Sinematik Tur | [R]: Başa Sıfırla"),
            ]

            cy = py + 58
            for key_txt, desc_txt in controls:
                if not desc_txt and key_txt:
                    h_s = self.font_bold.render(key_txt, True, (255, 200, 80))
                    surf.blit(h_s, (px + 16, cy))
                    cy += 19
                elif not key_txt and not desc_txt:
                    cy += 3
                else:
                    k_s = self.font_bold.render(key_txt, True, (255, 255, 255))
                    kw = k_s.get_width() + 8
                    pygame.draw.rect(surf, (28, 42, 64, 240), (px + 16, cy, kw, 18), border_radius=4)
                    pygame.draw.rect(surf, (55, 85, 125, 200), (px + 16, cy, kw, 18), width=1, border_radius=4)
                    surf.blit(k_s, (px + 20, cy + 1))

                    d_s = self.font_norm.render(desc_txt, True, (220, 230, 245))
                    surf.blit(d_s, (px + 16 + kw + 8, cy + 1))
                    cy += 19

        return surf

    def render_gl(self, win_w, win_h, fps, num_verts, num_faces, cam_fov,
                  render_mode, is_flipped, ruler, recorder, room_bounds, top_down_view, culler):
        surf = self.build_surface(win_w, win_h, fps, num_verts, num_faces, cam_fov,
                                  render_mode, is_flipped, ruler, recorder, room_bounds, top_down_view, culler)

        glPushAttrib(GL_ALL_ATTRIB_BITS)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glDisable(GL_CULL_FACE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_TEXTURE_2D)

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, win_w, win_h, 0, -1, 1)

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        if hasattr(pygame.image, 'tobytes'):
            rgba_data = pygame.image.tobytes(surf, "RGBA", True)
        else:
            rgba_data = pygame.image.tostring(surf, "RGBA", True)

        if self.tex_id is None:
            self.tex_id = glGenTextures(1)

        glBindTexture(GL_TEXTURE_2D, self.tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, win_w, win_h, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba_data)

        glColor4f(1.0, 1.0, 1.0, 1.0)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(0, win_h)
        glTexCoord2f(1, 0); glVertex2f(win_w, win_h)
        glTexCoord2f(1, 1); glVertex2f(win_w, 0)
        glTexCoord2f(0, 1); glVertex2f(0, 0)
        glEnd()

        glBindTexture(GL_TEXTURE_2D, 0)

        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glPopAttrib()


def view_mast3r_map(ply_path="mast3r_map.ply"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    target_ply = os.path.join(base_dir, ply_path)
    cache_path = target_ply.replace(".ply", "_cache.npz")

    verts, cols, norms, faces = None, None, None, None
    render_mode = "mesh"

    if os.path.exists(cache_path):
        t0 = time.time()
        data = np.load(cache_path, allow_pickle=True)
        if "verts" in data and "faces" in data:
            verts = np.ascontiguousarray(data["verts"], dtype=np.float32)
            cols = np.ascontiguousarray(data["cols"], dtype=np.float32)
            faces = np.ascontiguousarray(data["faces"], dtype=np.uint32)
            norms = np.ascontiguousarray(data["norms"], dtype=np.float32) if "norms" in data else None
            print(f" ⚡ Katı Meş Önbellekten Yüklendi: {len(verts):,} Köşe | {len(faces):,} Üçgen ({time.time()-t0:.2f}s)")
        else:
            verts = np.ascontiguousarray(data["pts"], dtype=np.float32)
            cols = np.ascontiguousarray(data["cols"], dtype=np.float32)
            faces = None
            render_mode = "points"
    elif os.path.exists(target_ply):
        print(f" 📂 3B Model Yükleniyor: {target_ply}...")
        import trimesh
        mesh = trimesh.load(target_ply, process=False)
        verts = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
        if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            cols = np.ascontiguousarray(mesh.visual.vertex_colors[:, :3] / 255.0, dtype=np.float32)
        else:
            cols = np.ones_like(verts, dtype=np.float32) * 0.8
        if hasattr(mesh, 'faces') and len(mesh.faces) > 0:
            faces = np.ascontiguousarray(mesh.faces, dtype=np.uint32)
            render_mode = "mesh"
    else:
        print("❌ HATA: Harita dosyası bulunamadı!")
        return

    if norms is None:
        norms = np.zeros_like(verts, dtype=np.float32)
        norms[:, 1] = 1.0

    traj_path = os.path.join(base_dir, "mast3r_trajectory.npz")
    auto_waypoints = []
    if os.path.exists(traj_path):
        traj_data = np.load(traj_path, allow_pickle=True)
        auto_waypoints = traj_data["positions"].copy()

    # 🪞 SAĞ/SOL DÜZELTMESİ
    is_flipped = True
    verts[:, 0] = -verts[:, 0]
    if norms is not None:
        norms[:, 0] = -norms[:, 0]
    if len(auto_waypoints) > 0:
        auto_waypoints[:, 0] = -auto_waypoints[:, 0]

    room_bounds = FloorplanEstimator.calculate_bounds(verts)

    # Modül Yöneticileri
    culler = CeilingCuller()
    culler.init_from_bounds(verts)
    ruler = LaserRuler()
    recorder = VideoRecorder()
    hud = HUDControllerSLAMPro()
    hud.set_toast("✨ MASt3R-SLAM Hazır! [G] Tavanı Gizle, [T] Kuşbakışı, [E] Cetvel, [H] Menü", 4.0)

    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
    pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 2)

    win_w, win_h = 1280, 720
    pygame.display.set_mode((win_w, win_h), DOUBLEBUF | OPENGL | RESIZABLE)

    gpu_vendor = glGetString(GL_VENDOR).decode(errors='replace')
    gpu_renderer = glGetString(GL_RENDERER).decode(errors='replace')
    print(f" 🚀 Aktif 3B GPU Donanımı: {gpu_renderer} ({gpu_vendor})")

    pygame.display.set_caption(f"MASt3R-SLAM Ultimate [{gpu_renderer}]")

    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)
    glDisable(GL_CULL_FACE)
    glShadeModel(GL_SMOOTH)
    glEnable(GL_NORMALIZE)
    glEnable(GL_MULTISAMPLE)

    # Işıklandırma Ayarları
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_AMBIENT, [0.45, 0.45, 0.45, 1.0])
    glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.85, 0.85, 0.85, 1.0])
    glLightfv(GL_LIGHT0, GL_SPECULAR, [0.25, 0.25, 0.25, 1.0])
    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE)
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [0.3, 0.3, 0.3, 1.0])
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 32.0)

    # GPU VBO'ları
    vbo_verts = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_verts)
    glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)

    vbo_cols = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_cols)
    glBufferData(GL_ARRAY_BUFFER, cols.nbytes, cols, GL_STATIC_DRAW)

    vbo_norms = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_norms)
    glBufferData(GL_ARRAY_BUFFER, norms.nbytes, norms, GL_STATIC_DRAW)

    vbo_faces = None
    if faces is not None and len(faces) > 0:
        flat_faces = faces.flatten()
        vbo_faces = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_faces)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, flat_faces.nbytes, flat_faces, GL_STATIC_DRAW)
        num_indices = len(flat_faces)
    else:
        num_indices = 0

    # Zemin Izgarası VBO (Y=0)
    grid_verts = []
    for i in range(-50, 51, 2):
        fi = float(i)
        grid_verts += [fi, 0.0, -50.0,  fi, 0.0,  50.0]
        grid_verts += [-50.0, 0.0, fi,  50.0, 0.0, fi]
    grid_arr = np.array(grid_verts, dtype=np.float32)
    grid_n = len(grid_arr) // 3
    vbo_grid = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_grid)
    glBufferData(GL_ARRAY_BUFFER, grid_arr.nbytes, grid_arr, GL_STATIC_DRAW)
    glBindBuffer(GL_ARRAY_BUFFER, 0)

    # Yörünge Çizgisi VBO
    vbo_traj = None
    traj_n = 0
    if len(auto_waypoints) > 0:
        traj_arr = auto_waypoints.astype(np.float32)
        traj_n = len(traj_arr)
        vbo_traj = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_traj)
        glBufferData(GL_ARRAY_BUFFER, traj_arr.nbytes, traj_arr, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    glClearColor(0.06, 0.08, 0.12, 1.0)

    if len(auto_waypoints) > 0:
        drone_x, drone_y, drone_z = float(auto_waypoints[0][0]), float(auto_waypoints[0][1]), float(auto_waypoints[0][2])
    else:
        drone_x, drone_y, drone_z = 0.0, 1.5, 0.0

    cam_yaw, cam_pitch = 0.0, 0.0
    target_yaw, target_pitch = 0.0, 0.0
    vel_x, vel_y, vel_z = 0.0, 0.0, 0.0
    flight_speed = 0.22

    cam_fov = 60.0
    auto_tour = False
    tour_progress = 0.0
    tour_speed = 1.0
    lighting_enabled = True
    show_frustums = True
    top_down_view = False

    clock = pygame.time.Clock()
    mouse_down_left = False
    last_mouse_pos = (0, 0)
    running = True

    fps_timer = time.time()
    frame_count = 0
    current_fps = 144
    num_f = len(faces) if faces is not None else 0

    while running:
        dt = clock.tick(144) / 1000.0
        frame_count += 1
        if time.time() - fps_timer >= 0.5:
            current_fps = int(frame_count / (time.time() - fps_timer))
            pygame.display.set_caption(f"MASt3R-SLAM Ultimate [{current_fps} FPS | {num_f:,} Üçgen | FOV: {cam_fov:.0f}°]")
            frame_count = 0
            fps_timer = time.time()

        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == VIDEORESIZE:
                win_w, win_h = event.w, event.h
                glViewport(0, 0, win_w, win_h)
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:
                    if ruler.active:
                        picked = ruler.add_point_from_screen(event.pos[0], event.pos[1], win_w, win_h)
                        if picked is not None and ruler.last_distance is not None:
                            hud.set_toast(f"📏 Ölçüldü: {ruler.last_distance:.2f} Metre", 3.0)
                    else:
                        mouse_down_left = True
                        last_mouse_pos = event.pos
                elif event.button == 4:
                    cam_fov = max(15.0, cam_fov - 3.0)
                    hud.set_toast(f"🔍 Zoom In: {cam_fov:.0f}°", 1.2)
                elif event.button == 5:
                    cam_fov = min(110.0, cam_fov + 3.0)
                    hud.set_toast(f"🔍 Zoom Out: {cam_fov:.0f}°", 1.2)
            elif event.type == MOUSEBUTTONUP:
                if event.button == 1:
                    mouse_down_left = False
            elif event.type == MOUSEMOTION:
                dx = event.pos[0] - last_mouse_pos[0]
                dy = event.pos[1] - last_mouse_pos[1]
                last_mouse_pos = event.pos
                if mouse_down_left and not ruler.active:
                    target_yaw += dx * 0.32
                    target_pitch += dy * 0.32
                    target_pitch = max(-89.0, min(89.0, target_pitch))
            elif event.type == KEYDOWN:
                if event.key in (K_h, K_TAB):
                    hud.show_panel = not hud.show_panel
                elif event.key == K_g:
                    c_state = culler.toggle()
                    hud.set_toast(f"🏠 Tavan: {'GİZLENDİ (İç Mekan Açıldı)' if c_state else 'GÖSTERİLİYOR'}", 2.5)
                elif event.key in (K_7, K_KP7):
                    r = culler.adjust_cut(-0.05)
                    culler.enabled = True
                    hud.set_toast(f"✂️ Tavan Kesme Seviyesi: %{r*100:.0f} ({culler.cut_y:.2f}m)", 1.5)
                elif event.key in (K_8, K_KP8):
                    r = culler.adjust_cut(+0.05)
                    culler.enabled = True
                    hud.set_toast(f"✂️ Tavan Kesme Seviyesi: %{r*100:.0f} ({culler.cut_y:.2f}m)", 1.5)
                elif event.key == K_e:
                    ruler.toggle()
                    hud.set_toast("📏 Lazer Cetvel: " + ("AKTİF (Ölçmek İçin 2 Noktaya Tıklayın)" if ruler.active else "KAPATILDI"), 3.0)
                elif event.key == K_v:
                    res_msg = recorder.toggle(win_w, win_h, fps=60)
                    hud.set_toast(res_msg, 3.5)
                elif event.key == K_k:
                    html_file = export_to_standalone_html(verts, cols, output_html="3d_scene.html")
                    hud.set_toast(f"🌐 Web 3B Modeli Kaydedildi: {html_file}", 3.5)
                elif event.key in (K_PLUS, K_KP_PLUS, K_EQUALS):
                    cam_fov = max(15.0, cam_fov - 4.0)
                    hud.set_toast(f"🔍 Yakınlaşıldı (Zoom In): {cam_fov:.0f}°", 1.5)
                elif event.key in (K_MINUS, K_KP_MINUS):
                    cam_fov = min(110.0, cam_fov + 4.0)
                    hud.set_toast(f"🔍 Uzaklaşıldı (Zoom Out): {cam_fov:.0f}°", 1.5)
                elif event.key in (K_0, K_KP0):
                    cam_fov = 60.0
                    hud.set_toast("🔍 Zoom Sıfırlandı (60°)", 1.5)
                elif event.key in (K_y, K_a):
                    is_flipped = not is_flipped
                    verts[:, 0] = -verts[:, 0]
                    if norms is not None:
                        norms[:, 0] = -norms[:, 0]
                    glBindBuffer(GL_ARRAY_BUFFER, vbo_verts)
                    glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)
                    glBindBuffer(GL_ARRAY_BUFFER, vbo_norms)
                    glBufferData(GL_ARRAY_BUFFER, norms.nbytes, norms, GL_STATIC_DRAW)
                    glBindBuffer(GL_ARRAY_BUFFER, 0)
                    if len(auto_waypoints) > 0:
                        auto_waypoints[:, 0] = -auto_waypoints[:, 0]
                        traj_arr = auto_waypoints.astype(np.float32)
                        glBindBuffer(GL_ARRAY_BUFFER, vbo_traj)
                        glBufferData(GL_ARRAY_BUFFER, traj_arr.nbytes, traj_arr, GL_STATIC_DRAW)
                        glBindBuffer(GL_ARRAY_BUFFER, 0)
                    drone_x = -drone_x
                    room_bounds = FloorplanEstimator.calculate_bounds(verts)
                    culler.init_from_bounds(verts)
                    hud.set_toast(f"🪞 Yön Değiştirildi: {'TERS (DÜZELTİLDİ)' if is_flipped else 'ORİJİNAL'}", 2.5)
                elif event.key == K_p and len(auto_waypoints) > 0:
                    auto_tour = not auto_tour
                    target_yaw, target_pitch = 0.0, 0.0
                    cam_yaw, cam_pitch = 0.0, 0.0
                    hud.set_toast("🎥 Sinematik Tur: " + ("BAŞLATILDI" if auto_tour else "DURDURULDU"), 2.0)
                elif event.key == K_m:
                    if render_mode == "mesh" and vbo_faces is not None:
                        render_mode = "wireframe"
                    elif render_mode == "wireframe":
                        render_mode = "points"
                    else:
                        render_mode = "mesh" if vbo_faces is not None else "points"
                    hud.set_toast(f"🎨 Görünüm Modu: {render_mode.upper()}", 1.5)
                elif event.key == K_l:
                    lighting_enabled = not lighting_enabled
                    if lighting_enabled:
                        glEnable(GL_LIGHTING)
                    else:
                        glDisable(GL_LIGHTING)
                    hud.set_toast(f"💡 Işıklandırma: {'AÇIK' if lighting_enabled else 'KAPALI'}", 1.5)
                elif event.key == K_f:
                    show_frustums = not show_frustums
                    hud.set_toast(f"📐 Frustumlar: {'GÖSTERİLİYOR' if show_frustums else 'GİZLENDİ'}", 1.5)
                elif event.key == K_1 and len(auto_waypoints) > 0:
                    auto_tour = False; p = auto_waypoints[0]; drone_x, drone_y, drone_z = float(p[0]), float(p[1]), float(p[2])
                    hud.set_toast("🚪 Konum 1'e Işınlanıldı (Giriş)", 1.5)
                elif event.key == K_2 and len(auto_waypoints) > 0:
                    auto_tour = False; p = auto_waypoints[len(auto_waypoints)//3]; drone_x, drone_y, drone_z = float(p[0]), float(p[1]), float(p[2])
                    hud.set_toast("🚪 Konum 2'ye Işınlanıldı (Orta 1)", 1.5)
                elif event.key == K_3 and len(auto_waypoints) > 0:
                    auto_tour = False; p = auto_waypoints[2*len(auto_waypoints)//3]; drone_x, drone_y, drone_z = float(p[0]), float(p[1]), float(p[2])
                    hud.set_toast("🚪 Konum 3'e Işınlanıldı (Orta 2)", 1.5)
                elif event.key == K_4 and len(auto_waypoints) > 0:
                    auto_tour = False; p = auto_waypoints[-1]; drone_x, drone_y, drone_z = float(p[0]), float(p[1]), float(p[2])
                    hud.set_toast("🚪 Konum 4'e Işınlanıldı (Son Nokta)", 1.5)
                elif event.key == K_t:
                    top_down_view = not top_down_view
                    auto_tour = False
                    if top_down_view:
                        culler.enabled = True
                        target_pitch, target_yaw = 89.0, 0.0; cam_pitch, cam_yaw = 89.0, 0.0
                        drone_x = float(np.mean(verts[:, 0])); drone_y = float(np.max(verts[:, 1])) + 14.0; drone_z = float(np.mean(verts[:, 2]))
                        hud.set_toast("🗺️ Kuşbakışı Kat Planı & Tavan Kesildi (İç Mekan Görünür)", 3.0)
                    else:
                        target_pitch, target_yaw = 0.0, 0.0; cam_pitch, cam_yaw = 0.0, 0.0
                        hud.set_toast("🕹️ Serbest Uçuş Moduna Dönüldü", 1.5)
                elif event.key == K_r:
                    auto_tour = False; tour_progress = 0.0; top_down_view = False
                    culler.enabled = False
                    if len(auto_waypoints) > 0:
                        drone_x, drone_y, drone_z = float(auto_waypoints[0][0]), float(auto_waypoints[0][1]), float(auto_waypoints[0][2])
                    else:
                        drone_x, drone_y, drone_z = 0.0, 1.5, 0.0
                    target_yaw, target_pitch = 0.0, 0.0; cam_yaw, cam_pitch = 0.0, 0.0; vel_x, vel_y, vel_z = 0.0, 0.0, 0.0
                    cam_fov = 60.0
                    hud.set_toast("🔄 Konum ve Kamera Sıfırlandı", 1.5)
                elif event.key in (K_ESCAPE, K_q):
                    running = False

        cam_yaw = 0.85 * cam_yaw + 0.15 * target_yaw
        cam_pitch = 0.85 * cam_pitch + 0.15 * target_pitch

        rad_yaw, rad_pitch = math.radians(cam_yaw), math.radians(cam_pitch)
        fwd_x = math.sin(rad_yaw) * math.cos(rad_pitch)
        fwd_y = -math.sin(rad_pitch)
        fwd_z = math.cos(rad_yaw) * math.cos(rad_pitch)
        right_x = math.cos(rad_yaw)
        right_z = -math.sin(rad_yaw)

        if auto_tour and len(auto_waypoints) > 1:
            tour_progress += dt * (tour_speed * 85.0)
            if tour_progress >= len(auto_waypoints) - 1:
                tour_progress = 0.0
            idx0 = int(tour_progress)
            idx1 = min(idx0 + 1, len(auto_waypoints) - 1)
            alpha = tour_progress - idx0
            pos0, pos1 = auto_waypoints[idx0], auto_waypoints[idx1]
            t_pos = (1.0 - alpha) * pos0 + alpha * pos1
            drone_x, drone_y, drone_z = float(t_pos[0]), float(t_pos[1]), float(t_pos[2])
        else:
            keys = pygame.key.get_pressed()
            target_vx, target_vy, target_vz = 0.0, 0.0, 0.0
            spd = flight_speed * (60.0 * dt)
            if keys[K_w] or keys[K_UP]: target_vx += fwd_x * spd; target_vy += fwd_y * spd; target_vz += fwd_z * spd
            if keys[K_s] or keys[K_DOWN]: target_vx -= fwd_x * spd; target_vy -= fwd_y * spd; target_vz -= fwd_z * spd
            if keys[K_a] or keys[K_LEFT]: target_vx -= right_x * spd; target_vz -= right_z * spd
            if keys[K_d] or keys[K_RIGHT]: target_vx += right_x * spd; target_vz += right_z * spd
            if keys[K_SPACE]: target_vy += spd
            if keys[K_LSHIFT] or keys[K_RSHIFT]: target_vy -= spd

            vel_x = 0.78 * vel_x + 0.22 * target_vx
            vel_y = 0.78 * vel_y + 0.22 * target_vy
            vel_z = 0.78 * vel_z + 0.22 * target_vz
            drone_x += vel_x; drone_y += vel_y; drone_z += vel_z

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(cam_fov, (win_w / max(win_h, 1)), 0.02, 300.0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glRotatef(cam_pitch, 1, 0, 0)
        glRotatef(-cam_yaw, 0, 1, 0)
        glTranslatef(-drone_x, -drone_y, -drone_z)

        # 🏠 Tavan Kesme Düzlemi Uygula
        culler.apply_gl()

        # Işık Konumu
        glLightfv(GL_LIGHT0, GL_POSITION, [drone_x + 1.0, drone_y + 3.0, drone_z + 1.0, 1.0])

        # Zemin Izgarası
        glDisable(GL_LIGHTING)
        glColor4f(0.18, 0.28, 0.42, 0.35)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_grid)
        glVertexPointer(3, GL_FLOAT, 0, None)
        glEnableClientState(GL_VERTEX_ARRAY)
        glDrawArrays(GL_LINES, 0, grid_n)
        glDisableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        # Kat Planı Mimari Çerçeve (Kuşbakışında)
        if top_down_view:
            FloorplanEstimator.draw_blueprint_grid(room_bounds)

        # Yeşil Yörünge Çizgisi
        if vbo_traj is not None and traj_n > 0:
            glLineWidth(2.5)
            glColor3f(0.1, 0.95, 0.35)
            glBindBuffer(GL_ARRAY_BUFFER, vbo_traj)
            glVertexPointer(3, GL_FLOAT, 0, None)
            glEnableClientState(GL_VERTEX_ARRAY)
            glDrawArrays(GL_LINE_STRIP, 0, traj_n)
            glDisableClientState(GL_VERTEX_ARRAY)
            glBindBuffer(GL_ARRAY_BUFFER, 0)

        # Kamera Frustumları
        if show_frustums and len(auto_waypoints) > 0:
            step_kf = max(1, len(auto_waypoints) // 30)
            for kf_pos in auto_waypoints[::step_kf]:
                draw_frustum(kf_pos, size=0.10, color=(1.0, 0.25, 0.25))

        # 📏 3B Lazer Cetvel Çizimi
        ruler.draw_3d()

        # 3B Meş Çizimi
        if lighting_enabled:
            glEnable(GL_LIGHTING)

        glBindBuffer(GL_ARRAY_BUFFER, vbo_verts)
        glVertexPointer(3, GL_FLOAT, 0, None)
        glEnableClientState(GL_VERTEX_ARRAY)

        glBindBuffer(GL_ARRAY_BUFFER, vbo_cols)
        glColorPointer(3, GL_FLOAT, 0, None)
        glEnableClientState(GL_COLOR_ARRAY)

        glBindBuffer(GL_ARRAY_BUFFER, vbo_norms)
        glNormalPointer(GL_FLOAT, 0, None)
        glEnableClientState(GL_NORMAL_ARRAY)

        if render_mode == "mesh" and vbo_faces is not None:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_faces)
            glDrawElements(GL_TRIANGLES, num_indices, GL_UNSIGNED_INT, None)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        elif render_mode == "wireframe" and vbo_faces is not None:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_faces)
            glDrawElements(GL_TRIANGLES, num_indices, GL_UNSIGNED_INT, None)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        else:
            glPointSize(5.0)
            glDrawArrays(GL_POINTS, 0, len(verts))

        glDisableClientState(GL_NORMAL_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        # Tavan Kesme Düzlemini Kaldır (2B Arayüz için)
        culler.restore_gl()

        # 2B Yarı Saydam Kontrol Paneli & HUD Çizimi
        hud.render_gl(win_w, win_h, current_fps, len(verts), num_f, cam_fov,
                      render_mode, is_flipped, ruler, recorder, room_bounds, top_down_view, culler)

        # MP4 Video Kare Kaydı
        if recorder.recording:
            recorder.capture_frame(win_w, win_h)

        pygame.display.flip()

    if recorder.recording:
        recorder.toggle(win_w, win_h)

    pygame.quit()


if __name__ == "__main__":
    ply = sys.argv[1] if len(sys.argv) > 1 else "mast3r_map.ply"
    view_mast3r_map(ply)
