"""
========================================================================================
MASt3R-SLAM: 144+ FPS 3D Gaussian Splatting (3DGS) Donanımsal Görselleştirici
========================================================================================
Bu dosya, hesaplanan 3D Gaussian Splatting modelini (6.2M+ Splat) donanımsal GPU
OpenGL hızlandırması ile 144+ FPS akıcılıkta ekrana çizen ANA GÖRÜNTÜLEYİCİDİR.

İçerdiği Başlıca Sistemler:
1. ⚡ Donanımsal NVIDIA GeForce RTX 4060 GPU Hızlandırması & VBO Çizim Motoru
2. 🎮 HUDControllerPro: Yarı saydam 2B Kontrol Paneli, Cetvel Kartı & Toast Bildirimleri
3. 🏠 CeilingCuller: Tavan Gizleme / Kesme Sistemi ([G] ve [7]/[8] Tuşları)
4. 🗺️ FloorplanEstimator: 90° Kuşbakışı Mimari Kat Planı ve m² Alan Hesabı ([T] Tuşu)
5. 📏 LaserRuler: 3B Metrik Lazer Cetvel & Mesafe Ölçüm Aracı ([E] Tuşu)
6. 🎬 VideoRecorder: 60 FPS Pürüzsüz MP4 Video Kaydedici ([V] Tuşu)
7. 🌐 Web 3B Çıktı: Tek Tıkla Web/HTML 3B Model Dışa Aktarıcı ([K] Tuşu)
8. 🔍 Kamera Zoom ([+/- / Tekerlek]), Splat Boyutu ([X/C]), Sağ/Sol Ayna ([Y/A])
========================================================================================
"""

import sys
import os

# Eğer venv dışından doğrudan system python ile çalıştırıldıysa otomatik venv python'a geç
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_PY = os.path.join(CURR_DIR, "venv", "bin", "python3")
if os.path.exists(VENV_PY) and os.path.abspath(sys.executable) != os.path.abspath(VENV_PY):
    try:
        import OpenGL
    except ImportError:
        os.execv(VENV_PY, [VENV_PY] + sys.argv)

import math
import time
import numpy as np

# Linux / Windows Optimus Hibrit Laptoplar için Harici NVIDIA RTX GPU ve Fallback Ayarları
os.environ["SHIM_MCCOMPAT"] = "0x000000001"
os.environ["__GL_SYNC_TO_VBLANK"] = "0"
os.environ["vblank_mode"] = "0"
os.environ["SDL_VIDEO_X11_NODIRECTCOLOR"] = "1"
os.environ["SDL_VIDEODRIVER"] = "x11"
os.environ["GDK_BACKEND"] = "x11"
os.environ["QT_QPA_PLATFORM"] = "xcb"

import ctypes
import OpenGL
OpenGL.USE_ACCELERATE = False
OpenGL.ERROR_CHECKING = False
OpenGL.ERROR_LOGGING = False
OpenGL.CONTEXT_CHECKING = False
import OpenGL.raw.GL.VERSION.GL_1_1 as raw_gl

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

# Pro özellik modüllerinin içe aktarılması
from pro_features import (
    CeilingCuller, LaserRuler, VideoRecorder,
    FloorplanEstimator, export_to_standalone_html,
    ProximityHeatmapEngine, OctomapEngine, Drone3DModel,
    DronePhysicsEngine, OpticalFlowHUD
)
from tunnel_report_generator import generate_tunnel_report





def draw_frustum(pos, size=0.15, color=(1.0, 0.25, 0.25)):
    """
    Kameranın o anki konumunu ve baktığı yönü gösteren 3B piramit (frustum) çizer.
    
    Parametreler:
        pos   : Kameranın (X, Y, Z) dünya koordinatları
        size  : Çizilecek koninin boyutu
        color : RGB renk kodu (Örn: Kırmızı [1.0, 0.25, 0.25])
    """
    x, y, z = pos
    s = size
    glColor3f(*color)
    glBegin(GL_LINES)
    # Kameranın tepe noktasından 4 köşeye uzanan çizgiler
    glVertex3f(x, y, z); glVertex3f(x - s, y + s*0.6, z + s*1.5)
    glVertex3f(x, y, z); glVertex3f(x + s, y + s*0.6, z + s*1.5)
    glVertex3f(x, y, z); glVertex3f(x - s, y - s*0.6, z + s*1.5)
    glVertex3f(x, y, z); glVertex3f(x + s, y - s*0.6, z + s*1.5)
    # Taban dikdörtgenini oluşturan 4 kenar çizgisi
    glVertex3f(x - s, y + s*0.6, z + s*1.5); glVertex3f(x + s, y + s*0.6, z + s*1.5)
    glVertex3f(x + s, y + s*0.6, z + s*1.5); glVertex3f(x + s, y - s*0.6, z + s*1.5)
    glVertex3f(x + s, y - s*0.6, z + s*1.5); glVertex3f(x - s, y - s*0.6, z + s*1.5)
    glVertex3f(x - s, y - s*0.6, z + s*1.5); glVertex3f(x - s, y + s*0.6, z + s*1.5)
    glEnd()


class HUDControllerPro:
    """
    2B Yarı Saydam Kontrol Paneli, Durum Çubuğu, Bildirimler ve Cetvel Kartı Yöneticisi.
    Pygame 2B Surface üzerinde çizilip OpenGL 2D Texture olarak ekrana yansıtılır.
    """

    def __init__(self):
        pygame.font.init()
        # Sistem fontlarını yükle (Varsayılan Segoe UI veya Arial)
        try:
            self.font_title = pygame.font.SysFont("Segoe UI,Arial,sans-serif", 19, bold=True)
            self.font_bold = pygame.font.SysFont("Segoe UI,Arial,sans-serif", 15, bold=True)
            self.font_norm = pygame.font.SysFont("Segoe UI,Arial,sans-serif", 14)
            self.font_small = pygame.font.SysFont("Segoe UI,Arial,sans-serif", 13)
        except Exception:
            self.font_title = pygame.font.Font(None, 19)
            self.font_bold = pygame.font.Font(None, 15)
            self.font_norm = pygame.font.Font(None, 14)
            self.font_small = pygame.font.Font(None, 13)

        self.tex_id = None          # OpenGL Texture Kimliği (Lazy initialization ile GPU context oluştuktan sonra tahsis edilir)
        self.tex_w = 0
        self.tex_h = 0
        self.show_panel = False     # [H] veya [TAB] ile kontrol paneli açılır (başlangıçta ekranı kapatmasın)
        self.toast_msg = ""         # Üstte beliren anlık bildirim metni
        self.toast_timer = 0        # Bildirimin ekranda kalma süresi
        self.last_update = 0        # Arayüz dokusunun son güncellenme zamanı (CPU-GPU tasarrufu)
        self.dirty = True           # Yeniden çizim bayrağı

    def set_toast(self, msg, duration=2.8):
        """Kullanıcı bir tuşa bastığında üstte şık bir bildirim kutusu gösterir."""
        self.toast_msg = msg
        self.toast_timer = time.time() + duration
        self.dirty = True

    def build_surface(self, win_w, win_h, fps, num_splats, cam_fov, is_flipped,
                      ruler, recorder, room_bounds, top_down_view, culler, auto_tour=False,
                      object_badges=None):
        """Tüm 2B arayüz kartlarını saydam Pygame Surface üzerinde çizer."""
        surf = pygame.Surface((win_w, win_h), pygame.SRCALPHA)
        now = time.time()

        # ---------------------------------------------------------------------
        # 1. Bildirim Mesajı (Toast Notification)
        # ---------------------------------------------------------------------
        if now < self.toast_timer and self.toast_msg:
            tw, th = 620, 42
            tx = (win_w - tw) // 2
            ty = 16
            pygame.draw.rect(surf, (18, 24, 38, 235), (tx, ty, tw, th), border_radius=8)
            pygame.draw.rect(surf, (0, 180, 255, 200), (tx, ty, tw, th), width=2, border_radius=8)
            t_txt = self.font_bold.render(self.toast_msg, True, (255, 255, 255))
            surf.blit(t_txt, (tx + (tw - t_txt.get_width()) // 2, ty + 11))

        # ---------------------------------------------------------------------
        # 2. Üst Durum Çubuğu (FPS, Splat Sayısı, Aktif Mod, Tavan ve Kayıt)
        # ---------------------------------------------------------------------
        bar_w, bar_h = 750, 36
        bx, by = 16, 16
        pygame.draw.rect(surf, (12, 18, 28, 210), (bx, by, bar_w, bar_h), border_radius=6)
        pygame.draw.rect(surf, (40, 60, 85, 180), (bx, by, bar_w, bar_h), width=1, border_radius=6)

        mode_str = "🎬 OTOMATİK SİMÜLASYON" if auto_tour else "🌐 SERBEST GEZİNTİ"
        rec_icon = "🔴 KAYITTA" if recorder.recording else ""
        stat_text = f"⚡ {fps} FPS  |  🔮 {num_splats:,} Splats  |  {mode_str} ([M] Mod)  |  🔍 {cam_fov:.0f}° {rec_icon}"
        stat_surf = self.font_bold.render(stat_text, True, (0, 230, 190) if not auto_tour else (255, 200, 60))
        surf.blit(stat_surf, (bx + 12, by + 9))

        # Kontrol Paneli Butonu
        btn_x = bx + bar_w + 12
        btn_w, btn_h = 175, 36
        pygame.draw.rect(surf, (22, 32, 48, 220), (btn_x, by, btn_w, btn_h), border_radius=6)
        pygame.draw.rect(surf, (0, 150, 255, 180), (btn_x, by, btn_w, btn_h), width=1, border_radius=6)
        h_txt = self.font_bold.render("[H] Kontrol Paneli", True, (255, 255, 255))
        surf.blit(h_txt, (btn_x + 12, by + 9))

        # ---------------------------------------------------------------------
        # 3. 3B Lazer Cetvel Ölçüm Bilgi Kartı (Cetvel Aktifken Sağ Üstte)
        # ---------------------------------------------------------------------
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

        # ---------------------------------------------------------------------
        # 4. Kuşbakışı Kat Planı ve Net Alan Hesabı Kartı ([T] Kuşbakışında)
        # ---------------------------------------------------------------------
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

        # ---------------------------------------------------------------------
        # 5. Tam Kontrol Paneli Kartı ([H] veya [TAB] ile Açılır/Kapanır)
        # ---------------------------------------------------------------------
        if self.show_panel:
            pw, ph = 440, min(650, win_h - 75)
            px, py = 16, 62

            pygame.draw.rect(surf, (10, 15, 24, 235), (px, py, pw, ph), border_radius=10)
            pygame.draw.rect(surf, (0, 160, 255, 160), (px, py, pw, ph), width=2, border_radius=10)

            title_surf = self.font_title.render("🎮 3DGS KONTROL MERKEZİ", True, (0, 215, 255))
            surf.blit(title_surf, (px + 16, py + 12))

            sub_surf = self.font_small.render("[H] / [TAB] ile paneli gizle veya göster", True, (150, 170, 195))
            surf.blit(sub_surf, (px + 16, py + 34))

            pygame.draw.line(surf, (35, 50, 75), (px + 14, py + 52), (px + pw - 14, py + 52), 1)

            # Kontrol tuşlarının açıklamaları
            controls = [
                ("🌐 SERBEST GEZİNTİ & UÇUŞ KONTROLLERİ", ""),
                ("[W / A / S / D]", "İleri (Baktığın Yöne) / Sol / Geri / Sağ"),
                ("[SPACE]", "🚀 Yukarı Yüksel  |  [C] / [Q] / [CTRL] Alçal"),
                ("[SHIFT]", "⚡ Turbo Hızlı Uçuş (2.5x Hız)"),
                ("[Fare Sol Sürükle]", "360° Serbest Kamera Bakış Açısı"),
                ("", ""),
                ("🎮 COMMANDO 8 KUMANDA", ""),
                ("[Sol Stick]", "🕹️ İleri / Geri / Sağ / Sol Hareket"),
                ("[Sağ Stick]", "📷 Kamera Yaw / Pitch Kontrolü"),
                ("[R2 Tetik]", "🚀 Yukarı Gaz  |  [L2] Alçal"),
                ("[RB]", "⚡ Hızlı Uçuş  |  [D-Pad] Tavan/Splat"),
                ("[A]", "📋 Panel  [B] Flow  [X] Tavan  [Y] Sıfırla"),
                ("[LB]", "🧊 OctoMap  [Select] Tur  [Start] Çıkış"),
                ("", ""),
                ("🔍 BÜYÜTME & KÜÇÜLTME (ZOOM & SPLAT)", ""),
                ("[+] / [-]", "🔍 Kamera Yakınlaş (Büyüt) / Uzaklaş (Küçült)"),
                ("[Fare Tekerleği]", "🔍 Hızlı Kamera Büyüt / Küçült"),
                ("[0] (Sıfır)", "Standart Zoom Açısına Sıfırla (60°)"),
                ("[Z] / [X]", "🔮 Splat Boyutunu Büyüt (Dolgunlaştır) / Küçült"),
                ("[N] Tuşu", "🔮 3DGS Gaussian Splat / Nokta Modu"),
                ("[B] Tuşu", "⚡ Performans Modu (Hızlı 2M / Dengeli 3M / Ultra 6M)"),
                ("", ""),
                ("🏠 MİMARİ & TAVAN KESME ARAÇLARI", ""),
                ("[G] Tuşu", "🏠 Tavanı Gizle / Aç (İç Mekanı Kuşbakışı Gör)"),
                ("[7] / [8]", "✂️ Tavan Kesme Yüksekliğini Alçalt / Yükselt"),
                ("[T] Tuşu", "🗺️ 90° Kuşbakışı Kat Planı ve m² Alan Hesabı"),
                ("[E] Tuşu", "📏 3B Lazer Cetvel (İki Nokta Arası Metre Ölçümü)"),
                ("[U] Tuşu", "⚠️ Tünel Darboğaz & Tehlike Isı Haritası (<1.1m)"),
                ("[O] Tuşu", "🧊 OctoMap 3B Voksel + Çarpışma Haritası Yükle"),
                ("[J] Tuşu", "📄 Tünel İnceleme & PDF/HTML Raporu Üret"),
                ("", ""),
                ("🚀 PRO ÇIKTI VE SİMÜLASYON", ""),
                ("[M] / [P]", "🎬 Dron Otomatik Uçuş Simülasyonu"),
                ("[I] Tuşu", "🌊 Optik Flow HUD Aç / Kapat"),
                ("[V] Tuşu", "🎬 60 FPS MP4 Video Kaydını Başlat / Durdur"),
                ("[K] Tuşu", "🌐 Bağımsız Web / HTML 3B Modelini Dışa Aktar"),
                ("[F] Tuşu", "🚁 3. Şahıs / 1. Şahıs / Serbest Dron Kamerası"),
                ("[R] Tuşu", "🔄 Başlangıç Konumuna Sıfırla  |  [ESC]: Çıkış"),
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

    def render_gl(self, win_w, win_h, fps, num_splats, cam_fov, is_flipped,
                  ruler, recorder, room_bounds, top_down_view, culler, auto_tour=False):
        """2B Arayüzü OpenGL 2D Texture olarak 3B sahnenin üzerine çizer (Önbellekli)."""
        now = time.time()
        # Saniyede en fazla 10 kez veya bir etkileşimde GPU'ya doku aktarımı yap
        needs_update = self.dirty or (now - self.last_update >= 0.1) or (self.tex_id is None)
        if needs_update:
            surf = self.build_surface(win_w, win_h, fps, num_splats, cam_fov, is_flipped,
                                      ruler, recorder, room_bounds, top_down_view, culler, auto_tour=auto_tour)
            self.last_update = now
            self.dirty = False

            if hasattr(pygame.image, 'tobytes'):
                rgba_data = pygame.image.tobytes(surf, "RGBA", True)
            else:
                rgba_data = pygame.image.tostring(surf, "RGBA", True)

            if self.tex_id is None or self.tex_w != win_w or self.tex_h != win_h:
                if self.tex_id is None:
                    self.tex_id = glGenTextures(1)
                self.tex_w = win_w
                self.tex_h = win_h
                glBindTexture(GL_TEXTURE_2D, self.tex_id)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, win_w, win_h, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba_data)
            else:
                glBindTexture(GL_TEXTURE_2D, self.tex_id)
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, win_w, win_h, GL_RGBA, GL_UNSIGNED_BYTE, rgba_data)
        else:
            glBindTexture(GL_TEXTURE_2D, self.tex_id)

        # 3B derinlik testini geçici devre dışı bırakarak 2B arayüzü en üste bas
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


# ======================================================================================
# 🎮 ANA GÖRÜNTÜLEYİCİ VE OYUN DÖNGÜSÜ (view_gaussian_splats)
# ======================================================================================
def view_gaussian_splats(ply_path="gaussian_scene.ply"):
    """
    3D Gaussian Splats modelini GPU önbelleğinden 0.1 sn içinde yükler ve 144+ FPS ile açar.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    target_ply = os.path.join(base_dir, ply_path)
    cache_path = target_ply.replace(".ply", "_cache.npz")
    traj_path = os.path.join(base_dir, "gaussian_trajectory.npz")

    xyz, rgb, scales, opacity = None, None, None, None
    auto_waypoints = []

    # 1. ⚡ Hızlı Önbellek Yüklemesi (.npz)
    if os.path.exists(cache_path):
        t0 = time.time()
        data = np.load(cache_path, allow_pickle=True)
        xyz = np.ascontiguousarray(data["xyz"], dtype=np.float32)
        rgb = np.ascontiguousarray(data["rgb"], dtype=np.float32)
        scales = np.ascontiguousarray(data["scales"], dtype=np.float32) if "scales" in data else None
        opacity = np.ascontiguousarray(data["opacity"], dtype=np.float32) if "opacity" in data else None
        print(f" ⚡ 3D Gaussian Splats Önbellekten Yüklendi: {len(xyz):,} Splat ({time.time()-t0:.2f}s)")
    elif os.path.exists(os.path.join(base_dir, "gaussian_scene_cache.npz")):
        # İstenen dosya henüz taranmamışsa varsayılan hazır sahneyi yükle
        print(f" ℹ️ '{os.path.basename(target_ply)}' henüz taranmamış, hazır 'gaussian_scene' modeli yükleniyor...")
        data = np.load(os.path.join(base_dir, "gaussian_scene_cache.npz"), allow_pickle=True)
        xyz = np.ascontiguousarray(data["xyz"], dtype=np.float32)
        rgb = np.ascontiguousarray(data["rgb"], dtype=np.float32)
        scales = np.ascontiguousarray(data["scales"], dtype=np.float32) if "scales" in data else None
        opacity = np.ascontiguousarray(data["opacity"], dtype=np.float32) if "opacity" in data else None
    else:
        print(" ⏳ Gaussian Splatting önbelleği bulunamadı! 'mast3r_to_3dgs.py' ile üretiliyor...")
        import mast3r_to_3dgs
        out_f = mast3r_to_3dgs.build_gaussian_splats_from_mast3r()
        if out_f and os.path.exists(out_f.replace(".ply", "_cache.npz")):
            data = np.load(out_f.replace(".ply", "_cache.npz"), allow_pickle=True)
            xyz = np.ascontiguousarray(data["xyz"], dtype=np.float32)
            rgb = np.ascontiguousarray(data["rgb"], dtype=np.float32)

    if xyz is None or len(xyz) == 0:
        print(f"\n❌ HATA: '{ply_path}' için geçerli 3B nokta verisi bulunamadı!")
        print(" -> Lütfen önce 'calistir_canli_dron.bat' ile canlı bir tarama yapın veya 'ofis_videosunu_yeniden_isle.bat' çalıştırın.")
        return

    # 2. Kamera Uçuş Yörüngesini Yükle (Dronun taranan rotası)
    if os.path.exists(traj_path):
        traj_data = np.load(traj_path, allow_pickle=True)
        auto_waypoints = traj_data["positions"].copy()

    # Eğer yörünge dosyası yoksa tünelin ana Z ekseni boyunca otomatik pürüzsüz rota oluştur
    if len(auto_waypoints) < 2 and xyz is not None and len(xyz) > 0:
        z_min = float(np.min(xyz[:, 2]))
        z_max = float(np.max(xyz[:, 2]))
        y_mid = float(np.mean(xyz[:, 1]))
        x_mid = float(np.mean(xyz[:, 0]))
        ts = np.linspace(z_min + 0.5, z_max - 0.5, 600)
        auto_waypoints = np.column_stack([
            np.full_like(ts, x_mid),
            np.full_like(ts, y_mid),
            ts
        ]).astype(np.float32)

    # 🪞 Gerçek Dünya Sağ/Sol Hizalaması (Aynalama Düzeltildi)
    xyz[:, 0] = -xyz[:, 0]
    if len(auto_waypoints) > 0:
        auto_waypoints[:, 0] = -auto_waypoints[:, 0]
    is_flipped = True

    # 🚀 Performans ve Akıllı Seyreltme Sistemi
    raw_xyz = xyz.copy()
    raw_rgb = rgb.copy()
    raw_opacity = opacity.copy() if opacity is not None else None

    quality_strides = [5, 4, 3, 2, 1]
    quality_names = [
        "🚀 ULTRA AKICI (144+ FPS, ~1.2M Splat)",
        "⚡ HIZLI MOD (144+ FPS, ~1.5M Splat)",
        "⚖️ DENGELİ MOD (120+ FPS, ~2.0M Splat)",
        "💎 YÜKSEK KALİTE (90+ FPS, ~3.0M Splat)",
        "👑 TAM ÇÖZÜNÜRLÜK (6.0M Splat)"
    ]
    # ✅ Varsayılan olarak laptoplarda kasmasını önlemek için "Dengeli (stride=3)" modu aç
    # Kullanıcı isterse "B" tuşuyla Tam Çözünürlüğe çıkabilir.
    quality_idx = 2
    cur_stride = quality_strides[quality_idx]

    xyz = np.ascontiguousarray(raw_xyz[::cur_stride], dtype=np.float32)
    rgb = np.ascontiguousarray(raw_rgb[::cur_stride], dtype=np.float32)
    num_splats = len(xyz)
    room_bounds = FloorplanEstimator.calculate_bounds(xyz)

    # Splat nokta boyutu varsayılanı (Boşluksuz, katı ve dolgun kaya yüzeyi)
    splat_point_size = 15.0 if cur_stride <= 1 else (18.0 if cur_stride == 2 else 22.0)

    # Modül Yöneticilerini Başlat
    culler = CeilingCuller()
    culler.init_from_bounds(xyz)
    ruler = LaserRuler()
    recorder = VideoRecorder()
    heatmap_engine = ProximityHeatmapEngine()
    octomap_engine = OctomapEngine(voxel_size=0.15)
    vbo_octo_xyz = None
    vbo_octo_rgba = None
    octo_line_count = 0
    hud = HUDControllerPro()
    drone_model = Drone3DModel()
    drone_view_mode = 2  # 2: Serbest Gezgin Modu (Doğrudan kamera, önde dron gövdesi olmadan pürüzsüz uçuş)

    # 🚁 Drone Fizik Motoru ve Optik Flow
    physics = DronePhysicsEngine()
    optical_flow = OpticalFlowHUD(grid_cols=14, grid_rows=9)
    show_optical_flow = False  # [I] tuşuyla açılıp kapatılabilir (varsayılan temiz görünüm)

    # Zemin yüksekliğini fizik motoruna bildir
    if room_bounds is not None:
        physics.set_floor(room_bounds['min_y'])

    hud.set_toast("🌐 Serbest Gezinti Modu (W/A/S/D: Gezin | Boşluk/C: Yüksel/Alçal | Fare: Bak)", 5.0)

    # RGBA Renk Matrisini Hazırla (Renk + Opaklık)
    if raw_opacity is not None:
        rgba = np.column_stack([rgb, raw_opacity[::cur_stride]]).astype(np.float32)
    else:
        rgba = np.column_stack([rgb, np.ones(num_splats, dtype=np.float32)]).astype(np.float32)
    original_rgba = rgba.copy()
    rgba = np.ascontiguousarray(rgba, dtype=np.float32)

    # Pygame & OpenGL Penceresini Başlat
    pygame.init()
    win_w, win_h = 1280, 720

    # 🎮 Kumanda Başlatma (Gamepad / Commando 8)
    pygame.joystick.init()
    joystick = None
    joystick_name = "Yok"
    is_rc_controller = False
    if pygame.joystick.get_count() > 0:
        try:
            joystick = pygame.joystick.Joystick(0)
            joystick_name = joystick.get_name()
            is_rc_controller = any(w in joystick_name.lower() for w in [
                "opentx", "edgetx", "commando", "iflight", "radiomaster", 
                "taranis", "jumper", "frsky", "flysky", "elrs", "crossfire"
            ])
            print(f" 🎮 Kumanda Bulundu: {joystick_name} {'(RC Kumanda)' if is_rc_controller else '(Gamepad)'}")
            print(f"    Axis: {joystick.get_numaxes()} | Buton: {joystick.get_numbuttons()} | Hat: {joystick.get_numhats()}")
        except Exception as e:
            print(f" ⚠️ Kumanda başlatma uyarısı: {e}")
    else:
        print(" ⌨️  Kumanda bulunamadı — sadece klavye/fare ile kontrol.")

    # MSAA kapatıldı (nokta bulutlarında gereksiz GPU yükünü önler)
    pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 0)
    pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 0)
    pygame.display.gl_set_attribute(pygame.GL_SWAP_CONTROL, 0)
    try:
        pygame.display.set_mode((win_w, win_h), DOUBLEBUF | OPENGL | RESIZABLE)
    except pygame.error as e:
        print(f" ⚠️ İlk GL modu başarısız ({e}), standart mod deneniyor...")
        pygame.display.set_mode((win_w, win_h), DOUBLEBUF | OPENGL)
    pygame.key.set_repeat(0)

    gpu_vendor = glGetString(GL_VENDOR).decode(errors='replace')
    gpu_renderer = glGetString(GL_RENDERER).decode(errors='replace')
    print(f" 🚀 Aktif 3DGS GPU Donanımı: {gpu_renderer} ({gpu_vendor})")
    joy_info = f" | 🎮 {joystick_name}" if joystick else ""
    pygame.display.set_caption(f"3D Gaussian Splatting Ultimate [{gpu_renderer}{joy_info}]")

    # OpenGL Render Ayarları
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)
    glDisable(GL_CULL_FACE)
    glEnable(GL_POINT_SMOOTH)
    glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    # 🔮 3DGS Gaussian Splatting: Yumuşak Radyal Doku Üretimi (64x64)
    # Noktalar kaba kareler yerine kenarları yumuşakça şeffaflaşan gerçekçi Gaussian elipsleri olarak çizilir
    tex_gs_size = 64
    y_g, x_g = np.ogrid[-1:1:tex_gs_size*1j, -1:1:tex_gs_size*1j]
    r_sq_g = x_g**2 + y_g**2
    alpha_g = np.clip(np.exp(-2.6 * r_sq_g), 0.0, 1.0).astype(np.float32)
    alpha_g[r_sq_g > 1.0] = 0.0
    tex_gs_data = np.zeros((tex_gs_size, tex_gs_size, 4), dtype=np.uint8)
    tex_gs_data[..., 0:3] = 255
    tex_gs_data[..., 3] = (alpha_g * 255).astype(np.uint8)

    tex_gaussian_splat = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex_gaussian_splat)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, tex_gs_size, tex_gs_size, 0, GL_RGBA, GL_UNSIGNED_BYTE, tex_gs_data)
    glBindTexture(GL_TEXTURE_2D, 0)

    use_gaussian_splat = True   # True: Yumuşak 3DGS Splat + Mesafe Uyarlamalı, False: Nokta Bulutu

    # GPU VBO (Vertex Buffer Object) Bellek Tahsisleri
    vbo_xyz = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_xyz)
    glBufferData(GL_ARRAY_BUFFER, xyz.nbytes, xyz, GL_STATIC_DRAW)

    vbo_rgba = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_rgba)
    glBufferData(GL_ARRAY_BUFFER, rgba.nbytes, rgba, GL_STATIC_DRAW)
    glBindBuffer(GL_ARRAY_BUFFER, 0)

    # ⚡ Dinamik Kalite / Seyreltme Güncelleme Fonksiyonu
    def apply_quality_mode(idx):
        nonlocal xyz, rgb, rgba, original_rgba, num_splats, splat_point_size, room_bounds
        st = quality_strides[idx]
        xyz = np.ascontiguousarray(raw_xyz[::st], dtype=np.float32)
        rgb = np.ascontiguousarray(raw_rgb[::st], dtype=np.float32)
        if raw_opacity is not None:
            rgba = np.column_stack([rgb, raw_opacity[::st]]).astype(np.float32)
        else:
            rgba = np.column_stack([rgb, np.ones(len(xyz), dtype=np.float32)]).astype(np.float32)
        original_rgba = rgba.copy()
        rgba = np.ascontiguousarray(rgba, dtype=np.float32)
        num_splats = len(xyz)
        room_bounds = FloorplanEstimator.calculate_bounds(xyz)
        culler.init_from_bounds(xyz)

        if st >= 4:
            splat_point_size = 24.0
        elif st == 3:
            splat_point_size = 20.0
        elif st == 2:
            splat_point_size = 17.5
        else:
            splat_point_size = 15.0

        glBindBuffer(GL_ARRAY_BUFFER, vbo_xyz)
        glBufferData(GL_ARRAY_BUFFER, xyz.nbytes, xyz, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_rgba)
        glBufferData(GL_ARRAY_BUFFER, rgba.nbytes, rgba, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    # Zemin Izgarası VBO (Y=0 Düzlemi)
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

    # Yörünge Çizgisi VBO (Yeşil Rota)
    vbo_traj = None
    traj_n = 0
    if len(auto_waypoints) > 0:
        traj_arr = auto_waypoints.astype(np.float32)
        traj_n = len(traj_arr)
        vbo_traj = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_traj)
        glBufferData(GL_ARRAY_BUFFER, traj_arr.nbytes, traj_arr, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    glClearColor(0.04, 0.05, 0.08, 1.0)

    # Başlangıç Kamera Konumu: Tünel girişinde, göz hizasında (1.6m) ve tam karşıya bakar şekilde başla
    drone_x = 0.0
    drone_y = 1.6
    drone_z = -4.0
    cam_yaw = 0.0
    cam_pitch = 0.0
    target_yaw = 0.0
    target_pitch = 0.0

    # Fizik motorunu başlangıç konumuyla senkronize et
    physics.x, physics.y, physics.z = drone_x, drone_y, drone_z
    if room_bounds is not None:
        physics.set_floor(room_bounds['min_y'])

    # Serbest Gezinti Hızları
    free_vx = 0.0
    free_vy = 0.0
    free_vz = 0.0

    cam_fov = 60.0              # Görüş açısı (60°)
    target_fov = 60.0
    auto_tour = False           # [M] veya [P] ile açılır
    tour_progress = 0.0
    tour_speed = 1.0
    show_frustums = False
    top_down_view = False
    user_override_look = False

    # 💡 Tünel Aydınlatma / Fener Kademeleri (1.0x -> 1.45x -> 1.90x)
    brightness_levels = [1.0, 1.45, 1.90]
    brightness_idx = 0
    brightness_names = [
        "💡 Fener: KAPALI (Standart)",
        "💡 Fener: AÇIK (1.45x HDR Aydınlatma - İnsan & Eşyalar Parlatıldı)",
        "🔦 Ultra Gece Görüşü: AÇIK (1.90x Maksimum Parlaklık)"
    ]

    def apply_brightness(b_level):
        if b_level == 1.0:
            bright_data = original_rgba
        else:
            scale_v = np.array([b_level, b_level, b_level, 1.0], dtype=np.float32)
            bright_data = np.clip(original_rgba * scale_v, 0.0, 1.0).astype(np.float32)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_rgba)
        glBufferData(GL_ARRAY_BUFFER, bright_data.nbytes, bright_data, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    clock = pygame.time.Clock()
    mouse_down_left = False
    last_mouse_pos = (0, 0)
    held_keys = set()           # Basılı tutulan tuşları %100 garantili yakalama kümesi
    running = True

    fps_timer = time.time()
    frame_count = 0
    current_fps = 144
    last_frame_time = time.perf_counter()
    smooth_dt = 1.0 / 144.0

    # =========================================================================
    # 🔄 ANA ETKİLEŞİM VE RENDER DÖNGÜSÜ
    # =========================================================================
    while running:
        clock.tick(240)  # Aşırı GPU yükünü önleyen tavan (144Hz ekran tazelemesiyle v-blank çatışmasını önler)
        now_time = time.perf_counter()
        raw_dt = now_time - last_frame_time
        last_frame_time = now_time
        raw_dt = min(max(raw_dt, 0.0005), 0.05)
        smooth_dt = 0.88 * smooth_dt + 0.12 * raw_dt
        dt = smooth_dt
        frame_count += 1
        # FPS Sayacı Hesaplama
        if time.time() - fps_timer >= 0.5:
            current_fps = int(frame_count / (time.time() - fps_timer))
            pygame.display.set_caption(f"3DGS Ultimate [{current_fps} FPS | {num_splats:,} Gaussians | FOV: {cam_fov:.0f}°]")
            frame_count = 0
            fps_timer = time.time()

        # Olay (Event) Yakalama (Klavye & Fare)
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == VIDEORESIZE:
                win_w, win_h = event.w, event.h
                glViewport(0, 0, win_w, win_h)
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:  # Fare Sol Tık
                    if ruler.active:
                        # Lazer cetvel aktifken tıklanan noktanın metresini ölç
                        picked = ruler.add_point_from_screen(event.pos[0], event.pos[1], win_w, win_h)
                        if picked is not None and ruler.last_distance is not None:
                            hud.set_toast(f"📏 Ölçüldü: {ruler.last_distance:.2f} Metre", 3.0)
                    else:
                        mouse_down_left = True
                        last_mouse_pos = event.pos
                elif event.button == 4:  # Tekerlek Yukarı -> Optik Yakınlaş (Lens Zoom In / Büyüt)
                    target_fov = max(15.0, target_fov - 2.5)
                elif event.button == 5:  # Tekerlek Aşağı -> Optik Uzaklaş (Lens Zoom Out / Küçült)
                    target_fov = min(110.0, target_fov + 2.5)
            elif event.type == MOUSEBUTTONUP:
                if event.button == 1:
                    mouse_down_left = False
            elif event.type == MOUSEMOTION:
                dx = event.pos[0] - last_mouse_pos[0]
                dy = event.pos[1] - last_mouse_pos[1]
                last_mouse_pos = event.pos
                if mouse_down_left and not ruler.active:
                    user_override_look = True  # Tur sırasında fareyle serbestçe istenen yöne bakma yetkisi
                    target_yaw += dx * 0.22
                    target_pitch += dy * 0.22
                    target_pitch = max(-89.0, min(89.0, target_pitch))

            # ─────────────────────────────────────────────────────────
            # 🎮 COMMANDO 8 BUTON OLAYLARI (Tek Basış Aksiyonları)
            # ─────────────────────────────────────────────────────────
            # ─────────────────────────────────────────────────────────
            # 🎮 GAMEPAD BUTON OLAYLARI (Sadece RC Kumanda Değilse Dinle)
            # ─────────────────────────────────────────────────────────
            elif event.type == pygame.JOYBUTTONDOWN and joystick and not is_rc_controller:
                btn = event.button
                if btn == 0:    # A → Kontrol Paneli
                    hud.show_panel = not hud.show_panel
                    hud.dirty = True
                    hud.set_toast("🎮 Kontrol Paneli " + ("AÇIK" if hud.show_panel else "KAPALI"), 1.2)
                elif btn == 1:  # B → Optik Flow
                    show_optical_flow = not show_optical_flow
                    hud.set_toast(f"🌊 Optik Flow: {'AÇIK' if show_optical_flow else 'KAPALI'}", 1.5)
                elif btn == 2:  # X → Tavan Gizle/Göster
                    c_state = culler.toggle()
                    hud.set_toast(f"🏠 Tavan: {'GİZLENDİ' if c_state else 'GÖSTERİLİYOR'}", 2.0)
                elif btn == 3:  # Y → Sıfırla
                    auto_tour = False; tour_progress = 0.0; top_down_view = False
                    culler.enabled = False
                    drone_x, drone_y, drone_z = 0.0, 1.6, -4.0
                    target_yaw, target_pitch = 0.0, 0.0; cam_yaw, cam_pitch = 0.0, 0.0
                    physics.vx = physics.vy = physics.vz = 0.0
                    physics.x, physics.y, physics.z = drone_x, drone_y, drone_z
                    cam_fov = 60.0; target_fov = 60.0
                    hud.set_toast("🔄 Kamera Tünel Girişine Sıfırlandı", 2.0)
                elif btn == 6 or btn == 10:  # Select / Mode → Otomatik Simülasyon
                    if len(auto_waypoints) > 1:
                        auto_tour = not auto_tour
                        user_override_look = False
                        hud.dirty = True
                        hud.set_toast("🎬 Simülasyon: " + ("BAŞLATILDI" if auto_tour else "DURDURULDU"), 2.0)
                elif btn == 7:  # Start → Çıkış
                    running = False
                elif btn == 8:  # Sol Stick Bas → Zoom Sıfırla
                    target_fov = 60.0; cam_fov = 60.0
                    hud.set_toast("🔍 Zoom Sıfırlandı (60°)", 1.2)

            # D-Pad (Hat) → Tavan Kesme Seviyesi Ayarı (Yalnızca Gamepad)
            elif event.type == pygame.JOYHATMOTION and joystick and not is_rc_controller:
                hx, hy = event.value
                if hy > 0:   # D-Pad Yukarı → Tavan yükselt
                    r = culler.adjust_cut(+0.05); culler.enabled = True
                    hud.set_toast(f"✂️ Tavan: %{r*100:.0f}", 1.0)
                elif hy < 0: # D-Pad Aşağı → Tavan alçalt
                    r = culler.adjust_cut(-0.05); culler.enabled = True
                    hud.set_toast(f"✂️ Tavan: %{r*100:.0f}", 1.0)
                elif hx > 0: # D-Pad Sağ → Splat boyutu büyüt
                    splat_point_size = min(50.0, splat_point_size + 1.5)
                    hud.set_toast(f"🔮 Splat: {splat_point_size:.1f}", 1.0)
                elif hx < 0: # D-Pad Sol → Splat boyutu küçült
                    splat_point_size = max(2.0, splat_point_size - 1.5)
                    hud.set_toast(f"🔮 Splat: {splat_point_size:.1f}", 1.0)

            elif event.type == KEYDOWN:
                held_keys.add(event.key)

                if event.key in (K_h, K_TAB):
                    hud.show_panel = not hud.show_panel
                    hud.dirty = True
                elif event.key == K_g:
                    # 🏠 Tavanı Gizle / Aç
                    c_state = culler.toggle()
                    hud.set_toast(f"🏠 Tavan: {'GİZLENDİ (İç Mekan Açıldı)' if c_state else 'GÖSTERİLİYOR'}", 2.5)
                elif event.key in (K_7, K_KP7):
                    # ✂️ Tavan Kesme Yüksekliğini Alçalt
                    r = culler.adjust_cut(-0.05)
                    culler.enabled = True
                    hud.set_toast(f"✂️ Tavan Kesme Seviyesi: %{r*100:.0f} ({culler.cut_y:.2f}m)", 1.5)
                elif event.key in (K_8, K_KP8):
                    # ✂️ Tavan Kesme Yüksekliğini Yükselt
                    r = culler.adjust_cut(+0.05)
                    culler.enabled = True
                    hud.set_toast(f"✂️ Tavan Kesme Seviyesi: %{r*100:.0f} ({culler.cut_y:.2f}m)", 1.5)
                elif event.key == K_e:
                    # 📏 3B Lazer Cetvel Aç/Kapat
                    ruler.toggle()
                    hud.set_toast("📏 Lazer Cetvel: " + ("AKTİF (Ölçmek İçin 2 Noktaya Tıklayın)" if ruler.active else "KAPATILDI"), 3.0)
                elif event.key == K_v:
                    # 🎬 60 FPS MP4 Video Kaydı
                    res_msg = recorder.toggle(win_w, win_h, fps=60)
                    hud.set_toast(res_msg, 3.5)
                elif event.key == K_k:
                    # 🌐 Web / HTML 3B Model Dışa Aktar
                    html_file = export_to_standalone_html(xyz, rgb, output_html="3d_scene.html")
                    hud.set_toast(f"🌐 Web 3B Modeli Kaydedildi: {html_file}", 3.5)
                elif event.key in (K_PLUS, K_KP_PLUS, K_EQUALS):
                    # 🔍 Optik Zoom Yakınlaş / Büyüt (FOV Küçült)
                    target_fov = max(15.0, target_fov - 3.0)
                    hud.set_toast(f"🔍 Optik Zoom: {target_fov:.0f}°", 1.0)
                elif event.key in (K_MINUS, K_KP_MINUS):
                    # 🔍 Optik Zoom Uzaklaş / Küçült (FOV Büyüt)
                    target_fov = min(110.0, target_fov + 3.0)
                    hud.set_toast(f"🔍 Optik Zoom: {target_fov:.0f}°", 1.0)
                elif event.key == K_z:
                    # 🔮 Splat Nokta Boyutunu Büyüt (Daha Dolgun, Boşluksuz Yüzey)
                    splat_point_size = min(60.0, splat_point_size + 2.0)
                    hud.set_toast(f"🔮 Splat Boyutu Büyütüldü (Dolgun): {splat_point_size:.1f}", 1.5)
                elif event.key == K_x:
                    # 🔮 Splat Nokta Boyutunu Küçült (Daha İnce)
                    splat_point_size = max(2.0, splat_point_size - 2.0)
                    hud.set_toast(f"🔮 Splat Boyutu Küçültüldü: {splat_point_size:.1f}", 1.5)
                elif event.key == K_n:
                    # 🔮/📍 3DGS Gaussian Splat vs Nokta Bulutu Modu ([N] Tuşu)
                    use_gaussian_splat = not use_gaussian_splat
                    if use_gaussian_splat:
                        hud.set_toast("🔮 3DGS Gaussian Modu: AÇIK (Yumuşak, Dolgun & Mesafe Uyarlamalı)", 3.0)
                    else:
                        hud.set_toast("📍 Nokta Bulutu Modu: AÇIK (Klasik Point Cloud)", 2.0)
                elif event.key in (K_0, K_KP0):
                    # 🔍 Zoom Sıfırla (60°)
                    target_fov = 60.0
                    cam_fov = 60.0
                    hud.set_toast("🔍 Zoom Sıfırlandı (60°)", 1.5)
                elif event.key in (K_m, K_p):
                    # 🎬 Mod Değiştir: Serbest Gezinti <-> Dron Uçuş Simülasyonu ([M] veya [P])
                    if len(auto_waypoints) > 1:
                        auto_tour = not auto_tour
                        user_override_look = False
                        hud.dirty = True
                        if auto_tour:
                            hud.set_toast("🎬 OTOMATİK SİMÜLASYON BAŞLADI: Dron rotasında uçuluyor ([M] ile durdur)", 3.5)
                        else:
                            hud.set_toast("🌐 SERBEST GEZİNTİ MODU: W, A, S, D ile serbest gezinim", 2.5)
                    else:
                        hud.set_toast("⚠️ Taranmış rota bulunamadı!", 2.0)
                elif event.key == K_b:
                    # ⚡ Performans / Kalite Modu Değiştir ([B] Tuşu)
                    quality_idx = (quality_idx + 1) % len(quality_strides)
                    apply_quality_mode(quality_idx)
                    hud.set_toast(quality_names[quality_idx], 3.0)
                elif event.key == K_u:
                    # ⚠️ Tünel Darboğaz & Tehlike Isı Haritası ([U] Tuşu)
                    h_active = heatmap_engine.toggle()
                    if h_active:
                        hm_cols = heatmap_engine.compute_clearance_heatmap(xyz)
                        if hm_cols is not None:
                            hm_rgba = np.column_stack([hm_cols, rgba[:, 3]]).astype(np.float32)
                            glBindBuffer(GL_ARRAY_BUFFER, vbo_rgba)
                            glBufferData(GL_ARRAY_BUFFER, hm_rgba.nbytes, hm_rgba, GL_DYNAMIC_DRAW)
                            glBindBuffer(GL_ARRAY_BUFFER, 0)
                            hud.set_toast("⚠️ Tünel Tehlike / Açıklık Isı Haritası: AÇIK (Kırmızı = <1.1m)", 3.5)
                    else:
                        glBindBuffer(GL_ARRAY_BUFFER, vbo_rgba)
                        glBufferData(GL_ARRAY_BUFFER, original_rgba.nbytes, original_rgba, GL_STATIC_DRAW)
                        glBindBuffer(GL_ARRAY_BUFFER, 0)
                        hud.set_toast("🌈 Normal Fotogerçekçi Renk Moduna Dönüldü", 2.0)
                elif event.key == K_o:
                    # 🧊 OctoMap (3B Voksel / Doluluk Izgara Modu - [O] Tuşu)
                    o_active = octomap_engine.toggle()
                    if o_active:
                        if vbo_octo_xyz is None:
                            hud.set_toast("⏳ OctoMap Voksel Izgarası Hesaplanıyor...", 1.5)
                            c_verts, c_cols = octomap_engine.generate_octomap(xyz, voxel_size=0.15)
                            if c_verts is not None:
                                octo_line_count = len(c_verts)
                                vbo_octo_xyz = glGenBuffers(1)
                                glBindBuffer(GL_ARRAY_BUFFER, vbo_octo_xyz)
                                glBufferData(GL_ARRAY_BUFFER, c_verts.nbytes, c_verts, GL_STATIC_DRAW)

                                vbo_octo_rgba = glGenBuffers(1)
                                glBindBuffer(GL_ARRAY_BUFFER, vbo_octo_rgba)
                                glBufferData(GL_ARRAY_BUFFER, c_cols.nbytes, c_cols, GL_STATIC_DRAW)
                                glBindBuffer(GL_ARRAY_BUFFER, 0)

                                # 🚁 Çarpışma haritasını fizik motoruna yükle
                                if octomap_engine.voxel_centers is not None:
                                    physics.set_collision_map(
                                        octomap_engine.voxel_centers,
                                        voxel_size=octomap_engine.voxel_size
                                    )
                                    hud.set_toast(f"🧊 OctoMap + Çarpışma Haritası Yüklendi ({octomap_engine.num_cubes:,} Küp)", 3.5)
                                else:
                                    hud.set_toast(f"🧊 OctoMap 3B Voksel Modu: AÇIK ({octomap_engine.num_cubes:,} Küp)", 3.5)
                        else:
                            hud.set_toast(f"🧊 OctoMap 3B Voksel Modu: AÇIK ({octomap_engine.num_cubes:,} Küp)", 3.5)
                    else:
                        hud.set_toast("🔮 3DGS Fotogerçekçi Renk Moduna Dönüldü", 2.0)
                elif event.key == K_j:
                    # 📄 Otomatik Tünel İnceleme & PDF/HTML Raporu Üret ([J] Tuşu)
                    rep_path = generate_tunnel_report(input_file="gaussian_scene_cache.npz")
                    if rep_path and os.path.exists(rep_path):
                        import webbrowser
                        webbrowser.open(rep_path)
                        hud.set_toast(f"📄 Tünel İnceleme Raporu Üretildi & Tarayıcıda Açıldı!", 4.0)
                elif event.key == K_f:
                    # 🚁 Dron Kamera Modu (0: 3. Şahıs Takip, 1: 1. Şahıs FPV, 2: Serbest)
                    drone_view_mode = (drone_view_mode + 1) % 3
                    if drone_view_mode == 0:
                        hud.set_toast("🚁 Kamera: 3. Şahıs Dron Takip Modu (Chase Cam)", 2.5)
                    elif drone_view_mode == 1:
                        hud.set_toast("📷 Kamera: 1. Şahıs Dron Kokpit Modu (FPV)", 2.5)
                    else:
                        hud.set_toast("🌐 Kamera: Serbest Gezgin Modu (Free Orbit)", 2.5)
                elif event.key == K_l:
                    # 💡 Dron Feneri & Aydınlatma Kademesi ([L] Tuşu)
                    brightness_idx = (brightness_idx + 1) % len(brightness_levels)
                    b_val = brightness_levels[brightness_idx]
                    apply_brightness(b_val)
                    hud.set_toast(brightness_names[brightness_idx], 2.5)
                elif event.key == K_1:
                    auto_tour = False; drone_x, drone_y, drone_z = 0.0, 1.6, -4.0
                    target_yaw, target_pitch = 0.0, 0.0; cam_yaw, cam_pitch = 0.0, 0.0
                    physics.x, physics.y, physics.z = drone_x, drone_y, drone_z
                    physics.vx = physics.vy = physics.vz = 0.0
                    hud.set_toast("🚪 Konum 1'e Işınlanıldı (Giriş)", 1.5)
                elif event.key == K_2 and len(auto_waypoints) > 0:
                    auto_tour = False; p = auto_waypoints[len(auto_waypoints)//3]; drone_x, drone_y, drone_z = float(p[0]), 1.6, float(p[2])
                    physics.x, physics.y, physics.z = drone_x, drone_y, drone_z
                    physics.vx = physics.vy = physics.vz = 0.0
                    hud.set_toast("🚪 Konum 2'ye Işınlanıldı (Orta 1)", 1.5)
                elif event.key == K_3 and len(auto_waypoints) > 0:
                    auto_tour = False; p = auto_waypoints[2*len(auto_waypoints)//3]; drone_x, drone_y, drone_z = float(p[0]), 1.6, float(p[2])
                    physics.x, physics.y, physics.z = drone_x, drone_y, drone_z
                    physics.vx = physics.vy = physics.vz = 0.0
                    hud.set_toast("🚪 Konum 3'e Işınlanıldı (Orta 2)", 1.5)
                elif event.key == K_4 and len(auto_waypoints) > 0:
                    auto_tour = False; p = auto_waypoints[-1]; drone_x, drone_y, drone_z = float(p[0]), 1.6, float(p[2])
                    physics.x, physics.y, physics.z = drone_x, drone_y, drone_z
                    physics.vx = physics.vy = physics.vz = 0.0
                    hud.set_toast("🚪 Konum 4'e Işınlanıldı (Son Nokta)", 1.5)
                elif event.key == K_t:
                    # 🗺️ 90° Kuşbakışı Kat Planı
                    top_down_view = not top_down_view
                    auto_tour = False
                    if top_down_view:
                        culler.enabled = True
                        target_pitch, target_yaw = 89.0, 0.0; cam_pitch, cam_yaw = 89.0, 0.0
                        drone_x = float(np.mean(xyz[:, 0])); drone_y = float(np.max(xyz[:, 1])) + 14.0; drone_z = float(np.mean(xyz[:, 2]))
                        hud.set_toast("🗺️ Kuşbakışı Kat Planı & Tavan Kesildi (İç Mekan Görünür)", 3.0)
                    else:
                        target_pitch, target_yaw = 0.0, 0.0; cam_pitch, cam_yaw = 0.0, 0.0
                        hud.set_toast("🕹️ Serbest Uçuş Moduna Dönüldü", 1.5)
                elif event.key == K_r:
                    # 🔄 Konum ve Kamera Sıfırlama (Tünel Girişine Dön)
                    auto_tour = False; tour_progress = 0.0; top_down_view = False
                    culler.enabled = False
                    drone_x, drone_y, drone_z = 0.0, 1.6, -4.0
                    target_yaw, target_pitch = 0.0, 0.0; cam_yaw, cam_pitch = 0.0, 0.0
                    physics.vx = physics.vy = physics.vz = 0.0
                    physics.x, physics.y, physics.z = drone_x, drone_y, drone_z
                    cam_fov = 60.0
                    target_fov = 60.0
                    hud.set_toast("🔄 Kamera Tünel Girişine Sıfırlandı", 2.0)
                elif event.key == K_i:
                    # 🌊 Optik Flow HUD Aç / Kapat ([I] Tuşu)
                    show_optical_flow = not show_optical_flow
                    hud.set_toast(f"🌊 Optik Flow: {'AÇIK' if show_optical_flow else 'KAPALI'}", 1.5)
                elif event.key == K_ESCAPE:
                    running = False

            elif event.type == KEYUP:
                held_keys.discard(event.key)

            elif event.type == pygame.ACTIVEEVENT:
                if event.gain == 0:
                    held_keys.clear()


        # Keskin ve doğrudan kamera yönü (Sıfır gecikmeli fare bakışı)
        cam_yaw = target_yaw
        cam_pitch = target_pitch

        rad_yaw, rad_pitch = math.radians(cam_yaw), math.radians(cam_pitch)
        fwd_x = math.sin(rad_yaw) * math.cos(rad_pitch)
        fwd_y = -math.sin(rad_pitch)
        fwd_z = math.cos(rad_yaw) * math.cos(rad_pitch)
        right_x = math.cos(rad_yaw)
        right_z = -math.sin(rad_yaw)

        # ── 🎮 Kamera ve Hareket Kontrolü (Serbest Gezinti veya Otomatik Simülasyon) ──
        keys = pygame.key.get_pressed()

        # %100 Güvenilir Tuş Okuma (get_pressed + held_keys kümesi)
        is_w = bool(keys[K_w] or (K_w in held_keys) or keys[K_UP] or (K_UP in held_keys))
        is_s = bool(keys[K_s] or (K_s in held_keys) or keys[K_DOWN] or (K_DOWN in held_keys))
        is_a = bool(keys[K_a] or (K_a in held_keys) or keys[K_LEFT] or (K_LEFT in held_keys))
        is_d = bool(keys[K_d] or (K_d in held_keys) or keys[K_RIGHT] or (K_RIGHT in held_keys))
        is_up = bool(keys[K_SPACE] or (K_SPACE in held_keys))
        is_down = bool(keys[K_c] or (K_c in held_keys) or keys[K_q] or (K_q in held_keys) or keys[K_LCTRL] or (K_LCTRL in held_keys))
        fast_mode = bool(keys[K_LSHIFT] or keys[K_RSHIFT])

        # --- Joystick / RC Kumanda Giriş Okuma ---
        joy_fwd   = 0.0
        joy_right = 0.0
        joy_up    = 0.0
        joy_yaw   = 0.0
        joy_pitch = 0.0

        if joystick:
            num_axes = joystick.get_numaxes()
            JOY_DEADZONE = 0.28 if is_rc_controller else 0.20

            def joy_axis(idx, invert=False):
                if idx >= num_axes:
                    return 0.0
                v = joystick.get_axis(idx)
                v = 0.0 if abs(v) < JOY_DEADZONE else v
                return -v if invert else v

            if is_rc_controller:
                # RC Kumanda (iFlight Commando 8): Sadece analog çubuk hareketini al
                joy_right = joy_axis(0)
                joy_fwd   = joy_axis(1, invert=True)
                if num_axes >= 4:
                    joy_yaw = joy_axis(3) * 0.7
            else:
                # Standart Gamepad
                joy_fwd   = joy_axis(1, invert=True)
                joy_right = joy_axis(0)
                joy_yaw   = joy_axis(2) or joy_axis(3)
                joy_pitch = joy_axis(3 if num_axes <= 4 else 4, invert=True)
                if num_axes >= 6:
                    r2_raw = joystick.get_axis(5)
                    l2_raw = joystick.get_axis(4)
                    r2 = max(0.0, r2_raw) if r2_raw > 0.2 else 0.0
                    l2 = max(0.0, l2_raw) if l2_raw > 0.2 else 0.0
                    joy_up = r2 - l2

            # Sağ stick kamera yönlendirmesi
            cam_sensitivity = 90.0
            if abs(joy_yaw) > 0.0:
                target_yaw += joy_yaw * cam_sensitivity * dt
                user_override_look = True
            if abs(joy_pitch) > 0.0:
                target_pitch += joy_pitch * cam_sensitivity * dt * 0.6
                target_pitch = max(-89.0, min(89.0, target_pitch))
                user_override_look = True

        # Kullanıcı simülasyondayken klavyeden hareket tuşuna dokunursa otomatik simülasyondan çıkar
        if (is_w or is_s or is_a or is_d or is_up or is_down) and auto_tour:
            auto_tour = False
            hud.dirty = True
            hud.set_toast("🌐 Serbest Gezinti Moduna Geçildi (Manuel Kontrol)", 2.0)

        if auto_tour and len(auto_waypoints) > 1:
            # 🎬 OTOMATİK SİMÜLASYON MODU (Dronun gittiği rota boyunca akıcı uçuş)
            tour_progress += dt * (tour_speed * 110.0)
            if tour_progress >= len(auto_waypoints) - 1:
                tour_progress = 0.0
            idx0 = int(tour_progress)
            idx1 = min(idx0 + 1, len(auto_waypoints) - 1)
            alpha = tour_progress - idx0
            pos0, pos1 = auto_waypoints[idx0], auto_waypoints[idx1]
            t_pos = (1.0 - alpha) * pos0 + alpha * pos1
            drone_x, drone_y, drone_z = float(t_pos[0]), float(t_pos[1]), float(t_pos[2])

            # Kullanıcı fareyle serbest bakışa geçmediyse rotayı takip et
            if not user_override_look:
                look_idx = min(idx0 + 35, len(auto_waypoints) - 1)
                look_pos = auto_waypoints[look_idx]
                dir_v = look_pos - t_pos
                if np.linalg.norm(dir_v) > 0.01:
                    t_yaw = math.degrees(math.atan2(dir_v[0], dir_v[2]))
                    t_pitch = math.degrees(math.atan2(-dir_v[1], math.sqrt(dir_v[0]**2 + dir_v[2]**2)))
                    diff_yaw = (t_yaw - target_yaw + 180.0) % 360.0 - 180.0
                    target_yaw += diff_yaw * min(1.0, 10.0 * dt)
                    target_pitch += (t_pitch - target_pitch) * min(1.0, 10.0 * dt)
        else:
            # 🌐 SERBEST GEZİNTİ MODU: W, A, S, D + Boşluk / C ile doğrudan anında tepki
            fly_speed = 6.5 * (2.2 if fast_mode else 1.0)

            fwd_val = 0.0
            if is_w: fwd_val += 1.0
            if is_s: fwd_val -= 1.0
            if abs(joy_fwd) > 0.0: fwd_val += joy_fwd
            fwd_val = max(-1.0, min(1.0, fwd_val))

            side_val = 0.0
            if is_d: side_val += 1.0
            if is_a: side_val -= 1.0
            if abs(joy_right) > 0.0: side_val += joy_right
            side_val = max(-1.0, min(1.0, side_val))

            vert_val = 0.0
            if is_up: vert_val += 1.0
            if is_down: vert_val -= 1.0
            if not is_rc_controller and abs(joy_up) > 0.2: vert_val += joy_up
            vert_val = max(-1.0, min(1.0, vert_val))

            # Yatay (XZ) düzleminde ileri vektörü normalize et:
            # W/S yürüyüşü yükseklik (Y) değerini ASLA etkilemez!
            fwd_len = math.sqrt(fwd_x**2 + fwd_z**2)
            if fwd_len > 1e-4:
                norm_fwd_x = fwd_x / fwd_len
                norm_fwd_z = fwd_z / fwd_len
            else:
                norm_fwd_x = math.sin(rad_yaw)
                norm_fwd_z = math.cos(rad_yaw)

            drone_x += (norm_fwd_x * fwd_val + right_x * side_val) * fly_speed * dt
            drone_z += (norm_fwd_z * fwd_val + right_z * side_val) * fly_speed * dt
            drone_y += vert_val * fly_speed * dt
            if frame_count % 180 == 0:
                print(f"[STATUS] y={drone_y:.2f}m | vert={vert_val:.1f} | fwd={fwd_val:.1f} | auto_sim={auto_tour}")

        # 🎥 Kamera Konumunu Ayarla (Saf doğrudan serbest kamera)
        render_cam_x = drone_x
        render_cam_y = drone_y
        render_cam_z = drone_z
        eff_cam_pitch = cam_pitch

        # OpenGL Ekranını Temizle
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Optik Zoom Yumuşatması (Jitter ve ani sıçrama önleyici enterpolasyon)
        cam_fov += (target_fov - cam_fov) * min(1.0, 14.0 * dt)

        # Projeksiyon Matrisi (Perspektif / Zoom FOV)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(cam_fov, (win_w / max(win_h, 1)), 0.02, 300.0)

        # ModelView Matrisi (Kamera Dönüş ve Konum Dönüşümleri)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glRotatef(eff_cam_pitch, 1, 0, 0)
        glRotatef(-cam_yaw, 0, 1, 0)
        glTranslatef(-render_cam_x, -render_cam_y, -render_cam_z)

        # 🏠 Tavan Kesme Düzlemini Donanımsal Olarak Uygula
        culler.apply_gl()

        # Zemin Izgarasını Çiz
        glDisable(GL_LIGHTING)
        glColor4f(0.18, 0.28, 0.42, 0.25)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_grid)
        raw_gl.glVertexPointer(3, GL_FLOAT, 0, ctypes.c_void_p(0))
        glEnableClientState(GL_VERTEX_ARRAY)
        glDrawArrays(GL_LINES, 0, grid_n)
        glDisableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        # Kat Planı Mimari Çerçevesi (Kuşbakışında Çizilir)
        if top_down_view:
            FloorplanEstimator.draw_blueprint_grid(room_bounds)

        # 🚁 Dron Modeli: Kullanıcı talebi üzerine haritada serbest gezinim için kaldırıldı (Saf 3DGS haritası)
        # drone_model.draw_3d(floor_y=room_bounds['min_y'])

        # Yeşil Kamera Yörünge Çizgisi
        if vbo_traj is not None and traj_n > 0:
            glLineWidth(2.0)
            glColor4f(0.1, 0.95, 0.35, 0.6)
            glBindBuffer(GL_ARRAY_BUFFER, vbo_traj)
            raw_gl.glVertexPointer(3, GL_FLOAT, 0, ctypes.c_void_p(0))
            glEnableClientState(GL_VERTEX_ARRAY)
            glDrawArrays(GL_LINE_STRIP, 0, traj_n)
            glDisableClientState(GL_VERTEX_ARRAY)
            glBindBuffer(GL_ARRAY_BUFFER, 0)

        # 📏 3B Lazer Cetvel Çizimi
        ruler.draw_3d()


        # 🧊 OctoMap İçi Dolu 3B Voksel Küpleri (Solid 3D Cubes) / 🔮 3DGS Nokta Render
        if octomap_engine.active and vbo_octo_xyz is not None and octo_line_count > 0:
            glBindBuffer(GL_ARRAY_BUFFER, vbo_octo_xyz)
            raw_gl.glVertexPointer(3, GL_FLOAT, 0, ctypes.c_void_p(0))
            glEnableClientState(GL_VERTEX_ARRAY)

            glBindBuffer(GL_ARRAY_BUFFER, vbo_octo_rgba)
            raw_gl.glColorPointer(4, GL_FLOAT, 0, ctypes.c_void_p(0))
            glEnableClientState(GL_COLOR_ARRAY)

            glDrawArrays(GL_TRIANGLES, 0, octo_line_count)

            glDisableClientState(GL_COLOR_ARRAY)
            glDisableClientState(GL_VERTEX_ARRAY)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
        else:
            # 🔮 3D Gaussian Splats Çizimi (Yumuşak Gaussian Splat & Mesafe Uyarlamalı)
            zoom_scale = 60.0 / max(cam_fov, 15.0)
            
            # KASMAYI ÖNLEME: GPU Fill-rate darboğazını (lag) önlemek için max splat boyutunu 120'den 45'e çektik
            max_allowed_size = 45.0
            effective_point_size = max(1.0, min(max_allowed_size, splat_point_size * zoom_scale))
            glPointSize(effective_point_size)

            if use_gaussian_splat:
                # 3B Mesafe Azaltımı: Yaklaştıkça noktalar dinamik büyüyerek boşlukları kapatır ve kenetlenir
                atten_arr = (GLfloat * 3)(0.15, 0.0, 0.25)
                glPointParameterfv(GL_POINT_DISTANCE_ATTENUATION, atten_arr)
                glPointParameterf(GL_POINT_SIZE_MIN, 1.5)
                glPointParameterf(GL_POINT_SIZE_MAX, max_allowed_size)

                # Gaussian Point Sprite: Noktaları sert kareler yerine yumuşak dairesel splat olarak çiz
                glEnable(GL_POINT_SPRITE)
                glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE)
                glEnable(GL_TEXTURE_2D)
                glBindTexture(GL_TEXTURE_2D, tex_gaussian_splat)
                glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
                glEnable(GL_ALPHA_TEST)
                glAlphaFunc(GL_GREATER, 0.05)
            else:
                atten_arr = (GLfloat * 3)(1.0, 0.0, 0.0)
                glPointParameterfv(GL_POINT_DISTANCE_ATTENUATION, atten_arr)
                glDisable(GL_POINT_SPRITE)
                glDisable(GL_TEXTURE_2D)
                glDisable(GL_ALPHA_TEST)

            glBindBuffer(GL_ARRAY_BUFFER, vbo_xyz)
            raw_gl.glVertexPointer(3, GL_FLOAT, 0, ctypes.c_void_p(0))
            glEnableClientState(GL_VERTEX_ARRAY)

            glBindBuffer(GL_ARRAY_BUFFER, vbo_rgba)
            raw_gl.glColorPointer(4, GL_FLOAT, 0, ctypes.c_void_p(0))
            glEnableClientState(GL_COLOR_ARRAY)

            glDrawArrays(GL_POINTS, 0, num_splats)

            glDisableClientState(GL_COLOR_ARRAY)
            glDisableClientState(GL_VERTEX_ARRAY)
            glBindBuffer(GL_ARRAY_BUFFER, 0)

            if use_gaussian_splat:
                glDisable(GL_POINT_SPRITE)
                glDisable(GL_TEXTURE_2D)
                glDisable(GL_ALPHA_TEST)
                atten_std = (GLfloat * 3)(1.0, 0.0, 0.0)
                glPointParameterfv(GL_POINT_DISTANCE_ATTENUATION, atten_std)


        # Tavan Kesme Düzlemini Kaldır (2B Arayüz için)
        culler.restore_gl()

        # 🌊 Optik Flow Ok Izgara Çizimi (hareket ederken görünür)
        if show_optical_flow and not auto_tour:
            optical_flow.draw_gl(win_w, win_h, cam_yaw)

        # 💥 Çarpışma Kırmızı Flaş Efekti (ekranın kenarlarına kırmızı overlay)
        if physics.crash_timer > 0.0:
            flash_alpha = min(0.55, physics.crash_timer * 0.9)
            glPushAttrib(GL_ALL_ATTRIB_BITS)
            glMatrixMode(GL_PROJECTION)
            glPushMatrix()
            glLoadIdentity()
            glOrtho(0, win_w, win_h, 0, -1, 1)
            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            glLoadIdentity()
            glDisable(GL_DEPTH_TEST)
            glDisable(GL_TEXTURE_2D)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            border = 60
            glColor4f(1.0, 0.05, 0.05, flash_alpha)
            glBegin(GL_QUADS)
            # Sol kenar
            glVertex2f(0, 0); glVertex2f(border, 0)
            glVertex2f(border, win_h); glVertex2f(0, win_h)
            # Sağ kenar
            glVertex2f(win_w - border, 0); glVertex2f(win_w, 0)
            glVertex2f(win_w, win_h); glVertex2f(win_w - border, win_h)
            # Üst kenar
            glVertex2f(0, 0); glVertex2f(win_w, 0)
            glVertex2f(win_w, border); glVertex2f(0, border)
            # Alt kenar
            glVertex2f(0, win_h - border); glVertex2f(win_w, win_h - border)
            glVertex2f(win_w, win_h); glVertex2f(0, win_h)
            glEnd()
            glMatrixMode(GL_MODELVIEW)
            glPopMatrix()
            glMatrixMode(GL_PROJECTION)
            glPopMatrix()
            glPopAttrib()

        # 2B Yarı Saydam Kontrol Paneli & HUD Çizimi
        hud.render_gl(win_w, win_h, current_fps, num_splats, cam_fov, is_flipped,
                      ruler, recorder, room_bounds, top_down_view, culler, auto_tour=auto_tour)

        # MP4 Video Kare Kaydı
        if recorder.recording:
            recorder.capture_frame(win_w, win_h)

        pygame.display.flip()

    # Çıkışta kayıt devam ediyorsa düzgün kapat
    if recorder.recording:
        recorder.toggle(win_w, win_h)

    pygame.quit()


# Doğrudan terminalden çalıştırıldığında (Örn: python gaussian_renderer.py)
if __name__ == "__main__":
    ply = sys.argv[1] if len(sys.argv) > 1 else "gaussian_scene.ply"
    view_gaussian_splats(ply)
