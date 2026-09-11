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
import math
import time
import numpy as np

# Linux / Windows Optimus Hibrit Laptoplar için Harici NVIDIA RTX GPU'yu Zorlama
os.environ["SHIM_MCCOMPAT"] = "0x000000001"
os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
os.environ["__GL_SYNC_TO_VBLANK"] = "0"
os.environ["vblank_mode"] = "0"
os.environ["SDL_VIDEO_X11_NODIRECTCOLOR"] = "1"
# ✅ Wayland kaynaklı performans çöküşünü (10 FPS) önlemek için X11 zorlaması
os.environ["SDL_VIDEODRIVER"] = "x11"
os.environ["GDK_BACKEND"] = "x11"
os.environ["QT_QPA_PLATFORM"] = "xcb"

# Linux üzerinde libGL/libGLX yüklenmeden önce NVIDIA PRIME Offload ortam değişkenlerini garantileme
if sys.platform.startswith("linux") and os.environ.get("_3DGS_GPU_CHECK") != "1":
    os.environ["_3DGS_GPU_CHECK"] = "1"
    os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
    os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
    os.environ["__GL_SYNC_TO_VBLANK"] = "0"
    os.environ["vblank_mode"] = "0"
    os.environ["SDL_VIDEODRIVER"] = "x11"
    try:
        os.execvpe(sys.executable, [sys.executable] + sys.argv, os.environ)
    except Exception:
        pass

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
            self.font_title = pygame.font.SysFont("Segoe UI", 16, bold=True)
            self.font_bold = pygame.font.SysFont("Segoe UI", 13, bold=True)
            self.font_norm = pygame.font.SysFont("Segoe UI", 12)
            self.font_small = pygame.font.SysFont("Segoe UI", 11)
        except Exception:
            self.font_title = pygame.font.Font(None, 19)
            self.font_bold = pygame.font.Font(None, 15)
            self.font_norm = pygame.font.Font(None, 14)
            self.font_small = pygame.font.Font(None, 13)

        self.tex_id = None          # OpenGL Texture Kimliği (Lazy initialization ile GPU context oluştuktan sonra tahsis edilir)
        self.tex_w = 0
        self.tex_h = 0
        self.show_panel = True      # [H] veya [TAB] ile kontrol paneli açık/kapalı durumu
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
                      ruler, recorder, room_bounds, top_down_view, culler):
        """Tüm 2B arayüz kartlarını saydam Pygame Surface üzerinde çizer."""
        surf = pygame.Surface((win_w, win_h), pygame.SRCALPHA)
        now = time.time()

        # ---------------------------------------------------------------------
        # 1. Bildirim Mesajı (Toast Notification)
        # ---------------------------------------------------------------------
        if now < self.toast_timer and self.toast_msg:
            tw, th = 520, 42
            tx = (win_w - tw) // 2
            ty = 16
            pygame.draw.rect(surf, (18, 24, 38, 235), (tx, ty, tw, th), border_radius=8)
            pygame.draw.rect(surf, (0, 180, 255, 200), (tx, ty, tw, th), width=2, border_radius=8)
            t_txt = self.font_bold.render(self.toast_msg, True, (255, 255, 255))
            surf.blit(t_txt, (tx + (tw - t_txt.get_width()) // 2, ty + 11))

        # ---------------------------------------------------------------------
        # 2. Üst Durum Çubuğu (FPS, Splat Sayısı, Zoom, Tavan ve Kayıt Durumu)
        # ---------------------------------------------------------------------
        bar_w, bar_h = 670, 36
        bx, by = 16, 16
        pygame.draw.rect(surf, (12, 18, 28, 210), (bx, by, bar_w, bar_h), border_radius=6)
        pygame.draw.rect(surf, (40, 60, 85, 180), (bx, by, bar_w, bar_h), width=1, border_radius=6)

        tavan_durum = f"🏠 Tavan: {'KAPALI (%{:.0f})'.format(culler.cut_ratio*100) if culler.enabled else 'AÇIK'}"
        rec_icon = "🔴 KAYITTA" if recorder.recording else ""
        stat_text = f"⚡ {fps} FPS  |  🔮 {num_splats:,} Splats  |  🔍 Zoom: {cam_fov:.0f}°  |  {tavan_durum}  |  🪞 Ayna: {'DÜZELTİLDİ' if is_flipped else 'ORİJİNAL'} {rec_icon}"
        stat_surf = self.font_bold.render(stat_text, True, (0, 230, 190))
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
            pw, ph = 430, 500
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
                ("🕹️ HAREKET & UÇUŞ (FİZİK MOTORU)", ""),
                ("[W / A / S / D]", "İleri / Sol / Geri / Sağ Uçuş"),
                ("[SPACE]", "🚀 Yukarı Gaz (Bırakınca yerçekimi alçaltır)"),
                ("[SHIFT]", "⚡ Hızlı Uçuş Modu (2x Hız)"),
                ("[Fare Sol Sürükle]", "360° Serbest Kamera Açısı"),
                ("", ""),
                ("🎮 COMMANDO 8 KUMANDA", ""),
                ("[Sol Stick]", "🕹️ İleri / Geri / Sağ / Sol Hareket"),
                ("[Sağ Stick]", "📷 Kamera Yaw / Pitch Kontrolü"),
                ("[R2 Tetik]", "🚀 Yukarı Gaz  |  [L2] Alçal"),
                ("[RB]", "⚡ Hızlı Uçuş  |  [D-Pad] Tavan/Splat"),
                ("[A]", "📋 Panel  [B] Flow  [X] Tavan  [Y] Sıfırla"),
                ("[LB]", "🧊 OctoMap  [Select] Tur  [Start] Çıkış"),
                ("", ""),
                ("🔍 BÜYÜTME & KÜÇÜLTME (ZOOM)", ""),
                ("[+] / [-]", "🔍 Kamera Yakınlaş (Büyüt) / Uzaklaş (Küçült)"),
                ("[Fare Tekerleği]", "🔍 Hızlı Kamera Büyüt / Küçült"),
                ("[0] (Sıfır)", "Standart Zoom Açısına Sıfırla (60°)"),
                ("[X] / [C]", "🔮 Splat Nokta Boyutunu Büyüt / Küçült"),
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
                ("🚀 PRO ÇIKTI VE VİDEO", ""),
                ("[I] Tuşu", "🌊 Optik Flow HUD Aç / Kapat"),
                ("[V] Tuşu", "🎬 60 FPS MP4 Video Kaydını Başlat / Durdur"),
                ("[K] Tuşu", "🌐 Bağımsız Web / HTML 3B Modelini Dışa Aktar"),
                ("[M] Tuşu", "🪞 Sağ / Sol Yönünü Aynala / Düzelt"),
                ("[F] Tuşu", "🚁 3. Şahıs / 1. Şahıs / Serbest Dron Kamerası"),
                ("[P] Tuşu", "🎥 Sinematik Tur | [R]: Başa Sıfırla | [ESC]: Çıkış"),
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
                  ruler, recorder, room_bounds, top_down_view, culler):
        """2B Arayüzü OpenGL 2D Texture olarak 3B sahnenin üzerine çizer (Önbellekli)."""
        now = time.time()
        # Saniyede en fazla 10 kez veya bir etkileşimde (dirty) GPU'ya doku aktarımı yap
        if self.dirty or (now - self.last_update >= 0.1) or (self.tex_id is None):
            surf = self.build_surface(win_w, win_h, fps, num_splats, cam_fov, is_flipped,
                                      ruler, recorder, room_bounds, top_down_view, culler)
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

    # 2. Kamera Uçuş Yörüngesini Yükle
    if os.path.exists(traj_path):
        traj_data = np.load(traj_path, allow_pickle=True)
        auto_waypoints = traj_data["positions"].copy()

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

    # Splat nokta boyutu varsayılanı (Boşluk kalmaması için orantılı)
    splat_point_size = 5.4 if cur_stride == 5 else (4.8 if cur_stride == 4 else (4.3 if cur_stride == 3 else (3.8 if cur_stride == 2 else 3.2)))

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
    show_optical_flow = True  # [I] tuşuyla açılıp kapatılabilir

    # Zemin yüksekliğini fizik motoruna bildir
    if room_bounds is not None:
        physics.set_floor(room_bounds['min_y'])

    hud.set_toast(f"✨ 3DGS Başlatıldı! {quality_names[quality_idx]} ([F] Kamera, [B] Mod)", 5.0)

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

    # 🎮 Commando 8 Kumanda Başlatma
    pygame.joystick.init()
    joystick = None
    joystick_name = "Yok"
    if pygame.joystick.get_count() > 0:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        joystick_name = joystick.get_name()
        print(f" 🎮 Kumanda Bulundu: {joystick_name}")
        print(f"    Axis: {joystick.get_numaxes()} | Buton: {joystick.get_numbuttons()} | Hat: {joystick.get_numhats()}")
    else:
        print(" ⌨️  Kumanda bulunamadı — sadece klavye/fare ile kontrol.")

    # MSAA kapatıldı (nokta bulutlarında gereksiz GPU yükünü önler)
    pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 0)
    pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 0)
    # Linux Wayland/XWayland V-Sync tampon takılmasını önleme
    pygame.display.gl_set_attribute(pygame.GL_SWAP_CONTROL, 0)
    pygame.display.set_mode((win_w, win_h), DOUBLEBUF | OPENGL | RESIZABLE)
    # OPENGL modunda key.get_pressed() için event.pump() çağrılmalı
    # key.set_repeat kaldırıldı: get_pressed() kendi başına sürekli tuş okur
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
    glDisable(GL_POINT_SMOOTH)  # Modern GPU donanım hızlandırması için legacy point smooth kapatıldı
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

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

        if st == 5:
            splat_point_size = 5.4
        elif st == 4:
            splat_point_size = 4.8
        elif st == 3:
            splat_point_size = 4.3
        elif st == 2:
            splat_point_size = 3.8
        else:
            splat_point_size = 3.2

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

    # Başlangıç Kamera Konumu ve Çekim Yönü
    if len(auto_waypoints) > 1:
        drone_x, drone_y, drone_z = float(auto_waypoints[0][0]), float(auto_waypoints[0][1]), float(auto_waypoints[0][2])
        dir0 = auto_waypoints[min(60, len(auto_waypoints)-1)] - auto_waypoints[0]
        if np.linalg.norm(dir0) > 1e-3:
            target_yaw = math.degrees(math.atan2(dir0[0], dir0[2]))
            target_pitch = math.degrees(math.atan2(-dir0[1], math.sqrt(dir0[0]**2 + dir0[2]**2)))
            cam_yaw, cam_pitch = target_yaw, target_pitch
        else:
            cam_yaw, cam_pitch = 0.0, 0.0
            target_yaw, target_pitch = 0.0, 0.0
    else:
        drone_x, drone_y, drone_z = 0.0, 1.5, 0.0
        center = np.mean(xyz, axis=0)
        dir0 = center - np.array([drone_x, drone_y, drone_z])
        if np.linalg.norm(dir0) > 1e-3:
            target_yaw = math.degrees(math.atan2(dir0[0], dir0[2]))
            target_pitch = math.degrees(math.atan2(-dir0[1], math.sqrt(dir0[0]**2 + dir0[2]**2)))
            cam_yaw, cam_pitch = target_yaw, target_pitch
        else:
            cam_yaw, cam_pitch = 0.0, 0.0
            target_yaw, target_pitch = 0.0, 0.0

    # Fizik motorunu başlangıç konumuyla senkronize et
    physics.x, physics.y, physics.z = drone_x, drone_y, drone_z
    if room_bounds is not None:
        physics.set_floor(room_bounds['min_y'])


    cam_fov = 60.0              # Görüş alanı (FOV / Zoom)
    target_fov = 60.0           # Hedef görüş alanı (Yumuşak optik zoom)
    auto_tour = False           # Sinematik tur aktif mi?
    tour_progress = 0.0
    tour_speed = 1.0
    show_frustums = True        # Kamera piramitleri açık mı?
    top_down_view = False       # Kuşbakışı modu açık mı?
    user_override_look = False  # Kullanıcı fareyle serbest bakış modunda mı?

    clock = pygame.time.Clock()
    mouse_down_left = False
    last_mouse_pos = (0, 0)
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
            elif event.type == pygame.JOYBUTTONDOWN and joystick:
                btn = event.button
                # ┌─ Commando 8 Buton Haritası ──────────────────────┐
                # │  0 = A / Çapraz   → Kontrol Paneli aç/kapat [H]  │
                # │  1 = B / Daire    → Optik Flow aç/kapat [I]      │
                # │  2 = X / Kare     → Tavan gizle/göster [G]       │
                # │  3 = Y / Üçgen    → Sıfırla [R]                  │
                # │  4 = LB / L1      → OctoMap + Çarpışma Haritası  │
                # │  5 = RB / R1      → Hızlı Mod (polling'den okunur)│
                # │  6 = Select/Share → Sinematik Tur [P]             │
                # │  7 = Start/Options→ Çıkış [ESC]                  │
                # │  8 = Sol Stick Bas→ Zoom Sıfırla [0]             │
                # │  9 = Sağ Stick Bas→ Aynalama [M]                 │
                # │ 10 = Home/PS      → Kamera Modu [F]              │
                # └──────────────────────────────────────────────────┘
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
                    if len(auto_waypoints) > 1:
                        drone_x, drone_y, drone_z = float(auto_waypoints[0][0]), float(auto_waypoints[0][1]), float(auto_waypoints[0][2])
                    else:
                        drone_x, drone_y, drone_z = 0.0, 1.5, 0.0
                    physics.x, physics.y, physics.z = drone_x, drone_y, drone_z
                    physics.vx = physics.vy = physics.vz = 0.0
                    target_yaw, target_pitch = 0.0, 0.0; cam_yaw, cam_pitch = 0.0, 0.0
                    cam_fov = 60.0; target_fov = 60.0
                    hud.set_toast("🔄 Sıfırlandı (Commando 8: Y)", 2.0)
                elif btn == 4:  # LB → OctoMap + Çarpışma Haritası
                    o_active = octomap_engine.toggle()
                    if o_active and vbo_octo_xyz is None:
                        hud.set_toast("⏳ OctoMap hesaplanıyor... (bekle)", 2.0)
                    elif o_active:
                        hud.set_toast(f"🧊 OctoMap: AÇIK ({octomap_engine.num_cubes:,} Küp)", 2.0)
                    else:
                        hud.set_toast("🔮 OctoMap: KAPALI", 1.5)
                elif btn == 6:  # Select → Sinematik Tur
                    if len(auto_waypoints) > 0:
                        auto_tour = not auto_tour
                        user_override_look = False
                        hud.set_toast("🎥 Tur: " + ("BAŞLATILDI" if auto_tour else "DURDURULDU"), 2.0)
                elif btn == 7:  # Start → Çıkış
                    running = False
                elif btn == 8:  # Sol Stick Bas → Zoom Sıfırla
                    target_fov = 60.0; cam_fov = 60.0
                    hud.set_toast("🔍 Zoom Sıfırlandı (60°)", 1.2)
                elif btn == 9:  # Sağ Stick Bas → Aynalama
                    is_flipped = not is_flipped
                    raw_xyz[:, 0] = -raw_xyz[:, 0]
                    apply_quality_mode(quality_idx)
                    drone_x = -drone_x
                    physics.x = drone_x
                    hud.set_toast(f"🪞 Ayna: {'TERS' if is_flipped else 'ORİJİNAL'}", 2.0)
                elif btn == 10: # Home → Kamera Modu
                    drone_view_mode = (drone_view_mode + 1) % 3
                    modes = ["🚁 3. Şahıs Takip", "📷 1. Şahıs FPV", "🌐 Serbest"]
                    hud.set_toast(f"Kamera: {modes[drone_view_mode]}", 2.0)

            # D-Pad (Hat) → Tavan Kesme Seviyesi Ayarı
            elif event.type == pygame.JOYHATMOTION and joystick:
                hx, hy = event.value
                if hy > 0:   # D-Pad Yukarı → Tavan yükselt
                    r = culler.adjust_cut(+0.05); culler.enabled = True
                    hud.set_toast(f"✂️ Tavan: %{r*100:.0f}", 1.0)
                elif hy < 0: # D-Pad Aşağı → Tavan alçalt
                    r = culler.adjust_cut(-0.05); culler.enabled = True
                    hud.set_toast(f"✂️ Tavan: %{r*100:.0f}", 1.0)
                elif hx > 0: # D-Pad Sağ → Splat boyutu büyüt
                    splat_point_size = min(15.0, splat_point_size + 0.5)
                    hud.set_toast(f"🔮 Splat: {splat_point_size:.1f}", 1.0)
                elif hx < 0: # D-Pad Sol → Splat boyutu küçült
                    splat_point_size = max(1.0, splat_point_size - 0.5)
                    hud.set_toast(f"🔮 Splat: {splat_point_size:.1f}", 1.0)

            elif event.type == KEYDOWN:

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
                elif event.key in (K_x, K_z):
                    # 🔮 Splat Nokta Boyutunu Büyüt
                    splat_point_size = min(15.0, splat_point_size + 0.5)
                    hud.set_toast(f"🔮 Splat Boyutu Büyütüldü: {splat_point_size:.1f}", 1.5)
                elif event.key == K_c:
                    # 🔮 Splat Nokta Boyutunu Küçült
                    splat_point_size = max(1.0, splat_point_size - 0.5)
                    hud.set_toast(f"🔮 Splat Boyutu Küçültüldü: {splat_point_size:.1f}", 1.5)
                elif event.key in (K_0, K_KP0):
                    # 🔍 Zoom Sıfırla (60°)
                    target_fov = 60.0
                    cam_fov = 60.0
                    hud.set_toast("🔍 Zoom Sıfırlandı (60°)", 1.5)
                elif event.key == K_m:
                    # 🪞 Sağ/Sol Aynalama ([M] Tuşu)
                    is_flipped = not is_flipped
                    raw_xyz[:, 0] = -raw_xyz[:, 0]
                    apply_quality_mode(quality_idx)
                    if len(auto_waypoints) > 0:
                        auto_waypoints[:, 0] = -auto_waypoints[:, 0]
                        traj_arr = auto_waypoints.astype(np.float32)
                        glBindBuffer(GL_ARRAY_BUFFER, vbo_traj)
                        glBufferData(GL_ARRAY_BUFFER, traj_arr.nbytes, traj_arr, GL_STATIC_DRAW)
                        glBindBuffer(GL_ARRAY_BUFFER, 0)
                    drone_x = -drone_x
                    hud.set_toast(f"🪞 Sağ/Sol Yön: {'AYNALANDI (TERS)' if is_flipped else 'ORİJİNAL'}", 2.5)

                elif event.key == K_b:
                    # ⚡ Performans / Kalite Modu Değiştir ([B] Tuşu)
                    quality_idx = (quality_idx + 1) % len(quality_strides)
                    apply_quality_mode(quality_idx)
                    hud.set_toast(quality_names[quality_idx], 3.0)

                elif event.key == K_p and len(auto_waypoints) > 0:
                    # 🎥 Sinematik Otomatik Tur (Rayda İlerleme + Serbest Fare Bakışı)
                    auto_tour = not auto_tour
                    user_override_look = False
                    hud.set_toast("🎥 Kamera Turu: " + ("BAŞLATILDI (Fareyle İstediğin Yöne Bakabilirsin!)" if auto_tour else "DURDURULDU"), 3.0)

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
                    # 💡 Dron Feneri ve Lazer Aç / Kapa
                    drone_model.spotlight = not drone_model.spotlight
                    drone_model.laser = drone_model.spotlight
                    hud.set_toast(f"💡 Dron Feneri & Lazer: {'AÇIK' if drone_model.spotlight else 'KAPALI'}", 2.0)

                elif event.key == K_1 and len(auto_waypoints) > 0:
                    auto_tour = False; p = auto_waypoints[0]; drone_x, drone_y, drone_z = float(p[0]), float(p[1]), float(p[2])
                    physics.x, physics.y, physics.z = drone_x, drone_y, drone_z
                    physics.vx = physics.vy = physics.vz = 0.0
                    hud.set_toast("🚪 Konum 1'e Işınlanıldı (Giriş)", 1.5)
                elif event.key == K_2 and len(auto_waypoints) > 0:
                    auto_tour = False; p = auto_waypoints[len(auto_waypoints)//3]; drone_x, drone_y, drone_z = float(p[0]), float(p[1]), float(p[2])
                    physics.x, physics.y, physics.z = drone_x, drone_y, drone_z
                    physics.vx = physics.vy = physics.vz = 0.0
                    hud.set_toast("🚪 Konum 2'ye Işınlanıldı (Orta 1)", 1.5)
                elif event.key == K_3 and len(auto_waypoints) > 0:
                    auto_tour = False; p = auto_waypoints[2*len(auto_waypoints)//3]; drone_x, drone_y, drone_z = float(p[0]), float(p[1]), float(p[2])
                    physics.x, physics.y, physics.z = drone_x, drone_y, drone_z
                    physics.vx = physics.vy = physics.vz = 0.0
                    hud.set_toast("🚪 Konum 3'e Işınlanıldı (Orta 2)", 1.5)
                elif event.key == K_4 and len(auto_waypoints) > 0:
                    auto_tour = False; p = auto_waypoints[-1]; drone_x, drone_y, drone_z = float(p[0]), float(p[1]), float(p[2])
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
                    # 🔄 Konum ve Kamera Sıfırlama (Başlangıç rotasına dön)
                    auto_tour = False; tour_progress = 0.0; top_down_view = False
                    culler.enabled = False
                    if len(auto_waypoints) > 1:
                        drone_x, drone_y, drone_z = float(auto_waypoints[0][0]), float(auto_waypoints[0][1]), float(auto_waypoints[0][2])
                        dir0 = auto_waypoints[min(60, len(auto_waypoints)-1)] - auto_waypoints[0]
                        if np.linalg.norm(dir0) > 1e-3:
                            target_yaw = math.degrees(math.atan2(dir0[0], dir0[2]))
                            target_pitch = math.degrees(math.atan2(-dir0[1], math.sqrt(dir0[0]**2 + dir0[2]**2)))
                            cam_yaw, cam_pitch = target_yaw, target_pitch
                    else:
                        drone_x, drone_y, drone_z = 0.0, 1.5, 0.0
                        target_yaw, target_pitch = 0.0, 0.0; cam_yaw, cam_pitch = 0.0, 0.0
                    # Fizik motorunu da sıfırla
                    physics.vx = 0.0; physics.vy = 0.0; physics.vz = 0.0
                    physics.x, physics.y, physics.z = drone_x, drone_y, drone_z
                    cam_fov = 60.0
                    target_fov = 60.0
                    hud.set_toast("🔄 Kamera Başlangıç Rotasına Sıfırlandı", 2.0)

                elif event.key == K_i:
                    # 🌊 Optik Flow HUD Aç / Kapat ([I] Tuşu)
                    show_optical_flow = not show_optical_flow
                    hud.set_toast(f"🌊 Optik Flow: {'AÇIK' if show_optical_flow else 'KAPALI'}", 1.5)

                elif event.key == K_ESCAPE:
                    running = False


        # Keskin ve doğrudan kamera yönü (Sıfır gecikmeli fare bakışı)
        cam_yaw = target_yaw
        cam_pitch = target_pitch

        rad_yaw, rad_pitch = math.radians(cam_yaw), math.radians(cam_pitch)
        fwd_x = math.sin(rad_yaw) * math.cos(rad_pitch)
        fwd_y = -math.sin(rad_pitch)
        fwd_z = math.cos(rad_yaw) * math.cos(rad_pitch)
        right_x = math.cos(rad_yaw)
        right_z = -math.sin(rad_yaw)

        # ⚠️ OPENGL modunda pygame.key.get_pressed() çalışabilmesi için
        # event.pump() zorunludur — bu olmadan tuşlar hiç okunmaz!
        pygame.event.pump()

        # Kamera Hareketi (Sinematik Tur veya Fizik Tabanlı W/A/S/D Klavye Uçuşu)
        if auto_tour and len(auto_waypoints) > 1:
            tour_progress += dt * (tour_speed * 110.0)
            if tour_progress >= len(auto_waypoints) - 1:
                tour_progress = 0.0
            idx0 = int(tour_progress)
            idx1 = min(idx0 + 1, len(auto_waypoints) - 1)
            alpha = tour_progress - idx0
            pos0, pos1 = auto_waypoints[idx0], auto_waypoints[idx1]
            t_pos = (1.0 - alpha) * pos0 + alpha * pos1
            drone_x, drone_y, drone_z = float(t_pos[0]), float(t_pos[1]), float(t_pos[2])
            # Tur sırasında fizik motorunu da senkronize tut
            physics.x, physics.y, physics.z = drone_x, drone_y, drone_z
            physics.vx = physics.vy = physics.vz = 0.0

            # Kullanıcı fareyle serbest bakışa geçtiyse yönü kilitleme; dokunmadıysa rotayı takip etsin
            if not user_override_look:
                look_idx = min(idx0 + 50, len(auto_waypoints) - 1)
                look_pos = auto_waypoints[look_idx]
                dir_v = look_pos - t_pos
                if np.linalg.norm(dir_v) > 0.01:
                    t_yaw = math.degrees(math.atan2(dir_v[0], dir_v[2]))
                    t_pitch = math.degrees(math.atan2(-dir_v[1], math.sqrt(dir_v[0]**2 + dir_v[2]**2)))
                    diff_yaw = (t_yaw - target_yaw + 180.0) % 360.0 - 180.0
                    target_yaw += diff_yaw * 0.14
                    target_pitch = 0.86 * target_pitch + 0.14 * t_pitch
        else:
            # 🎮 Fizik Tabanlı Klavye + Commando 8 Joystick Kontrolü
            keys = pygame.key.get_pressed()
            fast_mode = bool(keys[K_LSHIFT] or keys[K_RSHIFT])

            # --- Joystick Giriş Okuma (Commando 8) ---
            # Deadzone: analog stick sürüklemesini önler
            JOY_DEADZONE = 0.12
            # Joystick'ten gelen normalize edilmiş giriş değerleri (-1..+1)
            joy_fwd    = 0.0   # Sol Stick Y: ileri(+) / geri(-)
            joy_right  = 0.0   # Sol Stick X: sağ(+) / sol(-)
            joy_up     = 0.0   # R2 Tetik: yukarı | L2 Tetik: aşağı
            joy_yaw    = 0.0   # Sağ Stick X: sola(-) / sağa(+) kamera dönüşü
            joy_pitch  = 0.0   # Sağ Stick Y: aşağı(-) / yukarı(+) kamera eğimi
            joy_fast   = False # RB (Sağ Omuz): hızlı uçuş
            joy_throttle = False  # R2 tetik basılıysa True (yukarı gaz)

            if joystick:
                num_axes = joystick.get_numaxes()

                def joy_axis(idx, invert=False):
                    if idx >= num_axes:
                        return 0.0
                    v = joystick.get_axis(idx)
                    v = 0.0 if abs(v) < JOY_DEADZONE else v
                    return -v if invert else v

                # ── Hareket Eksenleri ──
                # Sol Stick Y → İleri/Geri (eksen genellikle ters: aşağı + değer = geri)
                raw_fwd   = joy_axis(1, invert=True)   # Axis 1: Sol Stick Y (ters)
                raw_right = joy_axis(0)                 # Axis 0: Sol Stick X

                # ── Tetikler (L2 / R2) ──
                # Birçok gamepad'de tetikler -1 (bırakılmış) → +1 (tam basılı) arası döner
                # Normalize: (raw + 1) / 2  → 0..1
                if num_axes >= 6:
                    r2_raw = joystick.get_axis(5)   # R2 → yukarı gaz
                    l2_raw = joystick.get_axis(4)   # L2 → aşağı / alçal
                    r2 = max(0.0, (r2_raw + 1.0) / 2.0)  # 0 (bırakılmış) → 1 (tam)
                    l2 = max(0.0, (l2_raw + 1.0) / 2.0)
                else:
                    # Bazı controllerlarda tetikler button olarak gelir
                    r2 = float(joystick.get_button(7) if joystick.get_numbuttons() > 7 else 0)
                    l2 = float(joystick.get_button(6) if joystick.get_numbuttons() > 6 else 0)

                joy_up = r2 - l2  # +1 = tam gaz yukarı, -1 = tam alçal

                # ── Kamera Eksenleri ──
                raw_yaw   = joy_axis(2)             # Sağ Stick X → kamera yaw
                raw_pitch = joy_axis(3, invert=True) # Sağ Stick Y → kamera pitch (ters)

                joy_fwd    = raw_fwd
                joy_right  = raw_right
                joy_yaw    = raw_yaw
                joy_pitch  = raw_pitch
                joy_throttle = r2 > 0.15   # R2 basılıysa yukarı gaz aktif

                # ── Omuz Tuşları ──
                joy_fast = bool(joystick.get_button(5)) if joystick.get_numbuttons() > 5 else False   # RB → hızlı

                # ── Sağ Stick ile Kamera Kontrolü ──
                cam_sensitivity = 90.0  # derece/saniye
                if abs(joy_yaw) > 0.0:
                    target_yaw   += joy_yaw   * cam_sensitivity * dt
                    user_override_look = True
                if abs(joy_pitch) > 0.0:
                    target_pitch += joy_pitch * cam_sensitivity * dt * 0.6
                    target_pitch  = max(-89.0, min(89.0, target_pitch))
                    user_override_look = True

                # ── Joystick Buton Aksiyonları (tek basış) ──
                # Bu event'ler JOYBUTTON'dan değil, polling'den — sadece basılı değil
                # anlık geçiş için bir önceki frame değeri tutmak gerekir;
                # burada basit "basılı tutma" mantığı kullanıyoruz:

                # Buton 4 (LB) → Optik Flow aç/kapat (tek basış değil, tutma OK)
                # Buton 3 (Y)  → Reset
                # Buton 2 (X)  → Tavan gizle/göster
                # Buton 1 (B)  → OctoMap (basılıysa tetikle — rate limit yap)

            # ── Joystick + Klavye Birleşimi ──
            # Fizik motoruna giriş: joystick OR klavye, hangisi daha büyükse kazan
            combined_fwd   = max(min(joy_fwd,   1.0), -1.0) != 0.0
            combined_back  = joy_fwd  < -JOY_DEADZONE
            combined_right = joy_right > JOY_DEADZONE
            combined_left  = joy_right < -JOY_DEADZONE

            move_fwd   = bool(keys[K_w] or keys[K_UP])   or (joy_fwd  > JOY_DEADZONE)
            move_back  = bool(keys[K_s] or keys[K_DOWN])  or (joy_fwd  < -JOY_DEADZONE)
            move_left  = bool(keys[K_a] or keys[K_LEFT])  or (joy_right < -JOY_DEADZONE)
            move_right = bool(keys[K_d] or keys[K_RIGHT]) or (joy_right > JOY_DEADZONE)
            throttle_up = bool(keys[K_SPACE]) or joy_throttle
            fast_mode   = fast_mode or joy_fast

            # Joystick analog → fizik kuvvetini ölçekle (tam basış = tam kuvvet)
            # Bu için step() içindeki fwd_x/z vektörlerini joy büyüklüğüyle çarp
            joy_fwd_mag   = abs(joy_fwd)   if abs(joy_fwd)   > JOY_DEADZONE else 1.0
            joy_right_mag = abs(joy_right) if abs(joy_right) > JOY_DEADZONE else 1.0
            effective_fwd_x = fwd_x * (joy_fwd_mag if (move_fwd or move_back) else 1.0)
            effective_fwd_z = fwd_z * (joy_fwd_mag if (move_fwd or move_back) else 1.0)
            effective_right_x = right_x * (joy_right_mag if (move_left or move_right) else 1.0)
            effective_right_z = right_z * (joy_right_mag if (move_left or move_right) else 1.0)

            # İleri/Geri/Sağ/Sol/Yukarı/Aşağı girişlerini fizik motoruna ilet
            drone_x, drone_y, drone_z = physics.step(
                dt=dt,
                throttle_up=throttle_up,
                move_fwd=move_fwd,
                move_back=move_back,
                move_left=move_left,
                move_right=move_right,
                fwd_x=effective_fwd_x, fwd_y=fwd_y, fwd_z=effective_fwd_z,
                right_x=effective_right_x, right_z=effective_right_z,
                fast_mode=fast_mode
            )

            # 💥 Çarpışma bildirimi ve kırmızı flaş
            if physics.is_crashed:
                hud.set_toast("💥 ÇARPIŞMA! Duvarla temas — düşüyor!", 1.8)
                hud.dirty = True


        # 🌊 Optik Flow HUD güncelle
        optical_flow.update(physics.vx, physics.vy, physics.vz, dt)

        # 🚁 Dron Modelinin Konum ve Yönünü Güncelle
        drone_model.x, drone_model.y, drone_model.z = drone_x, drone_y, drone_z
        drone_model.yaw = cam_yaw
        drone_model.update(
            dt=dt,
            vx=physics.vx,
            vy=physics.vy,
            vz=physics.vz,
            throttle=physics.throttle,
            is_airborne=not physics.is_landed
        )

        rx, ry, rz, ryaw, rpitch, rroll = drone_model.get_render_pose()

        # 🎥 Kamera Görünüm Moduna Göre Kamera Konumunu Ayarla
        if drone_view_mode == 0:
            # 3. Şahıs Takip Kamerası (Dronun Arkasından) + Mikro Kamera Sarsıntısı
            cam_dist = 1.40
            cam_h = 0.42
            render_cam_x = rx - fwd_x * cam_dist + drone_model.cam_shake_x
            render_cam_y = ry + cam_h - fwd_y * cam_dist + drone_model.cam_shake_y
            render_cam_z = rz - fwd_z * cam_dist
            eff_cam_pitch = cam_pitch + drone_model.cam_shake_rot
        else:
            # 1. Şahıs Kokpit (FPV) veya Serbest
            render_cam_x = rx + drone_model.cam_shake_x
            render_cam_y = ry + drone_model.cam_shake_y
            render_cam_z = rz
            eff_cam_pitch = cam_pitch + drone_model.cam_shake_rot

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
        glVertexPointer(3, GL_FLOAT, 0, None)
        glEnableClientState(GL_VERTEX_ARRAY)
        glDrawArrays(GL_LINES, 0, grid_n)
        glDisableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        # Kat Planı Mimari Çerçevesi (Kuşbakışında Çizilir)
        if top_down_view:
            FloorplanEstimator.draw_blueprint_grid(room_bounds)

        # 🚁 3B Fotogerçekçi Dron Modeli Çizimi (Sadece 3. Şahıs Takip Modunda)
        if drone_view_mode == 0:
            drone_model.draw_3d(floor_y=room_bounds['min_y'])
        elif drone_view_mode == 1 and drone_model.spotlight:
            # FPV kokpit modundayken sadece fenerin ve lazerin ışığı sahneye vursun
            drone_model.draw_3d(floor_y=room_bounds['min_y'])

        # Yeşil Kamera Yörünge Çizgisi
        if vbo_traj is not None and traj_n > 0:
            glLineWidth(2.0)
            glColor4f(0.1, 0.95, 0.35, 0.6)
            glBindBuffer(GL_ARRAY_BUFFER, vbo_traj)
            glVertexPointer(3, GL_FLOAT, 0, None)
            glEnableClientState(GL_VERTEX_ARRAY)
            glDrawArrays(GL_LINE_STRIP, 0, traj_n)
            glDisableClientState(GL_VERTEX_ARRAY)
            glBindBuffer(GL_ARRAY_BUFFER, 0)

        # 📏 3B Lazer Cetvel Çizimi
        ruler.draw_3d()


        # 🧊 OctoMap İçi Dolu 3B Voksel Küpleri (Solid 3D Cubes) / 🔮 3DGS Nokta Render
        if octomap_engine.active and vbo_octo_xyz is not None and octo_line_count > 0:
            glBindBuffer(GL_ARRAY_BUFFER, vbo_octo_xyz)
            glVertexPointer(3, GL_FLOAT, 0, None)
            glEnableClientState(GL_VERTEX_ARRAY)

            glBindBuffer(GL_ARRAY_BUFFER, vbo_octo_rgba)
            glColorPointer(4, GL_FLOAT, 0, None)
            glEnableClientState(GL_COLOR_ARRAY)

            glDrawArrays(GL_TRIANGLES, 0, octo_line_count)

            glDisableClientState(GL_COLOR_ARRAY)
            glDisableClientState(GL_VERTEX_ARRAY)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
        else:
            # 🔮 Standart 3D Gaussian Splats Çizimi (VBO Üzerinden 144+ FPS)
            # Optik zoom ile splat boyutunu dinamik ölçekle (Büyütmede seyrekleşmeyi/boşlukları önler)
            zoom_scale = 60.0 / max(cam_fov, 15.0)
            effective_point_size = max(1.0, min(24.0, splat_point_size * zoom_scale))
            glPointSize(effective_point_size)

            glBindBuffer(GL_ARRAY_BUFFER, vbo_xyz)
            glVertexPointer(3, GL_FLOAT, 0, None)
            glEnableClientState(GL_VERTEX_ARRAY)

            glBindBuffer(GL_ARRAY_BUFFER, vbo_rgba)
            glColorPointer(4, GL_FLOAT, 0, None)
            glEnableClientState(GL_COLOR_ARRAY)

            glDrawArrays(GL_POINTS, 0, num_splats)

            glDisableClientState(GL_COLOR_ARRAY)
            glDisableClientState(GL_VERTEX_ARRAY)
            glBindBuffer(GL_ARRAY_BUFFER, 0)


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
                      ruler, recorder, room_bounds, top_down_view, culler)

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
