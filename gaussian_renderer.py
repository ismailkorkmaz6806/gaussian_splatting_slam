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

# Windows Optimus / Çift Ekran Kartlı Laptoplar için Harici NVIDIA RTX GPU'yu Zorlama
os.environ["SHIM_MCCOMPAT"] = "0x000000001"
os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

# Pro özellik modüllerinin içe aktarılması
from pro_features import (
    CeilingCuller, LaserRuler, VideoRecorder,
    FloorplanEstimator, export_to_standalone_html,
    ProximityHeatmapEngine, OctomapEngine, Drone3DModel
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
        self.show_panel = True      # [H] veya [TAB] ile kontrol paneli açık/kapalı durumu
        self.toast_msg = ""         # Üstte beliren anlık bildirim metni
        self.toast_timer = 0        # Bildirimin ekranda kalma süresi

    def set_toast(self, msg, duration=2.8):
        """Kullanıcı bir tuşa bastığında üstte şık bir bildirim kutusu gösterir."""
        self.toast_msg = msg
        self.toast_timer = time.time() + duration

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
                ("🕹️ HAREKET & UÇUŞ", ""),
                ("[W / A / S / D]", "İleri / Sol / Geri / Sağ Serbest Uçuş"),
                ("[SPACE / SHIFT]", "Yukarı Yüksel / Aşağı Alçal"),
                ("[Fare Sol Sürükle]", "360° Serbest Kamera Açısı"),
                ("", ""),
                ("🔍 BÜYÜTME & KÜÇÜLTME (ZOOM)", ""),
                ("[+] / [-]", "🔍 Kamera Yakınlaş (Büyüt) / Uzaklaş (Küçült)"),
                ("[Fare Tekerleği]", "🔍 Hızlı Kamera Büyüt / Küçült"),
                ("[0] (Sıfır)", "Standart Zoom Açısına Sıfırla (60°)"),
                ("[X] / [C]", "🔮 Splat Nokta Boyutunu Büyüt / Küçült"),
                ("", ""),
                ("🏠 MİMARİ & TAVAN KESME ARAÇLARI", ""),
                ("[G] Tuşu", "🏠 Tavanı Gizle / Aç (İç Mekanı Kuşbakışı Gör)"),
                ("[7] / [8]", "✂️ Tavan Kesme Yüksekliğini Alçalt / Yükselt"),
                ("[T] Tuşu", "🗺️ 90° Kuşbakışı Kat Planı ve m² Alan Hesabı"),
                ("[E] Tuşu", "📏 3B Lazer Cetvel (İki Nokta Arası Metre Ölçümü)"),
                ("[U] Tuşu", "⚠️ Tünel Darboğaz & Tehlike Isı Haritası (<1.1m)"),
                ("[O] Tuşu", "🧊 OctoMap 3B Voksel / Doluluk Izgara Modu"),
                ("[J] Tuşu", "📄 Tünel İnceleme & PDF/HTML Raporu Üret"),
                ("", ""),
                ("🚀 PRO ÇIKTI VE VİDEO", ""),
                ("[V] Tuşu", "🎬 60 FPS MP4 Video Kaydını Başlat / Durdur"),
                ("[K] Tuşu", "🌐 Bağımsız Web / HTML 3B Modelini Dışa Aktar"),
                ("[M] Tuşu", "🪞 Sağ / Sol Yönünü Aynala / Düzelt"),
                ("[F] Tuşu", "🚁 3. Şahıs / 1. Şahıs / Serbest Dron Kamerası"),
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

    def render_gl(self, win_w, win_h, fps, num_splats, cam_fov, is_flipped,
                  ruler, recorder, room_bounds, top_down_view, culler):
        """2B Arayüzü OpenGL 2D Texture olarak 3B sahnenin üzerine çizer."""
        surf = self.build_surface(win_w, win_h, fps, num_splats, cam_fov, is_flipped,
                                  ruler, recorder, room_bounds, top_down_view, culler)

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

        if hasattr(pygame.image, 'tobytes'):
            rgba_data = pygame.image.tobytes(surf, "RGBA", True)
        else:
            rgba_data = pygame.image.tostring(surf, "RGBA", True)

        # Lazy texture initialization (İlk render çağrısında OpenGL dokusu üretilir)
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

    # 🪞 Orijinal Çekim Yönü (Aynasız, gerçek kamera açısı)
    is_flipped = False
    num_splats = len(xyz)
    room_bounds = FloorplanEstimator.calculate_bounds(xyz)

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
    drone_view_mode = 0  # 0: 3. Şahıs Dron Takip, 1: 1. Şahıs FPV, 2: Serbest
    hud.set_toast("✨ 3B Dron Simülatörü Hazır! [W/A/S/D] Uç, [F] Kamera Modu, [P] Otonom Tur, [O] OctoMap", 5.0)


    # RGBA Renk Matrisini Hazırla (Renk + Opaklık)
    if opacity is not None:
        rgba = np.column_stack([rgb, opacity]).astype(np.float32)
    else:
        rgba = np.column_stack([rgb, np.ones(num_splats, dtype=np.float32)]).astype(np.float32)
    original_rgba = rgba.copy()

    rgba = np.ascontiguousarray(rgba, dtype=np.float32)

    # Pygame & OpenGL Penceresini Başlat
    pygame.init()
    win_w, win_h = 1280, 720
    pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
    pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 2)
    pygame.display.set_mode((win_w, win_h), DOUBLEBUF | OPENGL | RESIZABLE)

    gpu_vendor = glGetString(GL_VENDOR).decode(errors='replace')
    gpu_renderer = glGetString(GL_RENDERER).decode(errors='replace')
    print(f" 🚀 Aktif 3DGS GPU Donanımı: {gpu_renderer} ({gpu_vendor})")
    pygame.display.set_caption(f"3D Gaussian Splatting Ultimate [{gpu_renderer}]")

    # OpenGL Render Ayarları
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)
    glDisable(GL_CULL_FACE)
    glEnable(GL_POINT_SMOOTH)
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
        cam_yaw, cam_pitch = 0.0, 0.0
        target_yaw, target_pitch = 0.0, 0.0


    cam_yaw, cam_pitch = 0.0, 0.0
    target_yaw, target_pitch = 0.0, 0.0
    vel_x, vel_y, vel_z = 0.0, 0.0, 0.0
    flight_speed = 0.22

    cam_fov = 60.0              # Görüş alanı (FOV / Zoom)
    splat_point_size = 3.8      # Splat nokta büyüklüğü
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

    # =========================================================================
    # 🔄 ANA ETKİLEŞİM VE RENDER DÖNGÜSÜ
    # =========================================================================
    while running:
        dt = clock.tick(144) / 1000.0
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
                elif event.button == 4:  # Tekerlek Yukarı -> Zoom In (Büyüt)
                    cam_fov = max(15.0, cam_fov - 3.0)
                    hud.set_toast(f"🔍 Zoom In: {cam_fov:.0f}°", 1.2)
                elif event.button == 5:  # Tekerlek Aşağı -> Zoom Out (Küçült)
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
                    user_override_look = True  # Tur sırasında fareyle serbestçe istenen yöne bakma yetkisi
                    target_yaw += dx * 0.32
                    target_pitch += dy * 0.32
                    target_pitch = max(-89.0, min(89.0, target_pitch))

            elif event.type == KEYDOWN:
                if event.key in (K_h, K_TAB):
                    hud.show_panel = not hud.show_panel
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
                    # 🔍 Zoom Yakınlaş (Büyüt)
                    cam_fov = max(15.0, cam_fov - 4.0)
                    hud.set_toast(f"🔍 Yakınlaşıldı (Zoom In): {cam_fov:.0f}°", 1.5)
                elif event.key in (K_MINUS, K_KP_MINUS):
                    # 🔍 Zoom Uzaklaş (Küçült)
                    cam_fov = min(110.0, cam_fov + 4.0)
                    hud.set_toast(f"🔍 Uzaklaşıldı (Zoom Out): {cam_fov:.0f}°", 1.5)
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
                    cam_fov = 60.0
                    hud.set_toast("🔍 Zoom Sıfırlandı (60°)", 1.5)
                elif event.key == K_m:
                    # 🪞 Sağ/Sol Aynalama ([M] Tuşu)
                    is_flipped = not is_flipped
                    xyz[:, 0] = -xyz[:, 0]
                    glBindBuffer(GL_ARRAY_BUFFER, vbo_xyz)
                    glBufferData(GL_ARRAY_BUFFER, xyz.nbytes, xyz, GL_STATIC_DRAW)
                    glBindBuffer(GL_ARRAY_BUFFER, 0)
                    if len(auto_waypoints) > 0:
                        auto_waypoints[:, 0] = -auto_waypoints[:, 0]
                        traj_arr = auto_waypoints.astype(np.float32)
                        glBindBuffer(GL_ARRAY_BUFFER, vbo_traj)
                        glBufferData(GL_ARRAY_BUFFER, traj_arr.nbytes, traj_arr, GL_STATIC_DRAW)
                        glBindBuffer(GL_ARRAY_BUFFER, 0)
                    drone_x = -drone_x
                    room_bounds = FloorplanEstimator.calculate_bounds(xyz)
                    culler.init_from_bounds(xyz)
                    hud.set_toast(f"🪞 Sağ/Sol Yön: {'AYNALANDI (TERS)' if is_flipped else 'ORİJİNAL'}", 2.5)

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
                    vel_x, vel_y, vel_z = 0.0, 0.0, 0.0
                    cam_fov = 60.0
                    hud.set_toast("🔄 Kamera Başlangıç Rotasına Sıfırlandı", 2.0)
                elif event.key in (K_ESCAPE, K_q):
                    running = False

        # Kamera Yumuşak Açı Geçişi (Slerp / Damping)
        cam_yaw = 0.85 * cam_yaw + 0.15 * target_yaw
        cam_pitch = 0.85 * cam_pitch + 0.15 * target_pitch

        rad_yaw, rad_pitch = math.radians(cam_yaw), math.radians(cam_pitch)
        fwd_x = math.sin(rad_yaw) * math.cos(rad_pitch)
        fwd_y = -math.sin(rad_pitch)
        fwd_z = math.cos(rad_yaw) * math.cos(rad_pitch)
        right_x = math.cos(rad_yaw)
        right_z = -math.sin(rad_yaw)

        # Kamera Hareketi (Sinematik Tur veya W/A/S/D Klavye Uçuşu)
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

        # 🚁 Dron Modelinin Konum ve Yönünü Güncelle
        drone_model.x, drone_model.y, drone_model.z = drone_x, drone_y, drone_z
        drone_model.yaw = cam_yaw
        drone_model.pitch = cam_pitch
        drone_model.update(dt)

        # 🎥 Kamera Görünüm Moduna Göre Kamera Konumunu Ayarla
        if drone_view_mode == 0:
            # 3. Şahıs Takip Kamerası (Dronun Arkasından)
            cam_dist = 1.40
            cam_h = 0.42
            render_cam_x = drone_x - fwd_x * cam_dist
            render_cam_y = drone_y + cam_h - fwd_y * cam_dist
            render_cam_z = drone_z - fwd_z * cam_dist
        else:
            # 1. Şahıs Kokpit (FPV) veya Serbest
            render_cam_x = drone_x
            render_cam_y = drone_y
            render_cam_z = drone_z

        # OpenGL Ekranını Temizle
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Projeksiyon Matrisi (Perspektif / Zoom FOV)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(cam_fov, (win_w / max(win_h, 1)), 0.02, 300.0)

        # ModelView Matrisi (Kamera Dönüş ve Konum Dönüşümleri)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glRotatef(cam_pitch, 1, 0, 0)
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

        # 🚁 3B Fotogerçekçi Dron Modeli Çizimi (Pervaneler, Gövde, Fener & Lazer)
        if drone_view_mode != 1:  # FPV modunda kokpitteyken gövde kamerayı kapatmasın
            drone_model.draw_3d(floor_y=room_bounds['min_y'])
        elif drone_model.spotlight:
            # FPV modundayken de fenerin ve lazerin ışığı sahneye vursun
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
            glPointSize(splat_point_size)

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
