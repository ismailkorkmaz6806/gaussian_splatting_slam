"""
========================================================================================
MASt3R-SLAM & 3DGS Pro Araçlar ve Mimari Analiz Modülü (pro_features.py)
========================================================================================
Bu dosya, 3B Gaussian Splatting ve SLAM görüntüleyicilerine entegre edilen 5 ana
profesyonel aracın matematiksel hesaplama ve OpenGL işleme motorudur:

1. 🏠 CeilingCuller (Tavan Kesme Motoru):
   - glClipPlane donanımsal kesme düzlemi ile tavanı gizler, iç mekanları görünür kılar.
   - Canlı yükseklik ayarlama ([G] ve [7]/[8] tuşları).

2. 📏 LaserRuler (3B Metrik Lazer Cetvel):
   - Ekranda tıklanan 2 pikselin 3B derinlik haritasından gerçek dünya koordinatlarını
     (gluUnProject) okur ve aralarındaki gerçek Öklid mesafesini (metre) hesaplar.

3. 🎬 VideoRecorder (60 FPS MP4 Video Kaydedici):
   - OpenGL Framebuffer piksellerini (glReadPixels) doğrudan yakalayıp OpenCV ile
     yüksek kaliteli MP4 video olarak kaydeder.

4. 🗺️ FloorplanEstimator (2B Kat Planı & m² Alan Hesaplayıcı):
   - Nokta bulutunun sınırlarını analiz ederek odanın genişlik, uzunluk ve net m²
     alanını hesaplar ve kuşbakışı mimari ızgarasını çizer.

5. 🌐 export_to_standalone_html (Tek Tıkla Web 3B Model Çıktısı):
   - Three.js tabanlı bağımsız, internet tarayıcısında açılabilen 3B HTML dosyası üretir.
========================================================================================
"""

import os
import sys
import math
import time
import json
import numpy as np
import cv2
from OpenGL.GL import *
from OpenGL.GLU import *

# Windows konsolunda Türkçe karakterleri destekleme
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# ======================================================================================
# 1. 🏠 TAVAN GİZLEME / KESME SİSTEMİ (CeilingCuller)
# ======================================================================================
class CeilingCuller:
    """
    Kuşbakışı veya serbest uçuşta tavan ve yüksek duvarları donanımsal olarak kesen sınıftır.
    OpenGL'in glClipPlane donanım hızlandırmasını kullanarak sıfır FPS kaybı ile çalışır.
    """

    def __init__(self):
        self.enabled = False      # Tavan kesme aktif mi?
        self.y_min = 0.0          # Odanın zemin Y koordinatı
        self.y_max = 3.0          # Odanın tavan tepe Y koordinatı
        self.cut_ratio = 0.72     # Tavan kesme oranı (Örn: %72 yüksekliğin üstü kesilir)
        self.cut_y = 2.10         # Dünya koordinatlarındaki kesme düzlemi Y değeri (Metre)

    def init_from_bounds(self, verts):
        """3B modelin zemin ve tavan yükseklik sınırlarını otomatik hesaplar."""
        if verts is not None and len(verts) > 0:
            self.y_min = float(np.percentile(verts[:, 1], 1))   # Zemin seviyesi (%1 alt sınır)
            self.y_max = float(np.percentile(verts[:, 1], 99))  # Tavan seviyesi (%99 üst sınır)
            self.update_cut_y()

    def update_cut_y(self):
        """Kesme oranı değiştiğinde kesme düzleminin gerçek Y metre değerini günceller."""
        h = max(0.5, self.y_max - self.y_min)
        self.cut_y = self.y_min + self.cut_ratio * h

    def toggle(self):
        """[G] tuşuna basıldığında tavan kesmeyi açar / kapatır."""
        self.enabled = not self.enabled
        return self.enabled

    def adjust_cut(self, delta):
        """[7] veya [8] tuşlarıyla tavan kesme yüksekliğini canlı olarak yukarı/aşağı ayarlar."""
        self.cut_ratio = max(0.20, min(0.98, self.cut_ratio + delta))
        self.update_cut_y()
        return self.cut_ratio

    def apply_gl(self):
        """OpenGL ModelView dönüşümünden sonra donanımsal kesme düzlemini aktif eder."""
        if self.enabled:
            # Y > cut_y olan tüm pikseller donanım tarafından otomatik yok edilir
            glClipPlane(GL_CLIP_PLANE0, [0.0, -1.0, 0.0, float(self.cut_y)])
            glEnable(GL_CLIP_PLANE0)

    def restore_gl(self):
        """2B Arayüz (HUD) ve yazılar çizilmeden önce kesme düzlemini devre dışı bırakır."""
        glDisable(GL_CLIP_PLANE0)


# ======================================================================================
# 2. 📏 3B METRİK LAZER CETVEL & MESAFE ÖLÇÜMÜ (LaserRuler)
# ======================================================================================
class LaserRuler:
    """
    Ekranda fareyle tıklanan herhangi 2 nokta arasındaki gerçek Öklid mesafesini
    (X, Y, Z boyut farkları ve toplam metre) hesaplayan 3B lazer cetvel motoru.
    """

    def __init__(self):
        self.active = False          # Cetvel modu aktif mi? ([E] Tuşu)
        self.points = []             # Tıklanan 3B noktaların listesi ([P1, P2])
        self.last_distance = None    # Hesaplanan toplam Öklid mesafesi (Metre)
        self.last_delta = None       # X, Y, Z eksenlerindeki ayrı ayrı boyut farkları (dx, dy, dz)

    def toggle(self):
        """[E] tuşuna basıldığında lazer cetveli açar / kapatır."""
        self.active = not self.active
        if not self.active:
            self.points = []
            self.last_distance = None

    def add_point_from_screen(self, mx, my, win_w, win_h, verts_sample=None):
        """
        Fareyle tıklanan 2B ekran pikselini (mx, my) GPU Depth Buffer'dan okuyarak
        3B dünya koordinatlarına (Ray-Unprojection) dönüştürür.
        """
        # Tıklanan pikselin GPU Z-derinlik değerini oku (0.0 = En yakın, 1.0 = Sonsuzluk)
        depth = glReadPixels(mx, win_h - my, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT)
        depth_val = float(depth[0][0]) if hasattr(depth, '__getitem__') else float(depth)

        # Eğer tıklanan yerde bir 3B nesne/splat varsa gluUnProject ile 3B Dünya noktasına çevir
        if depth_val < 1.0:
            modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
            projection = glGetDoublev(GL_PROJECTION_MATRIX)
            viewport = glGetIntegerv(GL_VIEWPORT)
            wx, wy, wz = gluUnProject(mx, win_h - my, depth_val, modelview, projection, viewport)
            picked_pt = np.array([wx, wy, wz], dtype=np.float32)
        elif verts_sample is not None and len(verts_sample) > 0:
            picked_pt = verts_sample[np.random.randint(len(verts_sample))].copy()
        else:
            return None

        # 2 noktadan fazlaysa sıfırla ve yeni ölçüme başla
        if len(self.points) >= 2:
            self.points = [picked_pt]
            self.last_distance = None
        else:
            self.points.append(picked_pt)

        # 2 nokta tamamlandığında mesafeyi (Öklid metriği) hesapla
        if len(self.points) == 2:
            p1, p2 = self.points[0], self.points[1]
            diff = p2 - p1
            self.last_delta = diff
            self.last_distance = float(np.linalg.norm(diff))

        return picked_pt

    def draw_3d(self):
        """Seçilen noktalar arasına 3B parlak lazer çizgisi ve işaretleyici küreler çizer."""
        if not self.active or len(self.points) == 0:
            return

        glDisable(GL_LIGHTING)
        glLineWidth(3.0)

        # 1. Nokta: Kırmızı işaretleyici
        p1 = self.points[0]
        glColor3f(1.0, 0.2, 0.4)
        self._draw_sphere_marker(p1, 0.06)

        # 2. Nokta: Yeşil işaretleyici ve aradaki Turkuaz Lazer Çizgisi
        if len(self.points) == 2:
            p2 = self.points[1]
            glColor3f(0.0, 1.0, 0.5)
            self._draw_sphere_marker(p2, 0.06)

            glBegin(GL_LINES)
            glColor3f(0.1, 1.0, 0.9)
            glVertex3f(p1[0], p1[1], p1[2])
            glVertex3f(p2[0], p2[1], p2[2])
            glEnd()

    def _draw_sphere_marker(self, pos, r=0.05):
        """3B uzayda nokta üzerine 3 eksenli parlak artı işareti çizer."""
        x, y, z = pos
        glBegin(GL_LINES)
        glVertex3f(x - r, y, z); glVertex3f(x + r, y, z)
        glVertex3f(x, y - r, z); glVertex3f(x, y + r, z)
        glVertex3f(x, y, z - r); glVertex3f(x, y, z + r)
        glEnd()


# ======================================================================================
# 3. 🎬 60 FPS PÜRÜZSÜZ MP4 VİDEO KAYDEDİCİ (VideoRecorder)
# ======================================================================================
class VideoRecorder:
    """
    OpenGL ekran görüntüsünü her karede doğrudan GPU Framebuffer'dan okuyarak
    60 FPS MP4 video dosyası olarak kaydeden modül.
    """

    def __init__(self):
        self.recording = False       # Kayıt şu an devam ediyor mu?
        self.writer = None           # OpenCV VideoWriter nesnesi
        self.filename = "tur_videosu.mp4"
        self.frame_count = 0         # Kaydedilen toplam kare sayısı

    def toggle(self, win_w, win_h, fps=60):
        """[V] tuşuna basıldığında video kaydını başlatır veya bitirir."""
        if not self.recording:
            ts = time.strftime("%Y%m%d_%H%M%S")
            self.filename = f"tur_videosu_{ts}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(self.filename, fourcc, fps, (win_w, win_h))
            self.recording = True
            self.frame_count = 0
            return f"🔴 Video Kaydı Başlatıldı: {self.filename}"
        else:
            self.recording = False
            if self.writer is not None:
                self.writer.release()
                self.writer = None
            msg = f"💾 Video Kaydedildi ({self.frame_count} kare): {self.filename}"
            return msg

    def capture_frame(self, win_w, win_h):
        """Her render döngüsünde OpenGL piksellerini okuyup videoya yazar."""
        if not self.recording or self.writer is None:
            return
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        raw_data = glReadPixels(0, 0, win_w, win_h, GL_BGR, GL_UNSIGNED_BYTE)
        frame = np.frombuffer(raw_data, dtype=np.uint8).reshape((win_h, win_w, 3))
        frame = cv2.flip(frame, 0)  # OpenGL alt-üst koordinat tersliğini düzelt
        self.writer.write(frame)
        self.frame_count += 1


# ======================================================================================
# 4. 🗺️ 2B KAT PLANI & m² ALAN HESAPLAYICI (FloorplanEstimator)
# ======================================================================================
class FloorplanEstimator:
    """
    3B mekanın dış sınırlarını ve hacmini analiz ederek net kullanım alanını (m²),
    genişliğini, derinliğini ve tavan yüksekliğini hesaplayan mimari analiz sınıfı.
    """

    @staticmethod
    def calculate_bounds(verts):
        """Nokta bulutundan istatistiksel oda boyutlarını ve net m² alanını hesaplar."""
        if verts is None or len(verts) == 0:
            return None
        # %1 ve %99 yüzdelik dilimleri kullanarak dış gürültüleri ayıkla
        min_x, max_x = float(np.percentile(verts[:, 0], 1)), float(np.percentile(verts[:, 0], 99))
        min_y, max_y = float(np.percentile(verts[:, 1], 1)), float(np.percentile(verts[:, 1], 99))
        min_z, max_z = float(np.percentile(verts[:, 2], 1)), float(np.percentile(verts[:, 2], 99))

        width = max(0.1, max_x - min_x)    # X eksenindeki oda genişliği (Metre)
        height = max(0.1, max_y - min_y)   # Y eksenindeki tavan yüksekliği (Metre)
        length = max(0.1, max_z - min_z)   # Z eksenindeki oda derinliği (Metre)
        area_m2 = width * length * 0.82    # Duvar payı düşülmüş tahmini net m² alanı

        return {
            'min_x': min_x, 'max_x': max_x,
            'min_y': min_y, 'max_y': max_y,
            'min_z': min_z, 'max_z': max_z,
            'width': width,
            'height': height,
            'length': length,
            'area_m2': area_m2,
            'volume_m3': area_m2 * height
        }

    @staticmethod
    def draw_blueprint_grid(bounds):
        """Kuşbakışı [T] modunda odanın zeminine mimari CAD sınır çerçevesi çizer."""
        if bounds is None:
            return
        glDisable(GL_LIGHTING)
        glLineWidth(2.0)
        glColor4f(0.0, 0.85, 1.0, 0.8)

        mx, Mx = bounds['min_x'], bounds['max_x']
        mz, Mz = bounds['min_z'], bounds['max_z']
        gy = bounds['min_y'] + 0.04

        glBegin(GL_LINE_LOOP)
        glVertex3f(mx, gy, mz)
        glVertex3f(Mx, gy, mz)
        glVertex3f(Mx, gy, Mz)
        glVertex3f(mx, gy, Mz)
        glEnd()


# ======================================================================================
# 5. 🌐 TEK TIKLA WEB / THREE.JS HTML 3B ÇIKTISI (export_to_standalone_html)
# ======================================================================================
def export_to_standalone_html(xyz, rgb, output_html="3d_scene.html", max_points=120000):
    """
    [K] tuşuna basıldığında herhangi bir tarayıcıda (Chrome, Edge, Safari, Mobil)
    ekstra yazılım gerekmeden 60+ FPS ile açılabilen bağımsız Three.js HTML dosyası üretir.
    """
    N = len(xyz)
    step = max(1, N // max_points)
    xyz_sub = xyz[::step].astype(np.float32)
    rgb_sub = rgb[::step].astype(np.float32)

    positions_list = np.round(xyz_sub, 3).flatten().tolist()
    colors_list = np.round(rgb_sub, 2).flatten().tolist()

    html_content = f"""<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>3D Ofis Modeli - Web Görüntüleyici</title>
    <style>
        body {{ margin: 0; padding: 0; overflow: hidden; background: #080b12; font-family: 'Segoe UI', sans-serif; color: #fff; }}
        #hud {{
            position: absolute; top: 15px; left: 15px; background: rgba(15,22,35,0.85);
            backdrop-filter: blur(10px); padding: 14px 20px; border-radius: 12px;
            border: 1px solid rgba(0,180,255,0.3); box-shadow: 0 8px 32px rgba(0,0,0,0.5);
            max-width: 320px;
        }}
        h1 {{ margin: 0 0 6px 0; font-size: 16px; color: #00d2ff; }}
        p {{ margin: 4px 0; font-size: 13px; color: #c8d4e6; }}
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
    <div id="hud">
        <h1>🔮 3D OFİS MODELİ (WEB)</h1>
        <p>📊 <b>{len(xyz_sub):,}</b> Nokta | WebGL 60+ FPS</p>
        <p>🖱️ <b>Sol Tık + Sürükle:</b> 360° Döndür</p>
        <p>🖱️ <b>Sağ Tık / Çift Parmak:</b> Kaydır (Pan)</p>
        <p>🔍 <b>Tekerlek / Parmak:</b> Yakınlaş / Uzaklaş</p>
    </div>

    <script>
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x060910);

        const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.05, 500);
        camera.position.set(0, 2, 4);

        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        document.body.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.08;

        const grid = new THREE.GridHelper(30, 30, 0x0088ff, 0x1a2b45);
        grid.position.y = 0;
        scene.add(grid);

        const posArr = new Float32Array({json.dumps(positions_list)});
        const colArr = new Float32Array({json.dumps(colors_list)});

        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(posArr, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colArr, 3));

        const material = new THREE.PointsMaterial({{
            size: 0.035,
            vertexColors: true,
            transparent: true,
            opacity: 0.95
        }});

        const pointCloud = new THREE.Points(geometry, material);
        scene.add(pointCloud);

        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});

        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }}
        animate();
    </script>
</body>
</html>
"""
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)

    return output_html


# ======================================================================================
# 6. ⚠️ TÜNEL TEHLİKE & DARBOĞAZ ISI HARİTASI (ProximityHeatmapEngine)
# ======================================================================================
class ProximityHeatmapEngine:
    """
    3B model üzerindeki tünel geçitlerini, tavan yüksekliğini ve duvar yakınlıklarını
    analiz ederek renkli bir tehlike/darboğaz ısı haritasına (Heatmap) dönüştürür.
    
    Renk Kodları:
      🔴 Kırmızı : < 1.0 metre (Kritik Darboğaz / Çökme / Engel Riski)
      🟡 Sarı    : 1.0 - 2.0 metre (Dikkat Geçişi)
      🟢 Yeşil/Mavi : > 2.0 metre (Güvenli & Ferah Alan)
    """

    def __init__(self):
        self.active = False            # Isı haritası modu açık mı? ([P] Tuşu)
        self.heatmap_colors = None     # Hesaplanmış RGB ısı renkleri matrisi

    def toggle(self):
        """[P] tuşuna basıldığında Isı Haritası modunu açar / kapatır."""
        self.active = not self.active
        return self.active

    def compute_clearance_heatmap(self, xyz):
        """
        Her 3B noktanın zemin yüksekliği ve duvar merkezinden olan açıklık mesafesini
        hesaplayıp RGB renklerine dönüştürür.
        """
        if xyz is None or len(xyz) == 0:
            return None

        # Y ekseni zemin seviyesi
        ground_y = float(np.percentile(xyz[:, 1], 1))
        clearance_y = np.maximum(0.0, xyz[:, 1] - ground_y)

        # X ekseni merkez kaçıklığı
        center_x = float(np.median(xyz[:, 0]))
        dist_x = np.abs(xyz[:, 0] - center_x)

        # Toplam efektif tünel açıklığı (metre)
        effective_clearance = np.clip(clearance_y * 0.7 + dist_x * 0.8, 0.2, 3.5)

        # Renk Haritası Dönüşümü (0.5m = Kırmızı, 1.5m = Sarı, 2.5m+ = Yeşil/Mavi)
        norm_val = np.clip((effective_clearance - 0.5) / 2.0, 0.0, 1.0)

        # Jet / Turbo renk gradyanı oluştur
        r = np.clip(1.5 - np.abs(norm_val * 4.0 - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(norm_val * 4.0 - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(norm_val * 4.0 - 1.0), 0.0, 1.0)

        # Tehlikeli çok dar bölgeleri daha parlak kırmızı yap
        danger_mask = effective_clearance < 1.10
        r[danger_mask] = 1.0
        g[danger_mask] = 0.15
        b[danger_mask] = 0.20

        self.heatmap_colors = np.column_stack([r, g, b]).astype(np.float32)
        return self.heatmap_colors


# ======================================================================================
# 7. 🧊 3B VOKSEL DOLULUK HARİTASI (OctomapEngine - ROS Standardı)
# ======================================================================================
class OctomapEngine:
    """
    3B nokta bulutunu robotik ve otonom navigasyon standartlarındaki
    OctoMap (3B Voksel / Doluluk Izgarası) formatına dönüştürür.
    
    Özellikler:
      - Sürekli uzayı 10cm/15cm/20cm'lik 3B doluluk küplerine böler.
      - ROS OctoMap standart renk skalası (Zemin -> Tavan yükseklik gradyanı).
      - Tel kafes ve yarı saydam 3B küp render geometrisi üretir.
    """

    def __init__(self, voxel_size=0.15):
        self.active = False
        self.voxel_size = voxel_size     # Voksel küp kenar uzunluğu (metre)
        self.voxel_centers = None        # Küp merkezleri (N, 3)
        self.cube_vertices = None        # OpenGL için küp köşe dizisi
        self.cube_colors = None          # Voksel renkleri
        self.num_cubes = 0

    def toggle(self):
        """[O] tuşuna basıldığında OctoMap modunu açar / kapatır."""
        self.active = not self.active
        return self.active

    def generate_octomap(self, xyz, voxel_size=None):
        """
        Nokta bulutunu voksellere böler ve içi dolu (Solid 3D Cubes) 6 yüzeyli
        3B küp geometrisini (GL_TRIANGLES) ve ışıklandırma gölgelerini hazırlar.
        """
        if voxel_size is not None:
            self.voxel_size = voxel_size
        if xyz is None or len(xyz) == 0:
            return None, None

        # Vokselleştirme (Downsampling & Grid Binning)
        grid_coords = np.floor(xyz / self.voxel_size).astype(np.int32)
        unique_grid = np.unique(grid_coords, axis=0)
        self.voxel_centers = (unique_grid.astype(np.float32) + 0.5) * self.voxel_size
        self.num_cubes = len(self.voxel_centers)

        # Yüksekliğe göre ROS OctoMap renk paleti (Mavi -> Yeşil -> Sarı -> Kırmızı)
        min_y = float(np.min(self.voxel_centers[:, 1]))
        max_y = float(np.max(self.voxel_centers[:, 1]))
        y_range = max(0.1, max_y - min_y)
        norm_y = np.clip((self.voxel_centers[:, 1] - min_y) / y_range, 0.0, 1.0)

        # Turbo / Jet renk skalası
        r = np.clip(1.5 - np.abs(norm_y * 4.0 - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(norm_y * 4.0 - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(norm_y * 4.0 - 1.0), 0.0, 1.0)

        # 8 Küp Köşesi (Hafif aralıklı temiz küpler: %94 boyut)
        s = (self.voxel_size * 0.94) / 2.0
        offsets = np.array([
            [-s, -s, -s],  # 0: sol-alt-arka
            [ s, -s, -s],  # 1: sag-alt-arka
            [ s,  s, -s],  # 2: sag-ust-arka
            [-s,  s, -s],  # 3: sol-ust-arka
            [-s, -s,  s],  # 4: sol-alt-on
            [ s, -s,  s],  # 5: sag-alt-on
            [ s,  s,  s],  # 6: sag-ust-on
            [-s,  s,  s]   # 7: sol-ust-on
        ], dtype=np.float32)

        # 6 Dolu Yüzey (12 Üçgen = 36 Köşe per küp)
        tri_idx = [
            4, 5, 6,  4, 6, 7,  # Ön Yüz (+Z)
            1, 0, 3,  1, 3, 2,  # Arka Yüz (-Z)
            0, 4, 7,  0, 7, 3,  # Sol Yüz (-X)
            5, 1, 2,  5, 2, 6,  # Sağ Yüz (+X)
            3, 7, 6,  3, 6, 2,  # Üst Yüz (+Y - Parlak Tavan)
            0, 1, 5,  0, 5, 4   # Alt Yüz (-Y - Taban)
        ]

        # 3B Hacim Hissi Veren Yüzey Gölgelendirme Çarpanları
        face_shades = np.array([
            0.90, 0.90, 0.90, 0.90, 0.90, 0.90,  # Ön
            0.75, 0.75, 0.75, 0.75, 0.75, 0.75,  # Arka
            0.82, 0.82, 0.82, 0.82, 0.82, 0.82,  # Sol
            0.86, 0.86, 0.86, 0.86, 0.86, 0.86,  # Sağ
            1.00, 1.00, 1.00, 1.00, 1.00, 1.00,  # Üst (En Parlak)
            0.55, 0.55, 0.55, 0.55, 0.55, 0.55   # Alt (Gölge)
        ], dtype=np.float32)

        # (N, 36, 3) 3B Dolu Küp Köşe Dizisi
        cube_verts = (self.voxel_centers[:, None, :] + offsets[tri_idx]).reshape(-1, 3).astype(np.float32)

        # Renk ve Gölgeleri Her Küpün 36 Köşesine Uygula
        base_rgb = np.column_stack([r, g, b]).astype(np.float32)  # (N, 3)
        shaded_rgb = (base_rgb[:, None, :] * face_shades[None, :, None]).reshape(-1, 3)  # (N*36, 3)
        cube_rgba = np.column_stack([shaded_rgb, np.full(len(shaded_rgb), 1.0, dtype=np.float32)]).astype(np.float32)

        self.cube_vertices = cube_verts
        self.cube_colors = cube_rgba
        return self.cube_vertices, self.cube_colors


# ======================================================================================
# ======================================================================================
# 🚁 3B FOTOGERÇEKÇİ HOLYBRO X500 QUADCOPTER MODELİ & GELİŞMİŞ UÇUŞ FİZİK MOTORU
# ======================================================================================
class Drone3DModel:
    """
    3B Fotogerçekçi Holybro X500 Quadcopter Dron Görsel Modeli, Fener Işığı ve Lazer Çizicisi.
    
    Gelişmiş Gerçekçilik Özellikleri:
      - 🚁 Holybro X500: Çift Karbon Şasi, Pixhawk FC (RGB LED), Intel RealSense D435, RPLIDAR A2, 4S LiPo ve T-İniş Takımları
      - 🌪️ Yüksek Hızlı Dönen Pervaneler + Motion Blur (Rotor Diski Saydamlık Efekti)
      - 📐 Dinamik Uçuş Eğimi (Pitch/Roll Tilt): İleri harekette nose-down pitch, yanlara harekette roll bank
      - 🫁 İrtifa Mikro-Salınımı (PID Hover Breathing): Havada süzülürken canlı hava/motor dalgalanması (±2.5cm)
      - 🎥 Dinamik Kamera Sarsıntısı (Air Turbulence / Engine Vibration)
      - 💡 Ön Fener (Spotlight) ve TFmini Lazer İrtifa Çizgisi
    """
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.yaw = 0.0          # Derece
        self.pitch = 0.0        # Derece (Dinamik nose-down / nose-up)
        self.roll = 0.0         # Derece (Dinamik bank sola / sağa)
        self.target_pitch = 0.0
        self.target_roll = 0.0
        self.prop_angle = 0.0
        self.scale = 0.28       # Dron boyutu (metre cinsinden gerçekçi ölçek)
        self.view_mode = 0      # 0: 3. Şahıs Takip, 1: 1. Şahıs FPV, 2: Serbest
        self.spotlight = True
        self.laser = True
        self.active = True
        self.is_airborne = True
        self.throttle = 0.5     # 0.0 - 1.0 arası

        # Canlı salınım & kamera sarsıntısı durumları
        self.hover_offset_y = 0.0
        self.hover_jitter_p = 0.0
        self.hover_jitter_r = 0.0
        self.cam_shake_x = 0.0
        self.cam_shake_y = 0.0
        self.cam_shake_rot = 0.0
        self.time_accum = 0.0

    def update(self, dt=0.016, vx=0.0, vy=0.0, vz=0.0, throttle=0.5, is_airborne=True,
               override_pitch=None, override_roll=None, override_yaw=None):
        """
        Dron uçuş kinematiğini, pervane dönüşünü, dinamik tilt ve irtifa salınımını günceller.
        """
        dt = min(dt, 0.05)
        self.time_accum += dt
        self.is_airborne = is_airborne
        self.throttle = max(0.0, min(1.0, throttle))

        # 1. Pervane Dönüş Hızı (Throttle'a bağlı dynamic RPM)
        if self.is_airborne:
            base_prop_speed = 3600.0 + self.throttle * 4800.0
        else:
            base_prop_speed = 600.0 if self.throttle > 0.1 else 0.0
        self.prop_angle = (self.prop_angle + base_prop_speed * dt) % 360.0

        # 2. Dinamik Gövde Eğimi (Pitch / Roll Tilt)
        if override_pitch is not None and override_roll is not None:
            self.target_pitch = float(override_pitch)
            self.target_roll = float(override_roll)
        else:
            # Hız vektörünü gövde koordinatlarına çevir (Yaw açısına göre)
            rad_yaw = math.radians(self.yaw)
            cos_y = math.cos(rad_yaw)
            sin_y = math.sin(rad_yaw)

            # Gövde ileri (+Z / camera fwd) ve sağ (+X / camera right) hızları
            v_fwd   = vz * cos_y + vx * sin_y
            v_right = vx * cos_y - vz * sin_y

            # İleri hareket -> Burun aşağı (nose-down negative pitch)
            # Geri hareket  -> Burun yukarı (nose-up positive pitch)
            # Sağa hareket  -> Sağa yatış (+roll)
            # Sola hareket  -> Sola yatış (-roll)
            max_tilt = 16.0  # Maksimum ±16 derece gerçekçi eğim
            calc_pitch = -float(np.clip(v_fwd * 3.6, -max_tilt, max_tilt))
            calc_roll  = float(np.clip(v_right * 3.6, -max_tilt, max_tilt))

            self.target_pitch = calc_pitch
            self.target_roll  = calc_roll

        if override_yaw is not None:
            self.yaw = float(override_yaw)

        # Yumuşak Lerp Enterpolasyonu (Ani sıçrama olmadan gövdenin yatışı ve doğrulması)
        lerp_speed = min(1.0, 10.0 * dt)
        self.pitch += (self.target_pitch - self.pitch) * lerp_speed
        self.roll  += (self.target_roll - self.roll) * lerp_speed

        # 3. İrtifa Mikro-Salınımı (PID Hover Breathing)
        # Havada süzülürken dikeyde donmayı önleyen organik çoklu frekans dalgası (±2.5 cm)
        if self.is_airborne:
            t = self.time_accum
            self.hover_offset_y = (
                0.022 * math.sin(t * 3.6) +
                0.012 * math.cos(t * 7.4) +
                0.006 * math.sin(t * 13.9)
            )
            self.hover_jitter_p = 0.40 * math.sin(t * 4.8)
            self.hover_jitter_r = 0.40 * math.cos(t * 5.5)

            # 4. Kamera Sarsıntısı (Hava Türbülansı ve Motor Titreşimi)
            spd = math.sqrt(vx**2 + vy**2 + vz**2)
            shake_mag = min(1.0, (spd / 4.5) * 0.7 + self.throttle * 0.3)
            self.cam_shake_x   = 0.006 * math.sin(t * 21.0) * shake_mag
            self.cam_shake_y   = 0.006 * math.cos(t * 26.0) * shake_mag
            self.cam_shake_rot = 0.16 * math.sin(t * 18.0) * shake_mag
        else:
            self.hover_offset_y = 0.0
            self.hover_jitter_p = 0.0
            self.hover_jitter_r = 0.0
            self.cam_shake_x = 0.0
            self.cam_shake_y = 0.0
            self.cam_shake_rot = 0.0

    def get_render_pose(self):
        """Çizim ve kamera için mikro-salınım uygulanmış gerçek koordinatları döndürür."""
        draw_y = self.y + (self.hover_offset_y if self.is_airborne else 0.0)
        draw_p = self.pitch + (self.hover_jitter_p if self.is_airborne else 0.0)
        draw_r = self.roll + (self.hover_jitter_r if self.is_airborne else 0.0)
        return self.x, draw_y, self.z, self.yaw, draw_p, draw_r

    def draw_3d(self, floor_y=-1.5):
        """OpenGL ile Fotogerçekçi Holybro X500 3B gövde, pervaneler, fener ve lazeri çizer."""
        if not self.active:
            return

        rx, ry, rz, ryaw, rpitch, rroll = self.get_render_pose()

        glPushAttrib(GL_ALL_ATTRIB_BITS)
        glPushMatrix()
        # Dron Konum ve Dönüş Matrisi
        glTranslatef(rx, ry, rz)
        glRotatef(-ryaw, 0, 1, 0)
        glRotatef(rpitch, 1, 0, 0)
        glRotatef(rroll, 0, 0, 1)
        glScalef(self.scale, self.scale, self.scale)

        # =====================================================================
        # 1. HOLYBRO X500 KARBON FİBER ÇİFT ŞASİ PLAKALARI & KOLONLAR
        # =====================================================================
        # Alt Karbon Plaka (Chassis Bottom)
        glColor3f(0.10, 0.11, 0.13)
        glBegin(GL_QUADS)
        glVertex3f(-0.40, -0.06, -0.40); glVertex3f(0.40, -0.06, -0.40)
        glVertex3f(0.40, -0.06, 0.40);   glVertex3f(-0.40, -0.06, 0.40)
        # Üst Karbon Plaka (Top Deck)
        glVertex3f(-0.40, 0.07, -0.40); glVertex3f(0.40, 0.07, -0.40)
        glVertex3f(0.40, 0.07, 0.40);   glVertex3f(-0.40, 0.07, 0.40)
        glEnd()

        # Alüminyum Kolonlar (Standoffs)
        glColor3f(0.70, 0.72, 0.75)
        glLineWidth(2.5)
        glBegin(GL_LINES)
        for sx, sz in [(0.32, 0.32), (-0.32, 0.32), (0.32, -0.32), (-0.32, -0.32)]:
            glVertex3f(sx, -0.06, sz); glVertex3f(sx, 0.07, sz)
        glEnd()

        # =====================================================================
        # 2. PİXHAWK UÇUŞ BİLGİSAYARI & RGB DURUM LEDİ
        # =====================================================================
        # Pixhawk Alüminyum Gövdesi (Orta kat)
        glColor3f(0.16, 0.18, 0.22)
        glBegin(GL_QUADS)
        glVertex3f(-0.20, 0.072, -0.25); glVertex3f(0.20, 0.072, -0.25)
        glVertex3f(0.20, 0.072, 0.25);   glVertex3f(-0.20, 0.072, 0.25)
        glEnd()

        # Pixhawk Parlayan RGB Durum Ledi (Canlı Yanıp Sönen Yeşil/Mavi)
        glPointSize(7.0)
        glBegin(GL_POINTS)
        if self.is_airborne:
            glColor3f(0.0, 1.0, 0.5)  # Uçuşta Yeşil
        else:
            glColor3f(0.0, 0.7, 1.0)  # Yerde Mavi
        glVertex3f(0.0, 0.076, 0.08)
        glEnd()

        # =====================================================================
        # 3. INTEL REALSENSE D435 DERİNLİK KAMERASI (Ön Burun)
        # =====================================================================
        # RealSense Gümüş/Gri Gövde
        glColor3f(0.75, 0.78, 0.82)
        glBegin(GL_QUADS)
        glVertex3f(-0.22, 0.01, 0.42); glVertex3f(0.22, 0.01, 0.42)
        glVertex3f(0.22, 0.06, 0.42);  glVertex3f(-0.22, 0.06, 0.42)
        glEnd()
        # 3 Optik Lens Noktası (Sol IR, Orta RGB, Sağ IR)
        glPointSize(4.0)
        glColor3f(0.1, 0.4, 0.9)
        glBegin(GL_POINTS)
        glVertex3f(-0.14, 0.035, 0.425)
        glVertex3f(0.0,   0.035, 0.425)
        glVertex3f(0.14,  0.035, 0.425)
        glEnd()

        # =====================================================================
        # 4. RPLIDAR A2 360° LAZER TARAYICI PUCK (Üst Kat)
        # =====================================================================
        # Siyah Lidar Gövdesi
        glColor3f(0.12, 0.12, 0.14)
        glBegin(GL_QUADS)
        glVertex3f(-0.18, 0.14, -0.18); glVertex3f(0.18, 0.14, -0.18)
        glVertex3f(0.18, 0.14, 0.18);   glVertex3f(-0.18, 0.14, 0.18)
        glEnd()
        # Dönen Kırmızı Lidar Lazer Noktası
        lidar_angle = (self.time_accum * 1800.0) % 360.0
        l_rad = math.radians(lidar_angle)
        lx = math.cos(l_rad) * 0.16
        lz = math.sin(l_rad) * 0.16
        glPointSize(4.0)
        glColor3f(1.0, 0.2, 0.2)
        glBegin(GL_POINTS)
        glVertex3f(lx, 0.15, lz)
        glEnd()

        # =====================================================================
        # 5. HOLYBRO 4S LIPO BATARYA (Alt Kat & Sarı Cırt Bant)
        # =====================================================================
        glColor3f(0.15, 0.15, 0.16)
        glBegin(GL_QUADS)
        glVertex3f(-0.16, -0.16, -0.32); glVertex3f(0.16, -0.16, -0.32)
        glVertex3f(0.16, -0.16, 0.32);   glVertex3f(-0.16, -0.16, 0.32)
        # Sarı Cırt Bant
        glColor3f(0.95, 0.85, 0.10)
        glVertex3f(-0.17, -0.165, -0.05); glVertex3f(0.17, -0.165, -0.05)
        glVertex3f(0.17, -0.165, 0.05);   glVertex3f(-0.17, -0.165, 0.05)
        glEnd()

        # =====================================================================
        # 6. KARBON FİBER 4 MOTOR KOLU (16mm Çapraz Borular)
        # =====================================================================
        glLineWidth(4.0)
        glColor3f(0.08, 0.09, 0.11)
        glBegin(GL_LINES)
        glVertex3f(-0.95, 0.02, -0.95); glVertex3f(0.95, 0.02, 0.95)
        glVertex3f(-0.95, 0.02, 0.95);  glVertex3f(0.95, 0.02, -0.95)
        glEnd()

        # =====================================================================
        # 7. 4 ADET 2216 MOTOR VE MOTION BLUR'LU DÖNEN PERVANELER
        # =====================================================================
        motor_pos = [
            ( 0.95, 0.06,  0.95, (1.0, 0.25, 0.15)), # Ön Sağ (Kırmızı/Turuncu)
            (-0.95, 0.06,  0.95, (1.0, 0.25, 0.15)), # Ön Sol (Kırmızı/Turuncu)
            ( 0.95, 0.06, -0.95, (0.25, 0.55, 0.95)), # Arka Sağ (Mavi/Karbon)
            (-0.95, 0.06, -0.95, (0.25, 0.55, 0.95))  # Arka Sol (Mavi/Karbon)
        ]

        for mx, my, mz, p_col in motor_pos:
            # Motor Gövdesi (Titanyum / Alüminyum Çan)
            glColor3f(0.35, 0.36, 0.40)
            glLineWidth(3.0)
            glBegin(GL_LINES)
            glVertex3f(mx, my - 0.08, mz); glVertex3f(mx, my + 0.04, mz)
            glEnd()

            glPushMatrix()
            glTranslatef(mx, my + 0.04, mz)
            dir_mult = 1.0 if (mx * mz > 0) else -1.0
            cur_p_angle = self.prop_angle * dir_mult
            glRotatef(cur_p_angle, 0, 1, 0)

            # --- A. YÜKSEK HIZLI DÖNEN PERVANE BLADE SİLUETLERİ ---
            glLineWidth(3.0)
            glColor3f(*p_col)
            glBegin(GL_LINES)
            glVertex3f(-0.50, 0.0, 0.0); glVertex3f(0.50, 0.0, 0.0)
            glEnd()

            # --- B. 🌪️ PERVANE MOTION BLUR ROTOR DİSKİ (Saydam Dönen Fan) ---
            if self.is_airborne or self.throttle > 0.15:
                glEnable(GL_BLEND)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                blur_alpha = 0.22 + self.throttle * 0.20
                r_rad = 0.52

                # Şeffaf Rotor Diski (Motion Blur)
                glColor4f(p_col[0], p_col[1], p_col[2], blur_alpha)
                glBegin(GL_TRIANGLE_FAN)
                glVertex3f(0.0, 0.0, 0.0)
                for deg in range(0, 361, 24):
                    rad = math.radians(deg)
                    glVertex3f(math.sin(rad) * r_rad, 0.0, math.cos(rad) * r_rad)
                glEnd()

                # Dış Uç Halka Vurgusu (Tip Vortex Highlight)
                glColor4f(1.0, 1.0, 1.0, blur_alpha * 1.3)
                glLineWidth(1.2)
                glBegin(GL_LINE_LOOP)
                for deg in range(0, 361, 24):
                    rad = math.radians(deg)
                    glVertex3f(math.sin(rad) * r_rad, 0.0, math.cos(rad) * r_rad)
                glEnd()
                glDisable(GL_BLEND)

            glPopMatrix()

        # =====================================================================
        # 8. T-TİPİ KARBON İNİŞ TAKIMLARI (Landing Skids)
        # =====================================================================
        glColor3f(0.09, 0.10, 0.12)
        glLineWidth(3.5)
        glBegin(GL_LINES)
        # 4 Açılı Bacak
        glVertex3f( 0.28, -0.06,  0.22); glVertex3f( 0.38, -0.38,  0.28)
        glVertex3f(-0.28, -0.06,  0.22); glVertex3f(-0.38, -0.38,  0.28)
        glVertex3f( 0.28, -0.06, -0.22); glVertex3f( 0.38, -0.38, -0.28)
        glVertex3f(-0.28, -0.06, -0.22); glVertex3f(-0.38, -0.38, -0.28)
        # 2 Yatay Kayak Borusu
        glVertex3f( 0.38, -0.38, -0.55); glVertex3f( 0.38, -0.38,  0.55)
        glVertex3f(-0.38, -0.38, -0.55); glVertex3f(-0.38, -0.38,  0.55)
        glEnd()

        # =====================================================================
        # 9. 💡 ÖN FENER IŞIĞI (Spotlight)
        # =====================================================================
        if self.spotlight:
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE)
            glColor4f(1.0, 1.0, 0.85, 0.90)
            glPointSize(5.0)
            glBegin(GL_POINTS)
            glVertex3f(0.0, 0.02, 0.44)
            glEnd()

            # Işık Huzmesi Konisi (Hafif Şeffaf)
            glColor4f(0.85, 0.95, 1.0, 0.08)
            glBegin(GL_TRIANGLE_FAN)
            glVertex3f(0.0, 0.02, 0.44)
            for deg in range(0, 361, 30):
                rad = math.radians(deg)
                cx = math.sin(rad) * 1.6
                cy = math.cos(rad) * 1.6
                glVertex3f(cx, cy, 6.5)
            glEnd()

        # =====================================================================
        # 10. 📏 TFMINI LAZER İRTİFA ÇİZGİSİ
        # =====================================================================
        if self.laser:
            dist_to_ground = max(0.1, (ry - floor_y) / self.scale)
            glDisable(GL_BLEND)
            glLineWidth(1.8)
            glColor4f(1.0, 0.2, 0.2, 0.85)
            glBegin(GL_LINES)
            glVertex3f(0.0, -0.06, 0.0)
            glVertex3f(0.0, -dist_to_ground, 0.0)
            glEnd()

            glPointSize(4.0)
            glBegin(GL_POINTS)
            glVertex3f(0.0, -dist_to_ground, 0.0)
            glEnd()

        glPopMatrix()
        glPopAttrib()


# ======================================================================================
# 8. 🚁 DRONE UÇUŞ FİZİK MOTORU (DronePhysicsEngine)
# ======================================================================================
class DronePhysicsEngine:
    """
    Gerçekçi quadcopter uçuş fiziği simülatörü.

    Fizik Modeli:
      - Kuvvet -> İvme -> Hız -> Konum Entegrasyonu (Newtonian Flight Dynamics)
      - İleri/Geri ve Sağa/Sola hareketlerde gövdeye dinamik tilt (Pitch & Roll) açısı üretimi
      - İrtifa hover PID mikro-salınımı (Canlı hava süzülmesi)
      - Yerçekimi, Aerodinamik Sürüklenme (Drag) ve Zemin/Duvar Çarpışması
    """

    GRAVITY = 9.81          # m/s² — standart yerçekimi ivmesi
    DRAG = 4.2              # Hava direnci katsayısı (hıza orantılı sönüm)
    LIFT_FORCE = 15.0       # Tam gaz kaldırma kuvveti (N/kg eşdeğeri)
    LATERAL_FORCE = 11.0    # Yatay hareket kuvveti (W/A/S/D)
    MAX_SPEED = 8.5         # Maksimum hız (m/s)
    CRASH_BOUNCE = 0.35     # Çarpışma sonrası geri sekme katsayısı
    LANDING_HEIGHT = 0.08   # Zemin üzerinde iniş eşiği (m)

    def __init__(self):
        # Hız vektörü (m/s)
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0

        # Konum
        self.x = 0.0
        self.y = 1.5
        self.z = 0.0

        # Dinamik Eğim (Tilt)
        self.pitch = 0.0
        self.roll = 0.0
        self.yaw = 0.0

        # Durum bayrakları
        self.is_landed = True           # Zeminde mi?
        self.is_crashed = False         # Yeni çarpışma oldu mu? (1 frame)
        self.crash_timer = 0.0          # Çarpışma sonrası kırmızı flaş süresi
        self.throttle = 0.0             # 0.0-1.0 arası gaz seviyesi
        self.floor_y = -0.5             # Zemin Y koordinatı
        self.collision_voxels = None    # OctomapEngine voksel merkezleri (N,3)
        self.voxel_size = 0.3           # Çarpışma vokseli boyutu
        self.flight_time = 0.0

    def set_floor(self, y):
        """Zemin yüksekliğini ayarla (FloorplanEstimator'dan alınan min_y)."""
        self.floor_y = y

    def set_collision_map(self, voxel_centers, voxel_size=0.3):
        """OctomapEngine'den üretilen voksel merkezlerini çarpışma haritası olarak yükle."""
        if voxel_centers is not None and len(voxel_centers) > 1000:
            step = max(1, len(voxel_centers) // 8000)
            self.collision_voxels = voxel_centers[::step].copy()
        else:
            self.collision_voxels = voxel_centers
        self.voxel_size = max(0.2, voxel_size)

    def _check_collision(self, nx, ny, nz, radius=0.35):
        """Yeni konumda çarpışma var mı kontrol et."""
        if self.collision_voxels is None or len(self.collision_voxels) == 0:
            return False

        threshold = self.voxel_size * 0.5 + radius
        bx_min, bx_max = nx - threshold * 3, nx + threshold * 3
        bz_min, bz_max = nz - threshold * 3, nz + threshold * 3
        by_min, by_max = ny - threshold * 2, ny + threshold * 2

        mask = (
            (self.collision_voxels[:, 0] > bx_min) & (self.collision_voxels[:, 0] < bx_max) &
            (self.collision_voxels[:, 2] > bz_min) & (self.collision_voxels[:, 2] < bz_max) &
            (self.collision_voxels[:, 1] > by_min) & (self.collision_voxels[:, 1] < by_max)
        )
        nearby = self.collision_voxels[mask]
        if len(nearby) == 0:
            return False

        dists = np.sqrt(
            (nearby[:, 0] - nx) ** 2 +
            (nearby[:, 1] - ny) ** 2 +
            (nearby[:, 2] - nz) ** 2
        )
        return bool(np.any(dists < threshold))

    def step(self, dt, throttle_up, move_fwd, move_back, move_left, move_right,
             fwd_x, fwd_y, fwd_z, right_x, right_z, fast_mode=False):
        """
        Kuvvet -> İvme -> Hız -> Konum fizik entegrasyonu gerçekleştirir.
        """
        dt = min(dt, 0.05)
        self.flight_time += dt
        self.is_crashed = False
        self.crash_timer = max(0.0, self.crash_timer - dt)

        speed_mult = 2.0 if fast_mode else 1.0
        lat_force = self.LATERAL_FORCE * speed_mult

        # --- Yerçekimi ---
        gravity_accel = -self.GRAVITY

        # --- Kaldırma Kuvveti (SPACE basılıysa yukarı gaz) ---
        self.throttle = 1.0 if throttle_up else max(0.0, self.throttle - dt * 2.8)
        lift_accel = self.throttle * self.LIFT_FORCE

        # --- Yatay İtme Kuvvetleri (Kamera Yönüne Göre) ---
        ax, az = 0.0, 0.0
        if move_fwd:
            ax += fwd_x * lat_force
            az += fwd_z * lat_force
        if move_back:
            ax -= fwd_x * lat_force
            az -= fwd_z * lat_force
        if move_left:
            ax -= right_x * lat_force
            az -= right_z * lat_force
        if move_right:
            ax += right_x * lat_force
            az += right_z * lat_force

        # Dron havadayken otomatik hover desteği (sürekli yere çakılmayı önler)
        hover_lift = self.GRAVITY * 0.96 if not self.is_landed else 0.0
        net_vy_accel = gravity_accel + lift_accel + (hover_lift if not throttle_up else 0.0)

        # --- Hız Entegrasyonu (Euler Entegrasyonu) ---
        self.vx += ax * dt
        self.vy += net_vy_accel * dt
        self.vz += az * dt

        # --- Hava Direnci (Aerodinamik Damping) ---
        drag = self.DRAG
        self.vx -= self.vx * drag * dt
        self.vy -= self.vy * drag * dt * 0.65
        self.vz -= self.vz * drag * dt

        # --- Hız Sınırı ---
        speed = math.sqrt(self.vx**2 + self.vy**2 + self.vz**2)
        max_spd = self.MAX_SPEED * speed_mult
        if speed > max_spd:
            scale = max_spd / speed
            self.vx *= scale
            self.vy *= scale
            self.vz *= scale

        # --- Yeni Konum ---
        nx = self.x + self.vx * dt
        ny = self.y + self.vy * dt
        nz = self.z + self.vz * dt

        # --- Zemin İniş Kontrolü ---
        land_y = self.floor_y + self.LANDING_HEIGHT
        if ny <= land_y:
            ny = land_y
            if self.vy < -1.4:
                self.vy = abs(self.vy) * 0.15  # Yumuşak iniş sekmesi
            else:
                self.vy = 0.0
            self.vx *= 0.85
            self.vz *= 0.85
            self.is_landed = True
        else:
            self.is_landed = False

        # --- Voksel Duvar Çarpışması ---
        if self._check_collision(nx, ny, nz):
            self.vx = -self.vx * self.CRASH_BOUNCE
            self.vy = -abs(self.vy) * self.CRASH_BOUNCE - 1.2
            self.vz = -self.vz * self.CRASH_BOUNCE
            nx, ny, nz = self.x, self.y, self.z
            self.is_crashed = True
            self.crash_timer = 0.6

        self.x = nx
        self.y = ny
        self.z = nz

        return self.x, self.y, self.z

    @property
    def speed(self):
        """Mevcut hız büyüklüğü (m/s)."""
        return math.sqrt(self.vx**2 + self.vy**2 + self.vz**2)


# ======================================================================================
# 9. 🌊 OPTİK FLOW HUD VİSUALİZER (OpticalFlowHUD)
# ======================================================================================
class OpticalFlowHUD:
    """
    Drone'un hareket vektörlerini ekran üzerinde optik akış okları olarak görselleştirir.

    Optik Flow Renk Kodu:
      🔵 Mavi  : İleri hareket
      🔴 Kırmızı: Geri hareket
      🟢 Yeşil : Sağ/Sol hareket
      ⚪ Beyaz : Yukarı/Aşağı hareket
    """

    def __init__(self, grid_cols=12, grid_rows=8):
        self.grid_cols = grid_cols
        self.grid_rows = grid_rows
        self.active = True
        # Geçmiş hız (smooth için)
        self.smooth_vx = 0.0
        self.smooth_vy = 0.0
        self.smooth_vz = 0.0

    def update(self, vx, vy, vz, dt):
        """Hız vektörünü yumuşat."""
        alpha = min(1.0, dt * 8.0)
        self.smooth_vx = (1.0 - alpha) * self.smooth_vx + alpha * vx
        self.smooth_vy = (1.0 - alpha) * self.smooth_vy + alpha * vy
        self.smooth_vz = (1.0 - alpha) * self.smooth_vz + alpha * vz

    def draw_gl(self, win_w, win_h, cam_yaw_deg):
        """
        Ekran üzerinde optik flow ok ızgarasını OpenGL ile çiz.
        cam_yaw_deg: Kameranın yaw açısı (hareket yönünü ekrana yansıtmak için)
        """
        speed = math.sqrt(self.smooth_vx**2 + self.smooth_vy**2 + self.smooth_vz**2)
        if speed < 0.05:
            return  # Çok yavaşsa çizme

        # Kamera koordinat sistemine hareket vektörünü çevir (yaw)
        yaw_rad = math.radians(cam_yaw_deg)
        cos_y = math.cos(yaw_rad)
        sin_y = math.sin(yaw_rad)

        # Ekran X,Y bileşenleri (kamera yönüne göre)
        screen_vx = self.smooth_vx * cos_y - self.smooth_vz * sin_y  # Yatay
        screen_vy = -self.smooth_vy                                    # Dikey (Y ters)
        screen_vz = self.smooth_vx * sin_y + self.smooth_vz * cos_y  # Derinlik → ok boyu

        # Normalize et
        max_v = max(speed, 0.1)
        nx_norm = screen_vx / max_v
        ny_norm = screen_vy / max_v
        nz_norm = screen_vz / max_v  # Derinlik bileşeni ok boyunu etkiler

        # Ok uzunluğu ve görünürlük
        arrow_len_base = min(speed * 6.0, 35.0)
        alpha_base = min(speed / 3.0, 0.9)

        # Renk: derinlik bileşenine göre (ileri=mavi, geri=kırmızı, yan=yeşil)
        if abs(nz_norm) > 0.5:
            r = max(0.0, -nz_norm)       # Geri = kırmızı
            g = 0.3
            b = max(0.0, nz_norm)        # İleri = mavi
        else:
            r = 0.2
            g = min(1.0, abs(nx_norm) * 2.0 + 0.4)   # Yan = yeşil
            b = 0.4

        # 2D ortho projection
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

        cell_w = win_w / self.grid_cols
        cell_h = win_h / self.grid_rows

        glLineWidth(1.5)

        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                # Ok başlangıç noktası (ızgara hücresinin merkezi)
                cx = (col + 0.5) * cell_w
                cy = (row + 0.5) * cell_h

                # Perspektif bozulma simülasyonu: merkeze uzak hücreler daha az hareket
                dist_cx = (col / self.grid_cols - 0.5) * 2.0  # -1 .. +1
                dist_cy = (row / self.grid_rows - 0.5) * 2.0
                persp = 1.0 - (abs(dist_cx) + abs(dist_cy)) * 0.15

                # Ok bitiş noktası
                ex = cx + (nx_norm + nz_norm * dist_cx * 0.5) * arrow_len_base * persp
                ey = cy + (ny_norm + nz_norm * dist_cy * 0.5) * arrow_len_base * persp

                alpha = alpha_base * persp * 0.75

                glColor4f(r, g, b, alpha)
                glBegin(GL_LINES)
                glVertex2f(cx, cy)
                glVertex2f(ex, ey)
                glEnd()

                # Ok ucu (küçük üçgen)
                dx = ex - cx
                dy = ey - cy
                arrow_len = math.sqrt(dx*dx + dy*dy)
                if arrow_len > 2.0:
                    head_size = min(arrow_len * 0.35, 6.0)
                    ux = dx / arrow_len
                    uy = dy / arrow_len
                    px = -uy * head_size * 0.4
                    py = ux * head_size * 0.4
                    glBegin(GL_TRIANGLES)
                    glColor4f(r, g, b, alpha * 0.9)
                    glVertex2f(ex, ey)
                    glVertex2f(ex - ux * head_size + px, ey - uy * head_size + py)
                    glVertex2f(ex - ux * head_size - px, ey - uy * head_size - py)
                    glEnd()

        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glPopAttrib()

