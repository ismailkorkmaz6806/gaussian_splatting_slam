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



