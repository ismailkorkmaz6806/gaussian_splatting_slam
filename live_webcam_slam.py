"""
========================================================================================
MASt3R-SLAM: Bozulmasız, Akıllı Voksel Tamamlamalı Canlı 3B Gezgin (144+ FPS)
========================================================================================
Bu dosya, canlı kamera veya video akışından gerçek zamanlı olarak 3B harita çıkaran
ve OpenGL üzerinde 144+ FPS hızında akıcı bir şekilde görselleştiren SLAM motorudur.

Temel Mimarisi ve Özellikleri:
1. 🧠 Akıllı Uzamsal Voksel Tamamlama (Smart Voxel Hole-Filling):
   - Önceden taranmış bir alana tekrar bakıldığında üst üste sahte yüzey eklemez ve
     orijinal temiz geometriyi ASLA bozmaz.
   - Sadece taranmamış, eksik veya yeni görünen 3B bölgeleri tespit edip haritayı tamamlar.
2. ⚡ Çok İş Parçacıklı (Multi-Threaded) Asenkron Yapı:
   - GPU yapay zeka çıkarımı (MASt3R + YOLOv8) arka plan iş parçacığında çalışırken,
     ön plandaki OpenGL render döngüsü asla takılmaz veya donmaz (144+ FPS).
3. 🔒 Saf Dönüş & Statik Duvar Kilidi:
   - Kamera dönüşlerinde sahte derinlik kaymalarını ve duvar eğilmelerini engeller.
4. 👤 Dinamik İnsan & Nesne Tespiti:
   - YOLOv8 ile kameradaki insanları tespit eder; 3B imleçle ayrıştırır ([O] tuşu).
========================================================================================
"""

import os
import sys
import time
import math
import threading
import queue
import cv2
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import torch
import roma

# ==============================================================================
# 1. MODÜL YOLLARININ VE İÇE AKTARIMLARIN AYARLANMASI
# ==============================================================================
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURR_DIR)
SLAM_DIR = os.path.join(PARENT_DIR, "mast3r_slam")

for p in [CURR_DIR, SLAM_DIR]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

from pointmap_engine import get_mast3r_model, prepare_frame_dict
from global_alignment import frame_to_solid_mesh, compute_vertex_normals
from object_detector_3d import extract_3d_objects


# ==============================================================================
# 2. 3B ÇİZİM YARDIMCI FONKSİYONLARI (FRUSTUM & BOUNDING BOX)
# ==============================================================================
def draw_frustum(pos, size=0.10, color=(1.0, 0.25, 0.25)):
    """
    3B uzayda anlık kameranın konumunu ve bakış piramidini (Camera Frustum) tel kafes olarak çizer.
    
    Parametreler:
        pos (tuple/list): Kameranın (X, Y, Z) dünya konumu.
        size (float): Piramidin boyutu (varsayılan: 0.10m).
        color (tuple): RGB renk üçlüsü (varsayılan: Kırmızı).
    """
    glDisable(GL_LIGHTING)
    glColor3f(*color)
    glBegin(GL_LINES)
    px, py, pz = pos
    s = size
    # Kamera merkezinden 4 köşe noktasına çizgiler
    glVertex3f(px, py, pz); glVertex3f(px - s, py - s, pz - s)
    glVertex3f(px, py, pz); glVertex3f(px + s, py - s, pz - s)
    glVertex3f(px, py, pz); glVertex3f(px + s, py + s, pz - s)
    glVertex3f(px, py, pz); glVertex3f(px - s, py + s, pz - s)
    # Ön taban karesinin çizgileri
    glVertex3f(px - s, py - s, pz - s); glVertex3f(px + s, py - s, pz - s)
    glVertex3f(px + s, py - s, pz - s); glVertex3f(px + s, py + s, pz - s)
    glVertex3f(px + s, py + s, pz - s); glVertex3f(px - s, py + s, pz - s)
    glVertex3f(px - s, py + s, pz - s); glVertex3f(px - s, py - s, pz - s)
    glEnd()


def draw_human_marker(centroid, min_c, max_c, color=(1.0, 0.28, 0.38)):
    """
    YOLOv8 tarafından tespit edilen insanların etrafına 3B sınırlayıcı kutu (Bounding Box) çizer.
    
    Parametreler:
        centroid (np.ndarray): İnsanın 3B merkez konumu.
        min_c (np.ndarray): Kutunun minimum (X, Y, Z) köşesi.
        max_c (np.ndarray): Kutunun maksimum (X, Y, Z) köşesi.
        color (tuple): Tel kafes rengi.
    """
    glDisable(GL_LIGHTING)
    glColor3f(*color)
    glBegin(GL_LINES)
    x0, y0, z0 = min_c
    x1, y1, z1 = max_c
    # 3B kutunun 12 kenar çizgisi
    for (xa, ya, za), (xb, yb, zb) in [
        ((x0, y0, z0), (x1, y0, z0)), ((x1, y0, z0), (x1, y1, z0)), ((x1, y1, z0), (x0, y1, z0)), ((x0, y1, z0), (x0, y0, z0)),
        ((x0, y0, z1), (x1, y0, z1)), ((x1, y0, z1), (x1, y1, z1)), ((x1, y1, z1), (x0, y1, z1)), ((x0, y1, z1), (x0, y0, z1)),
        ((x0, y0, z0), (x0, y0, z1)), ((x1, y0, z0), (x1, y0, z1)), ((x1, y0, z1), (x1, y1, z1)), ((x0, y1, z0), (x0, y1, z1))
    ]:
        glVertex3f(xa, ya, za)
        glVertex3f(xb, yb, zb)
    glEnd()


# ==============================================================================
# 3. 🧠 AKILLI UZAMSAL VOKSEL BİRİKTİRİCİ (SmartVoxelMeshAccumulator)
# ==============================================================================
class SmartVoxelMeshAccumulator:
    """
    Önceden taranan alanları voksel ızgarasında saklar. Kamerayı aynı yere tekrar
    tuttuğunuzda mükerrer yüzey eklemez, temiz geometriyi korur ve sadece eksik
    kalan boşlukları (hole-filling) tamamlar.
    """
    def __init__(self, voxel_size=0.032):
        self.voxel_size = voxel_size       # Voksel küp boyutu (metre cinsinden: 3.2 cm)
        self.occupied_voxels = set()       # Daha önce doldurulmuş (X, Y, Z) voksel anahtarları
        self.all_verts = []                # Biriktirilen köşe listesi
        self.all_cols = []                 # Biriktirilen renk listesi
        self.all_norms = []                # Biriktirilen normal vektör listesi
        self.all_faces = []                # Biriktirilen üçgen yüzey indeksleri
        self.vertex_offset = 0             # Birleştirme için indis kaydırma değeri

    def add_mesh(self, verts, cols, norms, faces):
        """
        Yeni gelen meş parçasını filtreler; sadece yeni voksellere denk gelen kısımları ekler.
        
        Parametreler:
            verts (np.ndarray): [N, 3] köşe noktaları.
            cols (np.ndarray): [N, 3] RGB renkleri.
            norms (np.ndarray): [N, 3] yüzey normalleri.
            faces (np.ndarray): [M, 3] üçgen yüzeyler.
        Dönüş:
            tuple: (eklendi_mi (bool), eklenen_kose_sayisi (int))
        """
        if verts is None or len(verts) == 0 or faces is None or len(faces) == 0:
            return False, 0

        # Noktaları voksel ızgara koordinatlarına dönüştür
        v_idx = np.floor(verts / self.voxel_size).astype(np.int32)

        # İlk keyframe ise tamamını doğrudan kabul et
        if len(self.occupied_voxels) == 0:
            for vx, vy, vz in v_idx:
                self.occupied_voxels.add((int(vx), int(vy), int(vz)))
            self.all_verts.append(verts)
            self.all_cols.append(cols)
            if norms is not None:
                self.all_norms.append(norms)
            self.all_faces.append(faces)
            self.vertex_offset += len(verts)
            return True, len(verts)

        # Yeni / boşluk olan bölgeleri bul
        is_new_vert = np.zeros(len(verts), dtype=bool)
        new_keys = []
        for i, (vx, vy, vz) in enumerate(v_idx):
            k = (int(vx), int(vy), int(vz))
            if k not in self.occupied_voxels:
                is_new_vert[i] = True
                new_keys.append(k)

        num_new = np.sum(is_new_vert)
        # Eğer bu karedeki yeni taranan nokta sayısı çok azsa (< 40), haritayı bozmamak için atla
        if num_new < 40:
            return False, 0

        # Sadece yeni bölgeleri tamamlayan üçgenleri filtrele
        f0_new = is_new_vert[faces[:, 0]]
        f1_new = is_new_vert[faces[:, 1]]
        f2_new = is_new_vert[faces[:, 2]]
        valid_faces_mask = f0_new | f1_new | f2_new
        valid_faces = faces[valid_faces_mask]

        if len(valid_faces) == 0:
            return False, 0

        # Kullanılan noktaları kompakt indeksle
        used_indices = np.unique(valid_faces)
        remap = np.full(len(verts), -1, dtype=np.int32)
        remap[used_indices] = np.arange(len(used_indices), dtype=np.int32)

        c_verts = verts[used_indices]
        c_cols = cols[used_indices]
        c_norms = norms[used_indices] if norms is not None else None
        c_faces = remap[valid_faces]

        for k in new_keys:
            self.occupied_voxels.add(k)

        self.all_verts.append(c_verts)
        self.all_cols.append(c_cols)
        if c_norms is not None:
            self.all_norms.append(c_norms)
        self.all_faces.append(c_faces + self.vertex_offset)
        self.vertex_offset += len(c_verts)

        return True, len(c_verts)

    def get_merged(self):
        """Tüm biriktirilmiş parçaları tek bir NumPy VBO matrisinde toplar."""
        if len(self.all_verts) == 0:
            return None, None, None, None
        mv = np.vstack(self.all_verts).astype(np.float32)
        mc = np.vstack(self.all_cols).astype(np.float32)
        mn = np.vstack(self.all_norms).astype(np.float32) if len(self.all_norms) == len(self.all_verts) else None
        mf = np.vstack(self.all_faces).astype(np.uint32)
        return mv, mc, mn, mf



def update_global_people(global_people, new_detections, match_dist=1.8):
    for det in new_detections:
        matched = False
        det_c = det['centroid']
        for g in global_people:
            dist = np.linalg.norm(g['centroid'] - det_c)
            if dist < match_dist:
                g['centroid'] = 0.70 * g['centroid'] + 0.30 * det_c
                g['min_corner'] = 0.70 * g['min_corner'] + 0.30 * det['min_corner']
                g['max_corner'] = 0.70 * g['max_corner'] + 0.30 * det['max_corner']
                g['score'] = max(g['score'], det['score'])
                matched = True
                break
        if not matched:
            global_people.append(det)


def run_live_webcam_slam(cam_id=0, target_size=336, min_depth_dist=0.65, max_cam_dist=4.8):
    print("=" * 75)
    print(" 🛡️ MASt3R-SLAM: AKILLI VOKSEL TAMAMLAMALI CANLI 3B GEZGİN (144+ FPS)")
    print(f" 🎥 Kamera: #{cam_id} | Akıllı Boşluk Doldurma: AKTİF | Kasma Önleme: AÇIK")
    print("=" * 75)

    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"❌ HATA: Webcam #{cam_id} açılamadı!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_mast3r_model(device=device)

    from dust3r.inference import inference

    kf_request_queue = queue.Queue(maxsize=2)
    kf_result_queue = queue.Queue()
    worker_running = True

    keyframe_dicts = []
    keyframe_pts3d = []
    keyframe_confs = []
    cam_poses = []

    def slam_worker():
        while worker_running:
            try:
                raw_bgr, kf_idx = kf_request_queue.get(timeout=0.08)
            except queue.Empty:
                continue

            t_start = time.time()
            f_dict = prepare_frame_dict(raw_bgr, idx=kf_idx, target_size=target_size, is_bgr=True)
            keyframe_dicts.append(f_dict)

            with torch.inference_mode():
                if kf_idx == 0:
                    pairs = [(f_dict, f_dict)]
                    out = inference(pairs, model, device=device, batch_size=1, verbose=False)
                    p_local = out['pred1']['pts3d'][0].cpu().numpy()
                    c_local = out['pred1']['conf'][0].cpu().numpy()

                    keyframe_pts3d.append(p_local)
                    keyframe_confs.append(c_local)
                    cam_poses.append(np.eye(4, dtype=np.float32))

                    p_world = p_local.copy()
                    p_world[..., 1] = -p_world[..., 1]

                    c_mask = (c_local > 1.35) & (p_local[..., 2] >= min_depth_dist) & (np.linalg.norm(p_local, axis=-1) < max_cam_dist)
                    v, c, f = frame_to_solid_mesh(f_dict['rgb_np'], p_local, p_world, c_mask, max_edge_len=0.14)
                    vn = compute_vertex_normals(v, f) if v is not None and len(f) > 0 else None

                    p_obj = extract_3d_objects([f_dict], [p_local], [cam_poses[-1]], [c_local])
                    for p in p_obj:
                        p['centroid'][1] = -p['centroid'][1]
                        p['min_corner'][1] = -p['min_corner'][1]
                        p['max_corner'][1] = -p['max_corner'][1]
                        c_min_y = min(p['min_corner'][1], p['max_corner'][1])
                        c_max_y = max(p['min_corner'][1], p['max_corner'][1])
                        p['min_corner'][1] = c_min_y
                        p['max_corner'][1] = c_max_y

                    kf_result_queue.put({
                        'kf_idx': kf_idx,
                        'verts': v, 'cols': c, 'norms': vn, 'faces': f,
                        'cam_pos': np.array([0.0, 0.0, 0.0], dtype=np.float32),
                        'people': p_obj,
                        'latency': time.time() - t_start
                    })
                else:
                    prev_dict = keyframe_dicts[-2]
                    pairs = [(prev_dict, f_dict), (f_dict, prev_dict)]
                    out = inference(pairs, model, device=device, batch_size=2, verbose=False)

                    pts_curr_in_prev = out['pred2']['pts3d_in_other_view'][0].cpu()
                    conf_curr_in_prev = out['pred2']['conf'][0].cpu()
                    pts_curr_in_curr = out['pred1']['pts3d'][1].cpu()
                    conf_curr_in_curr = out['pred1']['conf'][1].cpu()

                    keyframe_pts3d.append(pts_curr_in_curr.numpy())
                    keyframe_confs.append(conf_curr_in_curr.numpy())

                    mask = (conf_curr_in_prev > 1.25) & (conf_curr_in_curr > 1.25) & (pts_curr_in_curr[..., 2] >= min_depth_dist)
                    p_src = pts_curr_in_curr[mask].view(-1, 3)
                    p_tgt = pts_curr_in_prev[mask].view(-1, 3)
                    weights = (conf_curr_in_prev[mask] * conf_curr_in_curr[mask]).view(-1)

                    if len(p_src) > 30:
                        R, t = roma.rigid_points_registration(p_src, p_tgt, weights=weights, compute_scaling=False)
                        T_step = np.eye(4, dtype=np.float32)
                        T_step[:3, :3] = R.numpy()

                        t_vec = t.numpy()
                        if np.linalg.norm(t_vec) < 0.08:
                            t_vec = np.zeros(3, dtype=np.float32)

                        T_step[:3, 3] = t_vec
                        T_curr_world = cam_poses[-1] @ T_step
                        cam_poses.append(T_curr_world)
                    else:
                        diff = (pts_curr_in_prev.mean(dim=(0, 1)) - out['pred1']['pts3d'][0].cpu().mean(dim=(0, 1))).numpy()
                        T_step = np.eye(4, dtype=np.float32)
                        T_step[:3, 3] = diff
                        cam_poses.append(cam_poses[-1] @ T_step)

                    T_w = cam_poses[-1]
                    p_loc = pts_curr_in_curr.numpy()
                    c_loc = conf_curr_in_curr.numpy()
                    H_f, W_f, _ = p_loc.shape
                    pts_flat = p_loc.reshape(-1, 3)
                    pts_world_flat = (T_w[:3, :3] @ pts_flat.T).T + T_w[:3, 3]
                    p_world = pts_world_flat.reshape(H_f, W_f, 3)
                    p_world[..., 1] = -p_world[..., 1]

                    c_mask = (c_loc > 1.35) & (p_loc[..., 2] >= min_depth_dist) & (np.linalg.norm(p_loc, axis=-1) < max_cam_dist)
                    v, c, f = frame_to_solid_mesh(f_dict['rgb_np'], p_loc, p_world, c_mask, max_edge_len=0.14)
                    vn = compute_vertex_normals(v, f) if v is not None and len(f) > 0 else None

                    p_obj = extract_3d_objects([f_dict], [p_loc], [T_w], [c_loc])
                    for p in p_obj:
                        p['centroid'][1] = -p['centroid'][1]
                        p['min_corner'][1] = -p['min_corner'][1]
                        p['max_corner'][1] = -p['max_corner'][1]
                        c_min_y = min(p['min_corner'][1], p['max_corner'][1])
                        c_max_y = max(p['min_corner'][1], p['max_corner'][1])
                        p['min_corner'][1] = c_min_y
                        p['max_corner'][1] = c_max_y

                    cam_p_gl = T_w[:3, 3].copy()
                    cam_p_gl[1] = -cam_p_gl[1]

                    kf_result_queue.put({
                        'kf_idx': kf_idx,
                        'verts': v, 'cols': c, 'norms': vn, 'faces': f,
                        'cam_pos': cam_p_gl,
                        'people': p_obj,
                        'latency': time.time() - t_start
                    })

            kf_request_queue.task_done()

    worker_th = threading.Thread(target=slam_worker, daemon=True)
    worker_th.start()

    # Pygame & OpenGL
    pygame.init()
    win_w, win_h = 1280, 720
    pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
    pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)
    pygame.display.set_mode((win_w, win_h), DOUBLEBUF | OPENGL | RESIZABLE)
    pygame.display.set_caption("MASt3R-SLAM: Akıllı Voksel Tamamlamalı 3B Gezgin (144 FPS)")

    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)
    glDisable(GL_CULL_FACE)
    glShadeModel(GL_SMOOTH)
    glEnable(GL_NORMALIZE)
    glEnable(GL_MULTISAMPLE)

    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_AMBIENT, [0.50, 0.50, 0.50, 1.0])
    glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.85, 0.85, 0.85, 1.0])
    glLightfv(GL_LIGHT0, GL_SPECULAR, [0.25, 0.25, 0.25, 1.0])
    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE)

    # Zemin Izgarası
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

    vbo_verts = glGenBuffers(1)
    vbo_cols = glGenBuffers(1)
    vbo_norms = glGenBuffers(1)
    vbo_faces = glGenBuffers(1)

    mesh_accumulator = SmartVoxelMeshAccumulator(voxel_size=0.030)
    cam_positions_gl = []
    global_tracked_people = []
    num_indices = 0
    kf_counter = 0

    cam_yaw, cam_pitch = 0.0, 0.0
    target_yaw, target_pitch = 0.0, 0.0
    drone_x, drone_y, drone_z = 0.0, 1.2, -1.2
    vel_x, vel_y, vel_z = 0.0, 0.0, 0.0
    flight_speed = 0.065

    mouse_down_left = False
    last_mouse_pos = (0, 0)
    auto_scan = True
    show_people = True
    lighting_enabled = True

    prev_gray = None
    accum_motion = 0.0
    motion_thresh = 4.5  # Hareket eşiği: Gereksiz mükerrer taramayı engeller
    last_kf_time = time.time()

    clock = pygame.time.Clock()
    running = True

    # İlk kareyi anında gönder
    ret, first_frame = cap.read()
    if ret:
        kf_request_queue.put((first_frame, kf_counter))
        kf_counter += 1
        prev_gray = cv2.cvtColor(cv2.resize(first_frame, (120, 90)), cv2.COLOR_BGR2GRAY)

    print(" 💡 [A]: Otomatik Tarama | [SPACE]: Manuel Keyframe | [S]: Kaydet (.ply) | [O]: İnsan İmleci")

    while running:
        dt = clock.tick(144) / 1000.0

        ret, live_frame = cap.read()

        # Otomatik hareket algılama
        if ret and auto_scan and prev_gray is not None and kf_request_queue.empty():
            gray = cv2.cvtColor(cv2.resize(live_frame, (120, 90)), cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 2, 6, 2, 5, 1.1, 0)
            mag = np.hypot(flow[..., 0], flow[..., 1])
            accum_motion += float(np.mean(mag))
            prev_gray = gray

            if accum_motion >= motion_thresh and (time.time() - last_kf_time) > 0.45:
                kf_request_queue.put((live_frame, kf_counter))
                kf_counter += 1
                accum_motion = 0.0
                last_kf_time = time.time()

        # Arka plandan gelen sonuçları birleştir
        while not kf_result_queue.empty():
            res = kf_result_queue.get()
            v, c, vn, f = res['verts'], res['cols'], res['norms'], res['faces']

            if v is not None and len(f) > 0:
                # Akıllı Voksel Filtresi: Sadece yeni/eksik kısımları ekle
                added_ok, added_verts = mesh_accumulator.add_mesh(v, c, vn, f)
                cam_positions_gl.append(res['cam_pos'])
                if res['people']:
                    update_global_people(global_tracked_people, res['people'], match_dist=1.8)

                if added_ok:
                    merged_v, merged_c, merged_n, merged_f = mesh_accumulator.get_merged()
                    if merged_v is not None:
                        glBindBuffer(GL_ARRAY_BUFFER, vbo_verts)
                        glBufferData(GL_ARRAY_BUFFER, merged_v.nbytes, merged_v, GL_DYNAMIC_DRAW)

                        glBindBuffer(GL_ARRAY_BUFFER, vbo_cols)
                        glBufferData(GL_ARRAY_BUFFER, merged_c.nbytes, merged_c, GL_DYNAMIC_DRAW)

                        if merged_n is not None:
                            glBindBuffer(GL_ARRAY_BUFFER, vbo_norms)
                            glBufferData(GL_ARRAY_BUFFER, merged_n.nbytes, merged_n, GL_DYNAMIC_DRAW)

                        flat_f = merged_f.flatten()
                        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_faces)
                        glBufferData(GL_ELEMENT_ARRAY_BUFFER, flat_f.nbytes, flat_f, GL_DYNAMIC_DRAW)
                        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
                        num_indices = len(flat_f)

                        sys.stdout.write(f"\r ⚡ [Keyframe #{res['kf_idx']+1}] Eksik Bölge Tamamlandı! (+{added_verts:,} Yeni Köşe | Toplam: {len(merged_v):,} Köşe)")
                        sys.stdout.flush()
                else:
                    sys.stdout.write(f"\r 🛡️ [Keyframe #{res['kf_idx']+1}] Zaten taranmış bölge algılandı -> Mevcut temiz geometri korundu.")
                    sys.stdout.flush()

        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == VIDEORESIZE:
                win_w, win_h = event.w, event.h
                glViewport(0, 0, win_w, win_h)
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouse_down_left = True
                    last_mouse_pos = event.pos
            elif event.type == MOUSEBUTTONUP:
                if event.button == 1:
                    mouse_down_left = False
            elif event.type == MOUSEMOTION:
                dx = event.pos[0] - last_mouse_pos[0]
                dy = event.pos[1] - last_mouse_pos[1]
                last_mouse_pos = event.pos
                if mouse_down_left:
                    target_yaw += dx * 0.32
                    target_pitch += dy * 0.32
                    target_pitch = max(-89.0, min(89.0, target_pitch))
            elif event.type == KEYDOWN:
                if event.key == K_SPACE and ret and kf_request_queue.empty():
                    kf_request_queue.put((live_frame, kf_counter))
                    kf_counter += 1
                    last_kf_time = time.time()
                elif event.key == K_a:
                    auto_scan = not auto_scan
                    print(f"\n 🔄 Otomatik Canlı Tarama: {'AÇIK' if auto_scan else 'KAPALI'}")
                elif event.key == K_o:
                    show_people = not show_people
                    print(f"\n 👤 İnsan İmleci: {'GÖSTERİLİYOR' if show_people else 'GİZLENDİ'}")
                elif event.key == K_l:
                    lighting_enabled = not lighting_enabled
                    print(f"\n 💡 Işıklandırma: {'AÇIK' if lighting_enabled else 'KAPALI'}")
                elif event.key == K_s and mesh_accumulator.vertex_offset > 0:
                    out_ply = "webcam_map.ply"
                    mv, mc, mn, mf = mesh_accumulator.get_merged()
                    if mv is not None:
                        mc_uint8 = (mc * 255).astype(np.uint8)
                        with open(out_ply, "w", encoding="utf-8") as f:
                            f.write(f"ply\nformat ascii 1.0\nelement vertex {len(mv)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nelement face {len(mf)}\nproperty list uchar int vertex_indices\nend_header\n")
                            for i in range(len(mv)):
                                f.write(f"{mv[i,0]:.4f} {mv[i,1]:.4f} {mv[i,2]:.4f} {mc_uint8[i,0]} {mc_uint8[i,1]} {mc_uint8[i,2]}\n")
                            for face in mf:
                                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
                        print(f"\n 💾 3B Meş Kaydedildi: {out_ply} ({len(mv):,} Köşe, {len(mf):,} Yüzey)")
                elif event.key in (K_ESCAPE, K_q):
                    running = False

        # Kamera Uçuş Kontrolleri
        cam_yaw = 0.85 * cam_yaw + 0.15 * target_yaw
        cam_pitch = 0.85 * cam_pitch + 0.15 * target_pitch

        keys = pygame.key.get_pressed()
        rad_yaw, rad_pitch = math.radians(cam_yaw), math.radians(cam_pitch)
        fwd_x = math.sin(rad_yaw) * math.cos(rad_pitch)
        fwd_y = -math.sin(rad_pitch)
        fwd_z = math.cos(rad_yaw) * math.cos(rad_pitch)
        right_x = math.cos(rad_yaw)
        right_z = -math.sin(rad_yaw)

        target_vx, target_vy, target_vz = 0.0, 0.0, 0.0
        spd = flight_speed * (60.0 * dt)
        if keys[K_w] or keys[K_UP]: target_vx += fwd_x * spd; target_vy += fwd_y * spd; target_vz += fwd_z * spd
        if keys[K_s] or keys[K_DOWN]: target_vx -= fwd_x * spd; target_vy -= fwd_y * spd; target_vz -= fwd_z * spd
        if keys[K_a] or keys[K_LEFT]: target_vx -= right_x * spd; target_vz -= right_z * spd
        if keys[K_d] or keys[K_RIGHT]: target_vx += right_x * spd; target_vz += right_z * spd
        if keys[K_e]: target_vy += spd
        if keys[K_c]: target_vy -= spd

        vel_x = 0.78 * vel_x + 0.22 * target_vx
        vel_y = 0.78 * vel_y + 0.22 * target_vy
        vel_z = 0.78 * vel_z + 0.22 * target_vz
        drone_x += vel_x; drone_y += vel_y; drone_z += vel_z

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60.0, (win_w / max(win_h, 1)), 0.02, 300.0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glRotatef(cam_pitch, 1, 0, 0)
        glRotatef(-cam_yaw, 0, 1, 0)
        glTranslatef(-drone_x, -drone_y, -drone_z)

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

        # Aktif Kamera Frustumu
        if len(cam_positions_gl) > 0:
            draw_frustum(cam_positions_gl[-1], size=0.10, color=(1.0, 0.25, 0.25))

        # 3B İnsan İmleci
        if show_people and len(global_tracked_people) > 0:
            for p in global_tracked_people:
                draw_human_marker(p['centroid'], p['min_corner'], p['max_corner'], color=p.get('color', (1.0, 0.28, 0.38)))

        # 3B Meş Çizimi
        if lighting_enabled:
            glEnable(GL_LIGHTING)

        if num_indices > 0:
            glBindBuffer(GL_ARRAY_BUFFER, vbo_verts)
            glVertexPointer(3, GL_FLOAT, 0, None)
            glEnableClientState(GL_VERTEX_ARRAY)

            glBindBuffer(GL_ARRAY_BUFFER, vbo_cols)
            glColorPointer(3, GL_FLOAT, 0, None)
            glEnableClientState(GL_COLOR_ARRAY)

            glBindBuffer(GL_ARRAY_BUFFER, vbo_norms)
            glNormalPointer(GL_FLOAT, 0, None)
            glEnableClientState(GL_NORMAL_ARRAY)

            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_faces)
            glDrawElements(GL_TRIANGLES, num_indices, GL_UNSIGNED_INT, None)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

            glDisableClientState(GL_NORMAL_ARRAY)
            glDisableClientState(GL_COLOR_ARRAY)
            glDisableClientState(GL_VERTEX_ARRAY)
            glBindBuffer(GL_ARRAY_BUFFER, 0)

        pygame.display.flip()

    worker_running = False
    cap.release()
    pygame.quit()
    print("\n👋 Canlı MASt3R-SLAM kapatıldı.")


if __name__ == "__main__":
    cam = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    run_live_webcam_slam(cam_id=cam)
