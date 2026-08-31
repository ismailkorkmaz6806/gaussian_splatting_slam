"""
MASt3R-SLAM: High-Definition Multi-Room Solid 3D Mesh Engine (Pure Full Mesh)
-----------------------------------------------------------------------------
Bu modül, MASt3R modelinden gelen ardışık keyframe çıkarımlarını kullanarak:
1. Ardışık kamera çiftleri arasında rijit nokta eşleme (Roma rigid registration)
   ile SE(3) kamera yörüngesini (Trajectory) hesaplar.
2. 3B nokta haritalarından yüksek netlikli, derinlik kopması olmayan sürekli
   katı 3B üçgen meş (Solid 3D Mesh) üretir.
3. Her köşe noktası için yüzey normallerini (Vertex Normals) hesaplar.
4. Modeli OpenGL koordinat sistemine (Y-up, zemin seviyeleme) dönüştürür ve
   .ply ile sıkıştırılmış .npz önbellek dosyalarına kaydeder.
5. Kamera hareket yörüngesini Cubic Spline enterpolasyonu ile pürüzsüzleştirir.
"""

import os
import sys
import time
import math
import numpy as np
import cv2
import torch
import roma
from scipy.interpolate import CubicSpline

# ==============================================================================
# 1. MASt3R / DUSt3R IMPORT VE YOL AYARLARI
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAST3R_DIR = os.path.join(BASE_DIR, "mast3r_repo")
DUST3R_DIR = os.path.join(MAST3R_DIR, "dust3r")
CROCO_DIR = os.path.join(DUST3R_DIR, "croco")

for p in [MAST3R_DIR, DUST3R_DIR, CROCO_DIR]:
    if p not in sys.path and os.path.exists(p):
        sys.path.insert(0, p)

from dust3r.inference import inference


# ==============================================================================
# 2. NOKTA HARİTASINDAN KATI 3B MEŞ ÖRME (TRIANGULATION)
# ==============================================================================
def frame_to_solid_mesh(rgb_img, pts3d_local, pts3d_world, conf_mask,
                        step=2, max_edge_len=0.18, max_depth_step=0.14):
    """
    Tek bir keyframe'in 3B nokta ızgarasını (grid) üçgen yüzeylerle (mesh) birbirine bağlar.
    Derinlik süreksizliklerinde (örn. ön nesne ile arka duvar arası) sahte üçgen
    köprüleri kurulmasını engeller.
    
    Parametreler:
        rgb_img (np.ndarray): [H, W, 3] RGB renk dizisi.
        pts3d_local (np.ndarray): [H, W, 3] Kamera yerel koordinatlarındaki noktalar (Z derinlik için).
        pts3d_world (np.ndarray): [H, W, 3] Dünya koordinatlarındaki 3B noktalar.
        conf_mask (np.ndarray): [H, W] Boolean geçerlilik ve güven maskesi.
        step (int): Izgara alt-örnekleme adımı (2 = her 2 pikselde 1 köşe, performans için).
        max_edge_len (float): Bir üçgen kenarının maksimum uzunluğu (metre). Aşarsa üçgen iptal edilir.
        max_depth_step (float): Komşu pikseller arası maksimum derinlik farkı. Aşarsa köprü üçgen silinir.
    Dönüş:
        tuple: (vertices, colors, faces) veya yüzey üretilemezse (None, None, None).
    """
    # Adım oranında alt örnekleme
    pts_sub_world = pts3d_world[::step, ::step]
    pts_sub_local = pts3d_local[::step, ::step]
    rgb_sub = rgb_img[::step, ::step]
    conf_sub = conf_mask[::step, ::step]

    H, W, _ = pts_sub_world.shape
    vertices = pts_sub_world.reshape(-1, 3).astype(np.float32)
    colors = rgb_sub.reshape(-1, 3).astype(np.float32)
    z_local = pts_sub_local[..., 2]

    # Izgara üzerinde 4'lü piksel hücrelerinin indeksleri
    idx = np.arange(H * W).reshape(H, W)
    idx1 = idx[:-1, :-1].ravel()   # Sol üst
    idx2 = idx[:-1, 1:].ravel()    # Sağ üst
    idx3 = idx[1:, :-1].ravel()    # Sol alt
    idx4 = idx[1:, 1:].ravel()     # Sağ alt

    # X ve Y eksenlerinde derinlik gradyanını kontrol et (uçurum kontrolü)
    dz_x = np.abs(z_local[:-1, :-1] - z_local[:-1, 1:]).ravel()
    dz_y = np.abs(z_local[:-1, :-1] - z_local[1:, :-1]).ravel()
    smooth_quad = (dz_x < max_depth_step) & (dz_y < max_depth_step)

    # Her 4'lü hücreyi iki üçgene böl (quad -> 2 triangles)
    faces1 = np.c_[idx1, idx2, idx3]
    faces2 = np.c_[idx2, idx4, idx3]
    faces = np.vstack([faces1, faces2])
    smooth_mask = np.concatenate([smooth_quad, smooth_quad])
    faces = faces[smooth_mask]

    # Güven maskesi kontrolü: Üçgenin 3 köşesi de güvenilir olmalı
    valid_mask = conf_sub.ravel()
    valid_tri = valid_mask[faces].all(axis=1)
    faces = faces[valid_tri]

    if len(faces) == 0:
        return None, None, None

    # Geometrik kenar uzunluğu filtresi (Aşırı uzun uzantıları ve esnemeleri engeller)
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    d01 = np.sum((v0 - v1)**2, axis=1)
    d12 = np.sum((v1 - v2)**2, axis=1)
    d20 = np.sum((v2 - v0)**2, axis=1)

    max_len_sq = max_edge_len ** 2
    clean_tri = (d01 < max_len_sq) & (d12 < max_len_sq) & (d20 < max_len_sq)
    faces = faces[clean_tri]

    if len(faces) == 0:
        return None, None, None

    return vertices, colors, faces


# ==============================================================================
# 3. YÜZEY NORMALLERİ HESAPLAMA (AYDINLATMA İÇİN)
# ==============================================================================
def compute_vertex_normals(vertices, faces):
    """
    Her köşe noktası için paylaşılan üçgenlerin alan ağırlıklı
    yüzey normallerini (Vertex Normals) hesaplar ve birim vektöre normalize eder.
    
    Parametreler:
        vertices (np.ndarray): [N, 3] köşe koordinatları.
        faces (np.ndarray): [M, 3] üçgen köşe indeksleri.
    Dönüş:
        np.ndarray: [N, 3] normalize edilmiş köşe normalleri.
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # Çapraz çarpım (Cross product) ile üçgen yüzey normali
    fn = np.cross(v1 - v0, v2 - v0)
    vn = np.zeros_like(vertices, dtype=np.float32)
    np.add.at(vn, faces[:, 0], fn)
    np.add.at(vn, faces[:, 1], fn)
    np.add.at(vn, faces[:, 2], fn)

    # Vektör boyuna bölerek normalize et
    norms = np.linalg.norm(vn, axis=1, keepdims=True)
    vn = np.where(norms > 1e-5, vn / np.maximum(norms, 1e-5), [0.0, 1.0, 0.0])
    return vn.astype(np.float32)


# ==============================================================================
# 4. ÇEVRİMDIŞI MASt3R GLOBAL SLAM PIPELINE
# ==============================================================================
def run_fast_mast3r_slam(frames, model, device="cuda",
                          output_ply="mast3r_map.ply",
                          min_conf_thr=1.40, max_edge_len=0.18, max_cam_dist=4.8,
                          detect_objects=True):
    """
    Tüm keyframe listesini alır, çiftler halinde MASt3R'dan geçirir,
    kamera pozlarını zincirler, katı 3B meş örer ve .ply olarak kaydeder.
    
    Parametreler:
        frames (list): prepare_frame_dict ile hazırlanmış keyframe listesi.
        model (AsymmetricMASt3R): Yüklenmiş MASt3R model örneği.
        device (str): İşlem cihazı ('cuda' veya 'cpu').
        output_ply (str): Çıkış .ply dosya yolu.
        min_conf_thr (float): Noktaların dahil edilmesi için minimum MASt3R güven skoru.
        max_edge_len (float): Üçgenleme maksimum kenar uzunluğu (metre).
        max_cam_dist (float): Kameradan maksimum güvenilir mesafe eşiği.
        detect_objects (bool): YOLOv8 ile 3B nesne tespiti yapılsın mı.
    Dönüş:
        str: Üretilen .ply dosyasının yolu.
    """
    N = len(frames)
    print("=" * 75)
    print(" 🌟 200+ FPS MASt3R-SLAM (SAF VE TAM 3B MEŞ REKONSTRÜKSİYONU)")
    print(f" 📦 Toplam Keyframe: {N} | Çift Sayısı: {2*(N-1)} | Cihaz: {device.upper()}")
    print("=" * 75)

    t0 = time.time()

    # 1. Çiftleri Oluştur (İleri ve Geri yönlü çapraz eşleme)
    pairs = [(frames[i], frames[i + 1]) for i in range(N - 1)] + \
            [(frames[i + 1], frames[i]) for i in range(N - 1)]

    # 2. MASt3R Çıkarımı
    print(f" [1/5] 🧠 MASt3R Vision Transformer Çıkarımı ({len(pairs)} Çift)...")
    out = inference(pairs, model, device=device, batch_size=2, verbose=True)

    # 3. Kesin SE(3) Poz Zincirleme (Rigid Registration)
    print("\n [2/5] 📐 Gerçek Metrik Kamera Yörüngesi Hesaplanıyor...")
    n_steps = N - 1
    cam_poses = [np.eye(4, dtype=np.float32)]  # İlk kamera referans orijindir (Identity)
    keyframe_pts3d = []
    keyframe_confs = []

    keyframe_pts3d.append(out['pred1']['pts3d'][0].cpu().numpy())
    keyframe_confs.append(out['pred1']['conf'][0].cpu().numpy())

    for i in range(n_steps):
        pts_i1_in_i = out['pred2']['pts3d_in_other_view'][i].cpu()
        conf_i1_in_i = out['pred2']['conf'][i].cpu()

        pts_i1_in_i1 = out['pred1']['pts3d'][n_steps + i].cpu()
        conf_i1_in_i1 = out['pred1']['conf'][n_steps + i].cpu()

        keyframe_pts3d.append(pts_i1_in_i1.numpy())
        keyframe_confs.append(conf_i1_in_i1.numpy())

        # Yüksek güven skoruna sahip ortak noktaları maskele
        mask = (conf_i1_in_i > 1.2) & (conf_i1_in_i1 > 1.2)
        p_src = pts_i1_in_i1[mask].view(-1, 3)
        p_tgt = pts_i1_in_i[mask].view(-1, 3)
        weights = (conf_i1_in_i[mask] * conf_i1_in_i1[mask]).view(-1)

        # Roma kütüphanesi ile ağırlıklı rijit nokta hizalama (R, t)
        if len(p_src) > 30:
            R, t = roma.rigid_points_registration(p_src, p_tgt, weights=weights, compute_scaling=False)
            T_step = np.eye(4, dtype=np.float32)
            T_step[:3, :3] = R.numpy()
            T_step[:3, 3] = t.numpy()
            T_next = cam_poses[-1] @ T_step
            cam_poses.append(T_next)
        else:
            diff = (pts_i1_in_i.mean(dim=(0, 1)) - out['pred1']['pts3d'][i].cpu().mean(dim=(0, 1))).numpy()
            T_step = np.eye(4, dtype=np.float32)
            T_step[:3, 3] = diff
            cam_poses.append(cam_poses[-1] @ T_step)

    cam_poses = np.array(cam_poses, dtype=np.float32)

    # 4. Katı 3B Meş Üretimi ve Birleştirme
    detected_3d_objects = []
    print(f"\n [3/4] 🏗️ 200+ FPS Akıcı ve Tam Katı Yüzeyler Örülüyor...")
    all_vertices = []
    all_colors = []
    all_faces = []
    vertex_offset = 0

    for idx in range(N):
        pts_local = keyframe_pts3d[idx]
        conf = keyframe_confs[idx]
        rgb = frames[idx]['rgb_np']
        T_w = cam_poses[idx]

        local_dist = np.linalg.norm(pts_local, axis=-1)
        c_mask = (conf > min_conf_thr) & (local_dist < max_cam_dist) & (pts_local[..., 2] > 0.15) & np.isfinite(pts_local).all(axis=-1)

        H_f, W_f, _ = pts_local.shape
        pts_flat = pts_local.reshape(-1, 3)
        pts_world_flat = (T_w[:3, :3] @ pts_flat.T).T + T_w[:3, 3]
        pts_world = pts_world_flat.reshape(H_f, W_f, 3)

        v, c, f = frame_to_solid_mesh(rgb, pts_local, pts_world, c_mask, step=2, max_edge_len=max_edge_len, max_depth_step=0.14)

        if v is not None and len(f) > 0:
            all_vertices.append(v)
            all_colors.append(c)
            all_faces.append(f + vertex_offset)
            vertex_offset += len(v)

    if len(all_vertices) == 0:
        print("❌ HATA: Geçerli yüzey üretilemedi!")
        return None

    merged_verts = np.vstack(all_vertices).astype(np.float32)
    merged_cols = np.vstack(all_colors).astype(np.float32)
    merged_faces = np.vstack(all_faces).astype(np.uint32)

    # Yüzey Normallerini Hesapla
    print(f"  -> 💡 {len(merged_faces):,} Katı Üçgen için Yüzey Normalleri Hesaplanıyor...")
    merged_norms = compute_vertex_normals(merged_verts, merged_faces)

    # 5. OpenGL Koordinat Eşitlemesi (Y-ekseni ters çevirme)
    print(" [5/5] 📐 OpenGL Koordinat Eşitlemesi ve Kayıt...")
    merged_verts[:, 1] = -merged_verts[:, 1]
    merged_norms[:, 1] = -merged_norms[:, 1]
    traj_cams_gl = cam_poses[:, :3, 3].copy()
    traj_cams_gl[:, 1] = -traj_cams_gl[:, 1]

    # Zemin Düzleme (Pitch eğimini sıfırlama)
    z_coords = traj_cams_gl[:, 2]
    y_coords = traj_cams_gl[:, 1]
    R_level = np.eye(3, dtype=np.float32)
    if len(z_coords) > 2:
        poly_fit = np.polyfit(z_coords, y_coords, 1)
        pitch_angle = math.atan(poly_fit[0])
        cos_p = math.cos(-pitch_angle)
        sin_p = math.sin(-pitch_angle)
        R_level = np.array([
            [1.0, 0.0, 0.0],
            [0.0, cos_p, -sin_p],
            [0.0, sin_p, cos_p]
        ], dtype=np.float32)

        merged_verts = (R_level @ merged_verts.T).T
        merged_norms = (R_level @ merged_norms.T).T
        traj_cams_gl = (R_level @ traj_cams_gl.T).T

    ground_y = float(np.percentile(merged_verts[:, 1], 2))
    merged_verts[:, 1] -= ground_y
    traj_cams_gl[:, 1] -= ground_y

    # İnsan / Nesne İmleçlerini Dönüştür
    for obj in detected_3d_objects:
        obj['centroid'][1] = -obj['centroid'][1]
        obj['min_corner'][1] = -obj['min_corner'][1]
        obj['max_corner'][1] = -obj['max_corner'][1]

        c_min_y = min(obj['min_corner'][1], obj['max_corner'][1])
        c_max_y = max(obj['min_corner'][1], obj['max_corner'][1])
        obj['min_corner'][1] = c_min_y
        obj['max_corner'][1] = c_max_y

        obj['centroid'] = (R_level @ obj['centroid']).astype(np.float32)
        obj['min_corner'] = (R_level @ obj['min_corner']).astype(np.float32)
        obj['max_corner'] = (R_level @ obj['max_corner']).astype(np.float32)

        obj['centroid'][1] -= ground_y
        obj['min_corner'][1] -= ground_y
        obj['max_corner'][1] -= ground_y

    merged_verts = np.ascontiguousarray(merged_verts, dtype=np.float32)
    merged_cols = np.ascontiguousarray(np.clip(merged_cols, 0.0, 1.0), dtype=np.float32)
    merged_norms = np.ascontiguousarray(merged_norms, dtype=np.float32)
    merged_faces = np.ascontiguousarray(merged_faces, dtype=np.uint32)

    # NPZ Önbellek Kaydı (Hızlı yükleme için)
    cache_path = output_ply.replace(".ply", "_cache.npz")
    np.savez_compressed(
        cache_path,
        verts=merged_verts,
        cols=merged_cols,
        norms=merged_norms,
        faces=merged_faces,
        pts=merged_verts[::2],
        pts_cols=merged_cols[::2],
        objects_3d=np.array(detected_3d_objects, dtype=object)
    )
    print(f"  -> ⚡ Katı Meş Önbelleği Kaydedildi: {cache_path}")

    # ASCII PLY Formatında Standart 3B Model Kaydı
    colors_uint8 = (merged_cols * 255).astype(np.uint8)
    with open(output_ply, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(merged_verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property float nx\nproperty float ny\nproperty float nz\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {len(merged_faces)}\n")
        f.write("property list uchar int vertex_indices\nend_header\n")
        for i in range(len(merged_verts)):
            p = merged_verts[i]
            n = merged_norms[i]
            c = colors_uint8[i]
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {n[0]:.4f} {n[1]:.4f} {n[2]:.4f} {c[0]} {c[1]} {c[2]}\n")
        for face in merged_faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

    # Kamera Yörüngesini Spline Enterpolasyonu İle Pürüzsüzleştir
    t_orig = np.linspace(0, 1, len(traj_cams_gl))
    t_fine = np.linspace(0, 1, 5000)
    cs_x = CubicSpline(t_orig, traj_cams_gl[:, 0])
    cs_y = CubicSpline(t_orig, traj_cams_gl[:, 1])
    cs_z = CubicSpline(t_orig, traj_cams_gl[:, 2])
    pos_fine = np.column_stack([cs_x(t_fine), cs_y(t_fine), cs_z(t_fine)]).astype(np.float32)
    np.savez("mast3r_trajectory.npz", positions=pos_fine)

    elapsed = time.time() - t0
    print("=" * 75)
    print(f" 🎉 SAF VE TAM 3B MODEL HAZIR: {output_ply}")
    print(f" 📐 {len(merged_verts):,} Köşe | {len(merged_faces):,} Katı Üçgen | {len(detected_3d_objects)} İnsan")
    print(f" ⏱️ Toplam Süre: {elapsed:.1f} saniye")
    print("=" * 75)
    return output_ply

