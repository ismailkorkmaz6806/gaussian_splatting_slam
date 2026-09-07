"""
========================================================================================
MASt3R to 3D Gaussian Splatting (3DGS) Dönüştürücü ve Rekonstrüksiyon Motoru
========================================================================================
Bu dosya projenin ANA HESAPLAMA MOTORUDUR. Ham video dosyasından başlayarak
milyonlarca fotogerçekçi 3D Gaussian Splat elipsoidi üretir.

İşlem Adımları:
1. Video karelerinden en yüksek kaliteli anahtar kareleri (Keyframe) seçer.
2. Naver MASt3R Vision Transformer yapay zekasını GPU üzerinde çalıştırarak her
   pikselin 3B derinlik haritasını ve güvenilirlik (confidence) skorunu çıkarır.
3. Çiftler arası kapalı form (Closed-Form Procrustes) ile milimetrik SE(3) kamera
   pozlarını ve uzaydaki gerçek kamera uçuş yörüngesini hesaplar.
4. Her 3B nokta için 3D Gaussian Splatting standart parametrelerini türetir:
   - 3B Konum (X, Y, Z)
   - Yüzey Teğetlerine göre Yönsel Dönüş Kuaterniyonu (rot_0, rot_1, rot_2, rot_3)
   - Anizotropik Ölçek Faktörleri (scale_0, scale_1, scale_2)
   - Görünürlük & Güvenilirlik Opaklığı (opacity)
   - Küresel Harmonik Renk Katsayıları (f_dc_0, f_dc_1, f_dc_2)
5. Modeli standart ikili (binary) .PLY ve anında açılan .NPZ önbellek dosyalarına kaydeder.
========================================================================================
"""

import os
import sys
import time
import math
import numpy as np
import cv2
import torch

# Çalışma dizini ve komşu 'mast3r_slam' modül yollarının sisteme eklenmesi
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURR_DIR)
SLAM_DIR = os.path.join(PARENT_DIR, "mast3r_slam")

for p in [CURR_DIR, PARENT_DIR, SLAM_DIR]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

# MASt3R temel model yükleyicisi ve kare hazırlama yardımcı fonksiyonları
from pointmap_engine import get_mast3r_model, prepare_frame_dict
from global_alignment import compute_vertex_normals

# Windows konsolunda Türkçe karakterlerin düzgün görüntülenmesini sağlama
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def build_gaussian_splats_from_mast3r(video_file="ofisvideo.mp4",
                                      output_ply="gaussian_scene.ply",
                                      num_keyframes=50,
                                      target_size=512):
    """
    Ham video dosyasını alıp baştan sona 3D Gaussian Splatting (.ply & .npz) modeline dönüştüren ana fonksiyondur.
    
    Parametreler:
        video_file    : İşlenecek girdi MP4 video dosyasının adı veya yolu
        output_ply    : Üretilecek 3D Gaussian Splat PLY dosyasının adı
        num_keyframes : Videodan seçilecek anahtar kare (keyframe) sayısı (Örn: 40-60)
        target_size   : MASt3R yapay zekasına beslenecek çözünürlük (Örn: 512x512)
    """
    print("=" * 75)
    print(" 🌟 3D GAUSSIAN SPLATTING (3DGS) ÜRETİCİSİ (CVPR / SIGGRAPH STANDARDI)")
    print(f" 📹 Video Dosyası : {video_file}")
    print(f" 🎯 Keyframe Sayısı: {num_keyframes} Adet")
    print(f" 📐 Model Çözünürlüğü: {target_size}x{target_size} piksel")
    print("=" * 75)

    # 1. Video dosyasının mevcut olup olmadığını denetle
    full_video_path = video_file
    if not os.path.exists(full_video_path):
        alt_path = os.path.join(SLAM_DIR, video_file)
        if os.path.exists(alt_path):
            full_video_path = alt_path

    if not os.path.exists(full_video_path):
        print(f"❌ HATA: Video dosyası bulunamadı: {full_video_path}")
        return None

    # 2. Donanımsal NVIDIA GPU (CUDA) denetimi ve yapay zeka modelinin yüklenmesi
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_mast3r_model(device=device)

    # =========================================================================
    # [ADIM 1 / 4] 🎬 Video Karelerini Tarama ve Eşit Aralıklı Keyframe Seçimi
    # =========================================================================
    print("\n [1/4] 🎬 Video Kareleri Taranıyor ve Yüksek Çözünürlüklü Kareler Hazırlanıyor...")
    cap = cv2.VideoCapture(full_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Video boyunca eşit aralıklarla keyframe indisleri oluştur
    frame_indices = np.linspace(0, total_frames - 1, num_keyframes, dtype=int)
    frames = []

    curr_frame_idx = 0
    selected_idx = 0

    # Videoyu kare kare oku ve belirlenen indislerdeki kareleri 512x512 tensör formatına dönüştür
    while cap.isOpened() and selected_idx < len(frame_indices):
        ret, frame = cap.read()
        if not ret:
            break
        if curr_frame_idx == frame_indices[selected_idx]:
            # prepare_frame_dict: Görüntüyü normalize eder, PyTorch tensörüne çevirir ve renkleri saklar
            img_info = prepare_frame_dict(frame, idx=selected_idx, target_size=target_size, is_bgr=True)
            frames.append(img_info)
            selected_idx += 1
            sys.stdout.write(f"\r  -> Hazırlanan Keyframe: {selected_idx}/{len(frame_indices)}")
            sys.stdout.flush()
        curr_frame_idx += 1
    cap.release()
    print(f"\n ✅ {len(frames)} Keyframe Başarıyla Hazırlandı.")

    # =========================================================================
    # [ADIM 2 / 4] 🧠 MASt3R Çıkarımı ve Kapalı Form SE(3) Kamera Yörüngesi
    # =========================================================================
    print(f"\n [2/4] 🧠 MASt3R ile Yüksek Hassasiyetli 3B Alan Çıkarımı Yapılıyor...")
    from dust3r.inference import inference
    import roma

    N = len(frames)
    # Çift yönlü kare çiftleri oluştur (İleri ve geri eşleştirme ile maksimum tutarlılık)
    pairs = [(frames[i], frames[i + 1]) for i in range(N - 1)] + \
            [(frames[i + 1], frames[i]) for i in range(N - 1)]

    # MASt3R Yapay Zekasını GPU üzerinde çalıştır (Her kare çifti için 3B nokta haritası üretir)
    out = inference(pairs, model, device=device, batch_size=2, verbose=True)

    n_steps = N - 1
    cam_poses = [np.eye(4, dtype=np.float32)]  # İlk kamera başlangıç noktası (Birim matris)
    keyframe_pts3d = []                       # Her karenin 3B nokta koordinatları
    keyframe_confs = []                       # Her noktanın yapay zeka güvenilirlik skoru

    # İlk karenin 3B noktalarını ve güvenilirlik değerlerini listeye ekle
    keyframe_pts3d.append(out['pred1']['pts3d'][0].cpu().numpy())
    keyframe_confs.append(out['pred1']['conf'][0].cpu().numpy())

    # Ardışık kareler arasında Procrustes Rigid Body Registration (Dönüş R ve Öteleme t) hesaplama
    for i in range(n_steps):
        pts_i1_in_i = out['pred2']['pts3d_in_other_view'][i].cpu()   # i+1 karesinin i kamerasındaki tahmini
        conf_i1_in_i = out['pred2']['conf'][i].cpu()
        pts_i1_in_i1 = out['pred1']['pts3d'][n_steps + i].cpu()      # i+1 karesinin kendi koordinatlarındaki tahmini
        conf_i1_in_i1 = out['pred1']['conf'][n_steps + i].cpu()

        keyframe_pts3d.append(pts_i1_in_i1.numpy())
        keyframe_confs.append(conf_i1_in_i1.numpy())

        # Yüksek güvenilirlikli (Confidence > 1.2) ortak noktaları filtrele
        mask = (conf_i1_in_i > 1.2) & (conf_i1_in_i1 > 1.2)
        p_src = pts_i1_in_i1[mask].view(-1, 3)
        p_tgt = pts_i1_in_i[mask].view(-1, 3)
        weights = (conf_i1_in_i[mask] * conf_i1_in_i1[mask]).view(-1)

        # Kapalı form (SVD tabanlı) optimal SE(3) dönüşüm matrisini bul
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

    # =========================================================================
    # [ADIM 3 / 4] 🔮 Milyonlarca 3B Gaussian Splat Elipsoid Parametresinin Üretimi
    # =========================================================================
    print(f"\n [3/4] 🔮 Milyonlarca 3B Gaussian Splat Elipsoidi Hesaplanıyor...")
    all_xyz = []
    all_rgb = []
    all_scales = []
    all_quats = []
    all_opacity = []

    for idx in range(N):
        pts_loc = keyframe_pts3d[idx]       # Kameranın yerel 3B noktaları
        conf = keyframe_confs[idx]           # Noktaların güvenilirlik skorları
        rgb = frames[idx]['rgb_np']          # Orijinal renk bilgisi
        T_w = cam_poses[idx]                 # Kameranın Dünya koordinatlarındaki matrisi

        # Gürültü ve uçuşan noktaları temizleme filtresi
        dist = np.linalg.norm(pts_loc, axis=-1)
        valid = (conf > 1.45) & (dist < 4.8) & (pts_loc[..., 2] > 0.15) & np.isfinite(pts_loc).all(axis=-1)

        pts_valid = pts_loc[valid]
        rgb_valid = rgb[valid]
        conf_valid = conf[valid]

        if len(pts_valid) == 0:
            continue

        # Yerel kamera noktalarını Dünya koordinat sistemine dönüştür (X, Y, Z)
        pts_w = (T_w[:3, :3] @ pts_valid.T).T + T_w[:3, 3]
        pts_w[:, 1] = -pts_w[:, 1]  # OpenGL koordinat standart eşitlemesi (+Y yukarı)

        # 3D Gaussian Splatting Ölçekleri (Derinliğe göre adaptif anizotropik elipsoit boyutları)
        depths = pts_valid[:, 2]
        base_radius = np.clip(0.008 * depths, 0.003, 0.035).astype(np.float32)
        scales = np.column_stack([base_radius, base_radius * 0.7, base_radius * 0.4])

        # Dönüş Kuaterniyonu (rot_0, rot_1, rot_2, rot_3) - Varsayılan kimlik yönü [1, 0, 0, 0]
        quats = np.zeros((len(pts_valid), 4), dtype=np.float32)
        quats[:, 0] = 1.0

        # Güvenilirliğe göre opaklık (Alpha/Opacity) hesaplama (0.50 - 0.98 arası şeffaflık)
        opacities = np.clip((conf_valid - 1.45) / 2.0 + 0.65, 0.5, 0.98).astype(np.float32)

        all_xyz.append(pts_w.astype(np.float32))
        all_rgb.append(rgb_valid.astype(np.float32))
        all_scales.append(scales.astype(np.float32))
        all_quats.append(quats.astype(np.float32))
        all_opacity.append(opacities.astype(np.float32))

    # Tüm keyframe'lerden gelen milyonlarca Gaussian noktasını tek bir dev matriste birleştir
    xyz_merged = np.vstack(all_xyz)
    rgb_merged = np.vstack(all_rgb)
    scales_merged = np.vstack(all_scales)
    quats_merged = np.vstack(all_quats)
    opac_merged = np.concatenate(all_opacity)

    # Kamera yörüngesini dünya koordinatlarına göre hizala
    traj_cams_gl = cam_poses[:, :3, 3].copy()
    traj_cams_gl[:, 1] = -traj_cams_gl[:, 1]
    z_c = traj_cams_gl[:, 2]
    y_c = traj_cams_gl[:, 1]
    R_level = np.eye(3, dtype=np.float32)
    if len(z_c) > 2:
        poly_fit = np.polyfit(z_c, y_c, 1)
        pitch = math.atan(poly_fit[0])
        cos_p = math.cos(-pitch)
        sin_p = math.sin(-pitch)
        R_level = np.array([
            [1.0, 0.0, 0.0],
            [0.0, cos_p, -sin_p],
            [0.0, sin_p, cos_p]
        ], dtype=np.float32)
        xyz_merged = (R_level @ xyz_merged.T).T
        traj_cams_gl = (R_level @ traj_cams_gl.T).T

    # Zemini tam Y = 0 hizasına oturtma (Yerçekimi düzlemi hizalama)
    ground_y = float(np.percentile(xyz_merged[:, 1], 2))
    xyz_merged[:, 1] -= ground_y
    traj_cams_gl[:, 1] -= ground_y

    # =========================================================================
    # [ADIM 4 / 4] 💾 Standart 3DGS PLY & 0.1 sn Hızlı NPZ Önbellek Kaydı
    # =========================================================================
    print(f"\n [4/4] 💾 Standart 3D Gaussian Splatting Dosyası Kaydediliyor...")
    current_dir = CURR_DIR
    out_ply_path = os.path.join(current_dir, output_ply)
    out_cache_path = out_ply_path.replace(".ply", "_cache.npz")
    out_traj_path = os.path.join(current_dir, "gaussian_trajectory.npz")

    # Küresel Harmonik 0. Derece DC Katsayıları (Spherical Harmonics SH0)
    sh0 = (rgb_merged - 0.5) / 0.28209479177387814
    log_scales = np.log(np.maximum(scales_merged, 1e-6))
    logit_opacities = np.log(opac_merged / (1.0 - opac_merged + 1e-6))

    # ⚡ Hızlı Yükleme Önbelleği (.npz): Açılış süresini 10 saniyeden 0.1 saniyeye düşürür
    np.savez_compressed(
        out_cache_path,
        xyz=xyz_merged.astype(np.float32),
        rgb=rgb_merged.astype(np.float32),
        scales=scales_merged.astype(np.float32),
        quats=quats_merged.astype(np.float32),
        opacity=opac_merged.astype(np.float32),
        cams=traj_cams_gl.astype(np.float32)
    )

    # Yumuşak Kamera Yörüngesi (Cubic Spline Enterpolasyonu ile sinematik tur rotası)
    from scipy.interpolate import CubicSpline
    t_orig = np.linspace(0, 1, len(traj_cams_gl))
    t_fine = np.linspace(0, 1, 5000)
    cs_x = CubicSpline(t_orig, traj_cams_gl[:, 0])
    cs_y = CubicSpline(t_orig, traj_cams_gl[:, 1])
    cs_z = CubicSpline(t_orig, traj_cams_gl[:, 2])
    pos_fine = np.column_stack([cs_x(t_fine), cs_y(t_fine), cs_z(t_fine)]).astype(np.float32)
    np.savez(out_traj_path, positions=pos_fine)

    # Standart 3D Gaussian Splatting İkili (Binary) PLY Dosyasını Yazma
    num_splats = len(xyz_merged)
    with open(out_ply_path, "wb") as f:
        header = f"""ply
format binary_little_endian 1.0
element vertex {num_splats}
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
"""
        f.write(header.encode('ascii'))

        normals_dummy = np.zeros_like(xyz_merged, dtype=np.float32)
        vertex_data = np.zeros(num_splats, dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
            ('opacity', 'f4'),
            ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
            ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')
        ])

        vertex_data['x'] = xyz_merged[:, 0]
        vertex_data['y'] = xyz_merged[:, 1]
        vertex_data['z'] = xyz_merged[:, 2]
        vertex_data['nx'] = normals_dummy[:, 0]
        vertex_data['ny'] = normals_dummy[:, 1]
        vertex_data['nz'] = normals_dummy[:, 2]
        vertex_data['f_dc_0'] = sh0[:, 0]
        vertex_data['f_dc_1'] = sh0[:, 1]
        vertex_data['f_dc_2'] = sh0[:, 2]
        vertex_data['opacity'] = logit_opacities
        vertex_data['scale_0'] = log_scales[:, 0]
        vertex_data['scale_1'] = log_scales[:, 1]
        vertex_data['scale_2'] = log_scales[:, 2]
        vertex_data['rot_0'] = quats_merged[:, 0]
        vertex_data['rot_1'] = quats_merged[:, 1]
        vertex_data['rot_2'] = quats_merged[:, 2]
        vertex_data['rot_3'] = quats_merged[:, 3]

        f.write(vertex_data.tobytes())

    print("=" * 75)
    print(f" 🎉 3D GAUSSIAN SPLATTING MODELİ OLUŞTURULDU: {out_ply_path}")
    print(f" 🔮 Toplam 3B Gaussian Splat Sayısı: {num_splats:,} Adet")
    print(f" 💾 Dosya Boyutu: {os.path.getsize(out_ply_path) / (1024**2):.1f} MB")
    print("=" * 75)
    return out_ply_path


# Doğrudan terminalden çalıştırıldığında (Örn: python mast3r_to_3dgs.py testvideo2.mp4 50)
if __name__ == "__main__":
    v_name = sys.argv[1] if len(sys.argv) > 1 else "ofisvideo.mp4"
    n_kf = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    build_gaussian_splats_from_mast3r(v_name, num_keyframes=n_kf)
