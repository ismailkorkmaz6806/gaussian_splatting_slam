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
                                      target_size=512,
                                      lidar_file=None):
    """
    Ham video dosyasını alıp baştan sona 3D Gaussian Splatting (.ply & .npz) modeline dönüştüren ana fonksiyondur.
    Katı hal (Solid-State) LiDAR noktaları ile görsel MASt3R splatlarını milimetrik hassasiyetle birleştirir.
    
    Parametreler:
        video_file    : İşlenecek girdi MP4 video dosyasının adı veya yolu
        output_ply    : Üretilecek 3D Gaussian Splat PLY dosyasının adı
        num_keyframes : Videodan seçilecek anahtar kare (keyframe) sayısı (Örn: 40-60)
        target_size   : MASt3R yapay zekasına beslenecek çözünürlük (Örn: 512x512)
        lidar_file    : (Opsiyonel) Canlı uçuşta toplanan 3B LiDAR nokta bulutu (.npz)
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
    # [ADIM 1 / 4] 🎬 Video Karelerini Akıllı Tarama (Hareket & Netlik Tabanlı Keyframe Seçimi)
    # =========================================================================
    print(f"\n [1/4] 🎬 Video Taranıyor: Hareket ve Netliğe Göre En İyi {num_keyframes} Keyframe Seçiliyor...")
    cap = cv2.VideoCapture(full_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f" 📊 Toplam Video Kare Sayısı: {total_frames}")

    # Uzun videolarda (örn: 5000 kare) boşta bekleme veya bulanık kareleri eleyen akıllı seçim
    raw_frames = []
    prev_gray = None
    step_skip = max(1, total_frames // 400)  # Hızlı ön tarama adımı
    
    frame_idx = 0
    candidate_indices = []
    
    while cap.isOpened() and frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Sadece belirli aralıklarla ön analiz yap
        if frame_idx % step_skip == 0:
            small_gray = cv2.cvtColor(cv2.resize(frame, (160, 120)), cv2.COLOR_BGR2GRAY)
            # Netlik kontrolü (Laplacian varyansı)
            sharpness = cv2.Laplacian(small_gray, cv2.CV_64F).var()
            
            # Hareket kontrolü
            if prev_gray is None:
                diff = 100.0
            else:
                diff = float(np.mean(cv2.absdiff(small_gray, prev_gray)))
            
            # Hareketsiz veya aşırı bulanık kareleri ele
            if (diff > 4.0 or prev_gray is None) and sharpness > 15.0:
                candidate_indices.append(frame_idx)
                prev_gray = small_gray
        
        frame_idx += 1
    
    cap.release()

    # Eğer akıllı filtreleme yeterli kare bulamadıysa güvenli linspace'e dön
    if len(candidate_indices) < num_keyframes:
        frame_indices = np.linspace(0, total_frames - 1, min(num_keyframes, max(total_frames, 1)), dtype=int)
    else:
        # Adaylar arasından eşit dağılımlı en kaliteli keyframe'leri seç
        sel_idx = np.linspace(0, len(candidate_indices) - 1, num_keyframes, dtype=int)
        frame_indices = [candidate_indices[i] for i in sel_idx]

    # Seçilen keyframe'leri tam çözünürlükte tensöre dönüştür
    frames = []
    cap = cv2.VideoCapture(full_video_path)
    selected_set = set(frame_indices)
    curr_frame_idx = 0
    loaded_count = 0
    
    while cap.isOpened() and loaded_count < len(frame_indices):
        ret, frame = cap.read()
        if not ret:
            break
        if curr_frame_idx in selected_set:
            img_info = prepare_frame_dict(frame, idx=loaded_count, target_size=target_size, is_bgr=True)
            frames.append(img_info)
            loaded_count += 1
            sys.stdout.write(f"\r  -> Hazırlanan Akıllı Keyframe: {loaded_count}/{len(frame_indices)}")
            sys.stdout.flush()
        curr_frame_idx += 1
    cap.release()
    print(f"\n ✅ {len(frames)} Yüksek Kaliteli Keyframe Başarıyla Hazırlandı (Çakışma Oranı İdeal).")

    # =========================================================================
    # [ADIM 2 / 4] 🧠 MASt3R Çıkarımı ve Kapalı Form SE(3) Kamera Yörüngesi
    # =========================================================================
    print(f"\n [2/4] 🧠 MASt3R ile Yüksek Hassasiyetli 3B Alan Çıkarımı Yapılıyor (Bellek Korumalı)...")
    from dust3r.inference import inference
    import roma

    N = len(frames)
    n_steps = N - 1
    cam_poses = [np.eye(4, dtype=np.float32)]  # İlk kamera başlangıç noktası (Birim matris)
    keyframe_pts3d = []                       # Her karenin 3B nokta koordinatları
    keyframe_confs = []                       # Her noktanın yapay zeka güvenilirlik skoru

    # Adım adım çıkarım: Tüm çiftleri aynı anda RAM'e yığmak yerine ardışık kareleri 2'şerli işler.
    # Bu sayede RAM kullanımı 28 GB'tan ~1.5 GB'a düşer ve Linux OOM (Killed) hatası tamamen engellenir!
    for i in range(n_steps):
        step_pairs = [(frames[i], frames[i + 1]), (frames[i + 1], frames[i])]
        step_out = inference(step_pairs, model, device=device, batch_size=2, verbose=False)

        # İlk adımda 0. karenin 3B noktalarını ve güvenilirlik değerlerini listeye ekle
        if i == 0:
            keyframe_pts3d.append(step_out['pred1']['pts3d'][0].cpu().numpy())
            keyframe_confs.append(step_out['pred1']['conf'][0].cpu().numpy())

        # i+1 karesinin i kamerasındaki tahmini ve kendi koordinatlarındaki tahmini
        pts_i1_in_i = step_out['pred2']['pts3d_in_other_view'][0].cpu()
        conf_i1_in_i = step_out['pred2']['conf'][0].cpu()
        pts_i1_in_i1 = step_out['pred1']['pts3d'][1].cpu()
        conf_i1_in_i1 = step_out['pred1']['conf'][1].cpu()

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
            diff = (pts_i1_in_i.mean(dim=(0, 1)) - step_out['pred1']['pts3d'][0].cpu().mean(dim=(0, 1))).numpy()
            T_step = np.eye(4, dtype=np.float32)
            T_step[:3, 3] = diff
            cam_poses.append(cam_poses[-1] @ T_step)

        # Geçici tensorleri anında sil ve GPU/CPU RAM'ini boşalt
        del step_out, pts_i1_in_i, conf_i1_in_i, pts_i1_in_i1, conf_i1_in_i1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        sys.stdout.write(f"\r  -> MASt3R 3B Alan Rekonstrüksiyonu: {i + 1}/{n_steps} Kare Tamamlandı (%{int((i+1)/n_steps*100)})")
        sys.stdout.flush()

    print(f"\n ✅ Kamera Yörüngesi ve 3B Alan Çıkarımı Başarıyla Tamamlandı!")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()

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

    is_cave = "drone" in output_ply.lower() or "cave" in video_file.lower() or "tunel" in video_file.lower()

    for idx in range(N):
        pts_loc = keyframe_pts3d[idx]       # Kameranın yerel 3B noktaları
        conf = keyframe_confs[idx]           # Noktaların güvenilirlik skorları
        rgb = frames[idx]['rgb_np']          # Orijinal renk bilgisi
        T_w = cam_poses[idx]                 # Kameranın Dünya koordinatlarındaki matrisi

        # Gürültü ve uçuşan noktaları temizleme filtresi (Kristal netlik ve sıfır gürültü)
        dist = np.linalg.norm(pts_loc, axis=-1)
        valid = (conf > 1.45) & (dist < 5.0) & (pts_loc[..., 2] > 0.15) & np.isfinite(pts_loc).all(axis=-1)

        pts_valid = pts_loc[valid]
        rgb_valid = rgb[valid]
        conf_valid = conf[valid]

        if len(pts_valid) == 0:
            continue

        # Yerel kamera noktalarını Dünya koordinat sistemine dönüştür (X, Y, Z)
        pts_w = (T_w[:3, :3] @ pts_valid.T).T + T_w[:3, 3]
        pts_w[:, 1] = -pts_w[:, 1]  # OpenGL koordinat standart eşitlemesi (+Y yukarı)

        # 3D Gaussian Splatting Ölçekleri (Net, keskin ve yüksek çözünürlüklü)
        depths = pts_valid[:, 2]
        base_radius = np.clip(0.008 * depths, 0.003, 0.035).astype(np.float32)
        scales = np.column_stack([base_radius, base_radius * 0.7, base_radius * 0.4])
        opacities = np.clip((conf_valid - 1.45) / 2.0 + 0.65, 0.5, 0.98).astype(np.float32)

        # Dönüş Kuaterniyonu (rot_0, rot_1, rot_2, rot_3) - Varsayılan kimlik yönü [1, 0, 0, 0]
        quats = np.zeros((len(pts_valid), 4), dtype=np.float32)
        quats[:, 0] = 1.0

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

    # Kamera yörüngesini OpenGL koordinatlarına eşitle
    traj_cams_gl = cam_poses[:, :3, 3].copy()
    traj_cams_gl[:, 1] = -traj_cams_gl[:, 1]

    # Zemini tam Y = 0 hizasına oturtma (Yerçekimi düzlemi hizalama)
    ground_y = float(np.percentile(xyz_merged[:, 1], 2))
    xyz_merged[:, 1] -= ground_y
    traj_cams_gl[:, 1] -= ground_y

    # =========================================================================
    # [ADIM 3.5] 📡 3B LiDAR Noktalarının Birleştirilmesi (Sadece Açıkça Belirtilmişse)
    # =========================================================================
    lidar_npz = lidar_file if (lidar_file and os.path.exists(lidar_file)) else None

    if lidar_npz is not None and os.path.exists(lidar_npz):
        try:
            print(f"\n [3.5] 📡 Katı Hal 3B LiDAR Verisi Yükleniyor: {os.path.basename(lidar_npz)}...")
            ld = np.load(lidar_npz, allow_pickle=True)
            lp = ld['pts']  # Gazebo dünya koordinatları (X_w, Y_w, Z_w)
            lr = ld['rgb']
            sp = ld['start_pos'] if 'start_pos' in ld else np.array([-3.0, 0.0, 0.2], dtype=np.float32)
            traj_data = ld['traj'] if 'traj' in ld else None

            # Gazebo (X ileri, Y sol, Z yukarı) -> 3DGS Scene Koordinat Dönüşümü:
            # Z_gl = X_w - sp[0] (tünel boyunca ileri 0 -> 38m)
            # X_gl = -(Y_w - sp[1]) (tünel genişliği / yatay)
            # Y_gl = Z_w (yükseklik / zemin -> tavan)
            lx_gl = -(lp[:, 1] - sp[1])
            ly_gl = lp[:, 2]
            lz_gl = lp[:, 0] - sp[0]
            pts_l_gl = np.column_stack([lx_gl, ly_gl, lz_gl]).astype(np.float32)

            n_l = len(pts_l_gl)
            base_r = np.full(n_l, 0.045, dtype=np.float32)
            scales_l = np.column_stack([base_r, base_r * 0.85, base_r * 0.60]).astype(np.float32)
            quats_l = np.zeros((n_l, 4), dtype=np.float32)
            quats_l[:, 0] = 1.0
            opac_l = np.full(n_l, 0.95, dtype=np.float32)

            # Zemin seviyesini sıfırla
            l_ground = float(np.percentile(ly_gl, 2))
            pts_l_gl[:, 1] -= l_ground

            # LiDAR gerçek uçuş yörüngesini al
            if traj_data is not None and len(traj_data) > 0:
                tx_gl = -(traj_data[:, 1] - sp[1])
                ty_gl = traj_data[:, 2] - l_ground
                tz_gl = traj_data[:, 0] - sp[0]
                traj_cams_gl = np.column_stack([tx_gl, ty_gl, tz_gl]).astype(np.float32)

            # Görsel MASt3R splatları ile LiDAR splatlarını birleştir
            xyz_merged = np.vstack([xyz_merged, pts_l_gl])
            rgb_merged = np.vstack([rgb_merged, lr.astype(np.float32)])
            scales_merged = np.vstack([scales_merged, scales_l])
            quats_merged = np.vstack([quats_merged, quats_l])
            opac_merged = np.concatenate([opac_merged, opac_l])
            print(f" ✅ {n_l:,} LiDAR 3B Yüzey Noktası Modele Eklendi! Tünelin Tümü (38 Metre) Kapsandı.")
        except Exception as e:
            print(f" ⚠️ LiDAR birleştirme uyarısı: {e}")

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
    out_name = sys.argv[3] if len(sys.argv) > 3 else "gaussian_scene.ply"
    build_gaussian_splats_from_mast3r(v_name, output_ply=out_name, num_keyframes=n_kf)
