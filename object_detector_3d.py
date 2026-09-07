"""
MASt3R-SLAM: 3D Multi-Class Office Object & Person Detection Engine
--------------------------------------------------------------------
Bu modül, YOLOv8 (Ultralytics) derin öğrenme modelini MASt3R'ın 3B nokta
haritaları (pointmap) ile birleştirir.

İşlevler:
1. 2B görüntü karelerinde insan, sandalye, masa, monitör vb. nesneleri tespit eder.
2. 2B sınırlayıcı kutuların (bounding box) piksellerini MASt3R 3B dünya koordinatlarına eşler.
3. Aykırı değerleri (outlier) temizler, nesnenin 3B merkez noktasını (centroid)
   ve 3B sınırlayıcı kutusunu (min/max corner bounding box) hesaplar.
4. Zamansal ve uzamsal kümeleme ile mükerrer tespitleri tek bir 3B nesne altında birleştirir.
"""

import os
import sys
import numpy as np
import cv2
import torch
from ultralytics import YOLO

# Global singleton YOLO model referansı
_yolo_model = None

# ==============================================================================
# NESNE SINIFLARI VE GÖRSELLEŞTİRME YAPILANDIRMASI
# ==============================================================================
# COCO veri seti sınıf ID'leri, Türkçe etiketleri, renkleri ve kümeleme yarıçapları
OBJECT_CONFIG = {
    0:  {'name': 'İnsan',    'color': (1.0, 0.28, 0.38), 'cluster_dist': 1.2,  'icon': '👤'},
    56: {'name': 'Sandalye', 'color': (0.2, 0.8, 1.0),  'cluster_dist': 0.65, 'icon': '🪑'},
    57: {'name': 'Koltuk',   'color': (0.9, 0.5, 0.1),  'cluster_dist': 1.1,  'icon': '🛋️'},
    59: {'name': 'Bitki',    'color': (0.2, 0.9, 0.4),  'cluster_dist': 0.6,  'icon': '🪴'},
    60: {'name': 'Masa',     'color': (1.0, 0.8, 0.2),  'cluster_dist': 1.1,  'icon': '🪵'},
    62: {'name': 'Monitör',  'color': (0.8, 0.3, 1.0),  'cluster_dist': 0.6,  'icon': '🖥️'},
    63: {'name': 'Laptop',   'color': (0.3, 0.9, 0.9),  'cluster_dist': 0.5,  'icon': '💻'},
}


# ==============================================================================
# YOLOv8 MODELİNİ YÜKLEME (SINGLETON)
# ==============================================================================
def get_detector():
    """
    YOLOv8 nesne algılama modelini yükler veya önbellekten döner.
    Yerel dizinde 'yolov8m.pt' ağırlığı arar; yoksa internetten otomatik indirir.
    """
    global _yolo_model
    if _yolo_model is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(base_dir, "yolov8m.pt")
        if not os.path.exists(weights_path):
            weights_path = "yolov8m.pt"
        print(" 🧠 YOLOv8m Çoklu Nesne Tespit Modeli Yükleniyor...")
        _yolo_model = YOLO(weights_path)
    return _yolo_model


# ==============================================================================
# 3B NESNE TESPİT VE UZAMSAL KÜMELEME MOTORU
# ==============================================================================
def extract_3d_objects(frames, keyframe_pts3d, cam_poses, keyframe_confs, conf_thr=0.35):
    """
    Tüm keyframe'lerdeki nesneleri 2B tespit eder ve MASt3R 3B pozları ile dünya koordinatlarına aktarır.
    
    Parametreler:
        frames (list): Keyframe sözlükleri listesi (içinde 'rgb_np' barındırır).
        keyframe_pts3d (list[np.ndarray]): Kamera koordinat sistemindeki [H, W, 3] nokta haritaları.
        cam_poses (list[np.ndarray]): Her keyframe'in [4, 4] kamera -> dünya dönüşüm matrisleri.
        keyframe_confs (list[np.ndarray]): [H, W] piksel güven haritaları.
        conf_thr (float): YOLOv8 minimum güven eşiği (varsayılan: 0.35).
    Dönüş:
        list[dict]: 3B merkez, boyut, güven skoru ve sınıf bilgilerini içeren tespit listesi.
    """
    model = get_detector()
    target_classes = list(OBJECT_CONFIG.keys())
    raw_detections = {cls_id: [] for cls_id in target_classes}
    num_frames = len(frames)

    print(f"\n [3D-DETECTION] 🔍 {num_frames} Keyframe Boyunca Ofis Nesneleri ve İnsanlar Taranıyor...")

    # -------------------------------------------------------------------------
    # ADIM 1: Her Keyframe İçin 2B Tespit ve 3B Dünya Koordinatlarına Projeksiyon
    # -------------------------------------------------------------------------
    for idx, frame_dict in enumerate(frames):
        # RGB dizisini OpenCV BGR formatına dönüştür
        rgb_img = (frame_dict['rgb_np'] * 255).astype(np.uint8)
        bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

        # YOLOv8 ile hedef sınıfları tara
        results = model(bgr_img, classes=target_classes, conf=conf_thr, verbose=False)[0]

        if len(results.boxes) == 0:
            continue

        pts_local = keyframe_pts3d[idx]       # [H, W, 3] yerel 3B noktalar
        conf_map = keyframe_confs[idx]        # [H, W] güven değerleri
        T_w = cam_poses[idx]                  # [4, 4] kamera dış parametre matrisi

        H_f, W_f, _ = pts_local.shape
        pts_flat = pts_local.reshape(-1, 3)
        # Kamera koordinatlarından Dünya koordinatlarına dönüşüm: P_world = R * P_local + T
        pts_world_flat = (T_w[:3, :3] @ pts_flat.T).T + T_w[:3, 3]
        pts_world = pts_world_flat.reshape(H_f, W_f, 3)

        # Tespit edilen her sınırlayıcı kutuyu işle
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in OBJECT_CONFIG:
                continue

            score = float(box.conf[0])
            #.cpu(): Bu koordinatlar PyTorch tensörü olarak GPU belleğindeyse (CUDA), 
            #veriyi işlemcinin (CPU) erişebileceği ana belleğe taşır.
            #astype() virgüllü değer üretmeyi sağlar
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            # Görüntü sınırlarına kırpma
            x1 = max(0, min(W_f - 1, x1))
            x2 = max(0, min(W_f - 1, x2))
            y1 = max(0, min(H_f - 1, y1))
            y2 = max(0, min(H_f - 1, y2))

            # Bounding box içindeki 3B noktaları ve güven değerlerini kes
            pts_crop = pts_world[y1:y2, x1:x2].reshape(-1, 3)
            conf_crop = conf_map[y1:y2, x1:x2].reshape(-1)

            # Güven değeri > 1.2 ve sonlu (NaN olmayan) geçerli noktaları filtrele
            valid = (conf_crop > 1.2) & np.isfinite(pts_crop).all(axis=-1)
            pts_valid = pts_crop[valid]

            if len(pts_valid) < 25:
                continue

            # İstatistiksel aykırı değer filtreleme (Yüzde 10-90 persentil arası çekirdek)
            dists = np.linalg.norm(pts_valid, axis=1)
            p10 = np.percentile(dists, 10)
            p90 = np.percentile(dists, 90)
            core_pts = pts_valid[(dists >= p10) & (dists <= p90)]

            if len(core_pts) < 15:
                continue

            # 3B Medyan Merkez ve Sınır Noktalarını Hesapla
            centroid = np.median(core_pts, axis=0)
            min_c = np.percentile(core_pts, 5, axis=0)
            max_c = np.percentile(core_pts, 95, axis=0)

            cfg = OBJECT_CONFIG[cls_id]
            raw_detections[cls_id].append({
                'cls_id': cls_id,
                'label': cfg['name'],
                'icon': cfg['icon'],
                'color': cfg['color'],
                'frame_idx': idx,
                'score': score,
                'centroid': centroid.astype(np.float32),
                'min_corner': min_c.astype(np.float32),
                'max_corner': max_c.astype(np.float32),
            })

    # -------------------------------------------------------------------------
    # ADIM 2: Uzamsal Kümeleme ve Mükerrer Tespitleri Birleştirme (Merging)
    # -------------------------------------------------------------------------
    fused_objects = []

    for cls_id, det_list in raw_detections.items():
        if not det_list:
            continue

        cfg = OBJECT_CONFIG[cls_id]
        c_dist = cfg['cluster_dist']
        merged = [False] * len(det_list)

        for i in range(len(det_list)):
            if merged[i]:
                continue

            item_i = det_list[i]
            cluster = [item_i]
            merged[i] = True

            # Birbirine cluster_dist mesafesinden yakın olan aynı sınıf tespitleri grupla
            for j in range(i + 1, len(det_list)):
                if merged[j]:
                    continue
                item_j = det_list[j]
                dist = np.linalg.norm(item_i['centroid'] - item_j['centroid'])
                if dist < c_dist:
                    cluster.append(item_j)
                    merged[j] = True

            # Küme içindeki en yüksek skoru ve ortalama 3B sınırları hesapla
            best_score = max(p['score'] for p in cluster)
            mean_centroid = np.mean([p['centroid'] for p in cluster], axis=0)
            mean_min = np.mean([p['min_corner'] for p in cluster], axis=0)
            mean_max = np.mean([p['max_corner'] for p in cluster], axis=0)

            fused_objects.append({
                'label': cfg['name'],
                'icon': cfg['icon'],
                'score': best_score,
                'centroid': mean_centroid.astype(np.float32),
                'min_corner': mean_min.astype(np.float32),
                'max_corner': mean_max.astype(np.float32),
                'color': cfg['color'],
                'count': len(cluster)
            })

    print(f" ✅ Toplam {len(fused_objects)} 3B Nesne/Mobilya Tespit Edildi!")
    for idx, obj in enumerate(fused_objects, 1):
        print(f"   -> {obj['icon']} {obj['label']} #{idx}: %{obj['score']*100:.0f} (3B Konum: X={obj['centroid'][0]:.2f}m, Z={obj['centroid'][2]:.2f}m)")

    return fused_objects

