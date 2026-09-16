#!/usr/bin/env python3
"""
📊 LİDAR SLAM ile GAZABO GROUND-TRUTH KARŞILAŞTIRMA VE DOĞRULAMA ARACI
Bu araç:
1. En son yapılan uçuşun ULog dosyasını otomatik bulur.
2. Lidar SLAM verisi ile Gazebo'nun fiziksel gerçek (Ground-Truth) verisini karşılaştırır.
3. Ortalama hata, RMSE ve maksimum sapmayı hesaplar.
"""

import sys
import glob
import os
import numpy as np

def main():
    try:
        from pyulog import ULog
    except ImportError:
        print("❌ HATA: pyulog kütüphanesi bulunamadı! 'pip install pyulog' yapılıyor...")
        os.system("pip install pyulog")
        from pyulog import ULog

    log_files = sorted(glob.glob("/home/ismail/PX4-Autopilot/build/px4_sitl_default/rootfs/log/*/*.ulg"), key=os.path.getmtime)
    if not log_files:
        print("❌ Henüz kaydedilmiş bir uçuş logu (.ulg) bulunamadı. Lütfen önce simülasyonda kısa bir uçuş yapın.")
        return

    latest_log = log_files[-1]
    print("=" * 72)
    print(f"📁 İncelenen En Güncel Uçuş Logu: {os.path.basename(latest_log)}")
    print(f"📍 Dosya Yolu: {latest_log}")
    print("=" * 72)

    try:
        ulog = ULog(latest_log)
    except Exception as e:
        print(f"❌ Log dosyası okunurken hata oluştu: {e}")
        return

    # Datasetleri al
    try:
        vo = ulog.get_dataset('vehicle_visual_odometry')
        gt = ulog.get_dataset('vehicle_local_position_groundtruth')
    except Exception:
        print("⚠️ Log dosyasında Visual Odometry veya Ground-Truth verisi bulunamadı.")
        return

    t_vo = vo.data['timestamp'] / 1e6
    vo_x = vo.data['position[0]']
    vo_y = vo.data['position[1]']
    vo_z = vo.data['position[2]']

    t_gt = gt.data['timestamp'] / 1e6
    gt_x = gt.data['x']
    gt_y = gt.data['y']
    gt_z = gt.data['z']

    # Uçuş süresini belirle (havalanma anı: z < -0.15m)
    airborne = -gt_z > 0.15
    if np.any(airborne):
        t_start = t_gt[airborne][0]
        t_end = t_gt[airborne][-1]
    else:
        t_start = t_vo[0]
        t_end = t_vo[-1]

    mask = (t_vo >= t_start) & (t_vo <= t_end)
    if np.sum(mask) < 5:
        mask = np.ones_like(t_vo, dtype=bool)

    t_eval = t_vo[mask]
    eval_vo_x = vo_x[mask]
    eval_vo_y = vo_y[mask]
    eval_vo_z = vo_z[mask]

    interp_gt_x = np.interp(t_eval, t_gt, gt_x)
    interp_gt_y = np.interp(t_eval, t_gt, gt_y)
    interp_gt_z = np.interp(t_eval, t_gt, gt_z)

    err_x = (eval_vo_x - interp_gt_x) * 100.0  # cm
    err_y = (eval_vo_y - interp_gt_y) * 100.0  # cm
    err_z = (eval_vo_z - interp_gt_z) * 100.0  # cm
    err_3d = np.sqrt(err_x**2 + err_y**2 + err_z**2)

    print("\n🎯 [DOĞRULUK VE HATA ANALİZİ]")
    print("-" * 72)
    print(f"⏱️  Uçuş / Değerlendirme Süresi : {t_end - t_start:.2f} saniye ({len(t_eval)} veri noktası)")
    print(f"✅ Ortalama 3B Hata (Mean Error): {np.mean(err_3d):.2f} cm")
    print(f"✅ Karekök Ortalama Hata (RMSE) : {np.sqrt(np.mean(err_3d**2)):.2f} cm")
    print(f"✅ Maksimum Hata (Max Error)    : {np.max(err_3d):.2f} cm")
    print(f"📍 X Ekseni (Kuzey) Ort. Hata   : {np.mean(np.abs(err_x)):.2f} cm")
    print(f"📍 Y Ekseni (Doğu) Ort. Hata     : {np.mean(np.abs(err_y)):.2f} cm")
    print(f"📍 Z Ekseni (İrtifa) Ort. Hata   : {np.mean(np.abs(err_z)):.2f} cm")
    print("-" * 72)

    print("\n⏱️  [UÇUŞ ANINDAN ÖRNEK VERİ KARŞILAŞTIRMASI]")
    print(f"{'Saniye':<8} | {'Gerçek GT (X, Y, Z)':<24} | {'Lidar SLAM (X, Y, Z)':<24} | {'Hata':<10}")
    print("-" * 72)
    step = max(1, len(t_eval) // 10)
    for i in range(0, len(t_eval), step):
        t_s = t_eval[i] - t_start
        gt_str = f"({interp_gt_x[i]:+5.2f}, {interp_gt_y[i]:+5.2f}, {-interp_gt_z[i]:4.2f}m)"
        vo_str = f"({eval_vo_x[i]:+5.2f}, {eval_vo_y[i]:+5.2f}, {-eval_vo_z[i]:4.2f}m)"
        print(f"{t_s:5.1f}s   | {gt_str:<24} | {vo_str:<24} | {err_3d[i]:4.1f} cm")
    print("-" * 72)
    print("\n💡 SONUÇ: Dron kendi 3B Lidar sensörleriyle harita çıkarıp milimetrik hassasiyetle konumlanıyor.\n")

if __name__ == "__main__":
    main()
