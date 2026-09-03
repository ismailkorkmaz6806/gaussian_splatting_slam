"""
======================================================================================
 🎮 AIRSIM CANLI KLAVYE ILE DRON UCUS KONTROLCUSU & 3DGS KAYDEDICI
======================================================================================
W/A/S/D/SPACE tuslariyla dronu canli ucurun, [R] ile 3B harita kaydini baslatin!
"""

import time
import os
import sys
import numpy as np
import cv2
from airsim_client import AirSimRPCClient

def run_keyboard_drone_flight():
    client = AirSimRPCClient()
    print("\n -> AirSim baglantisi kuruluyor...")
    
    # Baglanana kadar 15 saniye dene
    connected = False
    for i in range(15):
        if client.connect():
            connected = True
            break
        print(f"  ... Simülatörün yüklenmesi bekleniyor ({i+1}/15)...")
        time.sleep(1.0)

    if not connected:
        print("\n [X] AirSim'e baglanilamadi! Lutfen once Blocks.exe dunyasini baslatin.")
        return

    # API ve Arm Kontrolunu Al
    client.enable_api_control()
    client.arm_disarm(True)
    client.takeoff(3.0)
    print("\n [OK] Dron Havalandi! Kontroller Aktif.")

    # Kayit Durumu
    recording = False
    video_writer = None
    output_video = "airsim_manual_flight.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    rec_frames = 0

    # Ucus Hizlari
    vx, vy, vz, vyaw = 0.0, 0.0, 0.0, 0.0
    speed = 2.0

    cv2.namedWindow("AirSim Dron Canli Kamera & Ucus Kontrolu", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AirSim Dron Canli Kamera & Ucus Kontrolu", 1024, 576)

    print("\n=============================================================")
    print(" 🎮 DRON KLAVYE KONTROLLER:")
    print("   [W / S]       : Ileri / Geri Uc")
    print("   [A / D]       : Sola / Saga Uc")
    print("   [SPACE / C]   : Yukari Yuksel / Asagi In")
    print("   [Q / E]       : Sola / Saga Don (Yaw)")
    print("   [R]           : 3DGS Video Kaydini Baslat / Durdur")
    print("   [ESC]         : Ucusu Bitir ve Inis Yap")
    print("=============================================================\n")

    running = True
    while running:
        # Kameradan canli goruntu al
        img = client.get_camera_image_bgr("front_center")
        if img is None:
            img = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Kayit devam ediyorsa kareyi MP4'e yaz
        if recording and video_writer is not None:
            video_writer.write(img)
            rec_frames += 1

        # HUD Bilgilerini Ciz
        disp_img = img.copy()
        h, w = disp_img.shape[:2]

        # Ust Bilgi Seridi
        cv2.rectangle(disp_img, (0, 0), (w, 50), (15, 20, 30), -1)
        status_txt = "🔴 KAYIT DEVAM EDIYOR (3DGS)" if recording else "🟢 MANUEL UCUS MODU"
        cv2.putText(disp_img, status_txt, (20, 34), cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 0, 255) if recording else (0, 255, 120), 2)
        cv2.putText(disp_img, "[R]: Kayit Baslat/Bitir | [W/A/S/D/SPACE]: Uc | [ESC]: Cikis", (w - 680, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 230, 245), 1)

        if recording:
            cv2.putText(disp_img, f"Kare: {rec_frames}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("AirSim Dron Canli Kamera & Ucus Kontrolu", disp_img)

        # Klavye Olaylarini Oku
        key = cv2.waitKey(20) & 0xFF

        target_vx, target_vy, target_vz, target_yaw = 0.0, 0.0, 0.0, 0.0

        if key in (ord('w'), ord('W')): target_vx = speed
        elif key in (ord('s'), ord('S')): target_vx = -speed
        
        if key in (ord('a'), ord('A')): target_vy = -speed
        elif key in (ord('d'), ord('D')): target_vy = speed

        if key == 32: # SPACE
            target_vz = -speed # Unreal z ekseninde -z yukaridir
        elif key in (ord('c'), ord('C')):
            target_vz = speed

        if key in (ord('q'), ord('Q')): target_yaw = -30.0
        elif key in (ord('e'), ord('E')): target_yaw = 30.0

        # [R] Tuşu: Kaydı Başlat / Bitir
        if key in (ord('r'), ord('R')):
            recording = not recording
            if recording:
                rec_frames = 0
                h_i, w_i = img.shape[:2]
                video_writer = cv2.VideoWriter(output_video, fourcc, 25, (w_i, h_i))
                print("\n 🔴 3DGS Video Kaydi Baslatildi! Dronu koridorda yavasca ucurun...")
            else:
                if video_writer:
                    video_writer.release()
                    video_writer = None
                print(f"\n 💾 Video Kaydedildi: {output_video} ({rec_frames} Kare)")
                print(" 🚀 3DGS Harita Motoru Baslatiliyor...")
                cv2.destroyAllWindows()
                client.land()
                client.close()

                # MASt3R 3DGS Motorunu Çalıştır
                import mast3r_to_3dgs
                out_ply = mast3r_to_3dgs.build_gaussian_splats_from_mast3r(output_video, 50)
                if out_ply and os.path.exists(out_ply):
                    import subprocess
                    subprocess.run([sys.executable, "gaussian_renderer.py", out_ply])
                return

        # [ESC] Tuşu: Çıkış
        elif key == 27:
            running = False

        # Dronu hareket ettir (moveToPosition/Velocity)
        if any([target_vx, target_vy, target_vz]):
            client.call("moveByVelocityBodyFrame", target_vx, target_vy, target_vz, 0.2, 0, [False, 0.0], "")

    cv2.destroyAllWindows()
    if video_writer:
        video_writer.release()
    client.land()
    client.arm_disarm(False)
    client.close()
    print("\n [OK] Ucus Tamamlandi.")

if __name__ == "__main__":
    run_keyboard_drone_flight()
