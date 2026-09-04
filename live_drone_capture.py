"""
========================================================================================
MASt3R-3DGS: Canlı Dron & Kamera Tarama ve Anında 3B Harita Çıkarıcı (live_drone_capture.py)
========================================================================================
Bu modül, Dron veya Canlı Kameradan (RTSP / USB / Web kamerası) gelen canlı yayını
ekranda gösterir. Kullanıcı [R] veya [SPACE] tuşuna bastığında kayıt alır ve kaydı
durdurduğu an otomatik olarak MASt3R 3DGS motorunu çalıştırıp 10 saniye içinde
kusursuz 3B harita görüntüleyicisini başlatır.

Kontroller:
  [R] veya [SPACE] : Taramayı Başlat / Durdur (Bitince otomatik 3B harita üretir)
  [Q] veya [ESC]   : Çıkış
========================================================================================
"""

import sys
import os
import time
import cv2
import numpy as np

# Çalışma dizinini sys.path'e ekle
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
if CURR_DIR not in sys.path:
    sys.path.insert(0, CURR_DIR)

# Windows konsolunda Türkçe karakterlerin düzgün görünmesini sağlama
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def run_drone_capture(camera_source=0, target_keyframes=45):
    """
    Canlı dron veya kamera akışını başlatır, kullanıcı taramayı bitirdiğinde
    otomatik olarak MASt3R yapay zekasını çalıştırıp 3B Gaussian Splatting modelini açar.
    
    Parametreler:
        camera_source    : Kamera indeksi (0, 1) veya RTSP URL'si ("rtsp://192.168.1.100:8554/stream")
        target_keyframes : 3B harita çıkarılırken videodan seçilecek anahtar kare sayısı
    """
    print("=" * 70)
    print(" 🚁 CANLI DRON / KAMERA 3B HARİTALAMA SİSTEMİ")
    print(f" 📹 Kaynak: {camera_source}")
    print(f" 🎯 Hedef Keyframe: {target_keyframes} Adet")
    print(" 💡 [R] veya [SPACE] : Taramayı Başlat / Bitir")
    print(" 💡 [Q] veya [ESC]   : Çıkış")
    print("=" * 70)

    # Kamera kaynağını sayıya çevirmeyi dene (USB Kamera için 0, 1 vs.)
    try:
        source = int(camera_source)
    except ValueError:
        source = camera_source

    # Windows'ta en kararlı backend olan Media Foundation (CAP_MSMF) veya varsayılan kullanılır
    if isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_MSMF)
        if not cap.isOpened():
            cap = cv2.VideoCapture(source)
    else:
        # RTSP / HTTP Canlı Video Akışı (Webots / Dron için 30 sn bağlantı bekleme döngüsü)
        print(f" ⏳ Canlı Video Yayınına Bağlanılıyor: {source}")
        print(" -> Webots/Dron başlatılıyor, lütfen bekleyin (Webots açılınca üstteki Play ▶️ tuşuna basın)...")
        cap = None
        t_start = time.time()
        while time.time() - t_start < 35:
            temp_cap = cv2.VideoCapture(source)
            if temp_cap.isOpened():
                ret, test_frame = temp_cap.read()
                if ret and test_frame is not None:
                    cap = temp_cap
                    print("\n ✅ Canlı Dron Kamera Akışına Başarıyla Bağlanıldı!")
                    break
                temp_cap.release()
            time.sleep(1.0)
            kalan = int(35 - (time.time() - t_start))
            print(f" -> Bağlantı bekleniyor... ({kalan} sn)", end='\r', flush=True)

    if cap is None or not cap.isOpened():
        print(f"\n❌ HATA: Kamera açılamadı veya canlı yayına bağlanılamadı: {camera_source}")
        print(" -> Webots simülasyonunun çalıştığından (Play ▶️) emin olun.")
        return


    recording = False      # Kayıt/tarama aktif mi?
    recorded_frames = []   # Taranan video karelerini bellekte tutan liste

    win_name = "CANLI DRON KAMERASI [R / SPACE: TARA | Q: CIKIS]"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
    print("\n ✅ Canlı görüntü açıldı! Taramak istediğinde [R] veya [SPACE]'e bas...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Kamera görüntüsü kesildi!")
            break

        display_frame = frame.copy()
        h, w = display_frame.shape[:2]

        # ---------------------------------------------------------------------
        # 2B DURUM VE BİLGİ PANELLERİ
        # ---------------------------------------------------------------------
        # Üst Bilgi Şeridi (Yarı saydam arka plan)
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), (15, 20, 30), -1)
        cv2.addWeighted(overlay, 0.75, display_frame, 0.25, 0, display_frame)

        if recording:
            # Taranan kareyi listeye ekle
            recorded_frames.append(frame.copy())
            # Ekrana Kırmızı "TARANIYOR" göstergesi çiz
            cv2.circle(display_frame, (30, 32), 12, (0, 0, 255), -1)
            cv2.putText(display_frame, f"TARANIYOR: {len(recorded_frames)} Kare Alindi",
                        (55, 38), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 255), 2)
            cv2.putText(display_frame, "Taramayi bitirmek icin tekrar [R] veya [SPACE]'e basin.",
                        (55, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 220, 240), 1)
        else:
            # Bekleme durumunda Yeşil nokta göster
            cv2.circle(display_frame, (30, 32), 10, (0, 255, 0), -1)
            cv2.putText(display_frame, "CANLI IZLEME (Taramak icin [R] tusuna bas)", 
                        (55, 38), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 0), 2)

        # Görüntüyü pencerede göster
        cv2.imshow(win_name, display_frame)

        # Klavyeden basılan tuşu dinle (1 milisaniye bekle)
        key = cv2.waitKey(1) & 0xFF

        # [R] veya [SPACE] tuşuna basıldığında:
        if key in (ord('r'), ord('R'), 32):
            if not recording:
                # 1. Basış: Taramayı başlat
                recording = True
                recorded_frames = []
                print("\n 🔴 [TARAMA BAŞLADI] Dron/Kamera ilerliyor, kareler hafızaya alınıyor...")
            else:
                # 2. Basış: Taramayı bitir ve anında 3B Harita Motorunu çalıştır!
                recording = False
                total_rec = len(recorded_frames)
                print(f"\n ⏹️ [TARAMA BİTTİ] Toplam {total_rec} kare alındı.")

                if total_rec < 15:
                    print(" ⚠️ UYARI: Kayıt çok kısa oldu (<15 kare). Biraz daha uzun tarama yapın.")
                else:
                    # Kaydedilen kareleri geçici bir MP4 videosuna dönüştür
                    temp_video_path = os.path.join(CURR_DIR, "temp_drone_scan.mp4")
                    print(f" 💾 Video hazırlanıyor: {temp_video_path}")
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(temp_video_path, fourcc, 30, (w, h))
                    for f in recorded_frames:
                        writer.write(f)
                    writer.release()

                    # Canlı kamera penceresini kapat
                    cap.release()
                    cv2.destroyAllWindows()

                    print("\n" + "=" * 70)
                    print(" ⚡ MASt3R YAPAY ZEKASI ÇALIŞIYOR: 3D GAUSSIAN SPLAT HARİTASI ÜRETİLİYOR...")
                    print("=" * 70)

                    # MASt3R motorunu çağırıp 10 saniyede 3B modeli üretiyoruz
                    from mast3r_to_3dgs import build_gaussian_splats_from_mast3r
                    out_ply = build_gaussian_splats_from_mast3r(
                        video_file=temp_video_path,
                        output_ply="drone_scene.ply",
                        num_keyframes=min(target_keyframes, total_rec),
                        target_size=512
                    )

                    # Model üretilince 144+ FPS görüntüleyiciyi otomatik aç
                    if out_ply and os.path.exists(out_ply):
                        print("\n 🚀 3B MODEL HAZIR! Görüntüleyici açılıyor...")
                        import subprocess
                        renderer_path = os.path.join(CURR_DIR, "gaussian_renderer.py")
                        subprocess.run([sys.executable, renderer_path, "drone_scene.ply"])
                    return

        # [Q] veya [ESC] tuşuna basıldığında programdan çık
        elif key in (ord('q'), ord('Q'), 27):
            print("\n 👋 Canlı akış kapatıldı.")
            break

    # Döngü biterse kamerayı ve pencereleri serbest bırak
    cap.release()
    cv2.destroyAllWindows()


# Dosya doğrudan terminalden çalıştırıldığında burası başlar
if __name__ == "__main__":
    # Terminalden kamera numarası veya RTSP linki alabilir (varsayılan: 0)
    src = sys.argv[1] if len(sys.argv) > 1 else "0"
    kfs = int(sys.argv[2]) if len(sys.argv) > 2 else 45
    run_drone_capture(camera_source=src, target_keyframes=kfs)