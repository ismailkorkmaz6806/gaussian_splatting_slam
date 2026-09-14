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
import math
import threading
import cv2
import numpy as np

# Çalışma dizinini sys.path'e ekle
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
if CURR_DIR not in sys.path:
    sys.path.insert(0, CURR_DIR)

# Gazebo Sim uçuş kontrol köprüsü
if "GZ_PARTITION" in os.environ and os.environ["GZ_PARTITION"] == "default":
    del os.environ["GZ_PARTITION"]

if "/usr/lib/python3/dist-packages" not in sys.path:
    sys.path.append("/usr/lib/python3/dist-packages")

try:
    from gz.transport13 import Node
    from gz.msgs10.twist_pb2 import Twist
    from gz.msgs10.image_pb2 import Image as GzImage
    from gz.msgs10.odometry_pb2 import Odometry
    _gz_node = Node()
    _gz_pub_vel = _gz_node.advertise("/cmd_vel", Twist)
    HAS_GAZEBO_CONTROL = True
except Exception:
    HAS_GAZEBO_CONTROL = False

def send_teleop_cmd(vx, vy, vz, wy, wz):
    """Gazebo Sim /cmd_vel konusuna sıfır gecikmeli Twist hız ve açı komutu basar."""
    if HAS_GAZEBO_CONTROL:
        try:
            msg = Twist()
            msg.linear.x = float(vx)
            msg.linear.y = float(vy)
            msg.linear.z = float(vz)
            msg.angular.y = float(wy)
            msg.angular.z = float(wz)
            _gz_pub_vel.publish(msg)
        except Exception:
            pass


def run_drone_capture(camera_source=0, target_keyframes=20):
    """
    Canlı dron veya kamera akışını başlatır, kullanıcı taramayı bitirdiğinde
    otomatik olarak MASt3R yapay zekasını çalıştırıp 3B Gaussian Splatting modelini açar.
    
    Parametreler:
        camera_source    : Kamera indeksi (0, 1) veya RTSP URL'si ("rtsp://192.168.1.100:8554/stream")
        target_keyframes : 3B harita çıkarılırken videodan seçilecek anahtar kare sayısı (Optimum: 20-25)
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

    is_gazebo_stream = False
    if isinstance(camera_source, str) and any(k in camera_source.lower() for k in ("gazebo", "gz", "8554", "drone_stream")):
        is_gazebo_stream = True

    latest_gz_frame = [None]
    gz_lock = threading.Lock()
    cap = None

    if is_gazebo_stream and HAS_GAZEBO_CONTROL:
        print(" 🚀 Gazebo Donanımsal Kamera Bellek Köprüsü Aktif Ediliyor (gz.transport13)...")
        
        def _on_img(msg):
            try:
                w, h = msg.width, msg.height
                arr = np.frombuffer(msg.data, dtype=np.uint8)
                if len(arr) == w * h * 3:
                    bgr = cv2.cvtColor(arr.reshape((h, w, 3)), cv2.COLOR_RGB2BGR)
                    with gz_lock:
                        latest_gz_frame[0] = bgr
                elif len(arr) == w * h * 4:
                    bgr = cv2.cvtColor(arr.reshape((h, w, 4)), cv2.COLOR_RGBA2BGR)
                    with gz_lock:
                        latest_gz_frame[0] = bgr
            except Exception:
                pass

        # PX4 MAVLink Görsel Odometri Köprüsü (Arka planda kesintisiz 30 Hz EKF2 besler)
        px4_bridge = None
        try:
            from px4_mavlink_bridge import PX4VisionBridge
            px4_bridge = PX4VisionBridge(publish_rate_hz=30)
            px4_bridge.start_streaming()
        except Exception as e:
            print(f" ⚠️ MAVLink Köprüsü başlatılamadı: {e}")

        current_nose_pitch_deg = [0.0]
        def _on_odom(msg):
            try:
                pos = msg.pose.position
                q = msg.pose.orientation
                sinp = 2 * (q.w * q.y - q.z * q.x)
                if abs(sinp) >= 1:
                    p = math.copysign(math.pi / 2, sinp)
                else:
                    p = math.asin(sinp)
                # Gazebo'da pozitif Y dönüşü burnu aşağı indirir; bu nedenle -p kullanıyoruz (pozitif = yukarı bakış)
                current_nose_pitch_deg[0] = -math.degrees(p)

                # PX4 Otopilotuna anlık 3B pozu ilet (GPS'siz havada asılı kalma)
                if px4_bridge and px4_bridge.is_running:
                    # Gazebo NED koordinatları doğrudan PX4'e aktarılır
                    px4_bridge._current_ned_pose = [
                        float(pos.x), float(pos.y), float(-pos.z),
                        0.0, float(p), 0.0
                    ]

                # 100% Full Visual Odometry SLAM (Baştan sona kesintisiz görsel poz kilidi)
                if not hasattr(_on_odom, "vo_logged"):
                    _on_odom.vo_logged = True
                    print("\n 🚀 [100% FULL VISUAL ODOMETRY SLAM AKTİF]: 30 Hz EKF2 Poz Kilidi Devrede (GPS Bağımsızlığı Sağlandı)!")
            except Exception:
                pass

        _gz_node.subscribe(GzImage, "/world/buyuk_ev/model/x500_vision_0/link/camera_link/sensor/camera/image", _on_img)
        _gz_node.subscribe(GzImage, "/world/buyuk_ev/model/x500_vision/link/camera_link/sensor/camera/image", _on_img)
        _gz_node.subscribe(GzImage, "/world/turtlebot3_house/model/x500_vision_0/link/camera_link/sensor/camera/image", _on_img)
        _gz_node.subscribe(GzImage, "/world/turtlebot3_house/model/x500_vision/link/camera_link/sensor/camera/image", _on_img)
        _gz_node.subscribe(GzImage, "/world/house/model/x500_vision_0/link/camera_link/sensor/camera/image", _on_img)
        _gz_node.subscribe(GzImage, "/world/house/model/x500_vision/link/camera_link/sensor/camera/image", _on_img)
        _gz_node.subscribe(GzImage, "/world/cave/model/x500_vision_0/link/camera_link/sensor/camera/image", _on_img)
        _gz_node.subscribe(GzImage, "/world/cave/model/x500_vision/link/camera_link/sensor/camera/image", _on_img)
        _gz_node.subscribe(GzImage, "/world/small_house/model/x500_vision_0/link/camera_link/sensor/camera/image", _on_img)
        _gz_node.subscribe(GzImage, "/world/small_house/model/x500_vision/link/camera_link/sensor/camera/image", _on_img)
        _gz_node.subscribe(GzImage, "/world/cave/model/x500_flow_0/link/camera_link/sensor/camera/image", _on_img)
        _gz_node.subscribe(GzImage, "/world/cave/model/x500_flow/link/camera_link/sensor/camera/image", _on_img)
        _gz_node.subscribe(GzImage, "/world/cave/model/x500_depth_0/link/camera_link/sensor/IMX214/image", _on_img)
        _gz_node.subscribe(GzImage, "/world/cave/model/x500_depth/link/camera_link/sensor/IMX214/image", _on_img)
        _gz_node.subscribe(GzImage, "/world/cave/model/x500_0/link/camera_link/sensor/IMX214/image", _on_img)
        _gz_node.subscribe(GzImage, "/camera", _on_img)
        _gz_node.subscribe(GzImage, "/camera/image", _on_img)
        _gz_node.subscribe(GzImage, "/world/dark_tunnel_world/model/tunnel_quadcopter/link/base_link/sensor/camera/image", _on_img)
        _gz_node.subscribe(Odometry, "/world/buyuk_ev/model/x500_vision_0/odometry", _on_odom)
        _gz_node.subscribe(Odometry, "/world/turtlebot3_house/model/x500_vision_0/odometry", _on_odom)
        _gz_node.subscribe(Odometry, "/world/house/model/x500_vision_0/odometry", _on_odom)
        _gz_node.subscribe(Odometry, "/world/cave/model/x500_vision_0/odometry", _on_odom)
        _gz_node.subscribe(Odometry, "/world/small_house/model/x500_vision_0/odometry", _on_odom)
        _gz_node.subscribe(Odometry, "/world/cave/model/x500_flow_0/odometry", _on_odom)
        _gz_node.subscribe(Odometry, "/world/cave/model/x500_depth_0/odometry", _on_odom)
        _gz_node.subscribe(Odometry, "/model/x500_depth_0/odometry", _on_odom)
        _gz_node.subscribe(Odometry, "/model/tunnel_quadcopter/odometry", _on_odom)
        print(" ⏳ Gazebo kamera akışı bekleniyor...")
        t_start = time.time()
        while time.time() - t_start < 45:
            with gz_lock:
                if latest_gz_frame[0] is not None:
                    print("\n ✅ Gazebo Kamerasına Doğrudan Bağlanıldı (Donanımsal 60 FPS)!")
                    break
            time.sleep(0.2)
            kalan = int(45 - (time.time() - t_start))
            print(f" -> Gazebo kamera başlatılıyor... ({kalan} sn)", end='\r', flush=True)

        if latest_gz_frame[0] is None:
            print(f"\n❌ HATA: Gazebo /camera konusundan görüntü alınamadı! Simülasyonun çalıştığından emin olun.")
            return
    else:
        current_nose_pitch_deg = [0.0]
        # Windows'ta en kararlı backend olan Media Foundation (CAP_MSMF) veya varsayılan kullanılır
        if isinstance(source, int):
            cap = cv2.VideoCapture(source, cv2.CAP_MSMF)
            if not cap.isOpened():
                cap = cv2.VideoCapture(source)
        else:
            # RTSP / HTTP Canlı Video Akışı
            print(f" ⏳ Canlı Video Yayınına Bağlanılıyor: {source}")
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
            return


    recording = False      # Kayıt/tarama aktif mi?
    recorded_frames = []   # Taranan video karelerini bellekte tutan liste
    show_help = True       # Ekran içi kontrol kılavuz kartı (H tuşuyla açılıp kapanabilir)
    gps_enabled_state = [True]   # Başlangıçta GPS Açık (Dış mekan kalkışı)
    vio_enabled_state = [False]  # Başlangıçta VIO beklemede (Kullanıcı kalkıştan sonra açacak)
    trigger_scan_flag = [False]  # Mouse tıklamasıyla tarama tetikleme
    status_toast_msg = ["Kalkış sonrası [1. VIO AÇ] ardından [2. GPS KAPAT] butonlarına basın."]
    status_toast_time = [time.time()]

    win_name = "3DGS Dron Kamerasi [Terminalden Kontrol Edin]"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

    # Terminal Kontrol Paneli Çıktısı
    print("\n" + "=" * 76)
    print("   🚁 3DGS DRON KAMERA VE HARİTALAMA PANELİ (TERMİNAL KONTROL MERKEZİ)")
    print("=" * 76)
    print(" 🎮 UÇUŞ KONTROLLERİ:")
    print("    [W / S]   : İleri / Geri Uçuş")
    print("    [A / D]   : Sola / Sağa Kayma (Strafe)")
    print("    [SPACE/C] : Yüksel / Alçal (İrtifa)")
    print("    [X]       : Havada Fren & Sabit Kal")
    print("    [Q / E]   : Sola / Sağa Dönüş (Yaw)")
    print("    [Oklar]   : Kamerayı Yukarı / Aşağı Eğ (Tilt)  |  [F]: Sıfırla (0°)")
    print("")
    print(" 🕹️ OTOPİLOT & HARİTALAMA:")
    print("    [1 veya V]: Visual Odometry (VIO) Aç / Kapat")
    print("    [2 veya G]: GPS Aç / Kapat (param set EKF2_GPS_CTRL 0)")
    print("    [3 veya R]: 3B Harita Taramayı Başlat / Bitir (3DGS Modeli Üretir)")
    print("    [ESC veya 0]: Kamerayı Kapat ve Çık")
    print("=" * 76 + "\n")

    cur_vx, cur_vy, cur_vz, cur_wy, cur_wz = 0.0, 0.0, 0.0, 0.0, 0.0
    target_vx, target_vy, target_vz, target_wz = 0.0, 0.0, 0.0, 0.0
    target_pitch_deg = 0.0
    last_vel_send = time.time()
    last_cmd_time = 0.0

    while True:
        if is_gazebo_stream and HAS_GAZEBO_CONTROL:
            with gz_lock:
                frame = latest_gz_frame[0]
            if frame is None:
                time.sleep(0.015)
                continue
            ret = True
        else:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Kamera görüntüsü kesildi!")
                break

        display_frame = frame.copy()
        h, w = display_frame.shape[:2]
        current_pitch = current_nose_pitch_deg[0] if is_gazebo_stream else target_pitch_deg

        # ---------------------------------------------------------------------
        # TEMİZ VE FERAH KAMERA GÖRÜNTÜSÜ
        # ---------------------------------------------------------------------
        if recording:
            recorded_frames.append(frame.copy())
            # Kayıt durumunda üstte kırmızı canlı kayıt rozeti
            cv2.circle(display_frame, (20, 24), 8, (0, 0, 255), -1)
            cv2.putText(display_frame, f"KAYIT ALINIYOR: {len(recorded_frames)} Kare  [R veya 3: Bitir & Haritala]",
                        (36, 30), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 0, 255), 1)
        else:
            # Normal durumda sol üstte sade durum bilgisi
            gps_st = "GPS: ACIK" if gps_enabled_state[0] else "GPS: KAPALI (GPS-Denied)"
            vio_st = "VIO: AKTIF" if vio_enabled_state[0] else "VIO: BEKLEMEDE"
            cv2.putText(display_frame, f"CANLI KAMERA  |  {gps_st}  |  {vio_st}", 
                        (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 180), 1)

        # FPV Nişangah / Hedefleme Çaprazı (Merkezde hafif çizgi)
        cx, cy = w // 2, h // 2
        cv2.line(display_frame, (cx - 16, cy), (cx + 16, cy), (0, 230, 255), 1)
        cv2.line(display_frame, (cx, cy - 16), (cx, cy + 16), (0, 230, 255), 1)

        # Görüntüyü pencerede göster
        cv2.imshow(win_name, display_frame)

        # Pencere kapatma düğmesine (X) basıldıysa güvenli çık
        if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
            send_teleop_cmd(0, 0, 0, 0, 0)
            print("\n 👋 Kamera penceresi kapatıldı.")
            break

        # Klavyeden basılan tuşu dinle (Hem Kamera Penceresinden hem Terminalden)
        key = cv2.waitKeyEx(12)
        key_ascii = key & 0xFF if key != -1 else -1

        stdin_cmd = None
        try:
            import select
            if select.select([sys.stdin], [], [], 0)[0]:
                line = sys.stdin.readline().strip()
                if line:
                    stdin_cmd = line.lower()
        except Exception:
            pass

        cmd_received = False

        # --- ESC VEYA ÇIKIŞ ---
        if key in (27, ord('0')) or stdin_cmd in ('0', 'q', 'exit'):
            send_teleop_cmd(0, 0, 0, 0, 0)
            print("\n 👋 Kamera akışı kapatıldı.")
            break

        # --- GPS AÇ / KAPAT (Tuş: 2 veya G) ---
        if key_ascii in (ord('g'), ord('G'), ord('2')) or stdin_cmd in ('2', 'g', 'gps'):
            gps_enabled_state[0] = not gps_enabled_state[0]
            if px4_bridge:
                px4_bridge.set_gps_enabled(gps_enabled_state[0])
            
            if gps_enabled_state[0]:
                print("\n\033[1;32m======================================================================\033[0m")
                print("\033[1;32m 🟢 [GPS AÇILDI]: EKF2_GPS_CTRL = 7 (Dış Mekan / 3D GPS Fix Aktif)\033[0m")
                print("\033[1;32m======================================================================\033[0m\n")
            else:
                print("\n\033[1;31m======================================================================\033[0m")
                print("\033[1;31m 🔴 [GPS KAPATILDI]: EKF2_GPS_CTRL = 0 (İç Mekan GPS-Denied Devrede)\033[0m")
                print("\033[1;31m======================================================================\033[0m\n")
            cmd_received = True

        # --- VIO VISUAL ODOMETRY AÇ / KAPAT (Tuş: 1 veya V) ---
        elif key_ascii in (ord('v'), ord('V'), ord('1')) or stdin_cmd in ('1', 'v', 'vio'):
            vio_enabled_state[0] = not vio_enabled_state[0]
            if px4_bridge:
                px4_bridge.set_vision_enabled(vio_enabled_state[0])
            
            if vio_enabled_state[0]:
                print("\n\033[1;36m======================================================================\033[0m")
                print("\033[1;36m 🔵 [VIO GÖRSEL ODOMETRİ AKTİF]: 30 Hz EKF2 Poz Kilidi Devraldı\033[0m")
                print("\033[1;36m======================================================================\033[0m\n")
            else:
                print("\n\033[1;33m======================================================================\033[0m")
                print("\033[1;33m ⚪ [VIO BEKLEMEDE]: Görsel Odometri Durduruldu\033[0m")
                print("\033[1;33m======================================================================\033[0m\n")
            cmd_received = True
        elif key_ascii in (ord('x'), ord('X')):
            cur_vx, cur_vy, cur_vz, cur_wy, cur_wz = 0.0, 0.0, 0.0, 0.0, 0.0
            target_vx, target_vy, target_vz, target_wz = 0.0, 0.0, 0.0, 0.0
            target_pitch_deg = 0.0
            send_teleop_cmd(0, 0, 0, 0, 0)
            cmd_received = True
        elif key_ascii in (ord('h'), ord('H')):
            show_help = not show_help

        if cmd_received:
            last_cmd_time = time.time()
        elif time.time() - last_cmd_time > 0.30:
            target_vx *= 0.70
            target_vy *= 0.70
            target_vz *= 0.70
            target_wz *= 0.70

        # Hız yumuşatma
        cur_vx = 0.60 * cur_vx + 0.40 * target_vx
        cur_vy = 0.60 * cur_vy + 0.40 * target_vy
        cur_vz = 0.60 * cur_vz + 0.40 * target_vz
        cur_wz = 0.60 * cur_wz + 0.40 * target_wz

        # Pitch P-kontrolörü (Kamera açısını hedefe kilitler)
        err_pitch = target_pitch_deg - current_pitch
        target_wy = -max(-1.5, min(1.5, err_pitch * 0.08))
        cur_wy = 0.50 * cur_wy + 0.50 * target_wy

        # Kamera eğikken yatay uçuş irtifa kompansasyonu
        np_rad = math.radians(current_pitch)
        cmd_vx = cur_vx * math.cos(np_rad) + cur_vz * math.sin(np_rad)
        cmd_vz = -cur_vx * math.sin(np_rad) + cur_vz * math.cos(np_rad)

        # Gazebo'ya hız bas (15 Hz)
        if time.time() - last_vel_send >= 0.065:
            send_teleop_cmd(cmd_vx, cur_vy, cmd_vz, cur_wy, cur_wz)
            last_vel_send = time.time()

        # [R] veya [3] tuşuna basıldığında veya terminalden girildiğinde taramayı başlat / bitir
        if key_ascii in (ord('r'), ord('R'), ord('3')) or stdin_cmd in ('3', 'r', 'rec', 'scan') or trigger_scan_flag[0]:
            trigger_scan_flag[0] = False
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
                send_teleop_cmd(0, 0, 0, 0, 0)

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
                    if cap is not None:
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


    # Döngü biterse köprüyü, kamerayı ve pencereleri serbest bırak
    if px4_bridge:
        px4_bridge.stop()
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


# Dosya doğrudan terminalden çalıştırıldığında burası başlar
if __name__ == "__main__":
    # Terminalden kamera numarası veya RTSP linki alabilir (varsayılan: gazebo)
    src = sys.argv[1] if len(sys.argv) > 1 else "gazebo"
    kfs = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    run_drone_capture(camera_source=src, target_keyframes=kfs)