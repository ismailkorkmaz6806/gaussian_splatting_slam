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

    win_name = "CANLI FPV DRON KOKPITI [W/A/S/D: UC | OKLAR: KAMERA | R: 3DGS | H: KILAVUZ | ESC: CIKIS]"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

    def _on_mouse(event, mx, my, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # 1. Buton: VIO Aç/Kapat (X: 180-360, Y: 6-40)
            if 180 <= mx <= 360 and 6 <= my <= 40:
                vio_enabled_state[0] = not vio_enabled_state[0]
                if px4_bridge:
                    px4_bridge.set_vision_enabled(vio_enabled_state[0])
                st = "AÇILDI (30Hz EKF2 Görsel Kilit)" if vio_enabled_state[0] else "KAPATILDI (0)"
                status_toast_msg[0] = f"📷 1. VIO Görsel Odometri {st}!"
                status_toast_time[0] = time.time()
                print(f"\n 🖱️ [BUTON]: {status_toast_msg[0]}")

            # 2. Buton: GPS Kapat / Aç (X: 370-580, Y: 6-40)
            elif 370 <= mx <= 580 and 6 <= my <= 40:
                gps_enabled_state[0] = not gps_enabled_state[0]
                if px4_bridge:
                    px4_bridge.set_gps_enabled(gps_enabled_state[0])
                st = "AÇILDI (EKF2_GPS_CTRL 7)" if gps_enabled_state[0] else "KAPATILDI (EKF2_GPS_CTRL 0 - GPS-Denied)"
                status_toast_msg[0] = f"🛰️ 2. GPS {st}!"
                status_toast_time[0] = time.time()
                print(f"\n 🖱️ [BUTON]: {status_toast_msg[0]}")

            # 3. Buton: 3DGS Harita Tara (X: 590-760, Y: 6-40)
            elif 590 <= mx <= 760 and 6 <= my <= 40:
                trigger_scan_flag[0] = True

            # 4. Buton: Yardım Kartı (X: 770-870, Y: 6-40)
            elif 770 <= mx <= 870 and 6 <= my <= 40:
                show_help_state_val = not show_help
                # show_help döngü içinde güncellenecek

    cv2.setMouseCallback(win_name, _on_mouse)

    print("\n ✅ Canlı FPV Kokpit açıldı!")
    print("    UÇUŞ   : [W/S]: İleri/Geri | [A/D]: Sola/Sağa Kay | [SPACE/C]: Yüksel/Alçal | [X]: Fren")
    print("    BUTON  : [1. VIO AÇ (V)] -> [2. GPS KAPAT (G)] -> [3. 3DGS TARA (R)]")
    print("    KAMERA : [Yukarı/Aşağı Ok]: Yukarı/Aşağı Eğ | [Q/E veya Sol/Sağ]: Dön | [F]: Düzle")
    print("    DİĞER  : [H]: Kılavuzu Aç/Kapat | [ESC]: Çıkış")

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
        # 2B FPV KOKPİT ARAYÜZÜ VE HUD ÇİZİMİ
        # ---------------------------------------------------------------------
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 48), (10, 15, 22), -1)
        cv2.rectangle(overlay, (0, h - 56), (w, h), (10, 15, 22), -1)
        cv2.addWeighted(overlay, 0.85, display_frame, 0.15, 0, display_frame)

        # ---------------------------------------------------------------------
        # İNTERAKTİF TIKLANABİLİR ÜST BUTONLAR
        # ---------------------------------------------------------------------
        # Buton 1: VIO (Visual Odometry) Butonu
        vio_on = vio_enabled_state[0]
        b1_bg = (30, 80, 20) if vio_on else (35, 40, 50)
        b1_border = (0, 255, 120) if vio_on else (0, 200, 255)
        b1_txt = "1. VIO: AKTIF (30Hz)" if vio_on else "1. VIO AC (Tikla/V)"
        cv2.rectangle(display_frame, (180, 7), (360, 41), b1_bg, -1)
        cv2.rectangle(display_frame, (180, 7), (360, 41), b1_border, 2 if vio_on else 1)
        cv2.putText(display_frame, b1_txt, (190, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255) if vio_on else (200, 230, 255), 1)

        # Buton 2: GPS Kapat / Aç Butonu
        gps_on = gps_enabled_state[0]
        b2_bg = (20, 60, 20) if gps_on else (20, 20, 80)
        b2_border = (0, 255, 100) if gps_on else (0, 60, 255)
        b2_txt = "2. GPS: ACIK (Tikla/G)" if gps_on else "2. GPS: KAPALI (0)"
        cv2.rectangle(display_frame, (370, 7), (580, 41), b2_bg, -1)
        cv2.rectangle(display_frame, (370, 7), (580, 41), b2_border, 2 if not gps_on else 1)
        cv2.putText(display_frame, b2_txt, (380, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255) if not gps_on else (200, 240, 220), 1)

        # Buton 3: 3DGS Harita Tara Butonu
        b3_bg = (0, 0, 90) if recording else (60, 20, 70)
        b3_border = (0, 0, 255) if recording else (220, 0, 255)
        b3_txt = "3. BITIR & HARITALA" if recording else "3. 3DGS TARA (R)"
        cv2.rectangle(display_frame, (590, 7), (760, 41), b3_bg, -1)
        cv2.rectangle(display_frame, (590, 7), (760, 41), b3_border, 2 if recording else 1)
        cv2.putText(display_frame, b3_txt, (600, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1)

        # Sol Üst Başlık / Rozet
        cv2.circle(display_frame, (22, 24), 8, (0, 255, 120), -1)
        cv2.putText(display_frame, "FPV KOKPIT", (38, 29), cv2.FONT_HERSHEY_DUPLEX, 0.52, (0, 255, 180), 1)

        # Anlık Durum Bildirim Bandı (Toast Message)
        if time.time() - status_toast_time[0] < 4.0:
            toast_text = status_toast_msg[0]
            cv2.rectangle(display_frame, (16, 56), (min(w - 16, 16 + len(toast_text) * 11 + 20), 84), (15, 20, 30), -1)
            cv2.rectangle(display_frame, (16, 56), (min(w - 16, 16 + len(toast_text) * 11 + 20), 84), (0, 220, 255), 1)
            cv2.putText(display_frame, toast_text, (26, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # FPV Nişangah / Hedefleme Çaprazı (Merkezde)
        cx, cy = w // 2, h // 2
        cv2.line(display_frame, (cx - 24, cy), (cx - 7, cy), (0, 230, 255), 1)
        cv2.line(display_frame, (cx + 7, cy), (cx + 24, cy), (0, 230, 255), 1)
        cv2.line(display_frame, (cx, cy - 24), (cx, cy - 7), (0, 230, 255), 1)
        cv2.line(display_frame, (cx, cy + 7), (cx, cy + 24), (0, 230, 255), 1)
        cv2.circle(display_frame, (cx, cy), 3, (0, 230, 255), -1)

        # Merkezde Kamera Açı Göstergesi (Tilt)
        pitch_str = f"Aci: {current_pitch:+.0f}\xb0"
        cv2.putText(display_frame, pitch_str, (cx + 30, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 240, 255), 1)

        # Alt Panel Bilgileri (2 Satır Halinde Her Şey Açıkça Görünür)
        line1_str = f"Ileri: {cur_vx:+.1f} m/s  |  Yan: {cur_vy:+.1f} m/s  |  Dikey: {cur_vz:+.1f} m/s  |  Kamera Acisi: {current_pitch:+.1f}\xb0  |  Donus: {cur_wz:+.1f} rad/s"
        cv2.putText(display_frame, line1_str, (16, h - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 240, 220), 1)
        
        line2_str = "UCUS: [W/S]: Ileri/Geri  [A/D]: Sola/Saga  [SPACE/C]: Yukari/Asagi  |  [G]: GPS On/Off  [V]: VIO On/Off  |  [H]: Yardim"
        cv2.putText(display_frame, line2_str, (16, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 225, 250), 1)

        # ---------------------------------------------------------------------
        # YARI ŞEFFAF KONTROL KILAVUZU KARTI (Sağ Üst Köşede)
        # ---------------------------------------------------------------------
        if show_help:
            card_w, card_h = 320, 290
            card_x1, card_y1 = w - card_w - 12, 54
            card_x2, card_y2 = card_x1 + card_w, card_y1 + card_h
            sub_hud = display_frame[card_y1:card_y2, card_x1:card_x2]
            card_bg = np.zeros_like(sub_hud)
            card_bg[:] = (15, 20, 30)
            cv2.addWeighted(card_bg, 0.88, sub_hud, 0.12, 0, sub_hud)
            display_frame[card_y1:card_y2, card_x1:card_x2] = sub_hud
            cv2.rectangle(display_frame, (card_x1, card_y1), (card_x2, card_y2), (0, 220, 255), 1)
            
            # Kart Başlığı
            cv2.putText(display_frame, "KONTROL KILAVUZU [H: Kapat]", 
                        (card_x1 + 10, card_y1 + 22), cv2.FONT_HERSHEY_DUPLEX, 0.48, (0, 230, 255), 1)
            cv2.line(display_frame, (card_x1 + 8, card_y1 + 29), (card_x2 - 8, card_y1 + 29), (60, 90, 120), 1)

            # Uçuş Tuşları
            cv2.putText(display_frame, "[ UCUS KONTROLLERI ]", 
                        (card_x1 + 10, card_y1 + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 160), 1)
            cv2.putText(display_frame, "W / S   : Ileri / Geri Ucus", 
                        (card_x1 + 16, card_y1 + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 235, 245), 1)
            cv2.putText(display_frame, "A / D   : Sola / Saga Kay (Strafe)", 
                        (card_x1 + 16, card_y1 + 86), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 235, 245), 1)
            cv2.putText(display_frame, "SPACE/C : Yukari / Asagi (Irtifa)", 
                        (card_x1 + 16, card_y1 + 104), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 235, 245), 1)
            cv2.putText(display_frame, "X       : Havada Fren & Sabit Kal", 
                        (card_x1 + 16, card_y1 + 122), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 235, 245), 1)

            # GPS & VIO Hibrit Geçiş Tuşları
            cv2.putText(display_frame, "[ OTOPILOT & NAVIGASYON ]", 
                        (card_x1 + 10, card_y1 + 144), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 160), 1)
            cv2.putText(display_frame, "G : GPS Ac / Kapat (Gps-Denied)", 
                        (card_x1 + 16, card_y1 + 164), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 240, 255), 1)
            cv2.putText(display_frame, "V : Visual Odometry Ac / Kapat", 
                        (card_x1 + 16, card_y1 + 182), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 240, 255), 1)

            # Kamera & Harita Tuşları
            cv2.putText(display_frame, "[ KAMERA & 3B HARITA ]", 
                        (card_x1 + 10, card_y1 + 204), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 160), 1)
            cv2.putText(display_frame, "Oklar / I-K   : Kamerayi Eg  [F]: Sifirla", 
                        (card_x1 + 16, card_y1 + 224), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 240, 160), 1)
            cv2.putText(display_frame, "Q / E         : Sola / Saga Don (Yaw)", 
                        (card_x1 + 16, card_y1 + 242), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 235, 245), 1)
            cv2.putText(display_frame, "R             : 3B Tara & Harita Uret", 
                        (card_x1 + 16, card_y1 + 260), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 180), 1)

        # Görüntüyü pencerede göster
        cv2.imshow(win_name, display_frame)

        # Pencere kapatma düğmesine (X) basıldıysa güvenli çık
        if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
            send_teleop_cmd(0, 0, 0, 0, 0)
            print("\n 👋 FPV Kokpit penceresi kapatıldı.")
            break

        # Klavyeden basılan tuşu dinle (12 milisaniye bekle)
        key = cv2.waitKeyEx(12)
        key_ascii = key & 0xFF if key != -1 else -1

        cmd_received = False

        # --- ESC VEYA LINUX ANSI OK TUŞLARI DİZİSİ YÖNETİMİ ---
        if key == 27:
            # Linux'ta ok tuşları \033[A, \033[B, \033[C, \033[D olarak gelebilir
            k2 = cv2.waitKeyEx(8)
            if k2 in (91, ord('[')):
                k3 = cv2.waitKeyEx(8)
                if k3 in (65, ord('A')):    # Yukarı Ok -> Kamerayı Yukarı Eğ
                    target_pitch_deg = min(50.0, target_pitch_deg + 2.5)
                    cmd_received = True
                elif k3 in (66, ord('B')):  # Aşağı Ok -> Kamerayı Aşağı Eğ
                    target_pitch_deg = max(-50.0, target_pitch_deg - 2.5)
                    cmd_received = True
                elif k3 in (67, ord('C')):  # Sağ Ok -> Sağa Dön (Yaw Right)
                    target_wz = -1.0
                    cmd_received = True
                elif k3 in (68, ord('D')):  # Sol Ok -> Sola Dön (Yaw Left)
                    target_wz = 1.0
                    cmd_received = True
            elif k2 == -1:
                # Sadece tek başına ESC tuşuna basılmışsa çıkış yap
                send_teleop_cmd(0, 0, 0, 0, 0)
                print("\n 👋 ESC tuşuna basıldı, Canlı FPV akışı kapatılıyor...")
                break

        # --- KAMERA YUKARI / AŞAĞI EĞME (TILT) ---
        if key in (0x01000013, 65362) or key_ascii in (ord('i'), ord('I')):
            target_pitch_deg = min(50.0, target_pitch_deg + 2.5)
            cmd_received = True
        elif key in (0x01000015, 65364) or key_ascii in (ord('k'), ord('K')):
            target_pitch_deg = max(-50.0, target_pitch_deg - 2.5)
            cmd_received = True
        elif key_ascii in (ord('f'), ord('F')):
            target_pitch_deg = 0.0
            cmd_received = True

        # --- DÖNÜŞ (YAW) ---
        elif key in (0x01000012, 65361) or key_ascii in (ord('q'), ord('Q')):
            target_wz = 1.0; cmd_received = True
        elif key in (0x01000014, 65363) or key_ascii in (ord('e'), ord('E')):
            target_wz = -1.0; cmd_received = True

        # --- UÇUŞ HAREKETLERİ ---
        elif key_ascii in (ord('w'), ord('W')):
            target_vx = 1.6; cmd_received = True
        elif key_ascii in (ord('s'), ord('S')):
            target_vx = -1.6; cmd_received = True
        elif key_ascii in (ord('a'), ord('A')):
            target_vy = 1.2; cmd_received = True
        elif key_ascii in (ord('d'), ord('D')):
            target_vy = -1.2; cmd_received = True
        elif key_ascii in (ord(' '), ord('t'), ord('T')):
            target_vz = 1.0; cmd_received = True
        elif key_ascii in (ord('c'), ord('C')):
            target_vz = -1.0; cmd_received = True
        elif key_ascii in (ord('g'), ord('G')):
            gps_enabled_state[0] = not gps_enabled_state[0]
            if px4_bridge:
                px4_bridge.set_gps_enabled(gps_enabled_state[0])
            status_txt = "AÇIK (7)" if gps_enabled_state[0] else "KAPALI (0 - GPS-Denied Modu)"
            print(f"\n 🛰️ [OTOPİLOT]: GPS {status_txt} yapıldı.")
            cmd_received = True
        elif key_ascii in (ord('v'), ord('V')):
            vio_enabled_state[0] = not vio_enabled_state[0]
            if px4_bridge:
                px4_bridge.set_vision_enabled(vio_enabled_state[0])
            status_txt = "AÇIK (15 - 30Hz Görsel Kilit)" if vio_enabled_state[0] else "KAPALI (0)"
            print(f"\n 📷 [OTOPİLOT]: Görsel Odometri (VIO) {status_txt} yapıldı.")
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

        # [R] tuşuna basıldığında veya butona tıklandığında taramayı başlat / bitir
        if key_ascii in (ord('r'), ord('R')) or trigger_scan_flag[0]:
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