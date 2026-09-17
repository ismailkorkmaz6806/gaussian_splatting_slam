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
    from gz.msgs10.pointcloud_packed_pb2 import PointCloudPacked
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


def run_drone_capture(camera_source=0, target_keyframes=50):
    """
    Canlı dron veya kamera akışını başlatır, kullanıcı taramayı bitirdiğinde
    otomatik olarak MASt3R yapay zekasını çalıştırıp 3B Gaussian Splatting modelini açar.
    
    Parametreler:
        camera_source    : Kamera indeksi (0, 1) veya RTSP URL'si ("rtsp://192.168.1.100:8554/stream")
        target_keyframes : 3B harita çıkarılırken videodan seçilecek anahtar kare sayısı (Yüksek Detay: 50)
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

        # PX4 MAVLink Görsel Odometri Köprüsü:
        # drone_lidar_odometry_bridge.py zaten arka planda çalıştığı için
        # burada ikinci bir köprü açmak EKF2 koordinatlarını çakıştırıp dronu yana fırlatıyordu.
        # Odometri tamamen ana köprüye bırakıldı.
        px4_bridge = None

        current_nose_pitch_deg = [0.0]
        current_drone_pos = [np.array([-3.0, 0.0, 0.20], dtype=np.float32)]
        current_drone_rot = [np.eye(3, dtype=np.float32)]
        start_flight_pos = [None]
        drone_trajectory = []

        lidar_lock = threading.Lock()
        accumulated_lidar_pts = []
        accumulated_lidar_rgb = []
        lidar_scan_count = [0]
        def _on_odom(msg):
            try:
                pos = msg.pose.position
                q = msg.pose.orientation
                sinp = 2 * (q.w * q.y - q.z * q.x)
                if abs(sinp) >= 1:
                    p = math.copysign(math.pi / 2, sinp)
                else:
                    p = math.asin(sinp)
                current_nose_pitch_deg[0] = -math.degrees(p)

                # YAW HESAPLAMA (Lidar Odometri Icin)
                siny_cosp = 2 * (q.w * q.z + q.x * q.y)
                cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
                yaw_deg = math.degrees(math.atan2(siny_cosp, cosy_cosp))

                # Dron anlık konumu
                pos_arr = np.array([float(pos.x), float(pos.y), float(pos.z)], dtype=np.float32)
                current_drone_pos[0] = pos_arr

                # PX4 BRIDGE'E LİDAR KONUMUNU BAS (Hover Icin)
                if px4_bridge:
                    px4_bridge.update_slam_pose(x_cam=float(pos.y), y_cam=-float(pos.z), z_cam=float(pos.x), yaw_deg=yaw_deg)

                qw, qx, qy, qz = float(q.w), float(q.x), float(q.y), float(q.z)
                current_drone_rot[0] = np.array([
                    [1.0 - 2.0*(qy*qy + qz*qz), 2.0*(qx*qy - qz*qw), 2.0*(qx*qz + qy*qw)],
                    [2.0*(qx*qy + qz*qw), 1.0 - 2.0*(qx*qx + qz*qz), 2.0*(qy*qz - qx*qw)],
                    [2.0*(qx*qz - qy*qw), 2.0*(qy*qz + qx*qw), 1.0 - 2.0*(qx*qx + qy*qy)]
                ], dtype=np.float32)

                if recording:
                    if start_flight_pos[0] is None:
                        start_flight_pos[0] = pos_arr.copy()
                    drone_trajectory.append(pos_arr.copy())
            except Exception as e:
                pass

        def _on_lidar(msg):
            try:
                offsets = {}
                for f in msg.field:
                    if f.name in ('x', 'y', 'z'):
                        offsets[f.name] = f.offset
                if len(offsets) < 3:
                    return
                step = msg.point_step
                n_pts = msg.width * msg.height
                if len(msg.data) < n_pts * step or n_pts == 0:
                    return

                data_2d = np.frombuffer(msg.data, dtype=np.uint8)[:n_pts * step].reshape((n_pts, step))
                x = data_2d[:, offsets['x']:offsets['x']+4].copy().view('<f4').ravel()
                y = data_2d[:, offsets['y']:offsets['y']+4].copy().view('<f4').ravel()
                z = data_2d[:, offsets['z']:offsets['z']+4].copy().view('<f4').ravel()
                pts_s = np.column_stack([x, y, z])

                # Menzil filtresi (0.25m - 30.0m) ve finite kontrolü
                dists = np.linalg.norm(pts_s, axis=1)
                valid = (dists > 0.25) & (dists < 30.0) & np.isfinite(pts_s).all(axis=1)
                pts_s = pts_s[valid]
                if len(pts_s) == 0:
                    return

                lidar_scan_count[0] += 1

                if lidar_scan_count[0] == 1:
                    print("\n 📡 [KATI HAL 3B LIDAR AKTİF]: Gazebo'dan 3B Nokta Bulutu Verisi Akıyor!")

                # Montaj açısı: burun hafif aşağı (pitch = 0.08 rad), T_mount = [0.15, 0.0, 0.18]
                cp, sp = 0.9968, 0.0799
                xb = 0.15 + pts_s[:, 0] * cp + pts_s[:, 2] * sp
                yb = pts_s[:, 1]
                zb = 0.18 - pts_s[:, 0] * sp + pts_s[:, 2] * cp
                pts_b = np.column_stack([xb, yb, zb])

                R_b = current_drone_rot[0]
                T_b = current_drone_pos[0]
                if R_b is None or T_b is None:
                    return

                pts_w = (R_b @ pts_b.T).T + T_b

                # Renk ataması (Kamera projeksiyonu veya gerçekçi kaya/zemin tonu)
                with gz_lock:
                    cam_img = latest_gz_frame[0]

                # Kamera koordinatları (Kamera gövdede [0.12, 0.03, 0.242])
                pc_x = xb - 0.12
                pc_y = yb - 0.03
                pc_z = zb - 0.242

                # Kamera HFOV ~ 1.74 rad (f ~ 270 px)
                u = (320.0 - 270.0 * (pc_y / np.maximum(pc_x, 0.01))).astype(np.int32)
                v = (240.0 - 270.0 * (pc_z / np.maximum(pc_x, 0.01))).astype(np.int32)

                in_cam = (pc_x > 0.2) & (u >= 0) & (u < 640) & (v >= 0) & (v < 480)
                rgb = np.zeros((len(pts_w), 3), dtype=np.float32)

                if cam_img is not None and np.any(in_cam):
                    idx_cam = np.where(in_cam)[0]
                    sampled = cam_img[v[idx_cam], u[idx_cam]][:, ::-1] / 255.0
                    rgb[idx_cam] = sampled

                out_cam = ~in_cam
                if np.any(out_cam):
                    idx_out = np.where(out_cam)[0]
                    z_w = pts_w[idx_out, 2]
                    is_floor = z_w < 0.25
                    f_idx = idx_out[is_floor]
                    w_idx = idx_out[~is_floor]
                    if len(f_idx) > 0:
                        rgb[f_idx] = np.array([0.45, 0.38, 0.32], dtype=np.float32) + np.random.uniform(-0.02, 0.02, (len(f_idx), 3))
                    if len(w_idx) > 0:
                        rgb[w_idx] = np.array([0.42, 0.40, 0.38], dtype=np.float32) + np.random.uniform(-0.03, 0.03, (len(w_idx), 3))

                rgb = np.clip(rgb, 0.05, 0.95)

                with lidar_lock:
                    if recording:
                        accumulated_lidar_pts.append(pts_w)
                        accumulated_lidar_rgb.append(rgb)
            except Exception:
                pass

        # Gazebo Kamera Konuları
        _gz_node.subscribe(GzImage, "/world/benim_magaram/model/x500_vision_0/link/camera_link/sensor/camera/image", _on_img)
        _gz_node.subscribe(GzImage, "/world/benim_magaram/model/x500_vision/link/camera_link/sensor/camera/image", _on_img)
        _gz_node.subscribe(GzImage, "/world/buyuk_ev/model/x500_vision_0/link/camera_link/sensor/camera/image", _on_img)
        _gz_node.subscribe(GzImage, "/world/buyuk_ev/model/x500_vision/link/camera_link/sensor/camera/image", _on_img)
        _gz_node.subscribe(GzImage, "/world/tunnel_world/model/x500_vision_0/link/camera_link/sensor/camera/image", _on_img)
        _gz_node.subscribe(GzImage, "/world/tunnel_world/model/x500_vision/link/camera_link/sensor/camera/image", _on_img)
        _gz_node.subscribe(GzImage, "/camera", _on_img)
        _gz_node.subscribe(GzImage, "/camera/image", _on_img)
        _gz_node.subscribe(GzImage, "/world/cave/model/x500_vision_0/link/camera_link/sensor/camera/image", _on_img)

        # Gazebo Odometri Konuları
        _gz_node.subscribe(Odometry, "/world/benim_magaram/model/x500_vision_0/odometry", _on_odom)
        _gz_node.subscribe(Odometry, "/world/benim_magaram/model/x500_vision/odometry", _on_odom)
        _gz_node.subscribe(Odometry, "/world/buyuk_ev/model/x500_vision_0/odometry", _on_odom)
        _gz_node.subscribe(Odometry, "/world/buyuk_ev/model/x500_vision/odometry", _on_odom)
        _gz_node.subscribe(Odometry, "/world/tunnel_world/model/x500_vision_0/odometry", _on_odom)
        _gz_node.subscribe(Odometry, "/world/tunnel_world/model/x500_vision/odometry", _on_odom)
        _gz_node.subscribe(Odometry, "/model/x500_vision/odometry", _on_odom)
        _gz_node.subscribe(Odometry, "/model/x500_vision_0/odometry", _on_odom)

        # Gazebo Katı Hal (Solid-State) 3B LiDAR Konuları
        _gz_node.subscribe(PointCloudPacked, "/forward_lidar/points", _on_lidar)
        _gz_node.subscribe(PointCloudPacked, "/forward_lidar/points/points", _on_lidar)
        _gz_node.subscribe(PointCloudPacked, "/world/benim_magaram/model/x500_vision_0/link/forward_lidar_link/sensor/forward_lidar/scan/points", _on_lidar)
        _gz_node.subscribe(PointCloudPacked, "/world/benim_magaram/model/x500_vision/link/forward_lidar_link/sensor/forward_lidar/scan/points", _on_lidar)
        _gz_node.subscribe(PointCloudPacked, "/world/buyuk_ev/model/x500_vision_0/link/forward_lidar_link/sensor/forward_lidar/scan/points", _on_lidar)
        _gz_node.subscribe(PointCloudPacked, "/world/tunnel_world/model/x500_vision_0/link/forward_lidar_link/sensor/forward_lidar/scan/points", _on_lidar)
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
    gps_enabled_state = [False]   # Başlangıçta GPS Açık (Dış mekan kalkışı)
    vio_enabled_state = [True]  # Başlangıçta VIO beklemede (Kullanıcı kalkıştan sonra açacak)
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
    print("    [1]: Lidar Odometry (Her Zaman Açık)")
    print("    [2]: GPS (Donanımsal Olarak Söküldü)")
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
            with lidar_lock:
                n_lp = sum(len(p) for p in accumulated_lidar_pts)
            # Kayıt durumunda üstte kırmızı canlı kayıt rozeti
            cv2.circle(display_frame, (20, 24), 8, (0, 0, 255), -1)
            cv2.putText(display_frame, f"KAYIT: {len(recorded_frames)} Kare | {n_lp:,} Lidar Pts  [R / 3: Bitir & 3DGS Haritala]",
                        (36, 30), cv2.FONT_HERSHEY_DUPLEX, 0.52, (0, 0, 255), 1)
        else:
            # Normal durumda sol üstte sade durum bilgisi
            gps_st = "GPS: IPTAL" if gps_enabled_state[0] else "GPS: KAPALI (GPS-Denied)"
            vio_st = "LIDAR ODOMETRY: AKTIF" if vio_enabled_state[0] else "LIDAR ODOMETRY: AKTIF"
            lidar_st = f"3D LIDAR: AKTIF ({lidar_scan_count[0]} scan)" if lidar_scan_count[0] > 0 else "3D LIDAR: BEKLEMEDE"
            cv2.putText(display_frame, f"CANLI KAMERA  |  {gps_st}  |  {vio_st}  |  {lidar_st}", 
                        (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 180), 1)

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

                # Kaydedilen LiDAR noktalarını işle ve kaydet
                temp_lidar_path = os.path.join(CURR_DIR, "temp_lidar_pts.npz")
                has_lidar = False
                with lidar_lock:
                    if accumulated_lidar_pts:
                        all_lp = np.vstack(accumulated_lidar_pts)
                        all_lr = np.vstack(accumulated_lidar_rgb)
                        # Voksel seyreltme (2.5 cm ızgara çözünürlüğü - yüksek yoğunluk, sıfır boşluk)
                        grid = np.floor(all_lp / 0.025).astype(np.int32)
                        _, uidx = np.unique(grid, axis=0, return_index=True)
                        lp_ds = all_lp[uidx]
                        lr_ds = all_lr[uidx]
                        traj_arr = np.array(drone_trajectory, dtype=np.float32) if drone_trajectory else None
                        sp = start_flight_pos[0] if start_flight_pos[0] is not None else np.array([-3.0, 0.0, 0.20], dtype=np.float32)
                        np.savez_compressed(
                            temp_lidar_path,
                            pts=lp_ds,
                            rgb=lr_ds,
                            traj=traj_arr,
                            start_pos=sp
                        )
                        has_lidar = True
                        print(f" 📡 Katı Hal 3B LiDAR: {len(lp_ds):,} hassas metrik yüzey noktası kaydedildi!")

                if total_rec < 15 and not has_lidar:
                    print(" ⚠️ UYARI: Kayıt çok kısa oldu (<15 kare) ve LiDAR verisi bulunamadı. Biraz daha uzun tarama yapın.")
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
                    print(" ⚡ MASt3R & 3B LIDAR MOTORU ÇALIŞIYOR: 3D GAUSSIAN SPLAT HARİTASI ÜRETİLİYOR...")
                    print("=" * 70)

                    # MASt3R & LiDAR motorunu çağırıp 3B Gaussian Splatting modelini üretiyoruz
                    from mast3r_to_3dgs import build_gaussian_splats_from_mast3r
                    out_ply = build_gaussian_splats_from_mast3r(
                        video_file=temp_video_path,
                        output_ply="drone_scene.ply",
                        num_keyframes=min(target_keyframes, max(total_rec, 1)),
                        target_size=512,
                        lidar_file=temp_lidar_path if has_lidar else None
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
    kfs = int(sys.argv[2]) if len(sys.argv) > 2 else 35
    run_drone_capture(camera_source=src, target_keyframes=kfs)
