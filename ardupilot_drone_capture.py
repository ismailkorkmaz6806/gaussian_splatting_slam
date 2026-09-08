import sys
import os
import time
import math
import threading
import cv2
import numpy as np

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
if CURR_DIR not in sys.path:
    sys.path.insert(0, CURR_DIR)

os.environ['GZ_IP'] = '127.0.0.1'
os.environ['GZ_PARTITION'] = 'default'

if '/usr/lib/python3/dist-packages' not in sys.path:
    sys.path.append('/usr/lib/python3/dist-packages')

try:
    from gz.transport13 import Node
    from gz.msgs10.twist_pb2 import Twist
    from gz.msgs10.image_pb2 import Image as GzImage
    from gz.msgs10.odometry_pb2 import Odometry
    from gz.msgs10.laserscan_pb2 import LaserScan
    _gz_node = Node()
    _gz_pub_vel = _gz_node.advertise('/cmd_vel', Twist)
    HAS_GAZEBO_CONTROL = True
except Exception as e:
    HAS_GAZEBO_CONTROL = False
    print('Gazebo transport uyarisi:', e)

# Otopilot & LIDAR Durum Değişkenleri
autopilot_mode = False  # Otomatik Tarama modu
lidar_min_front = 10.0
lidar_min_left = 10.0
lidar_min_right = 10.0

def send_teleop_cmd(vx, vy, vz, wy=0.0, wz=0.0):
    if HAS_GAZEBO_CONTROL:
        try:
            msg = Twist()
            msg.linear.x = float(vx)
            msg.linear.y = float(vy)
            msg.linear.z = float(vz)
            msg.angular.x = 0.0
            msg.angular.y = 0.0  # Ufka paralel ucus kilidi (0.0 rad)
            msg.angular.z = float(wz)
            _gz_pub_vel.publish(msg)
        except Exception:
            pass

def run_cave_capture(target_keyframes=45):
    global autopilot_mode, ap_vx, ap_vy, ap_vz, ap_yaw_rate, master_mavlink
    print('=' * 70)
    print(' 🦇 DARPA SUBT MAGARA KESIF DRONU & 3DGS HARITALAMA')
    print(' 💡 [W/A/S/D / Ok Tuslari] : Ucus & Yon | [Q-E] : Donus')
    print(' 💡 [R]                    : 3B Magara Taramasini Baslat / Bitir')
    print(' 💡 [ESC]                  : Cikis')
    print('=' * 70)

    latest_frame = [None]
    gz_lock = threading.Lock()
    drone_pose = [0.0, 0.0, 2.0, 0.0]  # lc_mine spawn: x=0, y=0, z=2m

    def _on_img(msg):
        try:
            iw, ih = msg.width, msg.height
            raw_bytes = bytes(msg.data)
            arr = np.frombuffer(raw_bytes, dtype=np.uint8)
            if len(arr) == iw * ih * 3:
                bgr = cv2.cvtColor(arr.reshape((ih, iw, 3)), cv2.COLOR_RGB2BGR)
            elif len(arr) == iw * ih * 4:
                bgr = cv2.cvtColor(arr.reshape((ih, iw, 4)), cv2.COLOR_RGBA2BGR)
            else:
                return

            mean_val = float(bgr.mean())
            std_val  = float(bgr.std())
            if mean_val > 242.0 or mean_val < 10.0 or std_val < 5.0:
                return

            with gz_lock:
                latest_frame[0] = bgr
        except Exception:
            pass

    def _on_lidar(msg):
        global lidar_min_front, lidar_min_left, lidar_min_right
        try:
            ranges = msg.ranges
            if len(ranges) > 0:
                n = len(ranges)
                mid = n // 2
                def get_min(arr):
                    valid = [r for r in arr if 0.15 < r < 14.9]
                    return min(valid) if valid else 15.0
                lidar_min_front = get_min(ranges[mid-25 : mid+25])
                lidar_min_left  = get_min(ranges[mid+30 : mid+100])
                lidar_min_right = get_min(ranges[mid-100 : mid-30])
        except Exception:
            pass

    if HAS_GAZEBO_CONTROL:
        _gz_node.subscribe(GzImage,   '/camera', _on_img)
        _gz_node.subscribe(LaserScan, '/lidar',  _on_lidar)

    # ==============================================================
    # ✈️ ARDUPILOT MAVLINK BAĞLANTISI (SITL)
    # ==============================================================
    global master_mavlink, ap_vx, ap_vy, ap_vz, ap_yaw_rate
    master_mavlink = None
    ap_vx = ap_vy = ap_vz = ap_yaw_rate = 0.0
    
    try:
        from pymavlink import mavutil
        print(" 🔌 ArduPilot SITL aranıyor (tcp:127.0.0.1:5760)...")
        master_mavlink = mavutil.mavlink_connection('tcp:127.0.0.1:5760')
        master_mavlink.wait_heartbeat(timeout=10)
        if master_mavlink.target_system == 0:
            print("❌ MAVLink Zaman Aşımı! ArduPilot çalışmıyor olabilir.")
        else:
            print(" ✅ ArduPilot Kalp Atışı (Heartbeat) Alındı!")
            master_mavlink.set_mode('GUIDED')
            print(" 🚀 Dron Arm Ediliyor...")
            master_mavlink.arducopter_arm()
            master_mavlink.motors_armed_wait()
            print(" ⬆️ Kalkış (Takeoff 2.0m) başlatıldı...")
            master_mavlink.mav.command_long_send(
                master_mavlink.target_system, master_mavlink.target_component,
                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0,
                0, 0, 0, 0, 0, 0, 2.0)
            
            def mavlink_thread():
                global drone_pose, current_pitch
                while True:
                    msg = master_mavlink.recv_match(type=['LOCAL_POSITION_NED', 'ATTITUDE'], blocking=False)
                    if msg:
                        if msg.get_type() == 'LOCAL_POSITION_NED':
                            drone_pose[0] = msg.x
                            drone_pose[1] = msg.y
                            drone_pose[2] = -msg.z
                        elif msg.get_type() == 'ATTITUDE':
                            drone_pose[3] = math.degrees(msg.pitch)
                            current_pitch = math.degrees(msg.pitch)
                            
                    ned_vx = ap_vx
                    ned_vy = -ap_vy
                    ned_vz = -ap_vz
                    ned_yw = -ap_yaw_rate
                    
                    master_mavlink.mav.set_position_target_local_ned_send(
                        0, master_mavlink.target_system, master_mavlink.target_component,
                        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                        0b010111000111, 0, 0, 0, ned_vx, ned_vy, ned_vz, 0, 0, 0, 0, ned_yw)
                    time.sleep(0.05)

            t = threading.Thread(target=mavlink_thread, daemon=True)
            t.start()
            print(" 📡 MAVLink Kontrol Döngüsü Başlatıldı.")

    except ImportError:
        print("❌ HATA: pymavlink kurulu değil!")
    except Exception as e:
        print(f"❌ ArduPilot Bağlantı Hatası: {e}")

    print(' ⏳ Gazebo /camera akisi bekleniyor...')
    t_start = time.time()
    while time.time() - t_start < 45:
        with gz_lock:
            if latest_frame[0] is not None:
                print(' ✅ /camera kamerasina basariyla baglanildi!')
                break
        time.sleep(0.1)

    if latest_frame[0] is None:
        print('❌ HATA: Gazebo kamerasindan goruntu alinamadi! Simulasyonun calistigindan emin olun.')
        return

    win_name = 'DARPA SubT Magara Dron Kokpiti - 60 FPS'
    # ✅ WINDOW_AUTOSIZE: pencere frame boyutuna göre otomatik ayarlanır.
    # WM'nin resize'ı engellemesi sorunu yok – frame biz 1280x720'ye ölçekliyoruz.
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

    # ✅ Ekranda gösterilecek sabit çözünürlük (kamera 800x600 → 1280x720'ye ölçekle)
    DISP_W, DISP_H = 1280, 720

    recording = False
    recorded_frames = []
    show_help = True

    cur_vx, cur_vy, cur_vz, cur_wy, cur_wz = 0.0, 0.0, 0.0, 0.0, 0.0
    target_vx, target_vy, target_vz, target_wz = 0.0, 0.0, 0.0, 0.0
    target_pitch_deg = 0.0
    last_vel_send = time.time()
    last_cmd_time = 0.0

    last_display = None
    headlight_on = True   # Fener başlangıçta AÇIK

    def toggle_headlight(on: bool):
        """Gazebo'daki drone_headlight spot ışığını aç/kapat."""
        if not HAS_GAZEBO_CONTROL:
            return
        import subprocess, json
        # Gazebo Harmonic'te model içi ışığı değiştirmek için
        # /world/default/set_light_state servisi kullanılır.
        if on:
            req = ('name: "drone_headlight" '
                   'diffuse { r: 0.40 g: 0.40 b: 0.42 a: 1.0 } '
                   'specular { r: 0.05 g: 0.05 b: 0.05 a: 1.0 } '
                   'range: 40.0')
        else:
            req = ('name: "drone_headlight" '
                   'diffuse { r: 0.0 g: 0.0 b: 0.0 a: 1.0 } '
                   'specular { r: 0.0 g: 0.0 b: 0.0 a: 1.0 } '
                   'range: 0.001')
        subprocess.Popen(
            ['gz', 'service', '-s', '/world/default/set_light_state',
             '--reqtype', 'gz.msgs.Light',
             '--reptype', 'gz.msgs.Boolean',
             '--timeout', '500', '--req', req],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

    while True:
        # ✅ Lock altında .copy() – callback race condition YOK
        with gz_lock:
            raw = latest_frame[0]
            frame = raw.copy() if raw is not None else None

        if frame is not None:
            last_display = frame
        elif last_display is not None:
            frame = last_display
        else:
            key = cv2.waitKeyEx(15)
            continue

        # ✅ Kamera frame'ini görüntüleme çözünürlüğüne ölçekle (INTER_LINEAR hızlı & yumuşak)
        # Tüm HUD koordinatları bundan sonra DISP_W × DISP_H üzerinden hesaplanır.
        display_frame = cv2.resize(frame, (DISP_W, DISP_H), interpolation=cv2.INTER_LINEAR)
        h, w = DISP_H, DISP_W
        current_pitch = drone_pose[3]

        # ✅ addWeighted in-place yazım hatası DÜZELTİLDİ:
        # Eski: addWeighted(..., display_frame) → output = input, bazı OpenCV sürümlerinde bozuk
        # Yeni: overlay → ayrı output buffer → display_frame'e ata
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 54), (12, 18, 26), -1)
        cv2.rectangle(overlay, (0, h - 64), (w, h), (12, 18, 26), -1)
        display_frame = cv2.addWeighted(overlay, 0.82, display_frame, 0.18, 0)

        if recording:
            recorded_frames.append(frame.copy())  # ham kamera frame'i kaydet (ölçeksiz)
            cv2.circle(display_frame, (30, 27), 12, (0, 0, 255), -1)
            cv2.putText(display_frame, f'MAST3R TARANIYOR: {len(recorded_frames)} Kare',
                        (56, 35), cv2.FONT_HERSHEY_DUPLEX, 0.80, (0, 240, 255), 2)
        else:
            cv2.circle(display_frame, (30, 27), 10, (0, 255, 120), -1)
            cv2.putText(display_frame, 'DARPA SUBT MAGARA KOKPITI [CANLI]',
                        (56, 35), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 180), 2)

        # Otopilot ve Lidar HUD Uyarısı
        ap_str = '[P] OTOPILOT: AKTIF' if autopilot_mode else '[P] OTOPILOT: KAPALI'
        if lidar_min_front < 1.2:
            lidar_str = '⚠️ LIDAR: ENGELE COK YAKIN!'
            lidar_color = (0, 0, 255) # Kırmızı
        elif lidar_min_front < 2.5:
            lidar_str = 'LIDAR: Engel Yaklasiyor'
            lidar_color = (0, 140, 255) # Turuncu
        else:
            lidar_str = 'LIDAR: Temiz'
            lidar_color = (0, 255, 120) # Yeşil
            
        cv2.putText(display_frame, f'{ap_str}  |  {lidar_str}', (w - 600, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, lidar_color, 2)

        telemetry_str = (f'X:{drone_pose[0]:.1f}m  Y:{drone_pose[1]:.1f}m  '
                         f'Z:{drone_pose[2]:.1f}m  |  Egim:{current_pitch:+.1f} deg')
        cv2.putText(display_frame, telemetry_str, (w - 480, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 230, 240), 1)

        cx, cy = w // 2, h // 2
        cv2.line(display_frame, (cx - 28, cy), (cx + 28, cy), (0, 255, 200), 1)
        cv2.line(display_frame, (cx, cy - 28), (cx, cy + 28), (0, 255, 200), 1)
        cv2.circle(display_frame, (cx, cy), 8, (0, 255, 200), 1)

        spd = math.sqrt(cur_vx**2 + cur_vy**2 + cur_vz**2)
        fener_str = '💡 ACIK' if headlight_on else '🔴 KAPALI'
        status_txt = f'Hiz: {spd:.1f} m/s  |  Fener: {fener_str}  |  FPV RTX-Offload'
        cv2.putText(display_frame, status_txt, (22, h - 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 230, 255), 1)
        cv2.putText(display_frame, '[L]: Fener  |  [R]: 3DGS Tara  |  [H]: Yardim  |  [ESC]: Cikis',
                    (w - 600, h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 220, 255), 1)

        if show_help:
            card_w, card_h = 370, 340
            card_x1 = w - card_w - 18
            card_y1 = 62
            card_x2 = card_x1 + card_w
            card_y2 = card_y1 + card_h
            roi = display_frame[card_y1:card_y2, card_x1:card_x2].copy()
            card_bg = np.full_like(roi, (15, 20, 30))
            blended = cv2.addWeighted(card_bg, 0.88, roi, 0.12, 0)
            display_frame[card_y1:card_y2, card_x1:card_x2] = blended
            cv2.rectangle(display_frame, (card_x1, card_y1), (card_x2, card_y2), (0, 220, 255), 1)

            cv2.putText(display_frame, 'UCUS KILAVUZU  [H: Kapat]',
                        (card_x1 + 12, card_y1 + 24), cv2.FONT_HERSHEY_DUPLEX, 0.52, (0, 230, 255), 1)
            cv2.line(display_frame, (card_x1 + 8, card_y1 + 32),
                     (card_x2 - 8, card_y1 + 32), (60, 90, 120), 1)

            cv2.putText(display_frame, '[ UCUS & YONLENDIRME ]',
                        (card_x1 + 12, card_y1 + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 160), 1)
            cv2.putText(display_frame, 'W / S  /  Yukari-Asagi : Ileri / Geri Ucus',
                        (card_x1 + 18, card_y1 + 78), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 235, 245), 1)
            cv2.putText(display_frame, 'A / D                  : Sola / Saga Kay (Strafe)',
                        (card_x1 + 18, card_y1 + 102), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 235, 245), 1)
            cv2.putText(display_frame, 'SPACE / C              : Yukari / Asagi (Irtifa)',
                        (card_x1 + 18, card_y1 + 126), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 235, 245), 1)
            cv2.putText(display_frame, 'Sol-Sag / Q-E          : Sola / Saga Don (Yaw)',
                        (card_x1 + 18, card_y1 + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 235, 245), 1)
            cv2.putText(display_frame, 'X                      : Havada Sabit Kal (Hover)',
                        (card_x1 + 18, card_y1 + 174), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 235, 245), 1)

            cv2.putText(display_frame, '[ FENER, OTOPILOT & HARITALAMA ]',
                        (card_x1 + 12, card_y1 + 206), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 160), 1)
            fener_label = 'KAPAT' if headlight_on else 'AC'
            cv2.putText(display_frame, f'L   : Feneri {fener_label} (toggle)',
                        (card_x1 + 18, card_y1 + 232), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 235, 80), 1)
            cv2.putText(display_frame, 'P   : Otopilot (Otomatik Tarama Ucusu)',
                        (card_x1 + 18, card_y1 + 258), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
            cv2.putText(display_frame, 'R   : 3DGS Taramayi Baslat / Bitir',
                        (card_x1 + 18, card_y1 + 284), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 240, 255), 1)
            cv2.putText(display_frame, 'ESC : Programdan Cikis',
                        (card_x1 + 18, card_y1 + 310), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 220, 240), 1)

        cv2.imshow(win_name, display_frame)

        if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
            send_teleop_cmd(0, 0, 0, 0, 0)
            print('FPV Kokpit penceresi kapatildi.')
            break

        key = cv2.waitKeyEx(12)
        key_ascii = key & 0xFF if key != -1 else -1
        cmd_received = False

        if key == 27:
            k2 = cv2.waitKeyEx(8)
            if k2 in (91, ord('[')):
                k3 = cv2.waitKeyEx(8)
                if k3 in (65, ord('A')):
                    target_vx = 2.4
                    cmd_received = True
                elif k3 in (66, ord('B')):
                    target_vx = -2.4
                    cmd_received = True
                elif k3 in (67, ord('C')):
                    target_wz = -1.0
                    cmd_received = True
                elif k3 in (68, ord('D')):
                    target_wz = 1.0
                    cmd_received = True
            elif k2 == -1:
                send_teleop_cmd(0, 0, 0, 0, 0)
                print('ESC tusuna basildi, FPV akisi kapatiliyor.')
                break

        if key in (0x01000013, 65362):
            target_vx = 2.4
            cmd_received = True
        elif key in (0x01000015, 65364):
            target_vx = -2.4
            cmd_received = True
        elif key in (0x01000012, 65361) or key_ascii in (ord('q'), ord('Q')):
            target_wz = 1.0; cmd_received = True
        elif key in (0x01000014, 65363) or key_ascii in (ord('e'), ord('E')):
            target_wz = -1.0; cmd_received = True

        elif key_ascii in (ord('w'), ord('W')):
            target_vx = 2.4; cmd_received = True
        elif key_ascii in (ord('s'), ord('S')):
            target_vx = -2.4; cmd_received = True
        elif key_ascii in (ord('a'), ord('A')):
            target_vy = 1.8; cmd_received = True
        elif key_ascii in (ord('d'), ord('D')):
            target_vy = -1.8; cmd_received = True
        elif key_ascii in (ord(' '), ord('t'), ord('T')):
            target_vz = 1.5; cmd_received = True
        elif key_ascii in (ord('c'), ord('C'), ord('g'), ord('G')):
            target_vz = -1.5; cmd_received = True
        elif key_ascii in (ord('x'), ord('X')):
            cur_vx, cur_vy, cur_vz, cur_wy, cur_wz = 0.0, 0.0, 0.0, 0.0, 0.0
            target_vx, target_vy, target_vz, target_wz = 0.0, 0.0, 0.0, 0.0
            send_teleop_cmd(0, 0, 0, 0, 0)
            cmd_received = True
        elif key_ascii in (ord('h'), ord('H')):
            show_help = not show_help
        elif key_ascii in (ord('l'), ord('L')):
            headlight_on = not headlight_on
            toggle_headlight(headlight_on)
            state_msg = 'ACILDI' if headlight_on else 'KAPATILDI'
            print(f'💡 Fener {state_msg}.')
        elif key_ascii in (ord('p'), ord('P')):
            autopilot_mode = not autopilot_mode
            state_msg = 'AKTIF (Otomatik Tarama)' if autopilot_mode else 'KAPALI'
            print(f'🤖 Otopilot {state_msg}.')
            cmd_received = True

        if cmd_received:
            last_cmd_time = time.time()
        else:
            # ✅ DJI Tarzı Otomatik Havada Kalma (Auto-Hover)
            target_vx *= 0.10
            target_vy *= 0.10
            target_vz *= 0.10
            target_wz *= 0.10

        # ✅ OTOPİLOT VE ÇARPIŞMA ÖNLEYİCİ (Collision Avoidance)
        if autopilot_mode:
            # Sabit Haritalama Uçuşu (0.8 m/s)
            target_vx = 0.8
            target_vy = 0.0
            target_vz = 0.0
            target_wz = 0.0
            
            # Duvarlardan Uzaklaşma (Tünel Ortalama)
            if lidar_min_left < 2.0:
                target_vy = -0.5
            elif lidar_min_right < 2.0:
                target_vy = 0.5
                
        # Manuel veya Otopilot fark etmez, Önde engel varsa DUR
        if target_vx > 0 and lidar_min_front < 1.2:
            target_vx = 0.0  # İleri gitmeyi engelle
            if autopilot_mode:
                autopilot_mode = False
                print('⚠️ LİDAR UYARISI: Önde engel var, Otopilot durduruldu!')

        # Yumuşak ivmelenme, ama sert frenleme
        cur_vx = 0.40 * cur_vx + 0.60 * target_vx
        cur_vy = 0.40 * cur_vy + 0.60 * target_vy
        cur_vz = 0.40 * cur_vz + 0.60 * target_vz
        cur_wz = 0.40 * cur_wz + 0.60 * target_wz
        
        # MAVLink değişkenlerini güncelle
        ap_vx = cur_vx
        ap_vy = cur_vy
        ap_vz = cur_vz
        ap_yaw_rate = cur_wz

        # Zamanlayıcı (artık GZ Velocity Control yok)
        time.sleep(0.01)

        if key_ascii in (ord('r'), ord('R')):
            if not recording:
                recording = True
                recorded_frames = []
                print('🔴 [TARAMA BASLADI] Magara ici kareler hafizaya aliniyor...')
            else:
                recording = False
                total_rec = len(recorded_frames)
                print(f'⏹️ [TARAMA BITTI] Toplam {total_rec} kare alindi.')
                send_teleop_cmd(0, 0, 0, 0, 0)

                if total_rec < 15:
                    print('⚠️ UYARI: Kayit cok kisa oldu (<15 kare). Lutfen biraz daha ucarak tarayin.')
                else:
                    temp_video_path = os.path.join(CURR_DIR, 'temp_cave_scan.mp4')
                    print(f'💾 Video hazirlaniyor: {temp_video_path}')
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    # ✅ Boyut DÜZELTMESİ: recorded_frames ham kamera kareleri içeriyor
                    # (800×600), display frame'i değil (1280×720). VideoWriter boyutu
                    # ilk karenin gerçek boyutundan alınmalı – aksi hâlde video bozuk
                    # çıkıyor ve MASt3R 0 keyframe buluyor.
                    raw_h, raw_w = recorded_frames[0].shape[:2]
                    writer = cv2.VideoWriter(temp_video_path, fourcc, 30, (raw_w, raw_h))
                    for f in recorded_frames:
                        writer.write(f)
                    writer.release()

                    cv2.destroyAllWindows()

                    print('=' * 70)
                    print('⚡ MAST3R-3DGS CALISIYOR: GERCEK MAGARA 3B GAUSSIAN MODELI URETILIYOR...')
                    print('=' * 70)

                    from mast3r_to_3dgs import build_gaussian_splats_from_mast3r
                    out_ply = build_gaussian_splats_from_mast3r(
                        video_file=temp_video_path,
                        output_ply='cave_scene.ply',
                        num_keyframes=min(target_keyframes, total_rec),
                        target_size=512
                    )

                    if out_ply and os.path.exists(out_ply):
                        print('🚀 MAGARA 3B MODELI HAZIR! 144+ FPS Goruntuleyici aciliyor...')
                        import subprocess
                        renderer_path = os.path.join(CURR_DIR, 'gaussian_renderer.py')
                        subprocess.run([sys.executable, renderer_path, 'cave_scene.ply'])
                    return

    cv2.destroyAllWindows()

if __name__ == '__main__':
    kfs = int(sys.argv[1]) if len(sys.argv) > 1 else 45
    run_cave_capture(target_keyframes=kfs)
