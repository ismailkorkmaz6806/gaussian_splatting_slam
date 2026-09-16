"""
========================================================================================
PX4 MAVLink Vision Poz ve Odometri Köprüsü (px4_mavlink_bridge.py)
========================================================================================
Bu modül, 3D Gaussian Splatting / SLAM veya Kamera Poz takip motorundan gelen anlık
3B kamera konumunu (X, Y, Z, Roll, Pitch, Yaw) MAVLink VISION_POSITION_ESTIMATE
mesajlarına dönüştürerek PX4 SITL otopilotuna iletir.

Özellikler:
1. UDP 14540 (PX4 SITL Offboard / Vision portu) üzerinden sıfır gecikmeli haberleşir.
2. OpenCV / SLAM kamera koordinat sistemini (X-sağ, Y-aşağı, Z-ileri)
   PX4 Havacılık NED koordinat sistemine (X-ileri, Y-sağ, Z-aşağı) dönüştürür.
3. Arka planda 30 Hz hızında EKF2'yi besleyerek GPS'siz havada asılı kalmayı (Position Hold) sağlar.
========================================================================================
"""

import time
import math
import threading
import numpy as np

try:
    from pymavlink import mavutil
    HAS_PYMAVLINK = True
except ImportError:
    HAS_PYMAVLINK = False


class PX4VisionBridge:
    def __init__(self, connection_url="udpout:127.0.0.1:14580", publish_rate_hz=30):
        """
        PX4 MAVLink Görsel Odometri Köprüsü
        Args:
            connection_url: PX4 SITL MAVLink adresi (Varsayılan: 127.0.0.1:14580)
            publish_rate_hz: EKF2'ye poz basma frekansı (Varsayılan: 30 Hz)
        """
        self.connection_url = connection_url
        self.publish_rate = publish_rate_hz
        self.mav = None
        self.is_connected = False
        self.is_running = False
        
        # Son bilinen poz (NED formatında: x, y, z, roll, pitch, yaw)
        self._current_ned_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self._lock = threading.Lock()
        self._thread = None
        self.gps_enabled = True
        self.vision_streaming = True

    def connect(self):
        """PX4 SITL otopilotuna bağlanır."""
        if not HAS_PYMAVLINK:
            print("⚠️ UYARI: 'pymavlink' kütüphanesi bulunamadı! Lütfen 'pip install pymavlink' çalıştırın.")
            return False

        try:
            print(f" 📡 PX4 MAVLink Köprüsü Bağlanıyor: {self.connection_url} ...")
            self.mav = mavutil.mavlink_connection(self.connection_url, source_system=255, source_component=197)
            self.is_connected = True
            print(" ✅ PX4 MAVLink Görsel Odometri Köprüsü Aktif (Port: 14580)!")
            self.configure_indoor_preflight()
            return True
        except Exception as e:
            print(f"❌ PX4 MAVLink Bağlantı Hatası: {e}")
            self.is_connected = False
            return False

    def set_px4_param_int(self, param_name, param_value):
        """PX4 EKF2 parametresini anlık olarak MAVLink üzerinden değiştirir."""
        if not self.is_connected or self.mav is None:
            return False
        try:
            param_id = param_name.encode('utf-8')[:16]
            # Hem autopilot component'e (1) hem broadcast (0) gönder
            self.mav.mav.param_set_send(
                1, # target_system
                1, # target_component
                param_id,
                float(param_value),
                mavutil.mavlink.MAV_PARAM_TYPE_INT32
            )
            # NSH Konsoluna doğrudan 'param set' komutu da bas
            self.send_nsh_command(f"param set {param_name} {int(param_value)}")
            return True
        except Exception as e:
            print(f"⚠️ MAVLink Parametre Değiştirme Hatası ({param_name}): {e}")
            return False

    def send_nsh_command(self, cmd_str):
        """PX4 NSH terminaline sıfır gecikmeli MAVLink shell komutu iletir."""
        if not self.is_connected or self.mav is None:
            return
        try:
            data = (cmd_str + "\n").encode('utf-8')
            payload = list(data) + [0] * max(0, 70 - len(data))
            self.mav.mav.serial_control_send(
                mavutil.mavlink.SERIAL_CONTROL_DEV_SHELL,
                mavutil.mavlink.SERIAL_CONTROL_FLAG_RESPOND | mavutil.mavlink.SERIAL_CONTROL_FLAG_EXCLUSIVE,
                0, 0,
                min(70, len(data)),
                payload[:70]
            )
        except Exception:
            pass

    def arm(self, force=True):
        """Dronu ARM eder (Motorları çalıştırır)."""
        if not self.is_connected or self.mav is None:
            return False
        try:
            force_param = 21196.0 if force else 0.0
            self.mav.mav.command_long_send(
                1, 1,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,
                1.0, force_param, 0.0, 0.0, 0.0, 0.0, 0.0
            )
            # NSH üzerinden de doğrudan tetikle (sıfır gecikme)
            self.send_nsh_command("commander arm -f" if force else "commander arm")
            print(" 🚀 [PX4 MAVLink]: ARM Komutu Gönderildi!")
            return True
        except Exception as e:
            print(f"❌ ARM Hatası: {e}")
            return False

    def disarm(self):
        """Dronu DISARM eder."""
        if not self.is_connected or self.mav is None:
            return False
        try:
            self.mav.mav.command_long_send(
                1, 1,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            )
            print(" 🛑 [PX4 MAVLink]: DISARM Komutu Gönderildi!")
        except Exception as e:
            print(f"❌ DISARM Hatası: {e}")
            return False

    def takeoff(self, altitude=1.5):
        """GPS olmadan 1.5 metreye otonom kalkış yapar ve çivi gibi asılı kalır."""
        if not self.is_connected or self.mav is None:
            return False
        try:
            # Önce Altitude moduna geçir (Yerde kilitlenmeyi engeller)
            self.send_nsh_command("commander mode altctl")
            time.sleep(0.3)
            self.arm(force=True)
            time.sleep(0.5)
            self.send_nsh_command("commander takeoff")
            self.mav.mav.command_long_send(
                1, 1,
                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                float(altitude)
            )
            print(f" 🛫 [PX4 MAVLink]: {altitude}m GPS-Siz Kalkış Komutu Gönderildi!")
            return True
        except Exception as e:
            print(f"❌ Kalkış Hatası: {e}")
            return False

    def configure_indoor_preflight(self):
        print(" ⚙️  [PX4] GPS İptal Ediliyor ve LIDAR (VIO) Otonomi Ayarları Zorlanıyor...")
        # Lidar Odometri
        self.send_nsh_command("param set SYS_FAILURE_EN 1")
        self.send_nsh_command("failure gps off")
        self.send_nsh_command("param set EKF2_GPS_CTRL 0")
        self.send_nsh_command("param set EKF2_EV_CTRL 15")
        self.send_nsh_command("param set NAV_RCL_ACT 0")
        self.send_nsh_command("param set NAV_DLL_ACT 0")
        
        # Kalkış / Arm İzinleri (Hayati Önem Taşıyor)
        self.send_nsh_command("param set COM_ARM_WO_GPS 1")   # GPS yokken ARM (Motor Çalıştırma) izni
        self.send_nsh_command("param set COM_ARM_MAG_STR 0")  # Pusula gücü kontrolünü kapat
        self.send_nsh_command("param set COM_ARM_MAG_ANG -1") # Pusula açısı kontrolünü kapat
        self.send_nsh_command("param set EKF2_MAG_CHECK 0")   # Sensör uyuşmazlığı hatasını yoksay
        
        self.send_nsh_command("param set COM_RC_IN_MODE 3")
        self.send_nsh_command("param set COM_RCL_EX_T 10")
        print(" ⚙️  [PX4] Lidar Odometri Kurulumu ve ARM İzinleri Tamamlandı.")

    def set_gps_enabled(self, enable: bool):
        """GPS'i otopilotta açar (7) veya tamamen devre dışı bırakır (0)."""
        val = 7 if enable else 0
        self.gps_enabled = enable
        
        # 1. Yöntem: Parametre Güncelleme (EKF2_GPS_CTRL)
        self.set_px4_param_int("EKF2_GPS_CTRL", val)
        self.send_nsh_command(f"param set EKF2_GPS_CTRL {val}")
        
        # GPS Kapatıldığında VIO'nun açık olduğundan emin ol
        if not enable:
            self.set_vision_enabled(True)
        
        # 2. Yöntem: NSH ve MAVLink üzerinden GPS Donanımını Tamamen Kapat (QGC'de uydu sayısı 0 olsun)
        self.send_nsh_command("param set SYS_FAILURE_EN 1")
        if enable:
            self.send_nsh_command("failure gps ok")
        else:
            self.send_nsh_command("failure gps off")

        durum = "AÇIK (7 - GPS Fix Aktif)" if enable else "KAPALI (0 - GPS-Denied Modu Devrede)"
        print(f" 🛰️ [PX4 EKF2 GPS]: {durum}")

    def set_vision_enabled(self, enable: bool):
        """Görsel Odometriyi (VIO) EKF2'de açar (15) veya devre dışı bırakır (0)."""
        val = 15 if enable else 0
        self.vision_streaming = enable
        self.set_px4_param_int("EKF2_EV_CTRL", val)
        durum = "AÇIK (15 - 30 Hz EKF2 Görsel Kilit)" if enable else "KAPALI (0)"
        print(f" 📷 [PX4 EKF2 VIO]: {durum}")

    def update_slam_pose(self, x_cam, y_cam, z_cam, roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0):
        """
        SLAM / Kamera eksenindeki pozu alır ve PX4 NED eksenine çevirir.
        Kamera Ekseni (OpenCV): X-Sağ, Y-Aşağı, Z-İleri
        PX4 Havacılık (NED):   X-İleri, Y-Sağ, Z-Aşağı
        """
        # Koordinat Dönüşümü:
        # x_ned = z_cam (İleri)
        # y_ned = x_cam (Sağ)
        # z_ned = y_cam (Aşağı)
        x_ned = float(z_cam)
        y_ned = float(x_cam)
        z_ned = float(y_cam)

        roll_rad = math.radians(roll_deg)
        pitch_rad = math.radians(pitch_deg)
        yaw_rad = math.radians(yaw_deg)

        with self._lock:
            self._current_ned_pose = [x_ned, y_ned, z_ned, roll_rad, pitch_rad, yaw_rad]

    def send_global_origin(self):
        """PX4'e harita başlangıç noktasını (Global Origin) tanımlar (GPS olmasa bile)."""
        if not self.is_connected or self.mav is None:
            return
        try:
            lat = int(47.3979710 * 1e7)
            lon = int(8.5461637 * 1e7)
            alt = int(488.0 * 1000)
            usec = int(time.time() * 1e6)
            self.mav.mav.set_gps_global_origin_send(1, lat, lon, alt, usec)
            self.mav.mav.set_home_position_send(
                1, lat, lon, alt,
                0.0, 0.0, 0.0,
                [1.0, 0.0, 0.0, 0.0],
                0.0, 0.0, 0.0,
                usec
            )
        except Exception:
            pass

    def send_pose_now(self):
        """MAVLink VISION_POSITION_ESTIMATE ve ODOMETRY mesajlarını basar."""
        if not self.is_connected or self.mav is None or not self.vision_streaming:
            return

        with self._lock:
            x, y, z, roll, pitch, yaw = self._current_ned_pose

        usec = int(time.time() * 1e6)
        try:
            # 1. VISION_POSITION_ESTIMATE (Message #102)
            self.mav.mav.vision_position_estimate_send(
                usec,
                x, y, z,
                roll, pitch, yaw,
                [0.01] * 21
            )

            # 2. ODOMETRY (Message #331) - EKF2 Tam Poz ve Hız Senkronizasyonu
            # Euler açılarını Kuaterniyona çevir
            cy = math.cos(yaw * 0.5)
            sy = math.sin(yaw * 0.5)
            cp = math.cos(pitch * 0.5)
            sp = math.sin(pitch * 0.5)
            cr = math.cos(roll * 0.5)
            sr = math.sin(roll * 0.5)
            qw = cr * cp * cy + sr * sp * sy
            qx = sr * cp * cy - cr * sp * sy
            qy = cr * sp * cy + sr * cp * sy
            qz = cr * cp * sy - sr * sp * cy

            self.mav.mav.odometry_send(
                usec,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                mavutil.mavlink.MAV_FRAME_BODY_FRD,
                x, y, z,
                [qw, qx, qy, qz],
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                [0.01] * 21,
                [0.01] * 21,
                0,
                mavutil.mavlink.MAV_ESTIMATOR_TYPE_VIO
            )
        except Exception:
            pass

    def _worker(self):
        """30 Hz frekansta düzenli poz basan arka plan iş parçacığı."""
        interval = 1.0 / self.publish_rate
        tick = 0
        while self.is_running:
            self.send_pose_now()
            # Başlangıçta sadece 3 kez Global Origin gönder (Terminali kirletmez)
            if tick < 3:
                self.send_global_origin()
            tick += 1
            time.sleep(interval)

    def start_streaming(self):
        """Arka plan yayın thread'ini başlatır."""
        if not self.is_connected:
            if not self.connect():
                return False

        self.is_running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        print(f" 🚀 EKF2 Vision Odometry Yayını Başlatıldı ({self.publish_rate} Hz)")
        return True

    def stop(self):
        """Köprüyü durdurur."""
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        print(" 🛑 PX4 Vision Köprüsü Durduruldu.")


# Bağımsız Test
if __name__ == "__main__":
    print("=" * 60)
    print(" 🚁 PX4 Vision Odometry Köprüsü Test Modu")
    print("=" * 60)
    bridge = PX4VisionBridge()
    if bridge.connect():
        bridge.start_streaming()
        print(" -> 10 saniye boyunca (X=1.0m, Y=0.0m, Z=-2.5m) sabit poz basılıyor...")
        bridge.update_slam_pose(x_cam=0.0, y_cam=-2.5, z_cam=1.0)
        time.sleep(10)
        bridge.stop()
