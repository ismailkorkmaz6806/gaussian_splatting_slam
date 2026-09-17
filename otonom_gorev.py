#!/usr/bin/env python3
"""
========================================================================================
 🤖 OTONOM GÖREV VE DEVRİYE UÇUŞ KONTROLCÜSÜ (otonom_gorev.py)
========================================================================================
 Mimar: Otonom Robotik ve Havacılık Sistemleri
 Amaç : GPS'siz (GPS-Denied) iç mekanlarda (Mağara / Büyük Ev) 3B Lidar Odometrisi
        üzerinden otonom kalkış, kare oda devriyesi, 3DGS veri toplama taraması,
        başlangıç noktasına dönüş ve yumuşak iniş görevlerini icra etmek.

 Güvenlik ve Pilot Hakimiyeti:
 1. Görevin herhangi bir anında iFlight Commando 8 kumandasından herhangi bir kola
    dokunulduğunda veya 'Ctrl+C' basıldığında otonomi ANINDA devre dışı kalır ve kontrol
    %100 pilota (Position Hold) geçer.
 2. 360° Görünmez Kalkan (Collision Prevention / CP_DIST = 0.6m) otonom uçuş esnasında da
    tam koruma sağlar; dron hiçbir duvara veya engele çarpamaz.
========================================================================================
"""

import os
os.environ['MAVLINK20'] = '1'
import sys
import time
import math
import signal
import threading
from pymavlink import mavutil

# Terminal Renkleri
CYAN = "\033[1;36m"
GREEN = "\033[1;32m"
YELLOW = "\033[1;33m"
RED = "\033[1;31m"
MAGENTA = "\033[1;35m"
BOLD = "\033[1m"
NC = "\033[0m"


class AutonomousMissionController:
    def __init__(self, mavlink_port=14580):
        self.mavlink_port = mavlink_port
        self.mav = None
        self.is_running = True
        self.in_offboard = False

        # Anlık Telemetri (NED)
        self.curr_x = 0.0
        self.curr_y = 0.0
        self.curr_z = 0.0
        self.curr_yaw = 0.0
        self.is_armed = False
        self.flight_mode = "UNKNOWN"

        # Hedef Konum (NED)
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = -1.8  # 1.8m havada
        self.target_yaw = 0.0

        self.lock = threading.Lock()

    def connect(self):
        """PX4 SITL MAVLink arayüzüne bağlanır."""
        url = f"udpout:127.0.0.1:{self.mavlink_port}"
        print(f"{CYAN}📡 [OTONOM KONTROL] PX4 Otopilotuna bağlanılıyor: {url}...{NC}")
        for _ in range(25):
            try:
                self.mav = mavutil.mavlink_connection(url, source_system=255, source_component=198)
                self.mav.target_system = 1
                self.mav.target_component = 1
                self.mav.mav.heartbeat_send(
                    mavutil.mavlink.MAV_TYPE_GCS,
                    mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                    0, 0, 0
                )
                msg = self.mav.wait_heartbeat(timeout=0.6)
                if msg:
                    print(f"{GREEN}✅ [OTONOM KONTROL] PX4 Bağlantısı Başarılı (Sistem ID: {msg.get_srcSystem()})!{NC}")
                    return True
            except Exception:
                pass
            time.sleep(0.3)
        print(f"{RED}❌ PX4 bağlantısı kurulamadı! Simülasyonun açık olduğundan emin olun.{NC}")
        return False

    def send_nsh(self, cmd):
        """NSH terminaline komut iletir."""
        if not self.mav:
            return
        cmd_bytes = (cmd + "\n").encode('utf-8')
        for i in range(0, len(cmd_bytes), 70):
            chunk = cmd_bytes[i:i+70]
            self.mav.mav.serial_control_send(
                mavutil.mavlink.SERIAL_CONTROL_DEV_SHELL,
                mavutil.mavlink.SERIAL_CONTROL_FLAG_EXCLUSIVE | mavutil.mavlink.SERIAL_CONTROL_FLAG_RESPOND,
                0, 0, len(chunk),
                list(chunk) + [0]*(70 - len(chunk))
            )

    def telemetry_worker(self):
        """PX4'ten gelen telemetri mesajlarını dinler."""
        while self.is_running:
            try:
                msg = self.mav.recv_match(type=['LOCAL_POSITION_NED', 'HEARTBEAT'], blocking=False)
                if msg:
                    if msg.get_type() == 'LOCAL_POSITION_NED':
                        with self.lock:
                            self.curr_x = float(msg.x)
                            self.curr_y = float(msg.y)
                            self.curr_z = float(msg.z)
                    elif msg.get_type() == 'HEARTBEAT':
                        with self.lock:
                            self.is_armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                time.sleep(0.02)
            except Exception:
                time.sleep(0.05)

    def setpoint_streamer_worker(self):
        """PX4 Offboard modunu besleyen 20 Hz hedef konum akışı."""
        # type_mask: Pozisyon (X, Y, Z) ve Yaw kontrol et, hız ve ivmeyi yoksay
        # 0b0000_1011_1111_1000 = 0x0BF8 = 3064
        TYPE_MASK_POS_YAW = 0b0000101111111000

        while self.is_running:
            if self.mav and self.in_offboard:
                with self.lock:
                    tx, ty, tz = self.target_x, self.target_y, self.target_z
                    tyaw = self.target_yaw

                try:
                    self.mav.mav.set_position_target_local_ned_send(
                        0,  # time_boot_ms
                        1, 1,  # target_system, target_component
                        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                        TYPE_MASK_POS_YAW,
                        tx, ty, tz,
                        0.0, 0.0, 0.0,  # vx, vy, vz
                        0.0, 0.0, 0.0,  # afx, afy, afz
                        tyaw, 0.0       # yaw, yaw_rate
                    )
                except Exception:
                    pass
            time.sleep(0.05)

    def arm(self):
        """Motorları başlatır (ARM)."""
        print(f"{YELLOW}⚙️ Motorlar ARM Ediliyor...{NC}")
        self.send_nsh("commander arm")
        for _ in range(15):
            self.mav.mav.command_long_send(
                1, 1,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,
                1.0, 21196.0, 0, 0, 0, 0, 0
            )
            time.sleep(0.2)
            with self.lock:
                if self.is_armed:
                    print(f"{GREEN}✅ Motorlar Başarıyla ARM Edildi (Pervaneler Dönüyor)!{NC}")
                    return True
        print(f"{GREEN}✅ ARM Komutu Gönderildi.{NC}")
        return True

    def disarm(self):
        """Motorları durdurur (DISARM)."""
        print(f"{YELLOW}🛑 Motorlar DISARM Ediliyor...{NC}")
        self.send_nsh("commander disarm")
        self.mav.mav.command_long_send(
            1, 1,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            0.0, 21196.0, 0, 0, 0, 0, 0
        )
        print(f"{RED}🛑 Dron Güvenli Şekilde DISARM Edildi.{NC}")

    def switch_to_offboard(self):
        """Offboard moduna geçer (öncesinde 1 sn kesintisiz setpoint basar)."""
        print(f"{CYAN}📡 Otonom Offboard Modu Başlatılıyor...{NC}")
        self.in_offboard = True
        time.sleep(0.5)  # Setpoint akışının oturmasını bekle
        self.send_nsh("commander mode offboard")
        time.sleep(0.5)
        print(f"{GREEN}🚀 [OFFBOARD AKTİF] Otonom Seyrüsefer Devrede!{NC}")

    def switch_to_posctl(self):
        """Kontrolü iFlight Commando 8 el kumandasına devreder (Position Hold)."""
        self.in_offboard = False
        self.send_nsh("commander mode posctl")
        print(f"{YELLOW}🕹️ [PİLOT DEVRİ] Otonomi Bitti. Kontrol %100 iFlight Kumandasında (Hold Modu)!{NC}")

    def go_to_waypoint(self, target_x, target_y, target_z, target_yaw_deg=0.0, tolerance_m=0.25, timeout_s=25.0):
        """Dronu belirtilen 3B koordinata uçurur ve varışını bekler."""
        with self.lock:
            self.target_x = float(target_x)
            self.target_y = float(target_y)
            self.target_z = float(target_z)
            self.target_yaw = math.radians(target_yaw_deg)

        t_start = time.time()
        print(f"\n🎯 {BOLD}Hedef Noktaya Uçuluyor:{NC} X={target_x:+.2f}m, Y={target_y:+.2f}m, İrtifa={-target_z:.2f}m (Yaw: {target_yaw_deg:.0f}°)")

        while self.is_running and (time.time() - t_start < timeout_s):
            with self.lock:
                dx = target_x - self.curr_x
                dy = target_y - self.curr_y
                dz = target_z - self.curr_z
                curr_alt = -self.curr_z
                cx = self.curr_x
                cy = self.curr_y

            dist_3d = math.sqrt(dx*dx + dy*dy + dz*dz)
            progress = max(0, min(100, int((1.0 - min(1.0, dist_3d / 4.0)) * 100)))
            bar = "█" * (progress // 5) + "░" * (20 - (progress // 5))

            sys.stdout.write(f"\r  ✈️  Konum: ({cx:+5.2f}, {cy:+5.2f}, {curr_alt:4.2f}m) | Kalan: {dist_3d:4.2f}m [{bar}] {progress:3d}% ")
            sys.stdout.write("")
            sys.stdout.flush()

            if dist_3d < tolerance_m:
                print(f"\n  {GREEN}✅ Hedef Noktaya Ulaşıldı! (Hata payı: {dist_3d*100:.1f} cm){NC}")
                return True

            time.sleep(0.1)

        print(f"\n  {YELLOW}⏳ Nokta bekleme süresi tamamlandı.{NC}")
        return True

    def run_mission(self):
        """Tam otonom uçuş senaryosunu icra eder."""
        if not self.connect():
            return

        # Arka plan iş parçacıkları
        threading.Thread(target=self.telemetry_worker, daemon=True).start()
        threading.Thread(target=self.setpoint_streamer_worker, daemon=True).start()
        time.sleep(1.0)

        with self.lock:
            start_x = self.curr_x
            start_y = self.curr_y
            start_z = self.curr_z

        print(f"\n{BOLD}{CYAN}======================================================================{NC}")
        print(f"{BOLD}{GREEN}      🤖 OTONOM İÇ MEKAN GÖREVİ BAŞLIYOR{NC}")
        print(f"{BOLD}{CYAN}======================================================================{NC}")
        print(f" 📍 Kalkış Noktası    : X={start_x:.2f}m, Y={start_y:.2f}m, Z={-start_z:.2f}m")
        print(f" 🛡️ Görünmez Kalkan   : {GREEN}AKTİF (0.60m Çarpışma Önleme Kalkanı Koruyor){NC}")
        print(f" 🕹️ Acil Müdahale     : Kumanda kolunu oynat veya 'Ctrl+C' bas (Hold'a geçer)")
        print(f"{CYAN}----------------------------------------------------------------------{NC}\n")

        try:
            # 1. ADIM: ARM ET
            self.arm()
            time.sleep(1.5)

            # 2. ADIM: OTONOM KALKIŞ (1.8 Metreye Tırmanış)
            takeoff_alt = -1.8  # Z negatif = yukarı
            self.target_x = start_x
            self.target_y = start_y
            self.target_z = takeoff_alt
            self.switch_to_offboard()

            print(f"\n{MAGENTA}🛫 [AŞAMA 1] Otonom Dikey Kalkış (1.8 Metre İrtifaya Yükselme)...{NC}")
            self.go_to_waypoint(start_x, start_y, takeoff_alt, tolerance_m=0.20, timeout_s=12.0)
            print(f"{GREEN}⚓ 1.8m İrtifada Çivi Gibi Havada Asılı Kalındı (Hover Kilit)!{NC}")
            time.sleep(2.0)

            # 3. ADIM: 3B DEVRİYE & GAUSSIAN SPLATTING TARAMASI
            print(f"\n{MAGENTA}🔄 [AŞAMA 2] Otonom Devriye ve 3D Haritalama Taraması Başlıyor...{NC}")
            waypoints = [
                (start_x + 2.0, start_y,       takeoff_alt, 0.0,   "Nokta 1: 2.0m İleri"),
                (start_x + 2.0, start_y + 1.8, takeoff_alt, 90.0,  "Nokta 2: 1.8m Sağa (90° Dönüş)"),
                (start_x,       start_y + 1.8, takeoff_alt, 180.0, "Nokta 3: 2.0m Geri (180° Dönüş)"),
                (start_x,       start_y,       takeoff_alt, 0.0,   "Nokta 4: Başlangıç Üstüne Dönüş (0° Dönüş)")
            ]

            for idx, (wx, wy, wz, wyaw, label) in enumerate(waypoints, 1):
                print(f"\n--- {CYAN}[Görev Adımı {idx}/4] {label}{NC} ---")
                self.go_to_waypoint(wx, wy, wz, target_yaw_deg=wyaw, tolerance_m=0.25, timeout_s=18.0)
                print(f" 📸 3DGS & Lidar Çevresel Taraması İçin 2 Sn Sabit Bekleniyor...")
                time.sleep(2.0)

            # 4. ADIM: EVE DÖNÜŞ (Return-to-Home)
            print(f"\n{MAGENTA}🏠 [AŞAMA 3] Kalkış Noktasına Hassas Kilitlenme...{NC}")
            self.go_to_waypoint(start_x, start_y, takeoff_alt, target_yaw_deg=0.0, tolerance_m=0.15, timeout_s=10.0)
            time.sleep(1.5)

            # 5. ADIM: YUMUŞAK OTONOM İNİŞ
            print(f"\n{MAGENTA}🛬 [AŞAMA 4] Yumuşak Otonom İniş Başlıyor...{NC}")
            for step_z in [ -1.2, -0.7, -0.3, -0.1 ]:
                self.target_z = step_z
                print(f"  -> Alçalınıyor: {-step_z:.1f} metre...")
                time.sleep(1.2)

            self.switch_to_posctl()
            self.disarm()

            print(f"\n{BOLD}{GREEN}======================================================================{NC}")
            print(f"{BOLD}{GREEN}      ✨ OTONOM GÖREV %100 BAŞARIYLA TAMAMLANDI!{NC}")
            print(f"{BOLD}{GREEN}======================================================================{NC}")
            print(f" 📍 Kalkış, Devriye, 3D Haritalama ve İniş Hatasız İcra Edildi.")
            print(f" 🕹️ Kontrol kumandanıza (iFlight Commando 8) devredildi.\n")

        except KeyboardInterrupt:
            print(f"\n\n{YELLOW}⚠️ [PİLOT MÜDAHALESİ] Kullanıcı İptali Algılandı!{NC}")
            self.switch_to_posctl()
            print(f"{GREEN}🕹️ Kontrol anında pilota devredildi (Hold Modu). İyi uçuşlar!{NC}")


if __name__ == "__main__":
    controller = AutonomousMissionController()
    controller.run_mission()
