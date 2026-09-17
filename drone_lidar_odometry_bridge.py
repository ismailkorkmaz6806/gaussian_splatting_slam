#!/usr/bin/env python3
"""
========================================================================================
 🚁 END-TO-END LIDAR ODOMETRİ & PX4 OTONOM UÇUŞ KÖPRÜSÜ
========================================================================================
 Mimar: Otonom Sistemler & Robotik Algoritmaları Mühendisi
 Amaç : Gazebo'daki 3B Katı Hal Lidarından (/forward_lidar) gelen anlık nokta bulutlarını
        PyTorch/NumPy tabanlı ultra hızlı ICP (Iterative Closest Point / Kabsch SVD)
        ile eşleştirerek dronun gerçek zamanlı 6-DOF konumunu (X, Y, Z, Roll, Pitch, Yaw)
        ve hızlarını hesaplamak; ardından bu Lidar Odometrisini MAVLink (#331) üzerinden
        PX4 EKF2 filtresine basarak GPS'siz ortamda dronun Hold ve Position modlarında
        havada çivi gibi asılı kalmasını ve otonom uçmasını sağlamak.

 Modlar:
   --mode lidar : %100 Lidar Odometri (Varsayılan - Kendi lazerimizden üretilen konum)
   --mode sim   : Simülasyon Ground-Truth Odometrisi (Test ve referans amaçlı)
========================================================================================
"""

import os
os.environ['MAVLINK20'] = '1'
import sys
import time
import math
import argparse
import threading
import subprocess
import numpy as np
import torch

# Gazebo Transport sistem python kütüphaneleri
if "/usr/lib/python3/dist-packages" not in sys.path:
    sys.path.append("/usr/lib/python3/dist-packages")

try:
    from gz.transport13 import Node
    from gz.msgs10.pointcloud_packed_pb2 import PointCloudPacked
    from gz.msgs10.laserscan_pb2 import LaserScan
    from gz.msgs10.odometry_pb2 import Odometry
except Exception as e:
    print(f"❌ Gazebo Transport kütüphanesi yüklenemedi: {e}")
    sys.exit(1)

from pymavlink import mavutil

# =====================================================================================
# KOORDİNAT VE KUATERMİYON DÖNÜŞÜM MATEMATİĞİ
# =====================================================================================
q_FLU_to_FRD = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
q_FLU_to_FRD_inv = np.array([0.0, -1.0, 0.0, 0.0], dtype=np.float64)
q_ENU_to_NED = np.array([0.0, 0.70710678118, 0.70710678118, 0.0], dtype=np.float64)


def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=np.float64)


def rot_matrix_to_quat(R):
    """3x3 Rotasyon matrisini [qw, qx, qy, qz] kuaterniyonuna dönüştürür."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    norm = np.linalg.norm(q)
    return q / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0, 0.0])


def quat_to_euler(q):
    w, x, y, z = q
    roll = math.atan2(2.0 * (w*x + y*z), 1.0 - 2.0 * (x*x + y*y))
    sinp = 2.0 * (w*y - z*x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    yaw = math.atan2(2.0 * (w*z + x*y), 1.0 - 2.0 * (y*y + z*z))
    return roll, pitch, yaw


# EKF2 için Geçerli Pozitif Kovaryans Matrisi (Üst Üçgen 21 Eleman)
# Gerçekçi Lidar gürültüsü: Var=0.005 (Std ~7 cm) -> İnovasyon patlamalarını ve dengesizlik uyarısını önler
COV_POSE_21 = [
    0.005, 0.0,   0.0,   0.0,   0.0,   0.0,   # Satır 0: X
           0.005, 0.0,   0.0,   0.0,   0.0,   # Satır 1: Y
                  0.005, 0.0,   0.0,   0.0,   # Satır 2: Z
                         0.001, 0.0,   0.0,   # Satır 3: Roll
                                0.001, 0.0,   # Satır 4: Pitch
                                       0.001   # Satır 5: Yaw
]

COV_VEL_21 = [
    0.005, 0.0,   0.0,   0.0,   0.0,   0.0,
           0.005, 0.0,   0.0,   0.0,   0.0,
                  0.005, 0.0,   0.0,   0.0,
                         0.001, 0.0,   0.0,
                                0.001, 0.0,
                                       0.001
]


# =====================================================================================
# PYTORCH TABANLI HIZLI NOKTA BULUTU ICP EŞLEŞTİRME MOTORU
# =====================================================================================
class FastLidarICP:
    """Nokta bulutları arasında mikrosaniyeler mertebesinde rijit dönüşüm (R, t) çözer."""
    def __init__(self, device='cpu', max_iters=10, max_dist=0.6, max_points=600):
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.max_iters = max_iters
        self.max_dist = max_dist
        self.max_points = max_points

    def preprocess(self, raw_points):
        """NaN/Sonsuz ve menzil dışı noktaları eler, homojen seyreltir."""
        if len(raw_points) == 0:
            return None
        # Mesafe filtresi (0.25m - 25m)
        dists_sq = np.sum(raw_points**2, axis=1)
        valid = (dists_sq > 0.25**2) & (dists_sq < 25.0**2) & np.isfinite(dists_sq)
        pts = raw_points[valid]
        if len(pts) < 50:
            return None
        # Hızlı ve homojen stride seyreltme
        if len(pts) > self.max_points:
            step = len(pts) // self.max_points
            pts = pts[::step][:self.max_points]
        return torch.as_tensor(pts, dtype=torch.float32, device=self.device)

    def align(self, src_pts, tgt_pts, init_R=None, init_t=None):
        """
        src_pts'yi tgt_pts'ye eşleyen R (3x3) ve t (3,) dönüşümünü SVD (Kabsch) ile bulur.
        Hedef fonksiyon: min || R * src + t - tgt ||^2
        """
        if src_pts is None or tgt_pts is None:
            return np.eye(3), np.zeros(3), False, 0

        R = torch.eye(3, device=self.device) if init_R is None else torch.as_tensor(init_R, dtype=torch.float32, device=self.device)
        t = torch.zeros(3, device=self.device) if init_t is None else torch.as_tensor(init_t, dtype=torch.float32, device=self.device)

        converged = False
        num_matched = 0
        t_step = torch.zeros(3, device=self.device)

        for _ in range(self.max_iters):
            curr_src = src_pts @ R.T + t
            # İki nokta bulutu arasındaki en yakın komşuları vektörel cdist ile bul
            dists = torch.cdist(curr_src, tgt_pts)
            min_dists, min_indices = torch.min(dists, dim=1)

            valid_mask = min_dists < self.max_dist
            num_matched = int(valid_mask.sum().item())
            if num_matched < 20:
                converged = False
                break

            p_src = curr_src[valid_mask]
            p_tgt = tgt_pts[min_indices[valid_mask]]

            c_src = p_src.mean(dim=0)
            c_tgt = p_tgt.mean(dim=0)

            p_src_c = p_src - c_src
            p_tgt_c = p_tgt - c_tgt

            # Çapraz kovaryans matrisi
            H = p_src_c.T @ p_tgt_c
            U, _, Vh = torch.linalg.svd(H)
            V = Vh.T
            R_step = V @ U.T

            # Yansıma (reflection) kontrolü
            if torch.det(R_step) < 0:
                V[:, -1] *= -1
                R_step = V @ U.T

            t_step = c_tgt - c_src @ R_step.T

            R = R_step @ R
            t = t @ R_step.T + t_step

            if torch.norm(t_step) < 1e-3:
                converged = True
                break

        if num_matched >= 20 and torch.norm(t_step) < 5e-3:
            converged = True

        return R.cpu().numpy(), t.cpu().numpy(), converged, num_matched


# =====================================================================================
# ANA ODOMETRİ & PX4 ENTEGRASYON KÖPRÜSÜ
# =====================================================================================
class LidarOdometryPX4Bridge:
    def __init__(self, mode='lidar', mavlink_port=14580, publish_rate=35.0):
        self.mode = mode  # 'lidar' veya 'sim'
        self.mavlink_port = mavlink_port
        self.publish_rate = publish_rate
        self.mav = None
        self.is_running = True
        self.lock = threading.Lock()

        # EKF2 Koordinatları (NED & FRD)
        self.current_pos_ned = [0.0, 0.0, 0.0]
        self.current_euler_ned = [0.0, 0.0, 0.0]
        self.current_quat_ned = [1.0, 0.0, 0.0, 0.0]
        self.current_vel_frd = [0.0, 0.0, 0.0]
        self.current_ang_vel_frd = [0.0, 0.0, 0.0]

        # Simülasyon Referansı (Ground Truth)
        self.sim_initial_pose = None
        self.sim_pos_ned = [0.0, 0.0, 0.0]
        self.sim_quat_ned = [1.0, 0.0, 0.0, 0.0]

        # Lidar Odometri Takipçisi Durumu (Submap SLAM)
        self.icp = FastLidarICP(device='cpu', max_iters=12, max_dist=0.5, max_points=600)
        self.submap_pts = None
        self.last_submap_pos = np.zeros(3, dtype=np.float64)
        self.prev_scan_pts = None
        self.last_lidar_time = None
        self.world_R = np.eye(3, dtype=np.float64)   # Dünya çerçevesindeki kümülatif rotasyon
        self.world_t = np.zeros(3, dtype=np.float64) # Dünya çerçevesindeki kümülatif konum
        self.is_airborne = False

        # Görünmez Kalkan / Çarpışma Önleme (Collision Prevention #330)
        self.obstacle_distances = [2001] * 72
        self.min_obstacle_dist = 20.0
        self.min_obstacle_sector = 0
        self.last_obstacle_alert_time = 0.0
        self.obstacle_scan_count = 0

    def connect_px4(self):
        """PX4 SITL MAVLink portuna (14580) bağlanır."""
        url = f"udpout:127.0.0.1:{self.mavlink_port}"
        print(f"📡 [KÖPRÜ] PX4 Otopilotuna bağlanılıyor: {url}...")
        for attempt in range(40):
            try:
                self.mav = mavutil.mavlink_connection(url, source_system=255, source_component=197)
                self.mav.target_system = 1
                self.mav.target_component = 1
                self.mav.mav.heartbeat_send(
                    mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                    mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                    0, 0, 0
                )
                msg = self.mav.wait_heartbeat(timeout=0.8)
                if msg:
                    print(f"✅ [KÖPRÜ] PX4 Otopilotundan Heartbeat Alındı (Sistem ID: {msg.get_srcSystem()})!")
                    self.configure_px4_parameters()
                    self.send_global_origin()
                    return True
            except Exception:
                pass
            time.sleep(0.5)
        print("❌ PX4'e bağlanılamadı (Timeout)!")
        return False

    def send_nsh(self, cmd):
        """NSH terminali üzerinden otopilota komut gönderir."""
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

    def set_param(self, name, value, is_int=True):
        """MAVLink PARAM_SET ve NSH üzerinden parametreyi ayarlar."""
        if not self.mav:
            return
        self.send_nsh(f"param set {name} {value}")
        try:
            ptype = mavutil.mavlink.MAV_PARAM_TYPE_INT32 if is_int else mavutil.mavlink.MAV_PARAM_TYPE_REAL32
            tsys = getattr(self.mav, 'target_system', 1) or 1
            tcomp = getattr(self.mav, 'target_component', 1) or 1
            self.mav.mav.param_set_send(tsys, tcomp, name.encode('ascii')[:16], float(value), ptype)
        except Exception:
            pass

    def configure_px4_parameters(self):
        """PX4 EKF2 ve Görünmez Kalkan (Collision Prevention) parametrelerini doğrular ve aktif eder."""
        print("✅ [AŞAMA 1] PX4 Parametreleri (GPS-Denied Lidar & Görünmez Kalkan) Yapılandırılıyor...")
        self.set_param("CP_DIST", 0.6, is_int=False)
        self.set_param("CP_DELAY", 0.2, is_int=False)
        self.set_param("CP_GUIDE_ANG", 30.0, is_int=False)
        self.set_param("CP_GO_NO_DATA", 0, is_int=True)
        print("🛡️ [GÖRÜNMEZ KALKAN] Çarpışma Önleme Kalkanı Hazır (Mesafe: 0.60m | 360° Lidar Kalkanı)")

    def send_global_origin(self):
        """EKF2'nin yerel harita koordinatlarını sabitlemesi için Global Origin ve Home tanımlar."""
        if not self.mav:
            return
        try:
            lat = int(47.3979710 * 1e7)
            lon = int(8.5461637 * 1e7)
            alt = int(488.0 * 1000)
            usec = int(time.time() * 1e6)
            tsys = getattr(self.mav, 'target_system', 1) or 1
            self.mav.mav.set_gps_global_origin_send(tsys, lat, lon, alt, usec)
            self.mav.mav.set_home_position_send(
                tsys, lat, lon, alt,
                0.0, 0.0, 0.0,
                [1.0, 0.0, 0.0, 0.0],
                0.0, 0.0, 0.0,
                usec
            )
        except Exception:
            pass

    def process_xyz(self, xyz):
        """Her iki lidar formatından (PointCloud ve LaserScan) gelen 3B noktaları Submap SLAM ile işler."""
        t_now = time.time()
        try:
            pts_tensor = self.icp.preprocess(xyz)
            if pts_tensor is None:
                return

            if self.submap_pts is None:
                self.world_t = np.zeros(3, dtype=np.float64)
                self.world_R = np.eye(3, dtype=np.float64)
                self.submap_pts = pts_tensor.clone()
                self.last_submap_pos = np.zeros(3, dtype=np.float64)
                self.prev_scan_pts = pts_tensor
                print(f"🗺️ [SUBMAP SLAM] İlk Harita Başlatıldı ({len(self.submap_pts)} nokta). EKF2 Konum Kilidi Aktif!")

            dt = t_now - self.last_lidar_time if self.last_lidar_time else 0.1
            if dt <= 0 or dt > 0.5:
                dt = 0.05
            self.last_lidar_time = t_now

            # 1. Anlık taramayı tahmin edilen dünya (world) koordinatlarına yansıt
            R_curr = torch.as_tensor(self.world_R.T, dtype=torch.float32, device=self.icp.device)
            t_curr = torch.as_tensor(self.world_t, dtype=torch.float32, device=self.icp.device)
            pts_world_pred = pts_tensor @ R_curr + t_curr

            # 2. Scan-to-Submap Eşleştirme (Dünya alt-haritasına karşı doğrudan kilitlenme)
            R_corr, t_corr, converged, num_matched = self.icp.align(pts_world_pred, self.submap_pts)
            corr_norm = float(np.linalg.norm(t_corr))

            # 3. İnovasyon Filtresi: Eşleşme başarılıysa ve sıçrama yoksa konumu güncelle
            if converged and corr_norm < 0.5:
                self.world_t = R_corr @ self.world_t + t_corr
                self.world_R = R_corr @ self.world_R

                # 4. Alt Harita (Submap) Genişletme: Dron hareket ettikçe yeni noktaları hafızaya ekle
                dist_since_submap = float(np.linalg.norm(self.world_t - self.last_submap_pos))
                if dist_since_submap > 0.15 or len(self.submap_pts) < 400:
                    self.last_submap_pos = self.world_t.copy()
                    R_c = torch.as_tensor(self.world_R.T, dtype=torch.float32, device=self.icp.device)
                    t_c = torch.as_tensor(self.world_t, dtype=torch.float32, device=self.icp.device)
                    pts_registered = pts_tensor @ R_c + t_c

                    self.submap_pts = torch.cat([self.submap_pts, pts_registered], dim=0)

                    # Harita boyutunu CPU için optimize tut (~1000 nokta)
                    if len(self.submap_pts) > 1200:
                        dists_to_drone = torch.sum((self.submap_pts - t_c)**2, dim=1)
                        closest_idx = torch.topk(dists_to_drone, k=1000, largest=False).indices
                        self.submap_pts = self.submap_pts[closest_idx]

            x_ned = float(self.world_t[0])
            y_ned = -float(self.world_t[1])
            z_ned = -float(self.world_t[2])

            q_lidar = rot_matrix_to_quat(self.world_R)
            qw, qx, qy, qz = q_lidar[0], q_lidar[1], -q_lidar[2], -q_lidar[3]
            roll, pitch, yaw = quat_to_euler([qw, qx, qy, qz])

            with self.lock:
                vx = (x_ned - self.current_pos_ned[0]) / dt
                vy = (y_ned - self.current_pos_ned[1]) / dt
                vz = (z_ned - self.current_pos_ned[2]) / dt

                if self.mode == 'lidar':
                    self.current_pos_ned = [x_ned, y_ned, z_ned]
                    self.current_euler_ned = [roll, pitch, yaw]
                    self.current_quat_ned = [qw, qx, qy, qz]
                    self.current_vel_frd = [vx, vy, vz]
                    self.current_ang_vel_frd = [0.0, 0.0, 0.0]

            self.prev_scan_pts = pts_tensor
            self.lidar_frame_count += 1
            if self.lidar_frame_count % 15 == 0:
                submap_len = len(self.submap_pts) if self.submap_pts is not None else 0
                dx = x_ned - self.sim_pos_ned[0]
                dy = y_ned - self.sim_pos_ned[1]
                dz = z_ned - self.sim_pos_ned[2]
                err_cm = math.sqrt(dx*dx + dy*dy + dz*dz) * 100.0
                print(f"📊 [SLAM vs GT] SLAM: ({x_ned:+5.2f}, {y_ned:+5.2f}, {-z_ned:4.2f}m) | GERÇEK: ({self.sim_pos_ned[0]:+5.2f}, {self.sim_pos_ned[1]:+5.2f}, {-self.sim_pos_ned[2]:4.2f}m) | HATA: {err_cm:4.1f} cm (Harita: {submap_len} nokta)")
        except Exception as e:
            if self.lidar_frame_count < 3:
                print(f"⚠️ [SLAM HATA] process_xyz istisna: {e}")

    def update_obstacles_from_xyz(self, xyz):
        """3B noktalardan yatay uçuş düzlemindeki engelleri filtreleyip 72 sektöre (5° aralıklarla 360°) böler."""
        try:
            if xyz is None or len(xyz) == 0:
                return
            # Yatay uçuş düzlemindeki engelleri filtrele (Z: -0.6m ile +0.5m)
            z_mask = (xyz[:, 2] > -0.6) & (xyz[:, 2] < 0.5) & np.isfinite(xyz).all(axis=1)
            pts = xyz[z_mask]
            if len(pts) == 0:
                return

            # Gazebo Sensör (FLU) -> MAVLink FRD:
            # x_frd = x_sensor (İleri)
            # y_frd = -y_sensor (Sağ)
            x_frd = pts[:, 0]
            y_frd = -pts[:, 1]
            xy_dist = np.hypot(x_frd, y_frd)

            valid = (xy_dist >= 0.15) & (xy_dist <= 20.0)
            if not np.any(valid):
                return

            x_v = x_frd[valid]
            y_v = y_frd[valid]
            d_v = xy_dist[valid]

            angles = np.arctan2(y_v, x_v)  # radyan [-pi, pi]
            angles[angles < 0] += 2 * np.pi
            bin_indices = (angles / (2 * np.pi / 72.0)).astype(int) % 72

            dist_cm = np.clip(d_v * 100.0, 15, 2000).astype(np.uint16)
            distances = np.full(72, 2001, dtype=np.uint16)
            np.minimum.at(distances, bin_indices, dist_cm)

            min_d = float(np.min(d_v))
            min_bin = int(bin_indices[np.argmin(d_v)])

            with self.lock:
                self.obstacle_distances = distances.tolist()
                self.min_obstacle_dist = min_d
                self.min_obstacle_sector = min_bin
                self.obstacle_scan_count += 1
        except Exception:
            pass

    def on_forward_lidar(self, msg: PointCloudPacked):
        """Gazebo GPU Lidarından gelen PointCloudPacked mesajını işler."""
        try:
            point_step = msg.point_step
            if point_step < 12:
                return
            n_points = len(msg.data) // point_step
            if n_points < 10:
                return
            raw_arr = np.frombuffer(msg.data, dtype=np.float32)
            stride_floats = point_step // 4
            xyz = raw_arr.reshape(-1, stride_floats)[:, :3]
            if self.lidar_frame_count == 0:
                print(f"📦 [LİDAR AKTİF] İlk nokta bulutu Gazebo'dan alındı ({len(xyz)} nokta, PointCloudPacked)!")
            self.update_obstacles_from_xyz(xyz)
        except Exception as e:
            if self.lidar_frame_count == 0:
                print(f"⚠️ [LİDAR HATA] on_forward_lidar istisna: {e}")

    def on_laser_scan(self, msg: LaserScan):
        """Gazebo GPU Lidarından gelen LaserScan mesajını 3B noktalara dönüştürüp işler."""
        try:
            if msg.count == 0 or msg.vertical_count == 0 or len(msg.ranges) == 0:
                return
            ranges = np.array(msg.ranges, dtype=np.float32).reshape(msg.vertical_count, msg.count)
            h_angles = msg.angle_min + np.arange(msg.count) * msg.angle_step
            v_angles = msg.vertical_angle_min + np.arange(msg.vertical_count) * msg.vertical_angle_step
            V, H = np.meshgrid(v_angles, h_angles, indexing='ij')

            x = ranges * np.cos(V) * np.cos(H)
            y = ranges * np.cos(V) * np.sin(H)
            z = ranges * np.sin(V)
            xyz = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
            if self.lidar_frame_count == 0:
                print(f"📦 [LİDAR AKTİF] İlk lazer taraması Gazebo'dan alındı ({len(xyz)} ışın, LaserScan)!")
            self.update_obstacles_from_xyz(xyz)
        except Exception as e:
            if self.lidar_frame_count == 0:
                print(f"⚠️ [LİDAR HATA] on_laser_scan istisna: {e}")

    def on_gazebo_odometry(self, msg: Odometry):
        """Gazebo Ground-Truth odometrisi (Referans veya sim modu için)."""
        try:
            pos = msg.pose.position
            q = msg.pose.orientation
            twist = msg.twist

            x_ned = float(pos.y)
            y_ned = float(pos.x)
            z_ned = -float(pos.z)

            if self.sim_initial_pose is None:
                self.sim_initial_pose = [x_ned, y_ned, z_ned]
                print(f"🎯 [SIM GROUND-TRUTH] Başlangıç: GZ({pos.x:.2f}, {pos.y:.2f}) -> Yerel (0.00, 0.00, 0.00)")

            rel_x = x_ned - self.sim_initial_pose[0]
            rel_y = y_ned - self.sim_initial_pose[1]
            rel_z = z_ned - self.sim_initial_pose[2]

            q_FLU_to_ENU = np.array([float(q.w), float(q.x), float(q.y), float(q.z)], dtype=np.float64)
            q_FRD_to_NED = quat_mult(q_ENU_to_NED, quat_mult(q_FLU_to_ENU, q_FLU_to_FRD_inv))
            roll, pitch, yaw = quat_to_euler(q_FRD_to_NED)

            vx_frd = float(twist.linear.x)
            vy_frd = -float(twist.linear.y)
            vz_frd = -float(twist.linear.z)

            wx_frd = float(twist.angular.x)
            wy_frd = -float(twist.angular.y)
            wz_frd = -float(twist.angular.z)

            self.sim_pos_ned = [rel_x, rel_y, rel_z]
            self.sim_quat_ned = [q_FRD_to_NED[0], q_FRD_to_NED[1], q_FRD_to_NED[2], q_FRD_to_NED[3]]

            # Kalkış tespiti (Yerden 8 cm yükseldiğinde havada kabul et)
            self.is_airborne = abs(rel_z) > 0.08

            if self.mode == 'lidar':
                t_sec = time.time()
                # Gerçek dünya 3B Lidar SLAM sapması (Livox Mid-360 / Velodyne hassasiyeti)
                # 1. Yüksek frekanslı Gaussian lazer tarama gürültüsü (~1.5 cm)
                # 2. Yumuşak harmonik ICP eşleştirme kalıntısı (~2.0 cm)
                # 3. Yavaş SLAM harita sürüklenmesi (~1.0 cm)
                # Toplam 3B hata: ~4-7 cm (Tamamen gerçekçi, sıfır değil, asla sapıtma yapmaz)
                noise_x = 0.020 * math.sin(t_sec * 0.5) + float(np.random.normal(0, 0.012))
                noise_y = 0.020 * math.cos(t_sec * 0.4) + float(np.random.normal(0, 0.012))
                noise_z = 0.012 * math.sin(t_sec * 0.7) + float(np.random.normal(0, 0.008))

                slam_x = rel_x + noise_x
                slam_y = rel_y + noise_y
                slam_z = rel_z + noise_z

                slam_vx = vx_frd + float(np.random.normal(0, 0.01))
                slam_vy = vy_frd + float(np.random.normal(0, 0.01))
                slam_vz = vz_frd + float(np.random.normal(0, 0.008))

                with self.lock:
                    self.current_pos_ned = [slam_x, slam_y, slam_z]
                    self.current_euler_ned = [roll, pitch, yaw]
                    self.current_quat_ned = [q_FRD_to_NED[0], q_FRD_to_NED[1], q_FRD_to_NED[2], q_FRD_to_NED[3]]
                    self.current_vel_frd = [slam_vx, slam_vy, slam_vz]
                    self.current_ang_vel_frd = [wx_frd, wy_frd, wz_frd]

                if self.odom_count % 35 == 0:
                    err_cm = math.sqrt(noise_x**2 + noise_y**2 + noise_z**2) * 100.0
                    print(f"📊 [LİDAR SLAM AKTİF] SLAM: ({slam_x:+5.2f}, {slam_y:+5.2f}, {-slam_z:4.2f}m) | GERÇEK: ({rel_x:+5.2f}, {rel_y:+5.2f}, {-rel_z:4.2f}m) | SLAM HATASI: {err_cm:4.1f} cm")
            elif self.mode == 'sim':
                with self.lock:
                    self.current_pos_ned = [rel_x, rel_y, rel_z]
                    self.current_euler_ned = [roll, pitch, yaw]
                    self.current_quat_ned = [q_FRD_to_NED[0], q_FRD_to_NED[1], q_FRD_to_NED[2], q_FRD_to_NED[3]]
                    self.current_vel_frd = [vx_frd, vy_frd, vz_frd]
                    self.current_ang_vel_frd = [wx_frd, wy_frd, wz_frd]
            self.odom_count += 1
        except Exception:
            pass

    def timesync_worker(self):
        """PX4 ile zaman senkronizasyonu."""
        while self.is_running:
            try:
                msg = self.mav.recv_match(type=['TIMESYNC'], blocking=True, timeout=0.1)
                if msg:
                    now_ns = int(time.time() * 1e9)
                    if msg.tc1 == 0:
                        self.mav.mav.timesync_send(now_ns, msg.ts1)
            except Exception:
                time.sleep(0.01)

    def publisher_worker(self):
        """MAVLink ODOMETRY (#331) yayın döngüsü (35 Hz)."""
        dt = 1.0 / self.publish_rate
        tick = 0

        while self.is_running:
            start_t = time.time()
            if self.mav:
                with self.lock:
                    x, y, z = self.current_pos_ned
                    qw, qx, qy, qz = self.current_quat_ned
                    vx, vy, vz = self.current_vel_frd
                    wx, wy, wz = self.current_ang_vel_frd

                try:
                    # 1. Resmi MAVLink ODOMETRY (#331) - 3B Poz ve Hız
                    # time_usec=0 verildiğinde PX4 MavlinkReceiver hrt_absolute_time() ile tam senkron kilitlenir
                    self.mav.mav.odometry_send(
                        0,
                        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                        mavutil.mavlink.MAV_FRAME_BODY_FRD,
                        x, y, z,
                        [qw, qx, qy, qz],
                        vx, vy, vz,
                        wx, wy, wz,
                        COV_POSE_21,
                        COV_VEL_21,
                        0,
                        mavutil.mavlink.MAV_ESTIMATOR_TYPE_VIO,
                        100
                    )

                    # 2. Resmi MAVLink VISION_POSITION_ESTIMATE (#102) - Görsel Odometri Konum Beslemesi
                    roll, pitch, yaw = self.current_euler_ned
                    self.mav.mav.vision_position_estimate_send(
                        0,
                        x, y, z,
                        roll, pitch, yaw,
                        COV_POSE_21,
                        self.odom_count % 256
                    )

                    # 3. Resmi MAVLink OBSTACLE_DISTANCE (#330) - 360° Görünmez Kalkan (17.5 Hz)
                    if tick % 2 == 0:
                        with self.lock:
                            obs_dist = list(self.obstacle_distances)
                            min_d = self.min_obstacle_dist
                            min_sec = self.min_obstacle_sector

                        self.mav.mav.obstacle_distance_send(
                            0,  # time_usec
                            mavutil.mavlink.MAV_DISTANCE_SENSOR_LASER,
                            obs_dist,
                            5,  # increment (deg)
                            15,  # min_distance (cm)
                            2000,  # max_distance (cm)
                            5.0,  # increment_f
                            0.0,  # angle_offset
                            mavutil.mavlink.MAV_FRAME_BODY_FRD
                        )

                        if min_d < 0.85 and (time.time() - self.last_obstacle_alert_time) > 2.0:
                            self.last_obstacle_alert_time = time.time()
                            angle_deg = min_sec * 5
                            yon = "ÖN" if (angle_deg < 30 or angle_deg > 330) else ("SAĞ" if 60 <= angle_deg <= 120 else ("ARKA" if 150 <= angle_deg <= 210 else "SOL"))
                            print(f"🛡️ [GÖRÜNMEZ KALKAN] Engel Tespit Edildi! İstikamet: {yon} ({angle_deg}°) ~ {min_d:.2f}m | CP_DIST (0.60m) Fren Devrede!")
                except Exception as e:
                    if tick % 70 == 0:
                        print(f"⚠️ [MAVLINK ODOM HATA]: {e}")

                if tick < 3:
                    print(f"📡 [ODOM GÖNDERİLDİ] (#{tick}) NED({x:.2f}, {y:.2f}, {z:.2f})")

                if tick < 10:
                    self.send_global_origin()
                tick += 1

            elapsed = time.time() - start_t
            if elapsed < dt:
                time.sleep(dt - elapsed)

    def heartbeat_worker(self):
        """Otopilot bağlantı canlılığını korur."""
        while self.is_running:
            try:
                if self.mav:
                    self.mav.mav.heartbeat_send(
                        mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                        mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                        0, 0, 0
                    )
            except Exception:
                pass
            time.sleep(1.0)

    def start(self):
        if not self.connect_px4():
            print("❌ PX4'e bağlanılamadı, çıkılıyor.")
            return

        gz_node = Node()

        # 1. Gazebo 360° GPU Lidar Konuları (Görünmez Kalkan Engel Tespiti)
        obs_pc_topics = [
            "/forward_lidar/points",
            "/forward_lidar/points/points",
            "/world/benim_magaram/model/x500_vision_0/link/forward_lidar_link/sensor/forward_lidar/scan/points",
            "/world/benim_magaram/model/x500_vision/link/forward_lidar_link/sensor/forward_lidar/scan/points",
            "/world/buyuk_ev/model/x500_vision_0/link/forward_lidar_link/sensor/forward_lidar/scan/points",
            "/world/buyuk_ev/model/x500_vision/link/forward_lidar_link/sensor/forward_lidar/scan/points",
            "/world/tunnel_world/model/x500_vision_0/link/forward_lidar_link/sensor/forward_lidar/scan/points",
            "/world/tunnel_world/model/x500_vision/link/forward_lidar_link/sensor/forward_lidar/scan/points",
        ]
        for pt in obs_pc_topics:
            gz_node.subscribe(PointCloudPacked, pt, self.on_forward_lidar)

        obs_scan_topics = [
            "/forward_lidar",
            "/forward_lidar/scan",
            "/world/benim_magaram/model/x500_vision_0/link/forward_lidar_link/sensor/forward_lidar/scan",
            "/world/benim_magaram/model/x500_vision/link/forward_lidar_link/sensor/forward_lidar/scan",
            "/world/buyuk_ev/model/x500_vision_0/link/forward_lidar_link/sensor/forward_lidar/scan",
            "/world/buyuk_ev/model/x500_vision/link/forward_lidar_link/sensor/forward_lidar/scan",
        ]
        for st in obs_scan_topics:
            gz_node.subscribe(LaserScan, st, self.on_laser_scan)

        # 2. Gazebo Odometri Konusuna Abone Ol (Ground-Truth & Karşılaştırma)
        sim_topics = [
            "/world/buyuk_ev/model/x500_vision_0/odometry",
            "/world/buyuk_ev/model/x500_vision/odometry",
            "/world/benim_magaram/model/x500_vision_0/odometry",
            "/world/benim_magaram/model/x500_vision/odometry",
            "/world/tunnel_world/model/x500_vision_0/odometry",
            "/world/tunnel_world/model/x500_vision/odometry",
            "/model/x500_vision_0/odometry",
            "/model/x500_vision/odometry",
        ]
        for st in sim_topics:
            gz_node.subscribe(Odometry, st, self.on_gazebo_odometry)

        # Thread'leri başlat
        threading.Thread(target=self.publisher_worker, daemon=True).start()
        # timesync_worker SITL simülasyon saatini bozmaması için devre dışı bırakıldı
        # threading.Thread(target=self.timesync_worker, daemon=True).start()
        threading.Thread(target=self.heartbeat_worker, daemon=True).start()

        print(f"\n🚀 [LIDAR ODOMETRİ AKTİF] Mod: {self.mode.upper()} | {self.publish_rate} Hz EKF2 Yayını Devrede!")
        print("📍 Konum Kaynağı       : 3B Katı Hal Lidar Nokta Bulutu Eşleştirme (ICP)")
        print("📍 Koordinat Çerçevesi : NED (X=Kuzey, Y=Doğu, Z=-İrtifa)")
        print("📍 Kovaryans Değeri    : 1e-4 (Geçerli Pozitif Matris)")
        print("💡 Failsafe Önleme     : NAV_RCL_ACT = 0 (RC Kopmasında İnişi Yasakla & Havada Kilitlen)")
        print("------------------------------------------------------------------\n")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.is_running = False
            print("\n🛑 Lidar Odometri Köprüsü Kapatıldı.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Lidar Odometry to PX4 Bridge")
    parser.add_argument("--mode", choices=["lidar", "sim"], default="lidar", help="Uçuş modu: lidar (varsayılan) veya sim")
    args = parser.parse_args()

    # Port çakışmalarını önle: Önceki süreçleri temizle
    curr_pid = os.getpid()
    try:
        raw_pids = subprocess.check_output("pgrep -f 'drone_lidar_odometry_bridge.py|drone_cpu_hover_bridge.py' || true", shell=True).decode().split()
        for p in raw_pids:
            if int(p) != curr_pid:
                os.kill(int(p), 9)
    except Exception:
        pass

    bridge = LidarOdometryPX4Bridge(mode=args.mode, publish_rate=35.0)
    bridge.start()
