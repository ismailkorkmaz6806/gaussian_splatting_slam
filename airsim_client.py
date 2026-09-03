import socket
import struct
import time
import os
import sys
import numpy as np
import cv2
import msgpack

class AirSimRPCClient:
    """
    Python 3.14 uyumlu, sifir harici bagimlilikli saf MessagePack RPC Istemcisi.
    """
    def __init__(self, ip="127.0.0.1", port=41451, timeout=10.0):
        self.ip = ip
        self.port = port
        self.timeout = timeout
        self.sock = None
        self.msg_id = 0

    def connect(self):
        """AirSim simulatorune TCP soketi ile baglanir."""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(self.timeout)
            self.sock.connect((self.ip, self.port))
            print(f" [OK] AirSim Simulatorune Baglanildi ({self.ip}:{self.port})")
            return True
        except Exception as e:
            print(f" [X] AirSim Baglanti Hatasi: {e}")
            print(" -> Lutfen once Unreal Engine / AirSim simulasyon dunyasini (.exe) baslatin.")
            self.sock = None
            return False

    def close(self):
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None

    def call(self, method_name, *args):
        """AirSim API fonksiyonunu cagirir ve cevabi dondurur."""
        if not self.sock:
            if not self.connect():
                return None

        self.msg_id += 1
        payload = msgpack.packb([0, self.msg_id, method_name, list(args)], use_bin_type=True)
        try:
            self.sock.sendall(payload)
            raw_resp = self.sock.recv(1024 * 1024 * 4)
            unpacker = msgpack.Unpacker()
            unpacker.feed(raw_resp)
            for msg in unpacker:
                if isinstance(msg, (list, tuple)) and len(msg) == 4:
                    msg_type, reply_id, err, result = msg
                    if err:
                        print(f" [!] AirSim RPC Hatasi ({method_name}): {err}")
                        return None
                    return result
        except Exception as e:
            print(f" [X] Iletisim Hatasi ({method_name}): {e}")
            self.sock = None
            return None

    def enable_api_control(self, vehicle_name=""):
        return self.call("enableApiControl", True, vehicle_name)

    def arm_disarm(self, arm=True, vehicle_name=""):
        return self.call("armDisarm", arm, vehicle_name)

    def takeoff(self, timeout_sec=5.0, vehicle_name=""):
        print(" -> Dron Havalaniyor (Takeoff)...")
        return self.call("takeoff", timeout_sec, vehicle_name)

    def land(self, timeout_sec=10.0, vehicle_name=""):
        print(" -> Dron Inis Yapiyor (Land)...")
        return self.call("land", timeout_sec, vehicle_name)

    def move_to_position(self, x, y, z, velocity=1.5, vehicle_name=""):
        return self.call("moveToPosition", x, y, z, velocity, 10.0, 0, [False, 0.0], -1, 1, vehicle_name)

    def get_camera_image_bgr(self, camera_name="front_center", vehicle_name=""):
        raw_bytes = self.call("simGetImage", camera_name, 0, vehicle_name)
        if raw_bytes and len(raw_bytes) > 0:
            np_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            return img
        return None

def record_airsim_tunnel_flight(output_video="airsim_tunnel_scan.mp4", duration_sec=18, fps=30):
    client = AirSimRPCClient()
    if not client.connect():
        print("\n[!] AirSim simulatoru calismiyor! Lutfen once AirSim dunyasini (.exe) acin.")
        return None

    client.enable_api_control()
    client.arm_disarm(True)
    client.takeoff(3.0)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = None

    print(f"\n -> Sanal Tunel Ucusu Basladi! Video Kaydediliyor: {output_video}")
    t0 = time.time()
    frame_count = 0

    target_x = 0.0
    while time.time() - t0 < duration_sec:
        elapsed = time.time() - t0
        target_x += 0.25
        target_y = float(np.sin(elapsed * 0.8) * 0.8)
        target_z = -1.30

        client.move_to_position(target_x, target_y, target_z, velocity=1.2)

        img = client.get_camera_image_bgr("front_center")
        if img is not None:
            if writer is None:
                h, w = img.shape[:2]
                writer = cv2.VideoWriter(output_video, fourcc, fps, (w, h))
            writer.write(img)
            frame_count += 1
            if frame_count % 15 == 0:
                print(f"  -> Kaydedilen Kare: {frame_count} ({elapsed:.1f}s / {duration_sec}s)")

        time.sleep(1.0 / fps)

    if writer:
        writer.release()
    client.land()
    client.arm_disarm(False)
    client.close()

    print(f"\n [OK] AirSim Ucus Kaydi Tamamlandi! ({frame_count} Kare)")
    return output_video

if __name__ == "__main__":
    out_v = record_airsim_tunnel_flight()
    if out_v and os.path.exists(out_v):
        print(f"\n [OK] 3DGS Motoru Baslatiliyor...")
        import mast3r_to_3dgs
        out_ply = mast3r_to_3dgs.build_gaussian_splats_from_mast3r(out_v, 50)
        if out_ply and os.path.exists(out_ply):
            import subprocess
            subprocess.run([sys.executable, "gaussian_renderer.py", out_ply])

