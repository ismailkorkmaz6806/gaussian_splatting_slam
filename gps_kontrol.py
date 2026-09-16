import time
from pymavlink import mavutil

print("📡 PX4 Otopilotuna bağlanılıyor...")
mav = mavutil.mavlink_connection('udpout:127.0.0.1:18570', source_system=253)
mav.wait_heartbeat(timeout=5)

print("🔍 EKF2_GPS_CTRL parametresi okunuyor...")
mav.mav.param_request_read_send(1, 1, b'EKF2_GPS_CTRL', -1)
msg = mav.recv_match(type='PARAM_VALUE', blocking=True, timeout=5)

if msg:
    deger = int(msg.param_value)
    print("="*50)
    if deger == 0:
        print(f"✅ SONUÇ: EKF2_GPS_CTRL = {deger} -> GPS TAMAMEN KAPALI (Sistem Kör)")
    elif deger == 7:
        print(f"❌ SONUÇ: EKF2_GPS_CTRL = {deger} -> GPS AÇIK (Sistem GPS kullanıyor)")
    else:
        print(f"ℹ️ SONUÇ: EKF2_GPS_CTRL = {deger}")
    print("="*50)
else:
    print("❌ HATA: Otopilottan cevap alınamadı. Simülasyon açık mı?")
