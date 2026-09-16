#!/bin/bash
echo "🧹 Tüm simülasyon, PX4, köprü ve QGC süreçleri temizleniyor..."
pkill -9 -f "drone_lidar_odometry_bridge.py" 2>/dev/null || true
pkill -9 -f "drone_cpu_hover_bridge.py" 2>/dev/null || true
pkill -9 -f "px4" 2>/dev/null || true
pkill -9 -f "gz sim" 2>/dev/null || true
pkill -9 -f "ruby" 2>/dev/null || true
pkill -9 -f "QGroundControl" 2>/dev/null || true
rm -f /tmp/px4_lock* /tmp/px4-sock* /tmp/px4_sitl.log 2>/dev/null || true
sleep 1
echo "✅ Tüm portlar (14540, 14550, 14580) ve süreçler serbest bırakıldı!"
