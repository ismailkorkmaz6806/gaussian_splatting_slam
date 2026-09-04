#!/bin/bash
# ========================================================================================
# 🤖 GAZEBO SIM 3B DRON VE CANLI HARİTALAMA SİSTEMİ - UBUNTU (calistir_gazebo_simulasyon.sh)
# ========================================================================================

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo "============================================================="
echo "  🤖 GAZEBO SIM 3B DRON VE CANLI HARİTALAMA (UBUNTU)"
echo "============================================================="
echo ""

echo " [1/3] Gazebo Sim Başlatılıyor..."
if which gz > /dev/null; then
    gz sim -r "$DIR/gazebo_tunnel_drone.sdf" &
    GZ_PID=$!
elif which gazebo > /dev/null; then
    gazebo "$DIR/gazebo_tunnel_drone.sdf" &
    GZ_PID=$!
else
    echo "⚠️ Gazebo komutu bulunamadı. Lütfen gazebo veya gz kurulu olduğundan emin olun."
fi

echo " [2/3] Dron Kamera Köprüsü Başlatılıyor..."
python3 "$DIR/gz_camera_bridge.py" &
BRIDGE_PID=$!

echo " [3/3] Canlı Kamera Arayüzüne Bağlanılıyor..."
sleep 3
python3 "$DIR/live_drone_capture.py" http://127.0.0.1:8554/drone_stream

# Temizlik
kill $GZ_PID $BRIDGE_PID 2>/dev/null || true