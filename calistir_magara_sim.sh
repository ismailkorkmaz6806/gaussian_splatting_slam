#!/bin/bash
# ========================================================================================
# 🏭 DARPA Terk Edilmiş Büyük Maden Simülasyonu & 3DGS Haritalama Başlatıcı
# ========================================================================================

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

if [ -f "$DIR/venv/bin/activate" ]; then
    source "$DIR/venv/bin/activate"
fi

# NVIDIA RTX Donanım Hızlandırması
export __NV_PRIME_RENDER_OFFLOAD=1
export __GL_SYNC_TO_VBLANK=0
export vblank_mode=0



# Gazebo Transport Yerel İletişim Değişkenleri
export GZ_IP=127.0.0.1
export GZ_PARTITION=default
export GZ_SIM_RESOURCE_PATH="$DIR/gazebo_cave_world/worlds/models:$HOME/ardupilot_gazebo/models:$HOME/ardupilot_gazebo/worlds:$GZ_SIM_RESOURCE_PATH"
export GZ_SIM_SYSTEM_PLUGIN_PATH="$HOME/ardupilot_gazebo/build:$GZ_SIM_SYSTEM_PLUGIN_PATH"

HEADLESS_FLAG="-s"
WORLD_FILE="$DIR/gazebo_cave_world/worlds/large_mine_3dgs.sdf"
WORLD_NAME="🏭 BÜYÜK TERK EDİLMİŞ MADEN (lc_mine)"

for arg in "$@"; do
    if [ "$arg" == "--gui" ] || [ "$arg" == "-g" ]; then
        HEADLESS_FLAG=""
    fi
    if [ "$arg" == "--cave" ] || [ "$arg" == "-c" ]; then
        WORLD_FILE="$DIR/gazebo_cave_world/worlds/cave_world_3dgs.sdf"
        WORLD_NAME="🦇 DARPA SubT MAGARA DÜNYASI"
    fi
done

echo "============================================================="
echo "  $WORLD_NAME"
echo "  3DGS HARITALAMA SIMÜLASYONU"
if [ -n "$HEADLESS_FLAG" ]; then
    echo "  ⚡ Mod: FPV KOKPİT (Gazebo Arka Planda)"
else
    echo "  🖥️  Mod: FULL GUI (Gazebo 3B Arayüzü Açık)"
fi
echo "  Dünya: $(basename $WORLD_FILE)"
echo "============================================================="
echo ""

# Eski takılı kalmış simülasyon süreçlerini temizle
killall -9 gz sim 2>/dev/null || true
pkill -f "ardupilot_drone_capture.py" 2>/dev/null || true
pkill -f "sim_vehicle.py" 2>/dev/null || true
pkill -f "arducopter" 2>/dev/null || true

echo " [1/3] Gazebo Sim Dünyası Başlatılıyor..."
if which gz > /dev/null; then
    gz sim $HEADLESS_FLAG -r "$WORLD_FILE" &
    GZ_PID=$!
else
    echo "⚠️ Gazebo (gz) komutu bulunamadı. Lütfen gz kurulu olduğundan emin olun."
    exit 1
fi

sleep 2

echo " [2/3] ArduPilot SITL (Sanal Uçuş Bilgisayarı) Başlatılıyor..."
# sim_vehicle.py ile ardupilot'u JSON model üzerinden Gazebo'ya bağla
python3 ~/ardupilot/Tools/autotest/sim_vehicle.py -v Copter -f gazebo-iris --model JSON --no-mavproxy --no-rebuild -I0 &
SITL_PID=$!

sleep 4

echo " [3/3] Canlı FPV Kokpiti & MAVLink Kontrolcüsü Başlatılıyor..."
# Wayland optimus cihazlarda Qt backend sorunları için çevre değişkenleri
QT_QPA_PLATFORM=xcb GDK_BACKEND=x11 python3 ardupilot_drone_capture.py

# Arayüz kapatıldığında (ESC basıldığında) arkada çalışan Gazebo ve SITL'yi temizle
echo " Temizleniyor..."
kill -9 $GZ_PID 2>/dev/null || true
kill -9 $SITL_PID 2>/dev/null || true
killall -9 arducopter 2>/dev/null || true
echo " Görev tamamlandı."
