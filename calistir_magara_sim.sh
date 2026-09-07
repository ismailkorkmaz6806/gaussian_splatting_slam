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
export GZ_SIM_RESOURCE_PATH="$DIR/gazebo_cave_world/worlds/models:$GZ_SIM_RESOURCE_PATH"

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
pkill -f "cave_drone_capture.py" 2>/dev/null || true

echo " [1/2] Gazebo Sim Dünyası Başlatılıyor..."
if which gz > /dev/null; then
    gz sim $HEADLESS_FLAG -r "$WORLD_FILE" &
    GZ_PID=$!
else
    echo "⚠️ Gazebo (gz) komutu bulunamadı. Lütfen gz kurulu olduğundan emin olun."
    exit 1
fi

# Güvenli kapanış için trap
trap "kill $GZ_PID 2>/dev/null || true; killall -9 gz sim 2>/dev/null || true" EXIT INT TERM

echo " [2/2] Canlı FPV Kokpiti & MASt3R-3DGS Haritalama Başlatılıyor..."
sleep 3.5
# Yalnızca Python (OpenCV) için X11 zorlamasını kullanarak çalıştır
QT_QPA_PLATFORM=xcb GDK_BACKEND=x11 python3 "$DIR/cave_drone_capture.py" 45

# Temizlik
kill $GZ_PID 2>/dev/null || true
killall -9 gz sim 2>/dev/null || true
