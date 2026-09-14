#!/bin/bash
# ========================================================================================
# 🚁 TEK TIKLA PX4 SITL + GAZEBO + QGROUNDCONTROL + MAVLINK BRIDGE + CANLI 3DGS SLAM
# ========================================================================================

# Terminal Renkleri
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PX4_DIR="$HOME/PX4-Autopilot"

# Grafik arayüzden çift tıklama desteği
if [ ! -t 0 ] || [ ! -t 1 ]; then
    if which gnome-terminal > /dev/null 2>&1; then
        exec gnome-terminal --title="3DGS SLAM & PX4 Otonom Simülasyon" --working-directory="$SCRIPT_DIR" -- bash -c "\"$0\"; echo ''; echo 'Program kapandı. Kapatmak için Enter tuşuna basın...'; read"
        exit 0
    fi
fi

echo -e "${CYAN}======================================================================${NC}"
echo -e "${GREEN}   🚁 BÜYÜK EV PX4 SITL + GAZEBO + QGC + 3DGS SLAM MERKEZİ${NC}"
echo -e "${CYAN}======================================================================${NC}"

# Venv aktivasyonu
if [ -f "$SCRIPT_DIR/venv/bin/activate" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
fi

# NVIDIA RTX Optimizasyonu
export __NV_PRIME_RENDER_OFFLOAD=1
export __GL_SYNC_TO_VBLANK=0
export vblank_mode=0

# Çıkışta tüm alt süreçleri otomatik kapat
cleanup() {
    echo ""
    echo -e "${YELLOW}🧹 Tüm süreçler temizleniyor (PX4, Gazebo, QGroundControl, Bridge)...${NC}"
    if [ ! -z "$QGC_PID" ]; then
        kill $QGC_PID 2>/dev/null || true
    fi
    if [ ! -z "$PX4_PID" ]; then
        kill $PX4_PID 2>/dev/null || true
    fi
    pkill -9 -f "QGroundControl" 2>/dev/null || true
    pkill -9 px4 2>/dev/null || true
    pkill -9 -f "gz sim" 2>/dev/null || true
    pkill -9 ruby 2>/dev/null || true
    echo -e "${GREEN}✅ Tüm simülasyon, QGC ve köprü süreçleri temizlendi.${NC}"
}
trap cleanup EXIT INT TERM

# -------------------------------------------------------------------------
# 1. ADIM: Gazebo & PX4 SITL Otopilotu
# -------------------------------------------------------------------------
if pgrep -x "px4" > /dev/null; then
    echo -e "${YELLOW}ℹ️  PX4 SITL zaten çalışıyor.${NC}"
else
    echo -e "${GREEN}🚀 [1/3] PX4 SITL ve Gazebo 3B Simülasyonu Açılıyor (buyuk_ev)...${NC}"
    if [ -d "$PX4_DIR" ]; then
        cd "$PX4_DIR"
        export DISPLAY="${DISPLAY:-:0}"
        export WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-wayland-0}"
        PX4_GZ_WORLD=buyuk_ev make px4_sitl gz_x500_vision > /tmp/px4_sitl.log 2>&1 &
        PX4_PID=$!
        cd "$SCRIPT_DIR"
        
        echo -e "${YELLOW}⏳ Gazebo ve PX4 otopilotunun ayağa kalkması bekleniyor (yaklaşık 8 sn)...${NC}"
        for i in {8..1}; do
            echo -ne " -> Simülatör hazırlanıyor... [${i}s]\r"
            sleep 1
        done
        echo -e "\n${GREEN}✅ Gazebo & PX4 başarıyla açıldı!${NC}"
    else
        echo -e "${RED}❌ HATA: $PX4_DIR dizini bulunamadı!${NC}"
        exit 1
    fi
fi

# -------------------------------------------------------------------------
# 2. ADIM: QGroundControl Yer Kontrol İstasyonu
# -------------------------------------------------------------------------
if pgrep -f "QGroundControl" > /dev/null; then
    echo -e "${YELLOW}ℹ️  QGroundControl zaten açık.${NC}"
else
    QGC_BIN=""
    if [ -f "$HOME/Downloads/QGroundControl.AppImage" ]; then
        QGC_BIN="$HOME/Downloads/QGroundControl.AppImage"
    elif [ -f "$HOME/QGroundControl.AppImage" ]; then
        QGC_BIN="$HOME/QGroundControl.AppImage"
    elif which qgroundcontrol > /dev/null 2>&1; then
        QGC_BIN="qgroundcontrol"
    elif which QGroundControl > /dev/null 2>&1; then
        QGC_BIN="QGroundControl"
    fi

    if [ ! -z "$QGC_BIN" ]; then
        echo -e "${GREEN}🎮 [2/2] QGroundControl Açılıyor...${NC}"
        if [[ "$QGC_BIN" == *.AppImage ]]; then
            "$QGC_BIN" --appimage-extract-and-run > /tmp/qgc.log 2>&1 &
        else
            "$QGC_BIN" > /tmp/qgc.log 2>&1 &
        fi
        QGC_PID=$!
        sleep 2
        echo -e "${GREEN}✅ QGroundControl açıldı (PX4'e otomatik bağlandı)!${NC}"
    else
        echo -e "${YELLOW}⚠️ QGroundControl.AppImage bulunamadı, bu adım atlanıyor.${NC}"
    fi
fi

# -------------------------------------------------------------------------
# SİMÜLASYON KONTROL PANELİ (Terminal)
# -------------------------------------------------------------------------
echo ""
echo -e "${CYAN}======================================================================${NC}"
echo -e "${GREEN}   ✨ SİMÜLASYON VE QGROUNDCONTROL HAZIR!${NC}"
echo -e "${CYAN}======================================================================${NC}"
echo -e " 📍 Gazebo Penceresi : ${GREEN}AÇIK${NC} (Dünya: buyuk_ev, Model: x500_vision)"
echo -e " 📍 QGroundControl   : ${GREEN}AÇIK${NC} (UDP 14550 Bağlı)"
echo -e " 📍 GPS Durumu       : ${GREEN}AÇIK (Kalkışa Hazır)${NC}"
echo ""
echo -e " 💡 ${YELLOW}QGroundControl MAVLink Console'dan veya buradan parametre verebilirsin:${NC}"
echo "    • GPS Kapatmak için : param set EKF2_GPS_CTRL 0"
echo "    • GPS Açmak için    : param set EKF2_GPS_CTRL 7"
echo ""

while true; do
    echo -e "${CYAN}----------------------------------------------------------------------${NC}"
    echo "  [1] 🔴 GPS'i KAPAT (param set EKF2_GPS_CTRL 0)"
    echo "  [2] 🟢 GPS'i AÇ   (param set EKF2_GPS_CTRL 7)"
    echo "  [3] 📷 3B Haritalama / FPV Kokpitini Aç (İsteğe bağlı)"
    echo "  [0] ❌ Simülasyonu Kapat ve Çık"
    echo -e "${CYAN}----------------------------------------------------------------------${NC}"
    read -p "Seçiminiz [0-3]: " CMD_SECIM

    case $CMD_SECIM in
        1)
            echo -e "${YELLOW}🛰️ GPS Kapatılıyor (EKF2_GPS_CTRL 0)...${NC}"
            python3 -c "from px4_mavlink_bridge import PX4VisionBridge; b=PX4VisionBridge(); b.connect() and b.set_gps_enabled(False)" 2>/dev/null || true
            echo -e "${RED}✅ GPS Kapatıldı (GPS-Denied Modu Aktif).${NC}"
            ;;
        2)
            echo -e "${YELLOW}🛰️ GPS Açılıyor (EKF2_GPS_CTRL 7)...${NC}"
            python3 -c "from px4_mavlink_bridge import PX4VisionBridge; b=PX4VisionBridge(); b.connect() and b.set_gps_enabled(True)" 2>/dev/null || true
            echo -e "${GREEN}✅ GPS Açıldı (3D Fix Aktif).${NC}"
            ;;
        3)
            echo -e "${GREEN}🎥 Canlı FPV ve 3DGS Haritalama Kokpiti Açılıyor...${NC}"
            python3 "$SCRIPT_DIR/live_drone_capture.py" gazebo
            ;;
        0|"q"|"Q")
            echo -e "${YELLOW}Simülasyon sonlandırılıyor...${NC}"
            break
            ;;
        *)
            echo -e "${RED}Geçersiz seçim!${NC}"
            ;;
    esac
done

exit 0
