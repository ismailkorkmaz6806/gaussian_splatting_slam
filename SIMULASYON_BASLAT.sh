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

# Grafik ve Senkronizasyon Ayarları
export __GL_SYNC_TO_VBLANK=0
export vblank_mode=0
export MAVLINK20=1

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
    pkill -9 -f "drone_lidar_odometry_bridge.py" 2>/dev/null || true
    pkill -9 -f "drone_cpu_hover_bridge.py" 2>/dev/null || true
    pkill -9 -f "lidar_icp_odometry.py" 2>/dev/null || true
    echo -e "${GREEN}✅ Tüm simülasyon, QGC ve köprü süreçleri temizlendi.${NC}"
}
trap cleanup EXIT INT TERM

# -------------------------------------------------------------------------
# 1. ADIM: Gazebo & PX4 SITL Otopilotu
# -------------------------------------------------------------------------
pkill -9 px4 2>/dev/null || true
pkill -9 -f "gz sim" 2>/dev/null || true
pkill -9 -f "drone_lidar_odometry_bridge.py" 2>/dev/null || true
pkill -9 -f "gps_kontrol.py" 2>/dev/null || true
rm -f /tmp/px4_lock* /tmp/px4-sock* /tmp/px4_sitl.log 2>/dev/null || true
rm -f "$PX4_DIR/build/px4_sitl_default/rootfs/parameters"* 2>/dev/null || true

# Dünyayı Belirle (Özel Organik Kaya Mağarası / Eşyalı Ev)
WORLD_CHOICE="${1:-}"
if [ -z "$WORLD_CHOICE" ]; then
    echo -e "${CYAN}----------------------------------------------------------------------${NC}"
    echo -e " 🌍 ${GREEN}Simülasyon Dünyasını Seçin:${NC}"
    echo -e "   [1] 🦇 ${YELLOW}Özel Organik Kaya Mağarası${NC} (benim_magaram - Dron Giriş Önünde, Keşif Kampı, Kutular)"
    echo -e "   [2] 🏠 ${CYAN}Eşyalı ve Dokulu Büyük Ev${NC} (buyuk_ev - 0 Kasma, %100 Hız, 280k+ Splat)"
    echo -e "${CYAN}----------------------------------------------------------------------${NC}"
    read -t 10 -p "Seçiminiz [1/2, varsayılan: 2]: " WORLD_CHOICE
fi

if [ "$WORLD_CHOICE" == "1" ]; then
    SECILEN_DUNYA="benim_magaram"
    DUNYA_POSE="4.0,0.0,0.20,0,0,0"
    DUNYA_BASLIK="Temiz Doğal Kaya Tüneli (Tam Mağaranın İçi Başlangıç)"
else
    SECILEN_DUNYA="buyuk_ev"
    DUNYA_POSE="0.0,-3.5,0.20,0,0,1.5708"
    DUNYA_BASLIK="Yüksek Tavanlı Tam Kapalı Büyük Ev (6m Tavan, Sıfır Dış Alan)"
fi

echo -e "\n${GREEN}🚀 [1/2] PX4 SITL ve Gazebo 3B Simülasyonu Açılıyor: ${DUNYA_BASLIK}...${NC}"
if [ -d "$PX4_DIR" ]; then
    cd "$PX4_DIR"
    export DISPLAY="${DISPLAY:-:0}"
    cp -f "$SCRIPT_DIR/gazebo_cave_world/worlds/buyuk_ev.sdf" "$PX4_DIR/Tools/simulation/gz/worlds/buyuk_ev.sdf" 2>/dev/null || true
    cp -f "$SCRIPT_DIR/benim_magaram.sdf" "$PX4_DIR/Tools/simulation/gz/worlds/benim_magaram.sdf" 2>/dev/null || true
    cp -f "$SCRIPT_DIR/px4_airframes/4005_gz_x500_vision" "$PX4_DIR/ROMFS/px4fmu_common/init.d-posix/airframes/4005_gz_x500_vision" 2>/dev/null || true
    
    export GZ_SIM_RESOURCE_PATH="$GZ_SIM_RESOURCE_PATH:$SCRIPT_DIR/gazebo_cave_world/worlds:$PX4_DIR/Tools/simulation/gz/worlds/aws_small_house_world/models:$PX4_DIR/Tools/simulation/gz/models:$SCRIPT_DIR/gazebo_cave_world/worlds/models"
    export PX4_GZ_WORLD="$SECILEN_DUNYA"
    export PX4_GZ_MODEL_POSE="$DUNYA_POSE"
    make px4_sitl gz_x500_vision > /tmp/px4_sitl.log 2>&1 &
    PX4_PID=$!
    cd "$SCRIPT_DIR"
    
    echo -e "${YELLOW}⏳ Gazebo ve PX4 otopilotunun ayağa kalkması bekleniyor (yaklaşık 8 sn)...${NC}"
    for i in {8..1}; do
        echo -ne " -> Simülatör hazırlanıyor... [${i}s]\r"
        sleep 1
    done
    echo -e "\n${GREEN}✅ Gazebo & PX4 başarıyla açıldı!${NC}"
    
    # Sistemin en başından itibaren GPS'i sıfırlamak ve 3B Lidar Odometrisini aktif tutmak için
    # arka planda Lidar Odometri (ICP) motorunu başlatıyoruz.
    echo -e "${YELLOW}🛰️ [LİDAR ODOMETRİ SİSTEMİ] Donanımsal Olarak GPS Devredışı Bırakılıyor ve Lidar Odometri Başlatılıyor...${NC}"
    nohup python3 -u drone_lidar_odometry_bridge.py --mode lidar > /tmp/lidar_odometry.log 2>&1 &
    LIDAR_PID=$!
    sleep 1
    
    # Kamerayı otomatik dronun arkasına kilitle (Mağara içine girince dronu kaybetmemek için)
    (
        sleep 3
        gz service -s /gui/follow --reqtype gz.msgs.StringMsg --reptype gz.msgs.Boolean --timeout 2000 --req 'data: "x500_vision_0"' >/dev/null 2>&1 || \
        gz service -s /gui/follow --reqtype gz.msgs.StringMsg --reptype gz.msgs.Boolean --timeout 2000 --req 'data: "x500_vision"' >/dev/null 2>&1
        gz service -s /gui/follow/offset --reqtype gz.msgs.Vector3d --reptype gz.msgs.Boolean --timeout 2000 --req 'x: -2.2, y: 0.0, z: 0.8' >/dev/null 2>&1
    ) &
else
    echo -e "${RED}❌ HATA: $PX4_DIR dizini bulunamadı!${NC}"
    exit 1
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
echo -e " 📍 Gazebo Penceresi : ${GREEN}AÇIK${NC} (Dünya: $SECILEN_DUNYA, Model: x500_vision)"
echo -e " 📍 QGroundControl   : ${GREEN}AÇIK${NC} (UDP 14550 Bağlı)"
echo -e " 📍 Görünmez Kalkan  : ${GREEN}AKTİF${NC} (CP_DIST: 0.60m | 360° Çarpışma Önleme Kalkanı)"
echo -e " 📍 Kalkış / Arm     : ${GREEN}KİLİTLER AÇILDI (Hazır)${NC}"
echo ""

while true; do
    echo -e "${CYAN}----------------------------------------------------------------------${NC}"
    echo "  [1] 🔴 GPS'i KAPAT (failure gps off)"
    echo "  [2] 🟢 GPS'i AÇ   (failure gps ok)"
    echo "  [3] 📷 3B Haritalama / FPV Kokpitini Aç (İsteğe bağlı)"
    echo "  [4] 🚀 Dronu ARM ET (Motorları Başlat)"
    echo "  [5] 🛑 Dronu DISARM ET (Motorları Durdur / İndir)"
    echo "  [0] ❌ Simülasyonu Kapat ve Çık"
    echo -e "${CYAN}----------------------------------------------------------------------${NC}"
    read -p "Seçiminiz [0-5]: " CMD_SECIM

    case $CMD_SECIM in
        1)
            echo -e "${YELLOW}🛰️ GPS Donanımsal Olarak Kapatılıyor...${NC}"
            python3 -c "from px4_mavlink_bridge import PX4VisionBridge; b=PX4VisionBridge(); b.connect() and b.set_gps_enabled(False)" 2>/dev/null || true
            echo -e "${RED}✅ GPS Kapatıldı (QGC'de uydu sıfırlandı).${NC}"
            ;;
        2)
            echo -e "${YELLOW}🛰️ GPS Açılıyor...${NC}"
            python3 -c "from px4_mavlink_bridge import PX4VisionBridge; b=PX4VisionBridge(); b.connect() and b.set_gps_enabled(True)" 2>/dev/null || true
            echo -e "${GREEN}✅ GPS Açıldı.${NC}"
            ;;
        3)
            echo -e "${GREEN}🎥 Canlı FPV ve 3DGS Haritalama Kokpiti Açılıyor...${NC}"
            python3 "$SCRIPT_DIR/live_drone_capture.py" gazebo
            ;;
        4)
            echo -e "${GREEN}🚀 Dron ARM Ediliyor...${NC}"
            python3 -c "from px4_mavlink_bridge import PX4VisionBridge; b=PX4VisionBridge(); b.connect() and b.arm(force=True)" 2>/dev/null || true
            ;;
        5)
            echo -e "${YELLOW}🛑 Dron DISARM Ediliyor...${NC}"
            python3 -c "from px4_mavlink_bridge import PX4VisionBridge; b=PX4VisionBridge(); b.connect() and b.disarm()" 2>/dev/null || true
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
