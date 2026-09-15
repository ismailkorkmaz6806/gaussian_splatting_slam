#!/bin/bash
# ========================================================================================
# 🚁 MASt3R-3DGS TÜNEL VE DRON HARİTALAMA MERKEZİ - UBUNTU / LINUX BAŞLATICI (BASLAT.sh)
# ========================================================================================

# Terminal Renkleri
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Proje dizinine geçiş ve venv aktivasyonu
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Eğer grafik arayüzden (çift tıklama vb.) açıldıysa ve bir terminal yoksa otomatik terminal aç
if [ ! -t 0 ] || [ ! -t 1 ]; then
    if which gnome-terminal > /dev/null 2>&1; then
        exec gnome-terminal --title="3DGS SLAM Merkezi" --working-directory="$SCRIPT_DIR" -- bash -c "\"$0\"; echo ''; echo 'Program kapandı. Kapatmak için Enter tuşuna basın...'; read"
        exit 0
    fi
fi

if [ -f "$SCRIPT_DIR/venv/bin/activate" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
fi

# NVIDIA RTX Donanım Hızlandırması
export __NV_PRIME_RENDER_OFFLOAD=1
export __GL_SYNC_TO_VBLANK=0
export vblank_mode=0

while true; do
    clear
    echo -e "${CYAN}=============================================================${NC}"
    echo -e "${GREEN}   🚁 MASt3R-3DGS TÜNEL VE DRON HARİTALAMA MERKEZİ (UBUNTU)${NC}"
    echo -e "${CYAN}=============================================================${NC}"
    echo ""
    echo "  [1] 📹 Canlı Dron / Kamera ile Tara & 3B Harita Çıkar"
    echo "  [2] 🎬 Bir MP4 Videosunu 3B Modele Dönüştür"
    echo "  [3] 🏢 Ofis 3B Haritasını Görüntüle (gaussian_scene.ply)"
    echo "  [4] 🚇 Eski Tünel 3B Haritasını Görüntüle (drone_scene.ply)"
    echo "  [5] 📄 Tünel İnceleme ve PDF/HTML Raporu Üret"
    echo "  [0] ❌ Çıkış"
    echo ""
    echo -e "${CYAN}=============================================================${NC}"
    read -p "Lütfen bir işlem seçin [0-5]: " SECIM

    case $SECIM in
        1)
            clear
            echo -e "${CYAN}=============================================================${NC}"
            echo -e "${GREEN}  📹 CANLI DRON / KAMERA İLE TARAMA VE HARİTALAMA${NC}"
            echo -e "${CYAN}=============================================================${NC}"
            echo ""
            echo "  [1] USB Web Kamerası (/dev/video0)"
            echo "  [2] Gerçek Dron Canlı Yayını (RTSP / Wi-Fi / Fiber)"
            echo "  [3] 🏡 Gazebo + PX4 SITL Büyük Ev (Tek Tık Otonom Uçuş & 3DGS)"
            echo ""
            read -p "Seçiminiz [1-3]: " K_SECIM
            if [ "$K_SECIM" == "3" ]; then
                ./SIMULASYON_BASLAT.sh
            elif [ "$K_SECIM" == "2" ]; then
                read -p "RTSP / HTTP Yayın Linki (Varsayılan: rtsp://192.168.1.100:8554/stream): " RTSP_URL
                if [ -z "$RTSP_URL" ]; then
                    RTSP_URL="rtsp://192.168.1.100:8554/stream"
                fi
                python3 live_drone_capture.py "$RTSP_URL"
            else
                python3 live_drone_capture.py 0
            fi
            read -p "Devam etmek için Enter'a basın..."
            ;;
        2)
            clear
            echo -e "${CYAN}=============================================================${NC}"
            echo -e "${GREEN}  🎬 MP4 VİDEOSUNDAN 3B HARİTA ÜRETİMİ${NC}"
            echo -e "${CYAN}=============================================================${NC}"
            echo ""
            read -p "İşlenecek video adı veya yolu (Varsayılan: ofisvideo.mp4): " V_NAME
            if [ -z "$V_NAME" ]; then
                V_NAME="ofisvideo.mp4"
            fi
            python3 mast3r_to_3dgs.py "$V_NAME" 50
            python3 gaussian_renderer.py gaussian_scene.ply
            read -p "Devam etmek için Enter'a basın..."
            ;;
        3)
            clear
            echo -e "${CYAN}=============================================================${NC}"
            echo -e "${GREEN}  🏢 OFİS VİDEOSU 3B HARİTASI AÇILIYOR...${NC}"
            echo -e "${CYAN}=============================================================${NC}"
            echo ""
            if [ -f "gaussian_scene.ply" ]; then
                python3 gaussian_renderer.py gaussian_scene.ply
            else
                echo -e "${RED}❌ Hata: gaussian_scene.ply bulunamadı!${NC}"
            fi
            read -p "Devam etmek için Enter'a basın..."
            ;;
        4)
            clear
            echo -e "${CYAN}=============================================================${NC}"
            echo -e "${GREEN}  🚇 ESKİ TÜNEL HARİTASI AÇILIYOR...${NC}"
            echo -e "${CYAN}=============================================================${NC}"
            echo ""
            if [ -f "drone_scene.ply" ]; then
                python3 gaussian_renderer.py drone_scene.ply
            else
                echo -e "${RED}❌ Hata: drone_scene.ply bulunamadı!${NC}"
            fi
            read -p "Devam etmek için Enter'a basın..."
            ;;
        5)
            clear
            echo -e "${CYAN}=============================================================${NC}"
            echo -e "${GREEN}  📄 TÜNEL İNCELEME RAPORU ÜRETİLİYOR...${NC}"
            echo -e "${CYAN}=============================================================${NC}"
            echo ""
            python3 tunnel_report_generator.py gaussian_scene_cache.npz
            if which xdg-open > /dev/null; then
                xdg-open tunel_inceleme_raporu.html &
            fi
            read -p "Devam etmek için Enter'a basın..."
            ;;
        0)
            echo -e "${YELLOW}Görüşmek üzere! Çıkış yapılıyor...${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}Geçersiz seçim! Lütfen 0-5 arasında bir rakam girin.${NC}"
            sleep 1
            ;;
    esac
done