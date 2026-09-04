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

while true; do
    clear
    echo -e "${CYAN}=============================================================${NC}"
    echo -e "${GREEN}   🚁 MASt3R-3DGS TÜNEL VE DRON HARİTALAMA MERKEZİ (UBUNTU)${NC}"
    echo -e "${CYAN}=============================================================${NC}"
    echo ""
    echo "  [1] 📹 Canlı Dron / Kamera ile Tara & 3B Harita Çıkar"
    echo "  [2] 🎬 Bir MP4 Videosunu 3B Modele Dönüştür"
    echo "  [3] 🔮 3B Haritayı Aç (144+ FPS Gezgin & 3B Dron Modu)"
    echo "  [4] 🎮 3B Dron & Tünel Simülatörü (Klavye ile Serbest Uçuş)"
    echo "  [5] 🧩 Çoklu Koridorları Tek Haritada Birleştir"
    echo "  [6] 📄 Tünel İnceleme ve PDF/HTML Raporu Üret"
    echo "  [0] ❌ Çıkış"
    echo ""
    echo -e "${CYAN}=============================================================${NC}"
    read -p "Lütfen bir işlem seçin [0-6]: " SECIM

    case $SECIM in
        1)
            clear
            echo -e "${CYAN}=============================================================${NC}"
            echo -e "${GREEN}  📹 CANLI DRON / KAMERA İLE TARAMA VE HARİTALAMA${NC}"
            echo -e "${CYAN}=============================================================${NC}"
            echo ""
            echo "  [1] USB Web Kamerası (/dev/video0)"
            echo "  [2] Gerçek Dron Canlı Yayını (RTSP / Wi-Fi / Fiber)"
            echo "  [3] 🎮 Sanal Dron Canlı Yayın Simülasyonu"
            echo ""
            read -p "Seçiminiz [1-3]: " K_SECIM
            if [ "$K_SECIM" == "3" ]; then
                python3 simulated_drone_streamer.py &
                STREAM_PID=$!
                sleep 2
                python3 live_drone_capture.py http://127.0.0.1:8554/drone_stream
                kill $STREAM_PID 2>/dev/null || true
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
            echo -e "${GREEN}  🔮 3B MODEL GÖRÜNTÜLEYİCİ AÇILIYOR...${NC}"
            echo -e "${CYAN}=============================================================${NC}"
            echo ""
            python3 gaussian_renderer.py gaussian_scene.ply
            read -p "Devam etmek için Enter'a basın..."
            ;;
        4)
            clear
            echo -e "${CYAN}=============================================================${NC}"
            echo -e "${GREEN}  🎮 3B DRON & TÜNEL UÇUŞ SİMÜLATÖRÜ AÇILIYOR (144+ FPS)...${NC}"
            echo -e "${CYAN}=============================================================${NC}"
            echo ""
            python3 dron_tunel_simulatoru.py
            read -p "Devam etmek için Enter'a basın..."
            ;;
        5)
            clear
            echo -e "${CYAN}=============================================================${NC}"
            echo -e "${GREEN}  🧩 ÇOKLU HARİTALARI BİRLEŞTİRME${NC}"
            echo -e "${CYAN}=============================================================${NC}"
            echo ""
            python3 map_stitcher.py
            read -p "Devam etmek için Enter'a basın..."
            ;;
        6)
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
            echo -e "${RED}Geçersiz seçim! Lütfen 0-6 arasında bir rakam girin.${NC}"
            sleep 1
            ;;
    esac
done