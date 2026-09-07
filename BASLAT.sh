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
    echo "  [3] 🔮 3B Haritayı Aç (144+ FPS Gezgin & 3B Dron Modu)"
    echo "  [4] 🎮 3B Dron & Tünel Simülatörü (Klavye ile Serbest Uçuş)"
    echo "  [5] 🧩 Çoklu Koridorları Tek Haritada Birleştir"
    echo "  [6] 📄 Tünel İnceleme ve PDF/HTML Raporu Üret"
    echo "  [7] 🦇 DARPA SubT Gerçek Mağara Simülasyonu & 3DGS Haritalama"
    echo "  [0] ❌ Çıkış"
    echo ""
    echo -e "${CYAN}=============================================================${NC}"
    read -p "Lütfen bir işlem seçin [0-7]: " SECIM

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
            echo "  [4] 🦇 DARPA SubT Gerçek Mağara Dronu"
            echo ""
            read -p "Seçiminiz [1-4]: " K_SECIM
            if [ "$K_SECIM" == "4" ]; then
                ./calistir_magara_sim.sh
            elif [ "$K_SECIM" == "3" ]; then
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
            echo "  [1] 🏢 Ofis Videosu Haritası (gaussian_scene.ply)"
            echo "  [2] 🚇 Eski Tünel Haritası (drone_scene.ply)"
            echo "  [3] 🦇 DARPA SubT Mağara Haritası (cave_scene.ply)"
            echo ""
            read -p "Hangi haritayı açmak istersiniz? [1-3]: " MAP_SECIM
            
            if [ "$MAP_SECIM" == "3" ]; then
                TARGET_PLY="cave_scene.ply"
            elif [ "$MAP_SECIM" == "2" ]; then
                TARGET_PLY="drone_scene.ply"
            else
                TARGET_PLY="gaussian_scene.ply"
            fi
            
            if [ -f "$TARGET_PLY" ]; then
                python3 gaussian_renderer.py "$TARGET_PLY"
            else
                echo -e "${RED}❌ Hata: $TARGET_PLY bulunamadı! Önce haritalama yapmalısınız.${NC}"
            fi
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
        7)
            clear
            echo -e "${CYAN}=============================================================${NC}"
            echo -e "${GREEN}  🦇 DARPA SUBT GERÇEK MAĞARA SİMÜLASYONU & 3DGS HARİTALAMA${NC}"
            echo -e "${CYAN}=============================================================${NC}"
            echo ""
            ./calistir_magara_sim.sh
            read -p "Devam etmek için Enter'a basın..."
            ;;
        0)
            echo -e "${YELLOW}Görüşmek üzere! Çıkış yapılıyor...${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}Geçersiz seçim! Lütfen 0-8 arasında bir rakam girin.${NC}"
            sleep 1
            ;;
    esac
done