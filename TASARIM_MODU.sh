#!/bin/bash
# ========================================================================================
# 🎨 3B DÜNYA VE HARİTA TASARIM MODU (GAZEBO 3D GÖRÜNTÜLEYİCİ)
# ========================================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f "$SCRIPT_DIR/venv/bin/activate" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
fi

export __NV_PRIME_RENDER_OFFLOAD=1
export __GL_SYNC_TO_VBLANK=0
export vblank_mode=0

echo -e "\033[0;36m=============================================================\033[0m"
echo -e "\033[0;32m   🎨 TASARIM MODU: YÜKSEK TAVANLI TAM KAPALI BÜYÜK EV\033[0m"
echo -e "\033[0;36m=============================================================\033[0m"
echo -e " 🏠 Tavan Yüksekliği : \033[1;33m6.0 Metre\033[0m"
echo -e " 🧱 Dış Alan         : \033[1;33mSıfır Dış Alan (%100 Kapalı ve İzolasyonlu)\033[0m"
echo -e " 🛋️ İç Mekan         : \033[1;33mGeniş Salon, Koltuklar, Halı, Laboratuvar, Raflar\033[0m"
echo -e "\033[0;36m-------------------------------------------------------------\033[0m"
echo " 🖱️  Fare Sol Tık   : Döndür (Orbit)"
echo " 🖱️  Fare Sağ Tık  : Yakınlaş / Uzaklaş (Zoom)"
echo " 🖱️  Fare Orta Tık : Kaydır (Pan)"
echo -e "\033[0;36m=============================================================\033[0m"
echo "Gazebo açılıyor..."

gz sim -v 4 gazebo_cave_world/worlds/buyuk_ev.sdf
