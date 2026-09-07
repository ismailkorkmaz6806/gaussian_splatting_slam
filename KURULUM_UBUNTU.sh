#!/bin/bash
# ========================================================================================
# 🚀 3D GAUSSIAN SPLATTING & DRON SLAM - UBUNTU / LINUX OTOMATİK KURULUM (KURULUM_UBUNTU.sh)
# ========================================================================================

# Proje çalışma dizinine geç
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Eğer grafik arayüzden (çift tıklama vb.) açıldıysa otomatik terminal aç
if [ ! -t 0 ] || [ ! -t 1 ]; then
    if which gnome-terminal > /dev/null 2>&1; then
        exec gnome-terminal --title="3DGS SLAM Kurulum Sihirbazı" --working-directory="$SCRIPT_DIR" -- bash -c "\"$0\"; echo ''; echo 'Kurulum tamamlandı. Kapatmak için Enter tuşuna basın...'; read"
        exit 0
    fi
fi

# 1. Sistem Paketlerini Güncelle ve Gerekli Kütüphaneleri Yükle
echo " [1/5] Ubuntu Sistem Paketleri ve OpenGL Kütüphaneleri Yükleniyor..."
sudo apt-get update -y
sudo apt-get install -y python3 python3-pip python3-dev python3-venv \
    libgl1-mesa-glx libgl1-mesa-dri libglib2.0-0 libsm6 libxext6 libxrender-dev \
    ffmpeg git

# Proje çalışma dizinine geç
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 2. Sanal Ortam (venv) Kurulumu & Pip Güncelleme
echo ""
echo " [2/5] Python Sanal Ortamı (venv) Hazırlanıyor..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install --upgrade pip

# 3. NVIDIA CUDA Destekli PyTorch Kurulumu
echo ""
echo " [3/5] NVIDIA CUDA Destekli PyTorch Yükleniyor..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Proje Gereksinimlerini Yükle
echo ""
echo " [4/5] Proje Bağımlılıkları Yükleniyor (requirements.txt)..."
pip install -r requirements.txt

# 5. Naver MASt3R Yapay Zeka Model Alt Modülü
echo ""
echo " [5/5] Naver MASt3R 3B Rekonstrüksiyon Modülü Hazırlanıyor..."
if [ ! -d "mast3r_repo" ]; then
    git clone --recursive https://github.com/naver/mast3r.git mast3r_repo
fi

# 6. Kurulum Testi
echo ""
echo "============================================================="
echo "  🔍 UBUNTU DONANIM VE GPU TESTİ YAPILIYOR..."
echo "============================================================="
python3 -c "import torch, cv2, pygame, OpenGL; print(' -> PyTorch Sürümü:', torch.__version__); print(' -> CUDA GPU Aktif mi?:', torch.cuda.is_available()); print(' -> GPU Modeli:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU Modu'); print('\n✨ TEBRİKLER! Ubuntu kurulumu başarıyla tamamlandı.'); print('🚀 Projeyi başlatmak için: ./BASLAT.sh')"
echo "============================================================="