#!/bin/bash
# ========================================================================================
# 🚀 3D GAUSSIAN SPLATTING & DRON SLAM - UBUNTU / LINUX OTOMATİK KURULUM (KURULUM_UBUNTU.sh)
# ========================================================================================

set -e

echo "============================================================="
echo "  🚀 3D GAUSSIAN SPLATTING & DRON SLAM UBUNTU KURULUMU"
echo "============================================================="
echo ""

# 1. Sistem Paketlerini Güncelle ve Gerekli Kütüphaneleri Yükle
echo " [1/4] Ubuntu Sistem Paketleri ve OpenGL Kütüphaneleri Yükleniyor..."
sudo apt-get update -y
sudo apt-get install -y python3 python3-pip python3-dev \
    libgl1-mesa-glx libgl1-mesa-dri libglib2.0-0 libsm6 libxext6 libxrender-dev \
    ffmpeg git

# 2. Pip Güncelle
echo ""
echo " [2/4] Pip Güncelleniyor..."
python3 -m pip install --upgrade pip

# 3. NVIDIA CUDA Destekli PyTorch Kurulumu
echo ""
echo " [3/4] NVIDIA CUDA Destekli PyTorch Yükleniyor..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Proje Gereksinimlerini Yükle
echo ""
echo " [4/4] Proje Bağımlılıkları Yükleniyor (requirements.txt)..."
pip install -r requirements.txt

# 5. Kurulum Testi
echo ""
echo "============================================================="
echo "  🔍 UBUNTU DONANIM VE GPU TESTİ YAPILIYOR..."
echo "============================================================="
python3 -c "import torch, cv2, pygame, OpenGL; print(' -> PyTorch Sürümü:', torch.__version__); print(' -> CUDA GPU Aktif mi?:', torch.cuda.is_available()); print(' -> GPU Modeli:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU Modu'); print('\n✨ TEBRİKLER! Ubuntu kurulumu başarıyla tamamlandı.'); print('🚀 Projeyi başlatmak için: ./BASLAT.sh')"
echo "============================================================="