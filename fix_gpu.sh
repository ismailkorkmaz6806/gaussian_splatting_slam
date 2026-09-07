#!/bin/bash
set -e

echo "=========================================================="
echo " 🔧 NVIDIA GPU & CUDA DÜĞÜM DÜZELTİCİ"
echo "=========================================================="
echo ""

echo " [1/3] 'nvidia-modprobe' paketi kuruluyor..."
sudo apt-get update -y
sudo apt-get install -y nvidia-modprobe

echo ""
echo " [2/3] 'nvidia-uvm' çekirdek modülü yükleniyor..."
sudo modprobe nvidia-uvm || true
sudo nvidia-modprobe -u -c=0 || true
sudo chmod 666 /dev/nvidia* 2>/dev/null || true

echo ""
echo " [3/3] Donanım Test Ediliyor..."
echo "----------------------------------------------------------"
nvidia-smi

echo "----------------------------------------------------------"
/home/ismail/gaussian_splatting_slam/venv/bin/python3 -c "import torch; print('CUDA Aktif mi?:', torch.cuda.is_available()); print('Aktif GPU Modeli:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'HATA: CUDA Algılanamadı!')"
echo "=========================================================="
