@echo off
chcp 65001 >nul
title ?? 3DGS & Dron SLAM Otomatik Kurulum Sihirbazi
cls
echo =============================================================
echo   ?? 3D GAUSSIAN SPLATTING & DRON SLAM OTOMATIK KURULUM
echo =============================================================
echo.
echo  Bu script projeyi yeni bir bilgisayara kurmak icin gereken
echo  tum kutuphaneleri (PyTorch CUDA, OpenGL, OpenCV) otomatik yukler.
echo.
pause
echo.
echo [1/3] Pip guncelleniyor...
python -m pip install --upgrade pip

echo.
echo [2/3] NVIDIA CUDA Destekli PyTorch Yukleniyor...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo [3/3] Proje Gereksinimleri Yukleniyor (requirements.txt)...
pip install -r requirements.txt

cls
echo =============================================================
echo   ?? KURULUM VE DONANIM TESTI YAPILIYOR...
echo =============================================================
python -c "import torch, cv2, pygame, OpenGL; print(' PyTorch Surumu:', torch.__version__); print(' CUDA Aktif mi?:', torch.cuda.is_available()); print(' GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU Modu'); print('\n TEBRIKLER! Tum kutuphaneler basariyla kuruldu.'); print(' Projeyi baslatmak icin: BASLAT.bat calistirin.')"
echo =============================================================
pause
