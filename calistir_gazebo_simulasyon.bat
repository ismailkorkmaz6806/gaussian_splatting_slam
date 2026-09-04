@echo off
chcp 65001 >nul
title 🤖 Gazebo Sim 3B Dron & 3DGS Haritalama İstasyonu
cls
echo =============================================================
echo  🤖 GAZEBO SIM (gz sim) 3B DRON VE CANLI HARİTALAMA SİSTEMİ
echo =============================================================
echo.
echo  [1/3] Conda gz_env Ortamı Etkinleştiriliyor...
call "C:\Users\ismai\anaconda3\condabin\conda.bat" activate gz_env

echo.
echo  [2/3] Gazebo Sim 3B Tünel Dünyası Başlatılıyor...
start "Gazebo Sim" gz sim -r "%~dp0gazebo_tunnel_drone.sdf"

echo.
echo  [3/3] Dron Kamera Köprüsü ve Canlı Yakalayıcı Açılıyor...
start "Gazebo Bridge" python "%~dp0gz_camera_bridge.py"
timeout /t 3 /nobreak >nul
python "%~dp0live_drone_capture.py" http://127.0.0.1:8554/drone_stream
pause
