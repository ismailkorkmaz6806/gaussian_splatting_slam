@echo off
chcp 65001 >nul
title 🤖 Gazebo Sim 3B Dron & 3DGS Haritalama İstasyonu
cls
echo =============================================================
echo  🤖 GAZEBO SIM 3B DRON VE CANLI HARİTALAMA SİSTEMİ
echo =============================================================
echo.
echo  [1/3] Gazebo Sim (Conda gz_env) Başlatılıyor...
start "Gazebo Sim GUI" "C:\Users\ismai\anaconda3\condabin\conda.bat" run -n gz_env --no-capture-output "C:\Users\ismai\anaconda3\envs\gz_env\Library\libexec\gz\sim10\gz-sim-main.exe" -r "%~dp0gazebo_tunnel_drone.sdf"

echo.
echo  [2/3] Dron Kamera Köprüsü Başlatılıyor...
start "Gazebo Camera Bridge" python "%~dp0gz_camera_bridge.py"

echo.
echo  [3/3] Canlı Kamera Arayüzüne Bağlanılıyor...
timeout /t 3 /nobreak >nul
python "%~dp0live_drone_capture.py" http://127.0.0.1:8554/drone_stream
pause
