@echo off
chcp 65001 >nul
title Gazebo Sim 3B Dron Haritalama
cls
echo =============================================================
echo  ?? GAZEBO SIM 3B DRON VE CANLI HAR?TALAMA S?STEM?
echo =============================================================
echo.
echo  [1/3] Gazebo Sim (gz_env) Ba?lat?l?yor...
call C:\Users\ismai\anaconda3\condabin\conda.bat activate gz_env
start "" gz sim -r "%~dp0gazebo_tunnel_drone.sdf"

echo.
echo  [2/3] Dron Kamera K?pr?s? Ba?lat?l?yor...
start "" python "%~dp0gz_camera_bridge.py"

echo.
echo  [3/3] Canl? Kamera Aray?z?ne Ba?lan?l?yor...
timeout /t 4 /nobreak >nul
python "%~dp0live_drone_capture.py" http://127.0.0.1:8554/drone_stream
pause
