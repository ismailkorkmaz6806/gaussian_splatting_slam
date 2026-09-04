@echo off
chcp 65001 >nul
title 🤖 Webots 3B Dron & 3DGS Haritalama İstasyonu
cls
echo =============================================================
echo  🤖 WEBOTS 3B DRON VE CANLI HARİTALAMA SİSTEMİ
echo =============================================================
echo.
set "WEBOTS_HOME=C:\Program Files\Webots"
set "PYTHONPATH=C:\Program Files\Webots\lib\controller\python;%PYTHONPATH%"

echo  [1/2] Webots 3B Dron Dünyası Başlatılıyor...
start "" "C:\Program Files\Webots\msys64\mingw64\bin\webots.exe" --mode=realtime "%~dp0webots_project\worlds\drone_tunnel_world.wbt"

echo.
echo  [2/2] Canlı Dron Kamera Yayınına Bağlanılıyor (Lütfen Webots açılınca Play ▶️ tuşuna basın)...
python live_drone_capture.py http://127.0.0.1:8554/drone_stream
pause
