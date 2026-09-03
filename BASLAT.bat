@echo off
chcp 65001 >nul
title ?? MASt3R-3DGS T?nel & Dron Haritalama ?stasyonu
:MENU
cls
echo =============================================================
echo    ?? MASt3R-3DGS T?NEL VE DRON HAR?TALAMA MERKEZ?
echo =============================================================
echo.
echo  [1] ?? Canl? Dron / Webcam ile Tara & 3B Harita ??kar
echo  [2] ?? Bir MP4 Videosunu 3B Modele D?n??t?r
echo  [3] ?? 3B Haritay? A? (144+ FPS Gezgin)
echo  [4] ?? Gazebo 3B T?nel Sim?lasyonu & Klavye U?u?u
echo  [5] ?? ?oklu Koridorlar? Tek Haritada Birle?tir
echo  [6] ?? T?nel ?nceleme ve PDF Raporu ?ret
echo  [0] ? ??k??
echo.
echo =============================================================
set /p SECIM=L?tfen bir i?lem se?in [0-6]: 

if "%SECIM%"=="1" goto CANLI
if "%SECIM%"=="2" goto VIDEO_ISLE
if "%SECIM%"=="3" goto GORUNTULE
if "%SECIM%"=="4" goto GAZEBO_SIM
if "%SECIM%"=="5" goto BIRLESTIR
if "%SECIM%"=="6" goto RAPOR
if "%SECIM%"=="0" exit
goto MENU

:CANLI
cls
echo =============================================================
echo  ?? CANLI DRON / KAMERA ?LE TARAMA
echo =============================================================
echo.
echo  [1] Laptop / USB Web Kameras? (#0)
echo  [2] Dron / RTSP Canl? Yay?n? (Wi-Fi / Fiber)
echo.
set /p K_SECIM=Se?iminiz [1 veya 2]: 
if "%K_SECIM%"=="2" goto CANLI_RTSP
goto CANLI_WEBCAM

:CANLI_RTSP
echo.
set /p RTSP_URL=RTSP Linki (Varsay?lan: rtsp://192.168.1.100:8554/stream): 
if "%RTSP_URL%"=="" set RTSP_URL=rtsp://192.168.1.100:8554/stream
python live_drone_capture.py "%RTSP_URL%"
pause
goto MENU

:CANLI_WEBCAM
python live_drone_capture.py 0
pause
goto MENU

:VIDEO_ISLE
cls
echo =============================================================
echo  ?? MP4 V?DEOSUNDAN 3B HAR?TA ?RET?M?
echo =============================================================
echo.
set /p V_NAME=??lenecek video ad? veya yolu (Varsay?lan: ofisvideo.mp4): 
if "%V_NAME%"=="" set V_NAME=ofisvideo.mp4
python mast3r_to_3dgs.py "%V_NAME%" 50
python gaussian_renderer.py gaussian_scene.ply
pause
goto MENU

:GORUNTULE
cls
echo =============================================================
echo  ?? 3B MODEL G?R?NT?LEY?C? A?ILIYOR...
echo =============================================================
echo.
python gaussian_renderer.py gaussian_scene.ply
pause
goto MENU

:GAZEBO_SIM
cls
echo =============================================================
echo  ?? GAZEBO 3B T?NEL S?M?LASYONU VE KLAVYE U?U?U
echo =============================================================
echo.
echo  [1] Gazebo 3B D?nyas? Ba?lat?l?yor...
start "" wsl -d Ubuntu-22.04 -u root -- /root/start_gazebo.sh
echo  [2] Sim?lasyonun Y?klenmesi Bekleniyor (5 Saniye)...
timeout /t 5 /nobreak >nul
echo  [3] Dron Klavye U?u? ?stasyonu A??l?yor...
python gazebo_keyboard_teleop.py
pause
goto MENU

:BIRLESTIR
cls
echo =============================================================
echo  ?? ?OKLU HAR?TALARI B?RLE?T?RME
echo =============================================================
echo.
python map_stitcher.py
pause
goto MENU

:RAPOR
cls
echo =============================================================
echo  ?? T?NEL ?NCELEME RAPORU ?RET?L?YOR...
echo =============================================================
echo.
python tunnel_report_generator.py gaussian_scene_cache.npz
start tunel_inceleme_raporu.html
pause
goto MENU
