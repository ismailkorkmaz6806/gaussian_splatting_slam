@echo off
chcp 65001 >nul
title ?? MASt3R-3DGS T?nel & Dron Haritalama ?stasyonu
:MENU
cls
echo =============================================================
echo    ?? MASt3R-3DGS T?NEL VE DRON HAR?TALAMA MERKEZ?
echo =============================================================
echo.
echo  [1] ?? Canl? Dron / Sanal Yay?n ile Tara & 3B Harita ??kar
echo  [2] ?? Bir MP4 Videosunu 3B Modele D?n??t?r
echo  [3] ?? 3B Haritay? A? (144+ FPS Gezgin & 3B Dron Modu)
echo  [4] ?? Gazebo Sim (gz sim) ile 3B Dronu A? & Canl? Haritala
echo  [5] ?? ?oklu Koridorlar? Tek Haritada Birle?tir
echo  [6] ?? T?nel ?nceleme ve PDF Raporu ?ret
echo  [0] ? ??k??
echo.
echo =============================================================
set /p SECIM=L?tfen bir i?lem se?in [0-6]: 

if "%SECIM%"=="1" goto CANLI
if "%SECIM%"=="2" goto VIDEO_ISLE
if "%SECIM%"=="3" goto GORUNTULE
if "%SECIM%"=="4" goto GAZEBO
if "%SECIM%"=="5" goto BIRLESTIR
if "%SECIM%"=="6" goto RAPOR
if "%SECIM%"=="0" exit
goto MENU

:GAZEBO
call calistir_gazebo_simulasyon.bat
goto MENU

:CANLI
cls
echo =============================================================
echo  ?? CANLI DRON / KAMERA ?LE TARAMA VE HAR?TALAMA
echo =============================================================
echo.
echo  [1] Laptop / USB Web Kameras? (#0)
echo  [2] Ger?ek Dron Canl? Yay?n? (RTSP / Wi-Fi / Fiber)
echo  [3] ?? Sanal Dron Canl? Yay?n Sim?lasyonu (PC ??i Test)
echo.
set /p K_SECIM=Se?iminiz [1-3]: 
if "%K_SECIM%"=="3" goto CANLI_SIMULE
if "%K_SECIM%"=="2" goto CANLI_RTSP
goto CANLI_WEBCAM

:CANLI_SIMULE
cls
echo =============================================================
echo  ?? SANAL DRON CANLI YAYINI VE HAR?TALAMA BA?LATILIYOR...
echo =============================================================
start "" python simulated_drone_streamer.py
timeout /t 2 /nobreak >nul
python live_drone_capture.py http://127.0.0.1:8554/drone_stream
pause
goto MENU

:CANLI_RTSP
echo.
set /p RTSP_URL=RTSP / HTTP Yay?n Linki (Varsay?lan: rtsp://192.168.1.100:8554/stream): 
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
