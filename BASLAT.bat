@echo off
chcp 65001 >nul
title ?? MASt3R-3DGS T?nel & Dron Haritalama ?stasyonu
:MENU
cls
echo =============================================================
echo    ?? MASt3R-3DGS T?NEL VE DRON HAR?TALAMA MERKEZ?
echo =============================================================
echo.
echo  [1] ?? Drondan Gelen Videoyu Kaydet & 3B Haritaya D?n??t?r
echo  [2] ?? Webcam ile Test Kayd? Al & 3B Harita ??kar
echo  [3] ?? Haz?r 3B Ofis Modelini A? (144+ FPS Gezgin)
echo  [4] ?? ?rnek Ofis Videosunu Ba?tan 3B Modele ?evir
echo  [5] ?? ?oklu Koridorlar? Tek Haritada Birle?tir
echo  [6] ?? T?nel ?nceleme ve PDF Raporu ?ret
echo  [0] ? ??k??
echo.
echo =============================================================
set /p SECIM=L?tfen bir i?lem se?in [0-6]: 

if "%SECIM%"=="1" goto DRON_KAYIT
if "%SECIM%"=="2" goto WEBCAM_TEST
if "%SECIM%"=="3" goto OFIS_AC
if "%SECIM%"=="4" goto OFIS_ISLE
if "%SECIM%"=="5" goto BIRLESTIR
if "%SECIM%"=="6" goto RAPOR
if "%SECIM%"=="0" exit
goto MENU

:DRON_KAYIT
cls
echo =============================================================
echo  ?? DRON CANLI YAYININDAN V?DEO KAYDI VE 3B HAR?TALAMA
echo =============================================================
echo.
set /p RTSP_URL=Dron RTSP Linki (Varsay?lan: rtsp://192.168.1.100:8554/stream): 
if "%RTSP_URL%"=="" set RTSP_URL=rtsp://192.168.1.100:8554/stream
python live_drone_capture.py "%RTSP_URL%"
pause
goto MENU

:WEBCAM_TEST
cls
echo =============================================================
echo  ?? WEBCAM ?LE TEST KAYDI VE 3B HAR?TALAMA
echo =============================================================
echo.
python live_drone_capture.py 0
pause
goto MENU

:OFIS_AC
cls
echo =============================================================
echo  ?? 3B OF?S MODEL? A?ILIYOR (144+ FPS)...
echo =============================================================
echo.
python gaussian_renderer.py gaussian_scene.ply
pause
goto MENU

:OFIS_ISLE
cls
echo =============================================================
echo  ?? OF?S V?DEOSU BA?TAN 3B MODELE D?N??T?R?L?YOR...
echo =============================================================
echo.
python mast3r_to_3dgs.py ofisvideo.mp4 50
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
