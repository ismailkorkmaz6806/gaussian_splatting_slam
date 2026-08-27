@echo off
chcp 65001 > nul
title 3D Gaussian Splatting (3DGS): Fotogercekci 144+ FPS Gezgin
echo ===================================================
echo   🔮 3D GAUSSIAN SPLATTING BASLATILIYOR...
echo ===================================================

python gaussian_renderer.py gaussian_scene.ply

pause
