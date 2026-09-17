# 🔮 3D Gaussian Splatting (3DGS) & Otonom 3B Lidar SLAM Dron Simülasyonu

Bu repo; ham monoküler videolardan **Naver MASt3R** yapay zeka çıkarımı ile fotogerçekçi **3D Gaussian Splatting (.ply / .npz)** modelleri üreten, bu modelleri **144+ FPS** donanımsal GPU hızlandırmasıyla görselleştiren ve **GPS Olmayan (GPS-Denied) Ortamlarda 3B Lidar SLAM ile Uçan PX4 Otonom Dron Simülasyonunu (Gazebo Sim)** içeren uçtan uca profesyonel bir robotik ve 3B rekonstrüksiyon paketidir.

---

## 🌟 Öne Çıkan Özellikler

### 🚁 1. GPS-Denied Otonom Dron & 3B Lidar SLAM Simülasyonu (Gazebo & PX4 SITL)
* **📡 360° 3D Lidar SLAM Odometrisi:** Dronun tepesindeki katı hal 3D Lidar verileri ile gerçek zamanlı 6-DOF konum (X, Y, Z, Roll, Pitch, Yaw) hesaplanır ve MAVLink `#331 ODOMETRY` üzerinden PX4 EKF2 filtresine basılır.
* **🎯 Milimetrik Konum Kilidi (~3.9 cm Hata Payı):** Tamamen kapalı iç mekanlarda (mağara, ev, tünel) GPS donanımsal olarak kapatılmış olsa dahi **Position** ve **Hold** modunda havada çivi gibi asılı kalır.
* **⚡ %100 Real-Time Factor (0 Kasma):** Optimize edilmiş hafif Lidar boru hattı ve sıfır CPU darboğazı ile simülasyon gerçek zamanlı %100 hızda (1.00x) çalışır.
* **🎮 Gerçek RC Kumanda Desteği & Failsafe:** iFlight Commando 8 (OpenTX/EdgeTX) USB kumandasıyla tam manuel/otonom kontrol. Kumanda kablosu çekildiğinde dron kesinlikle düşmez veya inişe geçmez; havada kilitlenir (**Stop & Wait**).
* **📊 Fiziksel Ground-Truth Doğrulama:** `kiyasla_ground_truth.py` aracıyla uçuş logları Gazebo'nun mutlak yer gerçeğiyle karşılaştırılır ve santimetre mertebesinde gerçekçi SLAM doğruluğu raporlanır.

### 🔮 2. 3D Gaussian Splatting (3DGS) Ultimate Viewer
* **⚡ 144+ FPS Donanımsal GPU Hızlandırması:** Pygame + OpenGL VBO (Vertex Buffer Object) mimarisi ile milyonlarca 3B Gaussian Splat'ı sıfır gecikmeyle render eder.
* **🏠 Donanımsal Tavan Gizleme / Kesme (`[G]` / `[7]` / `[8]`):** `glClipPlane` donanım hızlandırması ile tavanı tek tuşla keserek ofis ve odaların içini kuşbakışı görünür kılar.
* **🗺️ 2B Mimari Kat Planı & $m^2$ Alan Hesabı (`[T]`):** 90° dik kuşbakışı modunda odanın genişlik, uzunluk ve net kullanım $m^2$ alanını hesaplar.
* **📏 3B Metrik Lazer Cetvel (`[E]`):** GPU Depth Buffer (`gluUnProject`) ile ekranda tıklanan iki nokta arasındaki gerçek metre mesafesini ölçer.
* **🎬 60 FPS Pürüzsüz MP4 Video Kaydedici (`[V]`):** 3B gezinme ekranını doğrudan GPU'dan 60 FPS MP4 video olarak kaydeder.
* **🌐 Bağımsız Web 3B Model Çıktısı (`[K]`):** Three.js tabanlı, herhangi bir tarayıcıda veya telefonda açılabilen tek dosyalık interaktif HTML çıktısı üretir.
* **🔍 Kamera Zoom (`[+] / [-] / Fare Tekerleği`) & Splat Boyutu (`[X] / [C]`)**

---

## 🛠️ Dron Donanım Mimarisi (`x500_vision`)

| Bileşen | Özellik / Model |
| :--- | :--- |
| **Gövde (Frame)** | Holybro / NXP HoverGames x500 Karbon Fiber Quadcopter (500 mm) |
| **Tahrik Sistemi** | 4x 5010 Outrunner Fırçasız Motor + 10x4.5 Karbon Pervaneler (4S LiPo) |
| **Otopilot** | Pixhawk 6C / FMUv5-v6 (PX4 Autopilot v1.15+ EKF2 Odometry) |
| **3B Tepe Lidar** | Livox Mid-360 / Velodyne Puck sınıfı 360° x 30° 3D Lidar (25m menzil) |
| **Alt Altimetre** | Lightware LW20 Hassas Lazer Mesafe Sensörü (0.1 - 50m) |
| **FPV / 3DGS Kamera** | Monoküler HD Global Shutter RGB Kamera + 25m LED Keşif Projektörü |
| **Görev Bilgisayarı** | NVIDIA Jetson Orin / Xavier NX sınıfı Onboard AI/SLAM Ünitesi |
| **Kumanda (RC)** | iFlight Commando 8 (OpenTX/EdgeTX 2.4GHz ELRS / USB HID) |

---

## 📂 Dosya Yapısı

* **`SIMULASYON_BASLAT.sh`**: Simülasyonu, Lidar SLAM köprüsünü ve QGC'yi tek tıkla başlatan ana script.
* **`drone_lidar_odometry_bridge.py`**: 3B Lidar Odometrisini PX4 MAVLink (#331) EKF2'ye aktaran yüksek hızlı köprü.
* **`kiyasla_ground_truth.py`**: Uçuş logunu (.ulg) Gazebo gerçek konumuyla milimetrik kıyaslayan analiz aracı.
* **`gps_kontrol.py`**: Dronun GPS'inin donanımsal kapalı olduğunu kanıtlayan denetleyici.
* **`TEMIZLE.sh`**: Tüm arka plan simülasyon süreçlerini ve portları temizleme aracı.
* **`live_drone_capture.py`**: Dron kamerasından canlı FPV kokpit ve 3DGS veri toplama aracı.
* **`gaussian_renderer.py`**: 144+ FPS OpenGL 3B Gaussian Splatting Gezgini ve HUD.
* **`mast3r_to_3dgs.py`**: Ham videodan 3D Gaussian Splat (.ply / .npz) üreten yapay zeka motoru.
* **`pro_features.py`**: Tavan kesici, Lazer cetvel, Kat planı, Video kaydedici ve Web HTML modülleri.
* **`calistir_gaussian.bat`**: 3DGS Görüntüleyicisini doğrudan başlatan dosya.
* **`gazebo_cave_world/`**: Özel tünel, mağara ve eşyalı büyük ev dünyaları (.sdf).
* **`px4_models/`**: Simülasyon için optimize edilmiş 3D Lidar'lı x500_vision dron modeli.

---

## 🚀 Kurulum & Çalıştırma

### 1. Python Gereksinimlerini Yükleyin:
```bash
pip install torch torchvision pygame-ce PyOpenGL PyOpenGL_accelerate opencv-python numpy scipy pymavlink pyulog
```

### 2. Otonom Dron Simülasyonunu Başlatın (Gazebo + PX4 + QGC):
```bash
./SIMULASYON_BASLAT.sh
```
* Çıkan menüden dünyanızı seçin (`1: Mağara`, `2: Eşyalı Büyük Ev`).
* Gazebo, PX4 SITL, 3B Lidar Odometri Köprüsü ve QGroundControl otomatik olarak ayağa kalkar.
* Dron GPS olmadan, Lidar konum kilidiyle **Position** modunda hazır hale gelir.
* iFlight Commando 8 kumandanızla veya menüden `[4]` tuşlayarak motorları başlatabilirsiniz (Arm).

### 3. Lidar SLAM Doğruluğunu Analiz Edin:
Uçuştan sonra Lidar SLAM verinizin yer gerçeğiyle (Ground-Truth) ne kadar örtüştüğünü tek komutla raporlayın:
```bash
python3 kiyasla_ground_truth.py
```

### 4. 3D Gaussian Splatting Modelini Görüntüleyin:
```bash
python gaussian_renderer.py
```
*(veya Windows üzerinde `calistir_gaussian.bat` dosyasına çift tıklayın)*

### 5. Yeni Bir Videoyu 3DGS Modeline Dönüştürün:
```bash
python mast3r_to_3dgs.py yeni_video.mp4 50
```

---

## ⌨️ Klavye Kısayolları

| Tuş | Fonksiyon |
| :--- | :--- |
| **`[W / A / S / D]`** | İleri / Sol / Geri / Sağ Serbest Uçuş |
| **`[SPACE / SHIFT]`** | Yukarı Yüksel / Aşağı Alçal |
| **`[Fare Sol Sürükle]`** | 360° Serbest Kamera Açısı |
| **`[+]` / `[-]` / `Tekerlek`** | 🔍 Yakınlaş (Zoom In) / Uzaklaş (Zoom Out) |
| **`[0]`** | Zoom Açısını Sıfırla (60°) |
| **`[X]` / `[C]`** | 🔮 Splat / Nokta Boyutunu Büyüt / Küçült |
| **`[G]`** | 🏠 Tavanı Gizle / Aç |
| **`[7]` / `[8]`** | ✂️ Tavan Kesme Yüksekliğini Ayarla |
| **`[T]`** | 🗺️ 90° Kuşbakışı Kat Planı ve $m^2$ Hesabı |
| **`[E]`** | 📏 3B Lazer Cetvel (Metre Ölçümü) |
| **`[V]`** | 🎬 60 FPS MP4 Video Kaydı Başlat / Bitir |
| **`[K]`** | 🌐 Web / HTML 3B Modelini Dışa Aktar |
| **`[Y]` / `[A]`** | 🪞 Sağ / Sol Yönünü Tersine Çevir |
| **`[P]`** | 🎥 Sinematik Otomatik Tur |
| **`[H]` / `[TAB]`** | 🎮 Kontrol Panelini Gizle / Göster |
| **`[R]`** | 🔄 Başa Sıfırla |
