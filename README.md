# 🔮 3D Gaussian Splatting (3DGS) Ultimate Viewer & Reconstruction

Bu repo, ham monoküler videolardan **Naver MASt3R** yapay zeka çıkarımı ile fotogerçekçi **3D Gaussian Splatting (.ply / .npz)** modelleri üreten ve bu modelleri **144+ FPS** donanımsal GPU hızlandırmasıyla görselleştiren profesyonel bir 3B rekonstrüksiyon ve analiz paketidir.

---

## 🌟 Öne Çıkan Özellikler

1. **⚡ 144+ FPS Donanımsal GPU Hızlandırması:**
   * Pygame + OpenGL VBO (Vertex Buffer Object) mimarisi ile milyonlarca 3B Gaussian Splat'ı sıfır gecikmeyle render eder.
2. **🏠 Donanımsal Tavan Gizleme / Kesme (`[G]` / `[7]` / `[8]`):**
   * `glClipPlane` donanım hızlandırması ile tavanı tek tuşla keserek ofis ve odaların içini kuşbakışı görünür kılar.
3. **🗺️ 2B Mimari Kat Planı & $m^2$ Alan Hesabı (`[T]`):**
   * 90° dik kuşbakışı modunda odanın genişlik, uzunluk ve net kullanım $m^2$ alanını hesaplar.
4. **📏 3B Metrik Lazer Cetvel (`[E]`):**
   * GPU Depth Buffer (`gluUnProject`) ile ekranda tıklanan iki nokta arasındaki gerçek metre mesafesini ölçer.
5. **🎬 60 FPS Pürüzsüz MP4 Video Kaydedici (`[V]`):**
   * 3B gezinme ekranını doğrudan GPU'dan 60 FPS MP4 video olarak kaydeder.
6. **🌐 Bağımsız Web 3B Model Çıktısı (`[K]`):**
   * Three.js tabanlı, herhangi bir tarayıcıda veya telefonda açılabilen tek dosyalık interaktif HTML çıktısı üretir.
7. **🔍 Kamera Zoom (`[+] / [-] / Fare Tekerleği`) & Splat Boyutu (`[X] / [C]`)**

---

## 📂 Dosya Yapısı

* **`calistir_gaussian.bat`**: 3DGS Görüntüleyicisini doğrudan başlatan çift tıklamalık dosya.
* **`gaussian_renderer.py`**: 144+ FPS OpenGL 3B Gezgin ve şık yarı saydam Kontrol Paneli (HUD).
* **`mast3r_to_3dgs.py`**: Ham videodan 3D Gaussian Splat (.ply ve .npz) üreten yapay zeka motoru.
* **`pro_features.py`**: Tavan kesici, Lazer cetvel, Kat planı & $m^2$ hesabı, Video kaydedici ve Web HTML modülleri.

---

## 🚀 Kurulum & Çalıştırma

### 1. Gereksinimleri Yükleyin:
```bash
pip install torch torchvision pygame-ce PyOpenGL PyOpenGL_accelerate opencv-python numpy scipy
```

### 2. Modeli Görüntüleyin:
```bash
python gaussian_renderer.py
```
*(veya `calistir_gaussian.bat` dosyasına çift tıklayın)*

### 3. Yeni Bir Videoyu 3DGS Modeline Dönüştürün:
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
