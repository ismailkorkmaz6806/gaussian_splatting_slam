"""
MASt3R-SLAM: True Vision Foundation Pointmap & Feature Engine
-------------------------------------------------------------
Bu modül, Naver Labs tarafından geliştirilen resmi MASt3R yapay zeka modelini
kullanarak 2B video/kamera karelerinden doğrudan metrik 3B nokta bulutu (pointmap)
ve piksel bazlı güven (confidence) haritaları üretir.

İşlevler:
1. MASt3R ve DUSt3R alt modül yollarının sys.path'e dinamik eklenmesi.
2. MASt3R ViT-Large modelinin PyTorch ile GPU'ya yüklenmesi ve singleton önbelleklenmesi.
3. Giriş karelerinin Unsharp Masking (keskinleştirme) ve Vision Transformer için
   patch boyutuna göre ölçeklenip normalize tensörlere dönüştürülmesi.
"""

import os
import sys
import math
import numpy as np
import cv2
import PIL.Image
from PIL.ImageOps import exif_transpose
import torch
import torchvision.transforms as tvf

# ==============================================================================
# 1. MASt3R / DUSt3R REPO YOLLARININ BELİRLENMESİ VE IMPORT HAZIRLIĞI
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
SLAM_DIR = os.path.join(PARENT_DIR, "mast3r_slam")

# Olası repo ve alt modül yolları kontrol edilir
paths_to_check = [
    os.path.join(BASE_DIR, "mast3r_repo"),
    os.path.join(BASE_DIR, "mast3r_repo", "dust3r"),
    os.path.join(BASE_DIR, "mast3r_repo", "dust3r", "croco"),
    os.path.join(BASE_DIR, "dust3r_repo"),
    os.path.join(SLAM_DIR, "mast3r_repo"),
    os.path.join(SLAM_DIR, "mast3r_repo", "dust3r"),
    os.path.join(SLAM_DIR, "mast3r_repo", "dust3r", "croco"),
    os.path.join(SLAM_DIR, "dust3r_repo"),
]

for p in paths_to_check:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

# MASt3R model mimarisi ve çıkarım modüllerinin içe aktarılması
from mast3r.model import AsymmetricMASt3R
from dust3r.inference import inference

# ViT (Vision Transformer) için standart görüntü normalizasyonu [-1.0, 1.0] aralığı
ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Singleton model nesnesi tutucu
_mast3r_model = None

# ==============================================================================
# 2. MODEL YÜKLEME FONKSİYONU
# ==============================================================================
def get_mast3r_model(device="cuda", model_name="naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"):
    """
    MASt3R temel modelini Hugging Face / yerel önbellekten GPU'ya yükler.
    Model zaten yüklüyse tekrar yüklemez (Singleton tasarım deseni).
    
    Parametreler:
        device (str): 'cuda' veya 'cpu' donanım seçimi.
        model_name (str): İndirilecek/yüklenecek model ağırlık adı.
    Dönüş:
        AsymmetricMASt3R: Değerlendirme (eval) modundaki model örneği.
    """
    global _mast3r_model
    if _mast3r_model is None:
        print(f" 🌊 Naver MASt3R Temel Modeli Yükleniyor: '{model_name}'...")
        _mast3r_model = AsymmetricMASt3R.from_pretrained(model_name).to(device).eval()
        print(" ✅ MASt3R Modeli GPU'da Hazır!")
    return _mast3r_model


# ==============================================================================
# 3. GÖRÜNTÜ BOYUTLANDIRMA VE PATCH HİZALAMA
# ==============================================================================
def _resize_and_crop(pil_img, target_size=512, patch_size=16):
    """
    Görüntüyü en-boy oranını koruyarak yeniden boyutlandırır ve
    Vision Transformer'ın patch_size (16x16) katlarına tam bölünecek şekilde kırpar.
    
    Parametreler:
        pil_img (PIL.Image): Giriş görüntüsü.
        target_size (int): Uzun kenar hedef piksel boyutu (varsayılan: 512).
        patch_size (int): ViT patch katı (varsayılan: 16).
    Dönüş:
        PIL.Image: Boyutlandırılmış ve patch katına kırpılmış görüntü.
    """
    # EXIF yönlendirme bilgisini düzelt ve RGB'ye çevir
    pil_img = exif_transpose(pil_img).convert("RGB")
    W1, H1 = pil_img.size
    S = max(W1, H1)
    new_size = (int(round(W1 * target_size / S)), int(round(H1 * target_size / S)))
    resized = pil_img.resize(new_size, PIL.Image.LANCZOS)

    # 16'nın katı olacak şekilde merkezden kırpma (ViT sınır hatasını önler)
    W, H = resized.size
    cx, cy = W // 2, H // 2
    halfw = int(((2 * cx) // patch_size) * patch_size / 2)
    halfh = int(((2 * cy) // patch_size) * patch_size / 2)
    cropped = resized.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))
    return cropped


# ==============================================================================
# 4. KARE ÖN İŞLEME VE TENSÖR OLUŞTURMA
# ==============================================================================
def prepare_frame_dict(frame_bgr_or_rgb, idx, target_size=512, is_bgr=True):
    """
    NumPy kare dizisini (OpenCV görüntüsü) keskinleştirilmiş, yüksek netlikli
    MASt3R tensör sözlüğüne dönüştürür.
    
    Parametreler:
        frame_bgr_or_rgb (np.ndarray): Giriş OpenCV kare matrisi.
        idx (int): Karenin indeks numarası.
        target_size (int): Çıkarım çözünürlüğü (örn: 512 veya 336).
        is_bgr (bool): Kare BGR formatında mı (True) yoksa RGB mi (False).
    Dönüş:
        dict: MASt3R modeline girdi olarak verilecek tensör ve metadata sözlüğü.
    """
    if is_bgr:
        rgb_arr = cv2.cvtColor(frame_bgr_or_rgb, cv2.COLOR_BGR2RGB)
    else:
        rgb_arr = frame_bgr_or_rgb

    # Unsharp Masking ile kenar ve doku keskinliği artırma (Netlik yükseltme)
    blurred = cv2.GaussianBlur(rgb_arr, (0, 0), 1.2)
    sharp_rgb = cv2.addWeighted(rgb_arr, 1.35, blurred, -0.35, 0)

    # PIL formatına aktar ve ViT boyutlarına hizala
    pil_img = PIL.Image.fromarray(sharp_rgb)
    processed = _resize_and_crop(pil_img, target_size=target_size)
    
    # [-1.0, 1.0] aralığına normalize edilmiş [1, 3, H, W] PyTorch tensörü
    tensor_img = ImgNorm(processed)[None]

    return {
        'img': tensor_img,                                    # Model tensörü
        'true_shape': np.int32([processed.size[::-1]]),       # Gerçek [H, W] boyutu
        'idx': idx,                                           # Kare indeksi
        'instance': str(idx),                                 # Benzersiz örnek kimliği
        'rgb_np': np.array(processed, dtype=np.float32) / 255.0  # [0, 1] aralığında RGB dizi
    }

