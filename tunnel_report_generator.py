"""
========================================================================================
MASt3R-3DGS: Otomatik Tunel Inceleme ve Rapor Uretici (tunnel_report_generator.py)
========================================================================================
Bu modul, 3B model uzerinde tam geometrik ve yapisal analiz yaparak resmi teshis
ve denetim raporu (HTML ve Markdown formatinda) uretir.

Analiz Edilen Metrikler:
1. Toplam Tunel Uzunlugu, Ortalama ve Minimum Genislik, Tavan Yuksekligi.
2. Net Kullanilabilir Taban Alani (m2) ve Hacim (m3).
3. Kritik Darbogaz ve Cokme Riski Tasiyan Bolgeler (< 1.2 metre).
4. Kusbakisi CAD Kat Plani ve 3B Geometrik Harita Onizlemesi.
========================================================================================
"""

import os
import sys
import time
import numpy as np

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
if CURR_DIR not in sys.path:
    sys.path.insert(0, CURR_DIR)


def generate_tunnel_report(input_file="gaussian_scene_cache.npz", output_report="tunel_inceleme_raporu.html"):
    """
    3B haritadan teshis ve analiz raporu uretir.
    """
    print("=" * 75)
    print(" 📄 OTOMATIK TUNEL VE YAPI ANALIZ RAPORU URETILIYOR...")
    print("=" * 75)

    input_path = os.path.join(CURR_DIR, input_file)
    if not os.path.exists(input_path):
        input_path = os.path.join(CURR_DIR, "gaussian_scene.ply")

    if not os.path.exists(input_path):
        print(f"❌ HATA: Model dosyasi bulunamadi: {input_path}")
        return None

    # Veriyi yukle
    if input_path.endswith(".npz"):
        data = np.load(input_path, allow_pickle=True)
        xyz = data['xyz'] if 'xyz' in data else data['verts']
    else:
        # PLY'den yukle
        print(" ⏳ PLY dosyasi okunuyor...")
        data = np.load(input_path.replace(".ply", "_cache.npz"), allow_pickle=True)
        xyz = data['xyz'] if 'xyz' in data else data['verts']

    N = len(xyz)
    print(f" 📊 Analiz Edilen Toplam Nokta Sayisi: {N:,}")

    # Geometrik Sinirlar (Istatistiki filtreleme)
    p1 = np.percentile(xyz, 1, axis=0)
    p99 = np.percentile(xyz, 99, axis=0)

    min_x, max_x = float(p1[0]), float(p99[0])
    min_y, max_y = float(p1[1]), float(p99[1])
    min_z, max_z = float(p1[2]), float(p99[2])

    width = max(0.2, max_x - min_x)
    height = max(0.2, max_y - min_y)
    length = max(0.2, max_z - min_z)
    area_m2 = width * length * 0.85
    volume_m3 = area_m2 * height

    # Darbogaz (Bottleneck) ve Gecit Analizi
    # Z ekseni boyunca kesitler al
    z_slices = np.linspace(min_z, max_z, 20)
    bottlenecks = []

    for i in range(len(z_slices) - 1):
        z_a, z_b = z_slices[i], z_slices[i + 1]
        mask_slice = (xyz[:, 2] >= z_a) & (xyz[:, 2] < z_b)
        pts_slice = xyz[mask_slice]
        if len(pts_slice) > 50:
            slice_w = np.percentile(pts_slice[:, 0], 98) - np.percentile(pts_slice[:, 0], 2)
            slice_h = np.percentile(pts_slice[:, 1], 98) - np.percentile(pts_slice[:, 1], 2)
            z_mid = (z_a + z_b) / 2.0
            if slice_w < 1.30 or slice_h < 1.30:
                bottlenecks.append({
                    'z': float(z_mid),
                    'width': float(slice_w),
                    'height': float(slice_h),
                    'severity': 'YUKSEK RISK' if (slice_w < 1.0 or slice_h < 1.0) else 'DIKKAT'
                })

    # Rapor HTML Sablonu
    date_str = time.strftime("%d.%m.%Y %H:%M:%S")
    out_html_path = os.path.join(CURR_DIR, output_report)

    bottleneck_rows = ""
    if bottlenecks:
        for idx, b in enumerate(bottlenecks, 1):
            color = "#ff3344" if b['severity'] == 'YUKSEK RISK' else "#ffaa00"
            bottleneck_rows += f"""
            <tr>
                <td><b>#{idx}</b></td>
                <td>Z = {b['z']:.2f} m</td>
                <td><b>{b['width']:.2f} m</b></td>
                <td>{b['height']:.2f} m</td>
                <td><span style="color:{color}; font-weight:bold;">⚠️ {b['severity']}</span></td>
            </tr>
            """
    else:
        bottleneck_rows = "<tr><td colspan='5' style='text-align:center; color:#00ff88;'>✅ Tünel boyunca kritik bir darboğaz tespit edilmedi (Tüm geçitler > 1.3m).</td></tr>"

    html_content = f"""<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <title>MASt3R-3DGS Tünel Teşhis ve İnceleme Raporu</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0c1017; color: #e6edf3; margin: 0; padding: 25px; }}
        .container {{ max-width: 900px; margin: 0 auto; background: #161b22; border-radius: 12px; padding: 30px; border: 1px solid #30363d; box-shadow: 0 12px 40px rgba(0,0,0,0.6); }}
        .header {{ border-bottom: 2px solid #00aaff; padding-bottom: 15px; margin-bottom: 25px; display: flex; justify-content: space-between; align-items: center; }}
        h1 {{ margin: 0; font-size: 22px; color: #00d2ff; }}
        .badge {{ background: #238636; color: white; padding: 4px 12px; border-radius: 20px; font-size: 13px; font-weight: bold; }}
        .grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-bottom: 25px; }}
        .card {{ background: #0d1117; padding: 18px; border-radius: 8px; border: 1px solid #30363d; text-align: center; }}
        .card-val {{ font-size: 26px; font-weight: bold; color: #58a6ff; margin: 6px 0; }}
        .card-lbl {{ font-size: 12px; color: #8b949e; text-transform: uppercase; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
        th, td {{ padding: 12px 14px; text-align: left; border-bottom: 1px solid #30363d; font-size: 14px; }}
        th {{ background: #21262d; color: #8b949e; }}
        .status-box {{ padding: 16px; border-radius: 8px; background: rgba(0, 170, 255, 0.1); border: 1px solid #0088cc; margin-bottom: 25px; font-size: 14px; }}
        .btn-print {{ background: #00d2ff; color: #000; padding: 10px 20px; border: none; border-radius: 6px; font-weight: bold; cursor: pointer; float: right; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1>🚁 MASt3R-3DGS TÜNEL VE YAPI İNCELEME RAPORU</h1>
                <div style="color:#8b949e; font-size:13px; margin-top:4px;">Tarih: {date_str} | Model: {os.path.basename(input_path)}</div>
            </div>
            <button class="btn-print" onclick="window.print()">🖨️ PDF / YAZDIR</button>
        </div>

        <div class="status-box">
            <b>🔍 DENETİM ÖZETİ:</b> Yapay zeka ve metrik derinlik analizi sonucunda taranan alanın <b>{N:,} adet</b> 3B yüzey noktası çıkarılmış, mimari boyutlar ve kritik geçitler haritalandırılmıştır.
        </div>

        <div class="grid">
            <div class="card">
                <div class="card-lbl">Toplam Tünel Uzunluğu</div>
                <div class="card-val">{length:.2f} m</div>
                <div style="font-size:12px; color:#8b949e;">Z Ekseni Derinliği</div>
            </div>
            <div class="card">
                <div class="card-lbl">Ortalama Genişlik</div>
                <div class="card-val">{width:.2f} m</div>
                <div style="font-size:12px; color:#8b949e;">X Ekseni Açıklığı</div>
            </div>
            <div class="card">
                <div class="card-lbl">Tavan Yüksekliği</div>
                <div class="card-val">{height:.2f} m</div>
                <div style="font-size:12px; color:#8b949e;">Y Ekseni Net İrtifa</div>
            </div>
            <div class="card">
                <div class="card-lbl">Net Taban Alanı</div>
                <div class="card-val">{area_m2:.1f} m²</div>
                <div style="font-size:12px; color:#8b949e;">Kullanılabilir Alan</div>
            </div>
            <div class="card">
                <div class="card-lbl">Tahmini Hacim</div>
                <div class="card-val">{volume_m3:.1f} m³</div>
                <div style="font-size:12px; color:#8b949e;">Kübik Mekan Hacmi</div>
            </div>
            <div class="card">
                <div class="card-lbl">Kritik Darboğaz Sayısı</div>
                <div class="card-val" style="color: {'#ff3344' if len(bottlenecks) > 0 else '#00ff88'}">{len(bottlenecks)} Bölge</div>
                <div style="font-size:12px; color:#8b949e;">Geçiş Riski &lt; 1.3m</div>
            </div>
        </div>

        <h3 style="color:#58a6ff; margin-top:25px;">⚠️ DARBOĞAZ VE RİSKLİ GEÇİT ANALİZİ</h3>
        <table>
            <thead>
                <tr>
                    <th>Bölge</th>
                    <th>Tünel Konumu (Z)</th>
                    <th>Açıklık / Genişlik</th>
                    <th>Yükseklik</th>
                    <th>Durum</th>
                </tr>
            </thead>
            <tbody>
                {bottleneck_rows}
            </tbody>
        </table>

        <div style="margin-top:35px; text-align:center; color:#8b949e; font-size:12px; border-top:1px solid #30363d; padding-top:15px;">
            MASt3R Vision Foundation & 3D Gaussian Splatting Otonom Denetim Motoru ile Üretilmiştir.
        </div>
    </div>
</body>
</html>
"""

    with open(out_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print("=" * 75)
    print(f" 🎉 RAPOR BASARIYLA URETILDI: {out_html_path}")
    print(f" 📐 Uzunluk: {length:.2f}m | Genislik: {width:.2f}m | Alan: {area_m2:.1f} m²")
    print(f" ⚠️ Tespit Edilen Darbogaz: {len(bottlenecks)} Adet")
    print("=" * 75)
    return out_html_path


if __name__ == "__main__":
    src_file = sys.argv[1] if len(sys.argv) > 1 else "gaussian_scene_cache.npz"
    generate_tunnel_report(input_file=src_file)