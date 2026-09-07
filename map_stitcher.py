"""
========================================================================================
MASt3R-3DGS: Coklu Segment ve Oda Birlestirici (map_stitcher.py)
========================================================================================
Bu modul, dronun farkli zamanlarda veya farkli koridorlarda taradigi coklu 3B harita
parcalarini (segmentlerini) otomatik olarak birlestirir (Multi-Scan Merging / Stitching).

Ozellikler:
1. Birden fazla .npz veya .ply harita parcasini yukler.
2. Voksel izgarasi uzerinde mukerrer noktalari filtreler.
3. Birlesik buyuk tunel haritasini (unified_tunnel_map.ply) uretir.
========================================================================================
"""

import os
import sys
import glob
import numpy as np

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
if CURR_DIR not in sys.path:
    sys.path.insert(0, CURR_DIR)


def stitch_scans(scan_files=None, output_ply="unified_tunnel_map.ply", voxel_filter=0.015):
    """
    Verilen harita dosyalarini tek bir birlesik 3DGS modelinde toplar.
    """
    print("=" * 75)
    print(" 🧩 COKLU ODA VE TUNEL SEGMENTLERI BIRLESTIRICISI (MAP STITCHER)")
    print("=" * 75)

    if scan_files is None or len(scan_files) == 0:
        # Klasordeki tum scan onbelleklerini otomatik bul
        scan_files = sorted(glob.glob(os.path.join(CURR_DIR, "*_cache.npz")))
        if len(scan_files) == 0:
            scan_files = sorted(glob.glob(os.path.join(CURR_DIR, "*.ply")))

    if not scan_files:
        print("❌ HATA: Birlestirilecek harita dosyasi bulunamadi!")
        return None

    print(f" 📦 Birlestirilecek Segment Sayisi: {len(scan_files)} Adet")
    for idx, f in enumerate(scan_files, 1):
        print(f"   [{idx}] -> {os.path.basename(f)}")

    all_xyz = []
    all_rgb = []
    all_scales = []
    all_quats = []
    all_opacity = []

    for f_path in scan_files:
        if f_path.endswith("_cache.npz"):
            data = np.load(f_path, allow_pickle=True)
            if 'xyz' in data:
                all_xyz.append(data['xyz'])
                all_rgb.append(data['rgb'])
                all_scales.append(data['scales'])
                all_quats.append(data['quats'])
                all_opacity.append(data['opacity'])
            elif 'verts' in data:
                all_xyz.append(data['verts'])
                all_rgb.append(data['cols'])
                N = len(data['verts'])
                all_scales.append(np.full((N, 3), 0.015, dtype=np.float32))
                q = np.zeros((N, 4), dtype=np.float32)
                q[:, 0] = 1.0
                all_quats.append(q)
                all_opacity.append(np.full(N, 0.85, dtype=np.float32))

    if len(all_xyz) == 0:
        print("❌ HATA: Dosyalardan gecerli 3B nokta verisi okunamadi!")
        return None

    # Tum parcalari dev bir matriste birlestir
    xyz_cat = np.vstack(all_xyz).astype(np.float32)
    rgb_cat = np.vstack(all_rgb).astype(np.float32)
    scales_cat = np.vstack(all_scales).astype(np.float32)
    quats_cat = np.vstack(all_quats).astype(np.float32)
    opac_cat = np.concatenate(all_opacity).astype(np.float32)

    # Voksel Izgara Filtresi ile Mukerrer Noktalari Ayikla
    if voxel_filter > 0:
        print(f"\n 🧹 Voksel Izgara Filtresi Uygulaniyor ({voxel_filter*100:.1f} cm)...")
        v_coords = np.floor(xyz_cat / voxel_filter).astype(np.int32)
        _, unique_indices = np.unique(v_coords, axis=0, return_index=True)

        xyz_final = xyz_cat[unique_indices]
        rgb_final = rgb_cat[unique_indices]
        scales_final = scales_cat[unique_indices]
        quats_final = quats_cat[unique_indices]
        opac_final = opac_cat[unique_indices]
        print(f"  -> Toplam {len(xyz_cat):,} noktadan {len(xyz_final):,} temiz nokta elde edildi.")
    else:
        xyz_final, rgb_final, scales_final, quats_final, opac_final = xyz_cat, rgb_cat, scales_cat, quats_cat, opac_cat

    # Cikis Dosyalarini Kaydet
    out_ply_path = os.path.join(CURR_DIR, output_ply)
    out_cache_path = out_ply_path.replace(".ply", "_cache.npz")

    # NPZ Onbellegi
    np.savez_compressed(
        out_cache_path,
        xyz=xyz_final, rgb=rgb_final,
        scales=scales_final, quats=quats_final,
        opacity=opac_final,
        cams=np.array([[0, 1, 0]], dtype=np.float32)
    )

    # Standart Binary PLY Kaydi
    sh0 = (rgb_final - 0.5) / 0.28209479177387814
    log_scales = np.log(np.maximum(scales_final, 1e-6))
    logit_opacities = np.log(opac_final / (1.0 - opac_final + 1e-6))
    num_splats = len(xyz_final)

    with open(out_ply_path, "wb") as f:
        header = f"""ply
format binary_little_endian 1.0
element vertex {num_splats}
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
"""
        f.write(header.encode('ascii'))
        normals_dummy = np.zeros_like(xyz_final, dtype=np.float32)
        vertex_data = np.zeros(num_splats, dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
            ('opacity', 'f4'),
            ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
            ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')
        ])
        vertex_data['x'] = xyz_final[:, 0]
        vertex_data['y'] = xyz_final[:, 1]
        vertex_data['z'] = xyz_final[:, 2]
        vertex_data['f_dc_0'] = sh0[:, 0]
        vertex_data['f_dc_1'] = sh0[:, 1]
        vertex_data['f_dc_2'] = sh0[:, 2]
        vertex_data['opacity'] = logit_opacities
        vertex_data['scale_0'] = log_scales[:, 0]
        vertex_data['scale_1'] = log_scales[:, 1]
        vertex_data['scale_2'] = log_scales[:, 2]
        vertex_data['rot_0'] = quats_final[:, 0]
        vertex_data['rot_1'] = quats_final[:, 1]
        vertex_data['rot_2'] = quats_final[:, 2]
        vertex_data['rot_3'] = quats_final[:, 3]
        f.write(vertex_data.tobytes())

    print("=" * 75)
    print(f" 🎉 TUM SEGMENTLER BIRLESTIRILDI: {out_ply_path}")
    print(f" 🔮 Toplam Birlesik Gaussian Splat: {num_splats:,} Adet")
    print(f" 💾 Dosya Boyutu: {os.path.getsize(out_ply_path) / (1024**2):.1f} MB")
    print("=" * 75)
    return out_ply_path


if __name__ == "__main__":
    files = sys.argv[1:] if len(sys.argv) > 1 else None
    stitch_scans(scan_files=files)