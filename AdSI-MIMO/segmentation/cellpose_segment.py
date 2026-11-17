"""Automated CD8 reference / imputed-image segmentation with the Cellpose cyto model (Cellpose v3).

Each input .npy holds a (2, H, W) array: [0] = acquired CD8 image, [1] = imputed CD8 image.
The acquired image is segmented once and cached; every imputation method is segmented with the
same Cellpose settings and later compared against the same reference masks.

Example
  python cellpose_segment.py --npy-dir imputed/adsimimo_dapi --patch-list patch_list.csv \
      --ref-mask-dir masks/reference --out-mask-dir masks/adsimimo_dapi
"""
import argparse, os, random
import numpy as np, pandas as pd, torch
from cellpose import models

random.seed(1); np.random.seed(1); torch.manual_seed(1)

# Segmentation settings (cellpose 3.1.1.3; model_type is not supported in Cellpose >= 4)
MODEL_TYPE = "cyto"          # Cellpose pretrained cytoplasm model, no fine-tuning
DIAMETER = None              # None -> model default diameter (diam_mean = 30 px for 'cyto')
FLOW_THRESHOLD = 0.45
CELLPROB_THRESHOLD = 0.0
MIN_REFERENCE_CELLS = 2      # patches whose reference mask has <= 1 cell are skipped


def segment(model, img):
    masks, _flows, _styles = model.eval(img, diameter=DIAMETER, flow_threshold=FLOW_THRESHOLD,
                                       cellprob_threshold=CELLPROB_THRESHOLD, do_3D=False)
    return masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy-dir", required=True, help="folder of (2,H,W) .npy patches [acquired, imputed]")
    ap.add_argument("--patch-list", required=True, help="CSV with column Image_Name (patch IDs without .npy)")
    ap.add_argument("--ref-mask-dir", required=True, help="reference masks are read from / written to here")
    ap.add_argument("--out-mask-dir", required=True, help="masks of the imputed images")
    ap.add_argument("--gpu", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.ref_mask_dir, exist_ok=True); os.makedirs(a.out_mask_dir, exist_ok=True)
    model = models.CellposeModel(gpu=a.gpu, model_type=MODEL_TYPE)
    names = pd.read_csv(a.patch_list)["Image_Name"].tolist()
    for name in names:
        f = f"{name}.npy"; data = np.load(os.path.join(a.npy_dir, f))
        acquired, imputed = data[0], data[1]
        ref_path = os.path.join(a.ref_mask_dir, f)
        if os.path.exists(ref_path):
            ref = np.load(ref_path)
        else:
            ref = segment(model, acquired); np.save(ref_path, ref)
        if len(np.unique(ref)) - 1 < MIN_REFERENCE_CELLS:
            continue
        np.save(os.path.join(a.out_mask_dir, f), segment(model, imputed))


if __name__ == "__main__":
    main()
