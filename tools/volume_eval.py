"""Per-muscle volume error of main.py predictions (and of the template alone).

main.py writes <save_path>/predictions.npz (per test sample, point i = template point i moved).
The template mesh stored in mean_shape.npz by prepare_cactus.py is deformed with those
displacements (each mesh vertex follows its nearest template points) and its enclosed volume is
compared with the exact volume of the true muscle mesh (subjects.csv, <muscle>_mesh_ml).
Volumes are in each subject's own space: the normalised -> mm scale comes from affines.npz.

This needs the prediction to keep point correspondence with the template (point i stays near
template point i), which a model trained with Chamfer loss on absolute coordinates does not
guarantee. If the median point displacement from the template exceeds --max_disp_mm the model
volume is reported as NaN: it cannot be measured, and volumes from the bare point cloud are not
reliable for these thin muscles (-35 to -96 % on the true point clouds).

Usage:
  python tools/volume_eval.py --prepared data/cactus45 --results results/cactus_run
  python tools/volume_eval.py --prepared data/cactus45_loo_v2 --split_dir data/cactus45_loo_v2/folds/fold_03 \\
      --results results/cactus_loo_v2/fold_03
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_totalseg_mri import GRID_SHAPE  # noqa: E402


def mesh_volume(verts, faces):
    t = verts[faces]
    return abs(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum()) / 6


def ml_per_unit3(affine):
    """ml per (normalised unit)^3: |det| of normalised -> mm, as composed in convert_to_mm."""
    d = np.array(GRID_SHAPE, np.float64) - 1
    jac = affine[:3, :3] @ np.diag([-d[0] / 2, -d[1] / 2, d[2] / 2])
    return abs(np.linalg.det(jac)) / 1000


def volume_rows(prepared, split_dir, results, max_disp_mm=15.0):
    """One row per test sample, muscle and predictor (model / template)."""
    pred_path = os.path.join(results, "predictions.npz")
    if not os.path.isfile(pred_path):
        return []
    mean = np.load(os.path.join(split_dir, "mean_shape.npz"))
    if "mesh_verts" not in mean.files:
        raise SystemExit(f"{split_dir}/mean_shape.npz has no template mesh - rerun prepare_cactus.py")
    labels = json.load(open(glob.glob(os.path.join(prepared, "labels", "*.json"))[0]))
    subjects = pd.read_csv(os.path.join(prepared, "subjects.csv")).set_index("id")
    affines = np.load(os.path.join(prepared, "affines.npz"))
    preds = np.load(pred_path)
    tmpl, faces, vlab = mean["mean_pc_np"], mean["mesh_faces"], mean["mesh_label"]
    idx, w = mean["mesh_nn_idx"], mean["mesh_nn_w"]
    face_lab = vlab[faces[:, 0]]
    rows = []
    for sid in preds.files:
        disp = preds[sid] - tmpl  # (N_out, 3)
        scale = ml_per_unit3(affines[sid])
        disp_mm = np.median(np.linalg.norm(disp, axis=1)) * np.cbrt(scale * 1000)
        corresponds = disp_mm <= max_disp_mm
        if not corresponds:
            print(f"{sid}: median point displacement from the template {disp_mm:.0f} mm > {max_disp_mm:.0f} mm - "
                  f"the prediction does not keep template correspondence, model volume set to NaN")
        for pred_name, verts in (("model", mean["mesh_verts"] + (disp[idx] * w[..., None]).sum(1)),
                                 ("template", mean["mesh_verts"])):
            for lab, organ in labels.items():
                vol = mesh_volume(verts, faces[face_lab == int(lab)]) * scale
                if pred_name == "model" and not corresponds:
                    vol = np.nan
                true = subjects.loc[sid, f"{organ}_mesh_ml"]
                rows.append({"id": sid, "organ": organ, "pred": pred_name, "pred_ml": vol, "true_ml": true,
                             "err_ml": vol - true, "abs_err_pct": 100 * abs(vol - true) / true})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prepared", required=True, help="prepare_cactus.py output (data, affines, subjects.csv)")
    ap.add_argument("--split_dir", help="dir with mean_shape.npz used for training (default: --prepared)")
    ap.add_argument("--results", required=True, help="main.py --save_path containing predictions.npz")
    ap.add_argument("--max_disp_mm", type=float, default=15.0)
    args = ap.parse_args()
    rows = volume_rows(args.prepared, args.split_dir or args.prepared, args.results, args.max_disp_mm)
    if not rows:
        raise SystemExit(f"no predictions.npz in {args.results} - rerun main.py (or with --eval_only)")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.results, "csv", "volumes.csv"), index=False)
    t = df.groupby(["organ", "pred"]).agg(true_ml=("true_ml", "mean"), abs_err_ml=("err_ml", lambda e: e.abs().mean()),
                                          abs_err_pct=("abs_err_pct", "mean")).unstack("pred")
    print(t.round(1).to_string())


if __name__ == "__main__":
    main()
