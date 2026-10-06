"""Length of the TotalSegmentator triceps_brachii masks (and of the humerus) per arm, to check
whether the masks hold the whole muscle. Compares with Holzbaur et al. 2007 (J Biomech 40:742):
triceps length 27.0 +/- 3.2 cm (centroidal path, tendon excluded), humerus 33.2 +/- 2.5 cm.

Per arm (triceps components >= 1 ml, assigned to the nearer humerus as in segment_triceps.py):
  *_axis_cm  extent along the structure's own long axis (first principal axis; 0.5-99.5 percentile
             of the voxel positions), the closest analogue of a centroidal-path length
  *_si_cm    superior-inferior extent in scanner coordinates
  slice_mm   largest voxel spacing of the scan
Output: <out>/triceps_geometry.csv and triceps_length.pdf/.png

Usage:
  python tools/triceps_geometry.py                     # all segmented subjects in data/triceps/triceps_survey.csv
"""
import argparse
import os
import sys
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from segment_triceps import DATA, load_canonical  # noqa: E402

TRI_LEN, TRI_SD, HUM_LEN, HUM_SD = 27.0, 3.2, 33.2, 2.5
USABLE, CUT, SMALL = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, REF = "#0b0b0b", "#52514e", "#e4e2dc", "#8f8d86"


def world_points(mask, affine):
    idx = np.array(np.nonzero(mask)).T
    return idx @ affine[:3, :3].T + affine[:3, 3]


def extents(pts):
    """(length along the first principal axis, superior-inferior extent), in cm."""
    c = pts - pts.mean(0)
    axis = np.linalg.svd(c[:: max(1, len(c) // 20000)], full_matrices=False)[2][0]
    t = c @ axis
    lo, hi = np.percentile(t, [0.5, 99.5])
    zlo, zhi = np.percentile(pts[:, 2], [0.5, 99.5])
    return (hi - lo) / 10, (zhi - zlo) / 10


def measure(args):
    subject, data_root = args
    seg = os.path.join(data_root, subject, "segmentations")
    tri, aff, vml = load_canonical(os.path.join(seg, "triceps_brachii.nii.gz"))
    zooms = nib.load(os.path.join(data_root, subject, "mri.nii.gz")).header.get_zooms()[:3]
    hum = {}
    for side in ("left", "right"):
        p = os.path.join(seg, f"humerus_{side}.nii.gz")
        if os.path.isfile(p):
            m, ha, _ = load_canonical(p)
            if m.any():
                hp = world_points(m, ha)
                hum[side] = (hp.mean(0), extents(hp), any(m.take([0, -1], axis=a).any() for a in range(3)))
    rows = []
    if not hum or not tri.any():
        return rows
    lab, n = ndimage.label(tri, ndimage.generate_binary_structure(3, 3))
    sizes = ndimage.sum(tri, lab, range(1, n + 1)) * vml
    arms = {}
    for c in [i + 1 for i, s in enumerate(sizes) if s >= 1.0]:
        pts = world_points(lab == c, aff)
        side = min(hum, key=lambda s: np.linalg.norm(pts.mean(0) - hum[s][0]))
        arms.setdefault(side, []).append(pts)
    for side, plist in arms.items():
        pts = np.concatenate(plist)
        t_axis, t_si = extents(pts)
        h_axis, h_si = hum[side][1]
        rows.append({"subject": subject, "arm": side, "triceps_ml": len(pts) * vml,
                     "triceps_axis_cm": t_axis, "triceps_si_cm": t_si,
                     "humerus_axis_cm": h_axis, "humerus_si_cm": h_si, "humerus_cut": int(hum[side][2]),
                     "slice_mm": float(max(zooms))})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--survey", default="data/triceps/triceps_survey.csv")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--out", default="figures/triceps")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    s = pd.read_csv(args.survey)
    subjects = [x for x in s[s.status == "segmented"].subject
                if os.path.isfile(os.path.join(args.data, x, "segmentations", "triceps_brachii.nii.gz"))]
    with Pool(args.workers) as p:
        rows = [r for rs in p.map(measure, [(x, args.data) for x in subjects]) for r in rs]
    g = pd.DataFrame(rows)
    st = s.set_index("subject")
    g["status"] = [("usable" if st.loc[r.subject, f"{r.arm}_usable"] == 1 else
                    "cut" if st.loc[r.subject, f"{r.arm}_cut"] == 1 else "small") for r in g.itertuples()]
    os.makedirs(args.out, exist_ok=True)
    g.round(2).to_csv(os.path.join(args.out, "triceps_geometry.csv"), index=False)

    # ---- summary
    def line(name, x):
        return f"{name:<42} n={len(x):4d}  {x.mean():5.1f} ± {x.std():4.1f}  (median {x.median():5.1f}, max {x.max():5.1f})"
    u = g[g.status == "usable"]
    print(line("triceps length, long axis (cm), usable", u.triceps_axis_cm))
    print(line("triceps length, long axis (cm), all", g.triceps_axis_cm))
    print(line("triceps SI extent (cm), usable", u.triceps_si_cm))
    print(line("humerus length, long axis (cm), usable", u.humerus_axis_cm))
    print(line("humerus length, long axis (cm), all", g.humerus_axis_cm))
    full = g.humerus_axis_cm >= HUM_LEN - 2 * HUM_SD
    print(f"arms whose humerus is at least {HUM_LEN - 2 * HUM_SD:.1f} cm (complete upper arm): {int(full.sum())} of {len(g)}"
          f" ({int((full & (g.status == 'usable')).sum())} of {len(u)} usable)")
    tri_full = g.triceps_axis_cm >= TRI_LEN - 2 * TRI_SD
    print(f"arms whose triceps is at least {TRI_LEN - 2 * TRI_SD:.1f} cm long: {int(tri_full.sum())} of {len(g)}")
    if full.any():
        print(line("triceps volume (ml), complete humerus", g[full].triceps_ml))
        print(line("triceps length (cm), complete humerus", g[full].triceps_axis_cm))
    print(f"correlation volume vs triceps length (all arms): r = {g.triceps_ml.corr(g.triceps_axis_cm):.2f}")
    print("slice spacing (mm) of usable arms:", u.slice_mm.round(1).value_counts().head(6).to_dict())

    # ---- figure: length histograms + volume vs length
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                         "axes.axisbelow": True, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "pdf.fonttype": 42, "savefig.bbox": "tight", "legend.fontsize": 7, "xtick.labelsize": 7,
                         "ytick.labelsize": 7})
    groups = [("Usable", USABLE, "usable"), ("Below threshold", SMALL, "small"), ("Cut by field of view", CUT, "cut")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    for ax, col, ref, sd, title in ((axes[0], "triceps_axis_cm", TRI_LEN, TRI_SD, "Triceps length"),
                                    (axes[1], "humerus_axis_cm", HUM_LEN, HUM_SD, "Humerus length (same arm)")):
        xmax = ref + 4 * sd                                   # beyond: segmentation errors, counted in the title
        over = int((g[col] > xmax).sum())
        bins = np.arange(0, xmax + 1.0, 1.0)
        ax.set_xlim(0, xmax)
        ax.hist([g[g.status == k][col] for _, _, k in groups], bins=bins, stacked=True,
                color=[c for _, c, _ in groups], label=[n for n, _, _ in groups], edgecolor="white", linewidth=0.4)
        ax.axvspan(ref - sd, ref + sd, color=REF, alpha=0.15, linewidth=0)
        ax.axvline(ref, color=REF, linestyle="--", linewidth=1)
        ax.set_title(f"{title}\n(Holzbaur {ref} ± {sd} cm)" + (f"; {over} > {xmax:.0f} cm not shown" if over else ""), fontsize=8)
        ax.set_xlabel("Length along long axis (cm)")
    axes[0].set_ylabel("Arms")
    axes[0].legend(frameon=False, loc="upper left")
    ax = axes[2]
    for n, c, k in groups:
        x = g[g.status == k]
        ax.scatter(x.triceps_axis_cm, x.triceps_ml, s=7, color=c, edgecolor="white", linewidth=0.3, label=n)
    ax.axvspan(TRI_LEN - TRI_SD, TRI_LEN + TRI_SD, color=REF, alpha=0.15, linewidth=0)
    ax.axhspan(372.1 - 177.3, 372.1 + 177.3, color=REF, alpha=0.10, linewidth=0)
    ax.set_xlim(0, TRI_LEN + 4 * TRI_SD)
    ax.set_xlabel("Triceps length (cm)")
    ax.set_ylabel("Triceps volume (ml)")
    ax.set_title("Volume vs length\n(grey: Holzbaur mean ± SD)", fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.out, f"triceps_length.{ext}"), dpi=300)
    print(f"wrote {args.out}/triceps_geometry.csv and triceps_length.pdf/.png")


if __name__ == "__main__":
    main()
