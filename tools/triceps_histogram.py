"""Histogram of left and right triceps_brachii volumes from tools/segment_triceps.py.

Every segmented arm with a triceps (> 0 ml) is counted, stacked by its status:
  usable           >= --min_ml and not touching the scan border (the selection rule)
  cut by FOV       touches the volume border: a partial muscle
  below threshold  not cut, but < --min_ml
Reference: Holzbaur et al. 2007 (J Biomech 40:742), triceps 372 +/- 177 cm^3, dominant arm of 10
healthy young adults (manual MRI segmentation), shown as mean +/- 1 SD.

Usage:
  python tools/triceps_histogram.py                       # data/triceps/triceps_survey.csv -> figures/triceps/
  python tools/triceps_histogram.py --min_ml 150 --out figures/triceps
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

USABLE, CUT, SMALL = "#2a78d6", "#eb6834", "#1baf7a"   # validated categorical slots 1-3 (all-pairs)
INK, INK2, GRID, REF = "#0b0b0b", "#52514e", "#e4e2dc", "#8f8d86"
REF_MEAN, REF_SD = 372.1, 177.3

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "pdf.fonttype": 42, "savefig.bbox": "tight",
})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--survey", default="data/triceps/triceps_survey.csv")
    ap.add_argument("--min_ml", type=float, default=100, help="usability threshold used in the survey (ml)")
    ap.add_argument("--bin_ml", type=float, default=10)
    ap.add_argument("--out", default="figures/triceps")
    args = ap.parse_args()

    d = pd.read_csv(args.survey)
    d = d[d.status == "segmented"]
    os.makedirs(args.out, exist_ok=True)
    top = max(d.left_ml.max(), d.right_ml.max(), REF_MEAN + REF_SD)
    bins = np.arange(0, np.ceil(top / args.bin_ml) * args.bin_ml + args.bin_ml, args.bin_ml)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
    rows = []
    for ax, side in zip(axes, ("left", "right")):
        v, cut = d[f"{side}_ml"], d[f"{side}_cut"] == 1
        present = v > 0
        groups = [("Usable", USABLE, present & ~cut & (v >= args.min_ml)),
                  ("Below threshold", SMALL, present & ~cut & (v < args.min_ml)),
                  ("Cut by field of view", CUT, present & cut)]
        ax.hist([v[m] for _, _, m in groups], bins=bins, stacked=True, color=[c for _, c, _ in groups],
                label=[f"{n} (n={int(m.sum())})" for n, _, m in groups], edgecolor="white", linewidth=0.5)
        ax.axvspan(REF_MEAN - REF_SD, REF_MEAN + REF_SD, color=REF, alpha=0.12, linewidth=0)
        ax.axvline(REF_MEAN, color=REF, linewidth=1.0, linestyle="--")
        ax.axvline(args.min_ml, color=INK2, linewidth=0.8, linestyle=":")
        u = v[groups[0][2]]
        ax.set_title(f"{side.capitalize()} arm: usable {len(u)}, mean {u.mean():.0f} ± {u.std():.0f} ml")
        ax.set_xlabel("Triceps brachii volume (ml)")
        ax.legend(frameon=False, loc="upper right")
        rows.append({"arm": side, "with triceps": int(present.sum()), "usable": len(u),
                     "usable mean ml": round(u.mean(), 1), "usable sd ml": round(u.std(), 1),
                     "usable median ml": round(u.median(), 1), "usable min ml": round(u.min(), 1),
                     "usable max ml": round(u.max(), 1), "cut by FOV": int(groups[2][2].sum()),
                     "below threshold": int(groups[1][2].sum())})
    axes[0].set_ylabel("Arms")
    ymax = axes[0].get_ylim()[1]
    axes[1].text(REF_MEAN, ymax * 0.5, "Holzbaur et al. 2007\nmean ± SD", ha="center", va="center",
                 fontsize=6.5, color=INK2)
    axes[0].text(args.min_ml, ymax * 0.97, f" {args.min_ml:.0f} ml\n threshold", ha="left", va="top",
                 fontsize=6.5, color=INK2)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.out, f"triceps_volume_histogram.{ext}"), dpi=300)
    plt.close(fig)

    both = d[(d.left_usable == 1) & (d.right_usable == 1)]
    stats = pd.DataFrame(rows)
    stats.to_csv(os.path.join(args.out, "triceps_volume_stats.csv"), index=False)
    print(stats.to_string(index=False))
    print(f"both arms usable: {len(both)} subjects, left-right r = {both.left_ml.corr(both.right_ml):.2f}, "
          f"mean |L-R| = {(both.left_ml - both.right_ml).abs().mean():.0f} ml")
    print(f"wrote {args.out}/triceps_volume_histogram.pdf/.png and triceps_volume_stats.csv")


if __name__ == "__main__":
    main()
