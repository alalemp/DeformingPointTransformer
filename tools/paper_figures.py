"""Publication figures for the cactus-pose arm experiments (vector PDF + 300 dpi PNG).

Figures (written to --out):
  figS1_full_body         whole body per run (frame): skin, skeleton, target muscles; frame-to-frame skin motion
  fig1_data_overview      3D render of one arm per run (frame): skin crop, the four target muscles, humerus
  fig2_training_curves    per run: train / test loss, and test CD of the model against the template
  fig3_volume_scatter     per muscle: predicted vs true volume of every test arm, model and template
  fig4_error_surface      one test arm: template and predicted muscle surfaces coloured by distance (mm) to the true muscles
  table_metrics.csv/.tex  per run and muscle: CD, HD95, volume error (model and template)

Each run is "label=prepared_dir:results_dir[:frame]" (prepare_cactus.py output, main.py --save_path
or run_folds.py --save, and the simulation frame, default 160). For leave-N-out results (fold_XX
subdirectories) all finished folds are combined: every held-out arm appears once in the volume
scatter and the table (fold results weighted by their number of test arms); the training curves show
the median and interquartile range over folds; fig 4 uses the fold in which --sample was tested.
Runs without predictions yet are drawn where possible and skipped elsewhere.

Usage:
  python tools/paper_figures.py --out figures/paper \\
      --run "Frame 160=data/cactus160_s10:results/cactus160_s10_p0:160" \\
      --run "Frame 220=data/cactus220_s10:results/cactus220_s10_p0:220" \\
      --sample BB_0_TRL_-0.6_TRM_0_TRLN_0.6_T_1_R
"""
import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_cactus import (DATA, DEFAULT_MUSCLES, LAYOUTS, detect_layout, find_subjects,  # noqa: E402
                            load_muscles, parse_muscles, pick_humerus, read_ply, sample_surface, xyz)
from visualize_seg import to_world_mm  # noqa: E402
from volume_eval import volume_rows  # noqa: E402

# validated palette (dataviz reference instance): categorical slots 1-4, sequential blue ramp
MUSCLE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # adjacent-pair validated; always labelled
MODEL, TEMPLATE = "#2a78d6", "#eb6834"                         # all-pairs validated
SKIN, BONE = "#d9cbbd", "#ece8df"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e2dc"
BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
NAMES = {"biceps_brachii": "Biceps brachii", "triceps_lateral": "Triceps (lateral)",
         "triceps_medial": "Triceps (medial)", "triceps_long": "Triceps (long)"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "lines.linewidth": 1.4, "pdf.fonttype": 42, "savefig.bbox": "tight",
})


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=300)
    plt.close(fig)
    print(f"wrote {out}/{name}.pdf/.png")


def view(p):
    """world (x right, y up, z anterior) -> plot axes (x, z, y) so the plot is z-up."""
    return p[:, [0, 2, 1]]


def mesh(ax, verts, faces, color, alpha=1.0, light=(-0.4, -0.8, 0.6)):
    """Triangle mesh with simple two-sided Lambert shading (per face)."""
    tri = view(verts)[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    lit = np.abs(n @ (np.array(light) / np.linalg.norm(light)))
    rgb = np.array(matplotlib.colors.to_rgb(color))
    fc = np.c_[np.clip(rgb * (0.55 + 0.45 * lit[:, None]), 0, 1), np.full(len(tri), alpha)]
    ax.add_collection3d(Poly3DCollection(tri, facecolors=fc, edgecolors="none"))


VIEWS = {"Anterior": (8, 90), "Posterior": (8, -90)}   # (elevation, azimuth) in plot axes (x, z, y)


def setup_3d(ax, pts, title, elev=8, azim=90, zoom=1.3):
    """Axes box fitted to the data extent (no cube padding), equal scale on all axes."""
    lo, hi = view(pts).min(0), view(pts).max(0)
    pad = 0.02 * (hi - lo).max()
    lo, hi = lo - pad, hi + pad
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo, zoom=zoom)
    ax.view_init(elev=elev, azim=azim)
    ax.set_proj_type("ortho")   # no perspective: anatomical views, nothing pushed off-centre
    ax.set_axis_off()
    ax.set_title(title, pad=-4)


def render_view(draw, pts, elev, azim, width_in=6.0, dpi=300):
    """Render one 3D view on its own (draw(ax) adds the artists) and return it cropped to its
    visible pixels: mplot3d otherwise clips content at the panel edges or leaves uneven margins."""
    import io
    lo, hi = view(pts).min(0), view(pts).max(0)
    fig = plt.figure(figsize=(width_in, width_in))
    ax = fig.add_axes([0, 0, 1, 1], projection="3d")
    draw(ax)
    setup_3d(ax, pts, "", elev, azim, zoom=1.0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, transparent=True)
    plt.close(fig)
    buf.seek(0)
    img = plt.imread(buf)
    alpha = img[..., 3] > 0.02
    rows, cols = np.nonzero(alpha.any(1))[0], np.nonzero(alpha.any(0))[0]
    m = 6
    return img[max(rows[0] - m, 0):rows[-1] + m, max(cols[0] - m, 0):cols[-1] + m]


def image_grid(images, titles, width_in=7.2, title_h=0.22, gap=0.12, bottom=0.0):
    """Figure with images (rows of equal-width columns) at their own aspect; returns (fig, top-of-bottom-area)."""
    ncol = len(images[0])
    col_w = (width_in - gap * (ncol - 1)) / ncol
    row_h = [max(col_w * im.shape[0] / im.shape[1] for im in row) for row in images]
    height = sum(row_h) + title_h * len(images) + bottom
    fig = plt.figure(figsize=(width_in, height))
    y = height
    for row, ttl, h in zip(images, titles, row_h):
        y -= title_h + h
        for j, (im, t) in enumerate(zip(row, ttl)):
            x = j * (col_w + gap)
            ih = col_w * im.shape[0] / im.shape[1]
            ax = fig.add_axes([x / width_in, (y + (h - ih) / 2) / height, col_w / width_in, ih / height])
            ax.imshow(im, interpolation="lanczos")
            ax.set_axis_off()
            ax.set_title(t, pad=3)
    return fig, bottom / height


def parts(r):
    """[(split_dir, results_dir)]: every finished fold of a leave-N-out run, or the run itself."""
    folds = sorted(glob.glob(os.path.join(r["results"], "fold_*")))
    if folds:
        return [(os.path.join(r["prepared"], "folds", os.path.basename(f)), f) for f in folds
                if os.path.isfile(os.path.join(f, "csv", "trained_vs_target_arm4.csv"))]
    return [(r["prepared"], r["results"])]


def n_test(split_dir):
    with open(os.path.join(split_dir, "test.txt")) as f:
        return sum(1 for line in f if line.strip())


# ----------------------------------------------------------------------------- figure 1
def fig_data_overview(runs, sample, out, data_root):
    """Rows: runs (frames). Columns: anterior and posterior view (the medial triceps head is deep)."""
    subject, arm = sample.rsplit("_", 1)
    layout = detect_layout(data_root)
    muscles = parse_muscles(DEFAULT_MUSCLES, layout)
    flip = (lambda v: v) if arm == "R" else (lambda v: v * [-1, 1, 1])
    images, titles = [], []
    for label, run in runs.items():
        frame = run["frame"]
        sub = next(x for x in find_subjects(data_root, frame, layout, muscles) if x["subject"] == subject)
        organs = load_muscles(sub, arm, layout, muscles)   # right-arm frame; mirror back for L
        sv, sf = read_ply(sub["skin"])
        skin = xyz(sv)
        mus = np.concatenate([flip(v[np.unique(f)]) for v, f in organs.values()])
        keep = cKDTree(mus).query(skin, distance_upper_bound=0.1)[0] < 0.1
        sfc = sf[keep[sf].all(1)]
        bv, bf = read_ply(os.path.join(data_root, LAYOUTS[layout]["bone"].format(frame=frame)))
        bone = xyz(bv)
        hum = np.zeros(len(bone), bool)
        hum[pick_humerus(bone, bf, mus)] = True

        def draw(ax):
            mesh(ax, bone * 1000, bf[hum[bf].all(1)], BONE)
            for (name, (v, f)), col in zip(organs.items(), MUSCLE_COLORS):
                mesh(ax, flip(v) * 1000, f, col)
            mesh(ax, skin * 1000, sfc, SKIN, alpha=0.22)
        images.append([render_view(draw, skin[keep] * 1000, e, a) for e, a in VIEWS.values()])
        titles.append([f"{label}, {v.lower()}" for v in VIEWS])
    fig, b = image_grid(images, titles, bottom=0.45)
    handles = [matplotlib.patches.Patch(color=c, label=NAMES[n]) for n, c in zip([m[0] for m in muscles], MUSCLE_COLORS)]
    handles += [matplotlib.patches.Patch(color=BONE, label="Humerus"),
                matplotlib.patches.Patch(color=SKIN, label="Skin (input crop)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
    save(fig, out, "fig1_data_overview")


def fig_full_body(runs, sample, out, data_root):
    """Supplementary: the whole simulated body per frame (skin, skeleton, the four target muscles on
    both arms), anterior view, with the skin displacement between the first two frames."""
    subject = sample.rsplit("_", 1)[0]
    layout = detect_layout(data_root)
    muscles = parse_muscles(DEFAULT_MUSCLES, layout)
    images, titles, skins = [], [], []
    for label, run in runs.items():
        frame = run["frame"]
        sub = next(x for x in find_subjects(data_root, frame, layout, muscles) if x["subject"] == subject)
        sv, sf = read_ply(sub["skin"])
        skin = xyz(sv)
        skins.append(skin)
        bv, bf = read_ply(os.path.join(data_root, LAYOUTS[layout]["bone"].format(frame=frame)))
        organs = {a: load_muscles(sub, a, layout, muscles) for a in ("R", "L")}

        def draw(ax, skin=skin, sf=sf, bone=xyz(bv), bf=bf, organs=organs):
            mesh(ax, bone * 1000, bf, BONE)
            for a, flip in (("R", 1), ("L", -1)):
                for (name, (v, f)), col in zip(organs[a].items(), MUSCLE_COLORS):
                    mesh(ax, v * [flip, 1, 1] * 1000, f, col)
            mesh(ax, skin * 1000, sf, SKIN, alpha=0.12)
        images.append(render_view(draw, skin * 1000, 4, 90, width_in=5.0, dpi=250))
        titles.append(f"{label}, anterior")
    fig, b = image_grid([images], [titles], bottom=0.55)
    note = ""
    if len(skins) == 2:
        d = np.linalg.norm(skins[0] - skins[1], axis=1) * 1000
        arm = (skins[0][:, 1] > 1.4) & (skins[0][:, 1] < 1.6)
        note = (f"Skin displacement between frames: whole body median {np.median(d):.1f} mm (max {d.max():.0f} mm, "
                f"hands/wrists); upper-arm region median {np.median(d[arm]):.1f} mm")
    handles = [matplotlib.patches.Patch(color=c, label=NAMES[n]) for n, c in zip([m[0] for m in muscles], MUSCLE_COLORS)]
    handles += [matplotlib.patches.Patch(color=BONE, label="Skeleton"), matplotlib.patches.Patch(color=SKIN, label="Skin")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, b * 0.18))
    if note:
        fig.text(0.5, b * 0.05, note, ha="center", fontsize=6.5, color=INK2)
    save(fig, out, "figS1_full_body")


# ----------------------------------------------------------------------------- figure 2
def fig_training_curves(runs, out):
    """Per run: losses (train and validation, else test) and held-out CD of model vs template.
    Leave-N-out: median over folds (shaded: interquartile range), for epochs that at least half of
    the folds reached; the dotted line is the median selected (best-validation) epoch."""
    hist = {}
    for label, r in runs.items():
        hs = []
        for split_dir, res in parts(r):
            f = os.path.join(res, "csv", "loss_history.csv")
            if os.path.isfile(f):
                hs.append(pd.read_csv(f))
        if hs:
            hist[label] = hs
    if not hist:
        return
    fig, axes = plt.subplots(2, len(hist), figsize=(3.4 * len(hist), 4.4), squeeze=False, sharex=True, sharey="row")
    for j, (label, hs) in enumerate(hist.items()):
        held = "val" if "val_loss" in hs[0] else "test"
        allh = pd.concat([h.assign(fold=k) for k, h in enumerate(hs)])
        counts = allh.groupby("epoch").fold.nunique()
        allh = allh[allh.epoch.isin(counts[counts >= max(1, len(hs) / 2)].index)]
        g = allh.groupby("epoch")
        med, q1, q3 = g.median(numeric_only=True), g.quantile(0.25, numeric_only=True), g.quantile(0.75, numeric_only=True)

        def band(ax, col, color, label, scale=1.0, ls="-"):
            if col not in med:
                return
            ax.plot(med.index, med[col] * scale, color=color, label=label, linestyle=ls)
            if len(hs) > 1:
                ax.fill_between(med.index, q1[col] * scale, q3[col] * scale, color=color, alpha=0.18, linewidth=0)
        ax = axes[0, j]
        band(ax, "train_loss", MODEL, "Train", 1e4)
        band(ax, f"{held}_loss", TEMPLATE, "Validation" if held == "val" else "Test", 1e4)
        ax.set_title(label + (f" ({len(hs)} folds)" if len(hs) > 1 else ""))
        ax.set_ylabel("Chamfer loss (×10$^{-4}$)" if j == 0 else "")
        ax.legend(frameon=False)
        ax = axes[1, j]
        band(ax, f"{held}_cd_mm", MODEL, "Model")
        band(ax, "val_template_cd_mm" if held == "val" else "template_cd_mm", TEMPLATE, "Template", ls="--")
        best = []
        for split_dir, res in parts(runs[label]):
            f = os.path.join(res, "csv", "best_epoch.txt")
            if os.path.isfile(f):
                best.append(int(open(f).read().split()[0]))
        if best:
            for a in axes[:, j]:
                a.axvline(np.median(best), color=INK2, linestyle=":", linewidth=0.9)
        ax.legend(frameon=False)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(("Validation" if held == "val" else "Test") + " CD (mm)" if j == 0 else "")
    fig.align_ylabels(axes[:, 0])
    fig.tight_layout()
    save(fig, out, "fig2_training_curves")


# ----------------------------------------------------------------------------- figure 3 + table
def volumes(r):
    """Volume rows of every held-out arm (all finished folds for leave-N-out)."""
    rows = []
    for split_dir, res in parts(r):
        rows += [dict(x, fold=os.path.basename(res)) for x in volume_rows(r["prepared"], split_dir, res)]
    return pd.DataFrame(rows) if rows else None


def fig_volume_scatter(runs, vols, out):
    have = {k: v for k, v in vols.items() if v is not None}
    if not have:
        return
    organs = list(NAMES)
    fig, axes = plt.subplots(len(have), 4, figsize=(7.2, 1.95 * len(have) + 0.45), squeeze=False)
    for i, (label, v) in enumerate(have.items()):
        for j, o in enumerate(organs):
            ax = axes[i, j]
            d = v[v.organ == o]
            lo = min(d.true_ml.min(), d.pred_ml.min()) * 0.92
            hi = max(d.true_ml.max(), d.pred_ml.max()) * 1.05
            ax.plot([lo, hi], [lo, hi], color=INK2, linewidth=0.8, linestyle=":", zorder=1)
            for pred, col, mk in (("template", TEMPLATE, "s"), ("model", MODEL, "o")):
                x = d[d.pred == pred]
                ax.scatter(x.true_ml, x.pred_ml, s=16, color=col, marker=mk, edgecolor="white", linewidth=0.6,
                           zorder=3, label="Model" if pred == "model" else "Template")
            m = d[d.pred == "model"]
            r = np.corrcoef(m.true_ml, m.pred_ml)[0, 1] if m.pred_ml.std() > 0 else np.nan
            tm = d[d.pred == "template"]
            ax.text(0.04, 0.95, f"r = {r:.2f}\nerror {m.abs_err_pct.mean():.1f}%\n(template {tm.abs_err_pct.mean():.1f}%)",
                    transform=ax.transAxes, va="top", fontsize=6.5, color=INK2)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal")
            if i == 0:
                ax.set_title(NAMES[o])
            if i == len(have) - 1:
                ax.set_xlabel("True volume (ml)")
            if j == 0:
                ax.set_ylabel(f"{label}\nPredicted (ml)")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h[::-1], lab[::-1], loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, out, "fig3_volume_scatter")


def table_metrics(runs, vols, out):
    rows = []
    for label, r in runs.items():
        ms, ts, ws = [], [], []
        for split_dir, res in parts(r):
            f_m = os.path.join(res, "csv", "trained_vs_target_arm4.csv")
            f_t = os.path.join(res, "csv", "init_vs_target_arm4.csv")
            if os.path.isfile(f_m) and os.path.isfile(f_t):
                ms.append(pd.read_csv(f_m, index_col=0))
                ts.append(pd.read_csv(f_t, index_col=0))
                ws.append(n_test(split_dir))
        if not ms:
            continue
        w = np.array(ws, float) / sum(ws)   # fold means weighted by test arms = mean over all held-out arms
        m = sum(x * wi for x, wi in zip(ms, w))
        t = sum(x * wi for x, wi in zip(ts, w))
        v = vols.get(label)
        for o in NAMES:
            row = {"run": label, "muscle": NAMES[o], "folds": len(ms), "test arms": sum(ws),
                   "CD model (mm)": m.loc[o, "cd"], "CD template (mm)": t.loc[o, "cd"],
                   "HD95 model (mm)": m.loc[o, "hd95"], "HD95 template (mm)": t.loc[o, "hd95"]}
            if v is not None:
                vo = v[v.organ == o]
                row["Vol. err model (%)"] = vo[vo.pred == "model"].abs_err_pct.mean()
                row["Vol. err template (%)"] = vo[vo.pred == "template"].abs_err_pct.mean()
                mm = vo[vo.pred == "model"]
                row["r (vol.)"] = np.corrcoef(mm.true_ml, mm.pred_ml)[0, 1] if mm.pred_ml.std() > 0 else np.nan
            rows.append(row)
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.round(3).to_csv(os.path.join(out, "table_metrics.csv"), index=False)
    with open(os.path.join(out, "table_metrics.tex"), "w") as f:
        f.write(df.to_latex(index=False, float_format="%.2f"))
    print(df.round(2).to_string(index=False))
    print(f"wrote {out}/table_metrics.csv/.tex")


# ----------------------------------------------------------------------------- figure 4
def taubin_smooth(verts, faces, iters=30, lam=0.5, mu=-0.53):
    """Taubin smoothing (alternating shrink / inflate Laplacian steps): removes the voxel staircase
    of a marching-cubes mesh without shrinking it. Display only."""
    from scipy.sparse import coo_matrix
    n = len(verts)
    i = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2], faces[:, 1], faces[:, 2], faces[:, 0]])
    j = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0], faces[:, 0], faces[:, 1], faces[:, 2]])
    adj = coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n)).tocsr()
    adj.data[:] = 1.0
    deg = np.maximum(np.asarray(adj.sum(1)).ravel(), 1)[:, None]
    v = verts.astype(np.float64).copy()
    for _ in range(iters):
        for f in (lam, mu):
            v = v + f * (adj @ v / deg - v)
    return v


def mesh_values(ax, verts, faces, values, cmap, vmin, vmax, light=(-0.4, -0.8, 0.6), shade=0.25):
    """Triangle mesh coloured per face by the mean of its vertex values, with light shading only
    (shade = how much lighting may darken a face) so the colours stay readable against the colour bar."""
    tri = view(verts)[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    lit = np.abs(n @ (np.array(light) / np.linalg.norm(light)))
    rgba = cmap(matplotlib.colors.Normalize(vmin, vmax)(values[faces].mean(1)))
    rgba[:, :3] *= (1 - shade) + shade * lit[:, None]
    ax.add_collection3d(Poly3DCollection(tri, facecolors=rgba, edgecolors="none"))


def fig_error_surface(runs, sample, out, data_root, vmax=6.0, cmap_name="turbo"):
    """Per run: rows template / model, columns anterior / posterior. Each row is a closed surface:
    the template mesh, and the same mesh moved by the model's prediction (point i of the output is
    template point i moved; each mesh vertex follows its nearest template points, as in
    volume_eval.py). Colour = distance (mm) of the surface to the true muscle surface."""
    have = {}
    for k, r in runs.items():
        for split_dir, res in parts(r):
            f = os.path.join(res, "predictions.npz")
            if os.path.isfile(f) and sample in np.load(f).files:
                have[k] = dict(r, split=split_dir, pred_dir=res)
    if not have:
        print(f"no predictions for {sample}; skipping fig4")
        return
    cmap = matplotlib.colormaps[cmap_name]
    subject, arm = sample.rsplit("_", 1)
    layout = detect_layout(data_root)
    muscles = parse_muscles(DEFAULT_MUSCLES, layout)
    images, titles = [], []
    for label, r in have.items():
        sub = next(x for x in find_subjects(data_root, r["frame"], layout, muscles) if x["subject"] == subject)
        organs = load_muscles(sub, arm, layout, muscles)   # right-arm frame, metres
        rng = np.random.default_rng(0)
        true = {li: cKDTree(sample_surface(v * 1000, f, 300000, rng)) for li, (v, f) in enumerate(organs.values(), start=1)}
        aff = np.load(os.path.join(r["prepared"], "affines.npz"))[sample]
        mean = np.load(os.path.join(r["split"], "mean_shape.npz"))
        if "mesh_verts" not in mean.files:
            raise SystemExit(f"{r['split']}/mean_shape.npz has no template mesh - rerun prepare_cactus.py")
        pred = np.load(os.path.join(r["pred_dir"], "predictions.npz"))[sample]
        disp = pred - mean["mean_pc_np"]
        faces, vlab = mean["mesh_faces"], mean["mesh_label"]
        moved = (disp[mean["mesh_nn_idx"]] * mean["mesh_nn_w"][..., None]).sum(1)
        base = taubin_smooth(mean["mesh_verts"], faces)   # remove the voxel staircase (display only)
        for name, verts in (("template", base), ("model", base + moved)):
            w = to_world_mm(verts, aff)
            if arm == "L":
                w = w * [-1, 1, 1]   # into the right-arm frame of load_muscles
            err = np.zeros(len(w))
            for li, tree in true.items():
                err[vlab == li] = tree.query(w[vlab == li])[0]

            def draw(ax, w=w, err=err):
                mesh_values(ax, w, faces, err, cmap, 0, vmax)
            images.append([render_view(draw, w, el, az) for el, az in VIEWS.values()])
            titles.append([f"{label}, {name}, {v.lower()} (mean {err.mean():.2f} mm)" for v in VIEWS])
    fig, b = image_grid(images, titles, bottom=0.55)
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(0, vmax))
    cax = fig.add_axes([0.25, b * 0.62, 0.5, b * 0.12])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="max")
    cb.set_label("Distance to true muscle surface (mm)")
    cb.outline.set_visible(False)
    save(fig, out, "fig4_error_surface")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help='"label=prepared_dir:results_dir[:frame]" (repeatable)')
    ap.add_argument("--sample", default="BB_0_TRL_-0.6_TRM_0_TRLN_0.6_T_1_R", help="test arm for figs 1 and 4")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--out", default="figures/paper")
    ap.add_argument("--only", nargs="*", help="subset of: fullbody overview curves volumes error table")
    ap.add_argument("--error_cmap", default="turbo", help="colour map for fig4 (turbo = improved jet; or jet, viridis, ...)")
    ap.add_argument("--error_max_mm", type=float, default=6.0, help="top of the fig4 colour scale (mm)")
    args = ap.parse_args()

    runs = {}
    for spec in args.run:
        label, paths = spec.split("=", 1)
        parts = paths.split(":")
        frame = int(parts[2]) if len(parts) > 2 else LAYOUTS[detect_layout(args.data)]["frame"]
        runs[label] = {"prepared": parts[0], "results": parts[1], "frame": frame}
    os.makedirs(args.out, exist_ok=True)
    want = set(args.only or ["fullbody", "overview", "curves", "volumes", "error", "table"])
    vols = {k: volumes(r) for k, r in runs.items()} if want & {"volumes", "table"} else {}
    if "fullbody" in want:
        fig_full_body(runs, args.sample, args.out, args.data)
    if "overview" in want:
        fig_data_overview(runs, args.sample, args.out, args.data)
    if "curves" in want:
        fig_training_curves(runs, args.out)
    if "volumes" in want:
        fig_volume_scatter(runs, vols, args.out)
    if "error" in want:
        fig_error_surface(runs, args.sample, args.out, args.data, vmax=args.error_max_mm, cmap_name=args.error_cmap)
    if "table" in want:
        table_metrics(runs, vols, args.out)


if __name__ == "__main__":
    main()
