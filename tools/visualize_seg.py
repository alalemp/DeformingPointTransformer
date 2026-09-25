"""Visualise an MRI + segmentation: body surface vs. internal organs.

Two inputs are supported:
  * a single multi-label file (VIBESegmentator):    --img mri.nii.gz --seg seg.nii.gz
  * a TotalSegmentator MRI subject by id:            --id s0852 [--data <dataset root>]
    (loads <subject>/mri.nii.gz and every mask in <subject>/segmentations/)

Produces
  <out>.html  interactive 3D view (translucent body surface + organ meshes, world mm)
  <out>.png   axial / coronal / sagittal slices with body contour and organ overlay
and prints, per organ, how many voxels lie outside the body and the minimum
distance from the organ to the skin.

The body mask is taken from the MRI intensity (skin boundary), independent of
the segmentation, so it is a real check that the organs sit inside the surface.

With --prepared <dir> (output of tools/prepare_totalseg_mri.py) the sampled
body-surface and organ point clouds for that id are added to the 3D view, mapped
back to world mm, so they can be checked against the meshes.

Volumes are reoriented to RAS (closest canonical) so axis 2 is always axial.

Usage:
  python tools/visualize_seg.py --img mri3.nii.gz --seg seg3.nii.gz --out vis/scan3
  python tools/visualize_seg.py --id s0852 --out vis/s0852
  python tools/visualize_seg.py --id s0852 --organs liver spleen kidney_left kidney_right pancreas \
      --prepared data/tsmri --out vis/s0852
"""
import glob
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage
from skimage import filters, measure

# VIBESegmentator label ids
DEFAULT_LABELS = {1: "spleen", 2: "kidney_right", 3: "kidney_left", 5: "liver",
                  6: "stomach", 7: "pancreas", 24: "heart", 21: "urinary_bladder"}
COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
          "#f032e6", "#bfef45", "#9A6324", "#800000"]
DEFAULT_DATA = "/home/aalempij/Data/data/TotalsegmentatorMRI_dataset_v300"



def body_mask_from_mri(img, frac=0.1, close_iter=3):
    """Body mask of an RAS volume (axis 2 axial).

    The threshold is set per axial slice (frac x the slice's 99th percentile, median
    smoothed along z) because stitched multi-station scans differ in brightness per
    station. Per slice: close small skin gaps, fill holes (lungs, bowel gas), drop
    specks. Finally drop 3D components smaller than 1% of the body.
    """
    nz = img.shape[2]
    p99 = np.zeros(nz, np.float32)
    for z in range(nz):
        s = img[:, :, z]
        if (s > 0).any():
            p99[z] = np.percentile(s[s > 0], 99)
    t = ndimage.median_filter(p99, size=9, mode="nearest") * frac
    m = img > t[None, None, :]
    st = ndimage.generate_binary_structure(2, 1)
    for z in range(nz):
        s = ndimage.binary_opening(m[:, :, z], st, iterations=1)
        # pad before closing so the body is not glued to the array border
        s = ndimage.binary_closing(np.pad(s, close_iter), st, iterations=close_iter)[close_iter:-close_iter,
                                                                                    close_iter:-close_iter]
        s = ndimage.binary_fill_holes(s)
        lab, n = ndimage.label(s)
        if n > 1:
            sizes = ndimage.sum(s, lab, range(1, n + 1))
            s = np.isin(lab, 1 + np.nonzero(sizes >= 0.02 * sizes.max())[0])
        m[:, :, z] = s
    # bridge single-slice dropouts (station boundaries) along z, then refill
    m = ndimage.binary_closing(np.pad(m, ((0, 0), (0, 0), (2, 2)), mode="edge"),
                               np.ones((1, 1, 5), bool))[:, :, 2:-2]
    for z in range(nz):
        m[:, :, z] = ndimage.binary_fill_holes(m[:, :, z])
    lab, n = ndimage.label(m)
    if n > 1:
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        m = np.isin(lab, 1 + np.nonzero(sizes >= 0.01 * sizes.sum())[0])
    return m


def mesh_world(mask, affine, step=1):
    # pad so surfaces are closed at the volume border
    p = np.pad(mask, 1).astype(np.float32)
    verts, faces, _, _ = measure.marching_cubes(p, 0.5, step_size=step)
    verts -= 1
    verts = verts @ affine[:3, :3].T + affine[:3, 3]
    return verts, faces


def mesh_trace(verts, faces, name, color, opacity):
    return {"type": "mesh3d", "name": name, "showlegend": True,
            "x": np.round(verts[:, 0], 1).tolist(), "y": np.round(verts[:, 1], 1).tolist(),
            "z": np.round(verts[:, 2], 1).tolist(),
            "i": faces[:, 0].tolist(), "j": faces[:, 1].tolist(), "k": faces[:, 2].tolist(),
            "color": color, "opacity": opacity, "flatshading": False}


def write_html(traces, path, title):
    layout = {"title": title, "scene": {"aspectmode": "data",
              "xaxis": {"title": "x (mm)"}, "yaxis": {"title": "y (mm)"}, "zaxis": {"title": "z (mm)"}},
              "legend": {"itemclick": "toggle"}, "margin": {"l": 0, "r": 0, "t": 40, "b": 0}}
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>{title}</title>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<style>html,body{{margin:0;height:100%;background:#fff}}#p{{width:100%;height:100vh}}</style></head>
<body><div id="p"></div><script>
Plotly.newPlot("p", {json.dumps(traces)}, {json.dumps(layout)}, {{responsive:true}});
</script></body></html>"""
    with open(path, "w") as f:
        f.write(html)


def write_png(img, body, seg, labels, path, zooms):
    cx, cy, cz = [int(np.mean(np.where(np.isin(seg, list(labels)))[a])) if np.isin(seg, list(labels)).any()
                  else s // 2 for a, s in enumerate(seg.shape)]
    # arrays are RAS: index 1 grows anterior, index 2 superior (both up with origin="lower")
    views = [("axial", img[:, :, cz], body[:, :, cz], seg[:, :, cz], zooms[1] / zooms[0]),
             ("coronal", img[:, cy, :], body[:, cy, :], seg[:, cy, :], zooms[2] / zooms[0]),
             ("sagittal", img[cx, :, :], body[cx, :, :], seg[cx, :, :], zooms[2] / zooms[1])]
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    colors = palette(len(labels))
    cmap = matplotlib.colors.ListedColormap(colors)
    for ax, (name, im, b, s, aspect) in zip(axes, views):
        ax.imshow(im.T, cmap="gray", origin="lower", aspect=aspect)
        ov = np.full(s.shape, np.nan)
        for n, l in enumerate(labels):
            ov[s == l] = n
        ax.imshow(ov.T, cmap=cmap, vmin=0, vmax=len(colors) - 1, alpha=0.5, origin="lower",
                  aspect=aspect, interpolation="nearest")
        ax.contour(b.T.astype(float), levels=[0.5], colors="yellow", linewidths=1, origin="lower")
        ax.set_title(name)
        ax.axis("off")
    handles = [matplotlib.patches.Patch(color=colors[n], label=labels[l]) for n, l in enumerate(labels)]
    handles.append(matplotlib.lines.Line2D([], [], color="yellow", label="body surface (MRI)"))
    ncol = min(len(handles), 8)
    fig.legend(handles=handles, loc="lower center", ncol=ncol, fontsize=8)
    fig.tight_layout(rect=(0, 0.03 * (1 + (len(handles) - 1) // ncol), 1, 1))
    fig.savefig(path, dpi=100)
    plt.close(fig)


def palette(n):
    if n <= len(COLORS):
        return COLORS[:n]
    cm = plt.get_cmap("tab20")
    return [matplotlib.colors.to_hex(cm(i % 20)) for i in range(n)]


def load_multilabel(img_path, seg_path, wanted):
    img_nii = nib.as_closest_canonical(nib.load(img_path))
    seg_nii = nib.as_closest_canonical(nib.load(seg_path))
    assert img_nii.shape == seg_nii.shape and np.allclose(img_nii.affine, seg_nii.affine, atol=1e-3), \
        "img/seg grids differ"
    seg = np.asarray(seg_nii.dataobj).astype(np.int16)
    labels = {l: DEFAULT_LABELS.get(l, f"label_{l}") for l in wanted} if wanted else DEFAULT_LABELS
    return img_nii, seg, labels, os.path.basename(seg_path)


def load_totalseg(data, sid, organs):
    dirs = glob.glob(os.path.join(data, sid)) + glob.glob(os.path.join(data, "*", sid))
    if not dirs:
        raise SystemExit(f"subject {sid} not found under {data}")
    d = dirs[0]
    img_nii = nib.as_closest_canonical(nib.load(os.path.join(d, "mri.nii.gz")))
    names = organs or sorted(os.path.basename(p)[:-7] for p in glob.glob(os.path.join(d, "segmentations", "*.nii.gz")))
    seg = np.zeros(img_nii.shape[:3], np.int16)
    labels = {}
    for l, name in enumerate(names, start=1):
        m_nii = nib.as_closest_canonical(nib.load(os.path.join(d, "segmentations", f"{name}.nii.gz")))
        assert np.allclose(m_nii.affine, img_nii.affine, atol=1e-3), f"{name}: grid differs from mri"
        seg[np.asarray(m_nii.dataobj) > 0] = l
        labels[l] = name
    return img_nii, seg, labels, sid


def prepared_traces(prep_dir, sid, organs, colors):
    """Point clouds written by prepare_totalseg_mri.py, mapped back to world mm."""
    data, affines = np.load(os.path.join(prep_dir, "data.npz")), np.load(os.path.join(prep_dir, "affines.npz"))
    if sid not in affines:
        print(f"{sid} is not in {prep_dir} (not selected?) - no prepared points shown")
        return []
    aff = affines[sid]
    with open(glob.glob(os.path.join(prep_dir, "labels", "*.json"))[0]) as f:
        prep_organs = list(json.load(f).values())
    import prepare_totalseg_mri as prep
    d = np.array(prep.GRID_SHAPE, dtype=np.float32) - 1
    traces = []
    for key, color in [("input_points", "#806040")] + [(o, colors.get(o, "#000000")) for o in prep_organs]:
        p = data[f"{sid}__{key}"]
        v = (p + 1) / 2 * d
        v[:, :2] = d[:2] - v[:, :2]
        w = v @ aff[:3, :3].T + aff[:3, 3]
        traces.append({"type": "scatter3d", "mode": "markers", "name": f"points: {key}", "visible": "legendonly",
                       "x": np.round(w[:, 0], 1).tolist(), "y": np.round(w[:, 1], 1).tolist(),
                       "z": np.round(w[:, 2], 1).tolist(), "marker": {"size": 1.5, "color": color}})
    return traces


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--img", help="MRI (multi-label mode)")
    ap.add_argument("--seg", help="multi-label segmentation (multi-label mode)")
    ap.add_argument("--id", help="TotalSegmentator subject id, e.g. s0852")
    ap.add_argument("--data", default=DEFAULT_DATA, help="TotalSegmentator MRI dataset root (with --id)")
    ap.add_argument("--organs", nargs="*", help="with --id: organ names to show (default: all masks)")
    ap.add_argument("--prepared", help="with --id: output dir of prepare_totalseg_mri.py to overlay points")
    ap.add_argument("--out", required=True, help="output prefix (without extension)")
    ap.add_argument("--labels", type=int, nargs="*", help="label ids to show (default: main organs)")
    ap.add_argument("--body_step", type=int, default=2, help="marching-cubes step for the body (bigger = smaller file)")
    ap.add_argument("--organ_step", type=int, default=1, help="marching-cubes step for organs")
    args = ap.parse_args()

    if args.id:
        img_nii, seg, labels, title = load_totalseg(args.data, args.id, args.organs)
    elif args.img and args.seg:
        img_nii, seg, labels, title = load_multilabel(args.img, args.seg, args.labels)
    else:
        ap.error("give either --id or --img and --seg")
    img = np.asarray(img_nii.dataobj, dtype=np.float32)
    affine, zooms = img_nii.affine, img_nii.header.get_zooms()[:3]
    labels = {l: n for l, n in labels.items() if (seg == l).any()}

    body = body_mask_from_mri(img)
    seg_union = seg > 0
    print(f"body mask: {body.sum()} voxels; segmentation voxels outside body mask: "
          f"{(seg_union & ~body).sum()} ({100 * (seg_union & ~body).sum() / max(seg_union.sum(), 1):.2f}%)")

    # distance (mm) from each inside voxel to the skin
    dist_to_skin = ndimage.distance_transform_edt(body, sampling=zooms)
    print(f"{'organ':<30}{'voxels':>9}{'outside':>9}{'min skin dist (mm)':>20}{'touches FOV edge':>18}")
    for l, n in labels.items():
        m = seg == l
        outside = (m & ~body).sum()
        nz = [np.nonzero(m.any(tuple(a for a in range(3) if a != ax)))[0] for ax in range(3)]
        cut = any(n[0] == 0 or n[-1] == s - 1 for n, s in zip(nz, m.shape))
        print(f"{n:<30}{m.sum():>9}{outside:>9}{dist_to_skin[m].min():>20.1f}{'YES' if cut else 'no':>18}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    colors = dict(zip(labels.values(), palette(len(labels))))
    v, f = mesh_world(body, affine, step=args.body_step)
    traces = [mesh_trace(v, f, "body surface", "#c8a080", 0.15)]
    for l, name in labels.items():
        v, f = mesh_world(seg == l, affine, step=args.organ_step)
        traces.append(mesh_trace(v, f, name, colors[name], 1.0))
    if args.prepared:
        traces += prepared_traces(args.prepared, args.id, args.organs, colors)
    write_html(traces, args.out + ".html", title)
    write_png(img, body, seg, labels, args.out + ".png", zooms)
    print(f"wrote {args.out}.html and {args.out}.png")


if __name__ == "__main__":
    main()
