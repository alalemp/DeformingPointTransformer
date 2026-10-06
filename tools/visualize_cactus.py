"""Interactive 3D view (HTML) of a Houdini cactus-pose subject: skin, target muscles, humerus,
and optionally the prepared point clouds, a model's prediction and its template.

Meshes are read from the raw PLY export and shown in world mm. Point overlays come from a
prepare_cactus.py output (--prepared) and a main.py / run_folds.py result (--results); left-arm
samples (<subject>_L) are mirrored back onto the left arm. Every layer can be toggled in the legend.

Works with both export layouts (see tools/prepare_cactus.py); the layout is detected from --data.

Usage:
  # one subject, both arms: arm skin + the four target muscles + humerus (frame 160 by default)
  python tools/visualize_cactus.py --subject BB_0_TRL_0.6_TRM_-0.6_TRLN_0_T_1 --out vis/cactus_s1
  # one arm sample with its prepared points
  python tools/visualize_cactus.py --id BB_0_TRL_0.6_TRM_-0.6_TRLN_0_T_1_R --prepared data/cactus160 \\
      --out vis/cactus_s1_R
  # plus the model prediction (single split or leave-N-out: the fold holding the sample is found)
  python tools/visualize_cactus.py --id BB_0_TRL_0.6_TRM_-0.6_TRLN_0_T_1_R --prepared data/cactus160_loo \\
      --results results/cactus160_loo_res --out vis/cactus_s1_R_pred
  # the first export
  python tools/visualize_cactus.py --data /data/aalempij/data/OLD_datasets_cactus_poses --frame 45 \\
      --subject BB_0.6_TRL_-0.6_TRM_0.6_TRLN_-0.6_T_1 --out vis/old_s1
"""
import argparse
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_cactus import (DATA, DEFAULT_MUSCLES, DEFAULT_PIECES, LAYOUTS, detect_layout,  # noqa: E402
                            find_subjects, load_muscles, mirror, parse_muscles, pick_humerus, read_ply, xyz)
from visualize_seg import mesh_trace, palette, prediction_traces, prepared_traces, write_html  # noqa: E402


def submesh(verts, faces, keep):
    """Faces with all vertices kept, re-indexed onto the kept vertices."""
    f = faces[keep[faces].all(1)]
    used = np.unique(f)
    remap = np.full(len(verts), -1)
    remap[used] = np.arange(len(used))
    return verts[used], remap[f]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--subject", help="subject folder name (shows both arms)")
    ap.add_argument("--id", help="arm sample id <subject>_R / <subject>_L (shows that arm)")
    ap.add_argument("--frame", type=int, help="default 160 (separated layout) / 45 (entire layout)")
    ap.add_argument("--muscles", default=DEFAULT_MUSCLES, help="separated layout: name:folder,...")
    ap.add_argument("--pieces", default=DEFAULT_PIECES, help="entire layout: name:right_piece:left_piece,...")
    ap.add_argument("--all_muscles", action="store_true", help="also show all other muscles (grey, entire_muscles mesh)")
    ap.add_argument("--skin", choices=["arm", "full", "none"], default="arm",
                    help="arm: skin within --skin_mm of the target muscles (keeps the file small)")
    ap.add_argument("--skin_mm", type=float, default=150)
    ap.add_argument("--no_bone", action="store_true", help="do not show the humerus")
    ap.add_argument("--bone_file", help="default: the layout's shared skeleton")
    ap.add_argument("--prepared", help="prepare_cactus.py output dir, to overlay the sampled points")
    ap.add_argument("--results", help="main.py --save_path or run_folds.py --save dir, to overlay predictions")
    ap.add_argument("--to_mm", type=float, default=1000)
    ap.add_argument("--out", required=True, help="output prefix (writes <out>.html)")
    args = ap.parse_args()

    if not (args.subject or args.id):
        ap.error("give --subject or --id")
    layout = detect_layout(args.data)
    frame = args.frame if args.frame is not None else LAYOUTS[layout]["frame"]
    muscles = parse_muscles(args.muscles if layout == "separated" else args.pieces, layout)
    names = [m[0] for m in muscles]
    subject, arm = (args.id.rsplit("_", 1) if args.id else (args.subject, None))
    subs = {x["subject"]: x for x in find_subjects(args.data, frame, layout, muscles)}
    if subject not in subs:
        raise SystemExit(f"{subject}: no skin + muscles for frame {frame} under {args.data}")
    sub = subs[subject]
    arms = [arm] if arm else ["R", "L"]
    colors = dict(zip(names, palette(len(names))))
    to_world = lambda v, a: (v if a == "R" else mirror(v)) * args.to_mm  # loaders give the right-arm frame

    sv, sf = read_ply(sub["skin"])
    skin = xyz(sv) * args.to_mm
    organs = {a: load_muscles(sub, a, layout, muscles) for a in arms}
    traces = []
    target_pts = np.concatenate([to_world(v[np.unique(f)], a) for a in arms for v, f in organs[a].values()])
    if args.skin != "none":
        keep = (cKDTree(target_pts).query(skin, distance_upper_bound=args.skin_mm)[0] < args.skin_mm
                if args.skin == "arm" else np.ones(len(skin), bool))
        traces.append(mesh_trace(*submesh(skin, sf, keep), "skin", "#c8a080", 0.2))
    for a in arms:
        for name, (v, f) in organs[a].items():
            keep = np.zeros(len(v), bool)
            keep[np.unique(f)] = True
            traces.append(mesh_trace(*submesh(to_world(v, a), f, keep), f"{name} ({a})", colors[name], 1.0))
    if args.all_muscles:
        path = (os.path.join(args.data, "entire muscles", sub["group"], f"entire_muscles.{frame:05d}.ply")
                if layout == "separated" else sub["muscles"])
        if os.path.isfile(path):
            mv, mf = read_ply(path)
            traces.append(mesh_trace(xyz(mv) * args.to_mm, mf, "all muscles", "#999999", 0.25))
        else:
            print(f"no entire_muscles mesh at {path}")
    if not args.no_bone:
        bone_file = args.bone_file or LAYOUTS[layout]["bone"]
        bv, bf = read_ply(os.path.join(args.data, bone_file.format(subject=subject, frame=frame)))
        bone = xyz(bv) * args.to_mm
        for a in arms:
            # pick_humerus works in the export's units (metres)
            ref = np.concatenate([to_world(v[np.unique(f)], a) for v, f in organs[a].values()]) / args.to_mm
            keep = np.zeros(len(bone), bool)
            keep[pick_humerus(bone / args.to_mm, bf, ref)] = True
            traces.append(mesh_trace(*submesh(bone, bf, keep), f"humerus ({a})", "#f0f0e0", 1.0))

    if args.id and args.prepared:
        unmirror = (lambda w: mirror(w)) if arm == "L" else (lambda w: w)
        traces += prepared_traces(args.prepared, args.id, colors, world=unmirror)
        if args.results:
            traces += prediction_traces(args.prepared, args.results, args.id, colors, world=unmirror)
    elif args.prepared or args.results:
        print("--prepared / --results need --id (one arm sample); showing meshes only")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_html(traces, args.out + ".html", args.id or subject)
    print(f"wrote {args.out}.html ({os.path.getsize(args.out + '.html') / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
