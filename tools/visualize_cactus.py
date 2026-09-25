"""Interactive 3D view (HTML) of a Houdini cactus-pose subject: skin, target muscles, humerus,
and optionally the prepared point clouds, a model's prediction and its template.

Meshes are read from the raw PLY export and shown in world mm. Point overlays come from a
prepare_cactus.py output (--prepared) and a main.py / run_folds.py result (--results); left-arm
samples (<subject>_L) are mirrored back onto the left arm. Every layer can be toggled in the legend.

Usage:
  # one subject, both arms: arm skin + the four target muscles + humerus
  python tools/visualize_cactus.py --subject BB_0.6_TRL_-0.6_TRM_0.6_TRLN_-0.6_T_1 --out vis/cactus_s1
  # one arm sample with its prepared points
  python tools/visualize_cactus.py --id BB_0.6_TRL_-0.6_TRM_0.6_TRLN_-0.6_T_1_R --prepared data/cactus45 \\
      --out vis/cactus_s1_R
  # plus the model prediction (single split or leave-N-out: the fold holding the sample is found)
  python tools/visualize_cactus.py --id BB_0.6_TRL_-0.6_TRM_0.6_TRLN_-0.6_T_1_R --prepared data/cactus45_loo_v2 \\
      --results results/cactus_loo_res --out vis/cactus_s1_R_pred
"""
import argparse
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_cactus import (BONE_FILE, DEFAULT_PIECES, MUSCLE_DIR, SKIN_DIR, find_subjects,  # noqa: E402
                            mirror, pick_humerus, read_ply, xyz)
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
    ap.add_argument("--data", default="/data/aalempij/data/datasets_cactus_poses")
    ap.add_argument("--subject", help="subject folder name (shows both arms)")
    ap.add_argument("--id", help="arm sample id <subject>_R / <subject>_L (shows that arm)")
    ap.add_argument("--frame", type=int, default=45)
    ap.add_argument("--pieces", default=DEFAULT_PIECES, help="name:right_piece:left_piece,...")
    ap.add_argument("--all_muscles", action="store_true", help="also show every other muscle piece (grey)")
    ap.add_argument("--skin", choices=["arm", "full", "none"], default="arm",
                    help="arm: skin within --skin_mm of the target muscles (keeps the file small)")
    ap.add_argument("--skin_mm", type=float, default=150)
    ap.add_argument("--no_bone", action="store_true", help="do not show the humerus")
    ap.add_argument("--bone_file", default=BONE_FILE)
    ap.add_argument("--prepared", help="prepare_cactus.py output dir, to overlay the sampled points")
    ap.add_argument("--results", help="main.py --save_path or run_folds.py --save dir, to overlay predictions")
    ap.add_argument("--to_mm", type=float, default=1000)
    ap.add_argument("--out", required=True, help="output prefix (writes <out>.html)")
    args = ap.parse_args()

    if not (args.subject or args.id):
        ap.error("give --subject or --id")
    subject, arm = (args.id.rsplit("_", 1) if args.id else (args.subject, None))
    subs = {s: (sk, mu) for s, sk, mu in find_subjects(args.data, args.frame)}
    if subject not in subs:
        raise SystemExit(f"{subject}: no skin + muscles for frame {args.frame} under {args.data}")
    pieces = [(n, int(r), int(l)) for n, r, l in (p.split(":") for p in args.pieces.split(","))]
    arms = [arm] if arm else ["R", "L"]

    sv, sf = read_ply(subs[subject][0])
    mv, mf = read_ply(subs[subject][1], props=("x", "y", "z", "piece"))
    skin, mpts, mpiece = xyz(sv) * args.to_mm, xyz(mv) * args.to_mm, np.asarray(mv["piece"])
    colors = dict(zip([n for n, _, _ in pieces], palette(len(pieces))))
    targets = {(n, a): (r if a == "R" else l) for n, r, l in pieces for a in arms}

    traces = []
    target_pts = np.concatenate([mpts[mpiece == p] for p in targets.values()])
    if args.skin != "none":
        keep = (cKDTree(target_pts).query(skin, distance_upper_bound=args.skin_mm)[0] < args.skin_mm
                if args.skin == "arm" else np.ones(len(skin), bool))
        traces.append(mesh_trace(*submesh(skin, sf, keep), "skin", "#c8a080", 0.2))
    for (name, a), piece in targets.items():
        traces.append(mesh_trace(*submesh(mpts, mf, mpiece == piece), f"{name} ({a})", colors[name], 1.0))
    if args.all_muscles:
        others = ~np.isin(mpiece, list(targets.values()))
        traces.append(mesh_trace(*submesh(mpts, mf, others), "other muscles", "#999999", 0.3))
    if not args.no_bone:
        bv, bf = read_ply(os.path.join(args.data, args.bone_file.format(subject=subject, frame=args.frame)))
        bone = xyz(bv) * args.to_mm
        for a in arms:
            # pick_humerus works in the export's units (metres)
            ref = mpts[np.isin(mpiece, [targets[(n, a)] for n, _, _ in pieces])] / args.to_mm
            idx = pick_humerus(bone / args.to_mm, bf, ref)
            keep = np.zeros(len(bone), bool)
            keep[idx] = True
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
