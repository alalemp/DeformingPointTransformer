"""Build DeformingPointTransformer inputs from the TotalSegmentator MRI dataset.

Stages (run all by default, or pick with --stages):
  survey  read every subject's masks, record organ presence / extents / FOV cut  -> survey.csv
  select  keep large-FOV torso scans that contain the whole crop window and all
          five organs uncut                                                      -> selection.csv
          (build then also rejects scans whose torso is cut anterior/posterior by
          the FOV, or whose organs fall outside the body mask            -> qc.csv)
  build   per selected subject: resample to a common grid, body mask from MRI,
          sample 16384 body-surface points and 4096 points per organ; then the
          mean-shape template from the training subjects                        -> npz files

Common grid ("target grid"), identical for every subject up to a translation:
  RAS-aligned, shape GRID_SHAPE = (319, 259, 315) (the size hard-coded in
  util/evaluation_functions.py), in-plane spacing --xy_spacing, and the z axis
  spanning a fixed-height window anchored at the top of the sacrum (S1 endplate):
  [sacrum_top - below_mm, sacrum_top + above_mm]. The sacrum is a skeletal
  landmark, so the crop does not leak the target organs' position. The grid is
  centred left-right / anterior-posterior on the body-mask centroid.

Normalisation follows util/evaluation_functions.convert_to_mm exactly:
  v = voxel index in the target grid, axes 0 and 1 flipped,
  p = 2 * v' / (D - 1) - 1  ->  [-1, 1];  the stored affine maps v to world mm.

Outputs in --out:
  data.npz                    "<id>__input_points" (16384,3) and "<id>__<organ>" (4096,3)
  mean_shape.npz              mean_pc_np (20480,3), mean_label_np (20480,)
  affines.npz                 "<id>" -> 4x4 target-grid voxel-to-world affine
  labels/label_organs_5organs.json
  train.txt / test.txt        --split random (default, --test_frac of kept subjects) or meta (meta.csv)
  survey.csv, selection.csv, qc.csv

Usage:
  python tools/prepare_totalseg_mri.py \
      --data /home/aalempij/Data/data/TotalsegmentatorMRI_dataset_v300 --out data/tsmri
"""
import argparse
import csv
import glob
import json
import os
import sys
from multiprocessing import Pool

import nibabel as nib
import numpy as np
from scipy import ndimage
from skimage import measure

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visualize_seg import body_mask_from_mri  # noqa: E402  (same body mask as the visualiser)

ORGANS = ["liver", "spleen", "kidney_left", "kidney_right", "pancreas"]
LANDMARKS = ["sacrum", "lung_left", "lung_right"]
GRID_SHAPE = (319, 259, 315)
N_BODY, N_ORGAN = 16384, 4096
# organs smaller than this (ml) are treated as missing / failed segmentations
MIN_ORGAN_ML = {"liver": 700, "spleen": 50, "kidney_left": 60, "kidney_right": 60, "pancreas": 20}


# ----------------------------------------------------------------------------- helpers
def find_subjects(root):
    dirs = glob.glob(os.path.join(root, "s[0-9]*")) + glob.glob(os.path.join(root, "*", "s[0-9]*"))
    dirs = [d for d in dirs if os.path.isfile(os.path.join(d, "mri.nii.gz"))]
    return {os.path.basename(d): d for d in sorted(dirs)}


def read_splits(root):
    metas = glob.glob(os.path.join(root, "meta.csv")) + glob.glob(os.path.join(root, "*", "meta.csv"))
    split = {}
    for m in metas:
        with open(m, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f, delimiter=";"):
                split[row["image_id"]] = row["split"]
    return split


def load_canonical(path):
    return nib.as_closest_canonical(nib.load(path))


def world_z_range(mask, affine):
    """min/max world z (mm, superior positive) of a mask's voxels, and whether it touches the array edge."""
    idx = np.nonzero(mask.any((0, 1)))[0]
    z = [(affine @ [0, 0, k, 1])[2] for k in (idx[0], idx[-1])]
    nz = [np.nonzero(mask.any(tuple(a for a in range(3) if a != ax)))[0] for ax in range(3)]
    edge = any(n[0] == 0 or n[-1] == s - 1 for n, s in zip(nz, mask.shape))
    return min(z), max(z), edge


def resample(vol, src_affine, dst_affine, dst_shape, order, cval=0.0):
    """Resample vol (voxel grid src_affine) onto dst grid; float32 output."""
    m = np.linalg.inv(src_affine) @ dst_affine
    return ndimage.affine_transform(vol.astype(np.float32, copy=False), m[:3, :3], m[:3, 3],
                                    output_shape=dst_shape, order=order, cval=cval)


def largest_cc(mask):
    lab, n = ndimage.label(mask)
    if n <= 1:
        return mask
    return lab == (np.argmax(ndimage.sum(mask, lab, range(1, n + 1))) + 1)


def sample_surface(verts, faces, n, rng):
    """Area-weighted uniform sampling of n points on a triangle mesh."""
    tri = verts[faces]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    f = rng.choice(len(faces), size=n, p=area / area.sum())
    u, v = rng.random((2, n))
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    t = tri[f]
    return (t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0])).astype(np.float32)


def normalise(vox):
    """target-grid voxel coords -> [-1, 1], inverse of util/evaluation_functions.convert_to_mm."""
    d = np.array(GRID_SHAPE, dtype=np.float32) - 1
    v = vox.astype(np.float32).copy()
    v[:, 0] = d[0] - v[:, 0]
    v[:, 1] = d[1] - v[:, 1]
    return 2 * v / d - 1


def grid_affine(spacing, corner_world):
    a = np.diag([*spacing, 1.0])
    a[:3, 3] = corner_world
    return a


# ----------------------------------------------------------------------------- survey
def survey_one(item):
    sid, d = item
    img = load_canonical(os.path.join(d, "mri.nii.gz"))
    aff = img.affine
    ext = np.abs(aff[:3, :3]) @ np.array(img.shape[:3])
    zlo, zhi = sorted([(aff @ [0, 0, k, 1])[2] for k in (0, img.shape[2] - 1)])
    row = {"id": sid, "shape": "x".join(map(str, img.shape[:3])),
           "spacing": "x".join(f"{s:.2f}" for s in img.header.get_zooms()[:3]),
           "ext_x": round(ext[0]), "ext_y": round(ext[1]), "ext_z": round(ext[2]),
           "fov_zmin": round(zlo, 1), "fov_zmax": round(zhi, 1)}
    vox_ml = float(np.prod(img.header.get_zooms()[:3])) / 1000
    for o in ORGANS + LANDMARKS:
        m = np.asarray(load_canonical(os.path.join(d, "segmentations", f"{o}.nii.gz")).dataobj) > 0
        row[f"{o}_ml"] = round(m.sum() * vox_ml, 1)
        if m.any():
            a, b, e = world_z_range(m, aff)
            row[f"{o}_zmin"], row[f"{o}_zmax"], row[f"{o}_cut"] = round(a, 1), round(b, 1), int(e)
        else:
            row[f"{o}_zmin"] = row[f"{o}_zmax"] = row[f"{o}_cut"] = ""
    return row


def run_survey(subjects, splits, out, workers):
    with Pool(workers) as p:
        rows = p.map(survey_one, subjects.items(), chunksize=4)
    for r in rows:
        r["split"] = splits.get(r["id"], "")
    write_csv(rows, os.path.join(out, "survey.csv"))
    print(f"survey: {len(rows)} subjects -> {out}/survey.csv")


def write_csv(rows, path):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


# ----------------------------------------------------------------------------- select
def run_select(out, args):
    rows = read_csv(os.path.join(out, "survey.csv"))
    sel = []
    for r in rows:
        reason = ""
        if r["split"] not in ("train", "test"):
            reason = "no split"
        elif r["sacrum_zmax"] == "":
            reason = "no sacrum"
        else:
            top = float(r["sacrum_zmax"])
            w_lo, w_hi = top - args.below_mm, top + args.above_mm
            if float(r["fov_zmin"]) > w_lo or float(r["fov_zmax"]) < w_hi:
                reason = "FOV does not cover window"
            elif int(r["ext_x"]) < args.min_fov_x:
                reason = "FOV too narrow"
            else:
                for o in ORGANS:
                    if r[f"{o}_zmin"] == "" or float(r[f"{o}_ml"]) < MIN_ORGAN_ML[o]:
                        reason = f"{o} missing"
                    elif int(r[f"{o}_cut"]):
                        reason = f"{o} cut by FOV"
                    elif float(r[f"{o}_zmin"]) < w_lo or float(r[f"{o}_zmax"]) > w_hi:
                        reason = f"{o} outside window"
                    if reason:
                        break
        sel.append({"id": r["id"], "split": r["split"], "selected": int(not reason), "reason": reason})
    write_csv(sel, os.path.join(out, "selection.csv"))
    n = [s for s in sel if s["selected"]]
    print(f"select: {len(n)}/{len(sel)} subjects "
          f"(train {sum(s['split'] == 'train' for s in n)}, test {sum(s['split'] == 'test' for s in n)})")
    reasons = {}
    for s in sel:
        if s["reason"]:
            k = s["reason"].split(" ", 1)[1] if s["reason"].split(" ")[0] in ORGANS else s["reason"]
            reasons[k] = reasons.get(k, 0) + 1
    for k, v in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"  rejected, {k}: {v}  (first failing check only)")


# ----------------------------------------------------------------------------- build
def build_one(job):
    sid, d, sacrum_top, args = job
    rng = np.random.default_rng(int(sid[1:]))
    img_nii = load_canonical(os.path.join(d, "mri.nii.gz"))
    img = np.asarray(img_nii.dataobj, dtype=np.float32)
    src = img_nii.affine
    z0 = sacrum_top - args.below_mm
    sz = (args.below_mm + args.above_mm) / (GRID_SHAPE[2] - 1)

    # pass 1: coarse grid over the whole FOV in x/y to find the body centroid
    corners = np.array([[i, j, 0, 1] for i in (0, img.shape[0] - 1) for j in (0, img.shape[1] - 1)]) @ src.T
    cs = 4.0
    c_aff = grid_affine((cs, cs, sz * 4), (corners[:, 0].min(), corners[:, 1].min(), z0))
    c_shape = (int(np.ptp(corners[:, 0]) / cs) + 1, int(np.ptp(corners[:, 1]) / cs) + 1, (GRID_SHAPE[2] - 1) // 4 + 1)
    body_c = body_mask_from_mri(resample(img, src, c_aff, c_shape, order=1))
    cx, cy = (np.array(np.nonzero(body_c)[:2]).mean(1) * cs + c_aff[:2, 3])

    # pass 2: the target grid
    sp = (args.xy_spacing, args.xy_spacing, sz)
    corner = (cx - sp[0] * (GRID_SHAPE[0] - 1) / 2, cy - sp[1] * (GRID_SHAPE[1] - 1) / 2, z0)
    aff = grid_affine(sp, corner)
    body = body_mask_from_mri(resample(img, src, aff, GRID_SHAPE, order=1))
    fov = resample(np.ones(img.shape, np.uint8), src, aff, GRID_SHAPE, order=0) > 0.5
    del img

    # fraction of axial slices where the body touches the scan's FOV border: anterior/posterior
    # (torso cut -> unusable) and left/right (usually just the arms)
    def cut_frac(axis):
        st = np.zeros((3, 3, 3), bool)
        st[tuple(slice(None) if a == axis else 1 for a in range(3))] = True
        border = fov & ~ndimage.binary_erosion(fov, st, border_value=1)
        return float((body & border).any((0, 1)).mean())
    ap_cut, lr_cut = cut_frac(1), cut_frac(0)
    grid_cut_frac = float(np.mean(body[[0, -1]].any((0, 1)) | body[:, [0, -1]].any((0, 1))))

    # marching cubes without padding: surfaces are left open where the window cuts the body
    verts, faces, _, _ = measure.marching_cubes(body.astype(np.float32), 0.5)
    body_pts = sample_surface(verts, faces, N_BODY, rng)

    out = {"input_points": normalise(body_pts)}
    qc = {"id": sid, "body_ml": round(body.sum() * np.prod(sp) / 1000), "body_cut_ap_frac": round(ap_cut, 3),
          "body_cut_lr_frac": round(lr_cut, 3), "body_cut_by_grid_frac": round(grid_cut_frac, 3)}
    occ = {}
    for o in ORGANS:
        m_nii = load_canonical(os.path.join(d, "segmentations", f"{o}.nii.gz"))
        m = resample(np.asarray(m_nii.dataobj) > 0, m_nii.affine, aff, GRID_SHAPE, order=1) > 0.5
        m = largest_cc(m)
        qc[f"{o}_ml"] = round(m.sum() * np.prod(sp) / 1000, 1)
        qc[f"{o}_outside_body_pct"] = round(100 * (m & ~body).sum() / max(m.sum(), 1), 2)
        v, f, _, _ = measure.marching_cubes(np.pad(m, 1).astype(np.float32), 0.5)
        out[o] = normalise(sample_surface(v - 1, f, N_ORGAN, rng))
        occ[o] = m[::2, ::2, ::2]  # 2x-downsampled occupancy for the mean shape
    worst = max(qc[f"{o}_outside_body_pct"] for o in ORGANS)
    qc["rejected"] = ("torso cut by FOV (anterior/posterior)" if ap_cut > args.max_ap_cut else
                      f"organs outside body mask ({worst}%)" if worst > args.max_outside_pct else "")
    return sid, out, aff, qc, occ


def mean_shape(occ_sum, n_train, organ_ml, rng):
    """Per organ: the voxels most often occupied, keeping as many as the median organ volume."""
    pts, labels = [], []
    for li, o in enumerate(ORGANS, start=1):
        prob = occ_sum[o] / n_train
        k = int(round(np.median(organ_ml[o]) / organ_ml["_ds_vox_ml"]))
        thr = np.sort(prob.ravel())[-k]
        m = largest_cc(prob >= thr)
        v, f, _, _ = measure.marching_cubes(np.pad(m, 1).astype(np.float32), 0.5)
        p = sample_surface((v - 1) * 2, f, N_ORGAN, rng)  # back to full-res target-grid voxels
        pts.append(normalise(p))
        labels.append(np.full(N_ORGAN, li, np.int64))
        print(f"  mean {o:<13} threshold p>={thr:.2f}")
    return np.concatenate(pts), np.concatenate(labels)


def run_build(subjects, out, args):
    sel = [r for r in read_csv(os.path.join(out, "selection.csv")) if r["selected"] == "1"]
    sac = {r["id"]: float(r["sacrum_zmax"]) for r in read_csv(os.path.join(out, "survey.csv")) if r["sacrum_zmax"]}
    if args.limit:
        sel = sel[:args.limit]
    meta_split = {r["id"]: r["split"] for r in sel}
    jobs = [(r["id"], subjects[r["id"]], sac[r["id"]], args) for r in sel]

    data, affines, qcs = {}, {}, []
    occs, organ_ml = {}, {}
    with Pool(args.workers) as p:
        for i, (sid, pts, aff, qc, occ) in enumerate(p.imap_unordered(build_one, jobs)):
            qc["meta_split"] = meta_split[sid]
            qcs.append(qc)
            print(f"  [{i + 1}/{len(jobs)}] {sid}  AP cut {qc['body_cut_ap_frac']:.2f}  LR cut {qc['body_cut_lr_frac']:.2f}  "
                  f"max organ outside body {max(qc[f'{o}_outside_body_pct'] for o in ORGANS):.1f}%  "
                  f"{('REJECTED: ' + qc['rejected']) if qc['rejected'] else ''}", flush=True)
            if qc["rejected"]:
                continue
            for k, v in pts.items():
                data[f"{sid}__{k}"] = v
            affines[sid] = aff
            occs[sid] = occ
            organ_ml[sid] = {o: qc[f"{o}_ml"] for o in ORGANS}

    # split after QC so the test fraction holds for the subjects actually kept
    ids = sorted(affines)
    if args.split == "meta":
        split = {i: meta_split[i] for i in ids}
    else:
        perm = np.random.default_rng(args.seed).permutation(ids)
        n_test = int(round(args.test_frac * len(ids)))
        split = {i: ("test" if k < n_test else "train") for k, i in enumerate(perm)}
    for q in qcs:
        q["split"] = split.get(q["id"], "")
    np.savez(os.path.join(out, "data.npz"), **data)
    np.savez(os.path.join(out, "affines.npz"), **affines)
    qcs.sort(key=lambda q: q["id"])
    write_csv(qcs, os.path.join(out, "qc.csv"))
    for s in ("train", "test"):
        with open(os.path.join(out, f"{s}.txt"), "w") as f:
            f.writelines(f"{i}\n" for i in ids if split[i] == s)
    os.makedirs(os.path.join(out, "labels"), exist_ok=True)
    with open(os.path.join(out, "labels", "label_organs_5organs.json"), "w") as f:
        json.dump({str(i): o for i, o in enumerate(ORGANS, start=1)}, f, indent=1)

    train = [i for i in ids if split[i] == "train"]
    n_train = len(train)
    occ_sum = {o: sum(occs[i][o].astype(np.float32) for i in train) for o in ORGANS}
    ml = {o: [organ_ml[i][o] for i in train] for o in ORGANS}
    ml["_ds_vox_ml"] = float(np.prod(affines[ids[0]].diagonal()[:3] * 2)) / 1000
    mean_pc, mean_lab = mean_shape(occ_sum, n_train, ml, np.random.default_rng(0))
    np.savez(os.path.join(out, "mean_shape.npz"), mean_pc_np=mean_pc, mean_label_np=mean_lab)
    print(f"build: {len(ids)} kept, {sum(bool(q['rejected']) for q in qcs)} rejected in QC -> qc.csv")
    print(f"build: {len(ids)} subjects (train {sum(split[i] == 'train' for i in ids)}, "
          f"test {sum(split[i] == 'test' for i in ids)}), mean shape from {n_train} -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="/home/aalempij/Data/data/TotalsegmentatorMRI_dataset_v300")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stages", nargs="+", default=["survey", "select", "build"], choices=["survey", "select", "build"])
    ap.add_argument("--below_mm", type=float, default=40, help="window extent below the sacrum top")
    ap.add_argument("--above_mm", type=float, default=360, help="window extent above the sacrum top")
    ap.add_argument("--xy_spacing", type=float, default=1.7, help="target grid in-plane spacing (mm)")
    ap.add_argument("--min_fov_x", type=int, default=380, help="minimum left-right FOV (mm)")
    ap.add_argument("--max_ap_cut", type=float, default=0.1,
                    help="reject if the body touches the anterior/posterior FOV edge in more than this fraction of slices")
    ap.add_argument("--max_outside_pct", type=float, default=2.0,
                    help="reject if more than this %% of any organ lies outside the body mask")
    ap.add_argument("--split", choices=["random", "meta"], default="random",
                    help="random: --test_frac of the kept subjects; meta: meta.csv split (few test cases survive)")
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="build only the first N selected subjects (testing)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    subjects = find_subjects(args.data)
    if "survey" in args.stages:
        run_survey(subjects, read_splits(args.data), args.out, args.workers)
    if "select" in args.stages:
        run_select(args.out, args)
    if "build" in args.stages:
        run_build(subjects, args.out, args)


if __name__ == "__main__":
    main()
