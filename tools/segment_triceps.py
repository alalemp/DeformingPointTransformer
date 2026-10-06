"""Run TotalSegmentator's thigh_shoulder_muscles_mr task on the TotalSegmentator MRI subjects and
record which ones have a triceps_brachii large enough to use.

Per subject (<data>/sXXXX with mri.nii.gz and segmentations/):
  1. skip subjects with no humerus in the existing ground truth (humerus_left / humerus_right):
     the upper arm is not in the scan, so there can be no triceps (--all runs every subject)
  2. TotalSegmentator -i mri.nii.gz -o <subject>/<--out_name> --task thigh_shoulder_muscles_mr
     (skipped if triceps_brachii.nii.gz already exists, unless --redo). The task's 18 class names
     do not clash with the dataset's own masks, so writing into segmentations/ overwrites nothing.
  3. measure triceps_brachii: the mask holds both arms, so it is split into connected components
     (>= 1 ml); each is assigned to the left / right arm by the nearer humerus and checked for
     being cut by the scan's field of view (touching the volume border = partial muscle)

An arm is usable if its triceps is >= --min_ml and not cut by the FOV; a subject is usable if
--need (any | both) arms are. Results are appended per subject, so an interrupted run resumes.

Outputs (--report_dir, default data/triceps):
  triceps_survey.csv   one row per subject: status, volumes, per-arm usability, reasons
  usable_subjects.txt  subjects that pass
  failures.log         TotalSegmentator errors (last lines of its output)

The TotalSegmentator licence must already be set once (totalseg_set_license -l <key>); it is
stored in ~/.totalsegmentator/config.json and is not needed by this script.

Usage (on a GPU node, conda env bodyseg):
  python tools/segment_triceps.py                          # all subjects with a humerus
  python tools/segment_triceps.py --limit 3                # quick test
  python tools/segment_triceps.py --measure_only           # re-score existing outputs, e.g. new --min_ml
  python tools/segment_triceps.py --subjects s0852 s0925 --device gpu:1
"""
import argparse
import csv
import glob
import os
import subprocess
import sys
import time

import nibabel as nib
import numpy as np
from scipy import ndimage

DATA = "/home/aalempij/Data/data/TotalsegmentatorMRI_dataset_v300/Totalsegmentator_dataset_v300"
TASK = "thigh_shoulder_muscles_mr"
FIELDS = ["subject", "status", "humerus_left_ml", "humerus_right_ml", "triceps_ml", "n_components",
          "left_ml", "left_cut", "left_usable", "right_ml", "right_cut", "right_usable", "unassigned_ml",
          "usable", "reason", "seconds"]


def load_canonical(path):
    img = nib.as_closest_canonical(nib.load(path))
    return np.asarray(img.dataobj) > 0, img.affine, float(np.prod(img.header.get_zooms()[:3])) / 1000


def centroid_world(mask, affine):
    return (affine @ np.r_[np.array(np.nonzero(mask)).mean(1), 1])[:3]


def touches_border(mask):
    return any(mask.take([0, -1], axis=ax).any() for ax in range(3))


def humerus(seg_dir):
    """{side: (ml, world centroid)} for the humerus masks present."""
    out = {}
    for side in ("left", "right"):
        p = os.path.join(seg_dir, f"humerus_{side}.nii.gz")
        if os.path.isfile(p):
            m, aff, vml = load_canonical(p)
            if m.any():
                out[side] = (m.sum() * vml, centroid_world(m, aff))
    return out


def measure(triceps_path, hum, args):
    """Per-arm triceps volume / FOV cut / usability."""
    m, aff, vml = load_canonical(triceps_path)
    row = {"triceps_ml": round(m.sum() * vml, 1)}
    lab, n = ndimage.label(m, ndimage.generate_binary_structure(3, 3))
    sizes = ndimage.sum(m, lab, range(1, n + 1)) * vml
    comps = [i + 1 for i, s in enumerate(sizes) if s >= 1.0]
    row["n_components"] = len(comps)
    arm_ml, arm_cut, unassigned = {"left": 0.0, "right": 0.0}, {"left": False, "right": False}, 0.0
    for c in comps:
        cm = lab == c
        if not hum:
            unassigned += sizes[c - 1]
            continue
        cen = centroid_world(cm, aff)
        side = min(hum, key=lambda s: np.linalg.norm(cen - hum[s][1]))
        arm_ml[side] += sizes[c - 1]
        arm_cut[side] |= touches_border(cm)
    usable_arms = []
    for side in ("left", "right"):
        ok = arm_ml[side] >= args.min_ml and not arm_cut[side]
        row.update({f"{side}_ml": round(arm_ml[side], 1), f"{side}_cut": int(arm_cut[side]), f"{side}_usable": int(ok)})
        if ok:
            usable_arms.append(side)
    row["unassigned_ml"] = round(unassigned, 1)
    need = 2 if args.need == "both" else 1
    row["usable"] = int(len(usable_arms) >= need)
    if not row["usable"]:
        why = []
        for side in ("left", "right"):
            if arm_ml[side] < args.min_ml:
                why.append(f"{side} {arm_ml[side]:.0f} ml < {args.min_ml:.0f}")
            elif arm_cut[side]:
                why.append(f"{side} cut by FOV")
        row["reason"] = "; ".join(why)
    return row


def run_totalseg(subject_dir, out_dir, args):
    exe = os.path.join(os.path.dirname(sys.executable), "TotalSegmentator")
    cmd = [exe, "-i", os.path.join(subject_dir, "mri.nii.gz"), "-o", out_dir, "--task", args.task,
           "-d", args.device, "-q"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip().splitlines()[-15:]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--task", default=TASK)
    ap.add_argument("--out_name", default="segmentations", help="output folder inside each subject")
    ap.add_argument("--device", default="gpu", help="gpu, gpu:X or cpu")
    ap.add_argument("--min_ml", type=float, default=100, help="minimum triceps volume per arm (ml)")
    ap.add_argument("--need", choices=["any", "both"], default="any", help="usable arms required per subject")
    ap.add_argument("--all", action="store_true", help="also run subjects without a humerus in the ground truth")
    ap.add_argument("--subjects", nargs="*", help="only these subjects")
    ap.add_argument("--limit", type=int, default=0, help="process at most N subjects (testing)")
    ap.add_argument("--redo", action="store_true", help="rerun TotalSegmentator even if output exists")
    ap.add_argument("--measure_only", action="store_true", help="do not run TotalSegmentator, only score outputs")
    ap.add_argument("--report_dir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                         "data", "triceps"))
    args = ap.parse_args()

    subjects = sorted(os.path.basename(d) for d in glob.glob(os.path.join(args.data, "s[0-9]*"))
                      if os.path.isfile(os.path.join(d, "mri.nii.gz")))
    if args.subjects:
        subjects = [s for s in subjects if s in set(args.subjects)]
    os.makedirs(args.report_dir, exist_ok=True)
    report = os.path.join(args.report_dir, "triceps_survey.csv")
    done = {}
    if os.path.isfile(report) and not args.redo and not args.measure_only:
        with open(report) as f:
            done = {r["subject"]: r for r in csv.DictReader(f)}
    rows = dict(done)

    todo = [s for s in subjects if s not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(subjects)} subjects, {len(done)} already in the report, {len(todo)} to process", flush=True)
    for i, s in enumerate(todo, 1):
        t0 = time.time()
        d = os.path.join(args.data, s)
        out_dir = os.path.join(d, args.out_name)
        tri = os.path.join(out_dir, "triceps_brachii.nii.gz")
        hum = humerus(os.path.join(d, "segmentations"))
        row = {"subject": s, "humerus_left_ml": round(hum["left"][0], 1) if "left" in hum else 0,
               "humerus_right_ml": round(hum["right"][0], 1) if "right" in hum else 0, "usable": 0}
        if not hum and not args.all:
            row.update(status="skipped", reason="no humerus in ground truth (arm not in scan)")
        elif args.measure_only and not os.path.isfile(tri):
            row.update(status="not segmented", reason="no triceps_brachii.nii.gz")
        else:
            rc, tail = 0, []
            if not args.measure_only and (args.redo or not os.path.isfile(tri)):
                rc, tail = run_totalseg(d, out_dir, args)
            if rc != 0 or not os.path.isfile(tri):
                row.update(status="failed", reason=f"TotalSegmentator exit {rc}")
                with open(os.path.join(args.report_dir, "failures.log"), "a") as f:
                    f.write(f"== {s} (exit {rc})\n" + "\n".join(tail) + "\n")
            else:
                row.update(status="segmented", **measure(tri, hum, args))
        row["seconds"] = round(time.time() - t0, 1)
        rows[s] = row
        print(f"[{i}/{len(todo)}] {s}: {row['status']}"
              + (f", triceps L {row.get('left_ml')} ml R {row.get('right_ml')} ml -> "
                 f"{'USABLE' if row['usable'] else 'not usable (' + row.get('reason', '') + ')'}"
                 if row["status"] == "segmented" else f" ({row.get('reason', '')})")
              + f"  [{row['seconds']} s]", flush=True)
        # rewrite after every subject so an interruption loses nothing
        with open(report, "w", newline="") as f:
            w = csv.DictWriter(f, FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows[k] for k in sorted(rows))

    usable = sorted(k for k, r in rows.items() if str(r.get("usable")) == "1")
    with open(os.path.join(args.report_dir, "usable_subjects.txt"), "w") as f:
        f.writelines(f"{s}\n" for s in usable)
    status = {}
    for r in rows.values():
        status[r["status"]] = status.get(r["status"], 0) + 1
    print(f"\n{len(rows)} subjects in report: {status}; usable: {len(usable)} -> {args.report_dir}")


if __name__ == "__main__":
    main()
