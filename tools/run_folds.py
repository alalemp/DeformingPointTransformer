"""Train and evaluate main.py on every fold written by prepare_cactus.py --leave_out N,
then combine the per-fold metrics.

Each fold is run as its own main.py process (one GPU, sequentially) with save path
<save>/fold_XX. Folds that already have results are skipped, so an interrupted run can be
restarted with the same command. Arguments after "--" are passed to main.py unchanged.

Summary (<save>/summary_<label_set>.csv, also printed): per organ and metric, the mean over all
test samples (fold means weighted by fold size) and the standard deviation across folds, for
the trained model and for the template alone. Metrics: cd, hd95 (mm) and, when the folds have
predictions and a template mesh (tools/volume_eval.py), vol_abs_err_ml and vol_abs_err_pct.

--eval_only re-evaluates existing fold models (main.py --eval_only) without retraining, e.g. to
add predictions / volume errors to runs made before those existed.

Usage:
  python tools/run_folds.py --prepared data/cactus45_loo --label_set arm4 --save results/cactus_loo \\
      -- --batch_size 4 --epochs 100
  python tools/run_folds.py --prepared data/cactus45_loo --label_set arm4 --save results/cactus_loo --summary_only
"""
import argparse
import glob
import os
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from volume_eval import volume_rows  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_fold(fold_dir, prepared, label_set, save, main_args, log_name="train.log"):
    cmd = [sys.executable, os.path.join(REPO, "main.py"),
           "--training_list_file_path", os.path.join(fold_dir, "train.txt"),
           "--testing_list_file_path", os.path.join(fold_dir, "test.txt"),
           "--init_mean_shape_path", os.path.join(fold_dir, "mean_shape.npz"),
           "--label_json_file_path", os.path.join(prepared, "labels"),
           "--which_label", label_set,
           "--data_root", os.path.join(prepared, "data.npz"),
           "--conversion_path", os.path.join(prepared, "affines.npz"),
           "--save_path", save] + main_args
    if os.path.isfile(os.path.join(fold_dir, "val.txt")):   # validation-based checkpoint selection
        cmd += ["--validation_list_file_path", os.path.join(fold_dir, "val.txt")]
    os.makedirs(save, exist_ok=True)
    with open(os.path.join(save, log_name), "w") as log:
        return subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=REPO).returncode


def summarise(fold_dirs, save, label_set):
    rows = []
    for fd in fold_dirs:
        name = os.path.basename(fd)
        n_test = sum(1 for line in open(os.path.join(fd, "test.txt")) if line.strip())
        for pred, fname in (("model", "trained_vs_target"), ("template", "init_vs_target")):
            path = os.path.join(save, name, "csv", f"{fname}_{label_set}.csv")
            if not os.path.isfile(path):
                continue
            df = pd.read_csv(path, index_col=0)
            for organ, r in df.iterrows():
                for metric in ("cd", "hd95"):
                    rows.append({"fold": name, "n_test": n_test, "pred": pred, "organ": organ,
                                 "metric": metric, "value": r[metric]})
        vol = pd.DataFrame(volume_rows(os.path.dirname(os.path.dirname(fd)), fd, os.path.join(save, name)))
        if len(vol):
            vol.to_csv(os.path.join(save, name, "csv", "volumes.csv"), index=False)
            vol["abs_err_ml"] = vol.err_ml.abs()
            for (pred, organ), g in vol.groupby(["pred", "organ"]):
                for metric, col in (("vol_abs_err_ml", "abs_err_ml"), ("vol_abs_err_pct", "abs_err_pct")):
                    rows.append({"fold": name, "n_test": n_test, "pred": pred, "organ": organ,
                                 "metric": metric, "value": g[col].mean()})
    if not rows:
        print("no fold results found")
        return
    df = pd.DataFrame(rows)
    out = (df.groupby(["metric", "organ", "pred"], sort=False)
             .apply(lambda g: pd.Series({"mean": np.average(g.value, weights=g.n_test),
                                         "std_across_folds": g.value.std(ddof=0),
                                         "folds": g.fold.nunique()}), include_groups=False)
             .reset_index())
    out.to_csv(os.path.join(save, f"summary_{label_set}.csv"), index=False)
    for metric in out.metric.unique():
        t = out[out.metric == metric].pivot(index="organ", columns="pred", values=["mean", "std_across_folds"])
        unit = {"cd": "mm", "hd95": "mm", "vol_abs_err_ml": "ml", "vol_abs_err_pct": "%"}[metric]
        print(f"\n{metric.upper()} ({unit}): mean over test samples, std across folds")
        print(t.round(2).to_string())
    print(f"\nfolds with results: {df.fold.nunique()}/{len(fold_dirs)} -> {save}/summary_{label_set}.csv")


def main():
    argv = sys.argv[1:]
    main_args = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prepared", required=True, help="output dir of prepare_cactus.py --leave_out N")
    ap.add_argument("--label_set", required=True)
    ap.add_argument("--save", required=True)
    ap.add_argument("--folds", nargs="*", help="only these folds, e.g. fold_00 fold_03")
    ap.add_argument("--summary_only", action="store_true")
    ap.add_argument("--eval_only", action="store_true", help="re-evaluate existing fold models, no training")
    args = ap.parse_args(argv)

    fold_dirs = sorted(glob.glob(os.path.join(args.prepared, "folds", "fold_*")))
    if not fold_dirs:
        raise SystemExit(f"no folds under {args.prepared}/folds - run prepare_cactus.py with --leave_out N")
    todo = [fd for fd in fold_dirs if not args.folds or os.path.basename(fd) in args.folds]
    if not args.summary_only:
        for fd in todo:
            name = os.path.basename(fd)
            save = os.path.join(args.save, name)
            if args.eval_only:
                if not os.path.isfile(os.path.join(save, "models", "model_weights_best.pth")):
                    print(f"{name}: no trained model, skipping")
                    continue
                print(f"{name}: evaluating ... (log: {save}/eval.log)", flush=True)
                rc = run_fold(fd, args.prepared, args.label_set, save, main_args + ["--eval_only"], "eval.log")
            elif os.path.isfile(os.path.join(save, "csv", f"trained_vs_target_{args.label_set}.csv")):
                print(f"{name}: done already, skipping")
                continue
            else:
                print(f"{name}: training ... (log: {save}/train.log)", flush=True)
                rc = run_fold(fd, args.prepared, args.label_set, save, main_args)
            print(f"{name}: {'finished' if rc == 0 else f'FAILED (exit {rc}), see log'}", flush=True)
    summarise(fold_dirs, args.save, args.label_set)


if __name__ == "__main__":
    main()
