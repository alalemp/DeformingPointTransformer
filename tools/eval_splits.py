"""Per-organ CD / HD95 (mm) of a trained model on every train and test subject.

Comparing train vs test separates underfitting (both high) from overfitting
(train low, test high). Unlike main.py, no subject is dropped (batch size 1).

Usage: python tools/eval_splits.py --weights results/tsmri_b8/models/model_weights_best.pth
Run from the repo root with the same environment as main.py.
"""
import argparse
import json
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, ".")
from model.pointtransformer.pointtransformerlayer import pt_repro as Model
from util.evaluation_functions import evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--data", default="data/tsmri")
ap.add_argument("--out", help="optional CSV with per-subject results")
args = ap.parse_args()

cfg_mean = SimpleNamespace(num_encoder=6, planes=[16, 32, 64, 128, 256, 512], blocks=[2, 3, 4, 5, 6, 3],
                           share_planes=8, stride=[1, 5, 4, 4, 4, 4], nsample=[8, 8, 16, 16, 16, 16])
cfg_body = SimpleNamespace(num_encoder=6, planes=[16, 32, 64, 128, 256, 512], blocks=[2, 3, 4, 5, 6, 3],
                           share_planes=8, stride=[1, 4, 4, 4, 4, 4], nsample=[8, 8, 16, 16, 16, 16])
cfg_dec = SimpleNamespace(num_decoder=6, planes=cfg_body.planes[::-1], blocks=cfg_body.blocks[::-1],
                          share_planes=8, nsample=cfg_body.nsample[::-1])
dev = torch.device("cuda")
model = Model(cfg_mean, cfg_body, cfg_dec, c=3, k=3).to(dev)
model.load_state_dict(torch.load(args.weights, map_location=dev, weights_only=True))
model.eval()

organs = list(json.load(open(f"{args.data}/labels/label_organs_5organs.json")).values())
data = np.load(f"{args.data}/data.npz")
aff = np.load(f"{args.data}/affines.npz")
mean = np.load(f"{args.data}/mean_shape.npz")
mpts = torch.from_numpy(mean["mean_pc_np"]).float().to(dev)
mlab = torch.from_numpy(mean["mean_label_np"]).long().to(dev)
tlab = torch.from_numpy(np.repeat(np.arange(1, 6), 4096)).to(dev)
boff, moff = torch.IntTensor([16384]).to(dev), torch.IntTensor([20480]).to(dev)

rows = []
with torch.no_grad():
    for split in ("train", "test"):
        for sid in [l.strip() for l in open(f"{args.data}/{split}.txt")]:
            body = torch.from_numpy(data[f"{sid}__input_points"]).to(dev)
            tgt = torch.from_numpy(np.concatenate([data[f"{sid}__{o}"] for o in organs])).to(dev)
            out = model([body, body, boff], [mpts, mpts, moff])
            for name, pred in (("model", out), ("template", mpts)):
                r = evaluate(pred, tgt, mlab, tlab, batch_size=1, affine=[aff[sid]], device=dev)
                for k, o in enumerate(organs):
                    rows.append({"split": split, "id": sid, "pred": name, "organ": o,
                                 "cd": r["per_organ_cd"][k].item(), "hd95": r["per_organ_hd95"][k].item()})

import pandas as pd
df = pd.DataFrame(rows)
if args.out:
    df.to_csv(args.out, index=False)
for metric in ("cd", "hd95"):
    t = df.pivot_table(index="organ", columns=["split", "pred"], values=metric, aggfunc="mean").loc[organs]
    print(f"\n{metric.upper()} (mm), mean over subjects")
    print(t.round(1).to_string())
n = df.groupby("split").id.nunique()
print(f"\nsubjects: train {n.get('train', 0)}, test {n.get('test', 0)}")
