"""Measure peak GPU memory of one training step (forward + Chamfer loss + backward) per batch size.

Usage: python tools/mem_probe.py [batch sizes...]   (default: 2 4 8)
Run from the repo root with the same environment as main.py.
"""
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, ".")
from model.pointtransformer.pointtransformerlayer import pt_repro as Model
from util.chamfer_loss import calc_chamfer_stacked_objectwise as criterion

cfg_mean = SimpleNamespace(num_encoder=6, planes=[16, 32, 64, 128, 256, 512], blocks=[2, 3, 4, 5, 6, 3],
                           share_planes=8, stride=[1, 5, 4, 4, 4, 4], nsample=[8, 8, 16, 16, 16, 16])
cfg_body = SimpleNamespace(num_encoder=6, planes=[16, 32, 64, 128, 256, 512], blocks=[2, 3, 4, 5, 6, 3],
                           share_planes=8, stride=[1, 4, 4, 4, 4, 4], nsample=[8, 8, 16, 16, 16, 16])
cfg_dec = SimpleNamespace(num_decoder=6, planes=cfg_body.planes[::-1], blocks=cfg_body.blocks[::-1],
                          share_planes=8, nsample=cfg_body.nsample[::-1])

dev = torch.device("cuda")
model = Model(cfg_mean, cfg_body, cfg_dec, c=3, k=3).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
data = np.load("data/tsmri/data.npz")
mean = np.load("data/tsmri/mean_shape.npz")
ids = [l.strip() for l in open("data/tsmri/train.txt")]
organs = ["liver", "spleen", "kidney_left", "kidney_right", "pancreas"]

for B in [int(b) for b in sys.argv[1:]] or [2, 4, 8]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    body = torch.cat([torch.from_numpy(data[f"{i}__input_points"]) for i in ids[:B]]).to(dev)
    tgt = torch.cat([torch.from_numpy(np.concatenate([data[f"{i}__{o}"] for o in organs])) for i in ids[:B]]).to(dev)
    tlab = torch.from_numpy(np.tile(np.repeat(np.arange(1, 6), 4096), B)).to(dev)
    mpts = torch.from_numpy(np.tile(mean["mean_pc_np"], (B, 1))).float().to(dev)
    mlab = torch.from_numpy(np.tile(mean["mean_label_np"], B)).long().to(dev)
    boff = torch.IntTensor([16384 * (k + 1) for k in range(B)]).to(dev)
    moff = torch.IntTensor([20480 * (k + 1) for k in range(B)]).to(dev)
    try:
        out = model([body, body, boff], [mpts, mpts, moff])
        loss = criterion(out, tgt, mlab, tlab, B)
        opt.zero_grad()
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        print(f"batch {B}: peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB allocated, "
              f"{torch.cuda.max_memory_reserved() / 2**30:.2f} GiB reserved", flush=True)
    except torch.cuda.OutOfMemoryError:
        print(f"batch {B}: OUT OF MEMORY", flush=True)
        break
    finally:
        del body, tgt, tlab, mpts, mlab
        out = loss = None
