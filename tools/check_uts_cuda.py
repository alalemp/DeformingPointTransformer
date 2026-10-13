"""Run on a PBS-allocated GPU after installing pointops; also JIT-build Chamfer3D."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

if not torch.cuda.is_available():
    raise SystemExit("No CUDA GPU visible: run this check in a GPU PBS job.")

from lib.pointops.functions import pointops
from Chamfer3D.dist_chamfer_3D import chamfer_3DDist

print("PyTorch", torch.__version__, "CUDA", torch.version.cuda)
print("GPU", torch.cuda.get_device_name(0))
points = torch.rand(1000, 3, device="cuda")
offset = torch.tensor([1000], dtype=torch.int32, device="cuda")
neighbors, distances = pointops.knnquery(8, points, points, offset, offset)
assert tuple(neighbors.shape) == (1000, 8)
assert torch.isfinite(distances).all()
sampled = pointops.furthestsampling(
    points, offset, torch.tensor([100], dtype=torch.int32, device="cuda")
)
assert tuple(sampled.shape) == (100,)

x = torch.rand(1, 128, 3, device="cuda", requires_grad=True)
y = torch.rand(1, 96, 3, device="cuda", requires_grad=True)
d1, d2, _, _ = chamfer_3DDist()(x, y)
assert tuple(d1.shape) == (1, 128) and tuple(d2.shape) == (1, 96)
loss = d1.mean() + d2.mean()
assert torch.isfinite(loss)
loss.backward()
assert torch.isfinite(x.grad).all() and torch.isfinite(y.grad).all()
torch.cuda.synchronize()
print("CUDA_CHECK_OK: pointops sampling/kNN and Chamfer forward/backward")
