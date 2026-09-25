"""Build DeformingPointTransformer inputs from the Houdini cactus-pose arm simulations.

Expected layout under --data (as written by export_cactus_pose_ply_mirrored_tree.py):
  final skins for 45th and 56th frames/<subject>/final_skin.<frame:05d>.ply
  entire muscles for 45th and 56th frames/<subject>/entire_muscles.<frame:05d>.ply
Subject folder names carry the simulation parameters, e.g. BB_0.6_TRL_-0.6_TRM_0.6_TRLN_-0.6_T_1;
they are parsed into subjects.csv. Skin and muscle meshes may differ in topology per subject.

Per arm sample:
  input   skin within --crop_mm of a fixed reference position of the target muscles
          (taken once from --reference, so the crop never depends on the sample's own
          muscles), --n_body points sampled uniformly by area
  target  each target muscle piece (--pieces), --n_organ points sampled uniformly by area
The left arm is mirrored (x -> -x) onto the right, so --arms both gives two samples per
subject; both arms of a subject always go to the same split.

All samples share one isotropic normalisation to [-1, 1] (centre and scale of the reference
crop). affines.npz holds, per sample, a 4x4 that util/evaluation_functions.convert_to_mm turns
into mm: that function assumes the MRI voxel grid GRID_SHAPE, so the affine is composed to
undo its grid mapping and apply ours instead.

The template (mean_shape.npz) is built from the training samples only: per muscle, the voxels
most often occupied, keeping as many as the mean muscle volume.

Bone alignment (--align bone): each arm is first mapped onto the reference subject's humerus by
the least-squares similarity transform (rotation, translation, uniform scale) between
corresponding humerus vertices, so differences in bone size / placement are removed before
cropping; the learned task is then muscle shape relative to the skeleton. The humerus is found
automatically (the bone component closest to the target muscles, per arm) and bone meshes must
share one topology. Evaluation mm stay in each subject's own space (the per-sample affine undoes
the alignment). --bone_file may contain {subject} and {frame} for per-subject skeletons.

Leave-N-out (--leave_out N): subjects are shuffled and split into folds of N subjects (both arms
of a subject stay together). Each fold gets folds/fold_XX/{train.txt, test.txt, mean_shape.npz},
with the template built from that fold's training subjects; data.npz, affines.npz and labels are
shared. Run all folds with tools/run_folds.py. The crop / normalisation reference is one fixed
subject for all folds (--reference, default the first subject); in the fold where it is a test
subject its muscles only set where the 10 cm crop boundary lies.

Outputs in --out (same formats as prepare_totalseg_mri.py, for main.py):
  data.npz, mean_shape.npz, affines.npz, labels/label_organs_<label_set>.json,
  train.txt, test.txt, subjects.csv, qc.png          (single split)
  data.npz, affines.npz, labels/, folds/fold_XX/..., subjects.csv, qc.png   (--leave_out N)

Usage:
  python tools/prepare_cactus.py --data /data/aalempij/data/datasets_cactus_poses --out data/cactus45
  python tools/prepare_cactus.py --out data/cactus45_loo --leave_out 1     # leave-one-subject-out
"""
import argparse
import csv
import json
import os
import re
import sys
import zlib
from multiprocessing import Pool

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from skimage import measure

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_totalseg_mri import GRID_SHAPE, largest_cc, sample_surface, write_csv  # noqa: E402

SKIN_DIR = "final skins for 45th and 56th frames"
MUSCLE_DIR = "entire muscles for 45th and 56th frames"
# name:right_piece:left_piece in the 'piece' attribute of entire_muscles (checked on the v1 export:
# these pieces change most when BB / TRL / TRM / TRLN are varied)
DEFAULT_PIECES = "biceps_brachii:5:104,triceps_lateral:94:193,triceps_medial:96:195,triceps_long:95:194"
BONE_FILE = "bones for 45th and 56th frames/bones/v1/bones_v1.{frame:04d}.ply"
PITCH = 0.0015  # voxel size (m) for the mean-shape occupancy grid


# ----------------------------------------------------------------------------- io
def read_ply(path, props=("x", "y", "z")):
    """Binary PLY -> (dict of vertex properties, (F,3) triangles). Polygons are fan-triangulated."""
    with open(path, "rb") as f:
        fields, nv, nf, fmt = [], 0, 0, None
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("format"):
                fmt = ">" if "big_endian" in line else "<"
            elif line.startswith("element vertex"):
                nv = int(line.split()[-1])
            elif line.startswith("element face"):
                nf = int(line.split()[-1])
            elif line.startswith("property") and "list" not in line:
                _, t, name = line.split()
                fields.append((name, fmt + {"float": "f4", "int": "i4", "uchar": "u1", "double": "f8"}[t]))
            elif line == "end_header":
                break
        vdt = np.dtype(fields)
        v = np.frombuffer(f.read(nv * vdt.itemsize), vdt)
        buf = f.read()
    tri_dt = np.dtype([("n", "u1"), ("i", fmt + "i4", (3,))])
    if len(buf) == nf * tri_dt.itemsize and (np.frombuffer(buf, tri_dt, nf)["n"] == 3).all():
        faces = np.frombuffer(buf, tri_dt, nf)["i"].astype(np.int64)
    else:
        faces, pos = [], 0
        for _ in range(nf):
            n = buf[pos]
            idx = np.frombuffer(buf, fmt + "i4", n, pos + 1)
            pos += 1 + 4 * n
            faces += [(idx[0], idx[j], idx[j + 1]) for j in range(1, n - 1)]
        faces = np.array(faces, np.int64)
    return {p: np.asarray(v[p]) for p in props if p in v.dtype.names}, faces


def xyz(v):
    return np.c_[v["x"], v["y"], v["z"]].astype(np.float64)


def find_subjects(data, frame):
    subs = []
    for s in sorted(os.listdir(os.path.join(data, SKIN_DIR))):
        skin = os.path.join(data, SKIN_DIR, s, f"final_skin.{frame:05d}.ply")
        musc = os.path.join(data, MUSCLE_DIR, s, f"entire_muscles.{frame:05d}.ply")
        if os.path.isfile(skin) and os.path.isfile(musc):
            subs.append((s, skin, musc))
    return subs


def parse_params(name):
    """'BB_0.6_TRL_-0.6_TRM_0.6_TRLN_-0.6_T_1' -> {'BB': 0.6, 'TRL': -0.6, ...}"""
    return {k: float(v) for k, v in re.findall(r"([A-Za-z]+)_(-?\d+(?:\.\d+)?)", name)}


# ----------------------------------------------------------------------------- geometry
def submesh(points, faces, keep):
    """Triangles whose three vertices are all kept."""
    f = faces[keep[faces].all(1)]
    return points, f


def mirror(points):
    p = points.copy()
    p[:, 0] = -p[:, 0]
    return p


def organ_points(mpts, mpiece, mfaces, piece):
    keep = mpiece == piece
    if not keep.any():
        raise ValueError(f"muscle piece {piece} not found")
    return submesh(mpts, mfaces, keep)


def similarity_transform(src, dst):
    """4x4 M with dst ~ s R src + t in the least-squares sense (Umeyama 1991); rows correspond."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    x, y = src - mu_s, dst - mu_d
    u, sv, vt = np.linalg.svd(y.T @ x / len(src))
    d = np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))])
    r = u @ d @ vt
    scale = np.trace(np.diag(sv) @ d) / (x ** 2).sum(1).mean()
    m = np.eye(4)
    m[:3, :3] = scale * r
    m[:3, 3] = mu_d - scale * r @ mu_s
    return m


def apply(m, p):
    return p @ m[:3, :3].T + m[:3, 3]


def pick_humerus(bone, faces, muscles):
    """Vertex ids of the bone component with most vertices within 2 cm of the target muscles."""
    n = len(bone)
    adj = coo_matrix((np.ones(3 * len(faces)), (faces.ravel(), np.roll(faces, 1, 1).ravel())), shape=(n, n))
    _, lab = connected_components(adj, directed=False)
    near = cKDTree(muscles).query(bone, distance_upper_bound=0.02)[0] < 0.02
    best = np.bincount(lab[near], minlength=lab.max() + 1).argmax()
    return np.nonzero(lab == best)[0]


def mesh_volume(verts, faces):
    """Enclosed volume of a closed triangle mesh (divergence theorem), in verts units^3."""
    t = verts[faces]
    return abs(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum()) / 6


def humerus_length(pts):
    c = pts - pts.mean(0)
    axis = np.linalg.svd(c, full_matrices=False)[2][0]
    return float(np.ptp(c @ axis))


def occupancy(verts, faces, grid_origin, grid_shape, rng=None):
    """Exact voxelisation of a closed surface: a ray along z through every voxel column,
    voxels whose centre lies between an odd and the next even crossing are inside."""
    nx, ny, nz = grid_shape
    # voxel-centre coordinates; a tiny offset keeps rays off shared edges / vertices
    to_grid = lambda p: (p - grid_origin) / PITCH - 0.5 + np.array([1e-7 * np.pi, 1e-7 * np.e, 0])
    t = to_grid(verts)[faces]  # (F, 3, 3) in voxel units
    hits_i, hits_j, hits_z = [], [], []
    for a, b, c in t:
        lo = np.ceil(np.minimum(np.minimum(a, b), c)[:2]).astype(int)
        hi = np.floor(np.maximum(np.maximum(a, b), c)[:2]).astype(int)
        lo, hi = np.maximum(lo, 0), np.minimum(hi, [nx - 1, ny - 1])
        if (hi < lo).any():
            continue
        gi, gj = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1), indexing="ij")
        gi, gj = gi.ravel(), gj.ravel()
        det = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
        if det == 0:
            continue
        w1 = ((gi - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (gj - a[1])) / det
        w2 = ((b[0] - a[0]) * (gj - a[1]) - (gi - a[0]) * (b[1] - a[1])) / det
        inside = (w1 >= 0) & (w2 >= 0) & (w1 + w2 <= 1)
        hits_i.append(gi[inside])
        hits_j.append(gj[inside])
        hits_z.append((a[2] + w1 * (b[2] - a[2]) + w2 * (c[2] - a[2]))[inside])
    occ = np.zeros(grid_shape, bool)
    if not hits_i:
        return occ
    i, j, z = np.concatenate(hits_i), np.concatenate(hits_j), np.concatenate(hits_z)
    order = np.lexsort((z, j, i))
    i, j, z = i[order], j[order], z[order]
    col = i * ny + j
    start = np.r_[0, np.nonzero(np.diff(col))[0] + 1]
    rank = np.arange(len(col)) - np.repeat(start, np.diff(np.r_[start, len(col)]))
    counts = np.diff(np.r_[start, len(col)])
    ok = np.repeat(counts % 2 == 0, counts)  # skip columns with an odd number of crossings
    enter = ok & (rank % 2 == 0)
    k0 = np.clip(np.ceil(z[enter]).astype(int), 0, nz)
    k1 = np.clip(np.floor(z[np.nonzero(enter)[0] + 1]).astype(int) + 1, 0, nz)
    diff = np.zeros((nx, ny, nz + 1), np.int32)
    np.add.at(diff, (i[enter], j[enter], k0), 1)
    np.add.at(diff, (i[enter], j[enter], k1), -1)
    return np.cumsum(diff, axis=2)[:, :, :nz] > 0


class Frame:
    """Fixed crop, normalisation and voxel grid, all from the reference subject's right arm."""

    def __init__(self, ref_skin, ref_mpts, ref_mpiece, pieces, crop_m):
        self.ref = np.concatenate([ref_mpts[ref_mpiece == r] for _, r, _ in pieces])
        self.tree = cKDTree(self.ref)
        self.crop_m = crop_m
        region = ref_skin[self.tree.query(ref_skin, distance_upper_bound=crop_m)[0] < crop_m]
        lo, hi = region.min(0), region.max(0)
        self.center = (lo + hi) / 2
        self.scale = (hi - lo).max() / 2 * 1.05
        self.grid_origin = self.ref.min(0) - 0.03
        self.grid_shape = tuple(np.ceil((self.ref.max(0) + 0.03 - self.grid_origin) / PITCH).astype(int))
        self.ref_centroids = {name: ref_mpts[ref_mpiece == r].mean(0) for name, r, _ in pieces}
        self.humerus_idx, self.humerus_ref, self.bone_n = {}, {}, None

    def set_bone(self, bone, faces):
        """Reference humerus per arm (left arm mirrored onto the right), for --align bone."""
        self.bone_n = len(bone)
        for arm, pts in (("R", bone), ("L", mirror(bone))):
            self.humerus_idx[arm] = pick_humerus(pts, faces, self.ref)
            self.humerus_ref[arm] = pts[self.humerus_idx[arm]]

    def crop_mask(self, skin):
        return self.tree.query(skin, distance_upper_bound=self.crop_m)[0] < self.crop_m

    def normalise(self, p):
        return ((p - self.center) / self.scale).astype(np.float32)

    def eval_affine(self, to_mm, unalign=None):
        """4x4 for convert_to_mm: its virtual voxel index v -> world mm of our normalised point p.
        convert_to_mm uses v0 = (D0-1)(1-p0)/2, v1 = (D1-1)(1-p1)/2, v2 = (D2-1)(1+p2)/2.
        unalign (4x4) maps the aligned space back to the subject's own space (--align bone)."""
        d = np.array(GRID_SHAPE, np.float64) - 1
        a = np.eye(4)
        sign = np.array([-1.0, -1.0, 1.0])
        a[:3, :3] = np.diag(self.scale * sign * 2 / d)
        a[:3, 3] = self.center - sign * self.scale
        if unalign is not None:
            a = unalign @ a
        return np.diag([to_mm, to_mm, to_mm, 1.0]) @ a


# ----------------------------------------------------------------------------- per sample
def build_one(job):
    sid, subject, arm, skin_path, musc_path, frame, pieces, args, train = job
    rng = np.random.default_rng([args.seed, zlib.crc32(f"{subject}_{arm}".encode())])
    sv, sf = read_ply(skin_path)
    mv, mf = read_ply(musc_path, props=("x", "y", "z", "piece"))
    skin, mpts, mpiece = xyz(sv), xyz(mv), np.asarray(mv["piece"])
    if arm == "L":
        skin, mpts = mirror(skin), mirror(mpts)
    qc = {"id": sid, "subject": subject, "arm": arm}
    align = np.eye(4)
    if args.align == "bone":
        bv, _ = read_ply(os.path.join(args.data, args.bone_file.format(subject=subject, frame=args.frame)))
        bone = xyz(bv) if arm == "R" else mirror(xyz(bv))
        if len(bone) != frame.bone_n:
            raise ValueError(f"{sid}: bone mesh has {len(bone)} vertices, reference has {frame.bone_n}")
        humerus = bone[frame.humerus_idx[arm]]
        align = similarity_transform(humerus, frame.humerus_ref[arm])
        skin, mpts = apply(align, skin), apply(align, mpts)
        qc["humerus_length_mm"] = round(humerus_length(humerus) * args.to_mm, 1)
        qc["align_scale"] = round(float(np.cbrt(np.linalg.det(align[:3, :3]))), 4)
        qc["align_shift_mm"] = round(float(np.linalg.norm(apply(align, humerus) - humerus, axis=1).mean()) * args.to_mm, 2)
    aff = frame.eval_affine(args.to_mm, np.linalg.inv(align))

    keep = frame.crop_mask(skin)
    v, f = submesh(skin, sf, keep)
    out = {"input_points": frame.normalise(sample_surface(v, f, args.n_body, rng))}
    qc["skin_crop_vertices"] = int(keep.sum())
    occ = {}
    for name, right, left in pieces:
        v, f = organ_points(mpts, mpiece, mf, right if arm == "R" else left)
        used = np.unique(f)
        qc[f"{name}_centroid_shift_mm"] = round(float(np.linalg.norm(v[used].mean(0) - frame.ref_centroids[name])) * 1000, 1)
        out[name] = frame.normalise(sample_surface(v, f, args.n_organ, rng))
        # exact volume in the subject's own space (undo the alignment scale)
        qc[f"{name}_mesh_ml"] = round(mesh_volume(v, f) / abs(np.linalg.det(align[:3, :3])) * 1e6, 2)
        if train:
            occ[name] = np.packbits(occupancy(v, f, frame.grid_origin, frame.grid_shape, rng))
            qc[f"{name}_ml"] = round(float(np.unpackbits(occ[name]).sum()) * PITCH ** 3 * 1e6, 1)
    return sid, out, aff, qc, occ


def unpack(bits, shape):
    return np.unpackbits(bits)[:np.prod(shape)].reshape(shape).astype(np.float32)


def mean_template(occ_sum, n_train, volumes, frame, names, n_organ, rng, k_interp=8):
    """Template points per muscle, plus the closed template mesh they were sampled from.
    Each mesh vertex stores its k nearest template points of the same muscle and inverse-distance
    weights, so a predicted point set (point i = template point i moved) deforms the mesh and
    gives a predicted volume (tools/volume_eval.py)."""
    pts, labels, mesh_v, mesh_f, mesh_l, nn_idx, nn_w = [], [], [], [], [], [], []
    n_v = 0
    for li, name in enumerate(names, start=1):
        # light smoothing breaks the ties of a frequency map built from few samples, so that
        # exactly k voxels (the mean volume; volumes are bimodal per parameter) are selected
        prob = ndimage.gaussian_filter(occ_sum[name] / n_train, 1.0)
        k = int(round(np.mean(volumes[name])))
        thr = np.sort(prob.ravel())[-k]
        m = largest_cc(prob >= thr)
        v, f, _, _ = measure.marching_cubes(np.pad(m, 1).astype(np.float32), 0.5)
        world = (v - 1 + 0.5) * PITCH + frame.grid_origin
        p = frame.normalise(sample_surface(world, f, n_organ, rng))
        mv = frame.normalise(world)
        dist, idx = cKDTree(p).query(mv, k=k_interp)
        w = 1 / np.maximum(dist, 1e-9)
        nn_idx.append(idx + (li - 1) * n_organ)
        nn_w.append((w / w.sum(1, keepdims=True)).astype(np.float32))
        mesh_v.append(mv)
        mesh_f.append(f + n_v)
        mesh_l.append(np.full(len(mv), li, np.int64))
        n_v += len(mv)
        pts.append(p)
        labels.append(np.full(n_organ, li, np.int64))
        print(f"  mean {name:<16} threshold p>={thr:.2f}  ({m.sum() * PITCH ** 3 * 1e6:.1f} ml)")
    mesh = {"mesh_verts": np.concatenate(mesh_v), "mesh_faces": np.concatenate(mesh_f),
            "mesh_label": np.concatenate(mesh_l), "mesh_nn_idx": np.concatenate(nn_idx), "mesh_nn_w": np.concatenate(nn_w)}
    return np.concatenate(pts), np.concatenate(labels), mesh


def qc_plot(path, data, mean_pc, mean_lab, sid, names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for row, (title, body, organs) in enumerate([
            (sid, data[f"{sid}__input_points"], [data[f"{sid}__{n}"] for n in names]),
            ("mean shape (template)", None, [mean_pc[mean_lab == i + 1] for i in range(len(names))])]):
        for ax, (i, j, view) in zip(axes[row], [(0, 1, "front (x, y)"), (0, 2, "top (x, z)")]):
            if body is not None:
                ax.scatter(body[:, i], body[:, j], s=0.2, c="#bbbbbb")
            for k, (o, n) in enumerate(zip(organs, names)):
                ax.scatter(o[:, i], o[:, j], s=0.3, c=cols[k % len(cols)], label=n)
            ax.set_title(f"{title}: {view}")
            ax.set_aspect("equal")
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
    axes[0, 0].legend(markerscale=15, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="/data/aalempij/data/datasets_cactus_poses")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frame", type=int, default=45)
    ap.add_argument("--pieces", default=DEFAULT_PIECES, help="name:right_piece:left_piece,...")
    ap.add_argument("--label_set", default="arm4", help="labels/label_organs_<label_set>.json")
    ap.add_argument("--arms", choices=["right", "left", "both"], default="both")
    ap.add_argument("--crop_mm", type=float, default=100, help="skin kept within this distance of the target muscles")
    ap.add_argument("--n_body", type=int, default=16384)
    ap.add_argument("--n_organ", type=int, default=4096)
    ap.add_argument("--reference", help="subject defining crop / normalisation (default: first training subject)")
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--test_subjects", nargs="*", help="explicit test subjects (overrides --test_frac)")
    ap.add_argument("--align", choices=["none", "bone"], default="none",
                    help="bone: map each arm onto the reference humerus (similarity transform) first")
    ap.add_argument("--bone_file", default=BONE_FILE,
                    help="bone mesh relative to --data; may use {subject} and {frame}")
    ap.add_argument("--leave_out", type=int, default=0,
                    help="N > 0: leave-N-subjects-out folds instead of one split (1 = leave-one-out)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_shift_mm", type=float, default=50,
                    help="fail if a muscle centroid is further than this from the reference (piece ids changed?)")
    ap.add_argument("--to_mm", type=float, default=1000, help="mesh units -> mm (Houdini export is in metres)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    pieces = [(n, int(r), int(l)) for n, r, l in (p.split(":") for p in args.pieces.split(","))]
    names = [n for n, _, _ in pieces]
    subs = find_subjects(args.data, args.frame)
    if not subs:
        raise SystemExit(f"no subjects with both skin and muscles for frame {args.frame} under {args.data}")
    os.makedirs(os.path.join(args.out, "labels"), exist_ok=True)

    # split by subject so both arms of one subject are on the same side
    names_sub = [s for s, _, _ in subs]
    if args.leave_out:
        perm = list(np.random.default_rng(args.seed).permutation(names_sub))
        folds = [perm[i:i + args.leave_out] for i in range(0, len(perm), args.leave_out)]
        if len(folds) < 2:
            raise SystemExit(f"--leave_out {args.leave_out} leaves no training subjects ({len(perm)} subjects)")
        test = set()
    elif args.test_subjects:
        test = set(args.test_subjects)
        missing = test - set(names_sub)
        if missing:
            raise SystemExit(f"unknown test subjects: {sorted(missing)}")
    else:
        perm = np.random.default_rng(args.seed).permutation(names_sub)
        test = set(perm[:max(1, int(round(args.test_frac * len(perm))))])
    split = {s: "test" if s in test else "train" for s in names_sub}
    if args.leave_out:
        split = {s: f"fold_{k:02d}" for k, fold in enumerate(folds) for s in fold}
        ref = args.reference or names_sub[0]
        print(f"{len(subs)} subjects in {len(folds)} folds of up to {args.leave_out}, reference {ref}")
    else:
        ref = args.reference or next(s for s in names_sub if split[s] == "train")
        print(f"{len(subs)} subjects (train {sum(v == 'train' for v in split.values())}, test {len(test)}), "
              f"reference {ref}")

    _, ref_skin_path, ref_musc_path = next(x for x in subs if x[0] == ref)
    sv, _ = read_ply(ref_skin_path)
    mv, _ = read_ply(ref_musc_path, props=("x", "y", "z", "piece"))
    frame = Frame(xyz(sv), xyz(mv), np.asarray(mv["piece"]), pieces, args.crop_mm / args.to_mm)
    if args.align == "bone":
        bv, bf = read_ply(os.path.join(args.data, args.bone_file.format(subject=ref, frame=args.frame)))
        frame.set_bone(xyz(bv), bf)
        print(f"bone alignment: humerus {len(frame.humerus_idx['R'])} vertices (right), "
              f"{len(frame.humerus_idx['L'])} (left), length {humerus_length(frame.humerus_ref['R']) * args.to_mm:.0f} mm")
    print(f"normalisation: centre {np.round(frame.center, 3)}, half-width {frame.scale:.3f}; "
          f"occupancy grid {frame.grid_shape}")

    arms = {"right": ["R"], "left": ["L"], "both": ["R", "L"]}[args.arms]
    # occupancy is needed for every sample that is a training sample in some split
    jobs = [(f"{s}_{a}", s, a, sk, mu, frame, pieces, args, split[s] != "test") for s, sk, mu in subs for a in arms]

    data, affines, qcs = {}, {}, []
    occ_total = {n: np.zeros(frame.grid_shape, np.float32) for n in names}
    occ_packed, volumes = {}, {}
    with Pool(args.workers) as p:
        for i, (sid, pts, aff, qc, occ) in enumerate(p.imap_unordered(build_one, jobs)):
            for k, v in pts.items():
                data[f"{sid}__{k}"] = v
            affines[sid] = aff
            qc["split"] = split[qc["subject"]]
            qc.update(parse_params(qc["subject"]))
            qcs.append(qc)
            if occ:
                occ_packed[sid] = occ
                volumes[sid] = {}
                for n in names:
                    o = unpack(occ[n], frame.grid_shape)
                    occ_total[n] += o
                    volumes[sid][n] = int(o.sum())
            shift = max(qc[f"{n}_centroid_shift_mm"] for n in names)
            print(f"  [{i + 1}/{len(jobs)}] {sid}  crop {qc['skin_crop_vertices']} skin verts, "
                  f"max muscle centroid shift {shift} mm"
                  + (f", humerus {qc['humerus_length_mm']} mm, scale {qc['align_scale']}" if "align_scale" in qc else ""),
                  flush=True)
            if shift > args.max_shift_mm:
                raise SystemExit(f"{sid}: a muscle is {shift} mm from the reference position - "
                                 f"are the --pieces ids still right for this export?")

    ids = sorted(affines)
    qcs.sort(key=lambda q: q["id"])
    subject_of = {i: i.rsplit("_", 1)[0] for i in ids}

    def write_split(d, test_subjects):
        """train.txt / test.txt and the template from the training samples, into directory d."""
        os.makedirs(d, exist_ok=True)
        train_ids = [i for i in ids if subject_of[i] not in test_subjects]
        test_ids = [i for i in ids if subject_of[i] in test_subjects]
        occ_sum = {n: occ_total[n].copy() for n in names}  # total minus the held-out samples
        for i in test_ids:
            if i in occ_packed:
                for n in names:
                    occ_sum[n] -= unpack(occ_packed[i][n], frame.grid_shape)
        vols = {n: [volumes[i][n] for i in train_ids] for n in names}
        mean_pc, mean_lab, mesh = mean_template(occ_sum, len(train_ids), vols, frame, names, args.n_organ,
                                                np.random.default_rng(args.seed))
        np.savez(os.path.join(d, "mean_shape.npz"), mean_pc_np=mean_pc, mean_label_np=mean_lab, **mesh)
        for s, lst in (("train", train_ids), ("test", test_ids)):
            with open(os.path.join(d, f"{s}.txt"), "w") as f:
                f.writelines(f"{i}\n" for i in lst)
        return mean_pc, mean_lab, train_ids, test_ids

    np.savez(os.path.join(args.out, "data.npz"), **data)
    np.savez(os.path.join(args.out, "affines.npz"), **affines)
    with open(os.path.join(args.out, "labels", f"label_organs_{args.label_set}.json"), "w") as f:
        json.dump({str(i): n for i, n in enumerate(names, start=1)}, f, indent=1)
    write_csv(qcs, os.path.join(args.out, "subjects.csv"))
    if args.leave_out:
        for k, fold in enumerate(folds):
            print(f"fold_{k:02d}: test {', '.join(fold)}")
            mean_pc, mean_lab, train_ids, test_ids = write_split(
                os.path.join(args.out, "folds", f"fold_{k:02d}"), set(fold))
            if k == 0:
                first = (mean_pc, mean_lab, train_ids[0])
        print(f"wrote {len(ids)} samples in {len(folds)} folds -> {args.out}/folds")
    else:
        mean_pc, mean_lab, train_ids, test_ids = write_split(args.out, test)
        first = (mean_pc, mean_lab, train_ids[0])
        print(f"wrote {len(ids)} samples (train {len(train_ids)}, test {len(test_ids)}) -> {args.out}")
    qc_plot(os.path.join(args.out, "qc.png"), data, first[0], first[1], first[2], names)


if __name__ == "__main__":
    main()
