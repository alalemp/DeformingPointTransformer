"""Read-only CPU audit of existing arm predictions; no model training or changes."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_cactus import DEFAULT_MUSCLES, load_muscles, mesh_volume, parse_muscles
from visualize_seg import to_world_mm
from volume_eval import ml_per_unit3


def metrics(a, b):
    da = cKDTree(b).query(a)[0]
    db = cKDTree(a).query(b)[0]
    return (da.mean() + db.mean()) / 2, max(np.quantile(da, .95), np.quantile(db, .95))


def main():
    raw = Path.home() / 'datasets/houdini/cactus_BB_TRL_TRM_TRLN_n060p060s060_T_075100125_F_160_220'
    out = Path('results/diagnostics_arm')
    out.mkdir(parents=True, exist_ok=True)
    checks, rows = [], []
    runs = [(160, 'fold_00', Path('results/check160/fold_00'))]
    groups = {'neg': 'BB_-0.6_TRL_-0.6_TRM_-0.6_TRLN_-0.6', 'zero': 'BB_0_TRL_0_TRM_0_TRLN_0', 'pos': 'BB_0.6_TRL_0.6_TRM_0.6_TRLN_0.6'}
    runs += [(frame, fold, Path(f'results/compare{frame}_{groups[config]}/{fold}'))
             for frame in (160, 220)
             for config, fold in [('neg', 'fold_45'), ('zero', 'fold_80'), ('pos', 'fold_77')]]
    for frame, fold, result in runs:
        prepared = Path(f'data/cactus{frame}_loo_val')
        split = prepared / 'folds' / fold
        with np.load(prepared / 'data.npz') as data, np.load(prepared / 'affines.npz') as affines, \
                np.load(split / 'mean_shape.npz') as mean, np.load(result / 'predictions.npz') as pred:
            labels = json.loads((prepared / 'labels/label_organs_arm4.json').read_text())
            ids = {s: (split / f'{s}.txt').read_text().splitlines() for s in ('train', 'val', 'test')}
            groups = {s: {sid.rsplit('_T_', 1)[0] for sid in values} for s, values in ids.items()}
            assert not (groups['train'] & groups['val'] or groups['train'] & groups['test'] or groups['val'] & groups['test'])
            assert set(pred.files) == set(ids['test'])
            tmpl = mean['mean_pc_np']
            lab = mean['mean_label_np']
            assert set(np.unique(lab)) == {int(k) for k in labels}
            assert np.allclose(mean['mesh_nn_w'].sum(1), 1, atol=1e-6)
            assert mean['mesh_nn_idx'].min() >= 0 and mean['mesh_nn_idx'].max() < len(tmpl)
            for key in labels:
                assert np.all(lab[mean['mesh_nn_idx'][mean['mesh_label'] == int(key)]] == int(key))
            # Current experiments have no bone alignment; one affine per batch is safe only if these match.
            first_aff = affines[ids['test'][0]]
            affine_delta = max(float(np.abs(affines[s] - first_aff).max()) for s in sum(ids.values(), []))
            assert affine_delta < 1e-5, 'Different affines: original batch evaluation needs per-sample transforms'
            mesh_labels = mean['mesh_label'][mean['mesh_faces'][:, 0]]
            subjects = pd.read_csv(prepared / 'subjects.csv').set_index('id')
            # Verify actual PLY mesh volume, sampled coordinate bounds, and stored reports for one sample per run.
            sid0 = ids['test'][0]
            subject, arm = sid0.rsplit('_', 1)
            group = subject.rsplit('_T_', 1)[0]
            mapping = parse_muscles(DEFAULT_MUSCLES, 'separated')
            sub = {'muscles': {n: str(raw / 'separated muscles' / folder / group / f'{folder}.{frame:05d}.ply') for n, folder in mapping}}
            organs = load_muscles(sub, arm, 'separated', mapping)
            for name, (v, f) in organs.items():
                true_ml = mesh_volume(v, f) * 1e6
                assert abs(true_ml - subjects.loc[sid0, f'{name}_mesh_ml']) < .011
                target = to_world_mm(data[f'{sid0}__{name}'], affines[sid0])
                used = v[np.unique(f)] * 1000
                assert (target >= used.min(0) - .001).all() and (target <= used.max(0) + .001).all()
            for sid in ids['test']:
                assert pred[sid].shape == tmpl.shape and np.isfinite(pred[sid]).all()
                for key, organ in labels.items():
                    mask = lab == int(key)
                    target = to_world_mm(data[f'{sid}__{organ}'], affines[sid])
                    assert len(target) == mask.sum() == 4096
                    for predictor, pts in [('model', pred[sid][mask]), ('template', tmpl[mask])]:
                        cd, hd = metrics(to_world_mm(pts, affines[sid]), target)
                        rows.append(dict(frame=frame, fold=fold, id=sid, organ=organ, predictor=predictor, cd_mm=cd, hd95_mm=hd))
            frame_rows = [r for r in rows if r['frame'] == frame and r['fold'] == fold]
            independent = pd.DataFrame(frame_rows)
            differences = []
            for organ in labels.values():
                saved = pd.read_csv(result / 'csv/trained_vs_target_arm4.csv', index_col=0)
                calc = independent[(independent.organ == organ) & (independent.predictor == 'model')]
                differences.append(abs(calc.cd_mm.mean() - saved.loc[organ, 'cd']))
            # Zero displacement must reproduce exactly the stored template mesh volume.
            zero_disp = np.zeros_like(tmpl)
            rebuilt = mean['mesh_verts'] + (zero_disp[mean['mesh_nn_idx']] * mean['mesh_nn_w'][..., None]).sum(1)
            assert np.array_equal(rebuilt, mean['mesh_verts'])
            baseline = {}
            for key, organ in labels.items():
                baseline[organ] = mesh_volume(rebuilt, mean['mesh_faces'][mesh_labels == int(key)]) * ml_per_unit3(first_aff)
            checks.append(dict(frame=frame, fold=fold, group_disjoint=True, prediction_ids_match=True,
                               raw_mesh_volume_and_target_bounds_match=True, template_identity_volume_match=True,
                               max_affine_difference=affine_delta, max_model_cd_difference_mm=max(differences),
                               template_ml=baseline))
            print(json.dumps(checks[-1]), flush=True)
    pd.DataFrame(rows).to_csv(out / 'independent_test_metrics.csv', index=False)
    (out / 'audit.json').write_text(json.dumps(checks, indent=2))
    print('AUDIT_COMPLETED: see audit.json; template metrics omit the original random template jitter.', flush=True)


if __name__ == '__main__':
    main()
