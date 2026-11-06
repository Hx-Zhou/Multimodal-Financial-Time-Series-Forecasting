"""Evaluate the already exported candidate on the fixed internal temporal holdout."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch

from hs300_dataset import HS300Dataset
from experiments.validation_split import make_validation_split


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def metrics(prediction, target):
    error = prediction.astype(np.float64) - target.astype(np.float64)
    result = {'samples': len(error), 'mae': float(np.abs(error).mean()),
              'mse': float(np.square(error).mean())}
    result['horizons'] = {
        str(h): {'mae': float(np.abs(error[:, h-1]).mean()),
                 'mse': float(np.square(error[:, h-1]).mean())}
        for h in (1, 6, 12, 24)
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=HERE / 'holdout_20261101')
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error('batch size must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    candidate = HERE / 'candidate'
    source = candidate / 'model.py'
    weights = candidate / 'model.pt'
    frozen = read(HERE / 'ensemble/final_summary.json')
    hashes = {'model_py_sha256': digest(source), 'model_pt_sha256': digest(weights)}
    for key, value in hashes.items():
        if value != frozen[key]:
            raise ValueError(f'exported candidate changed: {key}')
    protocol = {
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'candidate': frozen['model'], 'seeds': frozen['seeds'],
        'members_per_seed': frozen['members_per_seed'], **hashes,
        'authorization': 'User requested test evaluation of the already delivered candidate.',
        'evaluation': 'One fixed nine-member equal prediction average; no refitting, seed, member or weight selection.',
        'dataset': 'Known internal temporal holdout from public val.npz; official hidden test is not supplied.',
        'expected_samples': 15390, 'expected_target_start_min': 20240124,
        'expected_target_start_max': 20240227,
        'metric': 'Uniform sample-horizon MAE/MSE in standardized Close, float64 error reductions.',
        'inference': 'Original fixed reader; remove y before calling exported prepare_batch and forward.',
        'references': 'Recompute E0 and verify historical score; report archived E1, E6 and optimized PatchTST scores without selection.',
        'device': 'cpu', 'torch': str(torch.__version__), 'numpy': np.__version__,
        'source_sha256': digest(Path(__file__)),
        'reader_sha256': digest(ROOT / 'hs300_dataset.py'),
        'split_sha256': digest(ROOT / 'experiments/validation_split.py'),
        'val_npz_sha256': digest(args.package_dir / 'data/val.npz'),
        'selection_after_holdout': False,
    }
    # Register the exact candidate and protocol before loading or scoring labels.
    write(args.output_dir / 'protocol.json', protocol)
    print('Frozen candidate verified; loading official reader for internal holdout...', flush=True)
    spec = importlib.util.spec_from_file_location('exported_holdout_candidate', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.build_model().eval().requires_grad_(False)
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
    dataset = HS300Dataset(args.package_dir, 'val')
    split = make_validation_split(dataset)
    indices = split.holdout_indices
    assert len(indices) == protocol['expected_samples']
    assert split.holdout_start_date == protocol['expected_target_start_min']
    assert split.holdout_end_date == protocol['expected_target_start_max']
    assert split.dev_target_end_date < split.holdout_start_date
    predictions = np.empty((len(indices), 24), dtype=np.float32)
    anchors = np.empty_like(predictions)
    target = np.asarray(dataset.y[indices], dtype=np.float32)
    for begin in range(0, len(indices), args.batch_size):
        selected = indices[begin:begin + args.batch_size]
        samples = [{k: v for k, v in dataset[int(index)].items() if k != 'y'}
                   for index in selected]
        batch = module.prepare_batch(samples, 'cpu')
        with torch.inference_mode():
            value = model(batch).numpy()
        if value.shape != (len(selected), 24) or not np.isfinite(value).all():
            raise ValueError('invalid prediction shape or non-finite values')
        predictions[begin:begin + len(selected)] = value
        anchors[begin:begin + len(selected)] = batch['anchor'].numpy()
        if begin == 0 or (begin // args.batch_size + 1) % 10 == 0 or begin + len(selected) == len(indices):
            print(f'{begin + len(selected)}/{len(indices)} samples; elapsed {time.monotonic()-start:.1f}s', flush=True)
    e0 = metrics(anchors, target)
    original = read(ROOT / 'experiments/e1_dlinear/results/holdout/summary.json')
    for key in ('mae', 'mse'):
        np.testing.assert_allclose(e0[key], original['e0_holdout'][key], rtol=0, atol=1e-10)
    score = metrics(predictions, target)
    e6 = read(ROOT / 'experiments/e1_dlinear/e6_plus/results/holdout/summary.json')['scores']['e6_average']
    patch = read(ROOT / 'experiments/transformer_optimization_common/results/holdout.json')['scores']['transformer']
    references = {'E0_last_value': e0, 'historical_enhanced_E1': original['e1_holdout'],
                  'historical_E6_average': e6, 'historical_optimized_PatchTST': patch}
    comparisons = {
        name: {key: 100 * (1 - score[key] / reference[key]) for key in ('mae', 'mse')}
        for name, reference in references.items()
    }
    dates = dataset.target_start_date[indices]
    stocks = dataset.stock_index[indices]
    for filename, values, label in [('per_target_date.csv', dates, 'target_start_date'),
                                    ('per_stock.csv', stocks, 'stock_code')]:
        with (args.output_dir / filename).open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=[label, 'samples', 'candidate_mae', 'candidate_mse', 'e0_mae', 'e0_mse'])
            writer.writeheader()
            for value in np.unique(values):
                mask = values == value
                c, b = metrics(predictions[mask], target[mask]), metrics(anchors[mask], target[mask])
                writer.writerow({label: dataset.stock_codes[int(value)] if label == 'stock_code' else int(value),
                                 'samples': int(mask.sum()), 'candidate_mae': c['mae'], 'candidate_mse': c['mse'],
                                 'e0_mae': b['mae'], 'e0_mse': b['mse']})
    for key, path in [('model_py_sha256', source), ('model_pt_sha256', weights)]:
        assert digest(path) == hashes[key], 'candidate changed during evaluation'
    np.save(args.output_dir / 'predictions.npy', predictions)
    np.save(args.output_dir / 'sample_indices.npy', indices)
    summary = {
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'candidate': frozen['model'], **hashes,
        'internal_holdout': score, 'historical_references': references,
        'improvement_percent_positive_is_better': comparisons,
        'development_reference': frozen['development'],
        'split': {'samples': len(indices), 'stocks': len(np.unique(stocks)),
                  'target_start_min': int(dates.min()), 'target_start_max': int(dates.max()),
                  'target_end_max': int(split.target_end_dates[indices].max()),
                  'development_target_end_max': split.dev_target_end_date},
        'validation': {'same_exported_checkpoint': True, 'labels_removed_before_inference': True,
                       'finite_predictions': True, 'e0_matches_historical': True},
        'selection_after_holdout': False, 'official_hidden_test_evaluated': False,
        'interpretation': 'Previously observed internal holdout, with overlapping windows and shared market dates; not a new blind test or official hidden test. Candidate was fixed before this evaluation.',
        'elapsed_seconds': time.monotonic() - start,
    }
    write(args.output_dir / 'summary.json', summary)
    print(json.dumps({'internal_holdout': score, 'improvement_percent_positive_is_better': comparisons,
                      'elapsed_seconds': summary['elapsed_seconds']}, ensure_ascii=False, indent=2), flush=True)
    dataset.close()


if __name__ == '__main__':
    main()
