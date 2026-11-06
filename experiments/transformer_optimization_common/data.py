"""Same split; Close and stationary-seven input routes, without large disk caches."""
import hashlib
import gc
import time
import numpy as np
import torch
from experiments.e1_dlinear.e6_plus.data import load_store
from experiments.e1_dlinear.e6_plus.data import Samples
from experiments.e1_dlinear.run import ROOT
from hs300_dataset import HS300Dataset, _load_embeddings
from experiments.validation_split import make_validation_split


class Adapter:
    def __init__(self, store):
        self.store = store
        self.train, self.dev = store.train, store.dev

    def batch(self, samples, rows, mode):
        if mode == "close":
            days = samples.starts[rows, None]+np.arange(96)
            return {"close": torch.from_numpy(self.store.close[days].copy())}
        if mode == "market":
            return self.store.batch(samples, rows, "stationary", news=False)
        raise ValueError(mode)

    def anchor(self, samples):
        return self.store.close[samples.starts+95, None]


def close_fingerprint(store, samples):
    digest = hashlib.sha256()
    for left in range(0, len(samples.y), 4096):
        starts = samples.starts[left:left+4096, None]+np.arange(96)
        digest.update(store.close[starts].tobytes())
    return digest.hexdigest()


def load_adapter(package):
    # with_news=False writes only a small JSON metadata file; no pooled embeddings.
    folder = ROOT / f"experiments/transformer_optimization_common/outputs/preprocessing_{time.time_ns()}"
    store = load_store(package, folder, with_news=False)
    return Adapter(store)


def holdout_samples(adapter, package):
    dataset = HS300Dataset(package, "val")
    try:
        split = make_validation_split(dataset)
        assert dataset.stock_codes == adapter.store.codes
        idx = split.holdout_indices
        assert len(idx) == adapter.store.metadata["split"]["holdout_samples"]
        assert split.dev_target_end_date < split.holdout_start_date
        offsets = np.r_[0, np.cumsum([len(dataset.price_values[c]) for c in dataset.stock_codes])].astype(np.int64)
        result = Samples(offsets[dataset.stock_index[idx]]+dataset.input_start[idx], dataset.y[idx].astype(np.float32),
                         dataset.stock_index[idx].copy(), dataset.target_start_date[idx].copy())
        for pos in np.linspace(0, len(idx)-1, 12, dtype=int):
            sample = dataset[int(idx[pos])]
            expected = (sample["numerical"][:, 3]-np.float32(sample["close_mean"]))/max(np.float32(sample["close_std"]), np.float32(1e-6))
            np.testing.assert_array_equal(adapter.store.close[result.starts[pos]:result.starts[pos]+96], expected)
            np.testing.assert_array_equal(result.y[pos], sample["y"])
        return result
    finally:
        dataset.close()
        HS300Dataset._reports.cache_clear()
        _load_embeddings.cache_clear()
        del dataset
        gc.collect()
