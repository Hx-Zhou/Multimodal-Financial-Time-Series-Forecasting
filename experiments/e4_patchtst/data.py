"""Reuse verified Close materialization and release unused news/report caches."""
import gc
from hs300_dataset import HS300Dataset, _load_embeddings
from experiments.e1_dlinear.data import load_data, materialize
from experiments.validation_split import make_validation_split


def release_reader_caches():
    HS300Dataset._reports.cache_clear()
    _load_embeddings.cache_clear()
    gc.collect()


def development_data(package):
    try:
        return load_data(package)
    finally:
        release_reader_caches()


def holdout_data(package, reference):
    dataset = HS300Dataset(package, "val")
    try:
        split = make_validation_split(dataset)
        assert len(split.holdout_indices) == reference["holdout_samples"]
        assert split.holdout_start_date == reference["holdout_target_start_date_min"]
        assert split.holdout_end_date == reference["holdout_target_start_date_max"]
        assert split.dev_target_end_date < split.holdout_start_date
        return materialize(dataset, split.holdout_indices)
    finally:
        dataset.close()
        del dataset
        release_reader_caches()
