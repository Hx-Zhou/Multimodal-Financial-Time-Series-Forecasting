"""Replicate fixed E4 training, calibrate on development, score holdout only if qualified."""
import argparse
from datetime import datetime, timezone
import gc
import importlib.util
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

from experiments.e1_dlinear.run import ROOT, metrics, write_json
from experiments.e1_dlinear.data import sha256_file
from experiments.e1_dlinear.model import build_model as build_e1
from experiments.e4_patchtst import run as original
from experiments.e4_patchtst.data import development_data, holdout_data
from experiments.e4_patchtst.model import build_model
from .model import shrink_state

HERE = Path(__file__).resolve().parent
SEEDS = (42, 43, 44)
ALPHAS = (0.5, 0.25)  # ordered primary and secondary, selected before new seed runs
KINDS = original.KINDS


def read(path):
    return original.read(path)


def replica_root(seed):
    return original.HERE if seed == 42 else HERE / f"replicas/seed{seed}"


def run_record(seed, kind):
    return read(replica_root(seed) / f"results/{kind}.json")


def source_checkpoint(seed, kind):
    path = replica_root(seed) / f"outputs/{kind}/best_trained.pt"
    assert sha256_file(path) == run_record(seed, kind)["checkpoint_sha256"]["best_trained.pt"]
    return path


def load_trainer(seed):
    # Execute the unchanged original trainer in its own namespace. Only its IO
    # root changes; original.run.HERE, training code and all hyperparameters stay intact.
    name = f"experiments.e4_patchtst._replica_seed{seed}"
    spec = importlib.util.spec_from_file_location(name, original.HERE / "run.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module.HERE = replica_root(seed)
    return module


def source_hashes():
    result = original.source_hashes()
    files = list(HERE.glob("*.py"))+[HERE / "README.md"]
    result.update({p.relative_to(ROOT).as_posix(): sha256_file(p) for p in files})
    return result


def halves(prediction, data):
    dates = np.unique(data.target_dates)
    boundary = int(dates[len(dates)//2])
    return {"boundary": boundary,
            "early": metrics(prediction[data.target_dates < boundary], data.y.numpy()[data.target_dates < boundary]),
            "late": metrics(prediction[data.target_dates >= boundary], data.y.numpy()[data.target_dates >= boundary])}


def calibrate(kind, seed, alpha, data, batch):
    path = source_checkpoint(seed, kind)
    model, saved = original.load_model(path)
    raw = np.load(replica_root(seed) / f"outputs/{kind}/dev_best_trained.npy")
    raw_score = metrics(raw, data.y.numpy())
    expected = run_record(seed, kind)["best_trained_development"]
    np.testing.assert_allclose(raw_score["mae"], expected["mae"], rtol=0, atol=1e-9)
    if alpha == 1.0:
        return raw
    model.load_state_dict(shrink_state(saved["state_dict"], kind, alpha))
    prediction = original.predict(model, data.x, batch)
    anchor = data.x[:, -1:].numpy().astype(np.float64)
    algebra = anchor+alpha*(raw.astype(np.float64)-anchor)
    np.testing.assert_allclose(prediction, algebra, rtol=0, atol=2e-5)
    return prediction


def select_candidate(candidates, reference, reference_halves):
    decisions = {}
    for alpha in ALPHAS:
        row = candidates[str(alpha)]
        checks = {"mae": row["average"]["mae"] <= reference["mae"]*0.999,
                  "mse": row["average"]["mse"] <= reference["mse"]*1.005,
                  "seeds": row["seeds_improved_vs_raw"] >= 2,
                  "periods": all(row["halves"][key]["mae"] <= reference_halves[key]["mae"]*1.005 for key in ("early", "late"))}
        decisions[str(alpha)] = {"passed": all(checks.values()), "checks": checks}
    eligible = [a for a in ALPHAS if decisions[str(a)]["passed"]]
    selected = None
    for alpha in eligible:
        if selected is None or candidates[str(alpha)]["average"]["mae"] < candidates[str(selected)]["average"]["mae"]-1e-5:
            selected = alpha
    return {"selected_alpha": selected, "checks": decisions}


def development(dev, batch):
    output = HERE / "outputs"
    output.mkdir(exist_ok=True)
    e0 = np.repeat(dev.x[:, -1:].numpy(), 24, axis=1)
    families, arrays = {}, {}
    for kind in KINDS:
        families[kind], arrays[kind] = {}, {}
        raw_scores = {s: run_record(s, kind)["best_trained_development"] for s in SEEDS}
        for alpha in (1.0, *ALPHAS):
            values = [calibrate(kind, seed, alpha, dev, batch) for seed in SEEDS]
            average = np.mean(values, axis=0)
            scores = [{"seed": seed, **metrics(pred, dev.y.numpy())} for seed, pred in zip(SEEDS, values)]
            families[kind][str(alpha)] = {"average": metrics(average, dev.y.numpy()), "seeds": scores,
                "halves": halves(average, dev),
                "seeds_improved_vs_raw": sum(row["mae"] < raw_scores[row["seed"]]["mae"] for row in scores)}
            arrays[kind][str(alpha)] = average
            np.save(output / f"dev_{kind}_alpha{alpha}.npy", average)
            print(f"development {kind} alpha={alpha} MAE={families[kind][str(alpha)]['average']['mae']:.9f} MSE={families[kind][str(alpha)]['average']['mse']:.9f}", flush=True)
    e1path = HERE.parent / "e1_dlinear/outputs/calibration/model.pt"
    e1ref = read(HERE.parent / "e1_dlinear/results/holdout/summary.json")
    assert sha256_file(e1path) == e1ref["checkpoint_sha256"]
    saved = torch.load(e1path, weights_only=True, map_location="cpu")
    e1 = build_e1(saved["config"])
    e1.load_state_dict(saved["state_dict"])
    e1pred = original.predict(e1, dev.x, batch)
    e1score = metrics(e1pred, dev.y.numpy())
    historical = read(HERE.parent / "e1_dlinear/results/calibration/summary.json")
    np.testing.assert_allclose(e1score["mae"], historical["averaged_model"]["development"]["mae"], rtol=0, atol=1e-8)
    control_options = {"0.0": e0, **arrays["dlinear_control"]}
    control_alpha = min(control_options, key=lambda a: metrics(control_options[a], dev.y.numpy())["mae"])
    control_prediction = control_options[control_alpha]
    control_score = metrics(control_prediction, dev.y.numpy())
    references = {"enhanced_e1": e1pred, "calibrated_dlinear": control_prediction,
                  "raw_patchtst_ensemble": arrays["patchtst"]["1.0"]}
    reference_name = min(references, key=lambda key: metrics(references[key], dev.y.numpy())["mae"])
    reference_prediction = references[reference_name]
    reference_score = metrics(reference_prediction, dev.y.numpy())
    decision = select_candidate(families["patchtst"], reference_score, halves(reference_prediction, dev))
    result = {"families": families, "e0": metrics(e0, dev.y.numpy()), "enhanced_e1": e1score,
              "control_alpha": float(control_alpha), "control_development": control_score,
              "reference": reference_name, "reference_development": reference_score,
              "decision": decision, "holdout_evaluated": False, "seeds": list(SEEDS),
              "normalization": "Same history transform as original E4; global alpha folded only into output heads"}
    path = HERE / "results/development.json"
    if path.exists():
        assert read(path) == result
    else:
        write_json(path, result)
    return result


def freeze(kind, alpha):
    paths, hashes = {}, {}
    for seed in SEEDS:
        path = HERE / f"outputs/frozen/{kind}_seed{seed}.pt"
        saved = torch.load(source_checkpoint(seed, kind), weights_only=True, map_location="cpu")
        saved["state_dict"] = shrink_state(saved["state_dict"], kind, alpha)
        saved["calibration"] = {"alpha": alpha, "source_sha256": sha256_file(source_checkpoint(seed, kind))}
        if path.exists():
            previous = torch.load(path, weights_only=True, map_location="cpu")
            assert previous["calibration"] == saved["calibration"]
            assert all(torch.equal(v, previous["state_dict"][k]) for k, v in saved["state_dict"].items())
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(saved, path)
        paths[seed], hashes[str(seed)] = path, sha256_file(path)
    return paths, hashes


def evaluate_holdout(args, metadata, development_result):
    alpha = development_result["decision"]["selected_alpha"]
    if alpha is None:
        print("No candidate passed development gates; holdout will not be evaluated.", flush=True)
        return None
    models, hashes = {}, {}
    for kind, value in (("patchtst", alpha), ("dlinear_control", development_result["control_alpha"])):
        models[kind], hashes[kind] = freeze(kind, value)
    protocol = {"alpha": alpha, "control_alpha": development_result["control_alpha"], "seeds": list(SEEDS),
                "checkpoint_sha256": hashes, "development_sha256": sha256_file(HERE / "results/development.json"),
                "candidates": "Only the development-qualified global alpha, fixed 3-seed equal average; unscaled PatchTST average and dev-selected calibrated DLinear as controls",
                "status": "Previously revealed internal holdout, not fresh blind/official test; no decisions or refitting after these scores"}
    path = HERE / "results/holdout_protocol.json"
    if path.exists():
        assert read(path) == protocol
    else:
        write_json(path, protocol)
    if (HERE / "results/holdout.json").exists():
        return read(HERE / "results/holdout.json")
    data = holdout_data(args.package_dir.resolve(), metadata)
    predictions, scores = {}, {}
    for kind in KINDS:
        values = []
        for seed in SEEDS:
            model, _ = original.load_model(models[kind][seed])
            value = original.predict(model, data.x, args.batch_size)
            values.append(value)
            scores[f"{kind}_seed{seed}"] = metrics(value, data.y.numpy())
        predictions[kind] = np.mean(values, axis=0)
    raw_values = []
    for seed in SEEDS:
        model, _ = original.load_model(source_checkpoint(seed, "patchtst"))
        raw_values.append(original.predict(model, data.x, args.batch_size))
    predictions["patchtst_raw"] = np.mean(raw_values, axis=0)
    predictions["e0"] = np.repeat(data.x[:, -1:].numpy(), 24, axis=1)
    for name, value in predictions.items():
        score = metrics(value, data.y.numpy())
        error = torch.from_numpy(value).double()-data.y.double()
        np.testing.assert_allclose([score["mae"], score["mse"]], [error.abs().mean(), error.square().mean()], rtol=1e-12, atol=1e-12)
        scores[name] = score
        np.save(HERE / f"outputs/holdout_{name}.npy", value)
        print(f"holdout {name} MAE={score['mae']:.9f} MSE={score['mse']:.9f}", flush=True)
    for kind in KINDS:
        for seed in SEEDS:
            assert sha256_file(models[kind][seed]) == hashes[kind][str(seed)]
    result = {"alpha": alpha, "scores": scores, "holdout_evaluated": True,
              "selection_after_holdout": False, "samples": len(data.x), "seeds": list(SEEDS)}
    write_json(HERE / "results/holdout.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if shutil.disk_usage(HERE).free < 512*1024**2:
        raise RuntimeError("Insufficient disk reserve")
    torch.set_flush_denormal(True)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.mha.set_fastpath_enabled(False)
    parent = read(original.HERE / "results/protocol.json")
    assert original.source_hashes() == parent["source_sha256"]
    settings = parent["settings"]
    hashes = source_hashes()
    protocol = {"seeds": list(SEEDS), "alphas": list(ALPHAS), "settings": settings, "source_sha256": hashes,
                "seed42_checkpoint_sha256": {kind: run_record(42, kind)["checkpoint_sha256"] for kind in KINDS},
                "checkpoint_selection": "Same original unscaled-development training/epoch0 early stopping; calibration uses best_trained checkpoint for each seed, alpha0 is E0 fallback",
                "alpha_selection": "One global alpha for all seeds; choose minimum ensemble dev MAE among qualified candidates; 0.5 wins differences <=1e-5",
                "gates": "MAE >=0.1% better than strongest of enhanced E1, dev-best calibrated DLinear and raw PatchTST ensemble; MSE no more than0.5% worse; >=2/3 seeds improve raw; both dev halves MAE no more than0.5% worse",
                "selection_bias": "Seed42 and prior internal holdout have been observed. This is follow-up development replication, not independent blind validation. No per-seed alpha choice, seed subset selection, or test tuning.",
                "holdout_rule": "Only score one frozen calibrated candidate after all development gates pass; otherwise stop without rescoring holdout"}
    path = HERE / "results/protocol.json"
    if path.exists():
        if not args.resume:
            raise FileExistsError(path)
        assert read(path) == protocol
        write_json(HERE / f"results/resume_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.json", {"protocol_sha256": sha256_file(path)})
    else:
        write_json(path, protocol)
    train, dev, metadata = development_data(args.package_dir.resolve())
    assert metadata == read(original.HERE / "results/data_protocol.json")
    write_json(HERE / "results/data_protocol.json", metadata)
    for seed in (43, 44):
        trainer = load_trainer(seed)
        trial_args = argparse.Namespace(seed=seed, resume=args.resume, epochs=settings["max_epochs"],
                                        patience=settings["patience"], batch_size=settings["batch_size"])
        for kind in KINDS:
            print(f"Replica seed={seed}, kind={kind}; original E4 training unchanged.", flush=True)
            trainer.fit(kind, train, dev, trial_args)
    del train
    gc.collect()
    result = development(dev, settings["batch_size"])
    assert source_hashes() == hashes
    args.batch_size = settings["batch_size"]
    evaluate_holdout(args, metadata, result)
    assert source_hashes() == hashes
    from .report import write_report
    write_report()
    print("Fixed-alpha replication complete. Results remain local.", flush=True)


if __name__ == "__main__":
    main()
