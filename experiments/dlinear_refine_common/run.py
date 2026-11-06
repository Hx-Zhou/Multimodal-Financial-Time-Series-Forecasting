"""Two predeclared regularization comparisons, development-only, conditional replication."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import time
import numpy as np
import torch

from experiments.dlinear_modal_common.run import (
    ROOT, OLD, DIRECTORIES, model_type, read_json, write_json, sha256_file, source_hashes as old_sources,
    verify_bases, base_path, predict, shuffled_rows, metrics, seed_statistics,
)
from experiments.dlinear_modal_common.data import load_store, add_reports, batch
from .audit import REFINED, PREFIXES, norms
from .optimizer import make_optimizer

HERE = Path(__file__).resolve().parent
STEPS = {"e8": ("F1", "F2"), "e9": ("M1", "M2"), "e10": ("S1", "S2")}
FAMILIES = {"e8": ("e6", "e8"), "e9": ("e6", "e7", "e8", "e9"), "e10": ("e6", "e7", "e10")}
SETTINGS = {"max_epochs": 20, "patience": 6, "min_delta": 1e-5, "batch_size": 1024,
            "learning_rate": 0.001, "weight_decay": 0.01, "loss": "MAE + 0.2*MSE", "clip_norm": 1.0,
            "screen_seed": 42, "replication_seeds": [43, 44], "mae_relative_improvement": 0.001,
            "max_mse_relative_degradation": 0.005, "content_mae_delta": 1e-5,
            "max_half_mae_relative_degradation": 0.005, "holdout_evaluated": False}


def home(stage):
    return ROOT / "experiments" / REFINED[stage]


def sources():
    result = old_sources()
    paths = list(HERE.glob("*.py"))
    for name in REFINED.values():
        paths += list((ROOT / "experiments" / name).glob("*.py"))
        paths.append(ROOT / "experiments" / name / "EXPERIMENT_PLAN.md")
    result.update({p.relative_to(ROOT).as_posix(): sha256_file(p) for p in paths})
    return result


def old_record(family, seed):
    if family in ("e6", "e7"):
        return read_json(OLD / f"results/{family}_stationary_seed{seed}.json")
    return read_json(ROOT / "experiments" / DIRECTORIES[family] / f"results/seed{seed}.json")


def old_predictions(family, seed):
    if family in ("e6", "e7"):
        return np.load(OLD / f"outputs/{family}_stationary_seed{seed}/dev_predictions.npy")
    return np.load(ROOT / "experiments" / DIRECTORIES[family] / f"outputs/seed{seed}/dev_predictions.npy")


def old_checkpoints():
    hashes = {}
    for family in ("e6", "e7", "e8", "e9", "e10"):
        for seed in (42, 43, 44):
            path = (OLD / f"outputs/{family}_stationary_seed{seed}/model.pt" if family in ("e6", "e7")
                    else ROOT / "experiments" / DIRECTORIES[family] / f"outputs/seed{seed}/model.pt")
            digest = sha256_file(path)
            assert digest == old_record(family, seed)["checkpoint_sha256"]
            hashes[f"{family}/{seed}"] = digest
    return hashes


def save_model(path, stage, step, seed, policy, model):
    torch.save({"stage": stage, "step": step, "seed": seed, "policy": policy,
                "settings": SETTINGS, "state_dict": model.state_dict()}, path)


def load_model(stage, step, seed):
    record = read_json(home(stage) / f"results/{step}/seed{seed}.json")
    path = home(stage) / f"outputs/{step}/seed{seed}/model.pt"
    assert sha256_file(path) == record["checkpoint_sha256"]
    saved = torch.load(path, weights_only=True, map_location="cpu")
    assert (saved["stage"], saved["step"], saved["seed"], saved["settings"]) == (stage, step, seed, SETTINGS)
    model = model_type(stage)()
    model.load_state_dict(saved["state_dict"], strict=True)
    return model.eval()


def half_scores(prediction, samples):
    dates = np.unique(samples.dates)
    boundary = int(dates[len(dates)//2])
    return {"boundary_target_start": boundary,
            "early": metrics(prediction[samples.dates < boundary], samples.y[samples.dates < boundary]),
            "late": metrics(prediction[samples.dates >= boundary], samples.y[samples.dates >= boundary])}


def run_diagnostics(model, store, samples, output):
    options = {}
    if model.with_reports:
        options.update(report_values_zero={"zero_report_values": True},
                       reports_shuffled={"report_shuffle": shuffled_rows(samples)},
                       report_lag270={"lag": 270})
    if model.with_news:
        options.update(news_semantics_zero={"zero_semantics": True},
                       news_shuffled={"news_shuffle": shuffled_rows(samples)})
    main_prediction = np.load(output / "dev_predictions.npy")
    result = {}
    for name, kwargs in options.items():
        changed = predict(model, store, samples, **kwargs)
        np.save(output / f"{name}.npy", changed)
        result[name] = {**metrics(changed, samples.y),
                        "max_abs_prediction_change": float(np.max(np.abs(changed-main_prediction))),
                        "mean_abs_prediction_change": float(np.mean(np.abs(changed-main_prediction)))}
    return result


def train_one(stage, step, policy, seed, store, resume):
    out = home(stage) / f"outputs/{step}/seed{seed}"
    record = home(stage) / f"results/{step}/seed{seed}.json"
    if record.exists():
        if not resume:
            raise FileExistsError(record)
        load_model(stage, step, seed)
        return read_json(record)
    torch.manual_seed(seed)
    model = model_type(stage)()
    model.load_market(torch.load(base_path(seed), weights_only=True, map_location="cpu"))
    initial = {key: value.clone() for key, value in model.state_dict().items()}
    optimizer, groups = make_optimizer(model, policy)
    rng = np.random.default_rng(seed)
    trainable = [p for p in model.parameters() if p.requires_grad]
    elapsed = 0.0
    if out.exists():
        if not resume:
            raise FileExistsError(out)
        state = torch.load(out / "training_state.pt", weights_only=True, map_location="cpu")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"])
        rng.bit_generator.state = state["numpy_rng"]
        history, best, best_epoch, stale, elapsed = (state[k] for k in ("history", "best", "best_epoch", "stale", "elapsed"))
    else:
        out.mkdir(parents=True)
        initial_score = metrics(predict(model, store, store.dev), store.dev.y)
        np.testing.assert_allclose(initial_score["mae"], old_record("e6", seed)["development"]["mae"], atol=1e-9, rtol=0)
        history = [{"epoch": 0, "development": initial_score}]
        best, best_epoch, stale = initial_score["mae"], 0, 0
        save_model(out / "model.pt", stage, step, seed, policy, model)
    started = time.perf_counter()

    def save_state():
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "torch_rng": torch.get_rng_state(), "numpy_rng": rng.bit_generator.state,
                    "history": history, "best": best, "best_epoch": best_epoch, "stale": stale,
                    "elapsed": elapsed+time.perf_counter()-started}, out / "training_state.pt")
        write_json(out / "history.json", history)

    save_state()
    for epoch in range(history[-1]["epoch"]+1, SETTINGS["max_epochs"]+1):
        if stale >= SETTINGS["patience"]:
            break
        tick = time.perf_counter()
        model.train()
        order = rng.permutation(len(store.train.y))
        total = 0.0
        for left in range(0, len(order), SETTINGS["batch_size"]):
            rows = order[left:left+SETTINGS["batch_size"]]
            item = batch(store, store.train, rows, model.with_news, model.with_reports)
            error = model(item)-torch.from_numpy(store.train.y[rows])
            loss = error.abs().mean()+0.2*error.square().mean()
            if not torch.isfinite(loss):
                raise ValueError("nonfinite loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            total += loss.item()*len(rows)
        score = metrics(predict(model, store, store.dev), store.dev.y)
        if score["mae"] < best-SETTINGS["min_delta"]:
            best, best_epoch, stale = score["mae"], epoch, 0
            save_model(out / "model.pt", stage, step, seed, policy, model)
        else:
            stale += 1
        history.append({"epoch": epoch, "train_loss": total/len(order), "development": score,
                        "seconds": time.perf_counter()-tick,
                        "module_norms": {p: norms(model.state_dict(), p) for p in PREFIXES[stage]},
                        "gates": {k: float(torch.sigmoid(v)) for k, v in model.state_dict().items() if k.endswith("gate")}})
        save_state()
        print(f"{stage}/{step} seed={seed} epoch={epoch:02d} dev_MAE={score['mae']:.9f} best={best_epoch} seconds={history[-1]['seconds']:.1f}", flush=True)
    model.load_state_dict(torch.load(out / "model.pt", weights_only=True, map_location="cpu")["state_dict"])
    for key, value in model.market.state_dict().items():
        torch.testing.assert_close(value, initial["market."+key], rtol=0, atol=0)
    prediction = predict(model, store, store.dev)
    np.save(out / "dev_predictions.npy", prediction)
    score = metrics(prediction, store.dev.y)
    np.testing.assert_allclose(score["mae"], best, rtol=0, atol=1e-9)
    result = {"stage": stage, "step": step, "policy": policy, "seed": seed, "best_epoch": best_epoch,
              "development": score, "halves": half_scores(prediction, store.dev), "history": history,
              "seconds": elapsed+time.perf_counter()-started, "optimizer_groups": groups,
              "checkpoint_sha256": sha256_file(out / "model.pt"), "market_sha256": sha256_file(base_path(seed)),
              "module_norm_ratios": {p: norms(model.state_dict(), p)["l2_norm"]/norms(initial, p)["l2_norm"] for p in PREFIXES[stage]},
              "holdout_evaluated": False}
    result["diagnostics"] = run_diagnostics(model, store, store.dev, out)
    write_json(record, result)
    return result


def content_supported(result):
    score = result["development"]
    content = []
    for a, b in (("report_values_zero", "reports_shuffled"), ("news_semantics_zero", "news_shuffled")):
        if a in result["diagnostics"]:
            content.append(all(result["diagnostics"][key]["mae"]-score["mae"] >= SETTINGS["content_mae_delta"] for key in (a, b)))
    return any(content)


def qualifies(result, reference, reference_halves):
    score = result["development"]
    checks = {"mae": score["mae"] <= reference["mae"]*(1-SETTINGS["mae_relative_improvement"]),
              "mse": score["mse"] <= reference["mse"]*(1+SETTINGS["max_mse_relative_degradation"]),
              "content_increment": content_supported(result),
              "both_dev_periods": all(result["halves"][key]["mae"] <= reference_halves[key]["mae"]*1.005 for key in ("early", "late"))}
    return {"passed": all(checks.values()), "checks": checks}


def finish_stage(stage, store, resume):
    reference_family = min(FAMILIES[stage], key=lambda f: old_record(f, 42)["development"]["mae"])
    reference_prediction = old_predictions(reference_family, 42)
    reference = metrics(reference_prediction, store.dev.y)
    np.testing.assert_allclose(reference["mae"], old_record(reference_family, 42)["development"]["mae"], atol=1e-9, rtol=0)
    reference_halves = half_scores(reference_prediction, store.dev)
    rows, decisions = [], {}
    for step, policy in zip(STEPS[stage], ("all", "matrix_only")):
        row = train_one(stage, step, policy, 42, store, resume)
        rows.append(row)
        decisions[step] = qualifies(row, reference, reference_halves)
        print(f"{stage}/{step} screen against {reference_family}: {decisions[step]}", flush=True)
    eligible = [row for row in rows if decisions[row["step"]]["passed"]]
    selected = None
    if eligible:
        selected = eligible[0]
        for row in eligible[1:]:
            if row["development"]["mae"] < selected["development"]["mae"]-1e-5:
                selected = row
    summary = {"stage": stage, "reference_family_seed42": reference_family, "reference_development": reference,
               "reference_halves": reference_halves, "screen_decisions": decisions,
               "selected_for_replication": None if selected is None else selected["step"],
               "holdout_evaluated": False, "new_trial_count_seed42": 2}
    if selected:
        runs = [selected]+[train_one(stage, selected["step"], selected["policy"], s, store, resume) for s in (43, 44)]
        predictions = [np.load(home(stage) / f"outputs/{selected['step']}/seed{s}/dev_predictions.npy") for s in (42, 43, 44)]
        average = np.mean(predictions, axis=0)
        old_averages = {f: np.mean([old_predictions(f, s) for s in (42, 43, 44)], axis=0) for f in FAMILIES[stage]}
        baseline_family = min(old_averages, key=lambda f: metrics(old_averages[f], store.dev.y)["mae"])
        baseline = metrics(old_averages[baseline_family], store.dev.y)
        np.save(home(stage) / f"outputs/{selected['step']}/dev_average.npy", average)
        ensemble_score = metrics(average, store.dev.y)
        improvements = [run["development"]["mae"] < old_record(baseline_family, run["seed"])["development"]["mae"] for run in runs]
        content_seeds = sum(content_supported(run) for run in runs)
        avg_halves, base_halves = half_scores(average, store.dev), half_scores(old_averages[baseline_family], store.dev)
        summary["replication"] = {"equal_average": ensemble_score, "seed_statistics": seed_statistics([r["development"] for r in runs]),
                                  "reference_family": baseline_family, "reference_equal_average": baseline,
                                  "seeds_improved": sum(improvements), "content_supported_seeds": content_seeds, "halves": avg_halves,
                                  "passed": ensemble_score["mae"] <= baseline["mae"]*0.999 and ensemble_score["mse"] <= baseline["mse"]*1.005
                                  and sum(improvements) >= 2 and content_seeds >= 2
                                  and all(avg_halves[k]["mae"] <= base_halves[k]["mae"]*1.005 for k in ("early", "late"))}
    else:
        summary["stop_reason"] = "Neither predeclared regularization policy met the development performance, content and temporal-stability gates; do not add seeds or inspect holdout. Follow the experiment-specific next diagnostic stage."
    write_json(home(stage) / "results/regularization_summary.json", summary)
    return summary


def main(default_stage=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--stages", nargs="+", choices=tuple(REFINED), default=[default_stage] if default_stage else list(REFINED))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not torch.set_flush_denormal(True):
        raise RuntimeError("CPU flush-denormal mode unavailable")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    source_hashes, originals = sources(), old_checkpoints()
    verify_bases()
    for stage in args.stages:
        protocol = {"stage": stage, "settings": SETTINGS, "steps": dict(zip(STEPS[stage], ("all", "matrix_only"))),
                    "source_sha256": source_hashes, "original_checkpoint_sha256": originals,
                    "screen_comparators": {f: old_record(f, 42)["development"] for f in FAMILIES[stage]},
                    "change": "Same architecture and inputs; compare AdamW all-parameter decay, then matrix-only decay. Bounded 20-epoch budget.",
                    "selection": "seed42 first; all four declared gates required; if multiple qualify choose lower dev MAE, first step wins ties <=1e-5; repeat only the selected policy for seeds43/44",
                    "holdout_policy": "This series follows previously revealed holdout results and cannot be considered blind. No holdout samples/scoring are constructed by this runner; all selection is development-only.",
                    "environment": {"torch": str(torch.__version__), "numpy": np.__version__, "device": "cpu", "threads": 4}}
        path = home(stage) / "results/regularization_protocol.json"
        if path.exists():
            if not args.resume:
                raise FileExistsError(path)
            assert read_json(path) == protocol
            write_json(home(stage) / f"results/resume_{time.time_ns()}.json", {"utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": sha256_file(path)})
        else:
            assert (home(stage) / "results/initial_audit.json").exists()
            write_json(path, protocol)
    print("Preparing unchanged data; this runner does not evaluate holdout.", flush=True)
    cache = HERE / f"outputs/cache_{time.time_ns()}"
    store = load_store(args.package_dir.resolve(), cache, with_news=True)
    add_reports(store, args.package_dir.resolve(), cache)
    metadata = {"market_news": store.metadata, "reports": store.report_metadata}
    assert metadata == read_json(ROOT / "experiments/e8_dlinear_reports/results/data_protocol.json")
    for stage in args.stages:
        path = home(stage) / "results/data_protocol.json"
        if path.exists():
            assert read_json(path) == metadata
        else:
            write_json(path, metadata)
    for stage in args.stages:
        path = home(stage) / "results/regularization_summary.json"
        if path.exists() and args.resume:
            print(f"Preserving completed {stage}", flush=True)
        else:
            finish_stage(stage, store, args.resume)
    assert sources() == source_hashes and old_checkpoints() == originals
    print("Completed declared development-only regularization trials; prior experiments and holdout unchanged.", flush=True)


if __name__ == "__main__":
    main()
