"""Run four bounded screening configurations, replicate one winner, then score once."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import shutil
import numpy as np
import torch
from experiments.e1_dlinear.run import ROOT, metrics, write_json
from experiments.e1_dlinear.data import sha256_file
from experiments.e4_patchtst.run import CONFIG as PATCH_CONFIG
from .data import load_adapter, close_fingerprint, holdout_samples
from .train import fit, predict, load_model, read_json

HERE = Path(__file__).resolve().parent
PATCH = ROOT / "experiments/e4_patchtst_capacity"
ITR = ROOT / "experiments/e5_itransformer"
SETTINGS = {"max_epochs": 8, "patience": 3, "batch_size": 512, "threads": 4, "alpha": .5, "min_delta": 1e-5,
            "loss": "MAE+0.2*MSE", "optimizer": "AdamW lr=.0003, ndim>=2 wd=.01, other parameters wd=0", "clip_norm": 1.0}


def home(spec):
    return ITR if spec["mode"] == "market" else PATCH


def path_for(spec, seed, filename="selected.pt"):
    return home(spec) / f"outputs/{spec['id']}/seed{seed}/{filename}"


def hashes():
    files = []
    for directory in (HERE, PATCH, ITR):
        files += list(directory.glob("*.py"))+[directory / "README.md"]
    files += [HERE / "PLAN.md", ROOT / "hs300_dataset.py", ROOT / "experiments/validation_split.py",
              ROOT / "experiments/e4_patchtst/model.py", ROOT / "experiments/e1_dlinear/e6_plus/data.py",
              ROOT / "experiments/e1_dlinear/e6_plus/model.py", ROOT / "experiments/e1_dlinear/data.py",
              ROOT / "experiments/e1_dlinear/run.py", ROOT / "experiments/e4_patchtst/run.py",
              ROOT / "experiments/e4_patchtst/data.py", ROOT / "experiments/e4_patchtst/results/data_protocol.json",
              ROOT / "experiments/e1_dlinear/outputs/calibration/dev_predictions.npy",
              ROOT / "experiments/e4_patchtst_shrinkage/outputs/dev_patchtst_alpha0.5.npy"]
    return {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in files}


def half_scores(value, samples):
    dates = np.unique(samples.dates)
    cutoff = int(dates[len(dates)//2])
    return {"cutoff_target_start": cutoff,
            "early": metrics(value[samples.dates < cutoff], samples.y[samples.dates < cutoff]),
            "late": metrics(value[samples.dates >= cutoff], samples.y[samples.dates >= cutoff])}


def references(adapter, controls):
    values = {
        "enhanced_e1": np.load(ROOT / "experiments/e1_dlinear/outputs/calibration/dev_predictions.npy"),
        "prior_calibrated_patch_ensemble": np.load(ROOT / "experiments/e4_patchtst_shrinkage/outputs/dev_patchtst_alpha0.5.npy"),
    }
    values.update(controls)
    assert all(v.shape == adapter.dev.y.shape for v in values.values())
    scores = {name: metrics(v, adapter.dev.y) for name, v in values.items()}
    strongest = min(scores, key=lambda name: scores[name]["mae"])
    return values, scores, strongest


def screening_gate(score, reference, e1):
    # A loose first-seed screen allows averaging gains; formal gates follow replication.
    return {"mae_within_half_percent": score["mae"] <= reference["mae"]*1.005,
            "mse_within_half_percent_of_e1": score["mse"] <= e1["mse"]*1.005}


def choose_spec(rows):
    selected = rows[0]
    for row in rows[1:]:
        gap = row["development"]["mae"]-selected["development"]["mae"]
        if gap < -1e-5 or (abs(gap) <= 1e-5 and row["parameters"] < selected["parameters"]):
            selected = row
    return selected


def run_screen(adapter, resume):
    close_control = dict(id="L_close", kind="linear_close", mode="close", config={}, parent=None)
    market_control = dict(id="L_market", kind="linear_market", mode="market", config={}, parent=None)
    fit(close_control, 42, adapter, home(close_control), SETTINGS, resume)
    p0 = dict(id="P0_selector", kind="patchtst", mode="close", config=deepcopy(PATCH_CONFIG), parent="historical_e4")
    row0 = fit(p0, 42, adapter, PATCH, SETTINGS, resume)
    p1 = deepcopy(p0);p1.update(id="P1_width64", parent=p0["id"]);p1["config"]["d_model"] = 64
    row1 = fit(p1, 42, adapter, PATCH, SETTINGS, resume)
    parent = choose_spec([row0, row1])
    p2 = deepcopy(parent["spec"]);p2.update(id="P2_dropout", parent=parent["spec"]["id"]);p2["config"]["dropout"] = .3
    row2 = fit(p2, 42, adapter, PATCH, SETTINGS, resume)
    fit(market_control, 42, adapter, ITR, SETTINGS, resume)
    i0 = dict(id="I0_variables", kind="itransformer", mode="market", parent="L_market",
              config=dict(d_model=64, nhead=4, layers=2, d_ff=256, dropout=.1))
    rowi = fit(i0, 42, adapter, ITR, SETTINGS, resume)
    controls = {s["id"]: np.load(path_for(s, 42, "dev_predictions.npy")) for s in (close_control, market_control)}
    _, refs, strongest = references(adapter, controls)
    rows, decisions, eligible = [row0, row1, row2, rowi], {}, []
    for row in rows:
        checks = screening_gate(row["development"], refs[strongest], refs["enhanced_e1"])
        passed = all(checks.values()) and row["best_epoch"] > 0
        decisions[row["spec"]["id"]] = {"eligible": passed, "checks": checks}
        if passed:
            eligible.append(row)
    selected = choose_spec(eligible) if eligible else None
    result = {"rows": [{k:r[k] for k in ("spec", "seed", "best_epoch", "parameters", "development", "checkpoint_sha256")} for r in rows],
              "references": refs, "strongest_reference": strongest, "decisions": decisions,
              "selected_spec": None if selected is None else selected["spec"], "holdout_evaluated": False,
              "control_spec": None if selected is None else (close_control if selected["spec"]["mode"] == "close" else market_control)}
    write_json(HERE / "results/screen.json", result)
    return result


def formal_gate(score, reference, e1, seed_wins, halves, ref_halves):
    checks = {"mae_improves_at_least_0p1_percent": score["mae"] <= reference["mae"]*.999,
              "mse_not_worse_than_e1": score["mse"] <= e1["mse"],
              "at_least_two_seed_wins_vs_matched_linear": seed_wins >= 2,
              "both_dev_periods": all(halves[k]["mae"] <= ref_halves[k]["mae"]*1.005 for k in ("early", "late"))}
    return {"passed": all(checks.values()), "checks": checks}


def replicate(adapter, screen, resume):
    spec, control = screen["selected_spec"], screen["control_spec"]
    if spec is None:
        result = {"passed": False, "reason": "No first-seed configuration passed the screening reserve gates", "holdout_evaluated": False}
        write_json(HERE / "results/replication.json", result)
        return result
    for seed in (43,44):
        fit(control, seed, adapter, home(control), SETTINGS, resume)
        fit(spec, seed, adapter, home(spec), SETTINGS, resume)
    values, records = {}, {}
    for label, trial in (("transformer", spec), ("matched_linear", control)):
        records[label] = [read_json(home(trial)/f"results/{trial['id']}/seed{s}.json") for s in (42,43,44)]
        values[label] = np.mean([np.load(path_for(trial,s,"dev_predictions.npy")) for s in (42,43,44)], axis=0)
        np.save(HERE / f"outputs/dev_{label}_average.npy", values[label])
    ref_values, refs, strongest = references(adapter, {"matched_linear_average": values["matched_linear"]})
    score = metrics(values["transformer"], adapter.dev.y)
    wins = sum(a["development"]["mae"] < b["development"]["mae"] for a,b in zip(records["transformer"], records["matched_linear"]))
    halves = half_scores(values["transformer"], adapter.dev)
    ref_halves = half_scores(ref_values[strongest], adapter.dev)
    decision = formal_gate(score, refs[strongest], refs["enhanced_e1"], wins, halves, ref_halves)
    result = {"spec": spec, "control_spec": control, "development": score, "records": records,
              "references": refs, "strongest_reference": strongest, "halves": halves, "reference_halves": ref_halves,
              "matched_linear_seed_wins": wins, "decision": decision, "passed": decision["passed"], "holdout_evaluated": False}
    write_json(HERE / "results/replication.json", result)
    return result


def evaluate(adapter, package, result, source_hashes):
    if not result["passed"]:
        print("Formal development gates failed; stopping without holdout scoring.", flush=True)
        return
    specs = {"transformer": result["spec"], "matched_linear": result["control_spec"]}
    checkpoint_hashes = {name: {str(s):sha256_file(path_for(spec,s)) for s in (42,43,44)} for name,spec in specs.items()}
    frozen = {"specs": specs, "checkpoint_sha256": checkpoint_hashes, "alpha": SETTINGS["alpha"],
              "development_sha256":sha256_file(HERE / "results/replication.json"), "sources": source_hashes,
              "policy":"One frozen winner only; all three seeds equally averaged; known internal holdout, no changes after scores"}
    protocol_path = HERE / "results/holdout_protocol.json"
    if protocol_path.exists():assert read_json(protocol_path)==frozen
    else:write_json(protocol_path, frozen)
    samples = holdout_samples(adapter, package)
    scores = {}
    for name,spec in specs.items():
        values=[]
        for seed in (42,43,44):
            path = path_for(spec,seed)
            assert sha256_file(path)==checkpoint_hashes[name][str(seed)]
            model,_=load_model(path)
            value=predict(model,adapter,samples,spec["mode"],SETTINGS["alpha"],SETTINGS["batch_size"])
            values.append(value);scores[f"{name}_seed{seed}"]=metrics(value,samples.y)
        mean=np.mean(values,axis=0)
        score=metrics(mean,samples.y)
        err=torch.from_numpy(mean).double()-torch.from_numpy(samples.y).double()
        np.testing.assert_allclose([score["mae"],score["mse"]],[err.abs().mean(),err.square().mean()],rtol=1e-12,atol=1e-12)
        scores[name]=score;np.save(HERE/f"outputs/holdout_{name}.npy",mean)
        print(f"holdout {name} MAE={score['mae']:.9f} MSE={score['mse']:.9f}",flush=True)
    for name,spec in specs.items():
        for seed in (42,43,44):assert sha256_file(path_for(spec,seed))==checkpoint_hashes[name][str(seed)]
    write_json(HERE / "results/holdout.json", {"scores":scores,"holdout_evaluated":True,"selection_after_holdout":False,"samples":len(samples.y)})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir",type=Path,required=True)
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    if shutil.disk_usage(HERE).free<512*1024**2:raise RuntimeError("Insufficient disk reserve")
    torch.set_flush_denormal(True);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.mha.set_fastpath_enabled(False)
    source=hashes()
    protocol={"settings":SETTINGS,"sources":source,"budget":"Four seed42 Transformer configurations; at most one winner replicated for seeds43/44; two cheap matched linear controls",
              "sequence":"P0 calibrated selector baseline -> P1 width64 only -> P2 dropout0.3 only on better P0/P1 -> I0 seven-variable iTransformer64",
              "screen":"MAE within0.5% of strongest reference and MSE within0.5% of E1, selected epoch>0; lowest MAE wins, ties<=1e-5 prefer fewer parameters",
              "formal":"3-seed MAE >=0.1% improvement over strongest E1/prior calibrated PatchTST/matched linear; MSE<=E1;>=2/3 paired wins; both dev halves <=0.5% worse",
              "limits":"Development and internal holdout have been observed before; no blind or significance claim. Only one qualified final candidate is scored on holdout.",
              "inference":"Checkpoint stores RAW weights and alpha metadata; batch alpha=0.5 applied once at inference. Training always alpha=1."}
    file=HERE / "results/protocol.json"
    if file.exists():
        if not args.resume:raise FileExistsError(file)
        assert read_json(file)==protocol
        write_json(HERE/f"results/resume_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.json",{"protocol_sha256":sha256_file(file)})
    else:write_json(file,protocol)
    print("Preparing shared numerical tables; no news/report embedding cache...",flush=True)
    adapter=load_adapter(args.package_dir.resolve())
    old=read_json(ROOT / "experiments/e4_patchtst/results/data_protocol.json")
    actual={"train_close_sha256":close_fingerprint(adapter.store,adapter.train),"dev_close_sha256":close_fingerprint(adapter.store,adapter.dev)}
    assert all(value==old[key] for key,value in actual.items())
    write_json(HERE / "results/data_protocol.json",{"numerical":adapter.store.metadata,"close_fingerprints":actual})
    screen_file=HERE / "results/screen.json"
    screen=read_json(screen_file) if args.resume and screen_file.exists() else run_screen(adapter,args.resume)
    replication_file=HERE / "results/replication.json"
    result=read_json(replication_file) if args.resume and replication_file.exists() else replicate(adapter,screen,args.resume)
    assert hashes()==source
    if not (HERE / "results/holdout.json").exists():evaluate(adapter,args.package_dir.resolve(),result,source)
    from .report import write_reports
    write_reports()
    assert hashes()==source
    print("Bounded Transformer plan completed; local results saved.",flush=True)


if __name__=="__main__":main()
