"""Inspect original best/final weights without accessing holdout records."""
from pathlib import Path
import torch
from experiments.dlinear_modal_common.run import ROOT, DIRECTORIES, model_type, read_json, write_json

REFINED = {stage: name + "_refine" for stage, name in DIRECTORIES.items()}
PREFIXES = {"e8": ("reports.encoder.",), "e9": ("news.mlp.", "reports.encoder."),
            "e10": ("news_mlp.", "query.", "coverage.", "value.")}


def norms(state, prefix):
    tensors = [value.double().flatten() for name, value in state.items() if name.startswith(prefix)]
    all_values = torch.cat(tensors)
    return {"l2_norm": float(all_values.norm()), "max_abs": float(all_values.abs().max())}


def audit(stage):
    old = ROOT / "experiments" / DIRECTORIES[stage]
    rows = []
    for seed in (42, 43, 44):
        torch.manual_seed(seed)
        initial = model_type(stage)().state_dict()
        best = torch.load(old / f"outputs/seed{seed}/model.pt", weights_only=True, map_location="cpu")["state_dict"]
        final = torch.load(old / f"outputs/seed{seed}/training_state.pt", weights_only=True, map_location="cpu")["model"]
        record = read_json(old / f"results/seed{seed}.json")
        row = {"seed": seed, "best_epoch": record["best_epoch"], "modules": {}, "gates": {}}
        for prefix in PREFIXES[stage]:
            a, b, c = (norms(state, prefix) for state in (initial, best, final))
            row["modules"][prefix] = {"initial": a, "best": b, "final": c,
                                       "best_to_initial_norm_ratio": b["l2_norm"]/a["l2_norm"],
                                       "final_to_initial_norm_ratio": c["l2_norm"]/a["l2_norm"]}
        for name in initial:
            if name.endswith("gate"):
                row["gates"][name] = {label: float(torch.sigmoid(state[name]))
                                      for label, state in (("initial", initial), ("best", best), ("final", final))}
        rows.append(row)
    result = {"stage": stage, "holdout_read": False, "runs": rows,
              "hypothesis": "Coupled Adam L2 shrank nonlinear encoders and penalized negative gate logits toward zero; test decoupled AdamW with bias/gates exempt.",
              "causal_limit": "Weight collapse is observed; relative importance of regularization vs data gradients must be tested by the new development-only run."}
    path = ROOT / "experiments" / REFINED[stage] / "results/initial_audit.json"
    if path.exists():
        assert read_json(path) == result
    else:
        write_json(path, result)
    return result


if __name__ == "__main__":
    for stage in DIRECTORIES:
        result = audit(stage)
        for row in result["runs"]:
            print(stage, row["seed"], "best epoch", row["best_epoch"],
                  {key: value["final_to_initial_norm_ratio"] for key, value in row["modules"].items()}, row["gates"])
