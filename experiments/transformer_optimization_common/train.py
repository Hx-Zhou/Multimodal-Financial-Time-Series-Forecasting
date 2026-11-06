"""One trainer, calibrated-development early stopping, bounded configurations."""
from pathlib import Path
import time
import numpy as np
import torch
from experiments.e1_dlinear.run import metrics, write_json
from experiments.e1_dlinear.data import sha256_file
from experiments.e4_patchtst_capacity.model import ScaleAwarePatchTST, ScaleAwareLinear
from experiments.e5_itransformer.model import ITransformer, ScaleAwareMarketLinear


def build(spec):
    return {"patchtst": ScaleAwarePatchTST, "linear_close": ScaleAwareLinear,
            "itransformer": ITransformer, "linear_market": ScaleAwareMarketLinear}[spec["kind"]](**spec["config"])


@torch.inference_mode()
def predict(model, adapter, samples, mode, alpha, batch_size=512):
    model.eval()
    out = []
    for left in range(0, len(samples.y), batch_size):
        rows = np.arange(left, min(left+batch_size, len(samples.y)))
        batch = adapter.batch(samples, rows, mode)
        batch["alpha"] = alpha
        out.append(model(batch).numpy())
    result = np.concatenate(out)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite predictions")
    return result


def read_json(path):
    import json
    return json.loads(path.read_text(encoding="utf-8"))


def load_model(path):
    saved = torch.load(path, weights_only=True, map_location="cpu")
    model = build(saved["spec"])
    model.load_state_dict(saved["state_dict"], strict=True)
    return model.eval(), saved


def fit(spec, seed, adapter, home, settings, resume=False):
    out, record = home / f"outputs/{spec['id']}/seed{seed}", home / f"results/{spec['id']}/seed{seed}.json"
    if record.exists():
        if not resume:
            raise FileExistsError(record)
        result = read_json(record)
        assert result["spec"] == spec and result["settings"] == settings
        assert sha256_file(out / "selected.pt") == result["checkpoint_sha256"]
        return result
    torch.manual_seed(seed)
    model = build(spec)
    decay, exempt = [], []
    for p in model.parameters():
        (decay if p.ndim >= 2 else exempt).append(p)
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": .01}, {"params": exempt, "weight_decay": 0.}], lr=.0003)
    rng = np.random.default_rng(seed)
    elapsed = 0.

    def save(epoch):
        torch.save({"spec": spec, "seed": seed, "epoch": epoch, "alpha": settings["alpha"],
                    "state_dict": model.state_dict()}, out / "selected.pt")

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
        initial = predict(model, adapter, adapter.dev, spec["mode"], settings["alpha"], settings["batch_size"])
        np.testing.assert_array_equal(initial, np.repeat(adapter.anchor(adapter.dev), 24, axis=1))
        score = metrics(initial, adapter.dev.y)
        history, best, best_epoch, stale = [{"epoch": 0, "development": score}], score["mae"], 0, 0
        save(0)
    started = time.perf_counter()

    def persist():
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
                    "numpy_rng": rng.bit_generator.state, "history": history, "best": best, "best_epoch": best_epoch,
                    "stale": stale, "elapsed": elapsed+time.perf_counter()-started}, out / "training_state.pt")
        write_json(out / "history.json", history)

    persist()
    for epoch in range(history[-1]["epoch"]+1, settings["max_epochs"]+1):
        if stale >= settings["patience"]:
            break
        tick = time.perf_counter()
        model.train()
        order, total = rng.permutation(len(adapter.train.y)), 0.
        for left in range(0, len(order), settings["batch_size"]):
            rows = order[left:left+settings["batch_size"]]
            batch = adapter.batch(adapter.train, rows, spec["mode"])
            # Raw training loss is unchanged. alpha is only used for checkpoint selection/inference.
            optimizer.zero_grad(set_to_none=True)
            error = model(batch)-torch.from_numpy(adapter.train.y[rows])
            loss = error.abs().mean()+.2*error.square().mean()
            if not torch.isfinite(loss):
                raise ValueError("nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += loss.item()*len(rows)
            number = left//settings["batch_size"]+1
            if spec["kind"] in ("patchtst", "itransformer") and number % 300 == 0:
                print(f"{spec['id']} seed={seed} epoch={epoch} batch={number}/{int(np.ceil(len(order)/settings['batch_size']))}", flush=True)
        score = metrics(predict(model, adapter, adapter.dev, spec["mode"], settings["alpha"], settings["batch_size"]), adapter.dev.y)
        if score["mae"] < best-settings["min_delta"]:
            best, best_epoch, stale = score["mae"], epoch, 0
            save(epoch)
        else:
            stale += 1
        history.append({"epoch": epoch, "train_loss": total/len(order), "development": score, "seconds": time.perf_counter()-tick})
        persist()
        print(f"{spec['id']} seed={seed} epoch={epoch:02d} calibrated_dev_MAE={score['mae']:.9f} MSE={score['mse']:.9f} best={best_epoch} seconds={history[-1]['seconds']:.1f}", flush=True)
    model, saved = load_model(out / "selected.pt")
    values = predict(model, adapter, adapter.dev, spec["mode"], settings["alpha"], settings["batch_size"])
    score = metrics(values, adapter.dev.y)
    np.testing.assert_allclose(score["mae"], best, rtol=0, atol=1e-9)
    np.save(out / "dev_predictions.npy", values)
    result = {"spec": spec, "settings": settings, "seed": seed, "best_epoch": best_epoch,
              "development": score, "history": history, "parameters": sum(p.numel() for p in model.parameters()),
              "seconds": elapsed+time.perf_counter()-started, "checkpoint_sha256": sha256_file(out / "selected.pt"),
              "holdout_evaluated": False}
    write_json(record, result)
    return result
