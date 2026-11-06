"""One fixed PatchTST configuration with a matched DLinear control; no sweep."""
import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from experiments.e1_dlinear.run import ROOT, metrics, write_json
from experiments.e1_dlinear.data import sha256_file
from experiments.e1_dlinear.model import build_model as build_e1
from .data import development_data, holdout_data
from .model import build_model

HERE = Path(__file__).resolve().parent
CONFIG = dict(patch_len=16, stride=8, d_model=128, nhead=4, layers=2, d_ff=256, dropout=0.1)
KINDS = ("dlinear_control", "patchtst")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def source_hashes():
    files = list(HERE.glob("*.py")) + [ROOT / "hs300_dataset.py", ROOT / "experiments/validation_split.py",
                                      HERE.parent / "e1_dlinear/data.py", HERE.parent / "e1_dlinear/model.py",
                                      HERE.parent / "e1_dlinear/run.py", HERE / "README.md"]
    return {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in files}


@torch.inference_mode()
def predict(model, x, batch_size):
    model.eval()
    result = torch.cat([model({"close": values}) for values in x.split(batch_size)]).numpy()
    if not np.isfinite(result).all():
        raise ValueError("nonfinite predictions")
    return result


def save_model(path, kind, seed, epoch, model):
    torch.save({"kind": kind, "config": CONFIG, "seed": seed, "epoch": epoch,
                "state_dict": model.state_dict()}, path)


def load_model(path):
    saved = torch.load(path, weights_only=True, map_location="cpu")
    assert saved["config"] == CONFIG
    model = build_model(saved["kind"], CONFIG)
    model.load_state_dict(saved["state_dict"], strict=True)
    return model.eval(), saved


def fit(kind, train, dev, args):
    out, record = HERE / f"outputs/{kind}", HERE / f"results/{kind}.json"
    if record.exists():
        if not args.resume:
            raise FileExistsError(record)
        result = read(record)
        for name, digest in result["checkpoint_sha256"].items():
            assert sha256_file(out / name) == digest
        print(f"Preserving completed {kind}", flush=True)
        return result
    torch.manual_seed(args.seed)
    model = build_model(kind, CONFIG)
    decay, exempt = [], []
    for p in model.parameters():
        (decay if p.ndim >= 2 else exempt).append(p)
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01},
                                   {"params": exempt, "weight_decay": 0.0}], lr=0.0003)
    rng = np.random.default_rng(args.seed)
    elapsed = 0.0
    if out.exists():
        if not args.resume:
            raise FileExistsError(out)
        state = torch.load(out / "training_state.pt", weights_only=True, map_location="cpu")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"])
        rng.bit_generator.state = state["numpy_rng"]
        history, best, best_epoch, trained_best, trained_epoch, stale, elapsed = (
            state[k] for k in ("history", "best", "best_epoch", "trained_best", "trained_epoch", "stale", "elapsed"))
    else:
        out.mkdir(parents=True)
        initial = predict(model, dev.x, args.batch_size)
        np.testing.assert_array_equal(initial, np.repeat(dev.x[:, -1:].numpy(), 24, axis=1))
        initial_score = metrics(initial, dev.y.numpy())
        history = [{"epoch": 0, "development": initial_score}]
        best, best_epoch, trained_best, trained_epoch, stale = initial_score["mae"], 0, float("inf"), None, 0
        save_model(out / "selected.pt", kind, args.seed, 0, model)
    started = time.perf_counter()

    def persist():
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
                    "numpy_rng": rng.bit_generator.state, "history": history, "best": best, "best_epoch": best_epoch,
                    "trained_best": trained_best, "trained_epoch": trained_epoch, "stale": stale,
                    "elapsed": elapsed+time.perf_counter()-started}, out / "training_state.pt")
        write_json(out / "history.json", history)

    persist()
    for epoch in range(history[-1]["epoch"]+1, args.epochs+1):
        if stale >= args.patience:
            break
        tick = time.perf_counter()
        model.train()
        order, total = rng.permutation(len(train.x)), 0.0
        for left in range(0, len(order), args.batch_size):
            rows = torch.from_numpy(order[left:left+args.batch_size])
            optimizer.zero_grad(set_to_none=True)
            error = model({"close": train.x[rows]})-train.y[rows]
            loss = error.abs().mean()+0.2*error.square().mean()
            if not torch.isfinite(loss):
                raise ValueError("nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()*len(rows)
            batch_number = left//args.batch_size+1
            if kind == "patchtst" and batch_number % 300 == 0:
                print(f"{kind} epoch={epoch} batch={batch_number}/{int(np.ceil(len(order)/args.batch_size))}", flush=True)
        score = metrics(predict(model, dev.x, args.batch_size), dev.y.numpy())
        if score["mae"] < trained_best-1e-5:
            trained_best, trained_epoch = score["mae"], epoch
            save_model(out / "best_trained.pt", kind, args.seed, epoch, model)
        if score["mae"] < best-1e-5:
            best, best_epoch, stale = score["mae"], epoch, 0
            save_model(out / "selected.pt", kind, args.seed, epoch, model)
        else:
            stale += 1
        history.append({"epoch": epoch, "train_loss": total/len(order), "development": score,
                        "seconds": time.perf_counter()-tick})
        persist()
        print(f"{kind} epoch={epoch:02d} dev_MAE={score['mae']:.9f} MSE={score['mse']:.9f} selected={best_epoch} seconds={history[-1]['seconds']:.1f}", flush=True)
    result = {"kind": kind, "seed": args.seed, "selected_epoch": best_epoch, "best_trained_epoch": trained_epoch,
              "parameters": sum(p.numel() for p in model.parameters()), "history": history,
              "seconds": elapsed+time.perf_counter()-started, "holdout_evaluated": False, "checkpoint_sha256": {}}
    for label, filename in (("selected", "selected.pt"), ("best_trained", "best_trained.pt")):
        chosen, saved = load_model(out / filename)
        values = predict(chosen, dev.x, args.batch_size)
        np.save(out / f"dev_{label}.npy", values)
        result[label+"_development"] = metrics(values, dev.y.numpy())
        expected = next(row["development"] for row in history if row["epoch"] == saved["epoch"])
        np.testing.assert_allclose(result[label+"_development"]["mae"], expected["mae"], atol=1e-9, rtol=0)
        result["checkpoint_sha256"][filename] = sha256_file(out / filename)
    write_json(record, result)
    return result


def evaluate_holdout(args, metadata, data_hashes, sources):
    frozen = {kind: read(HERE / f"results/{kind}.json")["checkpoint_sha256"] for kind in KINDS}
    e1_path = HERE.parent / "e1_dlinear/outputs/calibration/model.pt"
    previous = read(HERE.parent / "e1_dlinear/results/holdout/summary.json")
    assert sha256_file(e1_path) == previous["checkpoint_sha256"]
    protocol = {"checkpoints": frozen, "sources": sources, "seed": args.seed,
                "enhanced_e1_checkpoint_sha256": sha256_file(e1_path),
                "selection": "All checkpoints selected using development only before this scoring; epoch0 fallback and best trained epoch both predeclared; no choice after holdout",
                "status": "Previously observed internal temporal holdout, not a new blind or official hidden test",
                "metric": "standardized Close, uniform over sample/horizon positions; float64 reductions"}
    path = HERE / "results/holdout_protocol.json"
    if path.exists():
        assert read(path) == protocol
    else:
        write_json(path, protocol)
    if (HERE / "results/holdout.json").exists():
        return read(HERE / "results/holdout.json")
    data = holdout_data(args.package_dir.resolve(), metadata)
    target = data.y.numpy()
    values = {"e0": np.repeat(data.x[:, -1:].numpy(), 24, axis=1)}
    saved = torch.load(e1_path, weights_only=True, map_location="cpu")
    e1 = build_e1(saved["config"])
    e1.load_state_dict(saved["state_dict"])
    values["enhanced_e1"] = predict(e1, data.x, args.batch_size)
    for kind in KINDS:
        for label, filename in (("selected", "selected.pt"), ("best_trained", "best_trained.pt")):
            checkpoint = HERE / f"outputs/{kind}/{filename}"
            assert sha256_file(checkpoint) == frozen[kind][filename]
            model, _ = load_model(checkpoint)
            values[f"{kind}_{label}"] = predict(model, data.x, args.batch_size)
    scores = {}
    output = HERE / "outputs/holdout"
    output.mkdir(exist_ok=True)
    for name, prediction in values.items():
        score = metrics(prediction, target)
        error = torch.from_numpy(prediction).double()-torch.from_numpy(target).double()
        np.testing.assert_allclose([score["mae"], score["mse"]], [error.abs().mean(), error.square().mean()], rtol=1e-12, atol=1e-12)
        scores[name] = score
        np.save(output / f"{name}.npy", prediction)
        print(f"holdout {name}: MAE={score['mae']:.9f} MSE={score['mse']:.9f}", flush=True)
    for current, old in (("e0", "e0_holdout"), ("enhanced_e1", "e1_holdout")):
        for metric in ("mae", "mse"):
            np.testing.assert_allclose(scores[current][metric], previous[old][metric], atol=1e-8, rtol=0)
    for kind in KINDS:
        for filename, digest in frozen[kind].items():
            assert sha256_file(HERE / f"outputs/{kind}/{filename}") == digest
    result = {"scores": scores, "samples": len(target), "seed": args.seed, "holdout_evaluated": True,
              "selection_after_holdout": False, "baseline_reload_verified": True,
              "single_seed_screening": True, "data_sha256": data_hashes}
    write_json(HERE / "results/holdout.json", result)
    return result


def report():
    results = {kind: read(HERE / f"results/{kind}.json") for kind in KINDS}
    holdout = read(HERE / "results/holdout.json") if (HERE / "results/holdout.json").exists() else None
    original = read(HERE.parent / "e1_dlinear/results/calibration/summary.json")
    lines = ["# E4 PatchTST：固定配置首轮结果", "",
             "本轮使用原完整训练集及共享开发／内部留出划分，只比较一个 PatchTST 配置和同条件 DLinear。仅 seed42，未声称跨种子稳定或完整复现原论文。", "",
             "| 模型／权重 | epoch | 开发 MAE | 开发 MSE | 内部测试 MAE | 内部测试 MSE |", "|---|---:|---:|---:|---:|---:|"]
    refs = [("E0", None, original["e0_development"], "e0"),
            ("原增强 E1（三种子平均权重，历史参考）", None, original["averaged_model"]["development"], "enhanced_e1")]
    for label, epoch, d, key in refs:
        t = holdout["scores"][key] if holdout else None
        lines.append(f"| {label} | — | {d['mae']:.9f} | {d['mse']:.9f} | {f'{t["mae"]:.9f}' if t else '未评估'} | {f'{t["mse"]:.9f}' if t else '未评估'} |")
    for kind, record in results.items():
        for label, epoch_key in (("selected", "selected_epoch"), ("best_trained", "best_trained_epoch")):
            d = record[label+"_development"]
            t = holdout["scores"][kind+"_"+label] if holdout else None
            test_mae, test_mse = (f"{t['mae']:.9f}", f"{t['mse']:.9f}") if t else ("未评估", "未评估")
            lines.append(f"| {kind} / {label} | {record[epoch_key]} | {d['mae']:.9f} | {d['mse']:.9f} | {test_mae} | {test_mse} |")
    patch, control = results["patchtst"], results["dlinear_control"]
    lines += ["", "selected 包含 epoch0 的 E0 回退；best_trained 是实际训练轮中按开发 MAE 选出的权重。二者在训练前都已声明，不按测试成绩改变选择。最佳训练轮差于 E0 时，不能把回退成绩描述为 Transformer 学到了收益。", "",
              f"PatchTST 参数 {patch['parameters']:,}，匹配 DLinear 参数 {control['parameters']:,}。所有训练曲线、耗时与权重哈希见 results/，本地权重和预测数组在 outputs/。", ""]
    p, c = patch["selected_development"], control["selected_development"]
    best_reference = min(original["averaged_model"]["development"]["mae"], c["mae"])
    if p["mae"] < best_reference-1e-4:
        lines += ["开发集初筛优于较强参照，可作为后续三种子复验候选；单次结果不足以宣布最终最佳。", ""]
    else:
        lines += ["本固定配置未在开发集上明显超过增强 E1 与匹配 DLinear 中的较强者，暂不作为升级方案。一次固定配置也不能推断整个 PatchTST 模型族无效。", ""]
    lines += ["两模型共享：最后值锚点、仅输入窗口的方差缩放、h/24 残差缩放、损失、优化器、批量、早停与样本顺序。PatchTST 的注意力、位置编码、LayerNorm 和 dropout 属于架构；DLinear 没有对应层。原增强 E1 的训练后收缩、优化器与平均权重不同，所以另加同条件 DLinear 才能辅助判断架构差异。", "",
              "测试集是此前已经查看过的内部时间留出，不能称为新盲测或官方隐藏测试。所有配置和候选先固定，再评分；没有按测试选择 epoch、种子、归一化或融合权重。开发早停也不保证未曾过拟合开发集。", "",
              "本轮只输入 Close，未运行 E2/E3，也未使用新闻、财报。未新建或安装训练环境，未保存原始数据副本或大缓存。", ""]
    (HERE / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluate-holdout", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size) < 1:
        parser.error("epochs, patience and batch size must be positive")
    if shutil.disk_usage(HERE).free < 512*1024**2:
        raise RuntimeError("Insufficient disk reserve; preserve existing files and free space first")
    torch.set_flush_denormal(True)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.mha.set_fastpath_enabled(False)
    settings = {"config": CONFIG, "max_epochs": args.epochs, "patience": args.patience,
                "batch_size": args.batch_size, "seed": args.seed, "threads": 4,
                "optimizer": "AdamW lr=0.0003, ndim>=2 weights wd=0.01, scalar/1d parameters wd=0", "min_delta": 1e-5,
                "loss": "MAE+0.2*MSE", "clip_norm": 1.0, "epoch0_eligible": True,
                "normalization": "subtract input last value; divide by sqrt(input-window variance+1e-5); restore sigma and anchor; h/24 during train/inference"}
    hashes = source_hashes()
    protocol = {"settings": settings, "source_sha256": hashes, "kinds": list(KINDS),
                "selection": "development MAE only; save eligible epoch0/selected and separate best actual trained epoch; fixed seed42 single-configuration screening",
                "holdout_policy": "Freeze both models and all four checkpoint choices before scoring known internal holdout; no adaptation based on holdout",
                "environment": {"torch": str(torch.__version__), "numpy": np.__version__, "device": "cpu"}}
    path = HERE / "results/protocol.json"
    if path.exists():
        if not args.resume:
            raise FileExistsError("Existing experiment; use --resume to preserve it")
        assert read(path) == protocol
        write_json(HERE / f"results/resume_{time.time_ns()}.json", {"utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": sha256_file(path)})
    else:
        write_json(path, protocol)
    train, dev, metadata = development_data(args.package_dir.resolve())
    previous = read(HERE.parent / "e1_dlinear/results/calibration/summary.json")["split"]
    assert metadata == previous
    data_hashes = {key: value for key, value in metadata.items() if key.endswith("sha256")}
    write_json(HERE / "results/data_protocol.json", metadata)
    print(f"Ready: {len(train.x)} training, {len(dev.x)} development samples; inputs remain in RAM, no disk cache.", flush=True)
    for kind in KINDS:
        fit(kind, train, dev, args)
    del train
    gc.collect()
    if source_hashes() != hashes:
        raise ValueError("Experiment sources changed during training")
    if args.evaluate_holdout:
        evaluate_holdout(args, metadata, data_hashes, hashes)
    report()
    assert source_hashes() == hashes
    print("Fixed PatchTST trial completed; models/results are local only.", flush=True)


if __name__ == "__main__":
    main()
