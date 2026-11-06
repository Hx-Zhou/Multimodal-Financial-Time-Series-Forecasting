"""Decouple weight decay; avoid penalizing gates toward sigmoid(0)=0.5."""
import torch


def make_optimizer(model, policy="matrix_only"):
    if policy not in ("all", "matrix_only"):
        raise ValueError(policy)
    decay, no_decay = [], []
    names = {"decay": [], "no_decay": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group = "decay" if policy == "all" or parameter.ndim >= 2 else "no_decay"
        (decay if group == "decay" else no_decay).append(parameter)
        names[group].append(name)
    optimizer = torch.optim.AdamW([
        {"params": decay, "weight_decay": 0.01},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=0.001)
    return optimizer, names
