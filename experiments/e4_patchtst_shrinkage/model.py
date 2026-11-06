"""Fold a single global residual coefficient into the existing output heads."""
import torch


def shrink_state(state, kind, alpha):
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0,1]")
    prefixes = ("head.",) if kind == "patchtst" else ("trend.", "remainder.") if kind == "dlinear_control" else None
    if prefixes is None:
        raise ValueError(kind)
    result = {name: value.detach().cpu().clone() for name, value in state.items()}
    for name in result:
        if name.startswith(prefixes):
            result[name].mul_(alpha)
    # Existing h/24 and window-scale restoration are left untouched.
    return result
