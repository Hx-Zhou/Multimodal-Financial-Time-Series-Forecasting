"""Train the E6-B stationary-feature market GRU experiment."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.gru_e3.train import main as shared_main


def main() -> None:
    arguments = list(sys.argv[1:])
    if "--variant" not in arguments:
        arguments += ["--variant", "e6b"]
    if "--features" not in arguments:
        arguments += ["--features", "all"]
    if "--checkpoint" not in arguments:
        arguments += ["--checkpoint", "experiments/gru_e6_b/checkpoints/best.pt"]
    if "--output" not in arguments:
        name = "holdout_check.json" if "--evaluate-holdout" in arguments else "result.json"
        arguments += ["--output", f"experiments/gru_e6_b/{name}"]
    sys.argv = [sys.argv[0], *arguments]
    shared_main()


if __name__ == "__main__":
    main()
