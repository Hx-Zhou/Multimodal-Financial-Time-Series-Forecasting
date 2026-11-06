import unittest
import torch
from experiments.e4_patchtst.model import build_model
from experiments.e4_patchtst import run as original
from .model import shrink_state
from .run import load_trainer, HERE, select_candidate


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)

    def test_head_folding_scales_only_residual_once(self):
        for kind in ("patchtst", "dlinear_control"):
            model = build_model(kind, original.CONFIG).eval()
            heads = (model.head,) if kind == "patchtst" else (model.trend, model.remainder)
            for head in heads:
                torch.nn.init.normal_(head.weight, std=0.01)
                torch.nn.init.normal_(head.bias, std=0.01)
            x = torch.randn(4, 96)
            with torch.inference_mode():
                baseline, state = model({"close": x}), {k: v.clone() for k, v in model.state_dict().items()}
                for alpha in (0., 0.25, 0.5, 1.):
                    model.load_state_dict(shrink_state(state, kind, alpha))
                    value = model({"close": x})
                    torch.testing.assert_close(value, x[:, -1:]+alpha*(baseline-x[:, -1:]), rtol=1e-5, atol=1e-6)
                for key in state:
                    self.assertTrue(torch.equal(state[key], shrink_state(state, kind, 1.)[key]))

    def test_replica_output_root_does_not_modify_original_module(self):
        before = original.HERE
        trainer = load_trainer(43)
        self.assertEqual(original.HERE, before)
        self.assertEqual(trainer.HERE, HERE / "replicas/seed43")
        self.assertIsNot(trainer.fit.__globals__, original.fit.__globals__)

    def test_gate_rejects_one_seed_gain_or_excess_mse(self):
        reference = {"mae": 1., "mse": 1.}
        halves = {"early": reference, "late": reference}
        row = {"average": {"mae": .99, "mse": 1.}, "seeds_improved_vs_raw": 1,
               "halves": {"early": {"mae": .99}, "late": {"mae": .99}}}
        candidates = {"0.5": row, "0.25": row}
        self.assertIsNone(select_candidate(candidates, reference, halves)["selected_alpha"])
        row["seeds_improved_vs_raw"] = 2
        self.assertEqual(select_candidate(candidates, reference, halves)["selected_alpha"], .5)
        row["average"]["mse"] = 1.01
        self.assertIsNone(select_candidate(candidates, reference, halves)["selected_alpha"])


if __name__ == "__main__":
    unittest.main()
