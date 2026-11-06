import unittest
import torch
from experiments.e9_dlinear_multimodal.model import Model
from .optimizer import make_optimizer


class OptimizerTests(unittest.TestCase):
    def test_groups_exclude_frozen_market_and_exempt_biases_and_gates(self):
        model = Model()
        optimizer, names = make_optimizer(model)
        self.assertFalse(any(n.startswith("market.") for group in names.values() for n in group))
        self.assertIn("news.gate", names["no_decay"])
        self.assertIn("reports.gate", names["no_decay"])
        self.assertTrue(all(not n.endswith("bias") for n in names["decay"]))
        trainable = {id(p) for p in model.parameters() if p.requires_grad}
        actual = [id(p) for g in optimizer.param_groups for p in g["params"]]
        self.assertEqual(trainable, set(actual))
        self.assertEqual(len(actual), len(trainable))

    def test_zero_data_gradient_does_not_move_gate_or_collapse_encoder(self):
        torch.manual_seed(42)
        model = Model()
        optimizer, _ = make_optimizer(model)
        weight = model.news.mlp[0].weight
        initial = weight.detach().clone()
        gate = model.news.gate.detach().clone()
        expected = initial.clone()
        for _ in range(100):
            for group in optimizer.param_groups:
                for p in group["params"]:
                    p.grad = torch.zeros_like(p)
            optimizer.step()
            expected.mul_(1-0.001*0.01)
        torch.testing.assert_close(model.news.gate, gate, rtol=0, atol=0)
        torch.testing.assert_close(weight, expected, rtol=0, atol=0)
        self.assertGreater(float(weight.detach().norm()/initial.norm()), 0.998)

    def test_all_parameter_policy_is_an_isolated_optimizer_comparison(self):
        model = Model()
        optimizer, names = make_optimizer(model, "all")
        self.assertIn("news.gate", names["decay"])
        self.assertEqual(names["no_decay"], [])
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 0.01)


class SelectionTests(unittest.TestCase):
    def test_performance_alone_cannot_pass_without_content_and_period_stability(self):
        from .run import qualifies
        baseline = {"mae": 1.0, "mse": 1.0}
        halves = {"early": baseline, "late": baseline}
        result = {"development": {"mae": 0.99, "mse": 1.0},
                  "halves": {"early": {"mae": 0.99}, "late": {"mae": 0.99}},
                  "diagnostics": {"news_semantics_zero": {"mae": 0.991}, "news_shuffled": {"mae": 0.991}}}
        self.assertTrue(qualifies(result, baseline, halves)["passed"])
        result["diagnostics"]["news_shuffled"]["mae"] = 0.989
        self.assertFalse(qualifies(result, baseline, halves)["passed"])
        result["diagnostics"]["news_shuffled"]["mae"] = 0.991
        result["halves"]["late"]["mae"] = 1.01
        self.assertFalse(qualifies(result, baseline, halves)["passed"])


if __name__ == "__main__":
    unittest.main()
