import unittest
import torch
from .model import PatchTST, DLinearControl, history_transform


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)

    def test_initialization_is_exact_last_value_even_for_constant_windows(self):
        x = torch.cat((torch.randn(3, 96), torch.full((1, 96), 7.0)))
        for model in (PatchTST(), DLinearControl()):
            prediction = model({"close": x})
            self.assertEqual(prediction.shape, (4, 24))
            torch.testing.assert_close(prediction, x[:, -1:].expand(-1, 24), rtol=0, atol=0)

    def test_windows_use_only_history_and_restore_offset(self):
        x = torch.randn(3, 96)
        z, anchor, sigma = history_transform(x)
        shifted, other_anchor, other_sigma = history_transform(x+17)
        torch.testing.assert_close(z, shifted, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(sigma, other_sigma, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(other_anchor, anchor+17)
        self.assertEqual(x.unfold(1, 16, 8).shape, (3, 11, 16))

    def test_encoder_receives_finite_training_gradients_after_zero_head_warmup(self):
        model = PatchTST(d_model=32, d_ff=64)
        x = torch.randn(8, 96)
        target = x[:, -1:]+torch.randn(8, 24)*0.1
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        for _ in range(3):
            optimizer.zero_grad()
            error = model({"close": x})-target
            (error.abs().mean()+0.2*error.square().mean()).backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
            optimizer.step()
        self.assertGreater(float(model.projection.weight.grad.abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()
