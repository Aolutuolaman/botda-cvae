"""Checks using the training dataset included in this repository."""

from collections import Counter
from contextlib import redirect_stdout, redirect_stderr
import io
from pathlib import Path
import unittest

import torch

import train
import generate


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {"fading_0.00": 3689, "fading_0.10": 381, "fading_0.20": 611,
            "fading_0.30": 488, "fading_0.40": 401, "fading_0.50": 509}


class TrainingDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            cls.dataset = train.ConditionalFadingDataset(str(ROOT / "data0"))

    def test_dataset_files(self):
        counts = Counter()
        for path in (ROOT / "data0").glob("fading_*/*.txt"):
            code = path.read_text(encoding="utf-8").strip()
            self.assertTrue(code)
            self.assertLessEqual(set(code), {"0", "1"})
            self.assertIn(len(code), (120, 180, 240, 300, 360, 450, 540, 630, 720))
            counts[path.parent.name] += 1
        self.assertEqual(dict(counts), EXPECTED)
        self.assertEqual(len(self.dataset), sum(EXPECTED.values()))

    def test_training_labels_and_envelope(self):
        for fading in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
            sample = next(s for s in self.dataset.samples if s["fading"] == fading)
            length = sample["seq_len"]
            self.assertEqual(sample["fe"], int(sample["y"].sum()))
            expected = sample["y"][:length] * torch.exp(-fading * torch.linspace(0, 1, length))
            torch.testing.assert_close(sample["x"][:length], expected)
            self.assertEqual(int(torch.count_nonzero(sample["x"][length:])), 0)

    def test_train_and_generate_model_compatibility(self):
        training_model = train.UniversalCVAE().eval()
        generation_model = generate.UniversalCVAE().eval()
        generation_model.load_state_dict(training_model.state_dict(), strict=True)
        sample = self.dataset.samples[0]
        x, condition = sample["x"].unsqueeze(0), sample["cond"].unsqueeze(0)
        with torch.no_grad():
            torch.manual_seed(42)
            expected = training_model(x, condition)
            torch.manual_seed(42)
            actual = generation_model(x, condition)
        for left, right in zip(expected, actual):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_physical_loss_backward_on_training_samples(self):
        indices = [next(i for i, sample in enumerate(self.dataset.samples)
                        if sample["fading"] == fading) for fading in (0.0, 0.5)]
        batch = [torch.stack([self.dataset[i][j] for i in indices]) for j in range(6)]
        x, y, condition, lengths, fe, fading = batch
        model = train.UniversalCVAE()
        recon, mu, logvar = model(x, condition)
        loss, *_ = train.compute_loss(recon, y, mu, logvar, condition,
                                      lengths.squeeze(1), fe, fading, 500)
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                            for p in model.parameters()))

    def test_train_and_generate_physics(self):
        sample = next(s for s in self.dataset.samples if s["fading"] == 0.5)
        x = sample["x"].unsqueeze(0)
        length = sample["seq_len"]
        first = train.calculate_physics_metrics(x, length, length * 200)
        second = generate.calculate_physics_metrics(x, length, length * 200)
        for left, right in zip(first, second):
            torch.testing.assert_close(left, right, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
