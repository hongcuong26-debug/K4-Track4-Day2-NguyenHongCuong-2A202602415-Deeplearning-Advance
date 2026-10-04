"""Fast CPU checks for the completed student implementation."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[2]
sys.path.insert(0, str(CODE))

import benchmark  # noqa: E402
import dataset  # noqa: E402
import inference  # noqa: E402
import losses  # noqa: E402
import model  # noqa: E402
import train  # noqa: E402


class TestData(unittest.TestCase):
    def test_fold_zero_is_complete_and_disjoint(self):
        frames = dataset.load_split(ROOT / "data" / "labels", 0)
        stats = dataset.check_split(*frames, ROOT / "images")
        self.assertEqual(stats["n"], {"train": 10501, "val": 3501, "test": 3507})
        self.assertEqual(stats["union"], 17509)
        self.assertEqual(sum(stats["overlap"].values()), 0)


class TestLosses(unittest.TestCase):
    def test_focal_gamma_zero_matches_ce(self):
        torch.manual_seed(0)
        logits, labels = torch.randn(13, 9), torch.randint(0, 9, (13,))
        self.assertLess(abs(losses.FocalLoss(gamma=0)(logits, labels).item()
                            - F.cross_entropy(logits, labels).item()), 1e-6)

    def test_mix_and_class_weights(self):
        weights = losses.class_weights([100, 10, 20, 30, 40, 50, 60, 70, 80])
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)
        images, labels = torch.randn(8, 3, 32, 32), torch.arange(8)
        mixed, (_, _, lam) = losses.mix_batch(images, labels, mode="cutmix")
        self.assertEqual(mixed.shape, images.shape)
        self.assertTrue(0 <= lam <= 1)


class TestInference(unittest.TestCase):
    def test_temperature_and_aggregation(self):
        torch.manual_seed(1)
        logits = torch.randn(100, 9) * 3
        labels = logits.argmax(1)
        temperature = inference.fit_temperature(logits, labels)
        self.assertGreater(temperature, 0)
        probs = inference.aggregate_views([logits.numpy(), (logits + .1).numpy()])
        np.testing.assert_allclose(probs.sum(1), 1, atol=1e-6)

    def test_conv_bn_fusion(self):
        block = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1, bias=False), nn.BatchNorm2d(4), nn.ReLU()).eval()
        sample = torch.randn(2, 3, 16, 16)
        expected = block(sample)
        actual = inference.fuse_conv_bn(block)(sample)
        self.assertLess(float((expected - actual).abs().max().detach()), 1e-5)


class TestPipelineHelpers(unittest.TestCase):
    def test_model_and_overrides(self):
        network = model.build_model("mobilenetv3", init="scratch")
        self.assertGreater(model.count_params(network), 1)
        parsed = train.parse_overrides(["seed=2", "amp=false", "ema_decay=0.999", "mix=none"])
        self.assertEqual(parsed, {"seed": 2, "amp": False, "ema_decay": .999, "mix": None})

    def test_cpu_benchmark_contract(self):
        report = benchmark.bench(lambda: sum(range(10)), warmup=10, iters=50)
        self.assertEqual(report["n"], 50)
        self.assertLessEqual(report["p50"], report["p95"])
        self.assertLessEqual(report["p95"], report["p99"])


if __name__ == "__main__":
    unittest.main()
