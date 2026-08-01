import unittest

import numpy as np
import torch

from attack_algorithm.dct_decoder import (
    DCTMaskResidualCandidate,
    DCTMaskResidualDecoder,
    project_l2,
)
from attack_algorithm.dct_mask_residual_attack import DCTMaskResidualAttack
from attack_algorithm.multilabel_loss import is_success
from attack_algorithm.multilabel_loss import make_random_target
from attack_algorithm.query_manager import QueryManager
from attack_algorithm.sba_dct_attack import DCTLowFrequencyDecoder, SBA_DCT_Attack
from attack_algorithm.sba_wavelet_attack import SBA_Wavelet_Attack
from attack_algorithm.wavelet_utils import (
    add_scaled_direction,
    build_zero_coeffs,
    inverse_haar_reconstruct,
    inverse_haar_step,
    make_adv,
    zeros_like_coeffs_structure,
)


class DummyModel(torch.nn.Module):
    def forward(self, batch):
        mean = batch.mean(dim=(1, 2, 3))
        fixed_pos = torch.full_like(mean, 0.8)
        add_label = torch.clamp(0.2 + mean * 0.1, 0.0, 1.0)
        fixed_neg = torch.full_like(mean, 0.1)
        return torch.stack((fixed_pos, add_label, fixed_neg), dim=1)


class DCTMaskResidualAttackTest(unittest.TestCase):
    def test_decoder_shape_and_project_l2(self):
        decoder = DCTMaskResidualDecoder(
            dct_k=4, mask_size=4, mask_ratio=0.25, eps=3.0, device="cpu"
        )
        candidate = DCTMaskResidualCandidate(
            z_dct=torch.randn(3, 4, 4),
            mask_logits=torch.randn(1, 4, 4),
            z_residual=torch.randn(3, 4, 4),
        )
        x_adv, r = decoder.decode(candidate, torch.zeros(3, 448, 448))
        self.assertEqual(tuple(x_adv.shape), (3, 448, 448))
        self.assertEqual(tuple(r.shape), (3, 448, 448))

        projected = project_l2(torch.ones(3, 448, 448), eps=2.0)
        self.assertLessEqual(float(torch.linalg.vector_norm(projected)), 2.0 + 1e-5)

    def test_query_count_and_success(self):
        manager = QueryManager(DummyModel(), max_queries=3, device="cpu")
        scores = manager.query(torch.zeros(2, 3, 448, 448))
        self.assertEqual(tuple(scores.shape), (2, 3))
        self.assertEqual(manager.queries, 2)

        success, pred, l2 = is_success(
            scores=np.asarray([0.51, 0.1, 0.9]),
            target=np.asarray([1, 0, 1]),
            x_adv=np.zeros((3, 448, 448), dtype=np.float32),
            x_orig=np.zeros((3, 448, 448), dtype=np.float32),
            threshold=0.5,
            eps=1.0,
        )
        self.assertTrue(success)
        np.testing.assert_array_equal(pred, np.asarray([1, 0, 1]))
        self.assertEqual(l2, 0.0)

    def test_attack_one_stays_inside_budget(self):
        attack = DCTMaskResidualAttack(
            model=DummyModel(),
            max_queries=9,
            dct_k=4,
            mask_size=4,
            pop_size=4,
            stage1_queries=2,
            stage2_queries=5,
            seed=11,
            device="cpu",
        )
        result = attack.attack_one(torch.zeros(3, 448, 448))
        self.assertIn("success", result)
        self.assertEqual(tuple(result["x_adv"].shape), (3, 448, 448))
        self.assertLessEqual(result["queries"], 9)
        self.assertLessEqual(result["l2"], attack.eps + 1e-5)

    def test_sba_dct_decoder_target_and_budget(self):
        decoder = DCTLowFrequencyDecoder(dct_k=4, eps=2.0, device="cpu")
        x_adv, r = decoder.decode(torch.zeros(3, 448, 448), torch.randn(3, 4, 4) * 10.0)
        self.assertEqual(tuple(r.shape), (3, 448, 448))
        self.assertLessEqual(float(torch.linalg.vector_norm(r)), 2.0 + 1e-5)
        self.assertGreaterEqual(float(x_adv.min()), 0.0)
        self.assertLessEqual(float(x_adv.max()), 1.0)

        target, y_hide, y_add = make_random_target(
            np.asarray([1, 0, 1, 0]), np.random.default_rng(7)
        )
        self.assertEqual(int(target.sum()), 2)
        self.assertEqual(target[y_hide], 0)
        self.assertEqual(target[y_add], 1)

        attack = SBA_DCT_Attack(
            model=DummyModel(),
            max_queries=7,
            dct_k=4,
            patience=2,
            seed=13,
            device="cpu",
        )
        result = attack.attack_one(torch.zeros(3, 448, 448))
        self.assertLessEqual(result["queries"], 7)
        self.assertEqual(tuple(result["x_adv"].shape), (3, 448, 448))

        adapted_attack = SBA_DCT_Attack(
            model=DummyModel(),
            max_queries=7,
            dct_k=4,
            patience=2,
            seed=17,
            device="cpu",
        )
        result = adapted_attack.attack_one(
            torch.zeros(3, 448, 448),
            y_orig=np.asarray([1, -1, -1]),
            target=np.asarray([-1, 1, -1]),
        )
        self.assertNotEqual(result["y_hide"], -1)
        self.assertNotEqual(result["y_add"], -1)

    def test_sba_wavelet_reconstruct_and_budget(self):
        ll = torch.ones(3, 2, 2)
        zeros = torch.zeros_like(ll)
        step = inverse_haar_step(ll, zeros, zeros, zeros)
        self.assertEqual(tuple(step.shape), (3, 4, 4))
        self.assertTrue(torch.allclose(step, torch.full_like(step, 0.5)))

        coeffs = build_zero_coeffs(active_levels=(4, 5), device="cpu")
        coeffs["LL_5"][:] = 1.0
        raw = inverse_haar_reconstruct(coeffs)
        self.assertEqual(tuple(raw.shape), (3, 448, 448))
        direction = zeros_like_coeffs_structure(coeffs)
        direction["HH_4"][0, 0, 0] = 1.0
        shifted = add_scaled_direction(coeffs, direction, 2.0)
        self.assertEqual(tuple(shifted["HH_4"].shape), tuple(coeffs["HH_4"].shape))
        x_adv, r = make_adv(torch.zeros(3, 448, 448), shifted, eps=2.0)
        self.assertLessEqual(float(torch.linalg.vector_norm(r)), 2.0 + 1e-5)
        self.assertGreaterEqual(float(x_adv.min()), 0.0)
        self.assertLessEqual(float(x_adv.max()), 1.0)

        attack = SBA_Wavelet_Attack(
            model=DummyModel(),
            max_queries=7,
            active_levels=(4, 5),
            wavelet_levels=2,
            patience=2,
            seed=23,
            device="cpu",
        )
        result = attack.attack_one(
            torch.zeros(3, 448, 448),
            y_orig=np.asarray([1, -1, -1]),
            target=np.asarray([-1, 1, -1]),
        )
        self.assertLessEqual(result["queries"], 7)
        self.assertEqual(tuple(result["x_adv"].shape), (3, 448, 448))


if __name__ == "__main__":
    unittest.main()
