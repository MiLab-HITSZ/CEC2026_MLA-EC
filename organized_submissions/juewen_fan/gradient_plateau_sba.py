"""
GradientPlateauSBA — a black-box multilabel adversarial attack based on
low-frequency DCT gradient estimation
===================================================================

CEC 2026 Competition on Evolutionary Computation in Black-box Multilabel
Adversarial Examples.

Architecture:
  Phase 1: Antithetic gradient estimation in the low-frequency DCT space
           with Adam optimization, single-label rescue, 0.5-plateau escape,
           and random restarts for zero gradients.
  Phase 2: Coordinate-wise greedy SBA fallback search.
"""

import random
import numpy as np
from abc import ABC, abstractmethod
from scipy.fft import idctn
from attack_problem.one_image_problem import SingleImageProblem


# ==================== Base class ====================

class AttackAlgorithmBase(ABC):
    def __init__(self, config):
        self.rnd = config["rnd"]

    @abstractmethod
    def evolve(self, problem: SingleImageProblem) -> np.ndarray:
        pass


# ==================== Utilities ====================

def _build_dct_basis_1d(n, image_size):
    """Return the first n orthonormal DCT-II basis vectors."""
    basis = np.zeros((n, image_size))
    indices = np.arange(image_size)
    for k in range(n):
        if k == 0:
            alpha = 1.0 / np.sqrt(image_size)
        else:
            alpha = np.sqrt(2.0 / image_size)
        basis[k] = alpha * np.cos(np.pi * (2 * indices + 1) * k / (2 * image_size))
    return basis


# ==================== Main algorithm ====================

class GradientPlateauSBA(AttackAlgorithmBase):

    def __init__(self, config=None):
        if config is None:
            config = {"rnd": random.Random(1)}
        super().__init__(config)
        self.freq_dims = config.get("freq_dims", 30)
        self.sigma = config.get("sigma", 30.0)
        self.lr = config.get("lr", 1.0)
        self.k = config.get("k", 50)
        self.max_step_l2 = config.get("max_step_l2", 3.0)
        self.stagnation_steps = config.get("stagnation_steps", 10)
        self.switch_fitness = config.get("switch_fitness", 0.35)
        self.switch_remaining = config.get("switch_remaining", 2100)

        self.plateau_low = config.get("plateau_low", 0.49)
        self.plateau_high = config.get("plateau_high", 0.501)
        self.plateau_grad_tol = config.get("plateau_grad_tol", 1e-7)
        self.plateau_sigma = config.get("plateau_sigma", self.sigma * 2)
        self.plateau_sigma_max = config.get("plateau_sigma_max", self.sigma * 8)
        self.plateau_rounds = config.get("plateau_rounds", 12)
        self.plateau_tol = config.get("plateau_tol", 1e-4)
        self.sparse_ratio = config.get("sparse_ratio", 0.12)
        self.zero_grad_tol = config.get("zero_grad_tol", 1e-12)
        self.zero_kick_wait = config.get("zero_kick_wait", 3)
        self.zero_kick_l2_ratios = config.get(
            "zero_kick_l2_ratios", (0.35, 0.65, 0.95)
        )
        self.plateau_fail_limit = config.get("plateau_fail_limit", 1)
        self.rescue_best_threshold = config.get("rescue_best_threshold", 0.35)
        self.near_success_threshold = config.get("near_success_threshold", 0.02)

    def _z_to_pixel(self, z, image_size):
        freq = self.freq_dims
        pixel = np.zeros((3, image_size, image_size))
        for c in range(3):
            padded = np.zeros((image_size, image_size))
            padded[:freq, :freq] = z[c]
            pixel[c] = idctn(padded, type=2, norm="ortho")
        return pixel.flatten()

    def _project_pixel(self, x, epsilon):
        l2 = np.linalg.norm(x)
        if l2 > epsilon:
            return x * (epsilon / l2)
        return x

    def _make_directions(self, np_rng, k, z, at_cap, sparse=False):
        shape = z.shape
        directions = np_rng.randn(k, *shape).astype(np.float64)
        if sparse:
            flat_dim = int(np.prod(shape))
            active = max(1, int(flat_dim * self.sparse_ratio))
            for i in range(k):
                mask = np.zeros(flat_dim, dtype=bool)
                mask[np_rng.choice(flat_dim, active, replace=False)] = True
                d_flat = directions[i].reshape(-1)
                d_flat[~mask] = 0.0

        for i in range(k):
            d_norm = np.linalg.norm(directions[i])
            if d_norm > 0:
                directions[i] /= d_norm

        if at_cap:
            z_flat = z.flatten()
            z_l2 = np.linalg.norm(z_flat)
            if z_l2 > 0:
                z_unit = z_flat / z_l2
                for i in range(k):
                    d_flat = directions[i].flatten()
                    d_flat -= np.dot(d_flat, z_unit) * z_unit
                    d_norm = np.linalg.norm(d_flat)
                    if d_norm > 0:
                        d_flat /= d_norm
                    directions[i] = d_flat.reshape(shape)
        return directions

    def _evaluate_z(self, problem, z, image_size, epsilon, effective=True):
        x = self._project_pixel(self._z_to_pixel(z, image_size), epsilon)
        fitness, fit = problem.evaluate(x.reshape(1, -1), effective=effective)
        if fitness is None:
            return None, None, x
        return fitness, fit, x

    def _verify_success(self, problem, x):
        if problem.evaluations + 1 > problem.max_evaluation:
            return True, None, None
        res, fit = problem.evaluate(x.reshape(1, -1))
        if res is None:
            return True, None, None
        if res[0, 0] == 0:
            return True, res[0, 0], fit[0].copy()
        return False, res[0, 0], fit[0].copy()

    def _plateau_breakthrough(
            self, problem, z, image_size, epsilon, hard_label, current_hard_fit,
            np_rng, sigma):
        best_z = z
        best_x = None
        best_fitness = None
        best_fit = None
        best_hard_fit = current_hard_fit
        cur_sigma = sigma

        for round_idx in range(self.plateau_rounds):
            remaining = problem.max_evaluation - problem.evaluations
            if remaining <= 0:
                break
            batch_n = min(2 * self.k, remaining)
            if batch_n <= 0:
                break

            sparse = (round_idx % 2 == 1)
            directions = self._make_directions(
                np_rng, batch_n, z, True, sparse=sparse
            )
            x_batch = np.zeros((batch_n, problem.get_dimension()))
            z_batch = []
            for i in range(batch_n):
                z_try = z + cur_sigma * directions[i]
                x_try = self._project_pixel(
                    self._z_to_pixel(z_try, image_size), epsilon
                )
                x_batch[i] = x_try
                z_batch.append(z_try)

            fitness, fit = problem.evaluate(x_batch)
            if fitness is None:
                break

            hard_fits = fit[:, hard_label]
            idx = int(np.argmin(hard_fits))
            candidate_hard_fit = float(hard_fits[idx])
            candidate_fitness = float(fitness[idx, 0])

            if candidate_hard_fit < best_hard_fit - self.plateau_tol:
                best_hard_fit = candidate_hard_fit
                best_z = z_batch[idx]
                best_x = x_batch[idx].copy()
                best_fitness = fitness[idx, 0]
                best_fit = fit[idx].copy()
                z = best_z
                cur_sigma = max(self.plateau_sigma, cur_sigma * 0.85)
                if best_hard_fit <= 0.01 or best_fitness == 0:
                    break
            else:
                cur_sigma = min(cur_sigma * 1.5, self.plateau_sigma_max)

        return best_z, best_x, best_fitness, best_fit, best_hard_fit, cur_sigma

    def evolve(self, problem: SingleImageProblem) -> np.ndarray:
        rnd = self.rnd
        freq = self.freq_dims
        sigma = self.sigma
        lr = self.lr
        k = self.k
        max_step = self.max_step_l2
        epsilon = problem.epsilon * (1 - 1e-9)
        dim = problem.get_dimension()
        image_size = int(np.sqrt(dim / 3))

        np_rng = np.random.RandomState(rnd.randrange(2**31))

        z = np.zeros((3, freq, freq))
        result = problem.evaluate(self._z_to_pixel(z, image_size).reshape(1, -1))
        if result[0] is None:
            return self._z_to_pixel(z, image_size)
        best_fitness = result[0][0, 0]
        best_fit = result[1][0].copy()
        best_x = self._z_to_pixel(z, image_size)
        current_fit = best_fit.copy()
        if best_fitness == 0:
            confirmed, checked_fitness, checked_fit = self._verify_success(
                problem, best_x
            )
            if confirmed:
                return best_x
            best_fitness = checked_fitness
            best_fit = checked_fit
            current_fit = best_fit.copy()

        m = np.zeros_like(z)
        v = np.zeros_like(z)
        beta1, beta2, eps_adam = 0.9, 0.999, 1e-8

        eval_per_step = 2 * k
        step_count = 0
        rescue_mode = False
        rescue_label = None
        rescue_sigma = sigma * 2
        rescue_sigma_max = sigma * 4
        rescue_best_hard_fit = 1.0
        rescue_no_improve = 0
        cap_step = None
        cap_best = None
        zero_grad_steps = 0
        zero_kicks = 0
        plateau_failures = {}
        plateau_disabled_labels = set()

        while problem.evaluations + eval_per_step <= problem.max_evaluation:
            step_count += 1
            current_sigma = rescue_sigma if rescue_mode else sigma

            cur_l2 = np.linalg.norm(self._z_to_pixel(z, image_size))
            at_cap = cur_l2 >= epsilon * 0.99

            if at_cap:
                if cap_step is None:
                    cap_step = step_count
                    cap_best = best_fitness
                elif step_count - cap_step >= self.stagnation_steps:
                    remaining = problem.max_evaluation - problem.evaluations
                    no_total_gain = cap_best - best_fitness < 0.01
                    if no_total_gain and not rescue_mode:
                        if best_fitness <= self.switch_fitness or remaining <= self.switch_remaining:
                            break
                if best_fitness < cap_best - 0.01:
                    cap_step = step_count
                    cap_best = best_fitness

            directions = self._make_directions(np_rng, k, z, at_cap)

            x_batch = np.zeros((2 * k, dim))
            for i in range(k):
                x_p = self._project_pixel(
                    self._z_to_pixel(z + current_sigma * directions[i], image_size),
                    epsilon
                )
                x_m = self._project_pixel(
                    self._z_to_pixel(z - current_sigma * directions[i], image_size),
                    epsilon
                )
                x_batch[i] = x_p
                x_batch[k + i] = x_m

            result = problem.evaluate(x_batch)
            if result[0] is None:
                return best_x

            current_stuck = np.where(current_fit > self.near_success_threshold)[0]
            if len(current_stuck) == 0:
                break

            target_classes = np.array([rescue_label]) if rescue_mode else current_stuck

            fit_pc = result[1]
            p_plus = np.clip(0.5 - fit_pc[:k, target_classes], 1e-10, 1.0)
            p_minus = np.clip(0.5 - fit_pc[k:, target_classes], 1e-10, 1.0)
            f_plus = (-np.log(p_plus)).sum(axis=1)
            f_minus = (-np.log(p_minus)).sum(axis=1)

            grad = np.zeros_like(z)
            for i in range(k):
                grad += ((f_plus[i] - f_minus[i]) / (2 * current_sigma)) * directions[i]
            grad /= k
            grad_norm = np.linalg.norm(grad)

            for i in range(2 * k):
                if result[0][i, 0] < best_fitness:
                    candidate_x = x_batch[i].copy()
                    candidate_fitness = result[0][i, 0]
                    candidate_fit = result[1][i].copy()
                    if candidate_fitness == 0:
                        confirmed, checked_fitness, checked_fit = (
                            self._verify_success(problem, candidate_x)
                        )
                        if confirmed:
                            return candidate_x
                        if checked_fitness >= best_fitness:
                            continue
                        candidate_fitness = checked_fitness
                        candidate_fit = checked_fit
                    best_fitness = candidate_fitness
                    best_fit = candidate_fit
                    best_x = candidate_x

            if not rescue_mode and at_cap and best_fitness <= self.rescue_best_threshold:
                hard_candidates = np.where(current_fit >= 0.45)[0]
                if len(hard_candidates) > 0:
                    rescue_label = int(hard_candidates[np.argmax(current_fit[hard_candidates])])
                    rescue_mode = True
                    rescue_best_hard_fit = current_fit[rescue_label]
                    rescue_no_improve = 0
                    rescue_sigma = sigma * 2
                    m = np.zeros_like(z)
                    v = np.zeros_like(z)

            if grad_norm <= self.zero_grad_tol and cur_l2 < epsilon * 0.2:
                zero_grad_steps += 1
            else:
                zero_grad_steps = 0

            if zero_grad_steps >= self.zero_kick_wait:
                ratios = tuple(self.zero_kick_l2_ratios)
                ratio = ratios[min(zero_kicks, len(ratios) - 1)]
                z_rand = np_rng.randn(*z.shape).astype(np.float64)
                z_norm = np.linalg.norm(z_rand)
                if z_norm > 0:
                    z = z_rand * ((epsilon * ratio) / z_norm)
                    res, fit, x_cur = self._evaluate_z(
                        problem, z, image_size, epsilon
                    )
                    if res is not None:
                        current_fit = fit[0].copy()
                        if res[0, 0] < best_fitness:
                            candidate_fitness = res[0, 0]
                            candidate_fit = fit[0].copy()
                            accept_candidate = True
                            if candidate_fitness == 0:
                                confirmed, checked_fitness, checked_fit = (
                                    self._verify_success(problem, x_cur)
                                )
                                if confirmed:
                                    return x_cur
                                if checked_fitness >= best_fitness:
                                    accept_candidate = False
                                else:
                                    candidate_fitness = checked_fitness
                                    candidate_fit = checked_fit
                            if accept_candidate:
                                best_fitness = candidate_fitness
                                best_fit = candidate_fit
                                best_x = x_cur
                    m = np.zeros_like(z)
                    v = np.zeros_like(z)
                    rescue_mode = False
                    zero_grad_steps = 0
                    zero_kicks += 1
                    continue

            plateau_fit = current_fit[rescue_label] if rescue_mode else best_fitness
            plateau_stuck = np.where(current_fit > 0.01)[0]
            plateau_label = int(plateau_stuck[0]) if len(plateau_stuck) == 1 else None
            is_plateau = (
                at_cap
                and len(plateau_stuck) == 1
                and plateau_label not in plateau_disabled_labels
                and self.plateau_low <= plateau_fit <= self.plateau_high
                and grad_norm <= self.plateau_grad_tol
            )
            if is_plateau:
                hard_label = plateau_label
                out = self._plateau_breakthrough(
                    problem, z, image_size, epsilon, hard_label,
                    current_fit[hard_label], np_rng, self.plateau_sigma
                )
                z, px, pf, pfit, hard_fit, new_sigma = out
                self.plateau_sigma = new_sigma
                rescue_mode = False
                m = np.zeros_like(z)
                v = np.zeros_like(z)
                if px is not None:
                    plateau_failures[hard_label] = 0
                    current_fit = pfit.copy()
                    if pf < best_fitness:
                        candidate_x = px.copy()
                        candidate_fitness = pf
                        candidate_fit = pfit.copy()
                        if candidate_fitness == 0:
                            confirmed, checked_fitness, checked_fit = (
                                self._verify_success(problem, candidate_x)
                            )
                            if confirmed:
                                return candidate_x
                            if checked_fitness >= best_fitness:
                                continue
                            candidate_fitness = checked_fitness
                            candidate_fit = checked_fit
                        best_fitness = candidate_fitness
                        best_fit = candidate_fit
                        best_x = candidate_x
                    continue
                else:
                    plateau_failures[hard_label] = (
                        plateau_failures.get(hard_label, 0) + 1
                    )
                    if plateau_failures[hard_label] >= self.plateau_fail_limit:
                        plateau_disabled_labels.add(hard_label)
                    else:
                        continue

            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            m_hat = m / (1 - beta1 ** step_count)
            v_hat = v / (1 - beta2 ** step_count)
            update = lr * m_hat / (np.sqrt(v_hat) + eps_adam)
            upd_l2 = np.linalg.norm(update)
            if upd_l2 > max_step:
                update *= max_step / upd_l2

            z = z - update
            z_norm = np.linalg.norm(z)
            if z_norm > epsilon * 0.99:
                z *= epsilon * 0.99 / z_norm

            should_eval = rescue_mode or (step_count % 5 == 0)
            if should_eval and problem.evaluations + 1 <= problem.max_evaluation:
                res, fit, x_cur = self._evaluate_z(problem, z, image_size, epsilon)
                if res is not None:
                    current_fit = fit[0].copy()
                    if res[0, 0] < best_fitness:
                        candidate_fitness = res[0, 0]
                        candidate_fit = fit[0].copy()
                        accept_candidate = True
                        if candidate_fitness == 0:
                            confirmed, checked_fitness, checked_fit = (
                                self._verify_success(problem, x_cur)
                            )
                            if confirmed:
                                return x_cur
                            if checked_fitness >= best_fitness:
                                accept_candidate = False
                            else:
                                candidate_fitness = checked_fitness
                                candidate_fit = checked_fit
                        if accept_candidate:
                            best_fitness = candidate_fitness
                            best_fit = candidate_fit
                            best_x = x_cur

                    if rescue_mode and rescue_label is not None:
                        hard_fit_now = current_fit[rescue_label]
                        if hard_fit_now <= 0.01:
                            rescue_mode = False
                            m = np.zeros_like(z)
                            v = np.zeros_like(z)
                        elif hard_fit_now < rescue_best_hard_fit - 0.001:
                            rescue_best_hard_fit = hard_fit_now
                            rescue_no_improve = 0
                            rescue_sigma = max(sigma * 2, rescue_sigma * 0.9)
                        else:
                            rescue_no_improve += 1
                            if rescue_no_improve >= 3:
                                rescue_sigma = min(rescue_sigma * 1.5, rescue_sigma_max)
                                rescue_no_improve = 0

        # Evaluate the current z after Phase 1.
        if problem.evaluations < problem.max_evaluation:
            res, fit, x_cur = self._evaluate_z(
                problem, z, image_size, epsilon, effective=False
            )
            if res is not None and res[0, 0] < best_fitness:
                best_fitness = res[0, 0]
                best_fit = fit[0].copy()
                best_x = x_cur

        if best_fitness == 0:
            confirmed, checked_fitness, checked_fit = self._verify_success(
                problem, best_x
            )
            if confirmed:
                return best_x
            best_fitness = checked_fitness
            best_fit = checked_fit

        # Phase 2: SBA fallback.
        if best_fitness > 0 and problem.evaluations < problem.max_evaluation:
            sba_freq = 30
            sba_step = 1.0
            n_dims_sba = 3 * sba_freq * sba_freq
            n_pixel_per_ch = image_size * image_size
            row_basis = _build_dct_basis_1d(sba_freq, image_size)
            col_basis = _build_dct_basis_1d(sba_freq, image_size)

            while problem.evaluations < problem.max_evaluation:
                indices = list(range(n_dims_sba))
                rnd.shuffle(indices)
                for idx in indices:
                    if problem.evaluations >= problem.max_evaluation:
                        break
                    c = idx // (sba_freq * sba_freq)
                    rem = idx % (sba_freq * sba_freq)
                    r, col = rem // sba_freq, rem % sba_freq
                    delta = sba_step * np.outer(row_basis[r], col_basis[col]).ravel()
                    ch_s = c * n_pixel_per_ch
                    ch_e = (c + 1) * n_pixel_per_ch

                    for sign in (-1, 1):
                        x_try = best_x.copy()
                        x_try[ch_s:ch_e] += sign * delta
                        x_try = self._project_pixel(x_try, epsilon)
                        res = problem.evaluate(x_try.reshape(1, -1))
                        if res[0] is None:
                            return best_x
                        if res[0][0, 0] < best_fitness:
                            candidate_fitness = res[0][0, 0]
                            if candidate_fitness == 0:
                                confirmed, checked_fitness, checked_fit = (
                                    self._verify_success(problem, x_try)
                                )
                                if confirmed:
                                    return x_try
                                if checked_fitness >= best_fitness:
                                    continue
                                candidate_fitness = checked_fitness
                            best_fitness = candidate_fitness
                            best_x = x_try
                            break

        return best_x
