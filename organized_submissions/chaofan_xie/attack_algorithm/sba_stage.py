import math
import random

import numpy as np

from attack_algorithm.attack_algorithm_base import AttackAlgorithmBase
from attack_problem.one_image_problem import SingleImageProblem


class SBAGradientRefiner(AttackAlgorithmBase):
    """
    Clean SBA gradient-refine stage.

    This class extracts the gradient-estimation + line-search part from
    top_region_sba_attack.py. It starts from an optional full-dimensional
    initial_delta and refines it in a low-dimensional full-image latent space.
    """

    def __init__(self, config):
        super().__init__(config)
        self.initial_delta = config.get("initial_delta", None)
        self.initial_labels = config.get("initial_labels", None)
        self.latent_size = int(config.get("latent_size", 224))
        self.directions_per_round = int(config.get("directions_per_round", 60))
        self.step_size = float(config.get("step_size", 1))
        self.step_decay = float(config.get("step_decay", 0.5))
        self.min_step_size = float(config.get("min_step_size", 0.01))
        self.failure_patience = int(config.get("failure_patience", 20))
        self.log_every = int(config.get("log_every", 10))
        self.target_labels = config.get("target_labels", None)
        self.threshold = float(config.get("threshold", 0.5))
        self.gamma_hard = float(config.get("gamma_hard", 3.0))
        self.fitness_weight = float(config.get("fitness_weight", 1e-3))
        self.norm_weight = float(config.get("norm_weight", 1e-4))
        self.unlimited_queries = bool(config.get("unlimited_queries", False))
        self.stop_on_stagnation = bool(config.get("stop_on_stagnation", False))
        self.stagnation_patience = int(config.get("stagnation_patience", 8))
        self.stagnation_eps = float(config.get("stagnation_eps", 1e-4))
        self.last_result = None

    def evolve(self, problem: SingleImageProblem):
        solver = _SBAGradientRefinerSolver(problem, self.rnd, self)
        return solver.run()


class _SBAGradientRefinerSolver:
    def __init__(self, problem: SingleImageProblem, rnd: random.Random, alg: SBAGradientRefiner):
        self.problem = problem
        self.rnd = rnd
        self.alg = alg
        self.np_rng = np.random.default_rng(rnd.randrange(2**32))
        self.full_dim = problem.get_dimension()
        self.channels = 3
        self.image_size = int(round(math.sqrt(self.full_dim / self.channels)))
        self.image_shape = (self.channels, self.image_size, self.image_size)
        self.epsilon = float(problem.epsilon)
        self.latent_size = max(1, int(alg.latent_size))
        self.directions_per_round = max(1, int(alg.directions_per_round))
        self.step_size = max(float(alg.step_size), float(alg.min_step_size))
        self.step_decay = max(float(alg.step_decay), 1e-12)
        self.min_step_size = max(float(alg.min_step_size), 1e-8)
        self.failure_patience = max(1, int(alg.failure_patience))
        self.base_predict = None
        self.ever_failed_labels = set()

    def run(self):
        initial_delta = self._initial_delta()
        initial_delta = self._project(initial_delta.reshape(1, -1))[0]
        fitness, fit, probs = self.problem.evaluate_detail(initial_delta.reshape(1, -1), effective=False)
        if fitness is None or probs is None:
            return initial_delta, self.problem.evaluations

        labels = self._initial_labels(fit[0])
        self.ever_failed_labels = set(int(label) for label in labels)
        if len(labels) == 0:
            print("SBAGradientRefiner: initial delta already succeeds.")
            return initial_delta, self.problem.evaluations

        self.base_predict = probs[0].copy()
        specs, latent_dim = self._latent_specs()
        latent = self._encode_full_to_specs(initial_delta, specs, latent_dim)
        best = self._decode_project(latent, specs)
        best_fitness, best_fit, best_probs = self.problem.evaluate_detail(best.reshape(1, -1), effective=False)
        if best_probs is None:
            best = initial_delta.copy()
            best_fitness = fitness
            best_fit = fit
            best_probs = probs
        best_fitness_value = float(best_fitness[0, 0])
        best_fit_vec = best_fit[0].copy()
        best_probs_vec = best_probs[0].copy()
        best_score = float(
            self._scores(
                best_fitness,
                best_probs,
                best.reshape(1, -1),
                labels,
            )[0]
        )

        print(
            "SBAGradientRefiner start: "
            f"labels={labels}, latent_dim={latent_dim}, current_eval={self.problem.evaluations}, "
            f"initial_score={best_score:.6f}, initial_fitness={best_fitness_value:.6f}, "
            f"radius={float(np.linalg.norm(best)):.6f}"
        )

        iteration = 0
        accepted_count = 0
        failed_rounds = 0
        stop_reason = "budget"
        stagnation_count = 0
        last_max_fit = self._max_label_fit(best_fit_vec, labels)
        initial_stagnation_check = bool(self.alg.stop_on_stagnation)
        while self.alg.unlimited_queries or self.problem.evaluations < self.problem.max_evaluation:
            iteration += 1
            dirs = self._sample_directions(latent_dim)
            probe_latents = []
            for direction in dirs:
                probe_latents.append(np.clip(latent + self.step_size * direction, -1.0, 1.0))
                probe_latents.append(np.clip(latent - self.step_size * direction, -1.0, 1.0))

            probe_full = self._decode_project_batch(np.asarray(probe_latents, dtype=np.float32), specs)
            probe_fitness, probe_fit, probe_probs = self.problem.evaluate_detail(probe_full)
            if probe_probs is None:
                break

            probe_scores = self._scores(probe_fitness, probe_probs, probe_full, labels)
            grad = self._finite_difference_grad(dirs, probe_scores)
            grad_norm = float(np.linalg.norm(grad))

            line_latents = []
            if grad_norm > 0:
                grad_dir = grad / (grad_norm + 1e-12)
                if self.problem.evaluations < 5000:
                    line_muls = [0.4, 0.45, 0.6, 0.7, 1.2, 1.4, 1.6]
                else:
                    line_muls = [0.25, 0.35, 0.45, 0.55, 0.65, 0.75]
                remaining = 10**9 if self.alg.unlimited_queries else max(
                    0, self.problem.max_evaluation - self.problem.evaluations
                )
                for mul in line_muls[:remaining]:
                    cand = np.clip(latent - (mul * self.step_size) * grad_dir, -1.0, 1.0)
                    line_latents.append(cand.astype(np.float32))

            if len(line_latents) > 0:
                line_full = self._decode_project_batch(np.asarray(line_latents, dtype=np.float32), specs)
                line_fitness, line_fit, line_probs = self.problem.evaluate_detail(line_full)
                if line_probs is not None:
                    line_scores = self._scores(line_fitness, line_probs, line_full, labels)
                    probe_latents.extend(line_latents)
                    probe_full = np.concatenate([probe_full, line_full], axis=0)
                    probe_fitness = np.concatenate([probe_fitness, line_fitness], axis=0)
                    probe_fit = np.concatenate([probe_fit, line_fit], axis=0)
                    probe_probs = np.concatenate([probe_probs, line_probs], axis=0)
                    probe_scores = np.concatenate([probe_scores, line_scores], axis=0)

            idx = int(np.argmin(probe_scores))
            accepted = False
            if float(probe_scores[idx]) < best_score:
                latent = np.asarray(probe_latents[idx], dtype=np.float32)
                best = probe_full[idx].copy()
                best_score = float(probe_scores[idx])
                best_fitness_value = float(probe_fitness[idx, 0])
                best_fit_vec = probe_fit[idx].copy()
                best_probs_vec = probe_probs[idx].copy()
                labels = self._dynamic_active_labels(best_fit_vec)
                if len(labels) > 0:
                    best_score = float(
                        self._scores(
                            np.asarray([[best_fitness_value]], dtype=np.float32),
                            best_probs_vec.reshape(1, -1),
                            best.reshape(1, -1),
                            labels,
                        )[0]
                    )
                accepted = True
                accepted_count += 1
                failed_rounds = 0
            else:
                failed_rounds += 1
                if failed_rounds >= self.failure_patience:
                    self.step_size = max(self.step_size * self.step_decay, self.min_step_size)
                    failed_rounds = 0

            if best_fitness_value == 0:
                verify_fitness, verify_fit, verify_probs = self.problem.evaluate_detail(
                    best.reshape(1, -1),
                    effective=False,
                )
                if verify_fitness is not None and float(verify_fitness[0, 0]) == 0:
                    best_fitness_value = float(verify_fitness[0, 0])
                    best_fit_vec = verify_fit[0].copy()
                    best_probs_vec = verify_probs[0].copy()
                    print("SBAGradientRefiner official fitness reached zero.")
                    stop_reason = "success"
                    break
                if verify_fitness is not None:
                    best_fitness_value = float(verify_fitness[0, 0])
                    best_fit_vec = verify_fit[0].copy()
                    best_probs_vec = verify_probs[0].copy()
                    labels = self._dynamic_active_labels(best_fit_vec)
                    if len(labels) > 0:
                        best_score = float(
                            self._scores(
                                np.asarray([[best_fitness_value]], dtype=np.float32),
                                best_probs_vec.reshape(1, -1),
                                best.reshape(1, -1),
                                labels,
                            )[0]
                        )

            if self.alg.log_every > 0 and iteration % self.alg.log_every == 0:
                self._print_status(
                    iteration,
                    best_score,
                    best_fitness_value,
                    best,
                    best_fit_vec,
                    best_probs_vec,
                    labels,
                    accepted,
                    grad_norm,
                    failed_rounds,
                )

            if len(labels) == 0:
                stop_reason = "success"
                break

            current_max_fit = self._max_label_fit(best_fit_vec, labels)
            if initial_stagnation_check:
                if last_max_fit - current_max_fit > self.alg.stagnation_eps:
                    initial_stagnation_check = False
                    last_max_fit = current_max_fit
                else:
                    stagnation_count += 1
                if stagnation_count >= max(1, int(self.alg.stagnation_patience)):
                    stop_reason = "initial_max_fit_stagnation"
                    print(
                        "SBAGradientRefiner stop by initial max_fit stagnation: "
                        f"patience={self.alg.stagnation_patience}, "
                        f"eps={self.alg.stagnation_eps:.6e}, "
                        f"max_fit={current_max_fit:.6f}, "
                        f"Evaluation:{self.problem.evaluations}"
                    )
                    break

        print(
            "SBAGradientRefiner end: "
            f"accepted_count={accepted_count}, eval={self.problem.evaluations}, "
            f"fitness={best_fitness_value:.6f}, radius={float(np.linalg.norm(best)):.6f}, "
            f"stop_reason={stop_reason}"
        )
        self._print_failed_labels(best_fit_vec, best_probs_vec)
        self.alg.last_result = {
            "stop_reason": stop_reason,
            "best_delta": best.copy(),
            "fitness": best_fitness_value,
            "fit": best_fit_vec.copy(),
            "probs": best_probs_vec.copy(),
            "labels": list(labels),
            "max_fit": self._max_label_fit(best_fit_vec, labels),
            "evaluations": self.problem.evaluations,
        }
        return best, self.problem.evaluations

    def _initial_delta(self):
        if self.alg.initial_delta is None:
            return np.zeros((self.full_dim,), dtype=np.float32)
        return np.asarray(self.alg.initial_delta, dtype=np.float32).reshape(-1)

    def _initial_labels(self, fit):
        if self.alg.initial_labels is not None:
            return [int(label) for label in self.alg.initial_labels]
        return self._active_labels_from_fit(fit)

    def _active_labels_from_fit(self, fit):
        fit_vec = np.asarray(fit).reshape(-1)
        if self.alg.target_labels is not None:
            allowed = set(int(label) for label in self.alg.target_labels)
            return [int(i) for i in np.where(fit_vec > 1e-12)[0] if int(i) in allowed]
        return [int(i) for i in np.where(fit_vec > 1e-12)[0]]

    def _dynamic_active_labels(self, fit):
        current_failed = self._active_labels_from_fit(fit)
        self.ever_failed_labels.update(int(label) for label in current_failed)
        if self.alg.target_labels is not None:
            allowed = set(int(label) for label in self.alg.target_labels)
            return sorted(label for label in self.ever_failed_labels if label in allowed)
        return sorted(self.ever_failed_labels)

    def _max_label_fit(self, fit, labels):
        if fit is None or labels is None or len(labels) == 0:
            return 0.0
        fit_vec = np.asarray(fit).reshape(-1)
        values = [max(0.0, float(fit_vec[int(label)])) for label in labels if int(label) < len(fit_vec)]
        if len(values) == 0:
            return 0.0
        return max(values)

    def _latent_specs(self):
        latent_h = min(self.latent_size, self.image_size)
        latent_w = min(self.latent_size, self.image_size)
        region = (0, self.image_size, 0, self.image_size)
        size = self.channels * latent_h * latent_w
        return [(region, latent_h, latent_w, 0, size)], size

    def _sample_directions(self, latent_dim):
        dirs = self.np_rng.normal(0.0, 1.0, size=(self.directions_per_round, latent_dim)).astype(np.float32)
        norms = np.linalg.norm(dirs, axis=1)
        dirs /= np.maximum(norms, 1e-12)[:, None]
        return dirs

    def _finite_difference_grad(self, dirs, scores):
        grad = np.zeros_like(dirs[0], dtype=np.float32)
        for i, direction in enumerate(dirs):
            grad += (
                (float(scores[2 * i]) - float(scores[2 * i + 1]))
                / (2.0 * self.step_size)
            ) * direction
        grad /= max(float(len(dirs)), 1.0)
        return grad.astype(np.float32)

    def _decode_project_batch(self, latent_pop, specs):
        full = np.zeros((len(latent_pop), self.full_dim), dtype=np.float32)
        for row in range(len(latent_pop)):
            full[row] = self._decode(latent_pop[row], specs)
        return self._project(full)

    def _decode_project(self, latent, specs):
        return self._project(self._decode(latent, specs).reshape(1, -1))[0]

    def _decode(self, latent, specs):
        full = np.zeros(self.image_shape, dtype=np.float32)
        for region, latent_h, latent_w, start, end in specs:
            h0, h1, w0, w1 = region
            block = latent[start:end].reshape(self.channels, latent_h, latent_w)
            full[:, h0:h1, w0:w1] += self._resize_nearest(block, h1 - h0, w1 - w0)
        return full.reshape(-1)

    def _encode_full_to_specs(self, full, specs, latent_dim):
        full_arr = np.asarray(full, dtype=np.float32).reshape(self.image_shape)
        encoded = np.zeros((latent_dim,), dtype=np.float32)
        for region, spec_h, spec_w, start, end in specs:
            h0, h1, w0, w1 = region
            block = full_arr[:, h0:h1, w0:w1]
            small = np.zeros((self.channels, spec_h, spec_w), dtype=np.float32)
            for ly in range(spec_h):
                yy0 = int(round(ly * block.shape[1] / spec_h))
                yy1 = int(round((ly + 1) * block.shape[1] / spec_h))
                yy1 = max(yy1, yy0 + 1)
                for lx in range(spec_w):
                    xx0 = int(round(lx * block.shape[2] / spec_w))
                    xx1 = int(round((lx + 1) * block.shape[2] / spec_w))
                    xx1 = max(xx1, xx0 + 1)
                    small[:, ly, lx] = np.mean(block[:, yy0:yy1, xx0:xx1], axis=(1, 2))
            encoded[start:end] = small.reshape(-1)
        return np.clip(encoded, -1.0, 1.0)

    def _resize_nearest(self, block, out_h, out_w):
        _, in_h, in_w = block.shape
        y_idx = np.floor(np.arange(out_h) * in_h / out_h).astype(np.int64)
        x_idx = np.floor(np.arange(out_w) * in_w / out_w).astype(np.int64)
        return block[:, y_idx][:, :, x_idx]

    def _project(self, pop):
        projected = np.asarray(pop, dtype=np.float32).copy()
        norms = np.linalg.norm(projected, axis=1)
        safe_epsilon = self.epsilon * 0.9999
        too_large = norms > safe_epsilon
        if np.any(too_large):
            projected[too_large] *= (safe_epsilon / np.maximum(norms[too_large], 1e-12))[:, None]
        return np.clip(projected, -1.0, 1.0)

    def _scores(self, fitness, predict, probe_full, labels):
        show_labels, hide_labels = self._get_show_hide_labels(labels)
        target_margin = 0.02
        fitness_flat = np.asarray(fitness).reshape(-1)
        scores = []
        for b in range(predict.shape[0]):
            conf = predict[b]
            target_losses = []
            attack_loss = 0.0
            for label in show_labels:
                loss = max(0.0, self.alg.threshold + target_margin - float(conf[label]))
                attack_loss += loss
                target_losses.append(loss)
            for label in hide_labels:
                loss = max(0.0, float(conf[label]) - (self.alg.threshold - target_margin))
                attack_loss += loss
                target_losses.append(loss)
            hard_loss = max(target_losses) if len(target_losses) > 0 else 0.0
            official_loss = float(fitness_flat[b]) if b < len(fitness_flat) else 0.0
            if official_loss < 0.05:
                score = official_loss
            else:
                score = attack_loss + self.alg.gamma_hard * hard_loss + self.alg.fitness_weight * official_loss
            if probe_full is not None and self.alg.norm_weight > 0:
                score += self.alg.norm_weight * float(np.linalg.norm(probe_full[b]))
            scores.append(float(score))
        return np.asarray(scores, dtype=np.float32)

    def _get_show_hide_labels(self, labels):
        target = np.asarray(self.problem.y_target).reshape(-1)
        show_labels = []
        hide_labels = []
        for label in labels:
            if int(target[int(label)]) == 1:
                show_labels.append(int(label))
            else:
                hide_labels.append(int(label))
        return show_labels, hide_labels

    def _print_status(self, iteration, score, fitness, best, fit, probs, labels, accepted, grad_norm, failed_rounds):
        target = np.asarray(self.problem.y_target).reshape(-1)
        label_text = " | ".join(
            f"label {i}: fit={float(fit[i]):.6f}, prob={float(probs[i]):.6f}, target={int(target[i])}"
            for i in labels
        )
        print(
            f"SBAGradientRefiner Evaluation:{self.problem.evaluations}, Iteration:{iteration}, "
            f"score:{score:.6f}, fitness:{fitness:.6f}, radius:{float(np.linalg.norm(best)):.6f}, "
            f"step:{self.step_size:.6f}, grad_norm:{grad_norm:.6f}, accepted:{int(accepted)}, "
            f"failed_rounds:{failed_rounds}, {label_text}"
        )

    def _print_failed_labels(self, fit, probs):
        if fit is None or probs is None:
            return
        target = np.asarray(self.problem.y_target).reshape(-1)
        failed = np.where(np.asarray(fit).reshape(-1) > 1e-12)[0]
        if len(failed) == 0:
            print("SBAGradientRefiner final failed labels: none.")
            return
        failed = sorted(failed, key=lambda i: float(fit[i]), reverse=True)
        print(f"SBAGradientRefiner final failed labels top {min(20, len(failed))}/{len(failed)}:")
        for label in failed[:20]:
            print(
                f"  label {int(label)}: fit={float(fit[label]):.6f}, "
                f"prob={float(probs[label]):.6f}, target={int(target[label])}"
            )
