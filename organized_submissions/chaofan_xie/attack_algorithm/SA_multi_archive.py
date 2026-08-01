import math
import random

import numpy as np

from attack_algorithm.attack_algorithm_base import AttackAlgorithmBase
from attack_algorithm.sba_stage import SBAGradientRefiner
from attack_problem.one_image_problem import SingleImageProblem


class SA_MultiArchive(AttackAlgorithmBase):
    """
    Square-attack style random search with multi-objective archives.

    Each iteration evaluates one candidate only. The candidate parent is sampled
    from several archives, and the single evaluation is reused to update all
    objectives: total fitness, bottleneck fit, balanced score, and per-label fit.
    """

    def __init__(self, config):
        super().__init__(config)
        self.init_scale = float(config.get("init_scale", 0.02))
        self.patch_min = int(config.get("patch_min", 8))
        self.patch_max_ratio = float(config.get("patch_max_ratio", 0.35))
        self.patch_value = float(config.get("patch_value", 0.18))
        self.gamma_bottleneck = float(config.get("gamma_bottleneck", 3.5))
        self.threshold = float(config.get("threshold", 0.5))
        self.fitness_weight = float(config.get("fitness_weight", 1e-3))
        self.norm_weight = float(config.get("norm_weight", 1e-5))
        self.protect_weight = float(config.get("protect_weight", 5.0))
        self.protect_tol = float(config.get("protect_tol", 0.03))
        self.parent_current_prob = float(config.get("parent_current_prob", 0.5))
        self.parent_bottleneck_prob = float(config.get("parent_bottleneck_prob", 0.2))
        self.parent_total_prob = float(config.get("parent_total_prob", 0.08))
        self.log_every = int(config.get("log_every", 100))
        self.epsilon_margin = float(config.get("epsilon_margin", 1e-4))
        self.plateau_fit_threshold = float(config.get("plateau_fit_threshold", 0.499))
        self.label_best_eps = float(config.get("label_best_eps", 1e-6))
        self.label_improve_eps = float(config.get("label_improve_eps", 1e-6))
        self.plateau_patience = int(config.get("plateau_patience", 800))
        self.plateau_escape_budget = int(config.get("plateau_escape_budget", 1200))
        self.escape_patch_max_ratio = float(config.get("escape_patch_max_ratio", 0.65))
        self.escape_patch_value = float(config.get("escape_patch_value", 0.30))
        self.escape_sum_weight = float(config.get("escape_sum_weight", 0.1))
        self.escape_fitness_weight = float(config.get("escape_fitness_weight", 1e-4))
        self.escape_norm_weight = float(config.get("escape_norm_weight", 1e-6))
        self.escape_lowfreq_prob = float(config.get("escape_lowfreq_prob", 0.7))
        self.escape_lowfreq_size = int(config.get("escape_lowfreq_size", 8))
        self.escape_lowfreq_value = float(config.get("escape_lowfreq_value", 0.35))
        self.escape_micro_eps = float(config.get("escape_micro_eps", 1e-6))
        self.sba_to_sa_stagnation_patience = int(config.get("sba_to_sa_stagnation_patience", 15))
        self.sba_to_sa_stagnation_eps = float(config.get("sba_to_sa_stagnation_eps", 0.03))
        self.sba_switch_detect = bool(config.get("sba_switch_detect", True))
        self.sba_switch_eval = int(config.get("sba_switch_eval", 50))
        self.sba_start_max_fit = float(config.get("sba_start_max_fit", 0.499))
        self.sba_single_start_max_fit = float(config.get("sba_single_start_max_fit", 0.49))
        self.sba_min_remaining = int(config.get("sba_min_remaining", 1500))
        self.sba_switch_auto = bool(config.get("sba_switch_auto", True))
        self.sba_refine_latent_size = int(config.get("sba_refine_latent_size", 64))
        self.sba_refine_directions_per_round = int(config.get("sba_refine_directions_per_round", 50))
        self.sba_refine_step_size = float(config.get("sba_refine_step_size", 1))
        self.sba_refine_step_decay = float(config.get("sba_refine_step_decay", 0.5))
        self.sba_refine_min_step_size = float(config.get("sba_refine_min_step_size", 0.01))
        self.sba_refine_failure_patience = int(config.get("sba_refine_failure_patience", 20))
        self.sba_refine_log_every = int(config.get("sba_refine_log_every", 3))

    def evolve(self, problem: SingleImageProblem):
        solver = _SAMultiArchiveSolver(problem, self.rnd, self)
        return solver.run()


class _SAMultiArchiveSolver:
    def __init__(self, problem: SingleImageProblem, rnd: random.Random, alg: SA_MultiArchive):
        self.problem = problem
        self.rnd = rnd
        self.alg = alg
        self.dim = problem.get_dimension()
        image = np.asarray(problem.image, dtype=np.float32)
        self.image_shape = image.shape
        self.channels = int(image.shape[0])
        self.image_size = int(image.shape[1])
        margin = min(max(float(alg.epsilon_margin), 0.0), 0.1)
        self.epsilon = float(problem.epsilon) * (1.0 - margin)
        self.last_sba_result = None

    def run(self):
        zero = np.zeros((1, self.dim), dtype=np.float32)
        fitness, fit, probs = self.problem.evaluate_detail(zero)
        if fitness is None or fit is None:
            return np.zeros((self.dim,), dtype=np.float32), self.problem.evaluations

        labels = self._failed_labels(fit[0])
        if len(labels) == 0:
            print("SA_MultiArchive: no failed labels need optimization.")
            return zero[0], self.problem.evaluations

        print(
            "SA_MultiArchive initial route: SBA first, "
            f"labels={labels}, initial_fitness={float(fitness[0, 0]):.6f}, "
            f"max_fit={float(np.max(np.asarray(fit[0])[labels])):.6f}"
        )
        sba_delta, _ = self._run_initial_sba_refiner(
            zero[0],
            labels,
            fit[0],
            float(fitness[0, 0]),
            stop_on_stagnation=True,
        )
        if self.last_sba_result is None:
            return sba_delta, self.problem.evaluations
        if self.last_sba_result.get("stop_reason") != "initial_max_fit_stagnation":
            return sba_delta, self.problem.evaluations

        fitness, fit, probs = self.problem.evaluate_detail(sba_delta.reshape(1, -1), effective=False)
        if fitness is None or fit is None:
            return sba_delta, self.problem.evaluations
        labels = self._failed_labels(fit[0])
        if len(labels) == 0:
            print("SA_MultiArchive: SBA stagnation return already succeeds.")
            return sba_delta, self.problem.evaluations
        print(
            "SA_MultiArchive switch to SA after SBA stagnation: "
            f"labels={labels}, fitness={float(fitness[0, 0]):.6f}, "
            f"max_fit={float(np.max(np.asarray(fit[0])[labels])):.6f}, "
            f"Evaluation:{self.problem.evaluations}"
        )

        current = sba_delta.copy()
        current_fitness, current_fit, current_probs = fitness, fit, probs

        current_state = self._state(current, current_fitness[0], current_fit[0], current_probs[0], labels)
        zero_state = current_state.copy()

        total_best = current_state.copy()
        bottleneck_best = current_state.copy()
        balanced_best = current_state.copy()
        label_best = {int(label): current_state.copy() for label in labels}
        label_monitor = self._init_label_monitor(labels, zero_state, current_state)
        state = "SA_ARCHIVE_SEARCH"
        escape_label = None
        escape_start_eval = None
        escape_current_score = None
        portal_events = []
        sba_ready_reported = False

        iteration = 0
        while self.problem.evaluations < self.problem.max_evaluation:
            iteration += 1
            if state == "PLATEAU_ESCAPE":
                candidate = self._escape_candidate(
                    iteration,
                    current_state,
                    total_best,
                    bottleneck_best,
                )
            else:
                parent = self._select_parent(current_state, total_best, bottleneck_best, label_best)
                candidate = self._square_mutation(parent["delta"], iteration)
            cand_fitness, cand_fit, cand_probs = self.problem.evaluate_detail(candidate.reshape(1, -1))
            if cand_fitness is None:
                break

            cand_state = self._state(candidate, cand_fitness[0], cand_fit[0], cand_probs[0], labels)
            self._update_label_monitor(label_monitor, cand_state)


            if state == "PLATEAU_ESCAPE":
                cand_escape_score = self._escape_score(cand_state, escape_label)
                if escape_current_score is None:
                    escape_current_score = self._escape_score(current_state, escape_label)
                cand_label_fit = float(cand_state["label_fit"][int(escape_label)])
                current_label_fit = float(current_state["label_fit"][int(escape_label)])
                if (
                    cand_label_fit < current_label_fit - self.alg.escape_micro_eps
                    or cand_escape_score < escape_current_score
                ):
                    current_state = cand_state
                    escape_current_score = cand_escape_score
            else:
                if cand_state["balanced_score"] < current_state["balanced_score"]:
                    current_state = cand_state
            if cand_state["balanced_score"] < balanced_best["balanced_score"]:
                balanced_best = cand_state.copy()
            if cand_state["fitness"] < total_best["fitness"]:
                total_best = cand_state.copy()
            if cand_state["max_fit"] < bottleneck_best["max_fit"]:
                bottleneck_best = cand_state.copy()

            for label in labels:
                label = int(label)
                if cand_state["label_fit"][label] < label_best[label]["label_fit"][label] - self.alg.label_best_eps:
                    label_best[label] = cand_state.copy()
                if label_best[label]["label_fit"][label] < 1e-12:
                    label_best[label] = cand_state.copy()

            archive_candidates = self._archive_candidates(
                current_state,
                balanced_best,
                total_best,
                bottleneck_best,
                label_best,
            )
            best_return = self._select_return_state([state for _, state in archive_candidates])
            sba_start_name, sba_start_state = self._select_sba_start_archive(archive_candidates)

            if state == "PLATEAU_ESCAPE":
                exit_reason = self._escape_exit_reason(label_monitor, escape_label, escape_start_eval)
                if exit_reason is not None:
                    if exit_reason == "portal_found":
                        item = label_monitor[int(escape_label)]
                        portal_events.append({
                            "label": int(escape_label),
                            "initial_fit": float(item["initial_fit"]),
                            "best_fit": float(item["best_fit"]),
                            "eval": int(self.problem.evaluations),
                            "escape_trials": int(item["escape_trials"]),
                        })
                    print(
                        "SA退出PLATEAU_ESCAPE: "
                        f"label={escape_label}, reason={exit_reason}, "
                        f"best_fit={label_monitor[escape_label]['best_fit']:.6f}, "
                        f"Evaluation:{self.problem.evaluations}"
                    )
                    state = "SA_ARCHIVE_SEARCH"
                    escape_label = None
                    escape_start_eval = None
                    escape_current_score = None
            else:
                next_escape_label = self._select_plateau_label(label_monitor)
                if next_escape_label is not None:
                    state = "PLATEAU_ESCAPE"
                    escape_label = int(next_escape_label)
                    escape_start_eval = self.problem.evaluations
                    escape_current_score = self._escape_score(current_state, escape_label)
                    label_monitor[escape_label]["escape_trials"] += 1
                    stuck = self.problem.evaluations - label_monitor[escape_label]["best_eval"]
                    print(
                        "SA进入PLATEAU_ESCAPE: "
                        f"label={escape_label}, "
                        f"initial_fit={label_monitor[escape_label]['initial_fit']:.6f}, "
                        f"best_fit={label_monitor[escape_label]['best_fit']:.6f}, "
                        f"stuck_queries={stuck}, Evaluation:{self.problem.evaluations}"
                    )

            if (
                False
                and
                self.alg.sba_switch_detect
                and not sba_ready_reported
                and state == "SA_ARCHIVE_SEARCH"
            ):
                ready_info = self._fixed_sba_switch_ready(sba_start_state)
                if ready_info is not None:
                    ready_info["start_archive"] = sba_start_name
                    self._print_sba_ready(ready_info, sba_start_state, labels)
                    sba_ready_reported = True
                    if self.alg.sba_switch_auto:
                        self._print_portal_events(portal_events)
                        return self._run_sba_refiner(sba_start_name, sba_start_state)

            if self.alg.log_every > 0 and iteration % self.alg.log_every == 0:
                self._print_status(
                    iteration,
                    current_state,
                    balanced_best,
                    total_best,
                    bottleneck_best,
                    label_best,
                    best_return,
                    labels,
                    state,
                    escape_label,
                )

            if best_return["fitness"] <= 0.0:
                print(
                    "SA_MultiArchive success: "
                    f"Evaluation:{self.problem.evaluations}, "
                    f"fitness={best_return['fitness']:.6f}, radius={best_return['radius']:.6f}"
                )
                self._print_portal_events(portal_events)
                return best_return["delta"], self.problem.evaluations

        candidates = [current_state, balanced_best, total_best, bottleneck_best] + list(label_best.values())
        best_return = self._select_return_state(candidates)
        print(
            "SA_MultiArchive final: "
            f"Evaluation:{self.problem.evaluations}, fitness={best_return['fitness']:.6f}, "
            f"failed={best_return['failed_count']}, max_fit={best_return['max_fit']:.6f}, "
            f"sum_fit={best_return['sum_fit']:.6f}, radius={best_return['radius']:.6f}"
        )
        self._print_portal_events(portal_events)
        return best_return["delta"], self.problem.evaluations

    def _random_initial_delta(self):
        delta = np.zeros((self.dim,), dtype=np.float32)
        if self.alg.init_scale > 0:
            delta = np.asarray(
                [self.rnd.uniform(-self.alg.init_scale, self.alg.init_scale) for _ in range(self.dim)],
                dtype=np.float32,
            )
        return self._project(delta)

    def _square_mutation(self, parent, iteration, patch_max_ratio=None, patch_value=None):
        arr = np.asarray(parent, dtype=np.float32).reshape(self.image_shape).copy()
        side = self._patch_side(iteration, patch_max_ratio=patch_max_ratio)
        patch_value = self.alg.patch_value if patch_value is None else float(patch_value)
        y0 = self.rnd.randint(0, max(0, self.image_size - side))
        x0 = self.rnd.randint(0, max(0, self.image_size - side))
        mode = self.rnd.random()
        if mode < 0.5:
            value = self.rnd.uniform(-patch_value, patch_value)
            arr[:, y0:y0 + side, x0:x0 + side] += value
        else:
            patch = np.asarray(
                [
                    self.rnd.uniform(-patch_value, patch_value)
                    for _ in range(self.channels * side * side)
                ],
                dtype=np.float32,
            ).reshape(self.channels, side, side)
            arr[:, y0:y0 + side, x0:x0 + side] = patch
        return self._project(arr.reshape(-1))

    def _patch_side(self, iteration, patch_max_ratio=None):
        ratio = self.alg.patch_max_ratio if patch_max_ratio is None else float(patch_max_ratio)
        max_side = max(self.alg.patch_min, int(round(self.image_size * ratio)))
        progress = self.problem.evaluations / max(float(self.problem.max_evaluation), 1.0)
        decay = max(0.15, 1.0 - progress)
        side = int(round(max_side * decay))
        side = max(self.alg.patch_min, min(max_side, side))
        return min(side, self.image_size)

    def _project(self, delta):
        delta = np.asarray(delta, dtype=np.float32).reshape(-1)
        delta = np.clip(delta, -1.0, 1.0)
        norm = float(np.linalg.norm(delta))
        if norm > self.epsilon:
            delta = delta * (self.epsilon / (norm + 1e-12))
            norm = float(np.linalg.norm(delta))
            if norm > self.epsilon:
                delta = delta * ((self.epsilon - 1e-6) / (norm + 1e-12))
        return delta.astype(np.float32)

    def _state(self, delta, fitness, fit, probs, labels):
        fit_vec = np.asarray(fit, dtype=np.float32).reshape(-1)
        label_fit = {int(label): max(0.0, float(fit_vec[int(label)])) for label in labels}
        values = np.asarray([label_fit[int(label)] for label in labels], dtype=np.float32)
        failed_count = int(np.sum(values > 1e-12))
        sum_fit = float(np.sum(values))
        max_fit = float(np.max(values)) if len(values) > 0 else 0.0
        radius = float(np.linalg.norm(delta))
        protect = self._protect_penalty(label_fit, labels)
        official = float(np.asarray(fitness).reshape(-1)[0])
        balanced = (
            sum_fit
            + self.alg.gamma_bottleneck * max_fit
            + self.alg.fitness_weight * official
            + self.alg.norm_weight * radius
            + protect
        )
        return {
            "delta": np.asarray(delta, dtype=np.float32).reshape(-1).copy(),
            "fitness": official,
            "fit": fit_vec.copy(),
            "probs": np.asarray(probs, dtype=np.float32).reshape(-1).copy(),
            "label_fit": label_fit,
            "failed_count": failed_count,
            "sum_fit": sum_fit,
            "max_fit": max_fit,
            "radius": radius,
            "balanced_score": float(balanced),
        }

    def _protect_penalty(self, label_fit, labels):
        # Current SA archive version uses balanced current as the protection reference.
        # The per-label archive already keeps single-label gains from being lost.
        return 0.0

    def _select_parent(self, current, total_best, bottleneck_best, label_best):
        u = self.rnd.random()
        p_current = self.alg.parent_current_prob
        p_bottleneck = p_current + self.alg.parent_bottleneck_prob
        p_total = p_bottleneck + self.alg.parent_total_prob
        if u < p_current:
            return current
        if u < p_bottleneck:
            return bottleneck_best
        if u < p_total:
            return total_best
        return self.rnd.choice(list(label_best.values()))

    def _select_escape_parent(self, current, total_best, bottleneck_best):
        u = self.rnd.random()
        if u < 0.60:
            return current
        if u < 0.90:
            return bottleneck_best
        return total_best

    def _escape_candidate(self, iteration, current, total_best, bottleneck_best):
        parent = self._select_escape_parent(current, total_best, bottleneck_best)
        return self._escape_direct_mutation(parent["delta"], iteration)

    def _escape_direct_mutation(self, parent_delta, iteration):
        if self.rnd.random() < self.alg.escape_lowfreq_prob:
            return self._lowfreq_mutation(parent_delta)
        return self._square_mutation(
            parent_delta,
            iteration,
            patch_max_ratio=self.alg.escape_patch_max_ratio,
            patch_value=self.alg.escape_patch_value,
        )

    def _lowfreq_mutation(self, parent_delta):
        arr = np.asarray(parent_delta, dtype=np.float32).reshape(self.image_shape).copy()
        size = max(1, int(self.alg.escape_lowfreq_size))
        low = np.asarray(
            [
                self.rnd.uniform(-1.0, 1.0)
                for _ in range(self.channels * size * size)
            ],
            dtype=np.float32,
        ).reshape(self.channels, size, size)
        if self.rnd.random() < 0.35:
            channel = self.rnd.randrange(self.channels)
            mask = np.zeros_like(low, dtype=np.float32)
            mask[channel] = low[channel]
            low = mask
        y_idx = np.floor(np.arange(self.image_size) * size / self.image_size).astype(np.int64)
        x_idx = np.floor(np.arange(self.image_size) * size / self.image_size).astype(np.int64)
        full = low[:, y_idx][:, :, x_idx]
        sign = -1.0 if self.rnd.random() < 0.5 else 1.0
        arr += sign * self.alg.escape_lowfreq_value * full
        return self._project(arr.reshape(-1))

   
    def _init_label_monitor(self, labels, zero_state, current_state):
        monitor = {}
        current_eval = self.problem.evaluations
        for label in labels:
            label = int(label)
            zero_fit = float(zero_state["label_fit"][label])
            current_fit = float(current_state["label_fit"][label])
            best_fit = min(zero_fit, current_fit)
            monitor[label] = {
                "initial_fit": zero_fit,
                "best_fit": best_fit,
                "best_eval": current_eval,
                "improved": best_fit <= zero_fit - self.alg.label_improve_eps,
                "escape_trials": 0,
            }
        return monitor

    def _update_label_monitor(self, monitor, state):
        current_eval = self.problem.evaluations
        for label, item in monitor.items():
            fit_value = float(state["label_fit"][int(label)])
            if fit_value < float(item["best_fit"]) - self.alg.label_best_eps:
                item["best_fit"] = fit_value
                item["best_eval"] = current_eval
                if fit_value <= float(item["initial_fit"]) - self.alg.label_improve_eps:
                    item["improved"] = True

    def _select_plateau_label(self, monitor):
        candidates = []
        current_eval = self.problem.evaluations
        for label, item in monitor.items():
            initial_fit = float(item["initial_fit"])
            best_fit = float(item["best_fit"])
            stuck_queries = current_eval - int(item["best_eval"])
            if initial_fit < self.alg.plateau_fit_threshold:
                continue
            if best_fit < initial_fit - self.alg.label_improve_eps:
                continue
            if stuck_queries < self.alg.plateau_patience:
                continue
            candidates.append((int(label), best_fit, stuck_queries, int(item["escape_trials"])))
        if len(candidates) == 0:
            return None
        candidates.sort(key=lambda x: (-x[1], -x[2], x[3]))
        return candidates[0][0]

    def _escape_exit_reason(self, monitor, label, start_eval):
        if label is None:
            return "no_label"
        item = monitor[int(label)]
        if float(item["best_fit"]) <= 1e-12:
            return "label_success"
        if bool(item["improved"]):
            return "portal_found"
        if start_eval is not None and self.problem.evaluations - int(start_eval) >= self.alg.plateau_escape_budget:
            return "budget_used"
        return None

    def _escape_score(self, state, label):
        label = int(label)
        return float(
            state["label_fit"][label]
            + self.alg.escape_sum_weight * state["sum_fit"]
            + self.alg.escape_fitness_weight * state["fitness"]
            + self.alg.escape_norm_weight * state["radius"]
        )

    def _fixed_sba_switch_ready(self, start_state):
        remaining = self.problem.max_evaluation - self.problem.evaluations
        if self.problem.evaluations < self.alg.sba_switch_eval:
            return None
        if remaining < self.alg.sba_min_remaining:
            return None

        active_labels = [
            int(label)
            for label, value in start_state["label_fit"].items()
            if float(value) > 1e-12
        ]
        if len(active_labels) == 0:
            return None

        max_fit = float(start_state["max_fit"])
        if len(active_labels) == 1:
            if max_fit <= self.alg.sba_single_start_max_fit:
                return {
                    "reason": "fixed_scan_single",
                    "active_labels": active_labels,
                    "ready_labels": active_labels,
                    "remaining": int(remaining),
                    "max_fit": max_fit,
                }
            return None

        if max_fit <= self.alg.sba_start_max_fit:
            return {
                "reason": "fixed_scan_multi",
                "active_labels": active_labels,
                "ready_labels": active_labels,
                "remaining": int(remaining),
                "max_fit": max_fit,
            }
        return None

    def _select_return_state(self, states):
        return min(
            states,
            key=lambda s: (
                int(s["failed_count"]),
                float(s["max_fit"]),
                float(s["sum_fit"]),
                float(s["fitness"]),
                float(s["radius"]),
            ),
        )

    def _archive_candidates(self, current, balanced_best, total_best, bottleneck_best, label_best):
        candidates = [
            ("current", current),
            ("balanced_best", balanced_best),
            ("total_best", total_best),
            ("bottleneck_best", bottleneck_best),
        ]
        for label, state in label_best.items():
            candidates.append((f"label_best[{int(label)}]", state))
        return candidates

    def _select_sba_start_archive(self, candidates):
        return min(
            candidates,
            key=lambda item: (
                float(item[1]["max_fit"]),
                float(item[1]["sum_fit"]),
                int(item[1]["failed_count"]),
                float(item[1]["fitness"]),
                float(item[1]["radius"]),
            ),
        )

    def _run_sba_refiner(self, start_name, start_state, stop_on_stagnation=False):
        start_labels = [
            int(label)
            for label, value in start_state["label_fit"].items()
            if float(value) > 1e-12
        ]
        print(
            "SA切换到SBAGradientRefiner: "
            f"start_archive={start_name}, "
            f"labels={start_labels}, "
            f"start_failed={int(start_state['failed_count'])}, "
            f"start_max_fit={float(start_state['max_fit']):.6f}, "
            f"start_sum_fit={float(start_state['sum_fit']):.6f}, "
            f"start_fitness={float(start_state['fitness']):.6f}, "
            f"start_radius={float(start_state['radius']):.6f}, "
            f"Evaluation:{self.problem.evaluations}"
        )
        refiner = SBAGradientRefiner({
            "rnd": self.rnd,
            "initial_delta": start_state["delta"].copy(),
            "initial_labels": start_labels,
            "latent_size": self.alg.sba_refine_latent_size,
            "directions_per_round": self.alg.sba_refine_directions_per_round,
            "step_size": self.alg.sba_refine_step_size,
            "step_decay": self.alg.sba_refine_step_decay,
            "min_step_size": self.alg.sba_refine_min_step_size,
            "failure_patience": self.alg.sba_refine_failure_patience,
            "log_every": self.alg.sba_refine_log_every,
            "threshold": self.alg.threshold,
            "gamma_hard": self.alg.gamma_bottleneck,
            "fitness_weight": self.alg.fitness_weight,
            "norm_weight": self.alg.norm_weight,
            "stop_on_stagnation": stop_on_stagnation,
            "stagnation_patience": self.alg.sba_to_sa_stagnation_patience,
            "stagnation_eps": self.alg.sba_to_sa_stagnation_eps,
        })
        refiner.rnd = self.rnd
        result = refiner.evolve(self.problem)
        self.last_sba_result = getattr(refiner, "last_result", None)
        return result

    def _run_initial_sba_refiner(self, initial_delta, labels, initial_fit, initial_fitness, stop_on_stagnation=False):
        initial_delta = np.asarray(initial_delta, dtype=np.float32).reshape(-1)
        fit_vec = np.asarray(initial_fit, dtype=np.float32).reshape(-1)
        label_fit = {int(label): max(0.0, float(fit_vec[int(label)])) for label in labels}
        fit_values = list(label_fit.values())
        state = {
            "delta": initial_delta.copy(),
            "label_fit": label_fit,
            "failed_count": len(labels),
            "max_fit": max(fit_values) if len(fit_values) > 0 else 0.0,
            "sum_fit": sum(fit_values),
            "fitness": float(initial_fitness),
            "radius": float(np.linalg.norm(initial_delta)),
        }
        print(
            "SA_MultiArchive initial route enters SBAGradientRefiner: "
            f"labels={labels}, Evaluation:{self.problem.evaluations}"
        )
        return self._run_sba_refiner("initial_zero", state, stop_on_stagnation=stop_on_stagnation)

    def _failed_labels(self, fit):
        fit_vec = np.asarray(fit).reshape(-1)
        return [int(i) for i in np.where(fit_vec > 1e-12)[0]]

    def _print_status(
        self,
        iteration,
        current,
        balanced_best,
        total_best,
        bottleneck_best,
        label_best,
        best_return,
        labels,
        state="SA_ARCHIVE_SEARCH",
        escape_label=None,
    ):
        parts = []
        for label in labels:
            label = int(label)
            parts.append(f"label {label}: fit={best_return['label_fit'][label]:.6f}")
        escape_text = "" if escape_label is None else f", escape_label:{int(escape_label)}"
        print(
            f"Evaluation:{self.problem.evaluations}, Iteration:{iteration}, "
            f"state:{state}{escape_text}, "
            f"current_score:{current['balanced_score']:.6f}, "
            f"best_failed:{best_return['failed_count']}, "
            f"best_max_fit:{best_return['max_fit']:.6f}, "
            f"best_fitness:{best_return['fitness']:.6f}, "
            f"radius:{best_return['radius']:.6f}, "
            + " | ".join(parts)
        )
        for label in labels:
            label = int(label)
            self._print_archive_detail(
                f"label_best[{label}]",
                label_best[label],
                labels,
            )

    def _print_archive_detail(self, name, archive_state, labels):
        label_parts = []
        for label in labels:
            label = int(label)
            label_parts.append(f"L{label}={archive_state['label_fit'][label]:.6f}")
        print(
            "  archive "
            f"{name}: "
            f"failed={int(archive_state['failed_count'])}, "
            f"max_fit={float(archive_state['max_fit']):.6f}, "
            f"sum_fit={float(archive_state['sum_fit']):.6f}, "
            f"fitness={float(archive_state['fitness']):.6f}, "
            f"score={float(archive_state['balanced_score']):.6f}, "
            f"radius={float(archive_state['radius']):.6f}, "
            + " | ".join(label_parts)
        )

    def _print_portal_events(self, portal_events):
        if len(portal_events) == 0:
            print("SA_MultiArchive portal记录: 本图未出现PLATEAU_ESCAPE portal_found。")
            return
        print(f"SA_MultiArchive portal记录: 本图出现 {len(portal_events)} 次PLATEAU_ESCAPE portal_found。")
        for event in portal_events:
            gain = float(event["initial_fit"]) - float(event["best_fit"])
            print(
                "  portal "
                f"label={int(event['label'])}, "
                f"Evaluation={int(event['eval'])}, "
                f"initial_fit={float(event['initial_fit']):.6f}, "
                f"best_fit={float(event['best_fit']):.6f}, "
                f"gain={gain:.6e}, "
                f"escape_trials={int(event['escape_trials'])}"
            )

    def _print_sba_ready(self, ready_info, best_return, labels):
        fit_parts = []
        for label in labels:
            label = int(label)
            if label in best_return["label_fit"]:
                fit_parts.append(f"label {label}: fit={best_return['label_fit'][label]:.6f}")
        active_text = " ".join(str(label) for label in ready_info["active_labels"])
        ready_text = " ".join(str(label) for label in ready_info["ready_labels"])
        print(
            "SA建议切换SBA: "
            f"reason={ready_info['reason']}, "
            f"Evaluation:{self.problem.evaluations}, "
            f"remaining={ready_info['remaining']}, "
            f"max_fit={ready_info['max_fit']:.6f}, "
            f"active_labels=[{active_text}], "
            f"ready_labels=[{ready_text}], "
            f"start_archive={ready_info.get('start_archive', 'unknown')}, "
            f"start_failed={best_return['failed_count']}, "
            f"start_sum_fit={best_return['sum_fit']:.6f}, "
            f"start_fitness={best_return['fitness']:.6f}, "
            f"start_radius={best_return['radius']:.6f}, "
            + " | ".join(fit_parts)
        )
