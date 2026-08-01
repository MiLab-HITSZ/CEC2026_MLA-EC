import numpy as np
import torch
import torch.nn.functional as F

from attack_algorithm.attack_algorithm_base import AttackAlgorithmBase
from attack_problem.one_image_problem import SingleImageProblem


class StructuredActiveLabelAttack(AttackAlgorithmBase):
    def __init__(self, config):
        super().__init__(config)

        self.constraint = str(config.get("constraint", "l2"))
        self.verbose = int(config.get("verbose", 1))
        self.project_with_clip = bool(config.get("project_with_clip", True))
        self.clip_min = float(config.get("clip_min", 0.0))
        self.clip_max = float(config.get("clip_max", 1.0))
        self.eval_batch_size = int(config.get("eval_batch_size", 8))

        self.init_candidates = int(config.get("init_candidates", 18))
        self.init_radii = [float(v) for v in config.get("init_radii", [0.20, 0.45, 0.70, 0.92])]

        self.n_directions = int(config.get("n_directions", 10))
        self.sigma_init = float(config.get("sigma_init", 0.085))
        self.sigma_min = float(config.get("sigma_min", 0.010))
        self.sigma_max = float(config.get("sigma_max", 0.18))
        self.step_init = float(config.get("step_init", 0.18))
        self.step_min = float(config.get("step_min", 0.025))
        self.step_max = float(config.get("step_max", 0.30))
        self.line_search = [float(v) for v in config.get("line_search", [1.30, 0.90, 0.55, 0.25])]

        self.active_topk = int(config.get("active_topk", 3))
        self.memory_mix_every = int(config.get("memory_mix_every", 6))
        self.restart_patience = int(config.get("restart_patience", 12))
        self.max_restarts = int(config.get("max_restarts", 7))
        self.log_every = int(config.get("log_every", 8))

        self.coarse_parts = [int(v) for v in config.get("coarse_parts", [2, 4, 8])]
        self.fine_parts = [int(v) for v in config.get("fine_parts", [8, 16, 32])]
        self.dct_freq_coarse = int(config.get("dct_freq_coarse", 4))
        self.dct_freq_fine = int(config.get("dct_freq_fine", 10))

        self.rescue_enable = bool(config.get("rescue_enable", True))
        self.rescue_pos_threshold = int(config.get("rescue_pos_threshold", 2))
        self.rescue_trigger_patience = int(config.get("rescue_trigger_patience", 6))
        self.rescue_cooldown = int(config.get("rescue_cooldown", 20))
        self.max_rescue_runs = int(config.get("max_rescue_runs", 4))
        self.rescue_basis_size = int(config.get("rescue_basis_size", 32))
        self.rescue_coord_batch = int(config.get("rescue_coord_batch", 10))
        self.rescue_shuffle_every = int(config.get("rescue_shuffle_every", 32))
        self.rescue_query_cap = int(config.get("rescue_query_cap", 1200))
        self.rescue_focus_k = int(config.get("rescue_focus_k", 6))
        self.rescue_alpha = float(config.get("rescue_alpha", 1.0))
        self.rescue_alpha_expand = float(config.get("rescue_alpha_expand", 1.08))
        self.rescue_alpha_shrink = float(config.get("rescue_alpha_shrink", 0.72))
        self.rescue_min_alpha = float(config.get("rescue_min_alpha", 0.02))
        self.rescue_max_alpha = float(config.get("rescue_max_alpha", 6.0))

        seed = int(config.get("np_seed", self.rnd.getrandbits(63)))
        self.np_rng = np.random.default_rng(seed)

    def evolve(self, problem: SingleImageProblem):
        if self.constraint != "l2":
            raise ValueError("StructuredActiveLabelAttack only supports l2 constraint.")

        image_chw, channels, height, width = self._get_image_chw(problem)
        dim = int(problem.get_dimension())
        if image_chw is None:
            return np.zeros(dim, dtype=np.float32)

        epsilon = float(problem.epsilon)
        zero_delta = np.zeros(dim, dtype=np.float32)
        center_fitness, center_fit = self._evaluate_single(problem, zero_delta)
        if center_fitness is None:
            return zero_delta
        if center_fitness <= 0.0:
            return zero_delta

        center_delta = zero_delta.copy()
        best_delta = center_delta.copy()
        best_fitness = float(center_fitness)
        best_fit = center_fit.copy()
        label_memory = {}

        self._log(
            1,
            f"[Init] eval={int(problem.evaluations)} fitness={best_fitness:.6f} "
            f"pos={self._pos_count(best_fit)} max_pos={self._max_pos(best_fit):.6f}",
        )

        center_delta, center_fitness, center_fit = self._bootstrap(
            problem=problem,
            base_delta=center_delta,
            base_fitness=center_fitness,
            base_fit=center_fit,
            epsilon=epsilon,
            image_chw=image_chw,
            channels=channels,
            height=height,
            width=width,
        )
        if self._better(center_fit, center_fitness, best_fit, best_fitness):
            best_delta = center_delta.copy()
            best_fitness = float(center_fitness)
            best_fit = center_fit.copy()
        if best_fitness <= 0.0:
            return best_delta

        sigma = float(self.sigma_init)
        step = float(self.step_init)
        stagnation = 0
        restart_count = 0
        iter_id = 0
        rescue_runs = 0
        last_rescue_iter = -10**9
        best_pos_count = self._pos_count(best_fit)

        active = self._active_labels(center_fit, self._focus_budget(center_fit))
        self._update_label_memory(label_memory, center_fit, center_delta, active)

        while self._remaining(problem) > 0 and self._pos_count(best_fit) > 0:
            iter_id += 1
            active = self._active_labels(center_fit, self._focus_budget(center_fit))
            if len(active) == 0:
                break

            pair_dirs = self._sample_basis_pack(
                center_fit=center_fit,
                active_labels=active,
                channels=channels,
                height=height,
                width=width,
            )
            if len(pair_dirs) == 0:
                break

            pair_candidates = []
            for direction in pair_dirs:
                step_size = float(sigma) * float(epsilon)
                pair_candidates.append(self._project_delta(center_delta + step_size * direction, epsilon, image_chw))
                pair_candidates.append(self._project_delta(center_delta - step_size * direction, epsilon, image_chw))

            pair_fitness, pair_fit = self._evaluate_batch(problem, pair_candidates)
            if pair_fitness is None or pair_fit is None:
                break

            pair_fitness = np.asarray(pair_fitness, dtype=np.float32).reshape(-1)
            pair_fit = np.asarray(pair_fit, dtype=np.float32)
            pair_count = min(len(pair_dirs), len(pair_fitness) // 2, len(pair_fit) // 2)
            if pair_count <= 0:
                break

            agg_direction = np.zeros(dim, dtype=np.float32)
            local_delta = None
            local_fitness = None
            local_fit = None
            improved_global = False

            for i in range(pair_count):
                pos_idx = 2 * i
                neg_idx = pos_idx + 1
                pos_delta = pair_candidates[pos_idx]
                neg_delta = pair_candidates[neg_idx]
                pos_fit = pair_fit[pos_idx].copy()
                neg_fit = pair_fit[neg_idx].copy()
                pos_fitness = float(pair_fitness[pos_idx])
                neg_fitness = float(pair_fitness[neg_idx])

                if pos_fitness <= 0.0:
                    self._log(
                        1,
                        f"[Done] eval={int(problem.evaluations)} fitness={pos_fitness:.6f} "
                        f"pos={self._pos_count(pos_fit)} max_pos={self._max_pos(pos_fit):.6f}",
                    )
                    return pos_delta.copy()
                if neg_fitness <= 0.0:
                    self._log(
                        1,
                        f"[Done] eval={int(problem.evaluations)} fitness={neg_fitness:.6f} "
                        f"pos={self._pos_count(neg_fit)} max_pos={self._max_pos(neg_fit):.6f}",
                    )
                    return neg_delta.copy()

                self._update_label_memory(label_memory, pos_fit, pos_delta, active)
                self._update_label_memory(label_memory, neg_fit, neg_delta, active)

                if self._better(pos_fit, pos_fitness, best_fit, best_fitness):
                    best_delta = pos_delta.copy()
                    best_fitness = float(pos_fitness)
                    best_fit = pos_fit.copy()
                    improved_global = True
                if self._better(neg_fit, neg_fitness, best_fit, best_fitness):
                    best_delta = neg_delta.copy()
                    best_fitness = float(neg_fitness)
                    best_fit = neg_fit.copy()
                    improved_global = True

                if local_fit is None or self._better(pos_fit, pos_fitness, local_fit, local_fitness):
                    local_delta = pos_delta.copy()
                    local_fitness = float(pos_fitness)
                    local_fit = pos_fit.copy()
                if local_fit is None or self._better(neg_fit, neg_fitness, local_fit, local_fitness):
                    local_delta = neg_delta.copy()
                    local_fitness = float(neg_fitness)
                    local_fit = neg_fit.copy()

                pos_score = self._direction_score(center_fit, pos_fit, active)
                neg_score = self._direction_score(center_fit, neg_fit, active)
                agg_direction += float(pos_score - neg_score) * pair_dirs[i]

            extra_candidates = []
            if np.linalg.norm(agg_direction) > 1e-12:
                agg_direction = self._normalize_vec(agg_direction)
                for scale in self.line_search:
                    alpha = float(step) * float(scale) * float(epsilon)
                    extra_candidates.append(self._project_delta(center_delta + alpha * agg_direction, epsilon, image_chw))

            if iter_id % max(1, self.memory_mix_every) == 0:
                extra_candidates.extend(
                    self._memory_candidates(
                        center_delta=center_delta,
                        center_fit=center_fit,
                        active_labels=active,
                        label_memory=label_memory,
                        epsilon=epsilon,
                        image_chw=image_chw,
                    )
                )

            if len(extra_candidates) > 0 and self._remaining(problem) > 0:
                extra_fitness, extra_fit = self._evaluate_batch(problem, extra_candidates)
                if extra_fitness is not None and extra_fit is not None:
                    extra_fitness = np.asarray(extra_fitness, dtype=np.float32).reshape(-1)
                    extra_fit = np.asarray(extra_fit, dtype=np.float32)
                    extra_count = min(len(extra_candidates), len(extra_fitness), len(extra_fit))
                    for i in range(extra_count):
                        cand_delta = extra_candidates[i]
                        cand_fit = extra_fit[i].copy()
                        cand_fitness = float(extra_fitness[i])

                        if cand_fitness <= 0.0:
                            self._log(
                                1,
                                f"[Done] eval={int(problem.evaluations)} fitness={cand_fitness:.6f} "
                                f"pos={self._pos_count(cand_fit)} max_pos={self._max_pos(cand_fit):.6f}",
                            )
                            return cand_delta.copy()

                        self._update_label_memory(label_memory, cand_fit, cand_delta, active)

                        if self._better(cand_fit, cand_fitness, best_fit, best_fitness):
                            best_delta = cand_delta.copy()
                            best_fitness = float(cand_fitness)
                            best_fit = cand_fit.copy()
                            improved_global = True

                        if local_fit is None or self._better(cand_fit, cand_fitness, local_fit, local_fitness):
                            local_delta = cand_delta.copy()
                            local_fitness = float(cand_fitness)
                            local_fit = cand_fit.copy()

            improved_center = local_fit is not None and self._better(local_fit, local_fitness, center_fit, center_fitness)
            if improved_center:
                center_delta = local_delta.copy()
                center_fitness = float(local_fitness)
                center_fit = local_fit.copy()
                sigma = min(self.sigma_max, sigma * 1.05)
                step = min(self.step_max, step * 1.08)
                stagnation = 0
            else:
                sigma = max(self.sigma_min, sigma * 0.93)
                step = max(self.step_min, step * 0.92)
                stagnation += 1

            if iter_id == 1 or improved_center or improved_global or iter_id % max(1, self.log_every) == 0:
                self._log(
                    1,
                    f"[Iter] id={iter_id} eval={int(problem.evaluations)} best_fit={best_fitness:.6f} "
                    f"center_fit={center_fitness:.6f} pos={self._pos_count(best_fit)} "
                    f"max_pos={self._max_pos(best_fit):.6f} sigma={sigma:.4f} step={step:.4f} "
                    f"focus={self._format_focus(center_fit, active)}",
                )

            current_best_pos = self._pos_count(best_fit)
            should_rescue = (
                self.rescue_enable
                and rescue_runs < self.max_rescue_runs
                and current_best_pos > 0
                and current_best_pos <= max(1, self.rescue_pos_threshold)
                and self._remaining(problem) > 16
                and (
                    current_best_pos < best_pos_count
                    or stagnation >= self.rescue_trigger_patience
                    or (current_best_pos <= 1 and iter_id - last_rescue_iter >= self.rescue_cooldown)
                )
            )
            if should_rescue:
                rescue_delta, rescue_fitness, rescue_fit = self._simba_rescue(
                    problem=problem,
                    start_delta=best_delta,
                    start_fitness=best_fitness,
                    start_fit=best_fit,
                    epsilon=epsilon,
                    image_chw=image_chw,
                    channels=channels,
                    height=height,
                    width=width,
                )
                rescue_runs += 1
                last_rescue_iter = iter_id
                if rescue_fitness is not None:
                    if rescue_fitness <= 0.0:
                        self._log(
                            1,
                            f"[Done] eval={int(problem.evaluations)} fitness={float(rescue_fitness):.6f} "
                            f"pos={self._pos_count(rescue_fit)} max_pos={self._max_pos(rescue_fit):.6f}",
                        )
                        return rescue_delta.copy()
                    if self._better(rescue_fit, rescue_fitness, best_fit, best_fitness):
                        best_delta = rescue_delta.copy()
                        best_fitness = float(rescue_fitness)
                        best_fit = rescue_fit.copy()
                        if self._better(best_fit, best_fitness, center_fit, center_fitness):
                            center_delta = best_delta.copy()
                            center_fitness = float(best_fitness)
                            center_fit = best_fit.copy()
                            stagnation = 0
                        self._log(
                            1,
                            f"[Rescue] eval={int(problem.evaluations)} best_fit={best_fitness:.6f} "
                            f"pos={self._pos_count(best_fit)} max_pos={self._max_pos(best_fit):.6f}",
                        )
                active = self._active_labels(center_fit, self._focus_budget(center_fit))

            best_pos_count = min(best_pos_count, self._pos_count(best_fit))

            if stagnation >= self.restart_patience and restart_count < self.max_restarts and self._remaining(problem) > 0:
                restart_count += 1
                center_delta, center_fitness, center_fit = self._restart(
                    problem=problem,
                    best_delta=best_delta,
                    best_fitness=best_fitness,
                    best_fit=best_fit,
                    center_delta=center_delta,
                    center_fit=center_fit,
                    active_labels=active,
                    label_memory=label_memory,
                    epsilon=epsilon,
                    image_chw=image_chw,
                    channels=channels,
                    height=height,
                    width=width,
                )
                if center_fitness is None:
                    break
                sigma = float(self.sigma_init)
                step = float(self.step_init)
                stagnation = 0
                self._log(
                    1,
                    f"[Restart] id={restart_count} eval={int(problem.evaluations)} center_fit={center_fitness:.6f} "
                    f"best_fit={best_fitness:.6f} pos={self._pos_count(center_fit)} "
                    f"max_pos={self._max_pos(center_fit):.6f}",
                )

        self._log(
            1,
            f"[Done] eval={int(problem.evaluations)} fitness={best_fitness:.6f} "
            f"pos={self._pos_count(best_fit)} max_pos={self._max_pos(best_fit):.6f}",
        )
        return best_delta

    def _bootstrap(self, problem, base_delta, base_fitness, base_fit, epsilon, image_chw, channels, height, width):
        remain = self._remaining(problem)
        if remain <= 0:
            return base_delta, float(base_fitness), base_fit.copy()

        candidates = []
        n_base = min(max(1, int(self.init_candidates)), remain)
        for _ in range(n_base):
            direction = self._structured_direction(
                channels=channels,
                height=height,
                width=width,
                fine=False,
            )
            for ratio in self.init_radii:
                candidates.append(self._project_delta(float(ratio) * float(epsilon) * direction, epsilon, image_chw))

        if len(candidates) == 0:
            return base_delta, float(base_fitness), base_fit.copy()

        candidates = candidates[:remain]
        fitness_arr, fit_arr = self._evaluate_batch(problem, candidates)
        if fitness_arr is None or fit_arr is None:
            return base_delta, float(base_fitness), base_fit.copy()

        best_delta = base_delta.copy()
        best_fitness = float(base_fitness)
        best_fit = base_fit.copy()
        fitness_arr = np.asarray(fitness_arr, dtype=np.float32).reshape(-1)
        fit_arr = np.asarray(fit_arr, dtype=np.float32)
        count = min(len(candidates), len(fitness_arr), len(fit_arr))
        for i in range(count):
            cand_delta = candidates[i]
            cand_fit = fit_arr[i].copy()
            cand_fitness = float(fitness_arr[i])
            if cand_fitness <= 0.0:
                return cand_delta.copy(), float(cand_fitness), cand_fit.copy()
            if self._better(cand_fit, cand_fitness, best_fit, best_fitness):
                best_delta = cand_delta.copy()
                best_fitness = float(cand_fitness)
                best_fit = cand_fit.copy()

        self._log(
            1,
            f"[Boot] eval={int(problem.evaluations)} fitness={best_fitness:.6f} "
            f"pos={self._pos_count(best_fit)} max_pos={self._max_pos(best_fit):.6f}",
        )
        return best_delta, best_fitness, best_fit

    def _simba_rescue(self, problem, start_delta, start_fitness, start_fit, epsilon, image_chw, channels, height, width):
        remain = self._remaining(problem)
        if remain <= 1:
            return start_delta.copy(), float(start_fitness), start_fit.copy()

        local_budget = min(int(self.rescue_query_cap), remain)
        if local_budget < 2:
            return start_delta.copy(), float(start_fitness), start_fit.copy()

        basis = int(min(max(1, self.rescue_basis_size), height, width))
        if basis <= 0:
            return start_delta.copy(), float(start_fitness), start_fit.copy()

        coords = np.asarray([(c, i, j) for c in range(channels) for i in range(basis) for j in range(basis)], dtype=np.int32)
        if coords.size == 0:
            return start_delta.copy(), float(start_fitness), start_fit.copy()

        best_delta = np.asarray(start_delta, dtype=np.float32).reshape(-1).copy()
        best_fitness = float(start_fitness)
        best_fit = np.asarray(start_fit, dtype=np.float32).reshape(-1).copy()
        best_focus = self._focus_indices(best_fit, self.rescue_focus_k)
        best_focus_obj = self._focus_objective(best_fit, best_focus)

        alpha = float(np.clip(self.rescue_alpha, self.rescue_min_alpha, self.rescue_max_alpha))
        coord_batch = max(1, int(self.rescue_coord_batch))
        perm = self.np_rng.permutation(len(coords))
        perm_pos = 0
        rescue_start_eval = int(problem.evaluations)
        rounds = 0

        while int(problem.evaluations) - rescue_start_eval < local_budget and self._remaining(problem) > 1:
            remain_queries = min(local_budget - (int(problem.evaluations) - rescue_start_eval), self._remaining(problem))
            if remain_queries < 2:
                break
            if perm_pos >= len(perm):
                perm = self.np_rng.permutation(len(coords))
                perm_pos = 0

            batch_count = min(coord_batch, len(perm) - perm_pos, remain_queries // 2)
            if batch_count <= 0:
                break

            chosen = coords[perm[perm_pos: perm_pos + batch_count]]
            perm_pos += batch_count
            rounds += 1

            low = np.zeros((batch_count, channels, basis, basis), dtype=np.float32)
            for k in range(batch_count):
                c, i, j = int(chosen[k, 0]), int(chosen[k, 1]), int(chosen[k, 2])
                low[k, c, i, j] = 1.0

            up = F.interpolate(
                torch.from_numpy(low),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ).cpu().numpy()
            dirs = up.reshape(batch_count, -1)
            dirs = dirs / np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12)

            candidates = []
            for k in range(batch_count):
                direction = dirs[k].astype(np.float32, copy=False)
                candidates.append(self._project_delta(best_delta + alpha * direction, epsilon, image_chw))
                candidates.append(self._project_delta(best_delta - alpha * direction, epsilon, image_chw))

            fitness_arr, fit_arr = self._evaluate_batch(problem, candidates)
            if fitness_arr is None or fit_arr is None:
                break

            fitness_arr = np.asarray(fitness_arr, dtype=np.float32).reshape(-1)
            fit_arr = np.asarray(fit_arr, dtype=np.float32)
            success_idx = np.where(fitness_arr <= 0.0)[0]
            if success_idx.size > 0:
                success_delta = candidates[int(success_idx[0])]
                refined_delta = self._binary_refine(problem, success_delta, epsilon, image_chw)
                refined_fitness, refined_fit = self._evaluate_single(problem, refined_delta)
                if refined_fitness is None:
                    return success_delta.copy(), float(fitness_arr[int(success_idx[0])]), fit_arr[int(success_idx[0])].copy()
                return refined_delta, refined_fitness, refined_fit

            best_idx = None
            best_obj = None
            count = min(len(candidates), len(fitness_arr), len(fit_arr))
            for i in range(count):
                cand_fit = fit_arr[i].copy()
                cand_fitness = float(fitness_arr[i])
                focus_idx = self._focus_indices(cand_fit, self.rescue_focus_k)
                obj = self._focus_objective(cand_fit, focus_idx)
                if best_idx is None or obj < best_obj or (obj == best_obj and cand_fitness + 1e-9 < float(fitness_arr[best_idx])):
                    best_idx = i
                    best_obj = obj

            if best_idx is None:
                break

            cand_delta = candidates[best_idx]
            cand_fitness = float(fitness_arr[best_idx])
            cand_fit = fit_arr[best_idx].copy()
            cand_focus = self._focus_indices(cand_fit, self.rescue_focus_k)
            cand_focus_obj = self._focus_objective(cand_fit, cand_focus)

            if cand_focus_obj < best_focus_obj or self._better(cand_fit, cand_fitness, best_fit, best_fitness):
                best_delta = np.asarray(cand_delta, dtype=np.float32).reshape(-1).copy()
                best_fitness = float(cand_fitness)
                best_fit = cand_fit.copy()
                best_focus = cand_focus
                best_focus_obj = cand_focus_obj
                alpha = float(np.clip(alpha * self.rescue_alpha_expand, self.rescue_min_alpha, self.rescue_max_alpha))
            else:
                alpha = float(np.clip(alpha * self.rescue_alpha_shrink, self.rescue_min_alpha, self.rescue_max_alpha))

            if rounds % max(1, self.rescue_shuffle_every) == 0:
                perm = self.np_rng.permutation(len(coords))
                perm_pos = 0

        return best_delta, best_fitness, best_fit

    def _restart(
        self,
        problem,
        best_delta,
        best_fitness,
        best_fit,
        center_delta,
        center_fit,
        active_labels,
        label_memory,
        epsilon,
        image_chw,
        channels,
        height,
        width,
    ):
        remain = self._remaining(problem)
        if remain <= 0:
            return center_delta, float(best_fitness), best_fit.copy()

        seeds = []
        seeds.append(np.asarray(best_delta, dtype=np.float32).copy())
        seeds.append(np.asarray(center_delta, dtype=np.float32).copy())
        for label in active_labels:
            entry = label_memory.get(int(label))
            if entry is not None:
                seeds.append(np.asarray(entry["delta"], dtype=np.float32).copy())

        candidates = []
        ratios = [0.55, 0.75, 0.92]
        for seed in seeds[: max(2, 2 + len(active_labels))]:
            base = self._normalize_vec(seed)
            if np.linalg.norm(base) <= 1e-12:
                continue
            for ratio in ratios:
                candidates.append(self._project_delta(float(ratio) * float(epsilon) * base, epsilon, image_chw))
                jitter = self._structured_direction(
                    channels=channels,
                    height=height,
                    width=width,
                    fine=self._pos_count(center_fit) <= 1,
                )
                mixed = self._normalize_vec(0.75 * base + 0.25 * jitter)
                candidates.append(self._project_delta(float(ratio) * float(epsilon) * mixed, epsilon, image_chw))

        if len(candidates) == 0:
            for ratio in ratios:
                rand_dir = self._structured_direction(channels=channels, height=height, width=width, fine=False)
                candidates.append(self._project_delta(float(ratio) * float(epsilon) * rand_dir, epsilon, image_chw))

        candidates = candidates[:remain]
        fitness_arr, fit_arr = self._evaluate_batch(problem, candidates)
        if fitness_arr is None or fit_arr is None:
            return None, None, None

        fitness_arr = np.asarray(fitness_arr, dtype=np.float32).reshape(-1)
        fit_arr = np.asarray(fit_arr, dtype=np.float32)
        best_local_delta = None
        best_local_fitness = None
        best_local_fit = None
        count = min(len(candidates), len(fitness_arr), len(fit_arr))
        for i in range(count):
            cand_delta = candidates[i]
            cand_fit = fit_arr[i].copy()
            cand_fitness = float(fitness_arr[i])
            if best_local_fit is None or self._better(cand_fit, cand_fitness, best_local_fit, best_local_fitness):
                best_local_delta = cand_delta.copy()
                best_local_fitness = float(cand_fitness)
                best_local_fit = cand_fit.copy()

        return best_local_delta, best_local_fitness, best_local_fit

    def _sample_basis_pack(self, center_fit, active_labels, channels, height, width):
        pos_count = self._pos_count(center_fit)
        fine = pos_count <= 1
        n_dirs = self.n_directions + (4 if fine else 0)
        return [
            self._structured_direction(channels=channels, height=height, width=width, fine=fine)
            for _ in range(max(1, int(n_dirs)))
        ]

    def _structured_direction(self, channels, height, width, fine):
        parts = self.fine_parts if fine else self.coarse_parts
        freq_max = self.dct_freq_fine if fine else self.dct_freq_coarse
        mode = self.np_rng.choice(
            np.array(["block", "hstripe", "vstripe", "dct", "checker", "channel"], dtype=object),
            p=np.array([0.34, 0.16, 0.16, 0.18, 0.08, 0.08], dtype=np.float64),
        )
        arr = np.zeros((channels, height, width), dtype=np.float32)
        channel_mode = int(self.np_rng.integers(-1, channels))
        channel_ids = range(channels) if channel_mode < 0 else [channel_mode]

        if mode == "block":
            gh = int(self.np_rng.choice(np.asarray(parts, dtype=np.int32)))
            gw = int(self.np_rng.choice(np.asarray(parts, dtype=np.int32)))
            i = int(self.np_rng.integers(0, gh))
            j = int(self.np_rng.integers(0, gw))
            h0 = int(round(i * height / gh))
            h1 = int(round((i + 1) * height / gh))
            w0 = int(round(j * width / gw))
            w1 = int(round((j + 1) * width / gw))
            patch = self.np_rng.normal(size=(h1 - h0, w1 - w0)).astype(np.float32)
            for c in channel_ids:
                arr[int(c), h0:h1, w0:w1] = patch
        elif mode == "hstripe":
            gh = int(self.np_rng.choice(np.asarray(parts, dtype=np.int32)))
            i = int(self.np_rng.integers(0, gh))
            h0 = int(round(i * height / gh))
            h1 = int(round((i + 1) * height / gh))
            patch = self.np_rng.normal(size=(h1 - h0, width)).astype(np.float32)
            for c in channel_ids:
                arr[int(c), h0:h1, :] = patch
        elif mode == "vstripe":
            gw = int(self.np_rng.choice(np.asarray(parts, dtype=np.int32)))
            j = int(self.np_rng.integers(0, gw))
            w0 = int(round(j * width / gw))
            w1 = int(round((j + 1) * width / gw))
            patch = self.np_rng.normal(size=(height, w1 - w0)).astype(np.float32)
            for c in channel_ids:
                arr[int(c), :, w0:w1] = patch
        elif mode == "checker":
            gh = int(self.np_rng.choice(np.asarray(parts, dtype=np.int32)))
            gw = int(self.np_rng.choice(np.asarray(parts, dtype=np.int32)))
            yy = (np.arange(height)[:, None] * gh) // max(1, height)
            xx = (np.arange(width)[None, :] * gw) // max(1, width)
            base = ((yy + xx) % 2).astype(np.float32) * 2.0 - 1.0
            for c in channel_ids:
                arr[int(c)] = base
        elif mode == "channel":
            value = self.np_rng.normal(size=(height, width)).astype(np.float32)
            target_c = int(self.np_rng.integers(0, channels))
            arr[target_c] = value
        else:
            fy = int(self.np_rng.integers(0, max(1, freq_max)))
            fx = int(self.np_rng.integers(0, max(1, freq_max)))
            yy = ((np.arange(height, dtype=np.float32) + 0.5) / float(height))[:, None]
            xx = ((np.arange(width, dtype=np.float32) + 0.5) / float(width))[None, :]
            basis = (np.cos(np.pi * fy * yy) if fy > 0 else np.ones((height, 1), dtype=np.float32)) * (
                np.cos(np.pi * fx * xx) if fx > 0 else np.ones((1, width), dtype=np.float32)
            )
            for c in channel_ids:
                arr[int(c)] = basis

        return self._normalize_vec(arr.reshape(-1))

    def _memory_candidates(self, center_delta, center_fit, active_labels, label_memory, epsilon, image_chw):
        available = [int(label) for label in active_labels if int(label) in label_memory]
        if len(available) == 0:
            return []

        residual = np.maximum(np.asarray(center_fit, dtype=np.float32).reshape(-1), 0.0)
        weights = np.asarray([max(1e-6, float(residual[int(label)])) for label in available], dtype=np.float32)
        weights = weights / max(float(np.sum(weights)), 1e-12)

        merged = np.zeros_like(np.asarray(center_delta, dtype=np.float32).reshape(-1), dtype=np.float32)
        for w, label in zip(weights, available):
            merged += float(w) * np.asarray(label_memory[int(label)]["delta"], dtype=np.float32).reshape(-1)

        candidates = []
        merged = self._normalize_vec(merged)
        if np.linalg.norm(merged) > 1e-12:
            merged = self._project_delta(float(epsilon) * merged, epsilon, image_chw)
            candidates.append(self._project_delta(0.65 * np.asarray(center_delta, dtype=np.float32) + 0.35 * merged, epsilon, image_chw))
            candidates.append(self._project_delta(0.40 * np.asarray(center_delta, dtype=np.float32) + 0.60 * merged, epsilon, image_chw))

        if len(available) >= 2:
            first = np.asarray(label_memory[available[0]]["delta"], dtype=np.float32).reshape(-1)
            second = np.asarray(label_memory[available[1]]["delta"], dtype=np.float32).reshape(-1)
            pair_mix = self._project_delta(0.5 * first + 0.5 * second, epsilon, image_chw)
            candidates.append(pair_mix)

        return candidates

    def _update_label_memory(self, label_memory, fit_row, delta, labels):
        row = np.asarray(fit_row, dtype=np.float32).reshape(-1)
        pos = np.maximum(row, 0.0)
        pos_sum = float(np.sum(pos))
        for label in labels:
            idx = int(label)
            value = float(pos[idx])
            prev = label_memory.get(idx)
            score = (value, pos_sum)
            if prev is None or score < prev["score"]:
                label_memory[idx] = {
                    "score": score,
                    "delta": np.asarray(delta, dtype=np.float32).copy(),
                }

    def _direction_score(self, base_fit, cand_fit, active_labels):
        base_pos = np.maximum(np.asarray(base_fit, dtype=np.float32).reshape(-1), 0.0)
        cand_pos = np.maximum(np.asarray(cand_fit, dtype=np.float32).reshape(-1), 0.0)
        idx = np.asarray(active_labels, dtype=np.int64)
        if idx.size == 0:
            active_gain = float(np.sum(base_pos) - np.sum(cand_pos))
        else:
            weights = base_pos[idx] + 1e-6
            weights = weights / max(float(np.sum(weights)), 1e-12)
            active_gain = float(np.sum(weights * (base_pos[idx] - cand_pos[idx])))
        count_gain = float(np.sum(base_pos > 0) - np.sum(cand_pos > 0))
        global_gain = float(np.sum(base_pos) - np.sum(cand_pos))
        max_gain = float(np.max(base_pos) - np.max(cand_pos)) if np.any(base_pos > 0) else 0.0
        return active_gain + 0.30 * global_gain + 0.45 * count_gain + 0.20 * max_gain

    def _focus_indices(self, fit_row, k):
        row = np.maximum(np.asarray(fit_row, dtype=np.float32).reshape(-1), 0.0)
        idx = np.where(row > 0)[0]
        if idx.size == 0:
            return None
        k = max(1, int(k))
        if idx.size <= k:
            return idx
        vals = row[idx]
        order = np.argsort(-vals)
        return idx[order[:k]]

    def _focus_objective(self, fit_row, focus_idx):
        pos = np.maximum(np.asarray(fit_row, dtype=np.float32).reshape(-1), 0.0)
        active = pos[pos > 0]
        if active.size == 0:
            return (0, 0.0, 0.0, 0.0)
        if focus_idx is None:
            focus = active
        else:
            idx = np.asarray(focus_idx, dtype=np.int64).reshape(-1)
            idx = idx[(idx >= 0) & (idx < pos.size)]
            focus = pos[idx] if idx.size > 0 else active
        focus_pos = focus[focus > 0]
        if focus_pos.size == 0:
            focus_pos = active
        return (
            int(active.size),
            float(np.sum(focus_pos)),
            float(np.max(focus_pos)),
            float(np.sum(active)),
        )

    def _focus_budget(self, fit_row):
        pos_count = self._pos_count(fit_row)
        if pos_count <= 1:
            return 1
        if pos_count <= 3:
            return pos_count
        return min(pos_count, max(2, int(self.active_topk)))

    def _better(self, fit_a, fitness_a, fit_b, fitness_b):
        if fit_b is None:
            return True
        obj_a = self._objective(fit_a)
        obj_b = self._objective(fit_b)
        return obj_a < obj_b or (obj_a == obj_b and float(fitness_a) + 1e-9 < float(fitness_b))

    def _objective(self, fit_row):
        pos = np.maximum(np.asarray(fit_row, dtype=np.float32).reshape(-1), 0.0)
        active = pos[pos > 0]
        if active.size == 0:
            return (0, 0.0, 0.0)
        return (int(active.size), float(np.sum(active)), float(np.max(active)))

    def _active_labels(self, fit_row, topk):
        row = np.maximum(np.asarray(fit_row, dtype=np.float32).reshape(-1), 0.0)
        idx = np.where(row > 0)[0]
        if idx.size == 0:
            return []
        vals = row[idx]
        order = np.argsort(-vals)
        idx = idx[order]
        return [int(v) for v in idx[: max(1, int(topk))]]

    def _pos_count(self, fit_row):
        row = np.asarray(fit_row, dtype=np.float32).reshape(-1)
        return int(np.sum(row > 0))

    def _max_pos(self, fit_row):
        row = np.asarray(fit_row, dtype=np.float32).reshape(-1)
        pos = row[row > 0]
        return float(np.max(pos)) if pos.size > 0 else 0.0

    def _format_focus(self, fit_row, labels):
        row = np.asarray(fit_row, dtype=np.float32).reshape(-1)
        if labels is None or len(labels) == 0:
            return "[]"
        return "[" + ",".join(f"{int(label)}:{float(row[int(label)]):.4f}" for label in labels) + "]"

    def _normalize_vec(self, vec):
        vec = np.asarray(vec, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-12:
            return vec
        return (vec / norm).astype(np.float32, copy=False)

    def _delta_norm(self, delta):
        return float(np.linalg.norm(np.asarray(delta, dtype=np.float32).reshape(-1)))

    def _get_image_chw(self, problem):
        image = getattr(problem, "_SingleImageProblem__image", None)
        if image is not None:
            image_chw = np.asarray(image, dtype=np.float32)
            if image_chw.ndim == 3:
                channels, height, width = image_chw.shape
                return image_chw, int(channels), int(height), int(width)

        dim = int(problem.get_dimension())
        if dim % 3 != 0:
            return None, None, None, None
        side = int(round(np.sqrt(dim // 3)))
        if side * side * 3 != dim:
            return None, None, None, None
        image_chw = np.zeros((3, side, side), dtype=np.float32)
        return image_chw, 3, side, side

    def _project_delta(self, delta, epsilon, image_chw):
        delta = np.asarray(delta, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(delta))
        if norm > float(epsilon) and norm > 0.0:
            delta = delta * (float(epsilon) / norm)

        if not self.project_with_clip or image_chw is None:
            return delta.astype(np.float32, copy=False)

        delta_chw = delta.reshape(image_chw.shape)
        adv = np.clip(image_chw + delta_chw, self.clip_min, self.clip_max)
        projected = (adv - image_chw).reshape(-1)
        norm = float(np.linalg.norm(projected))
        if norm > float(epsilon) and norm > 0.0:
            projected = projected * (float(epsilon) / norm)
            adv = np.clip(image_chw + projected.reshape(image_chw.shape), self.clip_min, self.clip_max)
            projected = (adv - image_chw).reshape(-1)
        return projected.astype(np.float32, copy=False)

    def _remaining(self, problem):
        return max(0, int(problem.max_evaluation) - int(problem.evaluations))

    def _evaluate_single(self, problem, delta):
        fitness_arr, fit_arr = self._evaluate_batch(problem, [delta])
        if fitness_arr is None or fit_arr is None or len(fitness_arr) == 0 or len(fit_arr) == 0:
            return None, None
        fitness = float(np.asarray(fitness_arr, dtype=np.float32).reshape(-1)[0])
        fit = np.asarray(fit_arr[0], dtype=np.float32).reshape(-1)
        return fitness, fit

    def _binary_refine(self, problem, success_delta, epsilon, image_chw):
        best = self._project_delta(success_delta, epsilon, image_chw)
        low = 0.0
        high = 1.0
        for _ in range(15):
            if self._remaining(problem) <= 0:
                break
            mid = 0.5 * (low + high)
            cand = self._project_delta(success_delta * mid, epsilon, image_chw)
            fitness, _ = self._evaluate_single(problem, cand)
            if fitness is None:
                break
            if fitness <= 0.0:
                best = cand
                high = mid
            else:
                low = mid
        return best

    def _evaluate_batch(self, problem, deltas):
        remain = self._remaining(problem)
        if remain <= 0:
            return None, None
        deltas = deltas[:remain]
        if len(deltas) == 0:
            return None, None

        fitness_chunks = []
        fit_chunks = []
        idx = 0
        base_chunk = max(1, int(self.eval_batch_size))
        while idx < len(deltas):
            remain = self._remaining(problem)
            if remain <= 0:
                break
            chunk = min(base_chunk, len(deltas) - idx, remain)
            if chunk <= 0:
                break

            while True:
                batch = np.asarray(deltas[idx: idx + chunk], dtype=np.float32)
                try:
                    fitness, fit = problem.evaluate(batch, effective=True)
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower() and chunk > 1:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        chunk = max(1, chunk // 2)
                        continue
                    raise

                if fitness is None or fit is None:
                    return None, None
                fitness_chunks.append(np.asarray(fitness, dtype=np.float32).reshape(-1))
                fit_chunks.append(np.asarray(fit, dtype=np.float32))
                break
            idx += chunk

        if len(fitness_chunks) == 0 or len(fit_chunks) == 0:
            return None, None
        return np.concatenate(fitness_chunks, axis=0).astype(np.float32), np.concatenate(fit_chunks, axis=0).astype(np.float32)

    def _log(self, level, message):
        if int(self.verbose) >= int(level):
            print(message)
