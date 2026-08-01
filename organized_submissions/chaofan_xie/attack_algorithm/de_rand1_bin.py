import numpy as np
import random
from attack_algorithm.attack_algorithm_base import AttackAlgorithmBase
from attack_problem.one_image_problem import SingleImageProblem

class DE_RAND1(AttackAlgorithmBase):

    def __init__(self, config):
        super().__init__(config)
        # 默认参数
        self.pop_size = config.get("pop_size", 20)
        self.F = config.get("F", 0.5)
        self.CR = config.get("CR", 0.9)
        self.eps = config.get("eps", 0.01)
        self.single_label_mode = config.get("single_label_mode", True)
        self.use_low_dim = config.get("use_low_dim", True)
        self.low_dim_size = config.get("low_dim_size", 32)

    # =====================================================
    #   evolve: MLDE 的 DE 主循环
    # =====================================================
    def evolve(self, problem: SingleImageProblem):
        rnd = self.rnd
        return DE(
            self.pop_size,
            self.F,
            self.CR,
            self.eps,
            problem,
            rnd,
            self.single_label_mode,
            self.use_low_dim,
            self.low_dim_size,
        )


def mutation(pop, F, rnd):
    """经典 DE/rand/1 变异"""
    pop_size = len(pop)
    dim = pop.shape[1]
    mutant = np.zeros_like(pop)
    for i in range(pop_size):
        idxs = list(range(pop_size))
        idxs.remove(i)
        a, b, c = rnd.sample(idxs, 3)
        mutant[i] = pop[a] + F * (pop[b] - pop[c])
    return mutant


def crossover(pop, mutant, CR, rnd):
    """二进制交叉"""
    pop_size, dim = pop.shape
    trial = np.copy(pop)
    for i in range(pop_size):
        jrand = rnd.randint(0, dim - 1)
        for j in range(dim):
            if rnd.random() < CR or j == jrand:
                trial[i, j] = mutant[i, j]
    return trial


def problem_l2_norm(x):
    return [float(np.linalg.norm(np.asarray(r).reshape(-1))) for r in x]


class LowDimDecoder:
    def __init__(self, problem: SingleImageProblem, low_dim_size):
        image = np.asarray(problem.image)
        self.image_shape = image.shape
        self.channels = int(self.image_shape[0])
        self.height = int(self.image_shape[1])
        self.width = int(self.image_shape[2])
        self.low_dim_size = max(1, int(low_dim_size))
        self.latent_dim = self.channels * self.low_dim_size * self.low_dim_size
        self.full_dim = problem.get_dimension()
        self.y_idx = np.floor(
            np.arange(self.height) * self.low_dim_size / self.height
        ).astype(np.int64)
        self.x_idx = np.floor(
            np.arange(self.width) * self.low_dim_size / self.width
        ).astype(np.int64)

    def decode(self, latent):
        latent = np.asarray(latent, dtype=np.float32)
        small = latent.reshape((-1, self.channels, self.low_dim_size, self.low_dim_size))
        full = small[:, :, self.y_idx][:, :, :, self.x_idx]
        return full.reshape((len(latent), self.full_dim)).astype(np.float32)


def decode_population(pop, decoder):
    if decoder is None:
        return np.asarray(pop, dtype=np.float32)
    return decoder.decode(pop)


def single_label_scores(fitness, fit, eval_pop, target_label):
    label_score = np.asarray(fit)[:, int(target_label)].astype(np.float32)
    official = np.asarray(fitness).reshape(-1).astype(np.float32)
    radius = np.asarray(problem_l2_norm(eval_pop), dtype=np.float32)
    return label_score + 1e-4 * official + 1e-7 * radius


def select(pop, full_pop, trial, fitness, fit, scores, problem: SingleImageProblem, decoder=None, target_label=None):
    trial_full = decode_population(trial, decoder)
    trial_fitness, trial_fit = problem.evaluate(trial_full)
    if trial_fitness is None:
        return pop, full_pop, fitness, fit, scores

    if target_label is None:
        trial_scores = np.asarray(trial_fitness).reshape(-1)
    else:
        trial_scores = single_label_scores(trial_fitness, trial_fit, trial_full, target_label)

    new_pop = np.copy(pop)
    new_full_pop = np.copy(full_pop)
    new_fitness = np.copy(fitness)
    new_fit = np.copy(fit)
    new_scores = np.copy(scores)
    for i in range(len(pop)):
        if float(trial_scores[i]) < float(scores[i]):
            new_pop[i] = trial[i]
            new_full_pop[i] = trial_full[i]
            new_fitness[i] = trial_fitness[i]
            new_fit[i] = trial_fit[i]
            new_scores[i] = trial_scores[i]
    return new_pop, new_full_pop, new_fitness, new_fit, new_scores


def DE(
    pop_size,
    F,
    CR,
    eps,
    problem: SingleImageProblem,
    rnd: random,
    single_label_mode=True,
    use_low_dim=True,
    low_dim_size=32,
):
    """经典 DE 主循环"""
    generation_save = []

    full_dim = problem.get_dimension()
    decoder = None
    if use_low_dim:
        decoder = LowDimDecoder(problem, low_dim_size)
        dim = decoder.latent_dim
        print(
            "DE low-dim mode: "
            f"low_dim_size={decoder.low_dim_size}, latent_dim={dim}, full_dim={full_dim}"
        )
    else:
        dim = full_dim
    x_range = [(-1, 1)] * dim

    target_label = None
    if single_label_mode:
        zero = np.zeros((1, full_dim), dtype=np.float32)
        zero_fitness, zero_fit = problem.evaluate(zero)
        if zero_fit is not None:
            zero_fit_vec = np.asarray(zero_fit[0]).reshape(-1)
            failed = np.where(zero_fit_vec > 1e-12)[0]
            if len(failed) > 0:
                target_label = int(failed[np.argmax(zero_fit_vec[failed])])
                print(
                    "DE single-label mode: "
                    f"target_label={target_label}, "
                    f"initial_label_fit={float(zero_fit_vec[target_label]):.6f}, "
                    f"initial_total_fitness={float(zero_fitness[0, 0]):.6f}"
                )
            else:
                print("DE single-label mode: no failed label at zero perturbation, fallback to total fitness.")

    # 初始化种群
    pop = np.zeros((pop_size, dim))
    for i in range(dim):
        low, high = x_range[i]
        pop[:, i] = np.array([rnd.uniform(low, high) * eps for _ in range(pop_size)])

    full_pop = decode_population(pop, decoder)
    fitness, fit = problem.evaluate(full_pop)
    if fitness is None:
        return np.zeros((full_dim,), dtype=np.float32)

    if target_label is None:
        scores = np.asarray(fitness).reshape(-1)
    else:
        scores = single_label_scores(fitness, fit, full_pop, target_label)

    generation_save.append(np.min(fitness))

    while problem.evaluations < problem.max_evaluation:
        mutant = mutation(pop, F, rnd)
        trial = crossover(pop, mutant, CR, rnd)
        pop, full_pop, fitness, fit, scores = select(
            pop,
            full_pop,
            trial,
            fitness,
            fit,
            scores,
            problem,
            decoder,
            target_label,
        )
        best_idx = int(np.argmin(scores))
        pop_r= problem.l2_norm(full_pop) 
        r_min= np.min(pop_r)    
        if target_label is None:
            print(f"Evaluation:{problem.evaluations}, Best fitness:{float(fitness[best_idx, 0]):.6f}, Best radius:{float(r_min):.6f}")
        else:
            print(
                f"Evaluation:{problem.evaluations}, "
                f"target_label:{target_label}, "
                f"Best label fit:{float(fit[best_idx, target_label]):.6f}, "
                f"Best total fitness:{float(fitness[best_idx, 0]):.6f}, "
                f"Best radius:{float(pop_r[best_idx]):.6f}"
            )
        generation_save.append(fitness[best_idx])
        # 找到完全成功的解
        if np.any(fitness == 0):
            return full_pop[np.argwhere(fitness == 0)[0][0]]
        if target_label is not None and float(fit[best_idx, target_label]) <= 1e-12:
            print(
                "DE single-label mode reached target label success: "
                f"label={target_label}, Evaluation:{problem.evaluations}"
            )
            return full_pop[best_idx]

    return full_pop[int(np.argmin(scores))]
