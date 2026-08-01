from attack_problem.problem import EvolutionaryAttackProblem    

import random
import os

if __name__ == '__main__':
    range_start = int(os.environ.get("MLAE_START_INDEX", "0"))  # 0-based inclusive
    range_end_raw = os.environ.get("MLAE_END_INDEX", "").strip()  # 0-based exclusive
    range_end = int(range_end_raw) if range_end_raw != "" else None
    result_suffix = os.environ.get("MLAE_RESULT_SUFFIX", "").strip()
    resume_workers = 4

    proConfig = {
        "ml_model_name": "mlgcn",   # Multi-label classification model: mlgcn or mlliw
        "dataset_name": "nuswide",     # Dataset to attack: coco / voc2007 / voc2012 / nuswide
        "target_type": "random",    # Flip exactly one positive and one negative label
        "epsilon": 77.596,          # Perturbation limit (L2 norm bound)
        "max_eval": 10000,          # Maximum number of fitness evaluations per image 
        "batch_size": 1,
        "workers": resume_workers,
        "data_dir": os.environ.get("MLAE_CEC_DATA_DIR", "MLAE_cec_data")
    }
    try:
        import subprocess

        def _pick_gpu_with_most_free_mem():
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            )
            free = []
            for line in out.strip().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    free.append(int(line))
                except ValueError:
                    free.append(-1)
            if len(free) == 0:
                return 0, None
            idx = int(max(range(len(free)), key=lambda i: free[i]))
            return idx, free

        forced = os.environ.get("MLAE_CUDA_DEVICE", "").strip()
        if forced != "":
            gpu_id = int(forced)
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            print(f"[GPU] Using cuda:{gpu_id} (MLAE_CUDA_DEVICE)")
        else:
            gpu_id, free = _pick_gpu_with_most_free_mem()
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            if free is None:
                print(f"[GPU] Using cuda:{gpu_id}")
            else:
                print(f"[GPU] Using cuda:{gpu_id}, free_mem={free} MiB")
    except Exception as exc:
        print(f"[GPU] Auto-select skipped: {exc}")
    import sys

    class _Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for stream in self.streams:
                stream.write(data)
            return len(data)

        def flush(self):
            for stream in self.streams:
                stream.flush()

    if result_suffix != "":
        result_name = f"result_{result_suffix}.txt"
    elif range_end is None:
        result_name = f"result_{range_start}_end.txt"
    else:
        result_name = f"result_{range_start}_{range_end}.txt"
    result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), result_name)
    result_file = open(result_path, "w", encoding="utf-8", buffering=1)
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = _Tee(sys.stdout, result_file)
    sys.stderr = _Tee(sys.stderr, result_file)
    try:
        print(f"[Log] Writing output to {result_path}")
        problem = EvolutionaryAttackProblem(proConfig)
        if range_start > 0 or range_end is not None:
            import torch

            total_images = len(problem.dataset)
            if range_start < 0:
                raise ValueError(f"range_start={range_start} must be >= 0")
            if range_start >= total_images:
                raise ValueError(
                    f"range_start={range_start} is out of range for dataset size {total_images}"
                )
            actual_end = total_images if range_end is None else min(int(range_end), total_images)
            if actual_end <= range_start:
                raise ValueError(
                    f"invalid range: start={range_start}, end={actual_end}, dataset size={total_images}"
                )

            indices = list(range(range_start, actual_end))
            problem.dataset = torch.utils.data.Subset(problem.dataset, indices)
            problem.y_target = problem.y_target[range_start:actual_end]
            problem.y = problem.y[range_start:actual_end]
            problem.loader = torch.utils.data.DataLoader(
                problem.dataset,
                batch_size=1,
                shuffle=False,
                num_workers=resume_workers,
            )
            print(
                f"[Range] start={range_start} end={actual_end} "
                f"(1-based #{range_start + 1} to #{actual_end}), count={len(indices)}"
            )

        from attack_algorithm.structured_active_label_attack import StructuredActiveLabelAttack
        alg_cfg = {
            "constraint": "l2",
            "rnd": random.Random(1234),
            "verbose": 1,
            "project_with_clip": True,
            "eval_batch_size": 8,
            "init_candidates": 18,
            "init_radii": [0.20, 0.45, 0.70, 0.92],
            "n_directions": 10,
            "sigma_init": 0.085,
            "sigma_min": 0.010,
            "sigma_max": 0.18,
            "step_init": 0.18,
            "step_min": 0.025,
            "step_max": 0.30,
            "line_search": [1.30, 0.90, 0.55, 0.25],
            "active_topk": 3,
            "memory_mix_every": 6,
            "restart_patience": 12,
            "max_restarts": 6,
            "log_every": 8,
            "coarse_parts": [2, 4, 8],
            "fine_parts": [8, 16, 32],
            "dct_freq_coarse": 4,
            "dct_freq_fine": 10,
        }

        alg = StructuredActiveLabelAttack(alg_cfg)
        problem.attack(alg)
        print("attack_rate", problem.attack_rate())
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        result_file.close()
