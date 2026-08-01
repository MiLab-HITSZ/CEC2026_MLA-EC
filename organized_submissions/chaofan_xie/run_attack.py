import os
import random
from datetime import datetime

from attack_problem.problem import EvolutionaryAttackProblem
from attack_algorithm.SA_multi_archive import SA_MultiArchive


def build_lightweight_config():
    data_dir = os.environ.get("MLAE_DATA_DIR", "/home/dyy/723xcf/MLAE_cec_data")
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

    pro_config = {
        "ml_model_name": "mlgcn",
        "dataset_name": "nuswide",
        "target_type": "random",
        "image_size": 448,
        "epsilon": 77.596,
        "max_eval": 1000,
        "max_images": 5,
        "batch_size": 1,
        "workers": 0,
        "data_dir": data_dir,
    }

    alg_config = {
        "F": 0.5,
        "CR": 0.9,
        "pop_size": 20,
        "eps": 0.01,
        "rnd": random.Random(1234),
    }

    print(f"CUDA_VISIBLE_DEVICES={gpu_id}")
    print(f"data_dir={data_dir}")
    print(f"lightweight pro_config={pro_config}")
    alg_config_log = {k: v for k, v in alg_config.items() if k != "rnd"}
    print(f"lightweight alg_config={alg_config_log}")
    return pro_config, alg_config


def run_lightweight_attack():
    pro_config, alg_config = build_lightweight_config()
    problem = EvolutionaryAttackProblem(pro_config)
    attack = MyAttack(alg_config)
    problem.attack(attack)
    print("attack_rate", problem.attack_rate())


def run_one(model_name, dataset_name, data_dir, log_dir, seed):
    os.makedirs(log_dir, exist_ok=True)

    summary_log = os.path.join(log_dir, f"summary_{model_name}_{dataset_name}.txt")
    success_dir = os.path.join(log_dir, "success_images", model_name, dataset_name)

    pro_config = {
        "ml_model_name": model_name,
        "dataset_name": dataset_name,
        "target_type": "random",
        "epsilon": 77.596,
        "max_eval": 10000,
        "data_dir": data_dir,
        "save_success_adv_images": False,
        "success_adv_dir": success_dir,
        "attack_summary_log": summary_log,
    }

    alg_config = {
        "rnd": random.Random(seed),
    }

    print(
        "Starting attack: "
        f"model={model_name}, dataset={dataset_name}, seed={seed}, "
        f"summary_log={summary_log}"
    )
    problem = EvolutionaryAttackProblem(pro_config)
    alg = SA_MultiArchive(alg_config)
    problem.attack(alg)

    result = {
        "model": model_name,
        "dataset": dataset_name,
        "success_count": problem.success_count,
        "total_count": problem.total_count,
        "attack_rate": problem.attack_rate(),
        "summary_log": summary_log,
    }
    print(
        "Finished attack: "
        f"model={model_name}, dataset={dataset_name}, "
        f"success_count={result['success_count']}, total_count={result['total_count']}, "
        f"attack_rate={result['attack_rate']:.6f}"
    )
    return result


def append_all_summary(result, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, "summary_all.txt")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "a", encoding="utf-8") as f:
        f.write(
            "Attack summary: "
            f"time={timestamp}, "
            f"dataset={result['dataset']}, "
            f"model={result['model']}, "
            f"success_count={result['success_count']}, "
            f"total_count={result['total_count']}, "
            f"attack_rate={result['attack_rate']:.6f}, "
            f"summary_log={result['summary_log']}\n"
        )
    print(f"Appended summary to {path}")


if __name__ == '__main__':
    data_dir = os.environ.get("MLAE_DATA_DIR", "/home/dyy/fjw/MLAE_cec_data")
    log_dir = os.environ.get("ATTACK_LOG_DIR", "attack_logs")

    models = ["mlgcn", "mlliw"]
    datasets = ["voc2007", "voc2012", "coco", "nuswide"]

    results = []
    base_seed = 20250101
    for model_index, model_name in enumerate(models):
        for dataset_index, dataset_name in enumerate(datasets):
            seed = base_seed + model_index * 100 + dataset_index
            result = run_one(model_name, dataset_name, data_dir, log_dir, seed)
            results.append(result)
            append_all_summary(result, log_dir)

    print(f"Finished all attacks. Completed {len(results)} dataset/model runs.")
