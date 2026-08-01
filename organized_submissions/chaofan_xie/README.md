

# Evolutionary Computation for Multi-Label Adversarial Examples

This project presents the source code of the competition on "Evolutionary Computation in MultiLabel Adversarial Examples"

---

## 📁 Dataset Preparation

Please download the `MLAE_cec_data` folder from the following link:

🔗 **Download:**
[https://drive.google.com/file/d/1mTPTDYoMzpcCOxeuiZfOPkxpFYdk9zcV/view?usp=drive_link](https://drive.google.com/file/d/1mTPTDYoMzpcCOxeuiZfOPkxpFYdk9zcV/view?usp=drive_link)

After downloading, place the folder anywhere on your machine and specify its full path in `proConfig["data_dir"]`.

### Directory Structure

```
MLAE_cec_data
│
├── adv_filter_data
│   ├── coco
│   ├── nuswide
│   ├── voc2007
│   └── voc2012
│
└── checkpoint
    ├── mlgcn
    └── mlliw
```

---

## ⚙️ Usage Example

Below is a minimal example for running an evolutionary adversarial attack:

```python
import random
from attack_problem import EvolutionaryAttackProblem
from de import DE_RAND1

proConfig = {
    "ml_model_name": "mlgcn",        # mlgcn / mlliw
    "dataset_name": "coco",          # coco / voc2007 / voc2012 / nuswide
    "target_type": "random",
    "epsilon": 77.596,               # Perturbation limit ε
    "max_eval": 10000,               # Max fitness evaluations per image (query limit)
    "data_dir": "/home/dyy/code/MLAE_cec_data"  # Path to MLAE_cec_data
}

algConfig = {
    "F": 0.5,
    "CR": 0.9,
    "pop_size": 100,
    "rnd": random.Random(1234)
}

problem = EvolutionaryAttackProblem(proConfig)
mlde = DE_RAND1(algConfig)

problem.attack(mlde)
print("attack_rate:", problem.attack_rate())
```

---

## 🔧 Configuration Details

### **Problem Configuration (`proConfig`)**

| Key             | Description                                         |
| --------------- | --------------------------------------------------- |
| `ml_model_name` | Multi-label model to attack: `mlgcn`, `mlliw`       |
| `dataset_name`  | Dataset: `coco`, `voc2007`, `voc2012`, `nuswide`    |
| `target_type`   | Attack types (`random`)                             |
| `epsilon`       | L2 perturbation bound                               |
| `max_eval`      | Maximum number of fitness evaluations (query limit) |
| `data_dir`      | Local path to `MLAE_cec_data`                       |

---

