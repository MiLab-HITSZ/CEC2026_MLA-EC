# Gradient-Plateau SBA

## Competition

Competition on Evolutionary Computation in Multi-Label Adversarial Examples

## Entrant and Final Result

- Entrant: Juewen Fan
- Final place: Winner (1st place)
- Overall mean ASR: 0.962875

## Algorithm Overview

Gradient-Plateau SBA is a two-stage black-box multi-label adversarial attack.
The first stage reduces the 602,112-dimensional pixel search space to a
low-frequency DCT subspace and estimates a zeroth-order gradient through
antithetic queries. Adam-style updates then move the perturbation toward the
target label vector. The second stage uses coordinate-wise SBA search to spend
the remaining query budget on difficult cases.

## Key Components

- **Low-frequency DCT search:** uses a 30 x 30 frequency block per color
  channel, reducing the working dimension to 2,700.
- **Antithetic gradient estimation:** evaluates positive and negative random
  directions to estimate a query-efficient descent direction.
- **Single-label rescue:** concentrates the search on the hardest remaining
  label when most target labels have already been satisfied.
- **Plateau escape:** applies stronger sparse and dense perturbations when a
  label stalls near the 0.5 decision boundary.
- **Zero-gradient restart:** restarts from controlled random DCT points when
  the initial region provides no useful local signal.
- **SBA fallback:** performs coordinate-level greedy refinement with the
  remaining query budget.

## Benchmark Results

| Model | Dataset | Mean ASR |
| --- | --- | ---: |
| ML-GCN | VOC 2007 | 0.999 |
| ML-GCN | VOC 2012 | 0.996 |
| ML-GCN | NUS-WIDE | 0.913 |
| ML-GCN | COCO | 0.984 |
| ML-LIW | VOC 2007 | 0.992 |
| ML-LIW | VOC 2012 | 0.995 |
| ML-LIW | NUS-WIDE | 0.944 |
| ML-LIW | COCO | 0.880 |
| **Overall mean** | **Eight settings** | **0.962875** |

## Reproducibility Notes

The implementation is provided in `gradient_plateau_sba.py`. It expects the
competition framework to provide `SingleImageProblem`, SciPy's DCT routines,
and the standard perturbation and query-budget settings.
