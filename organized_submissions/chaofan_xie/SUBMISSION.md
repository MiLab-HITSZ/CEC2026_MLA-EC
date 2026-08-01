# Hybrid SBA-SA Multi-Archive Attack

## Competition

Competition on Evolutionary Computation in Multi-Label Adversarial Examples

## Entrant and Final Result

- Entrant: Chaofan Xie
- Final place: First Runner-Up (2nd place)
- Overall mean ASR: 0.823625

## Algorithm Overview

This method is a hybrid black-box attack that combines an SBA-based
gradient-estimation stage with a simulated-annealing multi-archive search. It
is designed for multi-label objectives whose target labels may have very
different fitness landscapes. The gradient stage efficiently follows useful
local descent information, while the archive-based stage handles stagnation,
bottleneck labels, and plateau regions.

## Key Components

- **SBA gradient-estimation stage:** jointly optimizes all currently failed
  labels using sampled directions and line search.
- **Stagnation detection:** switches search modes when the maximum label-wise
  fitness does not improve sufficiently for consecutive iterations.
- **Simulated-annealing search:** introduces randomized acceptance and stronger
  global exploration after local progress stalls.
- **Multiple archives:** retains the current solution, best overall solution,
  bottleneck-label solution, balanced solution, and label-specific solutions.
- **Query reuse:** a single evaluated candidate is used to update every
  relevant archive, improving information efficiency.
- **Plateau escape:** temporarily focuses on a stalled label and applies
  low-frequency and large-square mutations until improvement is found.

## Benchmark Results

| Model | Dataset | Mean ASR |
| --- | --- | ---: |
| ML-GCN | VOC 2007 | 0.933 |
| ML-GCN | VOC 2012 | 0.943 |
| ML-GCN | NUS-WIDE | 0.665 |
| ML-GCN | COCO | 0.822 |
| ML-LIW | VOC 2007 | 0.925 |
| ML-LIW | VOC 2012 | 0.908 |
| ML-LIW | NUS-WIDE | 0.715 |
| ML-LIW | COCO | 0.678 |
| **Overall mean** | **Eight settings** | **0.823625** |

## Reproducibility Notes

`run_attack.py` is the full benchmark entry point. Set `MLAE_DATA_DIR` to the
benchmark data directory and optionally use `ATTACK_LOG_DIR` for result logs.
