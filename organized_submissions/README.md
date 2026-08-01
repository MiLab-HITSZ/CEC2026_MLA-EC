# Competition on Evolutionary Computation in Multi-Label Adversarial Examples

This directory contains the three source-code submissions received for the
Competition on Evolutionary Computation in Multi-Label Adversarial Examples.
The original archives remain unchanged in the parent directory; this directory
provides a consistent, English-language layout for review and reproducibility
work.

## Submission index

| Directory | Entrant | Algorithm | Main file |
| --- | --- | --- | --- |
| `juewen_fan/` | Juewen Fan | Gradient-Plateau SBA | `gradient_plateau_sba.py` |
| `chaofan_xie/` | Chaofan Xie | Multi-Archive Search with SBA | `run_attack.py` |
| `gaoren_zhang/` | Gaoren Zhang | Structured Active-Label Attack | `main.py` |

## Benchmark protocol

- Models: ML-GCN and ML-LIW
- Datasets: PASCAL VOC 2007, PASCAL VOC 2012, MS-COCO, and NUS-WIDE
- Attack: black-box random multilabel attack
- Perturbation constraint: L2 norm at most 77.596
- Query budget: at most 10,000 fitness evaluations per image
- Primary metric: attack success rate (ASR)
- Final ranking metric: arithmetic mean of the eight submitted Mean ASR values

## Final competition ranking

The competition organizer's final decision is to rank entries by their overall
mean ASR across the eight model-dataset settings. Higher is better.

| Place | Entrant | Algorithm | Overall mean ASR |
| ---: | --- | --- | ---: |
| 1 | Juewen Fan | Gradient-Plateau SBA | 0.962875 |
| 2 | Chaofan Xie | Multi-Archive Search with SBA | 0.823625 |
| 3 | Gaoren Zhang | Structured Active-Label Attack | 0.763125 |

The complete source-backed result table and formula-driven ranking are in
`final_results.xlsx`.

## Reproduction notes

The benchmark data and trained checkpoints are not duplicated in each
submission. Configure the data directory using the environment variable or
configuration key documented by each entry point. Execute all methods in the
same benchmark environment and with the same image set before comparing ASR.

## Reproduction result schema

Use the following columns for the consolidated result file:

```text
entrant,algorithm,model,dataset,run,success_count,total_count,asr
```
