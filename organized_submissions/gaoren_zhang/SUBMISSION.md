# Structured Active-Label Attack

## Competition

Competition on Evolutionary Computation in Multi-Label Adversarial Examples

## Entrant and Final Result

- Entrant: Gaoren Zhang
- Final place: Second Runner-Up (3rd place)
- Overall mean ASR: 0.763125

## Algorithm Overview

Structured Active-Label Attack is a staged method for black-box multi-label
classifiers. During early search, it explores structured low-frequency block,
stripe, and DCT directions to obtain broad progress. It uses per-label residual
information to prioritize unsatisfied target labels. When only a few difficult
labels remain, it switches to low-frequency coordinate-level positive and
negative trials with adaptive step-size refinement.

## Key Components

- **Structured global exploration:** samples block, stripe, and low-frequency
  DCT directions instead of searching arbitrary pixel perturbations.
- **Active-label selection:** prioritizes labels with the largest remaining
  target residuals.
- **Label memory:** retains useful perturbation information associated with
  difficult labels.
- **Adaptive local refinement:** adjusts search radius and step size according
  to recent progress.
- **Restart strategy:** restores diversity when the current center stagnates.
- **Rescue search:** focuses coordinate-level symmetric trials on the final
  hard-to-flip labels.

## Benchmark Results

| Model | Dataset | Mean ASR |
| --- | --- | ---: |
| ML-GCN | VOC 2007 | 0.977 |
| ML-GCN | VOC 2012 | 0.978 |
| ML-GCN | NUS-WIDE | 0.020 |
| ML-GCN | COCO | 0.890 |
| ML-LIW | VOC 2007 | 0.918 |
| ML-LIW | VOC 2012 | 0.957 |
| ML-LIW | NUS-WIDE | 0.718 |
| ML-LIW | COCO | 0.647 |
| **Overall mean** | **Eight settings** | **0.763125** |

## Reproducibility Notes

`main.py` is the experiment entry point. The submitted runner supports dataset
path, image-range, result-suffix, and CUDA-device environment variables. The
algorithm module must be available in the benchmark framework's
`attack_algorithm` package.
