# Evolutionary-Based Neural Architecture Search Optimisation for Sustainable Deep Learning

This repository contains the implementation for my 4th Year dissertation project at the University of Edinburgh (2024), exploring how fine-tuning genetic algorithms within an evolutionary algorithm-based Neural Architecture Search (NAS) system can improve the trade-off between model accuracy and computational efficiency.

The work builds upon [GeneticNAS](https://arxiv.org/abs/1907.02871) (Habi & Rafalovich, 2019) and proposes four novel modifications to its genetic algorithm strategies, evaluated on CIFAR-10 and CIFAR-100 image classification benchmarks.

---

## Overview

Modern deep neural networks are computationally expensive to design and train. Neural Architecture Search automates this process, but most NAS systems optimise purely for accuracy while ignoring sustainability. This project investigates whether targeted modifications to the genetic algorithm component of an EA-based NAS can improve search efficiency — finding better architectures faster, with fewer resources — without sacrificing accuracy.
📄 [Full dissertation PDF](./Kian_Lee_dissertation_2024.pdf)

**Key contributions:**
- Three novel mutation strategies: Gaussian Flip, Uniform Random, and Adaptive Flip (v0–v2)
- A Blend Crossover operator adapted for cell-structured search spaces
- Empirical evaluation across CIFAR-10 and CIFAR-100, measuring validation accuracy, model complexity (parameter count), and search efficiency (time to threshold)
- A hybrid model combining Adaptive Flip v.2 + Blend Crossover, achieving the best results on CIFAR-10

---

## Results Summary

### CIFAR-10

| Model | Final Val. Accuracy | Params | Efficiency Gain vs Baseline |
|---|---|---|---|
| Baseline | 93.17% | 713,673 | — |
| Adaptive Flip v.2 (107x) | 93.19% | 649,793 (−8.95%) | +51.09% |
| Blend Crossover | 93.63% | 965,233 (+35.25%) | +48.61% |
| **Adaptive Flip v.2 + Blend (hybrid)** | **94.03%** | 852,833 (+19.5%) | **+54.98%** |

> The hybrid model improved accuracy by 0.86% over baseline while reducing search time by over half.

### CIFAR-100

| Model | Final Val. Accuracy | Params | Efficiency Gain vs Baseline |
|---|---|---|---|
| Baseline | 76.21% | 3,666,760 | — |
| Adaptive Flip v.2 (107x) | 77.69% | 3,787,936 (+3.3%) | +35.17% |
| Blend Crossover | 78.69% | 4,150,336 (+13.19%) | +25.11% |
| Adaptive Flip v.2 + Blend (hybrid) | 78.69% | 4,505,824 (+22.88%) | +23.91% |

> Adaptive Flip v.2 achieved the best accuracy improvement (+1.48%) with the smallest parameter overhead.

---

## Repository Structure

```
.
├── configs/                  # JSON config files for search and final training
├── gnas/                     # Core GeneticNAS implementation (EA logic, population management)
├── models/                   # Network architecture definitions (cells, DAG, CNN backbone)
├── modules/                  # Reusable building blocks (SE blocks, depthwise convolutions, etc.)
├── tests/                    # Unit tests
├── main.py                   # Entry point for architecture search and final training
├── config.py                 # Config loader
├── data.py                   # Dataset loading and augmentation (CIFAR-10/100)
├── cnn_utils.py              # CNN training utilities
├── rnn_utils.py              # RNN utilities
├── common.py                 # Shared helpers
├── plot_result.py            # Result visualisation scripts
└── gif_creator.py            # Visualisation of DAG architecture evolution
```

---

## Installation

Requires Python 3.7+ and a CUDA-capable GPU (experiments were run on a V100).

### Using conda (recommended)

```bash
conda create -n gnas python=3.7
conda activate gnas

conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
conda install graphviz
conda install pygraphviz
conda install numpy
```

### Using pip

```bash
pip install torch torchvision
pip install graphviz pygraphviz numpy
```

---

## Usage

### Architecture Search

Run the NAS training search phase to discover an optimal cell architecture.

**CIFAR-10:**
```bash
python main.py --dataset_name CIFAR10 --config_file ./configs/config_cnn_search_cifar10.json
```

**CIFAR-100:**
```bash
python main.py --dataset_name CIFAR100 --config_file ./configs/config_cnn_final_cifar100.json
```

A log directory is created under the current folder at the end of the search, containing the best discovered architecture and training history.

### Final Training

Once a search is complete, train the discovered architecture from scratch to evaluate its true performance. Replace `$LOG_DIR` with the path to your search output folder.

**CIFAR-10:**
```bash
python main.py --dataset_name CIFAR10 --final 1 --search_dir $LOG_DIR --config_file ./configs/config_cnn_final_cifar10.json
```

**CIFAR-100:**
```bash
python main.py --dataset_name CIFAR100 --final 1 --search_dir $LOG_DIR --config_file ./configs/config_cnn_final_cifar100.json
```

### Key Config Parameters

The config JSON files expose the main hyperparameters. The most relevant for reproducing the proposed experiments:

| Parameter | Description | Default (CIFAR-10) |
|---|---|---|
| `mutation_p` | Mutation probability | 0.02 |
| `cross_over_type` | `"Block"` or `"Blend"` | `"Block"` |
| `p_cross_over` | Crossover probability | 1.0 |
| `n_epochs` | Training epochs | 100 |
| `population_size` | EA population size | 20 |
| `generation_per_epoch` | EA generations per training epoch | 2 |
| `n_channels` | Base channel count | 20 |
| `n_nodes` | Nodes per cell | 5 |

---

## Proposed Methods

### Mutation Strategies

**Gaussian Flip** — augments the original signed flip mutation with a Gaussian-distributed perturbation, encouraging finer exploration of neighbouring solutions.

**Uniform Random** — replaces signed flip with uniform random gene values, promoting broader diversity in the population at the cost of exploitation.

**Adaptive Flip v.2** — dynamically adjusts the mutation rate during search based on population fitness stagnation. When mean fitness fails to improve beyond the current best for `x` consecutive generations, the mutation rate is scaled up by factor `f`. The rate resets every 20 epochs and is bounded at `default × 1.2` to prevent instability. Best-performing configuration: `f=1.07, x=4`.

### Crossover Strategy

**Blend Crossover (BLX-α-β)** — adapts the BLX operator to GeneticNAS's block-structured individuals. A random binary selector determines which blocks undergo blending; selected blocks have their genes blended within an interval determined by the distance between parent genes. Parameters: `α=0.5, β=0.1`.

---

## Experimental Setup

Both datasets use the standard CIFAR preprocessing pipeline from He et al. (2015): normalisation, 4-pixel padding, 32×32 random crop, random horizontal flip, and CutOut regularisation.

Training uses SGD with momentum (0.9), MultiStepLR scheduler (decay at 50% and 75% of total epochs), dropout (0.2), and weight decay (1e-4). All experiments were run for 100 epochs on an NVIDIA V100 GPU (≈12 hours for CIFAR-10, ≈16 hours for CIFAR-100).

---

## Citation

If you build on this work, please cite the original GeneticNAS paper this project extends:

```bibtex
@article{habi2019genetic,
  title={Genetic Neural Architecture Search},
  author={Habi, Hai Victor and Rafalovich, Gil},
  journal={arXiv preprint arXiv:1907.02871},
  year={2019}
}
```

---

## Acknowledgements

Supervised by Dr Michael Herrmann (University of Edinburgh). Additional feedback from David Symons. GeneticNAS baseline implementation by Habi & Rafalovich (2019).

---

## License

MIT
