# FedSCAM: Scam-resistant SAM for Robust Federated Optimization in Heterogeneous Environments

**Sameer Rahil\*, Zain Abdullah Ahmad\*, Talha Asif\*** (\*equal contribution)
Lahore University of Management Sciences (LUMS). Course research for Advanced Topics in Machine Learning, supervised by Prof. Muhammad Tahir, Fall 2025.

Paper: [arXiv:2601.00853](https://arxiv.org/abs/2601.00853) ([PDF in this repo](FedSCAM_paper.pdf))

## Summary

FedSAM brings sharpness-aware minimization (SAM) to federated learning, but it uses one perturbation radius ρ for every client. Under label skew, clients differ sharply in gradient scale and noise, so a single radius over-perturbs the unstable clients and amplifies client drift.

FedSCAM treats client heterogeneity as an explicit optimization signal:

1. **Heterogeneity score.** The server cannot see client labels, so each client gets a score from signals the server can compute: a cheap early-batch gradient norm and the alignment of a one-batch pilot update with the global direction.
2. **Per-client SAM radius.** Each client's radius shrinks as its heterogeneity score grows, so high-variance clients cannot destabilize the global model.
3. **Heterogeneity-aware aggregation.** Updates are reweighted by the same score and by their alignment with the global update direction, on top of the usual sample-size weights.

## Main results

| Setting | Result | Notebook |
|---|---|---|
| CIFAR-10, ResNet-18, Dirichlet α = 0.5 | FedSCAM 62.60% vs FedSAM 58.00% (+4.6 points) | `FedSCAM vs FedSAM.ipynb` |
| CIFAR-10, CNN, α = 0.5, aggregation-only variant | 70.63%, best of all 13 methods, above FedAvg (69.68%) and FedLW (69.42%) | `FedSCAM Ablation - 3.ipynb` |
| ResNet-18, 10 rounds, vs SAM-family baselines | 99.4% of FedLESAM's accuracy (65.41% vs 65.76%) at about 12% less time per round (144.6 s vs 163.9 s) | `FedSCAM - FashionMNIST - vs All baselines.ipynb` |

All methods in the ablations share one optimizer setup (SGD with momentum 0.9, weight decay, same local epochs and batch size), so differences come from the method rather than the training loop.

**Baselines re-implemented (9):** FedAvg, FedAvgM, FedProx, q-FedAvg, FedLW, FedSAM, FedLESAM, FedLWSAM, FedWMSAM. A uniform-weight average is also included as a sanity check.

## Repository map

Suggested reading order:

| Notebook | What it does |
|---|---|
| `Heterogeneity Score - Proof of Concept and Results.ipynb` | Tests which client-side signals track true label skew (entropy, KL divergence, gradient variance, cosine heterogeneity, L2 distance) on CIFAR-10 across α ∈ {0.1, 0.5, 1.0, 10.0}. This motivated the score used in FedSCAM. |
| `FedSCAM vs FedSAM.ipynb` | Head-to-head comparison on CIFAR-10 with ResNet-18 (the +4.6 point result). |
| `FedSCAM Hyperparameters.ipynb` | Sweeps of the radius and aggregation hyperparameters. |
| `FedSCAM Ablation - 3.ipynb` | Main ablation on CIFAR-10: full FedSCAM, SAM-only (`FedSCAM_SAM`) and aggregation-only (`FedSCAM_WA`) variants, each compared against its own family of baselines, for α ∈ {0.1, 0.5, 1.0}. |
| `FedSCAM - Ablation on FMNIST.ipynb` | The same ablation on Fashion-MNIST for α ∈ {0.1, 0.5}. |
| `FedSCAM - Ablation Continued.ipynb` | An earlier ablation run, superseded by `FedSCAM Ablation - 3.ipynb`; kept for the record. |
| `FedSCAM - FashionMNIST - vs All baselines.ipynb` | ResNet-18 comparison against the SAM-family and loss-weighted baselines, including time per round. |
| `FedSCAM_paper.pdf` | The paper. |

Each notebook is self-contained: it defines the data partitioning (Dirichlet label skew over 10 clients), the model, every method, and the training loop, then prints the results table and plots shown in the paper.

## Running

The notebooks were run on Google Colab with a single GPU. Datasets download automatically through `torchvision`.

```bash
pip install -r requirements.txt
jupyter notebook
```

Open any notebook and run all cells. The partition seed is fixed (42) so splits are reproducible.

## Citation

```bibtex
@misc{rahil2026fedscam,
  title         = {FedSCAM (Federated Sharpness-Aware Minimization with Clustered Aggregation and Modulation): Scam-resistant SAM for Robust Federated Optimization in Heterogeneous Environments},
  author        = {Rahil, Sameer and Ahmad, Zain Abdullah and Asif, Talha},
  year          = {2026},
  eprint        = {2601.00853},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```

## Related work by the authors

- [Label heterogeneity and FedAvg on CIFAR-10](https://github.com/SameerRahil/Advanced-Topics-in-Machine-Learning-PA4): the study of client drift under label skew that led to this project.
- [Sparse autoencoders and mechanistic interpretability](https://github.com/SameerRahil/Advanced-Topics-in-Machine-Learning-PA5): universal sparse autoencoders on ResNet and ViT activations, alongside decoding and alignment experiments on a small language model.
