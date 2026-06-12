# TrustFL-Chain

TrustFL-Chain is a reproducible PyTorch research prototype for studying lightweight poisoned-update detection, reputation-weighted aggregation, and compact hash-chain audit logging in non-IID federated learning.

The project is intentionally scoped as an empirical simulator. It does not implement production blockchain consensus, secure aggregation, zero-knowledge proofs, trusted execution, or a universal poisoning defense.

## What Is Included

- Federated learning simulator with IID and Dirichlet non-IID client partitions.
- Poisoning attacks: label flipping, sign flipping, Gaussian-noise updates, and model scaling.
- Defenses and baselines: FedAvg, trimmed mean, cosine filtering, History-Cosine, and TrustFL-Chain.
- TrustFL-Chain components: update feature extraction, anomaly scoring, soft client reputation, weighted aggregation, and hash-chain audit records.
- Reproducibility scripts for the reduced study matrix, final aggregation tables, statistical tests, and figures.
- Generated figures and reproducibility artifacts.

## Repository Layout

```text
figures/final_artifacts/ Final artifact figures
results/final_artifacts/ Final aggregate CSVs and supplemental runs
results/reduced_matrix/  Main reduced experiment matrix outputs
scripts/                 Experiment, aggregation, and plotting scripts
src/                     TrustFL-Chain simulator implementation
```

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The experiments use PyTorch and torchvision. Dataset files are downloaded locally into `data/`, which is intentionally ignored by Git.

## Quick Check

Run a one-round synthetic sanity check:

```powershell
.\.venv\Scripts\python.exe src/main.py --dataset synthetic --no_download --rounds 1 --num_clients 20 --clients_per_round 10 --malicious_ratio 0.2 --attack model_scaling --defense trustfl_chain --non_iid true --alpha 0.5
```

For real datasets, the simulator fails loudly if the dataset cannot be loaded. Use `--dataset synthetic` for offline smoke tests. The optional `--allow_synthetic_fallback` flag is available only when an explicit FakeData fallback is desired.

Run the History-Cosine baseline:

```powershell
.\.venv\Scripts\python.exe src/main.py --dataset synthetic --no_download --rounds 1 --num_clients 20 --clients_per_round 10 --malicious_ratio 0.2 --attack model_scaling --defense history_cosine --non_iid true --alpha 0.5
```

## Reproducing the Study Results

The final artifact evidence combines the reduced Fashion-MNIST matrix with supplemental CIFAR-10 and History-Cosine runs.

Generate the final supplemental manifest:

```powershell
.\.venv\Scripts\python.exe scripts/run_final_artifacts_experiments.py --manifest-only
```

Run or resume the supplemental experiments:

```powershell
.\.venv\Scripts\python.exe scripts/run_final_artifacts_experiments.py --execute
```

Aggregate the final results:

```powershell
.\.venv\Scripts\python.exe scripts/aggregate_final_artifacts_results.py
```

Regenerate final figures:

```powershell
.\.venv\Scripts\python.exe scripts/plot_final_artifacts_figures.py
```

To reproduce the main reduced matrix from scratch:

```powershell
.\.venv\Scripts\python.exe scripts/run_reduced_matrix.py --manifest-only
.\.venv\Scripts\python.exe scripts/run_reduced_matrix.py --execute
.\.venv\Scripts\python.exe scripts/aggregate_reduced_results.py
.\.venv\Scripts\python.exe scripts/plot_reduced_matrix_figures.py
```

## Repository Availability

This public repository is maintained as the implementation and reproducibility artifact for TrustFL-Chain. It includes the simulator, run manifests, aggregate outputs, figure-generation scripts, and recorded results needed to inspect and reproduce the released experiments.

## Main Empirical Takeaways

Fashion-MNIST is the primary multi-seed evidence base. Under Dirichlet non-IID alpha 0.5 and 20% malicious clients, TrustFL-Chain is strongest on direct update manipulation attacks. It improves final accuracy over the strongest baseline under Gaussian-noise and model-scaling attacks, while producing useful detection F1. The results are weaker for label flipping and sign flipping; cosine filtering remains a strong baseline for purely directional sign-flip attacks.

CIFAR-10 is used as supplemental validation with a shorter training budget. The CIFAR-10 results support the same broad trend for direct update attacks, but they should not be read as the central claim.

## Scope and Limitations

TrustFL-Chain assumes a trusted server that can inspect submitted client updates. It does not evaluate secure aggregation, adaptive white-box attackers, privacy leakage from update inspection, asynchronous FL, network failures, or production blockchain protocols. The hash-chain ledger records compact audit metadata for reproducibility and inspection; it is not a decentralized consensus mechanism.

## Development Note

Codex was used only to help manage the GitHub repository structure and release documentation.

## License

This project is released under the MIT License. See `LICENSE` for details.
