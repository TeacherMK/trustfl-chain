# Experiment Report

## Status

The final artifact-preparation matrix is complete. It combines the earlier reduced matrix with CIFAR-10 multi-seed supplementation and a FoolsGold-like `history_cosine` baseline.

- Reduced matrix: 191 / 191 runs completed.
- Final supplemental runs: 22 / 22 runs completed.
- Final combined summary rows: 213.
- Missing or incomplete rows: 0.
- PyTorch environment: CPU-only `torch 2.12.0+cpu`; CUDA was not visible in the virtual environment.

## Experimental Design

Fashion-MNIST is the main evidence base. CIFAR-10 is used as supplemental validation.

Main Fashion-MNIST matrix:

- Seeds: 42, 43, 44
- Rounds: 30
- Clients: 20
- Clients per round: 10
- Distributions: IID, Non-IID alpha 0.5, Non-IID alpha 0.1
- Malicious ratio: 20%
- Attacks: label flipping, sign flipping, Gaussian noise, model scaling
- Defenses: FedAvg, Trimmed Mean, Cosine filtering, TrustFL-Chain

final-artifact supplements:

- CIFAR-10 seeds 43 and 44 added to the existing seed 42 results.
- CIFAR-10 attacks: Gaussian noise and model scaling.
- CIFAR-10 defenses: FedAvg, Trimmed Mean, Cosine filtering, TrustFL-Chain.
- Added `history_cosine`, a FoolsGold-like baseline using historical update cosine similarity, on Fashion-MNIST for Gaussian noise and model scaling.

## RQ1: Robustness Across Attacks

Fashion-MNIST Non-IID alpha 0.5, 20% malicious clients, mean +/- std over 3 seeds:

| Attack | Strongest baseline final acc. | TrustFL-Chain final acc. | TrustFL-Chain mean F1 |
|---|---:|---:|---:|
| Gaussian noise | 0.720 +/- 0.013 | 0.782 +/- 0.023 | 0.802 |
| Model scaling | 0.681 +/- 0.015 | 0.710 +/- 0.015 | 0.803 |
| Label flipping | 0.686 +/- 0.025 | 0.712 +/- 0.016 | 0.644 |
| Sign flipping | 0.677 +/- 0.038 | 0.654 +/- 0.039 | 0.461 |

TrustFL-Chain is strongest for Gaussian noise and model scaling, and modestly improves label-flipping accuracy. Under sign flipping, Cosine filtering has higher final accuracy, while TrustFL-Chain still provides detection and audit information. This is a useful but mixed result rather than a universal win.

## RQ2: Multi-Feature Detection vs. Cosine-Only

Under Fashion-MNIST, Non-IID alpha 0.5, model scaling, and 20% malicious clients:

| Variant | Final acc. | Best acc. | Mean F1 | Final FPR |
|---|---:|---:|---:|---:|
| Full TrustFL-Chain | 0.710 +/- 0.015 | 0.724 +/- 0.008 | 0.803 | 0.131 |
| Cosine-only detector | 0.469 +/- 0.167 | 0.681 +/- 0.006 | 0.086 | 0.220 |

The cosine-only ablation is clearly weaker. This is the strongest evidence for the main methodological choice: under Non-IID updates, a single similarity score is not enough.

## RQ3: Dynamic Reputation

| Variant | Final acc. | Best acc. | Mean F1 |
|---|---:|---:|---:|
| Full TrustFL-Chain | 0.710 +/- 0.015 | 0.724 +/- 0.008 | 0.803 |
| No reputation | 0.592 +/- 0.133 | 0.713 +/- 0.011 | 0.768 |
| No temporal feature | 0.714 +/- 0.018 | 0.723 +/- 0.010 | 0.799 |

Removing reputation lowers final accuracy and increases variance. Removing the temporal feature has little effect on final accuracy in this setting but increases false positives. Reputation is therefore useful mainly as a stabilizing aggregation signal.

## RQ4: Audit Ledger Overhead

The ledger is storage-light. In the main Fashion-MNIST settings, TrustFL-Chain stores about 54-56 KB after 30 rounds. Baseline runs with audit hashes are around 49 KB. Removing the ledger keeps accuracy and F1 unchanged, as expected, because the ledger is not used for training decisions.

Runtime overhead is moderate in the CPU-only environment: TrustFL-Chain is typically around 1.44-1.47 seconds per round, while most baselines are around 1.30-1.40 seconds per round.

## History-Cosine Baseline

The FoolsGold-like `history_cosine` baseline is weaker than TrustFL-Chain in the evaluated direct-update attacks:

| Attack | History-Cosine final acc. | History-Cosine mean F1 | TrustFL-Chain final acc. | TrustFL-Chain mean F1 |
|---|---:|---:|---:|---:|
| Gaussian noise | 0.100 | 0.000 | 0.782 | 0.802 |
| Model scaling | 0.556 | 0.241 | 0.710 | 0.803 |

This baseline should be described as a lightweight history-cosine comparison, not as a strict reproduction of FoolsGold.

## CIFAR-10 Supplemental Results

CIFAR-10 now has 3 seeds for the two direct update attacks:

| Attack | Strongest baseline final acc. | TrustFL-Chain final acc. | TrustFL-Chain mean F1 |
|---|---:|---:|---:|
| Gaussian noise | 0.209 +/- 0.021 | 0.297 +/- 0.036 | 0.805 |
| Model scaling | 0.168 +/- 0.026 | 0.212 +/- 0.030 | 0.812 |

The absolute CIFAR-10 accuracies are low because the model and sample budget are intentionally lightweight, but the trend supports transfer beyond Fashion-MNIST.

## Exploratory Statistical Comparisons

Paired t-tests are reported as exploratory because there are only 3 seeds per main condition. TrustFL-Chain significantly improves detection F1 over the selected strongest baselines in the tested scenarios. Accuracy improvements are clearer for Gaussian noise and CIFAR-10 direct update attacks; model-scaling accuracy on Fashion-MNIST is positive but not statistically strong with 3 seeds.

Detailed results are in `results/final_artifacts/statistical_tests.csv`.

## Negative Results and Limitations

- Sign flipping: TrustFL-Chain is not the best final-accuracy method; Cosine filtering performs better in the main setting.
- Label flipping: detection is moderate because the attack changes local training labels rather than directly creating extreme update geometry.
- High malicious ratio: TrustFL-Chain remains useful at 40% malicious clients but false positives rise.
- CIFAR-10: the setup is intentionally lightweight; absolute accuracy is not competitive with large models.
- Ledger: this is a local hash-chain audit ledger, not a production blockchain or consensus protocol.

## Final Conclusion

The evidence supports a cautious simulator-level research artifact when framed carefully. TrustFL-Chain should be positioned as a reproducible, lightweight, auditable defense prototype. Its strongest claim is not universal robustness, but the combined benefit of multi-feature detection, dynamic reputation, and low-storage auditability under Non-IID FL.

## Sign-Flip Strengthening Rerun

We tested a configurable `direction_filter` aggregation variant to see whether TrustFL-Chain could be strengthened under sign flipping. The variant applies stricter direction-consensus filtering before aggregation:

```powershell
--trustfl_aggregation direction_filter --trustfl_keep_ratio 0.6
```

The original artifact-stable behavior is kept as:

```powershell
--trustfl_aggregation reputation_weighted
```

The focused rerun uses Fashion-MNIST, Non-IID alpha 0.5, malicious ratio 0.2, 30 rounds, and seeds 42/43/44. Summary files are stored in `results/comparison_summary/`.

| Setting | Gaussian noise final acc. | Label flip final acc. | Model scaling final acc. | Sign flip final acc. | Sign flip mean F1 | Sign flip final FPR |
|---|---:|---:|---:|---:|---:|---:|
| Original final-artifact result | 0.782 | 0.712 | 0.710 | 0.654 | 0.461 | 0.304 |
| Current reputation-weighted rerun | 0.698 | 0.392 | 0.542 | 0.345 | 0.235 | 0.345 |
| Direction-filter rerun | 0.089 | 0.437 | 0.402 | 0.462 | 0.264 | 0.387 |

The direction-filter variant slightly improves sign-flip accuracy compared with the current Cosine rerun recorded separately (`0.462` vs. `0.434` final accuracy), but it severely damages Gaussian-noise robustness and weakens model-scaling performance. Therefore it is not suitable as the main artifact-stable method.

Recommendation: keep the original `results/final_artifacts` evidence as the main artifact result and present direction filtering only, if needed, as a short negative/optional variant showing the trade-off between sign-flip robustness and general robustness.
