# Literature Review and Research Positioning

## Scope

TrustFL-Chain sits at the intersection of blockchain-enabled federated learning, poisoning-robust aggregation, reputation-aware FL, anomaly detection for malicious model updates, and Non-IID robustness.

## Representative Work

| Category | Representative work | Main idea | Relevance to TrustFL-Chain |
|---|---|---|---|
| Byzantine robust aggregation | Blanchard et al., *Machine Learning with Adversaries*, NeurIPS 2017 | Krum selects an update close to its nearest neighbors under bounded Byzantine workers. | Provides a distance-based robust baseline. |
| Robust statistics for distributed learning | Yin et al., *Byzantine-Robust Distributed Learning*, ICML 2018 | Coordinate-wise median and trimmed mean with statistical guarantees. | Provides median and trimmed-mean baselines. |
| Trust-bootstrapped FL | Cao et al., *FLTrust*, NDSS 2021 | Uses a server root dataset to score and normalize client updates. | Closest trust-weighted robust FL baseline, but assumes server-side clean data. |
| Blockchain FL | Kim et al., *Blockchained On-Device Federated Learning*, IEEE Communications Letters 2019 | Uses blockchain to exchange and verify model updates and studies latency. | Motivates auditability, but targets decentralized consensus rather than lightweight local audit. |
| Blockchain FL surveys | Recent surveys on blockchain-based/blockchained FL | Taxonomies of decentralized FL, auditability, incentives, and consensus issues. | Shows the area is active and that lightweight audit is a pragmatic subproblem. |
| Non-IID robust FL | Robust FL literature and empirical studies | Non-IID honest updates can look anomalous under single similarity/distance rules. | Motivates multi-feature detection and recoverable reputation instead of permanent exclusion. |

## Has a Similar Idea Been Done?

Partially. Prior work has studied robust aggregation, trust scores, reputation/incentive mechanisms, and blockchain-backed FL. However, the exact prototype here is narrower: a single-machine reproducible system that combines multi-feature update anomaly detection, dynamic reputation recovery, and an append-only hash ledger without putting full model weights on-chain. It should be described as an engineering synthesis and empirical study, not as a first-of-its-kind security primitive.

## Research Gap

1. Single-feature filters such as cosine similarity can falsely penalize honest Non-IID clients whose updates naturally differ from the mean.
2. Strong filtering can improve robustness but may permanently suppress rare but useful client distributions.
3. Many blockchain FL designs emphasize consensus, incentives, or decentralized architectures, which can be heavier than needed for controlled experimentation.
4. Reproducible prototypes that jointly report accuracy, detection quality, reputation dynamics, runtime overhead, and ledger storage overhead are still useful for empirical comparison.

## Research Questions

- **RQ1:** Under different malicious client ratios, does TrustFL-Chain improve global accuracy compared with FedAvg and lightweight robust baselines?
- **RQ2:** Is multi-feature anomaly detection more suitable than cosine-only detection under Dirichlet Non-IID data?
- **RQ3:** Does dynamic reputation recovery reduce the performance loss caused by false positives on honest heterogeneous clients?
- **RQ4:** Are the computation and storage overheads of a local hash-chain audit ledger acceptable for single-machine FL simulation?

## Claimed Contributions

1. A lightweight multi-feature anomaly detector for client model updates under Non-IID FL.
2. A dynamic reputation update rule that penalizes high-risk updates while allowing gradual recovery.
3. A local append-only hash-chain ledger that records update hashes, risk scores, reputation changes, aggregation hashes, and previous block hashes.
4. A reproducible PyTorch prototype with multiple attacks, robust baselines, metrics, ablations, and release-ready artifacts.

## Reference Management

Bibliographic notes are kept separate from this public source-code artifact.
