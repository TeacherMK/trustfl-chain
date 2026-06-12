from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

files = list(RESULTS.rglob("*.csv"))
main_files = [p for p in files if not p.name.endswith("_reputation.csv")]
if main_files:
    df = pd.concat([pd.read_csv(p) for p in main_files], ignore_index=True)
    plt.figure(figsize=(7, 4))
    for key, group in df.groupby(["defense", "attack"]):
        plt.plot(group["round"], group["accuracy"], marker="o", label=f"{key[0]}-{key[1]}")
    plt.xlabel("Federated round")
    plt.ylabel("Global test accuracy")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "accuracy_curves.png", dpi=180)

    plt.figure(figsize=(7, 4))
    for key, group in df.groupby(["defense", "attack"]):
        plt.plot(group["round"], group["f1"], marker="o", label=f"{key[0]}-{key[1]}")
    plt.xlabel("Federated round")
    plt.ylabel("Detection F1")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "detection_f1.png", dpi=180)

    plt.figure(figsize=(6, 4))
    last = df.sort_values("round").groupby(["defense", "attack"]).tail(1)
    plt.bar([f"{r.defense}\n{r.attack}" for r in last.itertuples()], last["runtime_sec"])
    plt.ylabel("Runtime per round (s)")
    plt.tight_layout()
    plt.savefig(FIGURES / "runtime_overhead.png", dpi=180)

rep_files = [p for p in files if p.name.endswith("_reputation.csv")]
if rep_files:
    rep = pd.concat([pd.read_csv(p) for p in rep_files], ignore_index=True)
    plt.figure(figsize=(7, 4))
    for malicious, group in rep.groupby("malicious"):
        mean = group.groupby("round")["reputation"].mean()
        plt.plot(mean.index, mean.values, marker="o", label="malicious" if malicious else "benign")
    plt.xlabel("Federated round")
    plt.ylabel("Mean reputation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "reputation_evolution.png", dpi=180)

print(f"Wrote figures to {FIGURES}")
