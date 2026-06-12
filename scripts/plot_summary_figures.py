from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "summary.csv"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

df = pd.read_csv(SUMMARY)


def save_bar(data, x, y, path, ylabel):
    plt.figure(figsize=(7.5, 4.2))
    plt.bar(data[x], data[y], color="#3b82f6")
    plt.ylabel(ylabel)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES / path, dpi=200)
    plt.close()


core = df[(df["group"] == "summary_core_fmnist") & (df["attack"] == "model_scaling")].copy()
if not core.empty:
    order = ["fedavg", "median", "trimmed_mean", "krum", "cosine", "trustfl_chain"]
    core["defense"] = pd.Categorical(core["defense"], categories=order, ordered=True)
    core = core.sort_values("defense")
    save_bar(core, "defense", "best_accuracy", "summary_core_best_accuracy.png", "Best accuracy")
    save_bar(core, "defense", "mean_runtime_sec", "summary_core_runtime.png", "Mean runtime per round (s)")

abl = df[df["group"].str.startswith("summary_ablation_fmnist")].copy()
if not abl.empty:
    abl["ablation"] = abl["group"].str.split("\\\\").str[-1]
    save_bar(abl.sort_values("ablation"), "ablation", "best_accuracy", "summary_ablation_best_accuracy.png", "Best accuracy")
    save_bar(abl.sort_values("ablation"), "ablation", "mean_f1", "summary_ablation_mean_f1.png", "Mean detection F1")

att = df[df["group"].str.startswith("summary_attack_fmnist")].copy()
if not att.empty:
    att["method"] = att["attack"] + "-" + att["defense"]
    save_bar(att.sort_values("method"), "method", "best_accuracy", "summary_attack_best_accuracy.png", "Best accuracy")
    save_bar(att.sort_values("method"), "method", "mean_f1", "summary_attack_mean_f1.png", "Mean detection F1")

ratio = df[df["group"].str.startswith("summary_ratio_fmnist")].copy()
if not ratio.empty:
    ratio["ratio"] = ratio["malicious_ratio"].astype(str)
    save_bar(ratio.sort_values("malicious_ratio"), "ratio", "mean_f1", "summary_ratio_mean_f1.png", "Mean detection F1")

print(f"Wrote study figures to {FIGURES}")
