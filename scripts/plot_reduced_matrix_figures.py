from __future__ import annotations

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "reduced_matrix"
DEFAULT_FIGURES = ROOT / "figures" / "reduced_matrix"


def bar(data: pd.DataFrame, x: str, y: str, title: str, filename: str, color="#2563eb") -> None:
    if data.empty:
        return
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    plt.bar(data[x].astype(str), data[y], color=color)
    plt.title(title)
    plt.ylabel(y)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES / filename, dpi=200)
    plt.close()


def line(data: pd.DataFrame, x: str, y: str, hue: str, title: str, filename: str) -> None:
    if data.empty:
        return
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    for key, group in data.groupby(hue):
        plt.plot(group[x], group[y], marker="o", label=str(key))
    plt.title(title)
    plt.ylabel(y)
    plt.xlabel(x)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / filename, dpi=200)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot reduced TrustFL-Chain study figures.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--figures-dir", default=str(DEFAULT_FIGURES))
    args = parser.parse_args()
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    global FIGURES
    FIGURES = Path(args.figures_dir)
    if not FIGURES.is_absolute():
        FIGURES = ROOT / FIGURES
    summary_path = output_root / "summary_all_runs.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}; run aggregate_reduced_results.py first.")
    summary = pd.read_csv(summary_path)
    completed = summary[summary["complete"].astype(str).str.lower() == "true"].copy()
    if completed.empty:
        print("No completed runs to plot.")
        return 0

    main = completed[(completed["phase"] == "main") & (completed["dataset"] == "fashion_mnist")]
    main_alpha05 = main[main["distribution"] == "alpha_0.5"]
    core = (
        main_alpha05.groupby(["defense", "attack"], as_index=False)
        .agg(best_accuracy=("best_accuracy", "mean"), mean_f1=("mean_f1", "mean"), mean_runtime_sec=("mean_runtime_sec", "mean"))
    )
    if not core.empty:
        core["method_attack"] = core["defense"] + "\n" + core["attack"]
        bar(core, "method_attack", "best_accuracy", "Fashion-MNIST robustness by defense/attack", "main_accuracy_by_defense_attack.png")
        bar(core, "method_attack", "mean_f1", "Fashion-MNIST detection F1 by defense/attack", "main_f1_by_defense_attack.png", "#059669")
        runtime = core.groupby("defense", as_index=False)["mean_runtime_sec"].mean()
        bar(runtime, "defense", "mean_runtime_sec", "Runtime overhead by defense", "runtime_by_defense.png", "#7c3aed")

    dist = (
        main.groupby(["distribution", "defense"], as_index=False)
        .agg(best_accuracy=("best_accuracy", "mean"), mean_f1=("mean_f1", "mean"))
    )
    if not dist.empty:
        dist["label"] = dist["distribution"] + "\n" + dist["defense"]
        bar(dist, "label", "best_accuracy", "IID vs Non-IID best accuracy", "distribution_accuracy.png", "#ea580c")

    ratio = completed[completed["phase"] == "ratio"].copy()
    if not ratio.empty:
        ratio["malicious_ratio"] = ratio["malicious_ratio"].astype(float)
        ratio_agg = ratio.groupby(["malicious_ratio", "defense"], as_index=False).agg(mean_f1=("mean_f1", "mean"), best_accuracy=("best_accuracy", "mean"))
        line(ratio_agg, "malicious_ratio", "mean_f1", "defense", "Malicious ratio vs detection F1", "ratio_f1.png")
        line(ratio_agg, "malicious_ratio", "best_accuracy", "defense", "Malicious ratio vs best accuracy", "ratio_accuracy.png")

    ablation = completed[completed["phase"] == "ablation"].copy()
    if not ablation.empty:
        abl = ablation.groupby("ablation", as_index=False).agg(best_accuracy=("best_accuracy", "mean"), mean_f1=("mean_f1", "mean"))
        bar(abl, "ablation", "best_accuracy", "Ablation best accuracy", "ablation_accuracy.png", "#0891b2")
        bar(abl, "ablation", "mean_f1", "Ablation detection F1", "ablation_f1.png", "#16a34a")

    cifar = completed[completed["phase"] == "cifar_supplement"].copy()
    if not cifar.empty:
        cifar_agg = cifar.groupby(["defense", "attack"], as_index=False).agg(best_accuracy=("best_accuracy", "mean"), mean_f1=("mean_f1", "mean"))
        cifar_agg["label"] = cifar_agg["defense"] + "\n" + cifar_agg["attack"]
        bar(cifar_agg, "label", "best_accuracy", "CIFAR-10 supplemental best accuracy", "cifar_supplement_accuracy.png", "#dc2626")
        bar(cifar_agg, "label", "mean_f1", "CIFAR-10 supplemental detection F1", "cifar_supplement_f1.png", "#65a30d")

    ledger = completed.groupby("defense", as_index=False)["final_ledger_bytes"].mean()
    bar(ledger, "defense", "final_ledger_bytes", "Ledger storage overhead", "ledger_bytes_by_defense.png", "#475569")
    print(f"Wrote reduced study figures to {FIGURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
