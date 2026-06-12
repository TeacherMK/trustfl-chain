from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
REDUCED = ROOT / "results" / "reduced_matrix"
FINAL = ROOT / "results" / "final_artifacts"


def load_manifest_summary(manifest_path: Path) -> pd.DataFrame:
    rows = []
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        for item in csv.DictReader(f):
            csv_path = ROOT / item["expected_csv"]
            item = dict(item)
            item["source_csv"] = str(csv_path.relative_to(ROOT))
            item["distribution"] = "iid" if str(item["non_iid"]).lower() == "false" else f"alpha_{item['alpha']}"
            item["complete"] = False
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                expected_rounds = int(float(item["rounds"]))
                item["completed_rounds"] = int(df["round"].max()) if not df.empty else 0
                item["complete"] = item["completed_rounds"] >= expected_rounds
                if not df.empty:
                    last = df.sort_values("round").iloc[-1]
                    item.update(
                        {
                            "final_accuracy": float(last["accuracy"]),
                            "best_accuracy": float(df["accuracy"].max()),
                            "final_f1": float(last["f1"]),
                            "mean_f1": float(df["f1"].mean()),
                            "final_fpr": float(last["fpr"]),
                            "mean_fpr": float(df["fpr"].mean()),
                            "mean_runtime_sec": float(df["runtime_sec"].mean()),
                            "final_ledger_bytes": int(last["ledger_bytes"]),
                        }
                    )
            else:
                item["completed_rounds"] = 0
            rows.append(item)
    return pd.DataFrame(rows)


def canonicalize(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()
    summary["phase_original"] = summary["phase"]
    summary.loc[summary["phase"] == "cifar_seed_extension", "phase"] = "cifar_supplement"
    summary.loc[summary["phase"] == "history_cosine_baseline", "phase"] = "main"
    summary["malicious_ratio"] = summary["malicious_ratio"].astype(float)
    summary["alpha"] = summary["alpha"].astype(float)
    summary["seed"] = summary["seed"].astype(int)
    summary["ablation"] = summary["ablation"].fillna("full")
    summary["distribution"] = summary["distribution"].fillna(
        summary.apply(lambda r: "iid" if str(r["non_iid"]).lower() == "false" else f"alpha_{r['alpha']}", axis=1)
    )
    return summary


def aggregate(summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "final_accuracy",
        "best_accuracy",
        "final_f1",
        "mean_f1",
        "final_fpr",
        "mean_fpr",
        "mean_runtime_sec",
        "final_ledger_bytes",
    ]
    group_cols = ["phase", "dataset", "distribution", "attack", "defense", "malicious_ratio", "ablation"]
    done = summary[summary["complete"].astype(bool)].copy()
    out = done.groupby(group_cols)[metrics].agg(["mean", "std", "count"]).reset_index()
    out.columns = ["_".join(c).rstrip("_") if isinstance(c, tuple) else c for c in out.columns]
    return out


def statistical_tests(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    done = summary[summary["complete"].astype(bool)].copy()
    scenarios = [
        ("fashion_mnist", "alpha_0.5", "gaussian_noise", 0.2, "trimmed_mean"),
        ("fashion_mnist", "alpha_0.5", "model_scaling", 0.2, "trimmed_mean"),
        ("fashion_mnist", "alpha_0.5", "label_flip", 0.2, "fedavg"),
        ("fashion_mnist", "alpha_0.5", "sign_flip", 0.2, "cosine"),
        ("cifar10", "alpha_0.5", "gaussian_noise", 0.2, "trimmed_mean"),
        ("cifar10", "alpha_0.5", "model_scaling", 0.2, "trimmed_mean"),
    ]
    for dataset, distribution, attack, ratio, baseline in scenarios:
        phase = "cifar_supplement" if dataset == "cifar10" else "main"
        trust = done[
            (done["dataset"] == dataset)
            & (done["phase"] == phase)
            & (done["distribution"] == distribution)
            & (done["attack"] == attack)
            & (done["defense"] == "trustfl_chain")
            & (done["ablation"] == "full")
            & (done["malicious_ratio"].astype(float) == ratio)
        ]
        base = done[
            (done["dataset"] == dataset)
            & (done["phase"] == phase)
            & (done["distribution"] == distribution)
            & (done["attack"] == attack)
            & (done["defense"] == baseline)
            & (done["ablation"] == "full")
            & (done["malicious_ratio"].astype(float) == ratio)
        ]
        merged = trust.merge(base, on="seed", suffixes=("_trustfl", "_baseline"))
        for metric in ["final_accuracy", "mean_f1"]:
            n = len(merged)
            if n >= 2:
                diffs = merged[f"{metric}_trustfl"] - merged[f"{metric}_baseline"]
                if n >= 3 and diffs.std(ddof=1) > 1e-12:
                    stat, pvalue = stats.ttest_rel(merged[f"{metric}_trustfl"], merged[f"{metric}_baseline"])
                    test = "paired_t"
                else:
                    stat, pvalue = float("nan"), float("nan")
                    test = "paired_t_unstable"
                mean_diff = float(diffs.mean())
            else:
                test, stat, pvalue, mean_diff = "insufficient_n", float("nan"), float("nan"), float("nan")
            rows.append(
                {
                    "dataset": dataset,
                    "distribution": distribution,
                    "attack": attack,
                    "baseline": baseline,
                    "metric": metric,
                    "n": n,
                    "mean_diff_trustfl_minus_baseline": mean_diff,
                    "test": test,
                    "statistic": stat,
                    "pvalue": pvalue,
                    "note": "exploratory statistical comparison",
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    FINAL.mkdir(parents=True, exist_ok=True)
    reduced = pd.read_csv(REDUCED / "summary_all_runs.csv")
    final_manifest = FINAL / "final_manifest.csv"
    supplemental = load_manifest_summary(final_manifest) if final_manifest.exists() else pd.DataFrame()
    combined = canonicalize(pd.concat([reduced, supplemental], ignore_index=True, sort=False))
    combined.to_csv(FINAL / "summary_all_runs.csv", index=False)
    missing = combined[~combined["complete"].astype(bool)].copy()
    missing.to_csv(FINAL / "missing_or_failed_runs.csv", index=False)
    agg = aggregate(combined)
    agg.to_csv(FINAL / "aggregate_mean_std.csv", index=False)
    tests = statistical_tests(combined)
    tests.to_csv(FINAL / "statistical_tests.csv", index=False)
    print(f"Wrote {FINAL / 'summary_all_runs.csv'}")
    print(f"Wrote {FINAL / 'aggregate_mean_std.csv'}")
    print(f"Wrote {FINAL / 'statistical_tests.csv'}")
    print(f"Missing/incomplete rows: {len(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
