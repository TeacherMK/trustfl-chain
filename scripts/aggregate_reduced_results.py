from __future__ import annotations

import csv
import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def completed_rounds(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        df = pd.read_csv(path)
        return int(df["round"].max()) if not df.empty else 0
    except Exception:
        return 0


def distribution(row) -> str:
    return "iid" if str(row["non_iid"]).lower() == "false" else f"alpha_{row['alpha']}"


def summarize_run(manifest_row: dict) -> dict:
    csv_path = ROOT / manifest_row["expected_csv"]
    expected_rounds = int(float(manifest_row["rounds"]))
    done = completed_rounds(csv_path)
    base = dict(manifest_row)
    base["distribution"] = distribution(manifest_row)
    base["completed_rounds"] = done
    base["complete"] = done >= expected_rounds
    base["csv_exists"] = csv_path.exists()
    base["config_exists"] = csv_path.with_name(csv_path.stem + "_config.json").exists()
    base["ledger_exists"] = (csv_path.with_name(csv_path.stem + "_ledger.jsonl").exists()) or manifest_row["ablation"] == "no_ledger"
    if done == 0:
        return base
    df = pd.read_csv(csv_path)
    last = df.sort_values("round").iloc[-1]
    base.update(
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
    return base


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate reduced TrustFL-Chain matrix results.")
    parser.add_argument("--output-root", default=str(ROOT / "results" / "reduced_matrix"))
    parser.add_argument("--expected-rounds", type=int, default=None)
    args = parser.parse_args()
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    manifest = output_root / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}. Run scripts/run_reduced_matrix.py first.")
    with manifest.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    summary_rows = []
    for row in rows:
        item = summarize_run(row)
        if args.expected_rounds is not None:
            item["complete"] = item["completed_rounds"] >= args.expected_rounds
            item["expected_rounds_for_aggregation"] = args.expected_rounds
        summary_rows.append(item)
    summary = pd.DataFrame(summary_rows)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "summary_all_runs.csv"
    missing_path = output_root / "missing_or_failed_runs.csv"
    aggregate_path = output_root / "aggregate_mean_std.csv"
    summary.to_csv(summary_path, index=False)
    missing = summary[~summary["complete"].astype(bool)].copy()
    missing.to_csv(missing_path, index=False)

    metric_cols = [
        "final_accuracy",
        "best_accuracy",
        "final_f1",
        "mean_f1",
        "final_fpr",
        "mean_fpr",
        "mean_runtime_sec",
        "final_ledger_bytes",
    ]
    completed = summary[summary["complete"].astype(bool)].copy()
    if completed.empty:
        aggregate = pd.DataFrame()
    else:
        group_cols = ["phase", "dataset", "distribution", "attack", "defense", "malicious_ratio", "ablation"]
        aggregate = completed.groupby(group_cols)[metric_cols].agg(["mean", "std", "count"]).reset_index()
        aggregate.columns = ["_".join(c).rstrip("_") if isinstance(c, tuple) else c for c in aggregate.columns]
    aggregate.to_csv(aggregate_path, index=False)
    print(f"Wrote {summary_path}")
    print(f"Wrote {aggregate_path}")
    print(f"Wrote {missing_path} ({len(missing)} incomplete runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
