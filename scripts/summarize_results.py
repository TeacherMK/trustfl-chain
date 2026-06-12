from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def run_label(path: Path) -> str:
    rel = path.parent.relative_to(RESULTS)
    return str(rel) if str(rel) != "." else "root"


rows = []
for path in RESULTS.rglob("*.csv"):
    if path.name.endswith("_reputation.csv"):
        continue
    df = pd.read_csv(path)
    if df.empty:
        continue
    last = df.sort_values("round").iloc[-1]
    rows.append(
        {
            "group": run_label(path),
            "file": path.name,
            "dataset": last["dataset"],
            "defense": last["defense"],
            "attack": last["attack"],
            "malicious_ratio": last["malicious_ratio"],
            "non_iid": last["non_iid"],
            "alpha": last["alpha"],
            "final_round": int(last["round"]),
            "final_accuracy": float(last["accuracy"]),
            "best_accuracy": float(df["accuracy"].max()),
            "final_f1": float(last["f1"]),
            "mean_f1": float(df["f1"].mean()),
            "final_fpr": float(last["fpr"]),
            "mean_runtime_sec": float(df["runtime_sec"].mean()),
            "final_ledger_bytes": int(last["ledger_bytes"]),
        }
    )

summary = pd.DataFrame(rows)
out = RESULTS / "summary.csv"
summary.sort_values(["group", "dataset", "attack", "defense", "file"]).to_csv(out, index=False)
print(summary.sort_values(["group", "dataset", "attack", "defense", "file"]).to_string(index=False))
print(f"Wrote {out}")
