from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "results" / "final_artifacts"


def run_name(row: dict) -> str:
    iid_tag = "noniid" if row["non_iid"] else "iid"
    return (
        f"{row['dataset']}_{row['defense']}_{row['attack']}_{iid_tag}"
        f"_a{row['alpha']}_mr{row['malicious_ratio']}_{row['ablation']}_seed{row['seed']}"
    )


def output_dir(root: Path, row: dict) -> Path:
    return root / "new_runs" / row["phase"] / row["dataset"] / row["attack"]


def expected_csv(root: Path, row: dict) -> Path:
    return output_dir(root, row) / f"{run_name(row)}.csv"


def completed_rounds(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        return max((int(float(r["round"])) for r in rows), default=0)
    except Exception:
        return 0


def add(rows: list[dict], **kwargs) -> None:
    row = {
        "phase": kwargs["phase"],
        "dataset": kwargs["dataset"],
        "seed": kwargs["seed"],
        "rounds": kwargs["rounds"],
        "num_clients": 20,
        "clients_per_round": 10,
        "attack": kwargs["attack"],
        "defense": kwargs["defense"],
        "non_iid": True,
        "alpha": 0.5,
        "malicious_ratio": 0.2,
        "ablation": "full",
        "batch_size": 64,
        "max_train_samples": 10000,
        "max_test_samples": 2000,
    }
    row["run_id"] = run_name(row)
    rows.append(row)


def build_manifest() -> list[dict]:
    rows: list[dict] = []
    for seed in [43, 44]:
        for attack in ["gaussian_noise", "model_scaling"]:
            for defense in ["fedavg", "trimmed_mean", "cosine", "trustfl_chain"]:
                add(rows, phase="cifar_seed_extension", dataset="cifar10", seed=seed, rounds=20, attack=attack, defense=defense)
    for seed in [42, 43, 44]:
        for attack in ["gaussian_noise", "model_scaling"]:
            add(
                rows,
                phase="history_cosine_baseline",
                dataset="fashion_mnist",
                seed=seed,
                rounds=30,
                attack=attack,
                defense="history_cosine",
            )
    return rows


def command(row: dict, root: Path, rounds_override: int | None = None) -> list[str]:
    rounds = rounds_override if rounds_override is not None else row["rounds"]
    return [
        "src/main.py",
        "--dataset",
        row["dataset"],
        "--rounds",
        str(rounds),
        "--num_clients",
        str(row["num_clients"]),
        "--clients_per_round",
        str(row["clients_per_round"]),
        "--max_train_samples",
        str(row["max_train_samples"]),
        "--max_test_samples",
        str(row["max_test_samples"]),
        "--malicious_ratio",
        str(row["malicious_ratio"]),
        "--attack",
        row["attack"],
        "--defense",
        row["defense"],
        "--non_iid",
        str(row["non_iid"]).lower(),
        "--alpha",
        str(row["alpha"]),
        "--seed",
        str(row["seed"]),
        "--batch_size",
        str(row["batch_size"]),
        "--ablation",
        row["ablation"],
        "--output_dir",
        str(output_dir(root, row).relative_to(ROOT)),
    ]


def write_manifest(path: Path, rows: list[dict], root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) + ["output_dir", "expected_csv"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            item = dict(row)
            item["output_dir"] = str(output_dir(root, row).relative_to(ROOT))
            item["expected_csv"] = str(expected_csv(root, row).relative_to(ROOT))
            writer.writerow(item)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run final-artifact supplemental experiments.")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rounds-override", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    root = Path(args.output_root)
    if not root.is_absolute():
        root = ROOT / root
    rows = build_manifest()
    if args.limit is not None:
        rows = rows[: args.limit]
    write_manifest(root / "final_manifest.csv", rows, root)
    print(f"Wrote final manifest with {len(rows)} runs to {root / 'final_manifest.csv'}")
    if args.manifest_only or not args.execute:
        return 0

    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    status = root / "run_status.jsonl"
    for index, row in enumerate(rows, start=1):
        csv_path = expected_csv(root, row)
        expected = args.rounds_override if args.rounds_override is not None else row["rounds"]
        if not args.force and completed_rounds(csv_path) >= expected:
            print(f"[{index}/{len(rows)}] skip completed {row['run_id']}")
            continue
        cmd = [args.python] + command(row, root, args.rounds_override)
        log_base = logs / f"{index:03d}_{row['run_id']}"
        print(f"[{index}/{len(rows)}] running {' '.join(cmd)}")
        start = time.time()
        with log_base.with_suffix(".out.log").open("w", encoding="utf-8") as out, log_base.with_suffix(".err.log").open(
            "w", encoding="utf-8"
        ) as err:
            proc = subprocess.run(cmd, cwd=str(ROOT), stdout=out, stderr=err)
        record = {
            "run_id": row["run_id"],
            "phase": row["phase"],
            "returncode": proc.returncode,
            "seconds": time.time() - start,
            "csv": str(csv_path.relative_to(ROOT)),
            "rounds_completed": completed_rounds(csv_path),
        }
        with status.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        if proc.returncode != 0:
            print(f"FAILED {row['run_id']} returncode={proc.returncode}")
        else:
            print(f"done {row['run_id']} in {record['seconds']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
