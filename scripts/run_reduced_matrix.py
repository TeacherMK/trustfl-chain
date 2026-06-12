from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "reduced_matrix"
SEEDS = [42, 43, 44]
ATTACKS = ["label_flip", "sign_flip", "gaussian_noise", "model_scaling"]
MAIN_DEFENSES = ["fedavg", "trimmed_mean", "cosine", "trustfl_chain"]
RATIO_DEFENSES = ["trustfl_chain", "cosine", "trimmed_mean"]
RATIOS = [0.1, 0.2, 0.3, 0.4]
ABLATIONS = ["full", "no_reputation", "no_temporal", "cosine_only", "no_ledger"]


def run_name(row: dict) -> str:
    iid_tag = "noniid" if row["non_iid"] else "iid"
    return (
        f"{row['dataset']}_{row['defense']}_{row['attack']}_{iid_tag}"
        f"_a{row['alpha']}_mr{row['malicious_ratio']}_{row['ablation']}_seed{row['seed']}"
    )


def output_dir(output_root: Path, row: dict) -> Path:
    if row["phase"] == "main":
        dist = "iid" if not row["non_iid"] else f"alpha{row['alpha']}"
        return output_root / "main" / row["dataset"] / dist / row["attack"]
    if row["phase"] == "ratio":
        return output_root / "ratio" / row["dataset"] / f"mr{row['malicious_ratio']}"
    if row["phase"] == "ablation":
        return output_root / "ablation" / row["dataset"] / row["ablation"]
    if row["phase"] == "cifar_supplement":
        return output_root / "cifar_supplement" / row["attack"]
    raise ValueError(row["phase"])


def expected_csv(output_root: Path, row: dict) -> Path:
    return output_dir(output_root, row) / f"{run_name(row)}.csv"


def completed_rounds(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return 0
        return max(int(float(r["round"])) for r in rows)
    except Exception:
        return 0


def add_run(rows: list[dict], **kwargs) -> None:
    row = {
        "phase": kwargs["phase"],
        "dataset": kwargs["dataset"],
        "seed": kwargs["seed"],
        "rounds": kwargs["rounds"],
        "num_clients": 20,
        "clients_per_round": 10,
        "attack": kwargs["attack"],
        "defense": kwargs["defense"],
        "non_iid": kwargs["non_iid"],
        "alpha": kwargs["alpha"],
        "malicious_ratio": kwargs["malicious_ratio"],
        "ablation": kwargs.get("ablation", "full"),
        "batch_size": 64,
        "max_train_samples": kwargs["max_train_samples"],
        "max_test_samples": kwargs["max_test_samples"],
    }
    row["run_id"] = run_name(row)
    rows.append(row)


def build_manifest(max_train_samples: int, max_test_samples: int, cifar_train_samples: int, cifar_test_samples: int) -> list[dict]:
    rows: list[dict] = []
    distributions = [
        {"non_iid": False, "alpha": 0.5},
        {"non_iid": True, "alpha": 0.5},
        {"non_iid": True, "alpha": 0.1},
    ]

    for seed in SEEDS:
        for dist in distributions:
            for attack in ATTACKS:
                for defense in MAIN_DEFENSES:
                    add_run(
                        rows,
                        phase="main",
                        dataset="fashion_mnist",
                        seed=seed,
                        rounds=30,
                        attack=attack,
                        defense=defense,
                        malicious_ratio=0.2,
                        max_train_samples=max_train_samples,
                        max_test_samples=max_test_samples,
                        **dist,
                    )

    for seed in SEEDS:
        for ratio in RATIOS:
            for defense in RATIO_DEFENSES:
                if ratio == 0.2:
                    # Covered by the main matrix for model_scaling / alpha=0.5.
                    continue
                add_run(
                    rows,
                    phase="ratio",
                    dataset="fashion_mnist",
                    seed=seed,
                    rounds=30,
                    attack="model_scaling",
                    defense=defense,
                    non_iid=True,
                    alpha=0.5,
                    malicious_ratio=ratio,
                    max_train_samples=max_train_samples,
                    max_test_samples=max_test_samples,
                )

    for seed in SEEDS:
        for ablation in ABLATIONS:
            if ablation == "full":
                # Covered by the main matrix for TrustFL-Chain / model_scaling / alpha=0.5.
                continue
            add_run(
                rows,
                phase="ablation",
                dataset="fashion_mnist",
                seed=seed,
                rounds=30,
                attack="model_scaling",
                defense="trustfl_chain",
                non_iid=True,
                alpha=0.5,
                malicious_ratio=0.2,
                ablation=ablation,
                max_train_samples=max_train_samples,
                max_test_samples=max_test_samples,
            )

    for attack in ["gaussian_noise", "model_scaling"]:
        for defense in MAIN_DEFENSES:
            add_run(
                rows,
                phase="cifar_supplement",
                dataset="cifar10",
                seed=42,
                rounds=20,
                attack=attack,
                defense=defense,
                non_iid=True,
                alpha=0.5,
                malicious_ratio=0.2,
                max_train_samples=cifar_train_samples,
                max_test_samples=cifar_test_samples,
            )
    return rows


def write_manifest(path: Path, rows: list[dict], output_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) + ["output_dir", "expected_csv"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            enriched = dict(row)
            enriched["output_dir"] = str(output_dir(output_root, row).relative_to(ROOT))
            enriched["expected_csv"] = str(expected_csv(output_root, row).relative_to(ROOT))
            writer.writerow(enriched)


def command_for(row: dict, output_root: Path, rounds_override: int | None = None) -> list[str]:
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
        str(output_dir(output_root, row).relative_to(ROOT)),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the reduced TrustFL-Chain study matrix with resume support.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only-phase", choices=["main", "ratio", "ablation", "cifar_supplement"], default=None)
    parser.add_argument("--rounds-override", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max-train-samples", type=int, default=10000)
    parser.add_argument("--max-test-samples", type=int, default=2000)
    parser.add_argument("--cifar-train-samples", type=int, default=10000)
    parser.add_argument("--cifar-test-samples", type=int, default=2000)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    rows = build_manifest(args.max_train_samples, args.max_test_samples, args.cifar_train_samples, args.cifar_test_samples)
    if args.only_phase:
        rows = [r for r in rows if r["phase"] == args.only_phase]
    if args.limit is not None:
        rows = rows[: args.limit]

    manifest_path = output_root / "manifest.csv"
    write_manifest(manifest_path, rows, output_root)
    print(f"Wrote manifest with {len(rows)} runs to {manifest_path}")

    if args.manifest_only or not args.execute:
        return 0

    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "run_status.jsonl"
    expected_rounds = args.rounds_override if args.rounds_override is not None else None

    for index, row in enumerate(rows, start=1):
        csv_path = expected_csv(output_root, row)
        needed_rounds = expected_rounds if expected_rounds is not None else row["rounds"]
        if not args.force and completed_rounds(csv_path) >= needed_rounds:
            print(f"[{index}/{len(rows)}] skip completed {row['run_id']}")
            continue

        cmd = [args.python] + command_for(row, output_root, args.rounds_override)
        log_base = logs_dir / f"{row['phase']}_{index:04d}_{row['run_id']}"
        print(f"[{index}/{len(rows)}] running {' '.join(cmd)}")
        start = time.time()
        with (log_base.with_suffix(".out.log")).open("w", encoding="utf-8") as stdout, (
            log_base.with_suffix(".err.log")
        ).open("w", encoding="utf-8") as stderr:
            proc = subprocess.run(cmd, cwd=str(ROOT), stdout=stdout, stderr=stderr)
        record = {
            "run_id": row["run_id"],
            "phase": row["phase"],
            "returncode": proc.returncode,
            "seconds": time.time() - start,
            "csv": str(csv_path.relative_to(ROOT)),
            "rounds_completed": completed_rounds(csv_path),
        }
        with status_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        if proc.returncode != 0:
            print(f"FAILED {row['run_id']} returncode={proc.returncode}; see {log_base.with_suffix('.err.log')}")
        else:
            print(f"done {row['run_id']} in {record['seconds']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
