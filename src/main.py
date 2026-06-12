from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parent))

from attacks.gaussian_noise import gaussian_noise
from attacks.model_scaling import model_scaling
from attacks.sign_flip import sign_flip
from datasets import load_datasets, make_loaders, partition_dataset
from defense.history_cosine import HistoryCosineDefense
from defense.trustfl_chain import TrustFLChainDefense
from federated.client import Client
from federated.server import aggregate
from ledger.blockchain import BlockchainLedger, hash_state
from models import build_model
from utils.metrics import classification_accuracy, detection_metrics
from utils.seed import set_seed


def str2bool(value):
    if isinstance(value, bool):
        return value
    return value.lower() in {"true", "1", "yes", "y"}


def parse_args():
    p = argparse.ArgumentParser(description="Single-machine TrustFL-Chain simulator")
    p.add_argument("--dataset", default="fashion_mnist", choices=["fashion_mnist", "cifar10", "synthetic"])
    p.add_argument("--num_clients", type=int, default=20)
    p.add_argument("--clients_per_round", type=int, default=10)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--local_epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--malicious_ratio", type=float, default=0.2)
    p.add_argument("--attack", default="label_flip", choices=["label_flip", "sign_flip", "gaussian_noise", "model_scaling", "none"])
    p.add_argument(
        "--defense",
        default="trustfl_chain",
        choices=["fedavg", "median", "trimmed_mean", "krum", "cosine", "trustfl_chain", "history_cosine"],
    )
    p.add_argument("--detector", default="isolation_forest", choices=["isolation_forest", "lof", "ocsvm", "cosine_only"])
    p.add_argument("--non_iid", type=str2bool, default=True)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_train_samples", type=int, default=2000)
    p.add_argument("--max_test_samples", type=int, default=1000)
    p.add_argument("--output_dir", default="results")
    p.add_argument("--no_download", action="store_true")
    p.add_argument("--allow_synthetic_fallback", action="store_true")
    p.add_argument("--ablation", default="full", choices=["full", "no_reputation", "no_temporal", "cosine_only", "no_ledger"])
    p.add_argument("--trustfl_aggregation", default="reputation_weighted", choices=["reputation_weighted", "direction_filter"])
    p.add_argument("--trustfl_keep_ratio", type=float, default=0.6)
    return p.parse_args()


def subset_if_needed(dataset, max_samples):
    if max_samples and len(dataset) > max_samples:
        return torch.utils.data.Subset(dataset, list(range(max_samples)))
    return dataset


def apply_post_attack(update, attack):
    if attack == "sign_flip":
        return sign_flip(update, scale=1.0)
    if attack == "gaussian_noise":
        return gaussian_noise(update, std=0.5)
    if attack == "model_scaling":
        return model_scaling(update, scale=5.0)
    return update


def main():
    args = parse_args()
    set_seed(args.seed)
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    iid_tag = "noniid" if args.non_iid else "iid"
    run_name = (
        f"{args.dataset}_{args.defense}_{args.attack}_{iid_tag}"
        f"_a{args.alpha}_mr{args.malicious_ratio}_{args.ablation}_seed{args.seed}"
    )
    csv_path = output_dir / f"{run_name}.csv"
    rep_path = output_dir / f"{run_name}_reputation.csv"
    ledger_path = output_dir / f"{run_name}_ledger.jsonl"
    for stale_path in (csv_path, rep_path, ledger_path):
        if stale_path.exists():
            stale_path.unlink()
    config_path = output_dir / f"{run_name}_config.json"
    config_path.write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, val, test = load_datasets(
        args.dataset,
        str(project_root / "data"),
        download=not args.no_download,
        allow_synthetic_fallback=args.allow_synthetic_fallback,
    )
    train = subset_if_needed(train, args.max_train_samples)
    test = subset_if_needed(test, args.max_test_samples)
    client_indices = partition_dataset(train, args.num_clients, iid=not args.non_iid, alpha=args.alpha, seed=args.seed)
    client_loaders, val_loader, test_loader = make_loaders(train, val, test, client_indices, args.batch_size)

    malicious_count = int(args.num_clients * args.malicious_ratio)
    rng = random.Random(args.seed)
    malicious_clients = set(rng.sample(range(args.num_clients), malicious_count))
    clients = [Client(i, client_loaders[i], i in malicious_clients) for i in range(args.num_clients)]

    global_model = build_model(args.dataset if args.dataset != "synthetic" else "fashion_mnist")
    global_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}
    defense = None
    if args.defense == "trustfl_chain":
        detector = "cosine_only" if args.ablation == "cosine_only" else args.detector
        defense = TrustFLChainDefense(
            args.num_clients,
            detector=detector,
            contamination=max(args.malicious_ratio, 0.05),
            seed=args.seed,
            use_reputation=args.ablation != "no_reputation",
            use_temporal=args.ablation != "no_temporal",
        )
    elif args.defense == "history_cosine":
        defense = HistoryCosineDefense(args.num_clients)
    ledger = None if args.ablation == "no_ledger" else BlockchainLedger(str(ledger_path))
    rows = []
    rep_rows = []

    for round_id in range(1, args.rounds + 1):
        start = time.time()
        selected = rng.sample(range(args.num_clients), min(args.clients_per_round, args.num_clients))
        updates = []
        y_true = []
        for cid in selected:
            local_model = build_model(args.dataset if args.dataset != "synthetic" else "fashion_mnist")
            train_attack = args.attack if clients[cid].malicious else "none"
            update = clients[cid].train(local_model, global_state, device, args.local_epochs, args.lr, train_attack)
            if clients[cid].malicious and args.attack != "label_flip":
                update = apply_post_attack(update, args.attack)
            updates.append(update)
            y_true.append(int(clients[cid].malicious))

        if args.defense == "trustfl_chain":
            risks, reputations, _ = defense.assess(selected, updates)
            y_pred = [int(r >= 0.5) for r in risks]
            agg_weights = reputations
        elif args.defense == "history_cosine":
            risks, weights = defense.assess(selected, updates)
            y_pred = [int(r >= 0.5) for r in risks]
            agg_weights = weights
        else:
            risks = [0.0] * len(selected)
            y_pred = [0] * len(selected)
            agg_weights = None
        global_state = aggregate(
            global_state,
            updates,
            args.defense,
            agg_weights,
            num_malicious=max(1, malicious_count),
            trustfl_aggregation=args.trustfl_aggregation,
            trustfl_keep_ratio=args.trustfl_keep_ratio,
        )
        global_model.load_state_dict(global_state)
        acc = classification_accuracy(global_model.to(device), test_loader, device)
        dmetrics = detection_metrics(y_true, y_pred)
        runtime = time.time() - start
        agg_hash = hash_state(global_state)
        if ledger is not None:
            ledger.append_round(round_id, selected, updates, risks, agg_weights or [1.0] * len(selected), agg_hash)
        ledger_bytes = ledger_path.stat().st_size if ledger_path.exists() else 0
        row = {
            "round": round_id,
            "dataset": args.dataset,
            "defense": args.defense,
            "attack": args.attack,
            "malicious_ratio": args.malicious_ratio,
            "non_iid": args.non_iid,
            "alpha": args.alpha,
            "accuracy": acc,
            "runtime_sec": runtime,
            "ledger_bytes": ledger_bytes,
            **dmetrics,
        }
        rows.append(row)
        if args.defense == "trustfl_chain":
            for cid in range(args.num_clients):
                rep_rows.append({"round": round_id, "client_id": cid, "reputation": defense.reputation.scores[cid], "malicious": int(cid in malicious_clients)})
        print(json.dumps(row))

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    if rep_rows:
        with rep_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rep_rows[0].keys()))
            writer.writeheader()
            writer.writerows(rep_rows)


if __name__ == "__main__":
    main()
