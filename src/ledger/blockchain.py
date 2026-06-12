from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch

from federated.aggregation import flatten_update


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_update(update) -> str:
    flat = flatten_update(update).numpy().tobytes()
    return _hash_bytes(flat)


def hash_state(state) -> str:
    payload = b"".join(v.detach().cpu().numpy().tobytes() for _, v in sorted(state.items()))
    return _hash_bytes(payload)


class BlockchainLedger:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.previous_hash = "0" * 64
        if self.path.exists():
            lines = self.path.read_text(encoding="utf-8").splitlines()
            if lines:
                self.previous_hash = json.loads(lines[-1])["block_hash"]

    def append_round(self, round_id, client_ids, updates, risks, reputations, aggregation_hash):
        records = []
        for cid, update, risk, rep in zip(client_ids, updates, risks, reputations):
            records.append({"client_id": cid, "update_hash": hash_update(update), "risk": float(risk), "reputation": float(rep)})
        block = {
            "round": round_id,
            "timestamp": time.time(),
            "previous_hash": self.previous_hash,
            "aggregation_hash": aggregation_hash,
            "client_records": records,
        }
        block["block_hash"] = _hash_bytes(json.dumps(block, sort_keys=True).encode("utf-8"))
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(block, ensure_ascii=False) + "\n")
        self.previous_hash = block["block_hash"]
        return block
