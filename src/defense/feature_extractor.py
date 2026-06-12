from __future__ import annotations

import torch

from federated.aggregation import flatten_update


class FeatureExtractor:
    def __init__(self):
        self.history = {}

    def extract(self, client_ids, updates, reputations, val_loss_deltas=None):
        flats = torch.stack([flatten_update(u) for u in updates])
        mean = flats.mean(dim=0)
        median = flats.median(dim=0).values
        val_loss_deltas = val_loss_deltas or [0.0] * len(updates)
        rows = []
        for i, flat in enumerate(flats):
            cid = client_ids[i]
            hist = self.history.get(cid)
            temporal = torch.norm(flat - hist).item() if hist is not None else 0.0
            self.history[cid] = flat.detach().clone()
            rows.append(
                [
                    torch.norm(flat).item(),
                    torch.nn.functional.cosine_similarity(flat, mean, dim=0).item(),
                    torch.norm(flat - median).item(),
                    float(val_loss_deltas[i]),
                    temporal,
                    float(reputations[i]),
                ]
            )
        return rows
