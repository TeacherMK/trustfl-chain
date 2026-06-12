from __future__ import annotations

import torch

from federated.aggregation import flatten_update


class HistoryCosineDefense:
    """FoolsGold-like history cosine weighting without a trusted root dataset."""

    def __init__(self, num_clients: int, momentum: float = 0.8):
        self.histories = [None] * num_clients
        self.momentum = momentum

    def assess(self, client_ids, updates):
        flats = [flatten_update(update) for update in updates]
        for cid, flat in zip(client_ids, flats):
            if self.histories[cid] is None:
                self.histories[cid] = flat.detach().clone()
            else:
                self.histories[cid] = self.momentum * self.histories[cid] + (1.0 - self.momentum) * flat

        selected_histories = torch.stack([self.histories[cid] for cid in client_ids])
        if len(client_ids) == 1:
            return [0.0], [1.0]

        sims = torch.nn.functional.cosine_similarity(
            selected_histories.unsqueeze(1),
            selected_histories.unsqueeze(0),
            dim=2,
        )
        sims.fill_diagonal_(-1.0)
        max_peer_sim = sims.max(dim=1).values.clamp(min=0.0, max=1.0)
        weights = (1.0 - max_peer_sim).clamp(min=0.05, max=1.0)
        weights = weights / (weights.mean() + 1e-12)
        risks = max_peer_sim.tolist()
        return risks, weights.tolist()
