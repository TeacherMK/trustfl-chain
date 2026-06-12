from __future__ import annotations


class ReputationManager:
    def __init__(self, num_clients: int, init: float = 1.0, floor: float = 0.05, penalty: float = 0.35, recovery: float = 0.05):
        self.scores = [init] * num_clients
        self.floor = floor
        self.penalty = penalty
        self.recovery = recovery

    def update(self, client_ids, risks):
        for cid, risk in zip(client_ids, risks):
            if risk >= 0.5:
                self.scores[cid] = max(self.floor, self.scores[cid] * (1.0 - self.penalty * risk))
            else:
                self.scores[cid] = min(1.0, self.scores[cid] + self.recovery * (1.0 - self.scores[cid]))
        return [self.scores[cid] for cid in client_ids]
