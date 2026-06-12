from __future__ import annotations

import torch
import torch.nn.functional as F

from attacks import label_flip_targets
from federated.aggregation import clone_state, state_subtract


class Client:
    def __init__(self, client_id: int, loader, malicious: bool = False):
        self.client_id = client_id
        self.loader = loader
        self.malicious = malicious

    def train(self, model, global_state, device, epochs: int, lr: float, attack: str, num_classes: int = 10):
        model.load_state_dict(global_state)
        model.to(device)
        model.train()
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)
        for _ in range(epochs):
            for x, y in self.loader:
                x, y = x.to(device), y.to(device)
                if self.malicious and attack == "label_flip":
                    y = label_flip_targets(y, num_classes)
                opt.zero_grad()
                loss = F.cross_entropy(model(x), y)
                loss.backward()
                opt.step()
        local_state = clone_state(model.state_dict())
        return state_subtract(local_state, global_state)
