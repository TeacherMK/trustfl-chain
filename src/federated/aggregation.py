from __future__ import annotations

import torch


def clone_state(state):
    return {k: v.detach().cpu().clone() for k, v in state.items()}


def flatten_update(update):
    return torch.cat([v.detach().float().cpu().reshape(-1) for v in update.values()])


def state_subtract(a, b):
    return {k: a[k].detach().cpu() - b[k].detach().cpu() for k in a}


def apply_update(state, update, lr: float = 1.0):
    return {k: state[k].detach().cpu() + lr * update[k].detach().cpu() for k in state}


def weighted_average(updates, weights=None):
    if weights is None:
        weights = [1.0] * len(updates)
    total = sum(weights) or 1.0
    out = {}
    for key in updates[0]:
        out[key] = sum(update[key] * (w / total) for update, w in zip(updates, weights))
    return out


def coordinate_median(updates):
    return {k: torch.stack([u[k] for u in updates], dim=0).median(dim=0).values for k in updates[0]}


def trimmed_mean(updates, trim_ratio: float = 0.2):
    n = len(updates)
    trim = int(n * trim_ratio)
    out = {}
    for k in updates[0]:
        stacked = torch.stack([u[k] for u in updates], dim=0)
        sorted_vals, _ = torch.sort(stacked, dim=0)
        kept = sorted_vals[trim : n - trim] if n - 2 * trim > 0 else sorted_vals
        out[k] = kept.mean(dim=0)
    return out


def krum(updates, num_malicious: int = 1):
    flats = torch.stack([flatten_update(u) for u in updates])
    n = len(updates)
    neighbor_count = max(1, n - num_malicious - 2)
    scores = []
    for i in range(n):
        dists = torch.norm(flats - flats[i], dim=1) ** 2
        scores.append(torch.sort(dists)[0][1 : neighbor_count + 1].sum().item())
    return updates[int(torch.tensor(scores).argmin().item())]


def cosine_filter(updates, reputations=None, keep_ratio: float = 0.7):
    flats = torch.stack([flatten_update(u) for u in updates])
    mean = flats.mean(dim=0)
    sims = torch.nn.functional.cosine_similarity(flats, mean.unsqueeze(0), dim=1)
    keep = max(1, int(len(updates) * keep_ratio))
    indices = torch.topk(sims, k=keep).indices.tolist()
    kept = [updates[i] for i in indices]
    weights = [reputations[i] for i in indices] if reputations is not None else None
    return weighted_average(kept, weights)
