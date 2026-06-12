from __future__ import annotations

from federated.aggregation import apply_update, coordinate_median, trimmed_mean, weighted_average, krum, cosine_filter


def aggregate(
    global_state,
    updates,
    defense: str,
    reputations=None,
    num_malicious: int = 1,
    trustfl_aggregation: str = "reputation_weighted",
    trustfl_keep_ratio: float = 0.6,
):
    if defense in {"fedavg", "none"}:
        agg = weighted_average(updates)
    elif defense == "median":
        agg = coordinate_median(updates)
    elif defense == "trimmed_mean":
        agg = trimmed_mean(updates)
    elif defense == "krum":
        agg = krum(updates, num_malicious=num_malicious)
    elif defense == "cosine":
        agg = cosine_filter(updates, reputations=None)
    elif defense == "trustfl_chain":
        if trustfl_aggregation == "direction_filter":
            agg = cosine_filter(updates, reputations=None, keep_ratio=trustfl_keep_ratio)
        else:
            agg = weighted_average(updates, reputations)
    elif defense == "history_cosine":
        agg = weighted_average(updates, reputations)
    else:
        raise ValueError(f"Unknown defense: {defense}")
    return apply_update(global_state, agg)
