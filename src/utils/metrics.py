from __future__ import annotations

import torch


def classification_accuracy(model, loader, device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / max(total, 1)


def detection_metrics(y_true, y_pred) -> dict:
    tp = sum(int(a == 1 and b == 1) for a, b in zip(y_true, y_pred))
    fp = sum(int(a == 0 and b == 1) for a, b in zip(y_true, y_pred))
    fn = sum(int(a == 1 and b == 0) for a, b in zip(y_true, y_pred))
    tn = sum(int(a == 0 and b == 0) for a, b in zip(y_true, y_pred))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    fpr = fp / max(fp + tn, 1)
    return {"precision": precision, "recall": recall, "f1": f1, "fpr": fpr}
