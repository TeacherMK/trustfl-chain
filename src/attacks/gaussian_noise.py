import torch


def gaussian_noise(update, std: float = 0.5):
    return {k: v + torch.randn_like(v) * std for k, v in update.items()}
