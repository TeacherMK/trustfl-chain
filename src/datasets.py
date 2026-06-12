from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


def _make_dataset(name: str, root: str, train: bool, download: bool):
    if name == "fashion_mnist":
        tfm = transforms.Compose([transforms.ToTensor()])
        return datasets.FashionMNIST(root=root, train=train, download=download, transform=tfm)
    if name == "cifar10":
        tfm = transforms.Compose([transforms.ToTensor()])
        return datasets.CIFAR10(root=root, train=train, download=download, transform=tfm)
    if name == "synthetic":
        size = 1200 if train else 300
        image_size = (1, 28, 28)
        return datasets.FakeData(size=size, image_size=image_size, num_classes=10, transform=transforms.ToTensor())
    raise ValueError(f"Unsupported dataset: {name}")


def load_datasets(name: str, data_dir: str, download: bool = True, allow_synthetic_fallback: bool = False):
    try:
        train = _make_dataset(name, data_dir, True, download)
        test = _make_dataset(name, data_dir, False, download)
    except Exception as exc:
        if not allow_synthetic_fallback:
            raise RuntimeError(
                f"Failed to load dataset '{name}'. Use --dataset synthetic for an offline smoke test, "
                "or pass --allow_synthetic_fallback to explicitly use FakeData after a dataset failure."
            ) from exc
        print(f"[datasets] Falling back to torchvision FakeData because {name} failed: {exc}")
        train = _make_dataset("synthetic", data_dir, True, False)
        test = _make_dataset("synthetic", data_dir, False, False)
    val_size = min(512, max(100, len(train) // 20))
    train_size = len(train) - val_size
    train, val = random_split(train, [train_size, val_size], generator=torch.Generator().manual_seed(123))
    return train, val, test


def partition_dataset(dataset, num_clients: int, iid: bool, alpha: float, seed: int):
    rng = np.random.default_rng(seed)
    if iid:
        indices = rng.permutation(len(dataset))
        return [indices[i::num_clients].tolist() for i in range(num_clients)]

    labels = []
    base = dataset.dataset if isinstance(dataset, Subset) else dataset
    subset_indices = dataset.indices if isinstance(dataset, Subset) else range(len(dataset))
    for idx in subset_indices:
        labels.append(int(base[idx][1]))
    by_class = defaultdict(list)
    for local_idx, y in enumerate(labels):
        by_class[y].append(local_idx)

    client_indices = [[] for _ in range(num_clients)]
    for cls_indices in by_class.values():
        rng.shuffle(cls_indices)
        proportions = rng.dirichlet([alpha] * num_clients)
        splits = (np.cumsum(proportions)[:-1] * len(cls_indices)).astype(int)
        for client_id, part in enumerate(np.split(np.array(cls_indices), splits)):
            client_indices[client_id].extend(part.tolist())
    return client_indices


def make_loaders(train, val, test, client_indices, batch_size: int):
    client_loaders = [DataLoader(Subset(train, idx), batch_size=batch_size, shuffle=bool(idx)) for idx in client_indices]
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    return client_loaders, val_loader, test_loader
