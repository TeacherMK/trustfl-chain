from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


class AnomalyDetector:
    def __init__(self, method: str = "isolation_forest", contamination: float = 0.2, seed: int = 42):
        self.method = method
        self.contamination = max(0.01, min(contamination, 0.49))
        self.seed = seed

    def score(self, features):
        x = StandardScaler().fit_transform(np.asarray(features, dtype=float))
        n = len(x)
        if n < 4:
            risk = np.zeros(n)
        elif self.method == "lof":
            nn = max(2, min(n - 1, 10))
            pred = LocalOutlierFactor(n_neighbors=nn, contamination=self.contamination).fit_predict(x)
            risk = (pred == -1).astype(float)
        elif self.method == "ocsvm":
            pred = OneClassSVM(nu=self.contamination, gamma="scale").fit_predict(x)
            risk = (pred == -1).astype(float)
        elif self.method == "cosine_only":
            cosine = np.asarray([row[1] for row in features])
            threshold = np.quantile(cosine, self.contamination)
            risk = (cosine <= threshold).astype(float)
        else:
            model = IsolationForest(contamination=self.contamination, random_state=self.seed)
            pred = model.fit_predict(x)
            raw = -model.score_samples(x)
            raw = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
            risk = np.maximum(raw, (pred == -1).astype(float))
        return risk.tolist()
