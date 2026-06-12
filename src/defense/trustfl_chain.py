from __future__ import annotations

from defense.anomaly_detector import AnomalyDetector
from defense.feature_extractor import FeatureExtractor
from defense.reputation import ReputationManager


class TrustFLChainDefense:
    def __init__(self, num_clients: int, detector: str, contamination: float, seed: int, use_reputation: bool = True, use_temporal: bool = True):
        self.extractor = FeatureExtractor()
        self.detector = AnomalyDetector(detector, contamination, seed)
        self.reputation = ReputationManager(num_clients)
        self.use_reputation = use_reputation
        self.use_temporal = use_temporal

    def assess(self, client_ids, updates):
        reps_before = [self.reputation.scores[cid] if self.use_reputation else 1.0 for cid in client_ids]
        features = self.extractor.extract(client_ids, updates, reps_before)
        if not self.use_temporal:
            for row in features:
                row[4] = 0.0
        risks = self.detector.score(features)
        reps_after = self.reputation.update(client_ids, risks) if self.use_reputation else [1.0] * len(client_ids)
        return risks, reps_after, features
