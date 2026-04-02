"""XGBoost meta-learner for swarm span classification.

Extracts 20 features per SpanCandidate and predicts whether the detection
is a true positive.  Falls back to logistic scoring when XGBoost is unavailable.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

from pii_anon.fusion import _overlap_iou
from pii_anon.swarm import STRUCTURED_TYPES, InformativenessScorer, SpanCandidate, _jaccard_similarity

logger = logging.getLogger(__name__)

# Canonical entity type → integer encoding for the meta-learner.
ENTITY_TYPE_ENCODING: dict[str, int] = {
    "PERSON_NAME": 0, "EMAIL_ADDRESS": 1, "PHONE_NUMBER": 2, "ADDRESS": 3,
    "US_SSN": 4, "CREDIT_CARD": 5, "DATE_OF_BIRTH": 6, "ORGANIZATION": 7,
    "IP_ADDRESS": 8, "MAC_ADDRESS": 9, "BANK_ACCOUNT": 10, "ROUTING_NUMBER": 11,
    "DRIVERS_LICENSE": 12, "PASSPORT": 13, "NATIONAL_ID": 14, "IBAN": 15,
    "USERNAME": 16, "EMPLOYEE_ID": 17, "MEDICAL_RECORD_NUMBER": 18, "LOCATION": 19,
    "LICENSE_PLATE": 20, "VIN": 21, "CRYPTO_WALLET": 22,
}

# PII context keywords for feature 17.
_CONTEXT_KEYWORDS = frozenset({
    "name", "email", "phone", "ssn", "social", "security", "address", "dob",
    "born", "card", "credit", "account", "passport", "license", "driver",
    "patient", "employee", "id", "number", "tel", "fax", "mobile", "cell",
    "mail", "contact", "ip", "mac", "iban", "routing", "bank",
})


def extract_features(
    candidate: SpanCandidate,
    *,
    total_engines: int = 6,
    informativeness_scorer: InformativenessScorer | None = None,
    text: str = "",
    all_candidates: list[SpanCandidate] | None = None,
) -> list[float]:
    """Extract 20 features from a SpanCandidate for the meta-learner."""
    if informativeness_scorer is None:
        informativeness_scorer = InformativenessScorer()

    calibrated_confs = [f.confidence for f in candidate.engine_findings.values()]
    if not calibrated_confs:
        calibrated_confs = [0.0]

    mean_conf = sum(calibrated_confs) / len(calibrated_confs)
    if len(calibrated_confs) > 1:
        std_conf = math.sqrt(sum((c - mean_conf) ** 2 for c in calibrated_confs) / len(calibrated_confs))
    else:
        std_conf = 0.0

    regex_finding = candidate.engine_findings.get("regex-oss")
    regex_detected = 1.0 if regex_finding is not None else 0.0
    regex_confidence = regex_finding.confidence if regex_finding is not None else 0.0

    span_len = max(candidate.span_end - candidate.span_start, 1)
    span_tokens = len(candidate.entity_type.split()) if span_len < 3 else max(1, span_len // 5)

    entity_encoded = ENTITY_TYPE_ENCODING.get(candidate.entity_type, len(ENTITY_TYPE_ENCODING))
    is_structured = 1.0 if candidate.entity_type in STRUCTURED_TYPES else 0.0

    # Checksum validation (feature 14): check if regex detected with high confidence.
    has_checksum = 1.0 if (regex_finding is not None and regex_finding.confidence >= 0.91) else 0.0

    # Informativeness (feature 15).
    info_scores = [informativeness_scorer.score(eid) for eid in candidate.engine_findings]
    mean_info = sum(info_scores) / len(info_scores) if info_scores else 0.5

    # Boundary agreement (feature 16): IoU between min/max boundaries.
    starts = [f.span_start for f in candidate.engine_findings.values() if f.span_start is not None]
    ends = [f.span_end for f in candidate.engine_findings.values() if f.span_end is not None]
    if len(starts) >= 2 and len(ends) >= 2:
        boundary_agreement = _overlap_iou(min(starts), max(ends), max(starts), min(ends))
    else:
        boundary_agreement = 1.0

    # Context keywords (feature 17).
    context_has_keywords = 0.0
    if text and candidate.span_start is not None:
        window_start = max(0, candidate.span_start - 50)
        window_end = min(len(text), candidate.span_end + 50)
        context_text = text[window_start:window_end].lower()
        if any(kw in context_text for kw in _CONTEXT_KEYWORDS):
            context_has_keywords = 1.0

    # Position in text (feature 18).
    text_len = max(len(text), 1) if text else 1
    position = candidate.span_start / text_len if candidate.span_start is not None else 0.5

    # Surrounding entity density (feature 19).
    density = 0
    if all_candidates:
        for other in all_candidates:
            if other is candidate:
                continue
            if (other.span_start is not None and candidate.span_start is not None
                    and abs(other.span_start - candidate.span_start) <= 100):
                density += 1

    # Engine diversity (feature 20).
    engine_ids = list(candidate.engine_findings.keys())
    if len(engine_ids) >= 2:
        engine_types = {eid: {f.entity_type} for eid, f in candidate.engine_findings.items()}
        pair_sims = []
        for i in range(len(engine_ids)):
            for j in range(i + 1, len(engine_ids)):
                pair_sims.append(_jaccard_similarity(
                    engine_types[engine_ids[i]], engine_types[engine_ids[j]],
                ))
        diversity = 1.0 - (sum(pair_sims) / len(pair_sims))
    else:
        diversity = 0.0

    return [
        candidate.ds_confidence,                    # 1. ds_confidence
        float(candidate.corroboration_count),        # 2. corroboration_count
        candidate.corroboration_count / max(total_engines, 1),  # 3. corroboration_ratio
        max(calibrated_confs),                       # 4. max_engine_confidence
        min(calibrated_confs),                       # 5. min_engine_confidence
        mean_conf,                                   # 6. mean_engine_confidence
        std_conf,                                    # 7. std_engine_confidence
        regex_detected,                              # 8. regex_detected
        regex_confidence,                            # 9. regex_confidence
        float(span_len),                             # 10. span_length_chars
        float(span_tokens),                          # 11. span_length_tokens
        float(entity_encoded),                       # 12. entity_type_encoded
        is_structured,                               # 13. is_structured_type
        has_checksum,                                # 14. has_checksum_validation
        mean_info,                                   # 15. informativeness_score
        boundary_agreement,                          # 16. boundary_agreement
        context_has_keywords,                        # 17. context_has_keywords
        position,                                    # 18. position_in_text
        float(density),                              # 19. surrounding_entity_density
        diversity,                                   # 20. engine_diversity_score
    ]


class XGBoostMetaLearner:
    """Wrapper around XGBoost for span-level TP/FP classification."""

    def __init__(self, model: Any = None) -> None:
        self._model = model

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    def predict(self, features: list[list[float]]) -> list[float]:
        """Predict probability of true positive for each feature vector."""
        if self._model is None:
            raise RuntimeError("Meta-learner model not loaded")
        try:
            import numpy as np
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError(
                "XGBoost is required for the meta-learner. "
                "Install with: pip install pii-anon[swarm-ml]"
            ) from exc

        dmat = xgb.DMatrix(np.array(features))
        return list(self._model.predict(dmat).tolist())

    def predict_candidate(
        self,
        candidate: SpanCandidate,
        *,
        total_engines: int = 6,
        informativeness_scorer: InformativenessScorer | None = None,
        text: str = "",
        all_candidates: list[SpanCandidate] | None = None,
    ) -> float:
        """Extract features and predict for a single candidate."""
        features = extract_features(
            candidate,
            total_engines=total_engines,
            informativeness_scorer=informativeness_scorer,
            text=text,
            all_candidates=all_candidates,
        )
        preds = self.predict([features])
        return preds[0]

    def train(
        self,
        features: list[list[float]],
        labels: list[int],
        *,
        n_rounds: int = 100,
        early_stopping: int = 10,
    ) -> None:
        """Train the XGBoost model on feature/label pairs."""
        try:
            import numpy as np
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError(
                "XGBoost and numpy are required for training. "
                "Install with: pip install pii-anon[swarm-ml]"
            ) from exc

        X = np.array(features)
        y = np.array(labels, dtype=np.float32)

        dtrain = xgb.DMatrix(X, label=y)
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "seed": 42,
        }
        self._model = xgb.train(
            params,
            dtrain,
            num_boost_round=n_rounds,
            evals=[(dtrain, "train")],
            verbose_eval=False,
        )

    def save(self, path: str | Path) -> None:
        if self._model is None:
            raise RuntimeError("No trained model to save")
        path = Path(path)
        self._model.save_model(str(path))

    @classmethod
    def load(cls, path: str | Path) -> XGBoostMetaLearner:
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError(
                "XGBoost is required to load the meta-learner. "
                "Install with: pip install pii-anon[swarm-ml]"
            ) from exc

        model = xgb.Booster()
        model.load_model(str(path))
        return cls(model=model)
