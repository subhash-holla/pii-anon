"""XGBoost meta-learner for swarm span classification.

Extracts 21 features per SpanCandidate and predicts whether the detection
is a true positive.  Falls back to logistic scoring when XGBoost is unavailable.

Feature-vector evolution
------------------------
Bump :data:`FEATURE_VERSION` whenever the shape of ``extract_features``
changes.  Trained artifacts in ``~/.pii_anon/swarm/xgboost_model.ubj``
are coupled to a specific version — a mismatch at inference time causes
the caller to fall back to the logistic scorer rather than silently
feeding mis-shaped vectors to XGBoost.

Tier 3 integration
------------------
Dataset v1.3.0+ ships per-record ``behavioral_signal_density`` and
``re_identification_resistance_score``.  Rather than inflate the
inference-time feature vector with fields that callers cannot supply at
prediction time, these are fed into training via
:func:`compute_sample_weights_from_records` — records with high RRS
(harder de-identification targets) receive a larger loss weight, so the
model sharpens its decision boundary on the challenging cases without
changing its input shape.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

from pii_anon.fusion import _overlap_iou
from pii_anon.swarm import STRUCTURED_TYPES, InformativenessScorer, SpanCandidate, _jaccard_similarity

logger = logging.getLogger(__name__)

#: Current meta-learner feature-vector version.  Persisted alongside the
#: trained model and checked at load time to detect incompatibilities
#: between training and inference code.
FEATURE_VERSION: int = 2

# Canonical entity type → integer encoding for the meta-learner.
ENTITY_TYPE_ENCODING: dict[str, int] = {
    "PERSON_NAME": 0, "EMAIL_ADDRESS": 1, "PHONE_NUMBER": 2, "ADDRESS": 3,
    "US_SSN": 4, "CREDIT_CARD": 5, "DATE_OF_BIRTH": 6, "ORGANIZATION": 7,
    "IP_ADDRESS": 8, "MAC_ADDRESS": 9, "BANK_ACCOUNT": 10, "ROUTING_NUMBER": 11,
    "DRIVERS_LICENSE": 12, "PASSPORT": 13, "NATIONAL_ID": 14, "IBAN": 15,
    "USERNAME": 16, "EMPLOYEE_ID": 17, "MEDICAL_RECORD_NUMBER": 18, "LOCATION": 19,
    "LICENSE_PLATE": 20, "VIN": 21, "CRYPTO_WALLET": 22,
}

# PII context keywords for feature 17 — English subset, used for the
# original ``context_has_keywords`` feature.
_CONTEXT_KEYWORDS = frozenset({
    "name", "email", "phone", "ssn", "social", "security", "address", "dob",
    "born", "card", "credit", "account", "passport", "license", "driver",
    "patient", "employee", "id", "number", "tel", "fax", "mobile", "cell",
    "mail", "contact", "ip", "mac", "iban", "routing", "bank",
})

# Multilingual PII context keywords for feature 21.  Covers the five
# largest non-English segments of the pii-anon-datasets v1.3.0 benchmark
# (es: ~10K, fr: ~8.7K, de: ~7.1K, zh: ~3.6K, ja: ~3.4K records) where
# the English-only feature 17 systematically produces zero signal and
# drags swarm precision down on non-Latin-script records.
_MULTILANG_CONTEXT_KEYWORDS = frozenset({
    # Spanish
    "nombre", "correo", "teléfono", "telefono", "dirección", "direccion",
    "cuenta", "tarjeta", "crédito", "credito", "pasaporte", "licencia",
    "conducir", "paciente", "empleado", "número", "numero", "móvil", "movil",
    "identificación", "identificacion",
    # French
    "nom", "courriel", "téléphone", "telephone", "adresse", "compte",
    "carte", "crédit", "credit", "passeport", "permis", "conduire", "patient",
    "employé", "employe", "numéro", "numero", "portable", "mobile",
    "identifiant",
    # German
    "name", "e-mail", "telefon", "adresse", "konto", "karte", "kreditkarte",
    "reisepass", "führerschein", "fuhrerschein", "patient", "mitarbeiter",
    "nummer", "handy", "mobilnummer", "ausweis", "kennung",
    # Chinese (Simplified — common hanzi for PII context words)
    "姓名", "名字", "电话", "地址", "邮箱", "电子邮件", "账户", "账号",
    "信用卡", "护照", "驾照", "驾驶证", "病人", "患者", "员工", "编号",
    "手机", "身份证",
    # Japanese (katakana/hiragana/kanji mix)
    "名前", "氏名", "電話", "住所", "メール", "口座", "カード", "クレジット",
    "パスポート", "運転免許", "患者", "従業員", "番号", "携帯",
    "身分証",
})


def extract_features(
    candidate: SpanCandidate,
    *,
    total_engines: int = 6,
    informativeness_scorer: InformativenessScorer | None = None,
    text: str = "",
    all_candidates: list[SpanCandidate] | None = None,
) -> list[float]:
    """Extract 21 features from a SpanCandidate for the meta-learner.

    The feature shape is pinned to :data:`FEATURE_VERSION` — callers that
    preallocate buffers or validate input dimensionality should read
    that constant rather than hardcoding 21 so future additions don't
    silently drift.
    """
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

    # Context keywords (features 17 + 21).
    context_has_keywords = 0.0
    context_has_multilang_keywords = 0.0
    if text and candidate.span_start is not None:
        window_start = max(0, candidate.span_start - 50)
        window_end = min(len(text), candidate.span_end + 50)
        context_text_raw = text[window_start:window_end]
        context_text = context_text_raw.lower()
        if any(kw in context_text for kw in _CONTEXT_KEYWORDS):
            context_has_keywords = 1.0
        # Feature 21 covers non-English segments where feature 17 is
        # systematically blind — Chinese/Japanese are checked against the
        # *unlowered* string because ``str.lower`` is a no-op for CJK
        # characters and we want to preserve the match intent.
        if any(kw in context_text for kw in _MULTILANG_CONTEXT_KEYWORDS) \
                or any(kw in context_text_raw for kw in _MULTILANG_CONTEXT_KEYWORDS):
            context_has_multilang_keywords = 1.0

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
        context_has_keywords,                        # 17. context_has_keywords (EN)
        position,                                    # 18. position_in_text
        float(density),                              # 19. surrounding_entity_density
        diversity,                                   # 20. engine_diversity_score
        context_has_multilang_keywords,              # 21. context_has_keywords (ES/FR/DE/ZH/JA)
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
        sample_weights: list[float] | None = None,
    ) -> None:
        """Train the XGBoost model on feature/label pairs.

        Parameters
        ----------
        features:
            List of 21-element feature vectors from :func:`extract_features`.
        labels:
            Binary TP (1) / FP (0) label for each vector.
        n_rounds, early_stopping:
            XGBoost training knobs.
        sample_weights:
            Optional per-example weight multiplier for the loss.  When
            present, must have ``len(sample_weights) == len(features)``.
            Produced by :func:`compute_sample_weights_from_records` to
            up-weight records with high Re-identification Resistance
            Score (Tier 3 hard cases) so the meta-learner sharpens on
            them without changing the inference-time feature shape.
        """
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

        if sample_weights is not None:
            if len(sample_weights) != len(features):
                raise ValueError(
                    f"sample_weights length {len(sample_weights)} "
                    f"does not match features length {len(features)}"
                )
            w = np.array(sample_weights, dtype=np.float32)
            dtrain = xgb.DMatrix(X, label=y, weight=w)
        else:
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
        # ``early_stopping_rounds`` is only meaningful when more than one
        # ``evals`` entry is supplied (XGBoost watches the last one).
        # ``early_stopping <= 0`` disables the feature outright so callers
        # can opt out without a separate flag.
        train_kwargs: dict[str, Any] = dict(
            params=params,
            dtrain=dtrain,
            num_boost_round=n_rounds,
            evals=[(dtrain, "train")],
            verbose_eval=False,
        )
        if early_stopping and early_stopping > 0:
            train_kwargs["early_stopping_rounds"] = early_stopping
        self._model = xgb.train(**train_kwargs)

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


# ---------------------------------------------------------------------------
# Training-time helpers — used by the swarm retrain pipeline
# ---------------------------------------------------------------------------

def compute_sample_weights_from_records(
    records: list[Any],
    *,
    rrs_boost: float = 2.0,
    paired_profile_boost: float = 1.5,
    default_weight: float = 1.0,
) -> list[float]:
    """Derive per-example sample weights from Tier 3 record annotations.

    Harder cases (low RRS = easy to re-identify, paired-profile records
    = ESRC adversarial eval target) receive larger weights so the
    meta-learner sharpens its decision boundary on them instead of
    optimising average log-loss on easy cases.

    Parameters
    ----------
    records:
        Iterable of :class:`~pii_anon.swarm_datasets.TrainingRecord`-like
        objects — anything with ``re_identification_resistance_score``
        and ``is_paired_profile`` attributes.
    rrs_boost:
        Maximum weight multiplier applied to the lowest-RRS record.
        Records with ``rrs=None`` or ``rrs=1.0`` receive *default_weight*.
        Records with ``rrs=0.0`` receive *rrs_boost*.
    paired_profile_boost:
        Additional multiplier for paired-profile records.
    default_weight:
        Base weight for records without Tier 3 annotations.

    Returns
    -------
    list[float]
        One weight per input record, ready for :meth:`XGBoostMetaLearner.train`.

    The weight formula::

        w = default_weight * (1 + (rrs_boost - 1) * (1 - rrs))
                           * (paired_profile_boost if is_paired else 1)

    A record with ``rrs=0.0`` (trivially re-identifiable — failed
    de-id) and ``is_paired_profile=True`` therefore contributes
    ``default_weight * rrs_boost * paired_profile_boost`` to the loss.
    """
    if rrs_boost < 1.0 or paired_profile_boost < 1.0:
        raise ValueError("boost factors must be >= 1.0")
    weights: list[float] = []
    for record in records:
        rrs = getattr(record, "re_identification_resistance_score", None)
        if rrs is None:
            base = default_weight
        else:
            rrs_clamped = max(0.0, min(1.0, float(rrs)))
            base = default_weight * (1.0 + (rrs_boost - 1.0) * (1.0 - rrs_clamped))
        if getattr(record, "is_paired_profile", False):
            base *= paired_profile_boost
        weights.append(base)
    return weights


def select_f2_threshold(
    scores: list[float],
    labels: list[int],
    *,
    min_threshold: float = 0.30,
    max_threshold: float = 0.70,
    step: float = 0.02,
    beta: float = 2.0,
) -> tuple[float, float]:
    """Sweep emission thresholds and pick the one that maximises F_beta.

    Implements the paper v10 recommendation: after training the
    meta-learner, select the ``SwarmConfig.emission_threshold`` that
    maximises F2 (β=2, privacy-first) on a held-out split rather than
    using the 0.50 default.  F2 double-weights recall, matching the
    regulatory cost model where a missed entity is worse than a false
    positive (TAB 2022, Lermen et al. 2026).

    Parameters
    ----------
    scores:
        Meta-learner output probabilities, one per candidate.
    labels:
        Binary TP (1) / FP (0) labels aligned with *scores*.
    min_threshold, max_threshold, step:
        Grid defining the sweep — inclusive of ``min_threshold``,
        exclusive of any value greater than ``max_threshold``.
    beta:
        F-score weight.  Default 2.0 matches the paper's privacy-first
        preset.

    Returns
    -------
    (threshold, f_beta)
        The threshold that maximises F_beta and its score.  If no
        threshold yields a positive F_beta (e.g. no TPs anywhere), the
        pair ``(0.5, 0.0)`` is returned so callers fall back to the
        default without crashing.
    """
    if len(scores) != len(labels):
        raise ValueError(
            f"scores and labels length mismatch: {len(scores)} vs {len(labels)}"
        )
    if min_threshold >= max_threshold:
        raise ValueError("min_threshold must be < max_threshold")
    if step <= 0:
        raise ValueError("step must be positive")
    if not scores:
        return 0.5, 0.0

    beta_sq = beta * beta
    best = (0.5, 0.0)
    t = min_threshold
    # Iterate on a rounded grid to avoid float drift pushing us past
    # ``max_threshold`` by ε.
    while t <= max_threshold + 1e-9:
        tp = fp = fn = 0
        for score, label in zip(scores, labels):
            predicted_positive = score >= t
            if label == 1 and predicted_positive:
                tp += 1
            elif label == 0 and predicted_positive:
                fp += 1
            elif label == 1 and not predicted_positive:
                fn += 1
        if tp == 0:
            f_beta = 0.0
        else:
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            denom = beta_sq * precision + recall
            f_beta = (1 + beta_sq) * precision * recall / denom if denom > 0 else 0.0
        if f_beta > best[1]:
            best = (round(t, 6), f_beta)
        t += step
    return best
