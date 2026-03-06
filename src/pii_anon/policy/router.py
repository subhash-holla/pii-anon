from __future__ import annotations

from dataclasses import dataclass, field

from pii_anon.types import ProcessingProfileSpec, Payload


@dataclass
class ExecutionPlan:
    plan_id: str
    engine_ids: list[str]
    fusion_mode: str
    segmentation_enabled: bool
    concurrency_cap: int = 4
    low_confidence_threshold: float = 0.78
    escalate_on_low_confidence: bool = True
    escalation_engine_ids: list[str] = field(default_factory=list)


class PolicyRouter:
    """Fast deterministic router that selects engines and fusion policy."""

    def select(
        self,
        payload: Payload,
        profile: ProcessingProfileSpec,
        capabilities: dict[str, dict[str, object]],
    ) -> ExecutionPlan:
        text = _primary_text(payload)
        token_count = len(text.split())
        language = (profile.language or "en").lower()
        use_case = profile.use_case or "default"
        objective = profile.objective

        available = _available_engines(capabilities, language=language)
        accuracy_engines = _preferred_accuracy_engines(available)
        speed_engines = _preferred_speed_engines(available)
        balanced_engines = _preferred_balanced_engines(available, use_case=use_case, language=language)

        if use_case == "default":
            return ExecutionPlan(
                plan_id="default_compat",
                engine_ids=list(available),
                fusion_mode=profile.mode,
                segmentation_enabled=False,
                concurrency_cap=1,
                low_confidence_threshold=0.0,
                escalate_on_low_confidence=False,
            )

        segmentation_enabled = token_count > 2000 and use_case in {"long_document", "multilingual_mix"}

        if objective == "accuracy":
            return ExecutionPlan(
                plan_id="accuracy_guarded",
                engine_ids=accuracy_engines,
                fusion_mode="weighted_consensus",
                segmentation_enabled=segmentation_enabled,
                concurrency_cap=min(4, max(1, len(accuracy_engines))),
                low_confidence_threshold=0.88,
                escalate_on_low_confidence=True,
                escalation_engine_ids=[engine_id for engine_id in available if engine_id not in accuracy_engines],
            )

        if objective == "speed":
            return ExecutionPlan(
                plan_id="speed_guarded",
                engine_ids=speed_engines,
                fusion_mode="union_high_recall",
                segmentation_enabled=False,
                concurrency_cap=1,
                low_confidence_threshold=0.0,
                escalate_on_low_confidence=False,
                escalation_engine_ids=[engine_id for engine_id in available if engine_id not in speed_engines],
            )

        return ExecutionPlan(
            plan_id="balanced_guarded",
            engine_ids=balanced_engines,
            fusion_mode="weighted_consensus",
            segmentation_enabled=segmentation_enabled,
            concurrency_cap=min(3, max(1, len(balanced_engines))),
            low_confidence_threshold=0.80,
            escalate_on_low_confidence=True,
            escalation_engine_ids=[engine_id for engine_id in available if engine_id not in balanced_engines],
        )


def _primary_text(payload: Payload) -> str:
    if "text" in payload and isinstance(payload["text"], str):
        return payload["text"]
    for value in payload.values():
        if isinstance(value, str):
            return value
    return ""


def _available_engines(capabilities: dict[str, dict[str, object]], *, language: str) -> list[str]:
    ranked = [
        "regex-oss",
        "presidio-compatible",
        "scrubadub-compatible",
        "spacy-ner-compatible",
        "stanza-ner-compatible",
        "llm-guard-compatible",
    ]
    out: list[str] = []
    for adapter_id in ranked:
        caps = capabilities.get(adapter_id)
        if not caps:
            continue
        supports = caps.get("supports_languages")
        if isinstance(supports, list) and supports and language not in supports:
            if "en" not in supports:
                continue
        dependency_available = bool(caps.get("dependency_available", True))
        if not dependency_available and adapter_id != "regex-oss":
            continue
        out.append(adapter_id)
    if not out:
        return ["regex-oss"]
    return out


def _preferred_accuracy_engines(available: list[str]) -> list[str]:
    # Accuracy profiles prioritize higher-quality detection first, then regex
    # augmentation only when additional coverage is available.
    order = [
        "presidio-compatible",
        "regex-oss",
        "spacy-ner-compatible",
        "stanza-ner-compatible",
    ]
    selected = [item for item in order if item in available]
    return selected or (["regex-oss"] if "regex-oss" in available else available)


def _preferred_speed_engines(available: list[str]) -> list[str]:
    # Prefer the lowest-latency external engines for strict speed floor profiles.
    for adapter_id in ("stanza-ner-compatible", "spacy-ner-compatible", "regex-oss"):
        if adapter_id in available:
            return [adapter_id]
    return available[:1]


def _preferred_balanced_engines(available: list[str], *, use_case: str, language: str) -> list[str]:
    if use_case == "structured_form_accuracy":
        return _preferred_accuracy_engines(available)
    if use_case == "structured_form_latency":
        return _preferred_speed_engines(available)

    selected = ["regex-oss"] if "regex-oss" in available else []
    if use_case in {"multilingual_mix", "long_document"} or language in {"es", "fr"}:
        if "presidio-compatible" in available:
            selected.append("presidio-compatible")
    if "spacy-ner-compatible" in available and "spacy-ner-compatible" not in selected:
        selected.append("spacy-ner-compatible")

    return selected or available
