"""Structured evidence citations for every metric and design decision.

Every component of the evaluation framework is backed by at least one
peer-reviewed paper, standard, or widely-adopted open-source benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResearchReference:
    """A single citable research artefact."""

    key: str
    title: str
    authors: str
    year: int
    venue: str
    doi_or_url: str
    relevance: str


# ---------------------------------------------------------------------------
# Master registry — maps framework feature names to supporting evidence
# ---------------------------------------------------------------------------

_REFS: dict[str, ResearchReference] = {
    "semeval13": ResearchReference(
        key="semeval13",
        title="SemEval-2013 Task 9.1: Recognition and Classification of Pharmacological Substances",
        authors="Segura-Bedmar, I., Martinez, P., Herrero-Zazo, M.",
        year=2013,
        venue="SemEval 2013",
        doi_or_url="https://aclanthology.org/S13-2056/",
        relevance="Defines strict/exact/partial/type entity matching modes used in span_metrics.",
    ),
    "nervaluate": ResearchReference(
        key="nervaluate",
        title="nervaluate: NER Evaluation Considering Partial Matching and Entity Types",
        authors="Batista, D.S.",
        year=2018,
        venue="GitHub / PyPI",
        doi_or_url="https://github.com/MantisAI/nervaluate",
        relevance="Entity-level evaluation library implementing SemEval matching modes; "
        "our span_metrics module provides a compatible superset.",
    ),
    "seqeval": ResearchReference(
        key="seqeval",
        title="seqeval: A Python framework for sequence labeling evaluation",
        authors="Nakayama, H.",
        year=2018,
        venue="GitHub / PyPI",
        doi_or_url="https://github.com/chakki-works/seqeval",
        relevance="Token-level NER evaluation using BIO tags; our TokenLevelF1Metric "
        "replicates its micro-averaged F1.",
    ),
    "openner10": ResearchReference(
        key="openner10",
        title="OpenNER 1.0: Standardized Open-Access Named Entity Recognition Datasets in 50+ Languages",
        authors="Malmasi, S., et al.",
        year=2024,
        venue="arXiv preprint (December 2024)",
        doi_or_url="https://arxiv.org/abs/2412.09587",
        relevance="52-language NER benchmark with 2.8 M+ entity mentions; guides our "
        "language coverage and cross-lingual evaluation design.",
    ),
    "tab2022": ResearchReference(
        key="tab2022",
        title="The Text Anonymization Benchmark (TAB): A Dedicated Corpus and Evaluation Framework for Text Anonymization",
        authors="Pil\u00e1n, I., Lison, P., \u00d8vrelid, L., Papadopoulou, A., S\u00e1nchez, D., Batet, M.",
        year=2022,
        venue="Computational Linguistics, 48(4):1053\u20131101",
        doi_or_url="https://doi.org/10.1162/coli_a_00458",
        relevance="Defines privacy-utility trade-off metrics for text anonymization; "
        "informs our utility_metrics module.",
    ),
    "ratbench2025": ResearchReference(
        key="ratbench2025",
        title="Unique Between the Lines: Benchmarking Re-identification Risk for Text Anonymization",
        authors="Various",
        year=2025,
        venue="OpenReview (2025)",
        doi_or_url="https://openreview.net/forum?id=FjbU4kLriN",
        relevance="Introduces RAT-Bench and TRIR (Text Re-Identification Risk Index); "
        "basis for our ReidentificationRiskMetric.",
    ),
    "nist800122": ResearchReference(
        key="nist800122",
        title="Guide to Protecting the Confidentiality of Personally Identifiable Information (PII)",
        authors="McCallister, E., Grance, T., Scarfone, K.",
        year=2010,
        venue="NIST Special Publication 800-122",
        doi_or_url="https://doi.org/10.6028/NIST.SP.800-122",
        relevance="Foundational US federal PII taxonomy; informs entity type "
        "classification and risk levels in taxonomy.py.",
    ),
    "gdpr_art4": ResearchReference(
        key="gdpr_art4",
        title="Regulation (EU) 2016/679 \u2013 General Data Protection Regulation, Article 4",
        authors="European Parliament and Council",
        year=2016,
        venue="Official Journal of the European Union",
        doi_or_url="https://gdpr-info.eu/art-4-gdpr/",
        relevance="Defines personal data and special categories (Art. 9); maps to "
        "entity categories in taxonomy.py.",
    ),
    "iso27701": ResearchReference(
        key="iso27701",
        title="ISO/IEC 27701:2019 \u2013 Privacy Information Management",
        authors="ISO/IEC JTC 1/SC 27",
        year=2019,
        venue="International Organization for Standardization",
        doi_or_url="https://www.iso.org/standard/71670.html",
        relevance="Privacy information management standard; informs PII de-identification "
        "controls and compliance validation.",
    ),
    "i2b2_2014": ResearchReference(
        key="i2b2_2014",
        title="Automated Systems for the De-identification of Longitudinal Clinical Narratives: "
        "Overview of 2014 i2b2/UTHealth Shared Task Track 1",
        authors="Stubbs, A., Kotfila, C., Uzuner, \u00d6.",
        year=2015,
        venue="Journal of Biomedical Informatics, 58:S11\u2013S19",
        doi_or_url="https://doi.org/10.1016/j.jbi.2015.06.007",
        relevance="Gold-standard clinical de-identification benchmark (1,304 records); "
        "informs medical entity types and evaluation methodology.",
    ),
    "crapii2024": ResearchReference(
        key="crapii2024",
        title="Enhancing the De-identification of Personally Identifiable Information in Educational Data",
        authors="Various",
        year=2024,
        venue="Educational Data Mining (EDM 2024)",
        doi_or_url="https://educationaldatamining.org/edm2024/proceedings/2024.EDM-posters.88/",
        relevance="CRAPII dataset (22,688 student writing samples, 14 PII types); "
        "validates educational domain entity coverage.",
    ),
    "regler2021": ResearchReference(
        key="regler2021",
        title="Regularization for Long Named Entity Recognition",
        authors="Various",
        year=2021,
        venue="arXiv",
        doi_or_url="https://arxiv.org/abs/2104.07249",
        relevance="Demonstrates bias in NER toward dataset length distributions; "
        "informs our context-length-aware evaluation.",
    ),
    "piibench2025": ResearchReference(
        key="piibench2025",
        title="PII-Bench: Evaluating Query-Aware Privacy Protection Systems",
        authors="Various",
        year=2025,
        venue="arXiv",
        doi_or_url="https://arxiv.org/abs/2502.18545",
        relevance="Two-stage PII annotation (identification + contextual relevance); "
        "informs contextual evaluation design.",
    ),
    "hybrid_pii_2025": ResearchReference(
        key="hybrid_pii_2025",
        title="A Hybrid Rule-Based NLP and Machine Learning Approach for PII Detection",
        authors="Various",
        year=2025,
        venue="Scientific Reports",
        doi_or_url="https://doi.org/10.1038/s41598-025-04971-9",
        relevance="Hybrid approach achieving 94.7% precision / 89.4% recall; "
        "validates multi-engine ensemble evaluation methodology.",
    ),
    "bitter_lesson_multilingual": ResearchReference(
        key="bitter_lesson_multilingual",
        title="The Bitter Lesson Learned from 2,000+ Multilingual Benchmarks",
        authors="Various",
        year=2024,
        venue="arXiv",
        doi_or_url="https://arxiv.org/abs/2504.15521",
        relevance="Reveals English overrepresentation in NER benchmarks; motivates "
        "equitable language coverage in our framework.",
    ),
    "kanonymity": ResearchReference(
        key="kanonymity",
        title="k-Anonymity: A Model for Protecting Privacy",
        authors="Sweeney, L.",
        year=2002,
        venue="International Journal of Uncertainty, Fuzziness and Knowledge-Based Systems, 10(5):557\u2013570",
        doi_or_url="https://doi.org/10.1142/S0218488502001648",
        relevance="Foundational privacy model; informs KAnonymityMetric.",
    ),
    "ldiversity": ResearchReference(
        key="ldiversity",
        title="l-Diversity: Privacy Beyond k-Anonymity",
        authors="Machanavajjhala, A., Kifer, D., Gehrke, J., Venkitasubramaniam, M.",
        year=2007,
        venue="ACM Transactions on Knowledge Discovery from Data, 1(1):3-es",
        doi_or_url="https://doi.org/10.1145/1217299.1217302",
        relevance="Extends k-anonymity with diversity requirement; informs LDiversityMetric.",
    ),
    "tcloseness": ResearchReference(
        key="tcloseness",
        title="t-Closeness: Privacy Beyond k-Anonymity and l-Diversity",
        authors="Li, N., Li, T., Venkatasubramanian, S.",
        year=2007,
        venue="ICDE 2007",
        doi_or_url="https://doi.org/10.1109/ICDE.2007.367856",
        relevance="Distribution-based privacy guarantee; informs TClosenessMetric.",
    ),
    # -- Dataset methodology & statistical rigor --
    "efron_bootstrap_1993": ResearchReference(
        key="efron_bootstrap_1993",
        title="An Introduction to the Bootstrap",
        authors="Efron, B., Tibshirani, R.J.",
        year=1993,
        venue="Chapman & Hall/CRC",
        doi_or_url="https://doi.org/10.1007/978-1-4899-4541-9",
        relevance="Foundation for bootstrap confidence intervals; used for "
        "generalised metric CIs across precision, recall, F1, and privacy metrics.",
    ),
    "berg_kirkpatrick_2012": ResearchReference(
        key="berg_kirkpatrick_2012",
        title="An Empirical Investigation of Statistical Significance in NLP",
        authors="Berg-Kirkpatrick, T., Burkett, D., Klein, D.",
        year=2012,
        venue="EMNLP 2012",
        doi_or_url="https://aclanthology.org/D12-1091/",
        relevance="Paired bootstrap significance testing for NLP system comparison; "
        "informs our paired_bootstrap_test in aggregation module.",
    ),
    "cohen_kappa_1960": ResearchReference(
        key="cohen_kappa_1960",
        title="A Coefficient of Agreement for Nominal Scales",
        authors="Cohen, J.",
        year=1960,
        venue="Educational and Psychological Measurement, 20(1):37-46",
        doi_or_url="https://doi.org/10.1177/001316446002000104",
        relevance="Inter-annotator agreement via Cohen's kappa; used for "
        "simulated annotator agreement on synthetic dataset labels.",
    ),
    "cohen_d_1988": ResearchReference(
        key="cohen_d_1988",
        title="Statistical Power Analysis for the Behavioral Sciences",
        authors="Cohen, J.",
        year=1988,
        venue="Lawrence Erlbaum Associates (2nd edition)",
        doi_or_url="https://doi.org/10.4324/9780203771587",
        relevance="Effect size measures (Cohen's d) for system comparison; "
        "quantifies practical significance beyond p-values.",
    ),
    "bender_friedman_2018": ResearchReference(
        key="bender_friedman_2018",
        title="Data Statements for Natural Language Processing: Toward Mitigating "
        "System Bias and Enabling Better Science",
        authors="Bender, E.M., Friedman, B.",
        year=2018,
        venue="Transactions of the ACL, 6:587-604",
        doi_or_url="https://doi.org/10.1162/tacl_a_00041",
        relevance="Dataset documentation standards; informs provenance, "
        "integrity verification, and data statement practices in our framework.",
    ),
    "gebru_datasheets_2021": ResearchReference(
        key="gebru_datasheets_2021",
        title="Datasheets for Datasets",
        authors="Gebru, T., Morgenstern, J., Vecchione, B., et al.",
        year=2021,
        venue="Communications of the ACM, 64(12):86-92",
        doi_or_url="https://doi.org/10.1145/3458723",
        relevance="Standardised dataset documentation; motivates our dataset_statistics, "
        "checksum verification, and distribution analysis functions.",
    ),
    # -- v1.0.0: PII-Rate-Elo composite metric & rating engine --
    "elo_1978": ResearchReference(
        key="elo_1978",
        title="The Rating of Chessplayers, Past and Present",
        authors="Elo, A.E.",
        year=1978,
        venue="Arco Publishing (2nd edition)",
        doi_or_url="https://en.wikipedia.org/wiki/Elo_rating_system",
        relevance="Foundational rating system for paired comparison; basis for "
        "PII-Rate-Elo pairwise system rating engine.",
    ),
    "glicko_2001": ResearchReference(
        key="glicko_2001",
        title="Parameter estimation in large dynamic paired comparison experiments",
        authors="Glickman, M.E.",
        year=2001,
        venue="Journal of the Royal Statistical Society: Series C, 48(3):377–394",
        doi_or_url="https://doi.org/10.1111/1467-9876.00159",
        relevance="Rating Deviation (RD) for uncertainty quantification in "
        "paired-comparison ratings; used in PII-Rate-Elo engine.",
    ),
    "bradley_terry_1952": ResearchReference(
        key="bradley_terry_1952",
        title="Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons",
        authors="Bradley, R.A., Terry, M.E.",
        year=1952,
        venue="Biometrika, 39(3/4):324–345",
        doi_or_url="https://doi.org/10.2307/2334029",
        relevance="Paired-comparison probability model underlying Elo expected "
        "score formula; statistical foundation for composite metric.",
    ),
    "nist_800_188": ResearchReference(
        key="nist_800_188",
        title="De-Identifying Government Datasets",
        authors="Garfinkel, S.",
        year=2016,
        venue="NIST Special Publication 800-188 (2nd Draft)",
        doi_or_url="https://csrc.nist.gov/publications/detail/sp/800-188/draft",
        relevance="Risk-calibrated de-identification thresholds; informs "
        "privacy normalization bounds in composite metric.",
    ),
    "pii_rate_elo_2026": ResearchReference(
        key="pii_rate_elo_2026",
        title="PII-Rate-Elo: Composite Metric for PII De-Identification System Evaluation",
        authors="pii-anon contributors",
        year=2026,
        venue="Internal design document",
        doi_or_url="docs/composite-metric-evidence.md",
        relevance="Internal research and design backing for the composite metric, "
        "Elo rating engine, and leaderboard system.",
    ),
}


EVIDENCE_REGISTRY: dict[str, list[ResearchReference]] = {
    "span_metrics": [_REFS["semeval13"], _REFS["nervaluate"], _REFS["seqeval"]],
    "token_level_metrics": [_REFS["seqeval"]],
    "privacy_metrics": [_REFS["ratbench2025"], _REFS["kanonymity"], _REFS["ldiversity"], _REFS["tcloseness"]],
    "utility_metrics": [_REFS["tab2022"]],
    "fairness_metrics": [_REFS["bitter_lesson_multilingual"], _REFS["openner10"]],
    "taxonomy": [_REFS["nist800122"], _REFS["gdpr_art4"], _REFS["iso27701"]],
    "language_support": [_REFS["openner10"], _REFS["bitter_lesson_multilingual"]],
    "context_evaluation": [_REFS["regler2021"], _REFS["piibench2025"]],
    "dataset": [_REFS["i2b2_2014"], _REFS["crapii2024"], _REFS["openner10"],
                _REFS["gebru_datasheets_2021"], _REFS["bender_friedman_2018"]],
    "dataset_methodology": [_REFS["gebru_datasheets_2021"], _REFS["bender_friedman_2018"]],
    "statistical_significance": [_REFS["berg_kirkpatrick_2012"], _REFS["efron_bootstrap_1993"],
                                 _REFS["cohen_d_1988"]],
    "inter_annotator_agreement": [_REFS["cohen_kappa_1960"]],
    "confidence_intervals": [_REFS["efron_bootstrap_1993"]],
    "reidentification_risk": [_REFS["ratbench2025"]],
    "standards_compliance": [_REFS["nist800122"], _REFS["gdpr_art4"], _REFS["iso27701"]],
    "composite_metric": [
        _REFS["pii_rate_elo_2026"],
        _REFS["bradley_terry_1952"],
        _REFS["tab2022"],
        _REFS["nist_800_188"],
    ],
    "elo_rating": [
        _REFS["elo_1978"],
        _REFS["glicko_2001"],
        _REFS["bradley_terry_1952"],
        _REFS["pii_rate_elo_2026"],
    ],
}


def get_references_for(feature: str) -> list[ResearchReference]:
    """Return all research references supporting *feature*."""
    return list(EVIDENCE_REGISTRY.get(feature, []))


def all_references() -> list[ResearchReference]:
    """Return all unique references sorted by key."""
    seen: set[str] = set()
    out: list[ResearchReference] = []
    for ref in _REFS.values():
        if ref.key not in seen:
            seen.add(ref.key)
            out.append(ref)
    return sorted(out, key=lambda r: r.key)
