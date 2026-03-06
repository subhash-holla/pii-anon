from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal, cast

from pii_anon import __version__
from pii_anon.benchmarks import run_benchmark
from pii_anon.config import ConfigManager
from pii_anon.eval_framework import EvaluationFramework, load_eval_dataset
from pii_anon.evaluation import (
    StrategyEvaluator,
    compare_competitors,
    evaluate_pipeline,
    run_benchmark_runtime_preflight,
)
from pii_anon.orchestrator import PIIOrchestrator
from pii_anon.tokenization import DeterministicHMACTokenizer, InMemoryTokenStore
from pii_anon.types import ProcessingProfileSpec, Payload, SegmentationPlan


def _dump_output(payload: dict[str, Any], output: str) -> None:
    if output == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    for key, value in payload.items():
        print(f"{key}: {value}")


def _build_orchestrator(token_key: str, config_path: str | None) -> PIIOrchestrator:
    if config_path:
        config = ConfigManager().load(config_path)
        return PIIOrchestrator(token_key=token_key, config=config)
    return PIIOrchestrator(token_key=token_key)


def create_app() -> Any:
    import typer

    app = typer.Typer(add_completion=False, help="pii-anon command line interface")

    @app.command("detect")
    def detect(
        text: str = typer.Argument(..., help="Input text to inspect"),
        token_key: str = typer.Option("dev-key", help="Tokenization key"),
        config: str | None = typer.Option(None, help="Path to JSON/YAML config file"),
        mode: str = typer.Option("weighted_consensus", help="Fusion mode"),
        language: str = typer.Option("en", help="Language code"),
        transform_mode: str = typer.Option("pseudonymize", help="Transform mode: anonymize|pseudonymize"),
        placeholder_template: str = typer.Option(
            "<{entity_type}:anon_{index}>",
            help="Placeholder template for anonymize mode",
        ),
        tracking_enabled: bool = typer.Option(True, "--tracking-enabled/--no-tracking", help="Enable alias tracking"),
        min_link_score: float = typer.Option(0.8, help="Minimum alias-link score"),
        output: str = typer.Option("json", help="Output format: json|text"),
    ) -> None:
        if transform_mode not in {"anonymize", "pseudonymize"}:
            raise typer.BadParameter("transform_mode must be one of: anonymize, pseudonymize")
        orchestrator = _build_orchestrator(token_key=token_key, config_path=config)
        orchestrator._async.config.tracking.enabled = tracking_enabled
        orchestrator._async.config.tracking.min_link_score = min_link_score
        result = orchestrator.run(
            {"text": text},
            profile=ProcessingProfileSpec(
                profile_id="cli",
                mode=mode,
                language=language,
                transform_mode=cast(Literal["pseudonymize", "anonymize"], transform_mode),
                placeholder_template=placeholder_template,
                entity_tracking_enabled=tracking_enabled,
            ),
            segmentation=SegmentationPlan(enabled=False),
            scope="cli",
            token_version=1,
        )
        _dump_output(result, output)

    @app.command("detect-stream")
    def detect_stream(
        file_path: Path = typer.Argument(..., help="File containing one text payload per line"),
        token_key: str = typer.Option("dev-key", help="Tokenization key"),
        config: str | None = typer.Option(None, help="Path to JSON/YAML config file"),
        mode: str = typer.Option("weighted_consensus", help="Fusion mode"),
        language: str = typer.Option("en", help="Language code"),
        transform_mode: str = typer.Option("pseudonymize", help="Transform mode: anonymize|pseudonymize"),
        placeholder_template: str = typer.Option(
            "<{entity_type}:anon_{index}>",
            help="Placeholder template for anonymize mode",
        ),
        tracking_enabled: bool = typer.Option(True, "--tracking-enabled/--no-tracking", help="Enable alias tracking"),
        min_link_score: float = typer.Option(0.8, help="Minimum alias-link score"),
        output: str = typer.Option("json", help="Output format: json|text"),
    ) -> None:
        if transform_mode not in {"anonymize", "pseudonymize"}:
            raise typer.BadParameter("transform_mode must be one of: anonymize, pseudonymize")
        orchestrator = _build_orchestrator(token_key=token_key, config_path=config)
        orchestrator._async.config.tracking.enabled = tracking_enabled
        orchestrator._async.config.tracking.min_link_score = min_link_score
        payloads: list[Payload] = []
        for line in file_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped:
                payloads.append({"text": stripped})

        records = []
        for item in orchestrator.run_stream(
            payloads,
            profile=ProcessingProfileSpec(
                profile_id="stream",
                mode=mode,
                language=language,
                transform_mode=cast(Literal["pseudonymize", "anonymize"], transform_mode),
                placeholder_template=placeholder_template,
                entity_tracking_enabled=tracking_enabled,
            ),
            segmentation=SegmentationPlan(enabled=False),
            scope="cli-stream",
            token_version=1,
        ):
            records.append(item)
        _dump_output({"items": records, "count": len(records)}, output)

    @app.command("process-file")
    def process_file(
        file_path: Path = typer.Argument(..., help="Input file (CSV, JSON, JSONL, or TXT)"),
        token_key: str = typer.Option("dev-key", help="Tokenization key"),
        config: str | None = typer.Option(None, help="Path to JSON/YAML config file"),
        mode: str = typer.Option("weighted_consensus", help="Fusion mode"),
        language: str = typer.Option("en", help="Language code"),
        transform_mode: str = typer.Option("pseudonymize", help="Transform mode: anonymize|pseudonymize"),
        text_column: str = typer.Option("text", help="Column/field name containing text to process"),
        output_file: Path | None = typer.Option(None, "--output", "-o", help="Output file path"),
        file_format: str | None = typer.Option(None, "--format", "-f", help="Force file format: csv|json|jsonl|txt"),
        whole_file: bool = typer.Option(False, help="TXT only: treat entire file as one record"),
        segmentation_enabled: bool = typer.Option(False, "--segmentation/--no-segmentation", help="Enable segmentation"),
        max_tokens: int = typer.Option(4096, help="Max tokens per segment"),
        output: str = typer.Option("json", help="Console output format: json|text"),
    ) -> None:
        """Process a file of records through the PII detection/anonymization pipeline.

        Supports CSV, JSON, JSONL, and TXT files. Large text blocks are
        automatically chunked for efficient processing.
        """
        from pii_anon.ingestion import FileFormat, IngestConfig

        if transform_mode not in {"anonymize", "pseudonymize"}:
            raise typer.BadParameter("transform_mode must be one of: anonymize, pseudonymize")

        fmt = None
        if file_format:
            try:
                fmt = FileFormat(file_format.lower())
            except ValueError:
                raise typer.BadParameter(f"Unsupported format: {file_format}. Use csv, json, jsonl, or txt.")

        orchestrator = _build_orchestrator(token_key=token_key, config_path=config)
        ingest_config = IngestConfig(
            format=fmt,
            text_column=text_column,
            whole_file=whole_file,
        )

        result = orchestrator.run_file(
            file_path,
            profile=ProcessingProfileSpec(
                profile_id="file",
                mode=mode,
                language=language,
                transform_mode=cast(Literal["pseudonymize", "anonymize"], transform_mode),
            ),
            segmentation=SegmentationPlan(enabled=segmentation_enabled, max_tokens=max_tokens),
            scope="cli-file",
            token_version=1,
            ingest_config=ingest_config,
            output_path=output_file,
        )
        _dump_output(
            {
                "input_path": result.input_path,
                "output_path": result.output_path,
                "format": result.format,
                "records_processed": result.records_processed,
                "records_failed": result.records_failed,
                "total_chars": result.total_chars,
                "total_chunks": result.total_chunks,
                "elapsed_seconds": result.elapsed_seconds,
                "records_per_second": round(result.records_per_second, 2),
                "errors": result.errors,
            },
            output,
        )

    @app.command("tokenize")
    def tokenize(
        text: str = typer.Argument(..., help="Plaintext to tokenize"),
        entity_type: str = typer.Option("GENERIC", help="Entity label"),
        token_key: str = typer.Option("dev-key", help="Tokenization key"),
        output: str = typer.Option("json", help="Output format: json|text"),
    ) -> None:
        tokenizer = DeterministicHMACTokenizer()
        store = InMemoryTokenStore()
        token = tokenizer.tokenize(entity_type, text, "cli", 1, token_key, store=store)
        recovered = tokenizer.detokenize(token, key=token_key, store=store)
        _dump_output(
            {
                "token": token.token,
                "entity_type": token.entity_type,
                "scope": token.scope,
                "version": token.version,
                "detokenized": recovered,
            },
            output,
        )

    @app.command("health")
    def health(
        token_key: str = typer.Option("dev-key", help="Tokenization key"),
        config: str | None = typer.Option(None, help="Path to JSON/YAML config file"),
        output: str = typer.Option("json", help="Output format: json|text"),
    ) -> None:
        orchestrator = _build_orchestrator(token_key=token_key, config_path=config)
        _dump_output({"engines": orchestrator.health_check_engines()}, output)

    @app.command("capabilities")
    def capabilities(
        token_key: str = typer.Option("dev-key", help="Tokenization key"),
        config: str | None = typer.Option(None, help="Path to JSON/YAML config file"),
        output: str = typer.Option("json", help="Output format: json|text"),
    ) -> None:
        orchestrator = _build_orchestrator(token_key=token_key, config_path=config)
        _dump_output({"capabilities": orchestrator.capabilities()}, output)

    @app.command("evaluate")
    def evaluate(
        strategies: str = typer.Option(
            "weighted_consensus,union_high_recall,intersection_consensus",
            help="Comma-separated fusion strategies",
        ),
        dataset: str = typer.Option("pii_anon_benchmark_v1", help="Benchmark dataset identifier"),
        token_key: str = typer.Option("dev-key", help="Tokenization key"),
        config: str | None = typer.Option(None, help="Path to JSON/YAML config file"),
        output: str = typer.Option("json", help="Output format: json|text"),
    ) -> None:
        orchestrator = _build_orchestrator(token_key=token_key, config_path=config)
        evaluator = StrategyEvaluator(orchestrator)
        try:
            report = evaluator.compare_strategies(
                [item.strip() for item in strategies.split(",") if item.strip()],
                dataset=dataset,
            )
        except FileNotFoundError as exc:
            raise typer.BadParameter(str(exc))
        _dump_output(
            {
                "winner": report.winner,
                "results": [r.__dict__ for r in report.results],
            },
            output,
        )

    @app.command("evaluate-pipeline")
    def evaluate_pipeline_command(
        dataset: str = typer.Option("pii_anon_benchmark_v1", help="Benchmark dataset identifier"),
        mode: str = typer.Option("weighted_consensus", help="Fusion mode"),
        transform_mode: str = typer.Option("pseudonymize", help="Transform mode: anonymize|pseudonymize"),
        language: str | None = typer.Option(None, help="Optional language filter"),
        max_samples: int = typer.Option(100, help="Sample cap for pipeline evaluation"),
        token_key: str = typer.Option("dev-key", help="Tokenization key"),
        config: str | None = typer.Option(None, help="Path to JSON/YAML config file"),
        output: str = typer.Option("json", help="Output format: json|text"),
    ) -> None:
        if transform_mode not in {"anonymize", "pseudonymize"}:
            raise typer.BadParameter("transform_mode must be one of: anonymize, pseudonymize")
        orchestrator = _build_orchestrator(token_key=token_key, config_path=config)
        try:
            report = evaluate_pipeline(
                orchestrator,
                dataset=dataset,
                mode=mode,
                transform_mode=cast(Literal["pseudonymize", "anonymize"], transform_mode),
                language=language,
                max_samples=max_samples if max_samples > 0 else None,
            )
        except FileNotFoundError as exc:
            raise typer.BadParameter(str(exc))
        _dump_output(report.__dict__, output)

    @app.command("eval-framework")
    def eval_framework_command(
        dataset: str = typer.Option("eval_framework_v1", help="Evaluation framework dataset identifier"),
        language: str | None = typer.Option(None, help="Optional language filter"),
        difficulty: str | None = typer.Option(None, help="Optional difficulty filter"),
        adversarial_only: bool = typer.Option(False, help="Evaluate adversarial records only"),
        max_records: int = typer.Option(500, help="Sample cap for framework-only evaluation"),
        output: str = typer.Option("json", help="Output format: json|text"),
    ) -> None:
        try:
            records = load_eval_dataset(
                dataset,
                language=language,
                difficulty=difficulty,
                adversarial_only=adversarial_only,
            )
        except FileNotFoundError as exc:
            raise typer.BadParameter(str(exc))
        if max_records > 0:
            records = records[:max_records]
        if not records:
            raise typer.BadParameter("No records available after filters")
        framework = EvaluationFramework()
        report = framework.evaluate_batch(records)
        _dump_output(
            {
                "dataset": dataset,
                "records_evaluated": report.records_evaluated,
                "micro_averaged": report.micro_averaged,
                "macro_averaged": report.macro_averaged,
                "weighted_averaged": report.weighted_averaged,
                "privacy_score": report.privacy_score,
                "fairness_score": report.fairness_score,
            },
            output,
        )

    @app.command("benchmark")
    def benchmark(
        mode: str = typer.Option("weighted_consensus", help="Fusion mode"),
        dataset: str = typer.Option("pii_anon_benchmark_v1", help="Benchmark dataset identifier"),
        max_samples: int = typer.Option(0, help="Optional sample cap for faster dry-runs (0 means full dataset)"),
        token_key: str = typer.Option("dev-key", help="Tokenization key"),
        config: str | None = typer.Option(None, help="Path to JSON/YAML config file"),
        output: str = typer.Option("json", help="Output format: json|text"),
    ) -> None:
        orchestrator = _build_orchestrator(token_key=token_key, config_path=config)
        try:
            summary = run_benchmark(
                orchestrator,
                dataset=dataset,
                mode=mode,
                max_samples=max_samples if max_samples > 0 else None,
            )
        except FileNotFoundError as exc:
            raise typer.BadParameter(str(exc))
        _dump_output(summary.__dict__, output)

    @app.command("compare-competitors")
    def compare_competitors_command(
        dataset: str = typer.Option("pii_anon_benchmark_v1", help="Benchmark dataset identifier"),
        warmup_samples: int = typer.Option(100, help="Warm-up samples per system"),
        measured_runs: int = typer.Option(3, help="Measured runs per system"),
        max_samples: int = typer.Option(0, help="Optional sample cap for faster dry-runs (0 means full dataset)"),
        matrix: str | None = typer.Option(None, help="Optional use-case matrix JSON path"),
        enforce_floors: bool = typer.Option(False, help="Fail command if any required profile floor check fails"),
        dataset_source: str = typer.Option("auto", help="Dataset source policy: auto|package-only"),
        require_all_competitors: bool = typer.Option(
            False,
            help="Fail qualification gate when any configured competitor is unavailable",
        ),
        require_native_competitors: bool = typer.Option(
            False,
            help="Require native competitor execution readiness (no fallback/proxy behavior)",
        ),
        include_end_to_end: bool = typer.Option(
            True,
            help="Include core end-to-end evaluation track in benchmark output",
        ),
        allow_core_native_engines: bool = typer.Option(
            True,
            help="Allow native optional engines in core benchmark execution path",
        ),
        use_case: str = typer.Option("default", help="Single-profile use-case identifier"),
        objective: str = typer.Option("balanced", help="Single-profile objective: accuracy|balanced|speed"),
        output: str = typer.Option("json", help="Output format: json|text"),
    ) -> None:
        if objective not in {"accuracy", "balanced", "speed"}:
            raise typer.BadParameter("objective must be one of: accuracy, balanced, speed")
        if dataset_source not in {"auto", "package-only"}:
            raise typer.BadParameter("dataset_source must be one of: auto, package-only")
        objective_mode = cast(Literal["accuracy", "balanced", "speed"], objective)
        try:
            report = compare_competitors(
                dataset=dataset,
                dataset_source=cast(Literal["auto", "package-only"], dataset_source),
                warmup_samples=warmup_samples,
                measured_runs=measured_runs,
                max_samples=max_samples if max_samples > 0 else None,
                matrix_path=matrix,
                enforce_floors=enforce_floors,
                require_all_competitors=require_all_competitors,
                require_native_competitors=require_native_competitors,
                allow_fallback_detectors=not require_native_competitors,
                include_end_to_end=include_end_to_end,
                allow_core_native_engines=allow_core_native_engines,
                use_case=use_case,
                objective=objective_mode,
            )
        except FileNotFoundError as exc:
            raise typer.BadParameter(str(exc))
        _dump_output(
            {
                "report_schema_version": report.report_schema_version,
                "dataset": report.dataset,
                "dataset_source": report.dataset_source,
                "warmup_samples": report.warmup_samples,
                "measured_runs": report.measured_runs,
                "floor_pass": report.floor_pass,
                "qualification_gate_pass": report.qualification_gate_pass,
                "mit_gate_pass": report.mit_gate_pass,
                "expected_competitors": report.expected_competitors,
                "available_competitors": report.available_competitors,
                "unavailable_competitors": report.unavailable_competitors,
                "all_competitors_available": report.all_competitors_available,
                "require_all_competitors": report.require_all_competitors,
                "require_native_competitors": report.require_native_competitors,
                "systems": [item.__dict__ for item in report.systems],
                "profiles": [
                    {
                        "profile": profile.profile,
                        "objective": profile.objective,
                        "floor_pass": profile.floor_pass,
                        "qualified_competitors": profile.qualified_competitors,
                        "mit_qualified_competitors": profile.mit_qualified_competitors,
                        "floor_checks": [check.__dict__ for check in profile.floor_checks],
                        "systems": [item.__dict__ for item in profile.systems],
                        "end_to_end_systems": [item.__dict__ for item in profile.end_to_end_systems],
                    }
                    for profile in report.profiles
                ],
            },
            output,
        )

    @app.command("benchmark-preflight")
    def benchmark_preflight(
        strict_runtime: bool = typer.Option(True, help="Require linux runtime with shared-memory support"),
        require_all_competitors: bool = typer.Option(
            True,
            help="Require all configured competitors to be available",
        ),
        require_native_competitors: bool = typer.Option(
            True,
            help="Require native competitor readiness (no fallback paths)",
        ),
        output_file: str | None = typer.Option(None, help="Optional JSON output path"),
        output: str = typer.Option("json", help="Output format: json|text"),
    ) -> None:
        report = run_benchmark_runtime_preflight(
            strict_runtime=strict_runtime,
            require_all_competitors=require_all_competitors,
            require_native_competitors=require_native_competitors,
        )
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _dump_output(report, output)
        if not bool(report.get("ready", False)):
            raise typer.Exit(1)

    @app.command("benchmark-publish-suite")
    def benchmark_publish_suite(
        dataset: str = typer.Option("pii_anon_benchmark_v1", help="Benchmark dataset identifier"),
        matrix: str = typer.Option(
            "src/pii_anon/benchmarks/matrix/use_case_matrix.v1.json",
            help="Use-case matrix path for canonical run",
        ),
        warmup_samples: int = typer.Option(100, help="Warm-up samples per system"),
        measured_runs: int = typer.Option(3, help="Measured runs per system"),
        dataset_source: str = typer.Option("package-only", help="Dataset source policy: auto|package-only"),
        artifacts_dir: str = typer.Option("artifacts/benchmarks", help="Output artifacts directory"),
        work_dir: str = typer.Option(".publish-suite", help="Temporary suite working directory"),
        strict_runtime: bool = typer.Option(True, help="Require Linux runtime with shared-memory support"),
        require_all_competitors: bool = typer.Option(
            True,
            help="Require all configured competitors for canonical suite pass",
        ),
        require_native_competitors: bool = typer.Option(
            True,
            help="Require native competitor readiness (no fallback detector proxies)",
        ),
        enforce_floors: bool = typer.Option(True, help="Fail suite if benchmark floor gates fail"),
        include_end_to_end: bool = typer.Option(
            True,
            help="Include core end-to-end benchmark track during suite execution",
        ),
        allow_core_native_engines: bool = typer.Option(
            True,
            help="Allow native optional engines in core benchmark execution path",
        ),
        enforce_publish_claims: bool = typer.Option(
            True,
            help="Require canonical publish-claim gates during summary/README validation",
        ),
        validate_readme_sync: bool = typer.Option(
            True,
            help="Validate README benchmark sync against generated summary",
        ),
        skip_build: bool = typer.Option(False, help="Reuse previously built wheels in suite work dir"),
        skip_readme_update: bool = typer.Option(False, help="Do not update README benchmark section"),
        reuse_current_env: bool = typer.Option(
            False,
            help="Run suite in current Python environment instead of isolated venv",
        ),
        install_no_deps: bool = typer.Option(
            False,
            help="Install built wheels with --no-deps (for pre-provisioned environments)",
        ),
        output: str = typer.Option("json", help="Output format: json|text"),
    ) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        script = repo_root / "scripts" / "run_publish_grade_suite.py"
        if not script.exists():
            raise typer.BadParameter(f"publish suite script not found at {script}")
        if dataset_source not in {"auto", "package-only"}:
            raise typer.BadParameter("dataset_source must be one of: auto, package-only")

        command = [
            sys.executable,
            str(script),
            "--dataset",
            dataset,
            "--matrix",
            matrix,
            "--warmup-samples",
            str(warmup_samples),
            "--measured-runs",
            str(measured_runs),
            "--dataset-source",
            dataset_source,
            "--artifacts-dir",
            artifacts_dir,
            "--work-dir",
            work_dir,
        ]
        command.append("--strict-runtime" if strict_runtime else "--no-strict-runtime")
        command.append("--require-all-competitors" if require_all_competitors else "--no-require-all-competitors")
        command.append(
            "--require-native-competitors" if require_native_competitors else "--no-require-native-competitors"
        )
        command.append("--enforce-floors" if enforce_floors else "--no-enforce-floors")
        command.append("--include-end-to-end" if include_end_to_end else "--no-include-end-to-end")
        command.append("--allow-core-native-engines" if allow_core_native_engines else "--no-allow-core-native-engines")
        command.append("--enforce-publish-claims" if enforce_publish_claims else "--no-enforce-publish-claims")
        command.append("--validate-readme-sync" if validate_readme_sync else "--no-validate-readme-sync")
        if skip_build:
            command.append("--skip-build")
        if skip_readme_update:
            command.append("--skip-readme-update")
        if reuse_current_env:
            command.append("--reuse-current-env")
        if install_no_deps:
            command.append("--install-no-deps")

        proc = subprocess.run(command, cwd=str(repo_root), check=False)
        payload = {
            "exit_code": proc.returncode,
            "command": command,
            "artifacts_dir": artifacts_dir,
        }
        _dump_output(payload, output)
        if proc.returncode != 0:
            raise typer.Exit(proc.returncode)

    @app.command("version")
    def version() -> None:
        print(__version__)

    return app


def main() -> None:
    try:
        app = create_app()
        app()
    except ImportError:
        raise SystemExit(
            "CLI dependencies are not installed. Install with `pip install pii-anon[cli]`."
        )


if __name__ == "__main__":
    main()
