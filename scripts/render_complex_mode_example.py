#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from pii_anon import PIIOrchestrator
from pii_anon.types import ProcessingProfileSpec, SegmentationPlan

README_START = "<!-- COMPLEX_MODE_EXAMPLE_START -->"
README_END = "<!-- COMPLEX_MODE_EXAMPLE_END -->"


def _complex_input() -> str:
    parts = [
        "Primary record owner is Jack Davis for account AC-7721.",
        "In review notes, the same person is referenced as alias Jack.",
        "Escalations and approvals route to jackdavis@example.com.",
        "Later timeline entries continue to refer to Jack during dispute handling.",
    ]
    return " ".join(parts)


def _render_block() -> str:
    text = _complex_input()
    orchestrator = PIIOrchestrator(token_key="readme-demo-key")

    pseudo = orchestrator.run(
        {"text": text},
        profile=ProcessingProfileSpec(
            profile_id="readme-demo-pseudo",
            mode="weighted_consensus",
            use_case="long_document",
            objective="accuracy",
            transform_mode="pseudonymize",
            entity_tracking_enabled=True,
        ),
        segmentation=SegmentationPlan(enabled=True, max_tokens=64, overlap_tokens=8),
        scope="readme-demo",
        token_version=1,
    )

    anon = orchestrator.run(
        {"text": text},
        profile=ProcessingProfileSpec(
            profile_id="readme-demo-anon",
            mode="weighted_consensus",
            use_case="long_document",
            objective="accuracy",
            transform_mode="anonymize",
            placeholder_template="<{entity_type}:anon_{index}>",
            entity_tracking_enabled=True,
        ),
        segmentation=SegmentationPlan(enabled=True, max_tokens=64, overlap_tokens=8),
        scope="readme-demo",
        token_version=1,
    )

    lines = [
        "Complex mode comparison (generated):",
        "",
        "Input:",
        "```text",
        text,
        "```",
        "",
        "Pseudonymize output:",
        "```text",
        str(pseudo["transformed_payload"]["text"]),
        "```",
        "",
        "Anonymize output:",
        "```text",
        str(anon["transformed_payload"]["text"]),
        "```",
        "",
        "Linking notes:",
        f"- pseudonymize link_audit entries: {len(pseudo.get('link_audit', []))}",
        f"- anonymize link_audit entries: {len(anon.get('link_audit', []))}",
        "",
        "This section is generated from deterministic demo input.",
    ]
    return "\n".join(lines).strip() + "\n"


def _inject(readme_path: Path, body: str) -> None:
    text = readme_path.read_text(encoding="utf-8")
    if README_START not in text or README_END not in text:
        raise SystemExit(f"README markers not found. Expected `{README_START}` and `{README_END}`")

    start = text.index(README_START) + len(README_START)
    end = text.index(README_END)
    updated = text[:start] + "\n\n" + body.rstrip() + "\n\n" + text[end:]
    readme_path.write_text(updated, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render complex mode comparison section")
    parser.add_argument("--output-markdown", default="docs/complex-mode-example.md")
    parser.add_argument("--update-readme", default="")
    args = parser.parse_args()

    body = _render_block()
    output = Path(args.output_markdown)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(body, encoding="utf-8")
    print(f"wrote {output}")

    if args.update_readme:
        _inject(Path(args.update_readme), body)
        print(f"updated README complex mode section in {args.update_readme}")


if __name__ == "__main__":
    main()
