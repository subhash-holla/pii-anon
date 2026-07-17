"""Renderers (Port 3). Each returns ``{relative_filename: text}``; the runner runs
the PII-egress gate over each artifact's bytes before writing it."""

from __future__ import annotations

from .html import render_html
from .json_bundle import render_json_bundle
from .markdown import render_markdown
from .summary import render_summary

__all__ = ["render_json_bundle", "render_markdown", "render_html", "render_summary"]
