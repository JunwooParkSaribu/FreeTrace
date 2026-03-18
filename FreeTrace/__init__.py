# Modified by Claude (claude-opus-4-6, Anthropic AI) — added package init, model download on first import
"""FreeTrace — single-molecule tracking with fBm inference."""
__version__ = "1.6.1.0"  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18

from FreeTrace.model_downloader import ensure_models as _ensure_models
_ensure_models()
del _ensure_models
