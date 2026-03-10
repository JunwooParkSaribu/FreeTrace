# Modified by Claude (claude-opus-4-6, Anthropic AI) — simplified installation
"""
FreeTrace installation (simplified).

With the new pyproject.toml, most of the installation is handled by pip:
    pip install .                  # installs deps + compiles Cython extensions
    pip install .[gui]             # also installs PyQt6
    pip install .[gpu]             # also installs cupy-cuda12x
    pip install .[macos-gpu]       # also installs tensorflow-metal

Models are downloaded automatically on first use.

This file is kept for backward compatibility:
    import FreeTrace.installation  # triggers model download
"""
from FreeTrace.model_downloader import ensure_models

print("***** FreeTrace: checking models... *****")
ensure_models()
print("***** FreeTrace: installation check complete. *****")
print("***** NOTE: Python dependencies are now managed via pip. *****")
print("*****       Run 'pip install .' from the FreeTrace root directory. *****")
