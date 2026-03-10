# Modified by Claude (claude-opus-4-6, Anthropic AI) — cross-platform lazy model downloader
"""
Download pre-trained models on first use.
Uses only stdlib (urllib) — no wget/curl/powershell needed.
"""
import os
import ssl
import sys
import zipfile
import shutil
from urllib.request import urlretrieve, install_opener, build_opener, HTTPSHandler

_MODEL_URLS = {
    "3.10": "https://psilo.sorbonne-universite.fr/index.php/s/o8SZrWt4HePY8js/download/models_2_14.zip",
    "default": "https://psilo.sorbonne-universite.fr/index.php/s/w9PrAQbxsNJrEFc/download/models_2_17.zip",
}

_REQUIRED_FILES = ["theta_hat.npz", "std_sets.npz"]


def _freetrace_dir():
    """Return the FreeTrace package directory."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)))


def _models_dir():
    return os.path.join(_freetrace_dir(), "models")


def _python_minor():
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def _model_url():
    minor = _python_minor()
    return _MODEL_URLS.get(minor, _MODEL_URLS["default"])


def _progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        print(f"\r  Downloading models: {mb:.1f}/{total_mb:.1f} MB ({pct}%)", end="", flush=True)
    else:
        mb = downloaded / (1024 * 1024)
        print(f"\r  Downloading models: {mb:.1f} MB", end="", flush=True)


def _download(url, dest):
    """Download url to dest, with SSL certificate fallback."""
    try:
        urlretrieve(url, dest, reporthook=_progress_hook)
    except Exception as first_err:
        # Retry with unverified SSL if certificate verification fails
        if "CERTIFICATE_VERIFY_FAILED" in str(first_err) or "SSL" in str(first_err):
            print("\n  SSL certificate verification failed, retrying without verification...")
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            opener = build_opener(HTTPSHandler(context=ctx))
            install_opener(opener)
            urlretrieve(url, dest, reporthook=_progress_hook)
        else:
            raise


def models_available():
    """Check if all required model files exist."""
    models = _models_dir()
    if not os.path.isdir(models):
        return False
    for f in _REQUIRED_FILES:
        if not os.path.isfile(os.path.join(models, f)):
            return False
    return True


def ensure_models():
    """Download models if not already present. Called automatically on first import."""
    if models_available():
        return

    models = _models_dir()
    url = _model_url()
    zip_path = os.path.join(_freetrace_dir(), "_models_download.zip")

    print(f"***** FreeTrace models not found. Downloading for Python {_python_minor()}... *****")

    try:
        _download(url, zip_path)
        print()  # newline after progress

        # Extract
        if os.path.isdir(models):
            shutil.rmtree(models)

        tmp_dir = os.path.join(_freetrace_dir(), "_models_tmp")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_dir)

        # The zip may contain a nested folder — find and move it
        extracted_items = os.listdir(tmp_dir)
        if len(extracted_items) == 1 and os.path.isdir(os.path.join(tmp_dir, extracted_items[0])):
            shutil.move(os.path.join(tmp_dir, extracted_items[0]), models)
            shutil.rmtree(tmp_dir)
        else:
            shutil.move(tmp_dir, models)

        # Verify
        for f in _REQUIRED_FILES:
            if not os.path.isfile(os.path.join(models, f)):
                print(f"***** WARNING: {f} not found after extraction. "
                      f"Please contact the author for pretrained models. *****")
                return

        print("***** Models downloaded successfully. *****")

    except Exception as e:
        print(f"\n***** WARNING: Failed to download models: {e} *****")
        print("***** FreeTrace will work for localization but fBm tracking requires models. *****")
        print(f"***** Please download manually from: {url} *****")
        print(f"***** Extract to: {models} *****")

    finally:
        if os.path.exists(zip_path):
            os.remove(zip_path)
