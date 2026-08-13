#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from airoa.constants import (
    DATASET_REVISION,
    LEROBOT_REVISION,
    MODEL_ID,
    MODEL_REVISION,
    PARC_REVISION,
    TRANSFORMERS_REVISION,
)
from airoa.pi05.checkpoint import TOKENIZER_PATTERNS, download_tokenizer, sha256_file

VENDOR_DISTRIBUTIONS = {
    "accelerate": "accelerate",
    "aiohappyeyeballs": "aiohappyeyeballs",
    "aiohttp": "aiohttp",
    "aiosignal": "aiosignal",
    "async-timeout": "async_timeout",
    "attrs": "attrs",
    "av": "av",
    "certifi": "certifi",
    "charset-normalizer": "charset_normalizer",
    "datasets": "datasets",
    "dill": "dill",
    "filelock": "filelock",
    "lerobot": "lerobot",
    "transformers": "transformers",
    "tokenizers": "tokenizers",
    "safetensors": "safetensors",
    "huggingface-hub": "huggingface_hub",
    "frozenlist": "frozenlist",
    "fsspec": "fsspec",
    "hf-xet": "hf_xet",
    "imageio": "imageio",
    "imageio-ffmpeg": "imageio_ffmpeg",
    "idna": "idna",
    "Jinja2": "jinja2",
    "jsonlines": "jsonlines",
    "MarkupSafe": "markupsafe",
    "mergedeep": "mergedeep",
    "multidict": "multidict",
    "multiprocess": "multiprocess",
    "mypy-extensions": "mypy_extensions",
    "numpy": "numpy",
    "packaging": "packaging",
    "pandas": "pandas",
    "Pillow": "PIL",
    "psutil": "psutil",
    "pyarrow": "pyarrow",
    "propcache": "propcache",
    "pyyaml-include": "yamlinclude",
    "python-dateutil": "dateutil",
    "pytz": "pytz",
    "PyYAML": "yaml",
    "regex": "regex",
    "requests": "requests",
    "six": "six",
    "toml": "toml",
    "tqdm": "tqdm",
    "typing-inspect": "typing_inspect",
    "typing_extensions": "typing_extensions",
    "tzdata": "tzdata",
    "urllib3": "urllib3",
    "xxhash": "xxhash",
    "yarl": "yarl",
    "draccus": "draccus",
    "einops": "einops",
    "sentencepiece": "sentencepiece",
    "torchvision": "torchvision",
}


def _copy_package(distribution_name: str, import_name: str, vendor: Path) -> str:
    spec = importlib.util.find_spec(import_name)
    if spec is None:
        raise ImportError(f"Required vendor package is not installed: {import_name}")
    if spec.submodule_search_locations:
        source = Path(next(iter(spec.submodule_search_locations)))
        shutil.copytree(source, vendor / source.name, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        library_names = {
            f"{import_name}.libs",
            f"{re.sub(r'[-_.]+', '_', distribution_name).lower()}.libs",
        }
        for name in library_names:
            shared_libraries = source.parent / name
            if shared_libraries.is_dir():
                shutil.copytree(shared_libraries, vendor / shared_libraries.name, dirs_exist_ok=True)
    elif spec.origin:
        source = Path(spec.origin)
        shutil.copy2(source, vendor / source.name)
    distribution = importlib.metadata.distribution(distribution_name)
    dist_info = Path(distribution._path)  # importlib exposes no public dist-info path
    shutil.copytree(dist_info, vendor / dist_info.name, dirs_exist_ok=True)
    return distribution.version


def _copy_checkpoint(checkpoint: Path, destination: Path) -> None:
    required = [
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_preprocessor_step_2_normalizer_processor.safetensors",
        "policy_postprocessor.json",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    ]
    destination.mkdir(parents=True)
    for name in required:
        source = checkpoint / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, destination / name)


def _zip_tree(root: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.zip")
    if temporary.exists():
        temporary.unlink()
    with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            relative = path.relative_to(root)
            compression = zipfile.ZIP_STORED if path.suffix in {".safetensors", ".so"} else zipfile.ZIP_DEFLATED
            archive.write(path, relative.as_posix(), compress_type=compression, compresslevel=6)
    temporary.replace(output)


def _verify_vendored_runtime(stage: Path) -> None:
    code = """
import importlib.metadata
import sys
from pathlib import Path
root = Path(sys.argv[1])
vendor = (root / 'vendor').resolve()
sys.path.insert(0, str(vendor))
import transformers
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.processor import PolicyProcessorPipeline
from airoa.pi05.checkpoint import load_processors
preprocessor, postprocessor = load_processors(
    root / 'model_weights' / 'pi05',
    root / 'model_weights' / 'pi05' / 'tokenizer',
    device='cpu',
)
for distribution, import_name in json.loads(sys.argv[2]).items():
    module = sys.modules.get(import_name)
    if module is None:
        __import__(import_name)
        module = sys.modules[import_name]
    origin = Path(module.__file__).resolve()
    if vendor not in origin.parents:
        raise RuntimeError(f'{distribution} resolved outside vendor: {origin}')
print('vendored offline runtime/processors: OK', PI05Policy.__name__, PolicyProcessorPipeline.__name__)
"""
    environment = os.environ.copy()
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    required = {
        distribution: import_name
        for distribution, import_name in VENDOR_DISTRIBUTIONS.items()
        if import_name not in {"av", "imageio", "imageio_ffmpeg", "jsonlines", "psutil"}
    }
    subprocess.run(
        [sys.executable, "-c", "import json\n" + code, str(stage), json.dumps(required)],
        check=True,
        env=environment,
    )


def build(selected_model_path: Path, output: Path) -> dict:
    root = Path(__file__).resolve().parents[1]
    selected = json.loads(selected_model_path.read_text(encoding="utf-8"))
    checkpoint = Path(selected["checkpoint"]).resolve()
    stage = root / "artifacts" / "submission_stage"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    shutil.copy2(root / "submission" / "policy_server.py", stage / "policy_server.py")
    shutil.copy2(root / "submission" / "requirements.txt", stage / "requirements.txt")
    model_dir = stage / "model_weights" / "pi05"
    _copy_checkpoint(checkpoint, model_dir)
    tokenizer_source = download_tokenizer()
    tokenizer_dir = model_dir / "tokenizer"
    tokenizer_dir.mkdir()
    for pattern in TOKENIZER_PATTERNS:
        source = tokenizer_source / pattern
        if source.is_file():
            shutil.copy2(source, tokenizer_dir / source.name)
    (model_dir / "runtime_config.json").write_text(
        json.dumps({"execution_horizon": int(selected["execution_horizon"]), "offline": True}, indent=2) + "\n"
    )
    vendor = stage / "vendor"
    vendor.mkdir()
    versions = {
        name: _copy_package(name, import_name, vendor)
        for name, import_name in VENDOR_DISTRIBUTIONS.items()
    }
    shutil.copytree(root / "src" / "airoa", vendor / "airoa", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    manifest = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dataset_revision": DATASET_REVISION,
        "lerobot_revision": LEROBOT_REVISION,
        "transformers_revision": TRANSFORMERS_REVISION,
        "parc_revision": PARC_REVISION,
        "execution_horizon": int(selected["execution_horizon"]),
        "model_sha256": sha256_file(model_dir / "model.safetensors"),
        "policy_server_sha256": sha256_file(stage / "policy_server.py"),
        "vendor_versions": versions,
        "python_export_version": sys.version,
    }
    (stage / "submission_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _verify_vendored_runtime(stage)
    for cache in stage.rglob("__pycache__"):
        shutil.rmtree(cache)
    for bytecode in stage.rglob("*.pyc"):
        bytecode.unlink()
    _zip_tree(stage, output)
    size = output.stat().st_size
    if size >= 20 * 1024**3:
        raise RuntimeError(f"submission.zip exceeds 20 GiB: {size}")
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        required_names = {"policy_server.py", "requirements.txt", "model_weights/pi05/model.safetensors"}
        if not required_names.issubset(names):
            raise RuntimeError(f"Zip root contract violated: missing {required_names - names}")
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"Corrupt zip member: {bad}")
    result = {
        "path": str(output.resolve()),
        "size_bytes": size,
        "size_gib": size / 1024**3,
        "sha256": sha256_file(output),
        "file_count": sum(1 for item in stage.rglob("*") if item.is_file()),
        **manifest,
    }
    (output.parent / "submission_build.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/submission.zip"))
    args = parser.parse_args()
    build(args.selected_model, args.output)


if __name__ == "__main__":
    main()
