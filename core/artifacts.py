"""Immutable model-bundle creation and strict serving-time validation."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class BundleError(RuntimeError):
    """A model bundle is missing, malformed, or incompatible."""


@dataclass(frozen=True)
class BundleRuntime:
    model: Any
    tokenizer: Any
    manifest: Mapping[str, Any]
    device: Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    return {"sha256": sha256_file(path), "bytes": path.stat().st_size}


def _read_checkpoint(path: Path) -> Mapping[str, Any]:
    import torch

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise BundleError(f"unable to inspect checkpoint {path}") from exc
    if not isinstance(checkpoint, Mapping):
        raise BundleError("checkpoint must contain a mapping")
    for key in ("config", "vocab_size", "context_length"):
        if key not in checkpoint:
            raise BundleError(f"checkpoint is missing required field {key!r}")
    if not isinstance(checkpoint["config"], Mapping):
        raise BundleError("checkpoint config must be an object")
    return checkpoint


def create_bundle(
    checkpoint_path: Path,
    tokenizer_path: Path,
    output_dir: Path,
    *,
    run_id: str | None = None,
    model_version: str | None = None,
) -> Path:
    """Create a new bundle atomically; refuse to overwrite an existing one."""
    checkpoint_path = Path(checkpoint_path).resolve()
    tokenizer_path = Path(tokenizer_path).resolve()
    output_dir = Path(output_dir).resolve()
    if not checkpoint_path.is_file():
        raise BundleError(f"checkpoint does not exist: {checkpoint_path}")
    if not tokenizer_path.is_file():
        raise BundleError(f"tokenizer does not exist: {tokenizer_path}")
    if output_dir.exists():
        raise BundleError(f"refusing to overwrite existing bundle: {output_dir}")

    checkpoint = _read_checkpoint(checkpoint_path)
    try:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    except Exception as exc:
        raise BundleError(f"unable to inspect tokenizer {tokenizer_path}") from exc

    model_config = dict(checkpoint["config"])
    model_version = model_version or str(model_config.get("name", "tinygpt"))
    run_id = run_id or model_version
    tokenizer_digest = sha256_file(tokenizer_path)
    tokenizer_id = f"vocab-{tokenizer.get_vocab_size()}-{tokenizer_digest[:12]}"

    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=parent))
    try:
        model_out = temp_dir / "model.pt"
        tokenizer_out = temp_dir / "tokenizer.json"
        shutil.copy2(checkpoint_path, model_out)
        shutil.copy2(tokenizer_path, tokenizer_out)
        manifest = {
            "format_version": 1,
            "model_version": model_version,
            "run_id": run_id,
            "tokenizer_id": tokenizer_id,
            "model": {
                "file": model_out.name,
                **_file_record(model_out),
                "vocab_size": int(checkpoint["vocab_size"]),
                "context_length": int(checkpoint["context_length"]),
                "architecture": model_config,
            },
            "tokenizer": {
                "file": tokenizer_out.name,
                **_file_record(tokenizer_out),
                "vocab_size": tokenizer.get_vocab_size(),
            },
        }
        (temp_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temp_dir, output_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return output_dir


def _bundle_file(bundle_dir: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise BundleError(f"manifest {label} file is invalid")
    path = (bundle_dir / value).resolve()
    if path.parent != bundle_dir or not path.is_file():
        raise BundleError(f"manifest {label} file is missing from bundle")
    return path


def _validate_file(path: Path, record: Any, label: str) -> None:
    if not isinstance(record, Mapping):
        raise BundleError(f"manifest {label} metadata is invalid")
    digest = record.get("sha256")
    size = record.get("bytes")
    if digest != sha256_file(path):
        raise BundleError(f"{label} digest does not match manifest")
    if size != path.stat().st_size:
        raise BundleError(f"{label} byte count does not match manifest")


def _resolve_device(requested: str):
    import torch

    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested not in {"cpu", "cuda", "mps"}:
        raise BundleError("TINYGPT_DEVICE must be one of auto, cpu, cuda, or mps")
    if requested == "cuda" and not torch.cuda.is_available():
        raise BundleError("TINYGPT_DEVICE=cuda but CUDA is unavailable")
    if requested == "mps" and (
        not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available()
    ):
        raise BundleError("TINYGPT_DEVICE=mps but MPS is unavailable")
    return torch.device(requested)


def load_bundle(bundle_dir: Path, requested_device: str = "auto") -> BundleRuntime:
    """Validate and load one immutable bundle, failing on every mismatch."""
    from tokenizers import Tokenizer

    bundle_dir = Path(bundle_dir).expanduser().resolve()
    if not bundle_dir.is_dir():
        raise BundleError(f"model bundle directory does not exist: {bundle_dir}")
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.is_file():
        raise BundleError(f"model bundle is missing manifest.json: {bundle_dir}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleError(f"unable to read model bundle manifest: {manifest_path}") from exc
    if not isinstance(manifest, Mapping) or manifest.get("format_version") != 1:
        raise BundleError("model bundle manifest has an unsupported format")
    for key in ("model_version", "run_id", "tokenizer_id", "model", "tokenizer"):
        if key not in manifest:
            raise BundleError(f"model bundle manifest is missing {key!r}")

    model_meta = manifest["model"]
    tokenizer_meta = manifest["tokenizer"]
    if not isinstance(model_meta, Mapping) or not isinstance(tokenizer_meta, Mapping):
        raise BundleError("model and tokenizer manifest entries must be objects")
    model_path = _bundle_file(bundle_dir, model_meta.get("file"), "model")
    tokenizer_path = _bundle_file(bundle_dir, tokenizer_meta.get("file"), "tokenizer")
    _validate_file(model_path, model_meta, "model")
    _validate_file(tokenizer_path, tokenizer_meta, "tokenizer")

    device = _resolve_device(requested_device)
    from .model import load_checkpoint

    try:
        model, checkpoint = load_checkpoint(model_path, device)
    except (ValueError, RuntimeError) as exc:
        raise BundleError(f"unable to load model checkpoint: {exc}") from exc
    checkpoint_vocab = int(checkpoint["vocab_size"])
    checkpoint_context = int(checkpoint["context_length"])
    if checkpoint_vocab != model_meta.get("vocab_size"):
        raise BundleError("checkpoint vocabulary size does not match manifest")
    if checkpoint_context != model_meta.get("context_length"):
        raise BundleError("checkpoint context length does not match manifest")
    if checkpoint_vocab != tokenizer_meta.get("vocab_size"):
        raise BundleError("checkpoint and tokenizer vocabulary sizes do not match")
    try:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    except Exception as exc:
        raise BundleError("unable to load tokenizer from model bundle") from exc
    if tokenizer.get_vocab_size() != checkpoint_vocab:
        raise BundleError("tokenizer vocabulary size does not match checkpoint")
    return BundleRuntime(model=model, tokenizer=tokenizer, manifest=manifest, device=device)
