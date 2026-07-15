from __future__ import annotations

# The model-specific tests are intentionally skipped when the optional model
# dependencies are not installed.
# ruff: noqa: E402

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tokenizers")

from core.artifacts import BundleError, create_bundle, load_bundle
from core.model import TinyGPT
from tinygpt_service.config import ServiceSettings
from tinygpt_service.generator import TinyGPTGenerator, load_generator


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    vocab = {"<bos>": 0, "hello": 1, "world": 2, "<eos>": 3, "<unk>": 4}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.decoder = decoders.WordPiece(prefix="##")
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    config = {
        "name": "tiny-test",
        "d_model": 4,
        "n_heads": 2,
        "n_layers": 1,
        "d_ff": 8,
        "dropout": 0.0,
    }
    model = TinyGPT(
        vocab_size=len(vocab),
        context_length=8,
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
    )
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "vocab_size": len(vocab),
            "context_length": 8,
        },
        checkpoint_path,
    )
    return checkpoint_path, tokenizer_path


def test_create_and_load_bundle_validates_digests(tmp_path):
    checkpoint, tokenizer = _write_inputs(tmp_path)
    bundle_dir = create_bundle(checkpoint, tokenizer, tmp_path / "bundle")

    loaded = load_bundle(bundle_dir, "cpu")
    assert loaded.manifest["model_version"] == "tiny-test"
    assert loaded.manifest["model"]["context_length"] == 8
    assert str(loaded.device) == "cpu"

    (bundle_dir / "tokenizer.json").write_text("tampered", encoding="utf-8")
    with pytest.raises(BundleError, match="tokenizer digest"):
        load_bundle(bundle_dir, "cpu")


def test_bundle_builder_refuses_overwrite(tmp_path):
    checkpoint, tokenizer = _write_inputs(tmp_path)
    output = tmp_path / "bundle"
    create_bundle(checkpoint, tokenizer, output)
    with pytest.raises(BundleError, match="refusing to overwrite"):
        create_bundle(checkpoint, tokenizer, output)


def test_bundle_rejects_manifest_compatibility_mismatch(tmp_path):
    checkpoint, tokenizer = _write_inputs(tmp_path)
    bundle = create_bundle(checkpoint, tokenizer, tmp_path / "bundle")
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tokenizer"]["vocab_size"] = 999
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(BundleError, match="vocabulary sizes do not match"):
        load_bundle(bundle, "cpu")


def test_generator_loads_once_and_returns_metadata(tmp_path):
    checkpoint, tokenizer = _write_inputs(tmp_path)
    bundle = create_bundle(checkpoint, tokenizer, tmp_path / "bundle", run_id="run-1")
    generator = TinyGPTGenerator(str(bundle), "cpu")
    assert generator.info.model_version == "tiny-test"
    assert generator.info.run_id == "run-1"
    assert generator.info.context_length == 8
    result = generator.generate(
        "hello", temperature=1.0, top_k=1, max_new_tokens=1
    )
    assert result.prompt_token_count == 1
    assert result.output_token_count == 1
    assert result.stop_reason in {"eos", "max_new_tokens"}


def test_generator_stream_emits_decoder_suffixes(tmp_path, monkeypatch):
    checkpoint, tokenizer = _write_inputs(tmp_path)
    bundle = create_bundle(checkpoint, tokenizer, tmp_path / "bundle")
    generator = TinyGPTGenerator(str(bundle), "cpu")

    def fake_tokens(*args, **kwargs):
        yield from (1, 2, 3)

    monkeypatch.setattr("core.generation.iter_generate_tokens", fake_tokens)
    chunks = list(
        generator.stream("hello", temperature=1.0, top_k=1, max_new_tokens=3)
    )
    result = chunks[-1].result
    assert result is not None
    assert result.stop_reason == "eos"
    assert result.output_token_count == 3
    assert "".join(chunk.delta for chunk in chunks) == result.text


def test_generator_rejects_prompt_over_context_limit(tmp_path):
    checkpoint, tokenizer = _write_inputs(tmp_path)
    bundle = create_bundle(checkpoint, tokenizer, tmp_path / "bundle")
    generator = TinyGPTGenerator(str(bundle), "cpu")
    with pytest.raises(ValueError, match="context limit"):
        generator.generate(
            "hello hello hello hello hello hello hello hello hello",
            temperature=1.0,
            top_k=1,
            max_new_tokens=1,
        )


def test_load_generator_requires_bundle_path():
    with pytest.raises(RuntimeError, match="TINYGPT_BUNDLE_DIR is required"):
        load_generator(ServiceSettings())
