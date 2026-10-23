import json
from pathlib import Path

import numpy as np
import pytest
import torch

from codeintel.config import ConfigError, ModelConfig, from_dict, load_model_config
from codeintel.models.base import load_model, verify_snapshot
from codeintel.models.coderank import CODERANK_SPEC, CodeRankEmbedAdapter, _extended_attention_mask
from codeintel.models.modernbert import GTE_MODERNBERT_SPEC

ROOT = Path(__file__).parent.parent


def test_configs_pin_full_shas_and_reviewed_code():
    coderank = load_model_config(ROOT / "configs/models/coderankembed.yaml")
    gte = load_model_config(ROOT / "configs/models/gte-modernbert.yaml")
    assert len(coderank.revision) == 40 and len(gte.revision) == 40
    assert coderank.trust_remote_code and set(coderank.remote_code_sha256) == {
        "configuration_hf_nomic_bert.py",
        "modeling_hf_nomic_bert.py",
    }
    assert not gte.trust_remote_code


def test_adapter_contracts_differ_where_model_cards_differ():
    assert CODERANK_SPEC.query_prefix == "Represent this query for searching relevant code: "
    assert GTE_MODERNBERT_SPEC.query_prefix == ""
    assert CODERANK_SPEC.pooling == GTE_MODERNBERT_SPEC.pooling == "cls"
    assert CODERANK_SPEC.requires_remote_code and not GTE_MODERNBERT_SPEC.requires_remote_code


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"revision": "main"}, "40-char commit SHA"),
        ({"trust_remote_code": True}, "remote_code_sha256"),
    ],
)
def test_config_validation(overrides, message):
    cfg = ModelConfig(**{"name": "m", "adapter": "modernbert", "repo_id": "x/y", "revision": "a" * 40, **overrides})
    with pytest.raises(ConfigError, match=message):
        cfg.validate()


def test_unknown_config_keys_rejected():
    with pytest.raises(ConfigError, match="unknown keys"):
        from_dict(ModelConfig, {"name": "m", "adapter": "hf", "repo_id": "x", "trust_remote_cod": True})


def _snapshot(tmp_path: Path, files: dict[str, str]) -> Path:
    for name, content in files.items():
        (tmp_path / name).write_text(content)
    (tmp_path / "model.safetensors").write_bytes(b"")
    return tmp_path


def _cfg(**kw) -> ModelConfig:
    return ModelConfig(name="m", adapter="coderank", repo_id="x/y", revision="a" * 40, **kw)


def test_verify_snapshot_rejects_pickles(tmp_path):
    snap = _snapshot(tmp_path, {})
    (snap / "pytorch_model.bin").write_bytes(b"\x80")
    with pytest.raises(ConfigError, match="pickle"):
        verify_snapshot(_cfg(), snap)


def test_verify_snapshot_remote_code_hashes(tmp_path):
    import hashlib

    code = "print('hi')\n"
    snap = _snapshot(tmp_path, {"modeling.py": code, "config.json": "{}"})
    good = hashlib.sha256(code.encode()).hexdigest()
    verify_snapshot(_cfg(trust_remote_code=True, remote_code_sha256={"modeling.py": good}), snap)
    with pytest.raises(ConfigError, match="does not match"):
        verify_snapshot(_cfg(trust_remote_code=True, remote_code_sha256={"modeling.py": "0" * 64}), snap)
    (snap / "extra.py").write_text("x = 1\n")
    with pytest.raises(ConfigError, match="not been reviewed"):
        verify_snapshot(_cfg(trust_remote_code=True, remote_code_sha256={"modeling.py": good}), snap)


def test_verify_snapshot_rejects_cross_repo_auto_map(tmp_path):
    import hashlib

    code = "x = 1\n"
    snap = _snapshot(tmp_path, {"m.py": code, "config.json": json.dumps({"auto_map": {"AutoModel": "other/repo--m.Model"}})})
    with pytest.raises(ConfigError, match="another repository"):
        verify_snapshot(_cfg(trust_remote_code=True, remote_code_sha256={"m.py": hashlib.sha256(code.encode()).hexdigest()}), snap)


def test_coderank_requires_explicit_remote_code_opt_in():
    with pytest.raises(ConfigError, match="requires remote code"):
        CodeRankEmbedAdapter(_cfg())


def test_extended_attention_mask_shim_matches_transformers_4_semantics():
    module = torch.nn.Linear(2, 2)
    mask = torch.tensor([[1, 1, 0]])
    ext = _extended_attention_mask(module, mask, mask.shape)
    assert ext.shape == (1, 1, 1, 3)
    assert ext[0, 0, 0, 0] == 0 and ext[0, 0, 0, 2] == torch.finfo(torch.float32).min


def test_hf_adapter_on_local_checkpoint(tiny_model_cfg, tmp_path):
    model = load_model(tiny_model_cfg)
    assert model.query_prefix == "query: "
    texts = ["def parse(file): return json", "class retry: pass", "x"]
    emb = model.encode_documents(texts)
    assert emb.shape == (3, 32)
    np.testing.assert_allclose(np.linalg.norm(emb, axis=1), 1.0, rtol=1e-5)
    # Batching/padding must not change embeddings.
    single = np.stack([model.encode_documents([t])[0] for t in texts])
    np.testing.assert_allclose(emb, single, atol=1e-5)
    model.save(tmp_path / "saved")
    reloaded = load_model(ModelConfig(name="re", adapter="hf", repo_id="local/x", local_dir=str(tmp_path / "saved"), device="cpu"))
    np.testing.assert_allclose(reloaded.encode_documents(texts), emb, atol=1e-6)


@pytest.mark.network
@pytest.mark.parametrize("config", ["coderankembed.yaml", "gte-modernbert.yaml"])
def test_real_checkpoint_ranks_relevant_code_first(config):
    model = load_model(load_model_config(ROOT / "configs/models" / config))
    q = model.encode_queries(["compute the factorial of n", "read a json file"])
    d = model.encode_documents(
        ["def fact(n):\n    return 1 if n == 0 else n * fact(n - 1)", "import json\ndef load(p):\n    return json.load(open(p))"]
    )
    sims = q @ d.T
    assert sims[0, 0] > sims[0, 1] and sims[1, 1] > sims[1, 0]


def test_local_fingerprint_tracks_weights(tiny_bert_dir, tmp_path):
    import shutil

    from safetensors.torch import load_file, save_file

    copy = tmp_path / "ckpt"
    shutil.copytree(tiny_bert_dir, copy)
    cfg = ModelConfig(name="t", adapter="hf", repo_id="local/t", local_dir=str(copy), device="cpu")
    before = load_model(cfg).fingerprint
    weights = load_file(copy / "model.safetensors")
    name = next(iter(weights))
    weights[name] = weights[name] + 1
    save_file(weights, copy / "model.safetensors")
    assert load_model(cfg).fingerprint != before
