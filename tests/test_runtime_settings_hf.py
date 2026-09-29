"""HuggingFace weight-source resolution in runtime settings (no network, no weights)."""
from types import SimpleNamespace

import pytest

from tcr_workbench import runtime_settings as rs


def _args(**overrides):
    base = dict(config=None, decoder_dir="/decoder", python_executable="/decoder/bin/python",
                model="esmc-300m", device="cpu", precision=None, checkpoint=None, mlx_python=None,
                batch_size=None, token_budget=None, cache_bytes=None, timeout=None,
                weight_source="registry", hf_repo=None, hf_revision=None)
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    # Do not pick up a real ./.tcr-workbench.json from the working directory.
    monkeypatch.chdir(tmp_path)


def test_registry_is_the_default_weight_source():
    options = rs.resolve_settings(_args())
    assert options["weight_source"] == "registry"
    assert options["hf_repo"] is None and options["hf_revision"] is None


def test_huggingface_source_resolves_the_published_repo():
    options = rs.resolve_settings(_args(weight_source="huggingface"))
    assert options["weight_source"] == "huggingface"
    # The explicit repo override is preserved verbatim.
    pinned = rs.resolve_settings(_args(weight_source="huggingface", hf_repo="me/custom", hf_revision="abc"))
    assert pinned["hf_repo"] == "me/custom" and pinned["hf_revision"] == "abc"


def test_huggingface_source_rejected_on_apple():
    with pytest.raises(ValueError, match="cpu or cuda only"):
        rs.resolve_settings(_args(weight_source="huggingface", device="apple", mlx_python="/mlx/bin/python"))


def test_huggingface_source_requires_a_published_or_explicit_repo():
    with pytest.raises(ValueError, match="no HuggingFace source"):
        rs.resolve_settings(_args(weight_source="huggingface", model="esmc-600m"))


def test_unknown_weight_source_fails_closed():
    # The RuntimeSettings Literal rejects an unknown source before it is used.
    with pytest.raises(ValueError, match="registry.*huggingface"):
        rs.resolve_settings(_args(weight_source="s3"))
