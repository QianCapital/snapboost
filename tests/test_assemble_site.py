"""Tests for resilient versioned documentation assembly."""

import importlib.util
from pathlib import Path
import urllib.error


SCRIPT = Path(__file__).parents[1] / "docs" / "_scripts" / "assemble_site.py"


def _load_assembler(monkeypatch, tmp_path):
    monkeypatch.setenv("DOCS_VERSION", "latest")
    monkeypatch.setenv("GITHUB_WORKSPACE", str(tmp_path))
    spec = importlib.util.spec_from_file_location("assemble_site", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repair_version_assets_prefers_live_version(monkeypatch, tmp_path):
    module = _load_assembler(monkeypatch, tmp_path)
    version_dir = module.SITE / "v1.2.3"

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return b"version asset"

    requested = []

    def fake_urlopen(url, timeout):
        requested.append((url, timeout))
        return Response()

    monkeypatch.setattr(module.urllib.request, "urlopen", fake_urlopen)
    module.repair_version_assets(version_dir)

    for rel in module.REQUIRED_STATIC:
        assert (version_dir / rel).read_bytes() == b"version asset"
    assert all("/v1.2.3/_static/" in url for url, _ in requested)


def test_repair_version_assets_falls_back_to_local_build(monkeypatch, tmp_path):
    module = _load_assembler(monkeypatch, tmp_path)
    version_dir = module.SITE / "v1.2.3"
    for rel in module.REQUIRED_STATIC:
        source = module.BUILD / rel
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(f"local:{rel}".encode())

    def unavailable(*args, **kwargs):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(module.urllib.request, "urlopen", unavailable)
    module.repair_version_assets(version_dir)
    module.assert_version_assets(version_dir)

    for rel in module.REQUIRED_STATIC:
        assert (version_dir / rel).read_bytes() == f"local:{rel}".encode()
