"""Tests for resilient versioned documentation assembly."""

import importlib.util
import urllib.error
from pathlib import Path

import pytest

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


def test_repair_root_assets_backfills_from_local_build(monkeypatch, tmp_path):
    """A tagged build mirrors the root with wget, which can drop stylesheets."""
    module = _load_assembler(monkeypatch, tmp_path)
    for rel in module.REQUIRED_STATIC:
        source = module.BUILD / rel
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(f"build:{rel}".encode())
    module.SITE.mkdir(parents=True, exist_ok=True)

    module.repair_root_assets()
    module.assert_version_assets(module.SITE, "Site root")

    for rel in module.REQUIRED_STATIC:
        assert (module.SITE / rel).read_bytes() == f"build:{rel}".encode()


def test_repair_root_assets_keeps_existing_files(monkeypatch, tmp_path):
    module = _load_assembler(monkeypatch, tmp_path)
    for rel in module.REQUIRED_STATIC:
        mirrored = module.SITE / rel
        mirrored.parent.mkdir(parents=True, exist_ok=True)
        mirrored.write_bytes(b"mirrored")
        source = module.BUILD / rel
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"build")

    module.repair_root_assets()

    for rel in module.REQUIRED_STATIC:
        assert (module.SITE / rel).read_bytes() == b"mirrored"


def test_copy_build_into_root_preserves_version_trees(monkeypatch, tmp_path):
    module = _load_assembler(monkeypatch, tmp_path)
    (module.BUILD).mkdir(parents=True, exist_ok=True)
    (module.BUILD / "index.html").write_bytes(b"fresh root")
    preserved = module.SITE / "v0.1.7"
    preserved.mkdir(parents=True, exist_ok=True)
    (preserved / "index.html").write_bytes(b"old version")
    stale = module.SITE / "index.html"
    stale.write_bytes(b"stale root")

    module.copy_build_into_root()

    assert (preserved / "index.html").read_bytes() == b"old version"
    assert stale.read_bytes() == b"fresh root"


def test_assert_version_assets_reports_missing_root(monkeypatch, tmp_path):
    module = _load_assembler(monkeypatch, tmp_path)
    module.SITE.mkdir(parents=True, exist_ok=True)

    with pytest.raises(SystemExit, match="Site root is missing"):
        module.assert_version_assets(module.SITE, "Site root")
