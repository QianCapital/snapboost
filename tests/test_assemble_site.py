import importlib.util
from pathlib import Path


def _load_assemble_site_module(tmp_path, monkeypatch):
    monkeypatch.setenv("DOCS_VERSION", "v0.2.0")
    monkeypatch.setenv("GITHUB_WORKSPACE", str(tmp_path))
    module_path = (
        Path(__file__).resolve().parents[1] / "docs" / "_scripts" / "assemble_site.py"
    )
    spec = importlib.util.spec_from_file_location("assemble_site", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_backfill_version_assets_copies_missing_required_assets(tmp_path, monkeypatch):
    module = _load_assemble_site_module(tmp_path, monkeypatch)

    root = tmp_path / "site"
    version_dir = root / "v0.1.7"
    for rel in module.REQUIRED_STATIC:
        source = root / rel
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("asset", encoding="utf-8")

    module.backfill_version_assets(version_dir, root)
    module.assert_version_assets(version_dir)
