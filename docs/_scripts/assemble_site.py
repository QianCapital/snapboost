#!/usr/bin/env python3
"""Assemble the GitHub Pages site tree for SnapBoost docs.

Preserves existing /v*/ trees from the live site when needed, overlays the
new Sphinx build, merges locally rebuilt snapshots, and writes versions.json.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

SITE_URL = os.environ.get("SITE_URL", "https://snapboost.qiancapital.com").rstrip("/")
DOCS_VERSION = os.environ["DOCS_VERSION"]
WORKSPACE = Path(os.environ["GITHUB_WORKSPACE"])
SITE = WORKSPACE / "site"
BUILD = WORKSPACE / "docs" / "_build" / "html"
SNAPSHOTS = WORKSPACE / "docs" / "_build" / "snapshots"
VERSION_RE = re.compile(r"^v\d+\.\d+\.\d+$")
REQUIRED_STATIC = [
    "_static/css/theme.css",
    "_static/css/qiancapital.css",
    "_static/js/theme.js",
]


def fetch_json(url: str):
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"Could not fetch {url}: {exc}")
        return None


def live_reachable() -> bool:
    try:
        req = urllib.request.Request(SITE_URL + "/", method="HEAD")
        with urllib.request.urlopen(req, timeout=30):
            return True
    except Exception:
        try:
            with urllib.request.urlopen(SITE_URL + "/", timeout=30):
                return True
        except Exception as exc:
            print(f"Live site unreachable: {exc}")
            return False


def wget_tree(url: str, dest_parent: Path) -> None:
    """Mirror a version tree, including CSS/JS page requisites."""
    dest_parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "wget",
        "--quiet",
        "--recursive",
        "--no-parent",
        "--page-requisites",
        "--no-host-directories",
        "--directory-prefix",
        str(dest_parent),
        "--reject",
        "genindex.html,search.html,searchindex.js,objects.inv",
        url,
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=False)


def list_version_dirs(site: Path) -> list[str]:
    return sorted(
        (p.name for p in site.iterdir() if p.is_dir() and VERSION_RE.fullmatch(p.name)),
        key=lambda v: tuple(int(x) for x in v[1:].split(".")),
        reverse=True,
    )


def local_snapshot_versions() -> set[str]:
    if not SNAPSHOTS.is_dir():
        return set()
    return {
        p.name
        for p in SNAPSHOTS.iterdir()
        if p.is_dir() and VERSION_RE.fullmatch(p.name)
    }


def write_versions_json(site: Path) -> Path:
    versions = [{"version": "latest", "url": "/"}]
    for name in list_version_dirs(site):
        versions.append({"version": name, "url": f"/{name}/"})
    if DOCS_VERSION != "latest" and not any(v["version"] == DOCS_VERSION for v in versions):
        versions.append({"version": DOCS_VERSION, "url": f"/{DOCS_VERSION}/"})
        versions[1:] = sorted(
            versions[1:],
            key=lambda e: tuple(int(x) for x in e["version"][1:].split(".")),
            reverse=True,
        )
    out = site / "versions.json"
    out.write_text(json.dumps(versions, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}:\n{out.read_text(encoding='utf-8')}")
    return out


def clear_root(site: Path) -> None:
    for child in list(site.iterdir()):
        if child.name.startswith("v") and VERSION_RE.fullmatch(child.name):
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def copy_build(dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(BUILD, dest)


def merge_local_snapshots() -> None:
    """Overlay locally built snapshots from docs/_build/snapshots/vX.Y.Z/."""
    if not SNAPSHOTS.is_dir():
        return
    for path in sorted(SNAPSHOTS.iterdir()):
        if path.is_dir() and VERSION_RE.fullmatch(path.name):
            dest = SITE / path.name
            if dest.exists():
                shutil.rmtree(dest)
            print(f"Installing local snapshot /{path.name}/ …")
            shutil.copytree(path, dest)
            (dest / ".nojekyll").touch()


def assert_version_assets(version_dir: Path) -> None:
    missing = [rel for rel in REQUIRED_STATIC if not (version_dir / rel).is_file()]
    if missing:
        raise SystemExit(
            f"Snapshot {version_dir.name} is missing required static assets: {missing}"
        )


def backfill_version_assets(version_dir: Path, root_dir: Path) -> None:
    for rel in REQUIRED_STATIC:
        dest = version_dir / rel
        if dest.is_file():
            continue
        source = root_dir / rel
        if not source.is_file():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        print(f"Backfilled {version_dir.name}/{rel} from root docs assets.")


def main() -> int:
    if not BUILD.is_dir():
        print(f"Build output missing: {BUILD}", file=sys.stderr)
        return 1

    if SITE.exists():
        shutil.rmtree(SITE)
    SITE.mkdir(parents=True)

    rebuilt = local_snapshot_versions()
    existing_versions: list[str] = []
    if live_reachable():
        live = fetch_json(SITE_URL + "/versions.json") or []
        existing_versions = [
            e["version"]
            for e in live
            if isinstance(e, dict) and VERSION_RE.fullmatch(str(e.get("version", "")))
        ]
        print("Live versions:", existing_versions)
        print("Locally rebuilt snapshots:", sorted(rebuilt))

        if DOCS_VERSION == "latest":
            for ver in existing_versions:
                if ver in rebuilt:
                    print(f"Skipping live preserve for /{ver}/ (local rebuild present)")
                    continue
                print(f"Preserving /{ver}/ …")
                wget_tree(f"{SITE_URL}/{ver}/", SITE)
        else:
            print("Preserving root docs …")
            wget_tree(f"{SITE_URL}/", SITE)
            for child in list(SITE.iterdir()):
                if child.is_dir() and VERSION_RE.fullmatch(child.name):
                    shutil.rmtree(child)
            for ver in existing_versions:
                if ver == DOCS_VERSION or ver in rebuilt:
                    continue
                print(f"Preserving /{ver}/ …")
                wget_tree(f"{SITE_URL}/{ver}/", SITE)
    else:
        print("Live site unavailable; starting from empty site tree.")

    if DOCS_VERSION == "latest":
        clear_root(SITE)
        copy_build(SITE)
    else:
        copy_build(SITE / DOCS_VERSION)
        (SITE / DOCS_VERSION / ".nojekyll").touch()
        if not (SITE / "index.html").is_file():
            copy_build(SITE)

    merge_local_snapshots()

    for ver in list_version_dirs(SITE):
        backfill_version_assets(SITE / ver, SITE)
        assert_version_assets(SITE / ver)

    if DOCS_VERSION == "latest":
        assert_version_assets(SITE)

    write_versions_json(SITE)
    (SITE / ".nojekyll").touch()
    print(f"Site assembled at {SITE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
