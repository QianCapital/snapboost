#!/usr/bin/env python3
"""Assemble the GitHub Pages site tree for SnapBoost docs.

Preserves existing /v*/ trees from the live site, overlays the new Sphinx
build (root for latest, /vX.Y.Z/ for tagged/snapshot builds), and writes
versions.json.
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
VERSION_RE = re.compile(r"^v\d+\.\d+\.\d+$")


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
    """Mirror url into dest_parent, preserving the URL path under dest_parent."""
    dest_parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "wget",
        "--quiet",
        "--recursive",
        "--no-parent",
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
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copytree(BUILD, dest, dirs_exist_ok=True)


def merge_local_snapshots() -> None:
    """Overlay locally built snapshots from docs/_build/snapshots/vX.Y.Z/."""
    snapshots = WORKSPACE / "docs" / "_build" / "snapshots"
    if not snapshots.is_dir():
        return
    for path in sorted(snapshots.iterdir()):
        if path.is_dir() and VERSION_RE.fullmatch(path.name):
            dest = SITE / path.name
            if dest.exists():
                shutil.rmtree(dest)
            print(f"Installing local snapshot /{path.name}/ …")
            shutil.copytree(path, dest)


def main() -> int:
    if not BUILD.is_dir():
        print(f"Build output missing: {BUILD}", file=sys.stderr)
        return 1

    if SITE.exists():
        shutil.rmtree(SITE)
    SITE.mkdir(parents=True)

    existing_versions: list[str] = []
    if live_reachable():
        live = fetch_json(SITE_URL + "/versions.json") or []
        existing_versions = [
            e["version"]
            for e in live
            if isinstance(e, dict) and VERSION_RE.fullmatch(str(e.get("version", "")))
        ]
        print("Live versions:", existing_versions)

        if DOCS_VERSION == "latest":
            for ver in existing_versions:
                print(f"Preserving /{ver}/ …")
                wget_tree(f"{SITE_URL}/{ver}/", SITE)
        else:
            print("Preserving root docs …")
            # Fetch root pages only: wget follows links, so pull then drop other trees
            # we will re-fetch explicitly.
            wget_tree(f"{SITE_URL}/", SITE)
            for child in list(SITE.iterdir()):
                if child.is_dir() and VERSION_RE.fullmatch(child.name):
                    shutil.rmtree(child)
            for ver in existing_versions:
                if ver == DOCS_VERSION:
                    continue
                print(f"Preserving /{ver}/ …")
                wget_tree(f"{SITE_URL}/{ver}/", SITE)
    else:
        print("Live site unavailable; starting from empty site tree.")

    if DOCS_VERSION == "latest":
        clear_root(SITE)
        copy_build(SITE)
    else:
        target = SITE / DOCS_VERSION
        if target.exists():
            shutil.rmtree(target)
        copy_build(target)
        if not (SITE / "index.html").is_file():
            copy_build(SITE)

    # Local bootstrapped snapshots win over preserved remote copies.
    merge_local_snapshots()

    versions_path = write_versions_json(SITE)
    (SITE / ".nojekyll").touch()
    print(f"Site assembled at {SITE} (versions: {versions_path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
