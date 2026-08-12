# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.abspath(".."))


def _package_version() -> str:
    """Read snapboost.__version__ without importing package dependencies."""
    init = Path(__file__).resolve().parent.parent / "snapboost" / "__init__.py"
    for line in init.read_text(encoding="utf-8").splitlines():
        if line.startswith("__version__"):
            return line.split("=", 1)[1].strip().strip("\"'")
    return "0.0.0"


project = "SnapBoost"
author = "Qian Capital"
copyright = f"{datetime.now().year}, Qian Capital"
release = _package_version()
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "tasklist",
    "html_image",
]
suppress_warnings = ["myst.xref_missing"]


source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "archive", "README.md"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["css/qiancapital.css"]
html_js_files = ["js/qiancapital.js"]
html_logo = "_static/img/qian-capital-logo.png"
html_favicon = "_static/img/qian-capital-mark.png"
html_title = "SnapBoost Documentation"
html_short_title = "SnapBoost"
html_baseurl = "https://snapboost.qiancapital.com/"
html_show_sphinx = False
html_show_sourcelink = False
html_copy_source = False

html_theme_options = {
    "logo_only": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": True,
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 2,
    "includehidden": True,
    "titles_only": False,
    "style_nav_header_background": "#1f271b",
}


def _load_versions():
    """Load version switcher entries from versions.json next to conf.py."""
    path = Path(__file__).resolve().parent / "versions.json"
    if not path.is_file():
        return [("latest", "/")]
    data = json.loads(path.read_text(encoding="utf-8"))
    return [(entry["version"], entry["url"]) for entry in data]


_docs_version = os.environ.get("DOCS_VERSION", "latest")

html_context = {
    "display_github": True,
    "github_user": "qiancapital",
    "github_repo": "snapboost",
    "github_version": "master",
    "conf_py_path": "/docs/",
    "display_lower_left": False,
    "current_version": _docs_version,
    "version": _docs_version,
    "versions": _load_versions(),
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
