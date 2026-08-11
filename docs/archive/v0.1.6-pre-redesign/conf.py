# Configuration file for the Sphinx documentation builder.
# Archived SnapBoost docs (pre-redesign), version 0.1.6.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath("../../.."))

project = "SnapBoost"
author = "Qian Capital"
copyright = f"{datetime.now().year}, Qian Capital"
release = "0.1.6"
version = "0.1.6"

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
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["css/qiancapital.css"]
html_js_files = ["js/qiancapital.js"]
html_logo = "_static/img/qian-capital-logo.png"
html_favicon = "_static/img/qian-capital-mark.png"
html_title = "SnapBoost Documentation (archive)"
html_short_title = "SnapBoost"
html_show_sphinx = False
html_show_sourcelink = False
html_copy_source = False

html_theme_options = {
    "logo_only": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": True,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 3,
    "includehidden": True,
    "titles_only": False,
    "style_nav_header_background": "#1f271b",
}

html_context = {
    "display_github": True,
    "github_user": "qiancapital",
    "github_repo": "snapboost",
    "github_version": "main",
    "conf_py_path": "/docs/archive/v0.1.6-pre-redesign/",
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
