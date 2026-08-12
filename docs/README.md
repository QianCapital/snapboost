# SnapBoost documentation

This directory contains the source, configuration, assets, and generated files
for the SnapBoost documentation site. The site is built with
[Sphinx](https://www.sphinx-doc.org/), uses
[MyST Parser](https://myst-parser.readthedocs.io/) for Markdown, and is styled
with the Read the Docs theme plus project-specific assets.

The published documentation starts at [`index.md`](index.md). This README is a
maintainer guide and is not included in the rendered site.

## Documentation pages

| File | Purpose |
| --- | --- |
| [`index.md`](index.md) | Landing page, project overview, and Sphinx table of contents. |
| [`installation.md`](installation.md) | PyPI, source, Docker, development, and documentation build instructions. |
| [`quickstart.md`](quickstart.md) | Classification and regression examples, training strategies, objectives, and preprocessing. |
| [`api.md`](api.md) | Public estimator, learner, preprocessing, and HNBM API reference. |
| [`parameters.md`](parameters.md) | Shared and SnapBoost-specific parameter reference. |
| [`examples.md`](examples.md) | Notebook links, benchmark results, and generated example figures. |
| [`references.md`](references.md) | Citation information, research links, and related projects. |
| [`license.md`](license.md) | License page, populated from the repository-level `LICENSE` file. |

The order of pages in the published navigation is controlled by the `toctree`
in [`index.md`](index.md).

## Supporting files and directories

| Path | Purpose |
| --- | --- |
| [`conf.py`](conf.py) | Sphinx configuration, extensions, theme settings, project metadata, and version-switcher context. |
| [`Makefile`](Makefile) | Convenience targets for Sphinx builds, such as `html`, `clean`, and `archive`. |
| [`requirements.txt`](requirements.txt) | Python dependencies required to build the documentation. |
| [`versions.json`](versions.json) | Versions displayed by the documentation version switcher. |
| [`_static/`](_static/) | Custom CSS, JavaScript, logos, and figures copied into the built site. |
| [`_templates/`](_templates/) | Sphinx HTML template overrides for layout, footer, and version selection. |
| [`_scripts/assemble_site.py`](_scripts/assemble_site.py) | Deployment helper that assembles the latest and versioned documentation trees. |
| [`archive/`](archive/) | Preserved pre-redesign documentation source and build for version 0.1.6. |
| [`_build/`](_build/) | Generated Sphinx output and intermediate files. Do not edit these files by hand. |

## Directory overview

```text
docs/
├── README.md                 # This maintainer guide
├── index.md                  # Documentation landing page and navigation
├── installation.md           # Installation instructions
├── quickstart.md             # Getting-started examples
├── api.md                    # API reference
├── parameters.md             # Parameter reference
├── examples.md               # Examples and benchmark figures
├── references.md             # Citations and related work
├── license.md                # License page
├── conf.py                   # Sphinx configuration
├── Makefile                  # Documentation build commands
├── requirements.txt          # Documentation dependencies
├── versions.json             # Published-version metadata
├── _static/                  # Site assets
├── _templates/               # HTML template overrides
├── _scripts/                 # Site assembly tooling
├── archive/                  # Historical documentation snapshot
└── _build/                   # Generated documentation output
```

## Build the documentation

From the repository root, install the documentation dependencies and build the
HTML site:

```bash
python -m pip install -r docs/requirements.txt
make -C docs html
```

The generated site is written to `docs/_build/html/`. Open
`docs/_build/html/index.html` in a browser to inspect it locally.

To remove generated output before rebuilding:

```bash
make -C docs clean
make -C docs html
```

To build the archived pre-redesign documentation separately:

```bash
make -C docs archive
```

## Updating the documentation

1. Edit the relevant Markdown source page.
2. Add new pages to the `toctree` in [`index.md`](index.md) so they appear in
   site navigation.
3. Put images under `_static/figures/`, branding under `_static/img/`, and
   custom styles or scripts under the corresponding `_static/` subdirectory.
4. Run `make -C docs html` and review the generated site.
5. Treat `_build/` as generated output; make changes in the source files,
   templates, or static assets instead.

When adding or removing published versions, keep [`versions.json`](versions.json)
and the deployment workflow in sync. The site assembly script also discovers
versioned directories and writes the final version metadata during deployment.
