# Archived docs (pre-redesign)

SnapBoost docs snapshot from before the August 2026 cleanup.

- **Version:** 0.1.6
- **Style:** branded hero, full Qian Capital CSS (serif fonts, chips, animations)

## Build

From this directory:

```bash
make html
# open _build/html/index.html
```

Or from `docs/`:

```bash
make archive
```

Shared pages (`installation`, `quickstart`, `api`, `parameters`) and figure/image assets are symlinked from the parent `docs/` tree. Output stays in `_build/` here and is not part of the main docs site.
