# Contributing

PRs welcome. For anything beyond a typo, please open an issue first so we
can scope the change before code lands.

## Ground rules

- Tests must pass on the CI matrix (Python 3.10, 3.11, 3.12).
- No new metrics or algorithms without a referenced source. EquiML wraps
  fairlearn, scikit-learn, and SHAP — additions need to live behind one of
  those (or have a peer-reviewed reference).
- Match the existing voice in docs: factual, plainspoken, no marketing
  language.

## Development setup

```bash
git clone https://github.com/mkupermann/EquiML.git
cd EquiML
pip install -e ".[dev]"
pytest tests/ -v
```

## Releasing

1. Update `CHANGELOG.md` (move `Unreleased` to a dated version block).
2. Bump version in `pyproject.toml` and `equiml/__init__.py`.
3. `git tag v1.0.x && git push --tags`.
4. `python -m build && twine upload dist/*`.

The repo isn't on PyPI yet — that's a maintainer decision and not a
release-blocker for the cleanup branch.

## Lint policy

CI currently runs flake8 with a syntax-error gate (`E9,F63,F7,F82`).
We are moving toward strict lint (`--max-line-length=120
--extend-ignore=E203,W503`) — see the `lint-strict` job in
`.github/workflows/ci.yml` for the staging gate.
