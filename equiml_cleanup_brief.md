# EquiML cleanup — Claude Code brief

You are Claude Code, working on the **EquiML** repo at `mkupermann/EquiML`. Michael ran a stakeholder review on the repo (nine personas, code-level audit) and produced a prioritised action list. This brief turns that list into a parallelised cleanup using **Teammates** — four independent subagents fan out, each owning a track, then a final synthesis pass integrates and verifies.

The point of the parallelism is not speed alone — it's that each Teammate has its own narrow brief and cannot drift into another's lane. A single agent doing everything would re-litigate decisions across tracks. Four agents, hard file boundaries, then one integrator.

---

## How to run this

```bash
# Clone fresh, branch, then hand this file to Claude Code
git clone https://github.com/mkupermann/EquiML.git
cd EquiML
git checkout -b cleanup/post-review

# In Claude Code:
#   /clear
#   Read this brief, then dispatch four Teammates per the plan below.
#   Wait for all four. Then run the synthesis pass yourself.
```

If you prefer, save this file as `.claude/commands/equiml-cleanup.md` and invoke as `/equiml-cleanup`.

---

## Pre-flight (do this before dispatching any Teammates)

1. Confirm you're on a clean branch off `main`.
2. Run the existing suite to capture the **current** baseline:
   ```bash
   pip install -e ".[dev]"
   pytest tests/ -v
   ```
   Save the output. The synthesis pass will compare against it.
3. Read `equiml/cli.py`, `equiml/model.py`, `equiml/data.py`, and `tests/test_audit_pipeline.py` end-to-end before dispatching. You need to know the lay of the land before you delegate, because two of the Teammates will touch overlapping files and you'll need to mediate the merge.

---

## File ownership map (binding for all Teammates)

To prevent merge conflicts, each Teammate owns specific files. **A Teammate must not edit a file outside its lane.** If a Teammate believes another file needs editing, it returns a note in its handoff for the synthesis pass.

| Track | Owns | Read-only |
|---|---|---|
| 1. Code Surgeon | `equiml/cli.py`, `equiml/model.py`, `equiml/data.py` | tests, README, packaging |
| 2. README & Positioning | `README.md`, `equiml/report_template.html`, `equiml/reporting.py` | code modules |
| 3. Packaging & Procurement | `pyproject.toml`, `setup.py`, `requirements.txt`, `.github/`, **new** `SECURITY.md`, **new** `CHANGELOG.md`, **new** `examples/` | code modules |
| 4. Tests & Verification | `tests/`, `conftest.py` | (read everything to write tests against it) |

---

## Teammate 1 — Code Surgeon

**Role.** Senior Python engineer. Sharp, conservative, will not introduce features. Fixes correctness bugs and removes dead code paths. Does **not** touch documentation, packaging, or tests (Teammate 4 owns tests).

**Files allowed to edit:** `equiml/cli.py`, `equiml/model.py`, `equiml/data.py`.

### Tasks

1. **Fix `predict_proba` for the fairness-mitigated model** (`equiml/model.py:72-80`). The current code averages predictor probabilities across `self.model.predictors_`, which is incorrect for `fairlearn.reductions.ExponentiatedGradient` — it produces a *randomised* classifier and the correct semantics are to sample a predictor according to the learned weights `_weights`, not arithmetic-mean them. Two acceptable resolutions:
   - **Preferred:** sample correctly. Use `self.model._weights` to draw one predictor per call (or per-row), then call its `predict_proba`.
   - **Acceptable fallback:** raise `NotImplementedError("predict_proba is not well-defined for the ExponentiatedGradient mitigator; use predict() and report fairness metrics from hard predictions only")` and update any internal call sites that rely on it.
   Pick one, document the choice in a code comment, and make sure the audit pipeline still runs end-to-end.

2. **Fix `cross_validate` semantics** (`equiml/model.py:82-86`). The current code falls back to `self.model.estimator` when the model is a mitigator, silently cross-validating the unconstrained baseline while the user thinks they're CV-ing the fair model. Either (a) refit the mitigator per fold honestly with sensitive features threaded through, or (b) raise a clear error when called on a mitigator. **Do not silently strip the constraint.**

3. **Fix the multi-sensitive feature bug in the CLI** (`equiml/cli.py:81-88`). The README's headline example is `--sensitive gender race`, but the code only audits the first column (`sensitive_train_cols[0]`). Two options:
   - **Preferred:** iterate the audit over each sensitive feature, and produce a per-feature section in the metrics dict (e.g., `metrics["per_sensitive"]["gender"] = {...}`, `metrics["per_sensitive"]["race"] = {...}`). The fairlearn constraint training still runs once with the *first* sensitive feature (or with a combined column) — document the choice in a comment.
   - **Acceptable fallback:** restrict the CLI to a single `--sensitive` argument (`nargs=1`) and update the help text. Then the README must change to match (Teammate 2's job, but flag it in handoff).

4. **Replace the substring match for encoded sensitive columns** (`equiml/cli.py:82-85`). The line `if any(sf in col for sf in sensitive_cols)` matches `"race"` against `"racing_score"`. Use exact-prefix match against the one-hot expansion convention scikit-learn uses (`f"{sf}_"` prefix or `col == sf`), or — better — track the encoded column names explicitly during `Data.preprocess()` and read them back here.

5. **Delete dead code paths in `equiml/data.py`.** The README says *"Tabular classification only. No text, image, or time-series."* The module retains:
   - NLTK setup and `_clean_text` (lines ~39-55, ~213-235)
   - OpenCV `_process_images` (lines ~237-262)
   - ARFF loading branch
   - Polynomial / interaction features (`feature_engineering`)
   - PCA / `reduce_dimensionality`
   - SMOTE / oversample / undersample / class-weight machinery in `handle_class_imbalance` and helpers
   
   Remove these. Also remove the `text_features` and `image_features` parameters from `Data.__init__` and the corresponding optional imports (`arff`, `nltk`, `cv2`). Keep `mitigate_bias` / `_apply_reweighing` (those are referenced by the test suite) and `_apply_correlation_removal` if it's reachable from `apply_bias_mitigation`. **Run `pytest tests/ -v` after this step** — anything you broke is real coverage, not dead code.

6. **Caveat the bias threshold banner** (`equiml/cli.py:158-167`). Replace the categorical "LOW BIAS / MODERATE / SIGNIFICANT / HIGH" with a numeric reading plus a one-line caveat:
   ```
   print(f"\n  Maximum group disparity: {max_bias:.3f}")
   print("  Note: thresholds for 'acceptable' bias are domain- and")
   print("  jurisdiction-specific. This number is a starting point for")
   print("  review, not a regulatory verdict.")
   ```
   Do not invent thresholds. The Compliance Officer persona was loudest on this; it's a real harm path.

7. **Add a version stamp to the metrics dict** so an audit artefact is reproducible. In `cmd_audit`, before saving JSON, attach:
   ```python
   import platform, sklearn, fairlearn
   from equiml import __version__ as equiml_version
   serializable["_meta"] = {
       "equiml_version": equiml_version,
       "python_version": platform.python_version(),
       "sklearn_version": sklearn.__version__,
       "fairlearn_version": fairlearn.__version__,
       "random_seed": 42,
       "dataset_path": dataset_path,
       "target": target,
       "sensitive_features": sensitive_cols,
       "algorithm": algorithm,
   }
   ```

### Don't

- Don't add features. No new metrics, no new algorithms, no new CLI flags beyond what's required by the bug fixes above.
- Don't reformat unrelated code. No "while I'm here" cleanup.
- Don't touch `tests/`, `README.md`, or `pyproject.toml`.

### Done when

- All seven tasks done with explanatory commit messages.
- `pytest tests/ -v` passes (or you've documented in handoff exactly which test broke and why — Teammate 4 will update tests in the synthesis pass).
- Handoff note lists: each task with the file/line touched, any test breakage, and any cross-cutting concern flagged for synthesis.

---

## Teammate 2 — README & Positioning

**Role.** Technical writer with consulting brand awareness. Will not invent claims. Demotes performative language. Knows that the author runs `kupermann.com` and has companion Medium pieces on Cortex HDC and FIRE Score.

**Files allowed to edit:** `README.md`, `equiml/report_template.html`, `equiml/reporting.py`.

### Tasks

1. **Demote the GitHub "About" tagline.** Currently *"A Framework for Equitable and Responsible Machine Learning & Validation of AI"*. Replace with: *"A focused CLI for fairness audits on tabular ML — wraps fairlearn, scikit-learn, and SHAP."* Note: the GitHub repo description is set in the GitHub UI, not in the README — leave a comment at the top of the README's H1 block reminding the maintainer to update it manually, or add a `.github/repo-description.txt` with the new text.

2. **Add a "How EquiML compares to fairlearn / aif360 / Aequitas" section** after the "What it wraps" table. Be honest, specific, no benchmarks unless you can produce them:
   - **fairlearn**: the underlying library. EquiML's job is to be a CLI you can pipe into CI; fairlearn is the toolbox you build with.
   - **aif360** (IBM): broader algorithm catalogue, heavier install, more academic. Choose aif360 if you need pre/in/post-processing variety beyond ExponentiatedGradient + reweighing.
   - **Aequitas** (CMU/DSSG): policy-audit-shaped, group-disparity dashboards, often the right pick for public-sector audits.
   - **EquiML**: opinionated three-command pipeline, single-author project, built for a specific audit cadence — not a research toolkit.
   
   End the section with one line: *"If you don't already know which of these you need, you probably need fairlearn or Aequitas, not EquiML."* That's honest and it's good positioning.

3. **Add a "Where this fits in your governance framework" section.** Map the audit artefacts to control vocabulary the buyer already uses:
   - The JSON metrics output → evidence for **EU AI Act Art. 9 (risk management)** and **Art. 15 (accuracy and robustness)** documentation.
   - The HTML report → input to **ISO/IEC 42001** AI-management-system review cycles.
   - Note that the tool itself is **not a Conformity Assessment**. It produces evidence; the assessment is a human and legal exercise.
   - Reference the **NIST AI Risk Management Framework** (Govern / Map / Measure / Manage) — the audit fits the Measure function.
   
   Keep it three short paragraphs, not a treatise.

4. **Add a "What this tool does not do" section.** Explicit list:
   - Not a legal opinion or regulatory verdict.
   - Not jurisdiction-specific — fairness definitions vary by domain and by law.
   - No intersectional fairness analysis (yet) — bias at the cross-product of two protected attributes is not assessed.
   - Assumes train/test are i.i.d. — not appropriate when distributions shift.
   - Tabular classification only. No text, image, time-series, or LLM evaluation.

5. **Update or remove the bias-assessment banner description** in the README's "What it does" section. If Teammate 1 has caveated the CLI banner (it should have), update the README to match — no claim that the tool tells you whether your model is "fair," only that it surfaces metrics.

6. **Cross-link to the author's body of work.** Add a short footer:
   ```
   ## About the author
   
   Built by Michael Kupermann at [Kupermann Consulting](https://kupermann.com).
   Companion writing on Medium: [Cortex HDC](URL_HERE), [FIRE Score](URL_HERE).
   Issues, ideas, and collaboration enquiries welcome.
   ```
   Leave `URL_HERE` placeholders — the maintainer will fill them. The headhunter persona convergence was that the repo is an island; this fixes it.

7. **Add a "Why I built this" paragraph** near the top of the README, in first person, two sentences max. Don't invent motivation — leave a structured placeholder for the maintainer:
   ```
   <!-- TODO(maintainer): two sentences in your own voice. Why fairness, why now,
        what client pattern made you tired of rebuilding this glue. The Coach
        persona was firm: this paragraph either lands as conviction or it
        weakens the rest of the README. Don't ghostwrite it. -->
   ```

8. **Update `report_template.html` and `reporting.py`** if Teammate 1 fixed the `fair_metrics` plumbing in `cli.py` (they should have, in task 3 of Teammate 1's brief — actually no, they're not allowed to touch `reporting.py`, so this falls to you). Make `generate_html_report` accept both `metrics` and `fair_metrics`, and render the comparison side-by-side. The Bereichsleiter persona's whole adoption argument hinges on this: the report has to show what the README says it shows.

### Don't

- Don't invent benchmarks. If you don't have numbers, don't write a benchmarks section.
- Don't write the "Why I built this" paragraph yourself — leave a TODO comment for the maintainer with explicit instructions.
- Don't edit code modules. If `reporting.py` needs a signature change, do that work — it's in your lane — but don't modify `cli.py` to call it; flag that in handoff for synthesis.

### Done when

- All sections added or updated.
- The README reads end-to-end and the new sections fit the existing voice (utility-over-cleverness — match the existing tone, don't escalate).
- `report_template.html` renders both baseline and fair metrics if both are passed; renders just baseline if only one is passed (backward compatible).
- Handoff note lists every section added, any cross-cutting concern (e.g., "Teammate 1 needs to update `cli.py:128-131` to pass `fair_metrics` into `generate_html_report` — flagged for synthesis").

---

## Teammate 3 — Packaging & Procurement

**Role.** Release engineer with enterprise procurement instincts. Knows that what blocks a vendor onboarding check is rarely the code — it's the absence of the metadata an SBOM tool wants to read.

**Files allowed to edit:** `pyproject.toml`, `setup.py`, `requirements.txt`, `.github/`, plus **new** files: `SECURITY.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `examples/`.

### Tasks

1. **Add `SECURITY.md`** at repo root. Standard template:
   - How to report a vulnerability (email — check `pyproject.toml` author block; if no email is listed, use a placeholder `security@kupermann.com` and flag it).
   - Supported versions (just `1.0.x` for now).
   - Disclosure timeline (90 days, standard).
   - One sentence acknowledging the project is single-maintainer and response is best-effort.

2. **Add Dependabot config** at `.github/dependabot.yml`:
   ```yaml
   version: 2
   updates:
     - package-ecosystem: "pip"
       directory: "/"
       schedule:
         interval: "weekly"
       open-pull-requests-limit: 5
     - package-ecosystem: "github-actions"
       directory: "/"
       schedule:
         interval: "monthly"
   ```

3. **Add `CHANGELOG.md`** following Keep-a-Changelog format. Backfill from the git log:
   - **1.0.0 (2026-04-19, current main):** "Polish for public release."
   - **Unreleased:** seed it with the changes this cleanup branch will introduce — the bug fixes from Teammate 1, the README work from Teammate 2, the testing additions from Teammate 4. Concrete entries, not "various improvements."

4. **Tag a `1.0.1` plan in the CHANGELOG `Unreleased` section** describing what the synthesis pass will release. The Procurement persona was clear: a 1.0.0 with no roadmap is harder to ingest than a 0.x with one. Add a brief "Roadmap" section to the CHANGELOG covering the next two quarters at a high level — don't overcommit.

5. **Add a minimal `examples/` directory.** One file: `examples/adult_census_audit.ipynb` — or, if Jupyter is heavy, `examples/adult_census_audit.py` as a runnable script that:
   - downloads the UCI Adult dataset (or uses scikit-learn's `fetch_openml`),
   - runs `equiml audit` programmatically against it,
   - prints the metrics,
   - has comments explaining the output.
   
   This is the artefact a Bereichsleiter's data scientist actually opens. Do not skip.

6. **Add a `CONTRIBUTING.md`** — short. Single-author project, so just: "PRs welcome, please open an issue first for anything beyond a typo. Tests must pass on the CI matrix. No new metrics or algorithms without a referenced source." That last sentence is the OSS Maintainer persona's main concern — bake it into the contribution policy.

7. **Update `pyproject.toml`:**
   - Add `keywords = ["fairness", "ml", "audit", "fairlearn", "responsible-ai"]` to the `[project]` block.
   - Add a `Repository` URL to `[project.urls]` matching `Homepage`.
   - Add `[project.urls] Changelog = "https://github.com/mkupermann/EquiML/blob/main/CHANGELOG.md"`.
   - If the maintainer email is missing from the `authors` block, leave a TODO comment.

8. **Verify `requirements.txt` matches `pyproject.toml`** dependencies. If they drift, the simpler convention is: `requirements.txt` exists for compatibility but pins are managed in `pyproject.toml`. Either keep both in sync or delete `requirements.txt` and document `pip install -e .` as the only path. Pick one and document it.

9. **Prepare for PyPI** but **do not actually publish.** Add a section to `CONTRIBUTING.md` titled "Releasing" with the checklist:
   ```
   1. Update CHANGELOG.md (Unreleased → version block, dated)
   2. Bump version in pyproject.toml and equiml/__init__.py
   3. git tag v1.0.x && git push --tags
   4. python -m build && twine upload dist/*
   ```
   The repo isn't on PyPI yet — that's a maintainer decision, not a Teammate decision. Set it up so it's a 5-minute job when the maintainer is ready.

### Don't

- Don't publish to PyPI. Don't push tags. Don't trigger releases.
- Don't add new dependencies. The `examples/` script should run on what's already declared.
- Don't touch code modules or `tests/`.

### Done when

- All new files added with sensible content (no boilerplate-only files).
- `pyproject.toml` updated with keywords, URLs, and any TODOs flagged.
- `examples/` has one runnable end-to-end demo against a real dataset.
- Handoff note lists each new file, the rationale, and any maintainer-action TODOs left in place.

---

## Teammate 4 — Tests & Verification

**Role.** Test engineer. Skeptical, defensive, wants the safety harness on the safety tool. Will read the code Teammate 1 changes and write tests against the *new* contract, not the old one.

**Files allowed to edit:** `tests/`, `conftest.py`.

### Tasks

1. **Add a CLI smoke test** at `tests/test_cli.py`. Use `subprocess.run` to invoke `equiml audit` against a fixture CSV and assert:
   - exit code is 0,
   - JSON output is valid and contains `baseline`, `fair`, and `_meta` keys (the `_meta` block was added by Teammate 1, task 7),
   - HTML report file is produced and is non-empty,
   - the `_meta.equiml_version` matches `equiml.__version__`.
   
   Use the existing `adult_sample` fixture or a similar synthetic CSV written to a tmp_path.

2. **Add a multi-sensitive integration test.** Whichever resolution Teammate 1 picked for task 3 (multi-feature audit *or* single-feature restriction with updated docs), write the test that locks the contract in. If multi-feature: assert `metrics["per_sensitive"]["gender"]` and `metrics["per_sensitive"]["race"]` both exist with valid values. If restricted to single: assert that passing two sensitive features raises a clear error (or the CLI warns and uses the first, depending on Teammate 1's choice).

3. **Add a substring-collision regression test.** Build a fixture with a sensitive column named `race` and a non-sensitive column named `racing_score`. Run the audit. Assert the audit only treats `race` as sensitive — not `racing_score`. This locks Teammate 1's task 4.

4. **Add a `predict_proba` contract test.** Whichever resolution Teammate 1 picked for task 1 (correct sampling vs. `NotImplementedError`), test the contract:
   - **If sampling:** assert `predict_proba` returns shape `(n, 2)` and values in `[0, 1]` summing to 1 per row, on a mitigated model.
   - **If raises:** assert `pytest.raises(NotImplementedError)` is hit, and that `predict()` still works.

5. **Add a `cross_validate` contract test.** Either CV produces honest scores against the fair model, or it raises. Test the chosen contract.

6. **Add an "adversarial" test fixture** at `tests/test_adversarial.py`. One test, not a battery — start small. Build a dataset where the protected attribute is **perfectly correlated** with a non-protected proxy feature (e.g., zip code with race). The audit should still flag bias against the protected attribute even though the model could "launder" through the proxy. This is the Anthropic persona's request — the safety harness on the safety tool. Document this test as the start of an `adversarial/` test directory; future cases get added one at a time.

7. **Verify the dead-code removal.** After Teammate 1's task 5, run `pytest --collect-only` to make sure no test references a deleted method. If a test does (e.g., a test of `feature_engineering` or `_clean_text`), delete it — it was testing dead code.

8. **Update `.github/workflows/ci.yml` flake8 rule** — currently `--select=E9,F63,F7,F82` is the bare-minimum syntax-error gate. Tighten to a real lint config: `--max-line-length=120 --extend-ignore=E203,W503` and the full default ruleset. If this surfaces too many existing issues to fix in this pass, leave the original gate, **add** a separate "lint (allowed-to-fail)" job at full strictness, and document in `CONTRIBUTING.md` that we're moving toward strict lint.

### Don't

- Don't add tests for behaviour that wasn't implemented. If Teammate 1 chose `NotImplementedError` for `predict_proba`, don't test sampling.
- Don't write sweeping property-based tests. Hypothesis tests on a fairness tool produce confusing failures; stick to specific examples.
- Don't touch source code outside `tests/` and `conftest.py`.

### Done when

- `pytest tests/ -v` passes and the suite has grown by at least the tests above.
- The new tests fail when reverted against `main` (i.e., they actually test the new behaviour).
- CI is green on the matrix.
- Handoff note lists each new test, the contract it locks, and any test that was deleted because the code it covered was removed.

---

## Synthesis pass (you, after all four Teammates return)

Once all four Teammates have completed and handed off, **you** integrate. The Teammates were forbidden from touching each other's files, so there will be cross-cutting work that only the synthesizer can do.

### Steps

1. **Re-read all four handoff notes** in one go. Look for:
   - Conflicts where two Teammates documented contradictory changes (rare, but possible — e.g., Teammate 1 chose single-sensitive restriction, Teammate 2 wrote multi-sensitive docs).
   - Cross-cutting work that was deferred to synthesis ("Teammate 2 flagged that Teammate 1 needs to pass `fair_metrics` to `generate_html_report` — apply now").
   - TODO comments left for the maintainer (you don't resolve those; you collate them in the final report).

2. **Wire the cross-cutting changes:**
   - Update `cli.py` to pass both `metrics` and `fair_metrics` into `generate_html_report` (this is the seam between Teammate 1 and Teammate 2).
   - Verify the README's example commands still work against the post-Teammate-1 CLI surface.
   - If Teammate 1 chose to restrict to a single `--sensitive` argument, make sure Teammate 2's README examples are consistent.

3. **Run the full suite:**
   ```bash
   pytest tests/ -v
   flake8 equiml/ --max-line-length=120 --extend-ignore=E203,W503
   python -m build  # smoke check that the package still builds
   ```
   If anything is red, fix it before continuing.

4. **Run the example end-to-end:**
   ```bash
   python examples/adult_census_audit.py
   # or
   jupyter nbconvert --execute examples/adult_census_audit.ipynb
   ```
   The example must run on a clean install. If it doesn't, the procurement persona's whole "ingestible tool" thesis breaks.

5. **Write a single squashed commit message** for the cleanup branch summarising what changed across all four tracks. Reference the four Teammates by their concern (correctness / positioning / packaging / verification). Don't reference internal personas — those are a workshop tool, not a public artefact.

6. **Produce a final report for the maintainer** with:
   - The four handoff notes verbatim.
   - The list of TODOs left in place (and which persona / Teammate raised them).
   - The list of out-of-scope items that were deliberately **not** addressed:
     - Essay: *"What I deleted from my fairness framework, and why"* (~1500 words). Anthropic-lens recommendation. Author-only work.
     - MCP server wrapper (`equiml-mcp`) exposing `audit_dataset` as an MCP tool. Strategic, ~200 lines, parked for a separate branch.
     - Rename / brand decision (the "EquiML name is forgettable in a crowded field" concern). Wait until the comparison table is written and positioning falls out.
     - PyPI publication. Maintainer decision; release process is now ready.
     - GitHub repo description and the cross-link URLs in the README footer (Medium piece URLs). Maintainer-only.

7. **Open a draft PR** against `main` with the cleanup branch. Body of the PR is the final report from step 6. Do not auto-merge.

---

## Anti-patterns (do not)

- **Don't run the Teammates sequentially.** The whole point is independent voices. Sequential execution lets the second agent see the first's output and reflexively conform.
- **Don't let any Teammate edit outside its lane.** If Teammate 1 wants to fix a typo in the README, the answer is "no, flag it for synthesis." Strict ownership is the merge-conflict prevention.
- **Don't ghostwrite the maintainer's voice.** The "Why I built this" paragraph and the Medium-link footer are author-only. Leave structured TODOs.
- **Don't apply the parked items.** The essay, the MCP server, the rename, the PyPI publish — all out of scope for this pass. Their place is the final report.
- **Don't auto-merge.** This is a substantial cleanup; a human should review before it lands on `main`.

---

## Verification checklist (before opening the PR)

- [ ] All seven Code Surgeon tasks implemented and tested.
- [ ] README has comparison table, governance section, "what it doesn't do" section, demoted title note, author footer with TODOs.
- [ ] `SECURITY.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `.github/dependabot.yml`, `examples/adult_census_audit.{py,ipynb}` all exist with non-trivial content.
- [ ] New tests added for CLI smoke, multi-sensitive contract, substring-collision regression, `predict_proba` contract, `cross_validate` contract, and one adversarial case.
- [ ] `pytest tests/ -v` is green on Python 3.10/3.11/3.12.
- [ ] flake8 (at the chosen strictness level) is green.
- [ ] `python -m build` succeeds.
- [ ] `examples/adult_census_audit.{py,ipynb}` runs end-to-end on a clean venv.
- [ ] Final report drafted with handoff notes, TODOs, and parked items listed explicitly.

That's the brief. Run pre-flight, dispatch the four Teammates, wait for all four, then synthesise. Don't skip the synthesis pass — that's where the cross-cutting work lives.
