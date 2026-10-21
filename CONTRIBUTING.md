# Contributing to AutoML Doctor

Contributions are welcome: bug reports, new diagnostics, documentation and tests.

## Development setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest && ruff check src tests && mypy
```

## Ground rules

1. **Statistical correctness first.** Every new numerical method needs tests that compare it
   against a known analytic value or a reference implementation (SciPy or scikit-learn are
   available in the dev extras). Test conclusions as well as numbers, not only that the code
   runs.
2. **Undefined is not zero.** Return `None` plus a reason. Never put NaN or Inf in reports.
3. **No false passes.** When evidence is missing, skip with a reason or record a skipped
   sub-check.
4. **Thresholds are heuristics.** Make them configurable, document them in
   `docs/statistical_methods.md`, and never describe them as significance tests.
5. **No causal or definitive language.** Write "possible target copy", not "leakage
   detected".
6. **Privacy.** Report aggregates only: no raw rows, IDs or group identifiers.
7. **Local-first.** No network calls, telemetry, LLM calls or pickle loading in the core
   package.
8. **Determinism.** Randomness only through `context.rng(stream)`.

## Workflow

- Branch from `main` and keep pull requests focused.
- Add or update tests: `tests/unit` for numerical functions and diagnostics,
  `tests/integration` for CLI behaviour, `tests/regression` for example snapshots
  (`UPDATE_SNAPSHOTS=1 pytest tests/regression` after an intended change, and explain why
  in the PR).
- Update `CHANGELOG.md` under "Unreleased".
- If you change public interfaces, the report schema, finding codes or default thresholds,
  say so explicitly in the PR description. Schema changes need a `SCHEMA_VERSION` bump.

## Adding a diagnostic

See [`docs/plugin_development.md`](docs/plugin_development.md). Built-in diagnostics are
registered in `src/mldoctor/diagnostics/__init__.py`. Their order there is the report order.

## Reporting bugs

Please include the command, the `mldoctor version` output, the relevant part of the JSON
report (it contains no raw data by design) and, if possible, a small synthetic CSV that
reproduces the problem. Please do not attach real or sensitive datasets.
