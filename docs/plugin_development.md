# Writing a diagnostic

A diagnostic is any object that satisfies `mldoctor.core.protocols.Diagnostic`. The easiest
way to write one is to subclass `BaseDiagnostic`.

```python
from typing import Any, ClassVar

from mldoctor.core.context import DiagnosticContext
from mldoctor.core.protocols import BaseDiagnostic, Requirement
from mldoctor.core.schemas import DiagnosticResult, Finding, Task
from mldoctor.recommendations.rules import make_finding


class TargetDriftDiagnostic(BaseDiagnostic):
    id = "target_drift"                       # snake_case, unique
    version = "0.1.0"
    title = "Target distribution drift"
    description = "Compares the target distribution between train and test."
    supported_tasks: ClassVar[frozenset[Task]] = frozenset({"regression"})
    requires: ClassVar[frozenset[Requirement]] = frozenset({"test", "test_target"})
    limitations: ClassVar[tuple[str, ...]] = ("Marginal comparison only.",)

    def run(self, context: DiagnosticContext) -> DiagnosticResult:
        # 1. compute: pure numbers, ideally a function in mldoctor.statistics
        # 2. interpret: map numbers to findings with configured thresholds
        findings: list[Finding] = []
        metrics: dict[str, Any] = {}
        ...
        return self.result(findings, metrics)
```

## Rules for diagnostics

1. **Keep computation and interpretation separate.** Put the numerical method in a pure,
   unit-tested function. The diagnostic only turns numbers into findings.
2. **Never put NaN or Inf in evidence.** Use `None` and add a reason. `jsonable` raises on
   non-finite floats, and the schema rejects them.
3. **Use the catalog.** `make_finding(code, severity, message, evidence)` takes the title and
   recommendation from `recommendations/rules.py:CATALOG`. Add new codes there and document
   them in `docs/report_schema.md`.
4. **Declare what you need.** `supported_tasks` and `requires` let the engine skip the
   check with a reason. Override `applicability()` for extra conditions (for example "fewer
   than two numeric features"). For partial runs, record what didn't run in
   `skipped_subchecks`. Never pass silently.
5. **Randomness** must come from `context.rng(self.id)`.
6. **Privacy.** Report aggregates only: no raw values, row IDs or group IDs.
7. **Do not mutate** `context.train` or `context.test`.
8. **Test with known answers.** Use synthetic data where the correct number is known
   analytically, and check the number as well as the code.

Exceptions raised by `run` become `error` results (exit code 3 unless `--continue-on-error`),
so do not catch broad exceptions just to hide failures.

## Registering a plugin

Expose the class through the `mldoctor.diagnostics` entry-point group in your package's
`pyproject.toml`:

```toml
[project.entry-points."mldoctor.diagnostics"]
target_drift = "my_package.drift:TargetDriftDiagnostic"
```

Loading a plugin runs its code, so plugins load only when the configuration enables it:

```toml
load_plugins = true
```

From Python you can also build a registry directly:

```python
from mldoctor.api import diagnose
from mldoctor.core.registry import default_registry

registry = default_registry()
registry.register(TargetDriftDiagnostic())
report = diagnose(train=..., test=..., target="y", task="regression", registry=registry)
```

## Context reference

`DiagnosticContext` provides:

| Attribute | Meaning |
|-----------|---------|
| `task` | `"classification"` or `"regression"` |
| `train`, `test` | private DataFrame copies (`test` may be `None`) |
| `target`, `row_id`, `group` | column names (`row_id`/`group` may be `None`) |
| `feature_columns`, `numeric_features` | features after metadata removal |
| `test_predictions`, `train_predictions` | Series aligned to the split index via row ID |
| `class_labels` | canonical string labels (classification) |
| `test_has_target` | whether the test split contains the target column |
| `config` | the validated `DoctorConfig` |
| `rng(stream)` | reproducible, independent `numpy.random.Generator` |
