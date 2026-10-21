"""Generate the bundled example datasets (deterministic, seed 2026).

Run from the repository root::

    python examples/generate_examples.py

Predictions are *simulated* with known error structure (no model is trained), so the
diagnostics have known ground truth. Each dataset plants specific, documented issues; see
``examples/README.md``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SEED = 2026


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.6g")


def regression(rng: np.random.Generator) -> None:
    """House-price-like regression.

    Planted: `sqft_m2` = sqft × 0.0929 (redundant feature), 35% missing `garage_area` in
    test, one extreme `lot_size` value, 12 test rows duplicated from train, an overfit
    model (near-perfect train predictions) and a +15k bias on test predictions.
    """
    n_train, n_test = 800, 200
    n = n_train + n_test
    sqft = rng.normal(1800, 450, n).clip(500)
    bedrooms = rng.integers(1, 6, n)
    age = rng.uniform(0, 80, n)
    lot = rng.lognormal(8.5, 0.4, n)
    garage = rng.normal(35, 10, n).clip(0)
    price = 50_000 + 120 * sqft + 9_000 * bedrooms - 800 * age + 3 * lot + 400 * garage
    price += rng.normal(0, 25_000, n)
    df = pd.DataFrame(
        {
            "row_id": [f"h{i:05d}" for i in range(n)],
            "sqft": sqft.round(0),
            "sqft_m2": (sqft.round(0) * 0.0929),
            "bedrooms": bedrooms,
            "age_years": age.round(1),
            "lot_size": lot.round(0),
            "garage_area": garage.round(1),
            "price": price.round(0),
        }
    )
    df.loc[17, "lot_size"] = 2_500_000  # data-entry error
    train, test = df.iloc[:n_train].copy(), df.iloc[n_train:].copy()
    dup = train.sample(12, random_state=SEED)
    test.iloc[:12, 1:-1] = dup.iloc[:, 1:-1].to_numpy()
    test.iloc[:12, -1] = dup["price"].to_numpy()
    missing = test.sample(frac=0.35, random_state=SEED).index
    test.loc[missing, "garage_area"] = np.nan

    train_pred = train["price"] + rng.normal(0, 4_000, n_train)
    test_pred = test["price"] + 15_000 + rng.normal(0, 20_000, n_test)
    out = ROOT / "regression"
    _write(train, out / "train.csv")
    _write(test, out / "test.csv")
    _write(pd.DataFrame({"row_id": train["row_id"], "y_pred": train_pred.round(0)})
           .sample(frac=1.0, random_state=1), out / "train_predictions.csv")
    _write(pd.DataFrame({"row_id": test["row_id"], "y_pred": test_pred.round(0)})
           .sample(frac=1.0, random_state=2), out / "test_predictions.csv")


def classification(rng: np.random.Generator) -> None:
    """Churn-like imbalanced binary classification.

    Planted: ~6% positive class, `refund_issued` copies the label as yes/no text (target
    copy), `notes_len` with 8% missing, a constant `region_code`, a model that misses most
    churners on test.
    """
    n_train, n_test = 1500, 400
    n = n_train + n_test
    tenure = rng.exponential(24, n).round(1)
    spend = rng.gamma(3, 30, n).round(2)
    tickets = rng.poisson(1.2, n)
    logit = -3.4 + 0.5 * tickets - 0.03 * tenure + 0.004 * spend
    churn = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    notes = rng.integers(0, 400, n).astype(float)
    notes[rng.uniform(size=n) < 0.08] = np.nan
    df = pd.DataFrame(
        {
            "customer_id": [f"c{i:05d}" for i in range(n)],
            "tenure_months": tenure,
            "monthly_spend": spend,
            "support_tickets": tickets,
            "notes_len": notes,
            "region_code": "AU-QLD",
            "refund_issued": np.where(churn == 1, "yes", "no"),
            "churned": churn,
        }
    )
    train, test = df.iloc[:n_train], df.iloc[n_train:]
    score = logit + rng.normal(0, 0.6, n)
    train_pred = np.where(score[:n_train] > -1.6, 1, train["churned"])  # memorised labels
    test_pred = (score[n_train:] > -1.2).astype(int)
    out = ROOT / "classification"
    _write(train, out / "train.csv")
    _write(test, out / "test.csv")
    _write(pd.DataFrame({"customer_id": train["customer_id"], "y_pred": train_pred}),
           out / "train_predictions.csv")
    _write(pd.DataFrame({"customer_id": test["customer_id"], "y_pred": test_pred}),
           out / "test_predictions.csv")


def grouped(rng: np.random.Generator) -> None:
    """Repeated measurements per participant (gait speed from wearable features).

    Planted: 6 participants appear in both splits (group leakage for unseen-person
    evaluation) and strong per-participant error offsets (clustered errors).
    """
    n_people, visits = 60, 8
    pid = np.repeat([f"p{i:03d}" for i in range(n_people)], visits)
    person = np.repeat(rng.normal(0, 0.15, n_people), visits)
    cadence = rng.normal(105, 8, pid.size)
    stride = rng.normal(1.35, 0.12, pid.size)
    speed = 0.004 * cadence + 0.55 * stride + person + rng.normal(0, 0.04, pid.size)
    df = pd.DataFrame(
        {
            "visit_id": [f"v{i:05d}" for i in range(pid.size)],
            "participant_id": pid,
            "cadence": cadence.round(2),
            "stride_length": stride.round(3),
            "gait_speed": speed.round(3),
        }
    )
    people = sorted(set(pid))
    train_people = set(people[:45])
    test_people = set(people[45:]) | set(people[39:45])  # 6 shared participants
    train = df[df["participant_id"].isin(train_people)]
    test = df[df["participant_id"].isin(test_people)]
    test = test.assign(visit_id=test["visit_id"] + "t")
    offsets = {p: rng.normal(0, 0.12) for p in people}
    test_pred = test["gait_speed"] + test["participant_id"].map(offsets) + rng.normal(
        0, 0.03, len(test))
    train_pred = train["gait_speed"] + rng.normal(0, 0.05, len(train))
    out = ROOT / "grouped_data"
    _write(train, out / "train.csv")
    _write(test, out / "test.csv")
    _write(pd.DataFrame({"visit_id": train["visit_id"], "y_pred": train_pred.round(4)}),
           out / "train_predictions.csv")
    _write(pd.DataFrame({"visit_id": test["visit_id"], "y_pred": test_pred.round(4)}),
           out / "test_predictions.csv")


def main() -> None:
    rng = np.random.default_rng(SEED)
    regression(rng)
    classification(rng)
    grouped(rng)
    print(f"examples written under {ROOT}")


if __name__ == "__main__":
    main()
