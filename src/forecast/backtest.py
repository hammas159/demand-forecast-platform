"""Rolling-origin backtesting and scale-free accuracy.

Two things this file exists to get right, both of which are usually got wrong.

**Leakage.** A backtest that lets the model see data after the forecast origin reports
an accuracy the system will never achieve in production, and it reports it
confidently. The split here is explicit and the forecaster is handed a *copy* of the
history up to the origin and nothing else — and there is a test that watches what it
was given.

**The metric.** MAPE is the default in industry and is broken in exactly the situation
forecasting is hardest: it divides by the actual, so a zero makes it infinite and a
near-zero makes it enormous. Intermittent demand is full of zeros, so MAPE reports
nonsense precisely where accuracy matters most.

MASE is used instead. It divides the model's error by the error a **seasonal naive**
forecast would have made on the training data:

    MASE < 1   better than seasonal naive
    MASE = 1   no better than doing nothing
    MASE > 1   worse than doing nothing

That is a number with a meaning, comparable across series of wildly different scale,
and it answers the question the project exists to answer.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

Forecaster = Callable[[Sequence[float], int], list[float]]


def mae(actual: Sequence[float], predicted: Sequence[float]) -> float:
    if not actual:
        return 0.0
    return statistics.fmean(abs(a - p) for a, p in zip(actual, predicted))


def rmse(actual: Sequence[float], predicted: Sequence[float]) -> float:
    if not actual:
        return 0.0
    return (statistics.fmean((a - p) ** 2 for a, p in zip(actual, predicted))) ** 0.5


def smape(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Symmetric MAPE, bounded at 200%. Still degenerate when both are zero, which is
    defined here as zero error rather than as a division by zero."""
    if not actual:
        return 0.0
    terms = []
    for a, p in zip(actual, predicted):
        denominator = (abs(a) + abs(p)) / 2
        terms.append(0.0 if denominator == 0 else abs(a - p) / denominator)
    return round(statistics.fmean(terms), 6)


def naive_scale(train: Sequence[float], *, period: int = 1) -> float:
    """Mean absolute error of a seasonal naive forecast on the training data.

    This is the denominator of MASE, and it must be computed **in-sample**: using the
    test period would let the difficulty of the test set flatter the model.
    """
    if len(train) <= period:
        return 0.0
    errors = [abs(train[i] - train[i - period]) for i in range(period, len(train))]
    return statistics.fmean(errors) if errors else 0.0


def mase(
    actual: Sequence[float], predicted: Sequence[float], train: Sequence[float],
    *, period: int = 1,
) -> float:
    scale = naive_scale(train, period=period)
    if scale == 0:
        # A perfectly flat training series has no naive error to compare against.
        # Reporting 0.0 would claim perfection; None says "not comparable", which is
        # the honest answer.
        return float("nan")
    return round(mae(actual, predicted) / scale, 6)


@dataclass
class FoldResult:
    origin: int
    actual: list[float]
    predicted: list[float]
    mae: float
    rmse: float
    smape: float
    mase: float


@dataclass
class BacktestResult:
    folds: list[FoldResult] = field(default_factory=list)

    def summary(self) -> dict:
        if not self.folds:
            return {"folds": 0}
        mases = [f.mase for f in self.folds if f.mase == f.mase]  # drop NaN
        return {
            "folds": len(self.folds),
            "mae": round(statistics.fmean(f.mae for f in self.folds), 6),
            "rmse": round(statistics.fmean(f.rmse for f in self.folds), 6),
            "smape": round(statistics.fmean(f.smape for f in self.folds), 6),
            "mase": round(statistics.fmean(mases), 6) if mases else None,
            # The only verdict that matters: did it beat doing nothing?
            "beats_naive": (statistics.fmean(mases) < 1.0) if mases else None,
        }


def rolling_origin(
    series: Sequence[float],
    forecaster: Forecaster,
    *,
    horizon: int = 4,
    initial: int | None = None,
    step: int = 1,
    period: int = 1,
) -> BacktestResult:
    """Expanding-window backtest.

    The origin advances through the series; at each position the forecaster sees only
    what precedes it. Expanding rather than sliding because that is what production
    does — a real system has all of its history available, not a fixed window of it.
    """
    series = list(series)
    initial = initial or max(period * 2, horizon * 2, 8)
    result = BacktestResult()

    origin = initial
    while origin + horizon <= len(series):
        train = series[:origin]
        actual = series[origin : origin + horizon]
        # A copy, so a forecaster that mutates its input cannot corrupt later folds.
        predicted = forecaster(list(train), horizon)

        if len(predicted) != horizon:
            raise ValueError(
                f"forecaster returned {len(predicted)} values for horizon {horizon}"
            )

        result.folds.append(
            FoldResult(
                origin=origin, actual=actual, predicted=predicted,
                mae=round(mae(actual, predicted), 6),
                rmse=round(rmse(actual, predicted), 6),
                smape=smape(actual, predicted),
                mase=mase(actual, predicted, train, period=period),
            )
        )
        origin += step

    return result


def compare(
    series: Sequence[float], forecasters: dict[str, Forecaster], **kw
) -> dict[str, dict]:
    """Score several forecasters on the same folds.

    Same series, same origins, same horizon — otherwise the comparison measures the
    split rather than the models.
    """
    return {
        name: rolling_origin(series, fn, **kw).summary()
        for name, fn in forecasters.items()
    }
