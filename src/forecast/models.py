"""Baseline forecasters.

Deliberately simple, and deliberately including the one nobody implements.

A forecasting project that opens with gradient boosting has skipped the question that
decides whether the project is worth doing: **does anything beat seasonal naive?** In
retail it frequently does not, and a team that never checked has no idea whether its
model is adding value or laundering the seasonality it was handed.

So the baselines are here, they are the thing everything else is scored against, and
`naive_seasonal` is the denominator of the accuracy metric.

Croston is included because intermittent demand — long runs of zeros punctuated by
occasional orders — is extremely common in spare parts and wholesale, and every
standard method handles it badly. Averaging a series that is 80% zeros produces a
forecast of 0.3 units for something that is only ever ordered in whole boxes.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence


def naive(history: Sequence[float], horizon: int) -> list[float]:
    """Last value, repeated. The floor any model must clear."""
    if not history:
        return [0.0] * horizon
    return [float(history[-1])] * horizon


def naive_seasonal(history: Sequence[float], horizon: int, *, period: int) -> list[float]:
    """Same period last season. In retail this is a genuinely strong competitor."""
    if not history or period <= 0:
        return naive(history, horizon)
    if len(history) < period:
        return naive(history, horizon)
    return [float(history[-period + (i % period)]) for i in range(horizon)]


def moving_average(history: Sequence[float], horizon: int, *, window: int = 4) -> list[float]:
    if not history:
        return [0.0] * horizon
    window = min(window, len(history))
    value = statistics.fmean(history[-window:])
    return [float(value)] * horizon


def drift(history: Sequence[float], horizon: int) -> list[float]:
    """Last value plus the average per-period change. A trend model with no fitting."""
    if len(history) < 2:
        return naive(history, horizon)
    slope = (history[-1] - history[0]) / (len(history) - 1)
    return [float(history[-1] + slope * (i + 1)) for i in range(horizon)]


def croston(
    history: Sequence[float], horizon: int, *, alpha: float = 0.1
) -> list[float]:
    """Croston's method for intermittent demand.

    Two exponentially smoothed series rather than one: the **size** of a non-zero
    demand, and the **interval** between non-zero demands. The forecast is size
    divided by interval.

    The reason this works where ordinary smoothing fails: smoothing the raw series
    lets the zeros drag the level toward zero between orders, so the forecast is
    always too low right before an order and too high right after. Separating size
    from timing removes that oscillation.
    """
    demands = [(i, v) for i, v in enumerate(history) if v > 0]
    if not demands:
        return [0.0] * horizon
    if len(demands) == 1:
        # One observation gives a size but no interval to estimate.
        return [float(demands[0][1])] * horizon

    size = float(demands[0][1])
    interval = float(demands[1][0] - demands[0][0])

    previous_index = demands[0][0]
    for index, value in demands[1:]:
        gap = index - previous_index
        size += alpha * (value - size)
        interval += alpha * (gap - interval)
        previous_index = index

    rate = size / interval if interval > 0 else size
    return [float(rate)] * horizon
