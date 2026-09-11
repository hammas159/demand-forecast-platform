# demand-forecast-platform

[![ci](https://github.com/hammas159/demand-forecast-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/hammas159/demand-forecast-platform/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![dependencies](https://img.shields.io/badge/dependencies-none-success)
![license](https://img.shields.io/badge/license-MIT-green)

**Hierarchical demand forecasting where the numbers actually add up — and a backtest
that cannot lie to you.**

Zero dependencies. Reconciliation is arithmetic and a backtest is a loop; importing a
numerics stack for either would exceed the code.

---

## The problem nobody mentions in the tutorial

```
total  = 1000      ← forecast independently
  north =  400     ← forecast independently
  south =  550     ← forecast independently
```

`400 + 550 = 950 ≠ 1000`. The regional plans and the national plan disagree by 50
units, and procurement, staffing and cash are now planned against two numbers that
cannot both be right.

Forecasting each level separately **always** does this — every level is fitted to its
own noise. Reconciliation makes them agree.

And coherence is *exact arithmetic*, which makes it one of the very few
machine-learning properties you can **assert** instead of measure:

```python
@pytest.mark.parametrize("method", ["bottom_up", "optimal"])
def test_every_method_produces_coherence(method):
    assert h.is_coherent(result)
```

| Method | Correct when |
|---|---|
| **bottom-up** | Leaves carry enough signal. Coherent by construction — and it throws away every forecast made above leaf level. |
| **top-down** | The total is stable and the leaves are sparse. Cannot represent a leaf whose share is changing. |
| **optimal** | Both levels carry real information. Keeps every forecast and distributes the disagreement. |

`optimal` runs bottom-up to blend, then top-down to distribute. **The order is the
whole trick** — a single top-down pass overwrites the value each node's parent just
fixed, so every level silently breaks the one above it. That was a real bug here,
caught by running the coherence check.

Setting every weight to zero makes `optimal` degenerate exactly to bottom-up — also a
test, because a knob that does not do what it claims is worse than no knob.

## Intermittent demand

Spare parts and wholesale produce series that are 80% zeros with occasional orders.
Ordinary smoothing lets the zeros drag the level toward zero between orders, so the
forecast is always too low just before an order and too high just after.

**Croston's method** smooths two series instead — the *size* of a non-zero demand and
the *interval* between them — and forecasts size ÷ interval. That removes the
oscillation, and it is why averaging gives you "0.3 units" for something only ever
ordered in whole boxes.

## The backtest cannot leak

A backtest that lets the model see data after the forecast origin reports an accuracy
the system will never achieve, and reports it *confidently*. So the property is
asserted directly, by watching what the forecaster was handed:

```python
def test_the_forecaster_never_sees_the_future():
    for window in seen:
        assert window == series[: len(window)]
        assert max(window) < series[len(window)]
```

A forecaster that mutates its input also cannot corrupt later folds — it gets a copy.
That too is a test.

## MAPE is the wrong metric, and it is the industry default

MAPE divides by the actual value, so a zero makes it infinite and a near-zero makes it
enormous. Intermittent demand is *full* of zeros — MAPE reports nonsense exactly where
forecasting is hardest.

**MASE** divides the model's error by what seasonal naive would have scored on the
training data:

```
MASE < 1    better than doing nothing
MASE = 1    no better than doing nothing
MASE > 1    worse than doing nothing
```

Scale-free, comparable across series, and it answers the question the project exists
to answer. The denominator is computed **in-sample** — using the test period would let
the difficulty of the test set flatter the model.

On a perfectly flat series, MASE returns `NaN` rather than `0.0`: there is no naive
error to compare against, and reporting zero would claim perfection.

## Does anything beat seasonal naive?

The question that decides whether a forecasting project is worth doing. In retail the
answer is frequently *no*, and a team that never checked cannot tell whether its model
adds value or launders the seasonality it was handed.

```python
compare(series, {"naive": naive, "seasonal": seasonal, "croston": croston})
# {"seasonal": {"mase": 1.0, ...}, "naive": {"mase": 31.7, "beats_naive": False}}
```

Seasonal naive scores exactly 1.0 — it *is* the denominator, so it cannot beat itself.
That is the line any model has to come in under.

## Usage

```python
h = Hierarchy()
h.add("total")
h.add("north", parent="total"); h.add("south", parent="total")
h.add("lahore", parent="north"); h.add("karachi", parent="south")

base = {name: forecast(history[name], horizon=4) for name in h.nodes}
plan = optimal(h, base)          # coherent everywhere

assert h.is_coherent(plan)

rolling_origin(history["lahore"], croston, horizon=4, period=52).summary()
```

## Tests

**41 tests. No dependencies, no fixtures, no data download.**

| Covered | |
|---|---|
| Hierarchy | structure, levels, duplicate/second-root/unknown-parent rejection, incoherence of independent forecasts |
| Reconciliation | coherence for every method, leaf and root preservation, weight-0 degeneracy, proportional absorption, deep trees, empty history |
| Models | naive, seasonal, moving average, drift, Croston on intermittent/single/all-zero demand |
| Metrics | sMAPE bounds and the 0-vs-0 case, MASE direction, in-sample denominator, flat-series `NaN` |
| Backtest | **leakage**, expanding window, wrong-length output, mutating forecaster, short series, identical folds |

## Limits

- `optimal` distributes disagreement proportionally rather than solving the MinT
  covariance system. It is coherent and it uses every level; it is not provably
  minimum-variance. The matrix version needs numpy and a covariance estimate that is
  itself hard to get right on short series.
- Baselines only. Gradient boosting and state-space models fit behind the same
  `Forecaster` signature — but the baselines are what they must be scored against,
  which is why they are here first.
- One seasonal period. Daily-and-weekly together needs decomposition.
- No exogenous regressors. Promotions and holidays are the obvious next feature and the
  backtest harness already accommodates them.

## License

MIT
