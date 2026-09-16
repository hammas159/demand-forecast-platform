"""Forecast every level of a hierarchy independently, then make the numbers add up.

    python demo.py

Two things nobody notices until a planner does: independent forecasts of
`total`, `north` and `south` do not sum to each other, and an intermittent
series -- a part that sells on 11 days out of 40 -- defeats every ordinary
forecaster, Croston included. No dependencies, no network.
"""

import random
import sys

sys.path.insert(0, "src")

from forecast.backtest import compare
from forecast.hierarchy import Hierarchy, bottom_up, optimal
from forecast.models import croston, drift, moving_average, naive

h = Hierarchy()
h.add("total")
h.add("north", parent="total")
h.add("south", parent="total")
for leaf in ("n1", "n2"):
    h.add(leaf, parent="north")
for leaf in ("s1", "s2"):
    h.add(leaf, parent="south")

# What independent, per-series forecasting actually produces: every level modelled
# on its own history, so no level agrees with any other.
INDEPENDENT = {
    "total": 1000.0,
    "north": 610.0,
    "south": 440.0,
    "n1": 300.0,
    "n2": 290.0,
    "s1": 250.0,
    "s2": 205.0,
}

print("INPUT")
print("   a 3-level hierarchy, each node forecast independently")
for name in ("total", "north", "south", "n1", "n2", "s1", "s2"):
    print(f"      {name:6} {INDEPENDENT[name]:>8.1f}")
print()

print("OUTPUT")
print(f"   coherent as forecast?   {h.is_coherent(INDEPENDENT)}")
print(
    f"      north + south = {INDEPENDENT['north'] + INDEPENDENT['south']:.1f}, "
    f"but total says {INDEPENDENT['total']:.1f}"
)
print(
    f"      n1 + n2       = {INDEPENDENT['n1'] + INDEPENDENT['n2']:.1f}, "
    f"but north says {INDEPENDENT['north']:.1f}"
)
print()

for name, fn in (("bottom_up", bottom_up), ("optimal", optimal)):
    rec = fn(h, INDEPENDENT)
    print(
        f"   {name:10} coherent={h.is_coherent(rec)}   "
        f"total={rec['total']:.1f}  north={rec['north']:.1f}  south={rec['south']:.1f}"
    )
print()

# --- intermittent demand -----------------------------------------------------
rng = random.Random(3)
spare_part = [rng.choice([0, 0, 0, 0, 0, 2, 3]) for _ in range(40)]
nonzero = sum(1 for v in spare_part if v)

print(f"   an intermittent series: {nonzero} non-zero days out of {len(spare_part)}")
print(f"      {spare_part}")
print()
results = compare(
    spare_part,
    {
        "naive": naive,
        "moving_average": moving_average,
        "drift": drift,
        "croston": croston,
    },
    horizon=4,
    initial=20,
)
print(f"   {'forecaster':16} {'MASE':>8}  {'beats doing nothing?':>20}")
print("   " + "-" * 50)
for name, stats in sorted(results.items(), key=lambda kv: kv[1]["mase"]):
    print(f"   {name:16} {stats['mase']:>8.3f}  {str(stats['beats_naive']):>20}")
print()
print("   MASE below 1.0 beats a naive forecast. Above 1.0 is worse than")
print("   doing nothing, and all four are above 1.0.")
print("   Croston is built for intermittent demand and is the least bad of")
print("   the four -- but on 29 zero days out of 40 it still loses to naive.")
print("   The honest output here is that this series should not be forecast")
print("   at all; it should be stocked to a service level.")
