"""The Streamlit demo the `ui` dependency group declared but never shipped.

Two tabs: hierarchical reconciliation (build a 3-node tree, enter base forecasts that
disagree, watch each method force agreement) and model backtesting (compare naive/
seasonal/drift/moving-average/Croston on a regular or intermittent-demand series via
the same rolling-origin folds).

Run: streamlit run ui/app.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from forecast.backtest import compare  # noqa: E402
from forecast.hierarchy import Hierarchy, bottom_up, optimal, top_down  # noqa: E402
from forecast.models import croston, drift, moving_average, naive, naive_seasonal  # noqa: E402

st.set_page_config(page_title="demand-forecast-platform demo", layout="wide")
st.title("demand-forecast-platform")
st.caption(
    "Hierarchical reconciliation where the numbers actually add up, and a backtest that cannot lie."
)

tab_hierarchy, tab_models = st.tabs(["Hierarchical reconciliation", "Model backtesting"])

# ---- Tab 1: reconciliation ----------------------------------------------------------

with tab_hierarchy:
    st.markdown(
        "`total`, `north`, and `south` were each forecast **independently** — they "
        "disagree, because every level was fit to its own noise. Reconciliation forces "
        "them to agree; the three methods differ in *how* they split the disagreement."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Independent (disagreeing) forecasts")
        total_fc = st.number_input("total", value=1000.0)
        north_fc = st.number_input("north", value=400.0)
        south_fc = st.number_input("south", value=550.0)
        st.metric("Disagreement", f"{total_fc - (north_fc + south_fc):+.0f}")

        st.subheader("Historical shares (for top-down)")
        north_hist = st.number_input("north historical total", value=3800.0)
        south_hist = st.number_input("south historical total", value=6200.0)

    hierarchy = Hierarchy()
    hierarchy.add("total")
    hierarchy.add("north", parent="total")
    hierarchy.add("south", parent="total")
    base = {"total": total_fc, "north": north_fc, "south": south_fc}
    proportions = {
        "north": north_hist / (north_hist + south_hist) if (north_hist + south_hist) else 0.5,
        "south": south_hist / (north_hist + south_hist) if (north_hist + south_hist) else 0.5,
    }

    with col2:
        st.subheader("Reconciled")
        for method_name, result in [
            ("Bottom-up (sum leaves)", bottom_up(hierarchy, base)),
            ("Top-down (historical share)", top_down(hierarchy, base, proportions)),
            ("Optimal (proportional adjustment)", optimal(hierarchy, base)),
        ]:
            coherent = hierarchy.is_coherent(result)
            st.write(f"**{method_name}** {'✅ coherent' if coherent else '❌'}")
            st.write(
                f"total={result['total']:.1f}, north={result['north']:.1f}, "
                f"south={result['south']:.1f}"
            )

# ---- Tab 2: model backtesting --------------------------------------------------------

with tab_models:
    st.markdown(
        "Same series, same rolling-origin folds, every model — otherwise the comparison "
        "measures the split rather than the models. **MASE < 1 means it beat doing "
        "nothing.**"
    )

    scenario = st.radio("Series", ["Regular weekly demand", "Intermittent demand (Croston's case)"])
    random.seed(3)
    if scenario == "Regular weekly demand":
        series = [50 + 15 * (1 if i % 7 in (5, 6) else 0) + random.gauss(0, 4) for i in range(120)]
        forecasters = {
            "naive": naive,
            "seasonal (period=7)": lambda h, n: naive_seasonal(h, n, period=7),
            "moving average": moving_average,
            "drift": drift,
        }
    else:
        series = [0.0] * 120
        for i in range(0, 120, 9):
            series[i] = random.uniform(20, 60)
        forecasters = {
            "naive": naive,
            "moving average": moving_average,
            "croston": croston,
        }

    st.line_chart(series)

    if st.button("Run backtest", type="primary"):
        results = compare(series, forecasters, horizon=4, period=7 if "Regular" in scenario else 1)
        rows = []
        for name, summary in results.items():
            rows.append(
                {
                    "model": name,
                    "folds": summary.get("folds", 0),
                    "MAE": summary.get("mae"),
                    "MASE": summary.get("mase"),
                    "beats naive": summary.get("beats_naive"),
                }
            )
        st.table(rows)
        scored = [r for r in rows if r["MASE"] is not None]
        best = min(scored, key=lambda r: r["MASE"], default=None)
        if best:
            st.success(f"Best by MASE: **{best['model']}** ({best['MASE']})")
