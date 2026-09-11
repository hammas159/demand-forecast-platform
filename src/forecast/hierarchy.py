"""Hierarchical forecasting and reconciliation.

The problem, which is easy to state and easy to get wrong:

    total = 1000        ← forecast independently
      north = 400       ← forecast independently
      south = 550       ← forecast independently

400 + 550 = 950 ≠ 1000. The regional plans and the national plan disagree by 50 units,
and every downstream decision — procurement, staffing, cash — is now made against one
of two numbers that cannot both be right.

Forecasting each level separately always produces this, because each level is fitted
to its own noise. **Reconciliation** adjusts the forecasts so they add up. Coherence is
an exact arithmetic property, which makes it one of the rare machine-learning
behaviours that can be asserted rather than measured.

Three methods, each correct for a different situation:

  bottom-up   Forecast the leaves, sum upward. Coherent by construction. Best when
              the leaves have enough signal to forecast; worst when they are sparse
              and noisy, because the noise sums too.
  top-down    Forecast the total, split by historical proportions. Best when the total
              is stable and the leaves are sparse; it cannot represent a leaf whose
              share is changing.
  optimal     Keep every level's forecast and distribute the disagreement. Better than
              either when both levels carry real information.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field


class HierarchyError(ValueError):
    pass


@dataclass
class Node:
    name: str
    children: list[str] = field(default_factory=list)
    parent: str | None = None


@dataclass
class Hierarchy:
    """A tree of series. Leaves hold data; internal nodes are sums of their children."""

    nodes: dict[str, Node] = field(default_factory=dict)
    root: str = ""

    def add(self, name: str, *, parent: str | None = None) -> Node:
        if name in self.nodes:
            raise HierarchyError(f"{name!r} already in the hierarchy")
        node = Node(name=name, parent=parent)
        self.nodes[name] = node
        if parent is None:
            if self.root:
                raise HierarchyError(f"hierarchy already has root {self.root!r}")
            self.root = name
        else:
            if parent not in self.nodes:
                raise HierarchyError(f"unknown parent {parent!r}")
            self.nodes[parent].children.append(name)
        return node

    def leaves(self) -> list[str]:
        return [n.name for n in self.nodes.values() if not n.children]

    def descendant_leaves(self, name: str) -> list[str]:
        node = self.nodes[name]
        if not node.children:
            return [name]
        out: list[str] = []
        for child in node.children:
            out.extend(self.descendant_leaves(child))
        return out

    def levels(self) -> list[list[str]]:
        """Nodes grouped by depth, root first."""
        out: list[list[str]] = []
        current = [self.root] if self.root else []
        while current:
            out.append(current)
            nxt: list[str] = []
            for name in current:
                nxt.extend(self.nodes[name].children)
            current = nxt
        return out

    def is_coherent(self, values: dict[str, float], *, tolerance: float = 1e-6) -> bool:
        """Does every parent equal the sum of its children?"""
        for name, node in self.nodes.items():
            if not node.children:
                continue
            total = sum(values.get(c, 0.0) for c in node.children)
            if abs(values.get(name, 0.0) - total) > tolerance:
                return False
        return True


# --- reconciliation ---------------------------------------------------------------


def bottom_up(hierarchy: Hierarchy, base: dict[str, float]) -> dict[str, float]:
    """Sum the leaves upward. Coherent by construction, and it discards every
    forecast made above leaf level."""
    out: dict[str, float] = {}

    def value(name: str) -> float:
        node = hierarchy.nodes[name]
        if not node.children:
            out[name] = float(base.get(name, 0.0))
        else:
            out[name] = sum(value(c) for c in node.children)
        return out[name]

    value(hierarchy.root)
    return out


def top_down(
    hierarchy: Hierarchy, base: dict[str, float], proportions: dict[str, float]
) -> dict[str, float]:
    """Split the root forecast by historical proportions.

    Proportions are of the *root*, per leaf, and must sum to 1. Requiring leaf-level
    proportions rather than level-by-level shares avoids compounding rounding error
    down a deep tree.
    """
    leaves = hierarchy.leaves()
    total_share = sum(proportions.get(leaf, 0.0) for leaf in leaves)
    if total_share <= 0:
        raise HierarchyError("proportions must sum to a positive number")

    root_value = float(base.get(hierarchy.root, 0.0))
    leaf_values = {leaf: root_value * proportions.get(leaf, 0.0) / total_share for leaf in leaves}
    return bottom_up(hierarchy, leaf_values)


def historical_proportions(
    hierarchy: Hierarchy, history: dict[str, Sequence[float]]
) -> dict[str, float]:
    """Each leaf's average share of the root over the observed history."""
    leaves = hierarchy.leaves()
    totals = {leaf: sum(history.get(leaf, [])) for leaf in leaves}
    grand = sum(totals.values())
    if grand <= 0:
        # No history to apportion by: an equal split is the only defensible default,
        # and it is better than dividing by zero or silently returning nothing.
        return {leaf: 1.0 / len(leaves) for leaf in leaves} if leaves else {}
    return {leaf: totals[leaf] / grand for leaf in leaves}


def optimal(
    hierarchy: Hierarchy, base: dict[str, float], *, weights: dict[str, float] | None = None
) -> dict[str, float]:
    """Distribute each parent's disagreement across its children, proportionally.

    This is the practical core of optimal reconciliation without the matrix algebra:
    a parent whose children disagree with it pushes the difference down, split by the
    children's own forecast magnitudes, so a large child absorbs more of the
    adjustment than a small one. Applied top-down, the result is coherent everywhere.

    `weights` lets a level be trusted more or less — a weight of 0 on a node means its
    own forecast is ignored and it simply takes the sum of its children.

    Two passes, and the order is the whole trick. A single top-down pass cannot work:
    adjusting a level overwrites the value its own parent just fixed, so every level
    silently breaks the one above it.

        1. bottom-up   blend each node's own forecast with its children's sum, so
                       information from every level reaches the root
        2. top-down    fix the root, then scale each node's children to match it,
                       preserving the relative proportions the blend produced

    After pass 2 every parent equals the sum of its children exactly, by construction.
    """
    weights = weights or {}
    blended: dict[str, float] = {}

    def blend(name: str) -> float:
        node = hierarchy.nodes[name]
        own = float(base.get(name, 0.0))
        if not node.children:
            blended[name] = own
            return own

        child_total = sum(blend(c) for c in node.children)
        parent_weight = weights.get(name, 1.0)
        # Weight 0 means the node's own forecast is ignored: it becomes its children.
        blended[name] = (
            child_total
            if parent_weight <= 0
            else (parent_weight * own + child_total) / (parent_weight + 1)
        )
        return blended[name]

    blend(hierarchy.root)

    out: dict[str, float] = {hierarchy.root: blended[hierarchy.root]}

    def distribute(name: str) -> None:
        children = hierarchy.nodes[name].children
        if not children:
            return
        child_total = sum(blended[c] for c in children)
        if child_total == 0:
            # Nothing to apportion by. An equal split keeps the level coherent, which
            # is better than leaving it inconsistent or dividing by zero.
            share = out[name] / len(children)
            for c in children:
                out[c] = share
        else:
            scale = out[name] / child_total
            for c in children:
                out[c] = blended[c] * scale
        for c in children:
            distribute(c)

    distribute(hierarchy.root)
    return out
