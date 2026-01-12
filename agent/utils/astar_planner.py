"""
A* grid planner.

Takes a numpy occupancy grid (int8: 0 free, 100 occupied, -1 unknown) and returns
an ordered list of grid cell coordinates from start to goal. Designed to be used
with occupancy grids produced by navmesh_occupancy.navmesh_to_occupancy_grid.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import heapq

import numpy as np


@dataclass(order=True)
class _PrioritizedNode:
    """Node for priority queue ordering by A* f-cost.

    Attributes:
        f: Total estimated cost (g + h) used for heap ordering.
        g: Accumulated path cost from the start node.
        h: Heuristic estimate to the goal.
        pos: Grid cell position as (row, col).
        parent: Optional parent cell used for path reconstruction.
    """
    f: float
    g: float
    h: float
    pos: Tuple[int, int]
    parent: Optional[Tuple[int, int]]


@dataclass
class AStarResult:
    """Result of an A* search.

    Attributes:
        path: Ordered list of (row, col) cells from start to goal.
        cost: Accumulated g-cost of the returned path.
        expanded: Number of nodes expanded during the search.
    """
    path: List[Tuple[int, int]]
    cost: float
    expanded: int


def astar(
    grid: np.ndarray,
    start: Sequence[int],
    goal: Sequence[int],
    allow_diagonal: bool = True,
    obstacle_threshold: int = 50,
    unknown_is_obstacle: bool = False,
    cost_grid: Optional[np.ndarray] = None,
    cost_weight: float = 1.0,
) -> Optional[AStarResult]:
    """Perform A* search on a 2D occupancy grid.

    Args:
        grid: 2D array with occupancy values (0 free, > obstacle_threshold occupied, -1 unknown).
        start: (row, col) start cell.
        goal: (row, col) goal cell.
        allow_diagonal: Use 8-connected moves when True, else 4-connected.
        obstacle_threshold: Values strictly greater than this are treated as obstacles.
        unknown_is_obstacle: When True, negative cells are treated as obstacles.
        cost_grid: Optional float grid aligned with ``grid``; contributes additive cost.
        cost_weight: Scalar multiplier for values pulled from ``cost_grid``.

    Returns:
        AStarResult with path, total cost, and expanded count; None if no path exists.
    """

    if grid is None or grid.ndim != 2:
        return None
    rows, cols = grid.shape
    sr, sc = int(start[0]), int(start[1])
    gr, gc = int(goal[0]), int(goal[1])

    cost_view: Optional[np.ndarray] = None
    if cost_grid is not None:
        try:
            cost_view = np.asarray(cost_grid, dtype=float)
            if cost_view.shape != grid.shape:
                cost_view = None
        except Exception:
            cost_view = None

    def _in_bounds(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols

    def _is_free(r: int, c: int) -> bool:
        val = float(grid[r, c])
        if unknown_is_obstacle and val < 0:
            return False
        return val <= obstacle_threshold

    def _cell_cost(r: int, c: int) -> float:
        if cost_view is None:
            return 0.0
        try:
            value = float(cost_view[r, c])
        except Exception:
            return 0.0
        if not np.isfinite(value):
            return 0.0
        return max(0.0, value)

    if not _in_bounds(sr, sc) or not _in_bounds(gr, gc):
        return None
    if not _is_free(sr, sc) or not _is_free(gr, gc):
        return None

    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if allow_diagonal:
        dirs.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])

    open_heap: List[_PrioritizedNode] = []
    start_h = _heuristic(sr, sc, gr, gc, allow_diagonal)
    heapq.heappush(open_heap, _PrioritizedNode(start_h, 0.0, start_h, (sr, sc), None))
    came_from: dict[Tuple[int, int], Tuple[int, int]] = {}
    g_score: dict[Tuple[int, int], float] = {(sr, sc): 0.0}
    closed: set[Tuple[int, int]] = set()

    expanded = 0

    while open_heap:
        current = heapq.heappop(open_heap)
        cr, cc = current.pos
        if current.pos in closed:
            continue
        closed.add(current.pos)
        expanded += 1

        if cr == gr and cc == gc:
            path = _reconstruct_path(came_from, current.pos)
            return AStarResult(path=path, cost=current.g, expanded=expanded)

        for dr, dc in dirs:
            nr, nc = cr + dr, cc + dc
            if not _in_bounds(nr, nc):
                continue
            if not _is_free(nr, nc):
                continue
            move = _move_cost(dr, dc)
            tentative_g = current.g + move * (1.0 + cost_weight * _cell_cost(nr, nc))
            if tentative_g >= g_score.get((nr, nc), float("inf")):
                continue
            came_from[(nr, nc)] = current.pos
            g_score[(nr, nc)] = tentative_g
            h = _heuristic(nr, nc, gr, gc, allow_diagonal)
            f = tentative_g + h
            heapq.heappush(open_heap, _PrioritizedNode(f, tentative_g, h, (nr, nc), current.pos))

    return None


def _heuristic(r: int, c: int, gr: int, gc: int, allow_diagonal: bool) -> float:
    """Admissible heuristic.

    Args:
        r: Current row.
        c: Current column.
        gr: Goal row.
        gc: Goal column.
        allow_diagonal: Whether diagonal motion is permitted.

    Returns:
        float: Octile distance when diagonals are allowed; Manhattan otherwise.
    """
    dr = abs(gr - r)
    dc = abs(gc - c)
    if allow_diagonal:
        diag = min(dr, dc)
        straight = max(dr, dc) - diag
        diag_cost = 1.41421356237
        return diag_cost * diag + straight
    return dr + dc


def _move_cost(dr: int, dc: int) -> float:
    """Movement cost for a step.

    Args:
        dr: Row delta.
        dc: Column delta.

    Returns:
        float: sqrt(2) for diagonal steps, 1.0 for straight steps.
    """
    return 1.41421356237 if dr != 0 and dc != 0 else 1.0


def _reconstruct_path(came_from: dict[Tuple[int, int], Tuple[int, int]], current: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Rebuild a path from predecessor links.

    Args:
        came_from: Mapping of node -> predecessor node.
        current: Goal node (row, col) to backtrack from.

    Returns:
        List[Tuple[int, int]]: Path from start to goal, inclusive.
    """
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path
