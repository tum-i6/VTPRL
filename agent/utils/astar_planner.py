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

import cv2
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
        path: Ordered list of (row, col) cells from start to goal (or to
            the nearest reachable cell when ``partial`` is True).
        cost: Accumulated g-cost of the returned path.
        expanded: Number of nodes expanded during the search.
        partial: True when the planner could not reach the goal and
            returned a best-effort path to the nearest reachable cell
            instead.  False for a complete start-to-goal path.
    """
    path: List[Tuple[int, int]]
    cost: float
    expanded: int
    partial: bool = False


def astar(
    grid: np.ndarray,
    start: Sequence[int],
    goal: Sequence[int],
    allow_diagonal: bool = True,
    obstacle_threshold: int = 50,
    unknown_is_obstacle: bool = False,
    cost_grid: Optional[np.ndarray] = None,
    cost_weight: float = 1.0,
    min_obstacle_distance_cells: float = 0.0,
    project_start_goal_to_valid: bool = True,
    max_projection_distance_cells: int = 8,
    allow_partial: bool = False,
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
        min_obstacle_distance_cells: Required minimum clearance from occupied
            cells in grid-cell units. This enables footprint-aware planning
            without changing the occupancy grid.
        project_start_goal_to_valid: If True, invalid start/goal cells are
            projected to the nearest valid cell within
            ``max_projection_distance_cells``.
        max_projection_distance_cells: Maximum search radius (in cells) used for
            start/goal projection.
        allow_partial: When True and the goal is unreachable, return a
            best-effort partial path to the explored cell closest to the
            goal (Euclidean).  The returned ``AStarResult.partial`` flag
            is set to True in this case.  When False (default), the
            function returns None for unreachable goals.

    Returns:
        AStarResult with path, total cost, and expanded count; None if no path exists
        (or no partial path when ``allow_partial`` is False).
    """

    if grid is None or grid.ndim != 2:
        return None
    rows, cols = grid.shape
    sr, sc = int(start[0]), int(start[1])
    gr, gc = int(goal[0]), int(goal[1])
    min_clearance_cells = max(float(min_obstacle_distance_cells), 0.0)

    # Pre-compute cost_view: clamp once (NaN/inf → 0, negatives → 0)
    cost_arr: Optional[np.ndarray] = None
    if cost_grid is not None:
        try:
            cost_arr = np.asarray(cost_grid, dtype=np.float64)
            if cost_arr.shape != grid.shape:
                cost_arr = None
            else:
                cost_arr = np.where(np.isfinite(cost_arr), np.maximum(cost_arr, 0.0), 0.0)
        except Exception:
            cost_arr = None

    # Pre-compute obstacle mask and distance transform
    obstacle_mask = grid > obstacle_threshold
    if unknown_is_obstacle:
        obstacle_mask = np.logical_or(obstacle_mask, grid < 0)

    if min_clearance_cells > 0.0:
        free_mask_u8 = np.logical_not(obstacle_mask).astype(np.uint8)
        distance_to_obstacle = cv2.distanceTransform(free_mask_u8, cv2.DIST_L2, 3)
    else:
        distance_to_obstacle = np.zeros_like(grid, dtype=np.float32)

    # Pre-compute a boolean traversable array so the main loop avoids
    # per-cell Python function-call overhead.
    traversable = np.logical_not(obstacle_mask)
    if min_clearance_cells > 0.0:
        traversable = np.logical_and(traversable, distance_to_obstacle >= min_clearance_cells)

    def _nearest_valid_cell(src_r: int, src_c: int) -> Optional[Tuple[int, int]]:
        """Project an invalid start/goal cell to a nearest valid free cell.

        Uses the pre-computed ``traversable`` mask and a vectorized
        distance search to avoid a Python double-loop.

        Args:
            src_r: Source row index to project.
            src_c: Source column index to project.

        Returns:
            The nearest valid cell ``(row, col)`` within
            ``max_projection_distance_cells``, or ``None`` if no valid cell is
            found in that neighborhood.
        """
        if 0 <= src_r < rows and 0 <= src_c < cols and traversable[src_r, src_c]:
            return src_r, src_c
        max_r = max(0, int(max_projection_distance_cells))
        r_lo = max(0, src_r - max_r)
        r_hi = min(rows, src_r + max_r + 1)
        c_lo = max(0, src_c - max_r)
        c_hi = min(cols, src_c + max_r + 1)
        patch = traversable[r_lo:r_hi, c_lo:c_hi]
        valid_indices = np.argwhere(patch)
        if valid_indices.size == 0:
            return None
        # Shift indices to grid frame
        valid_indices[:, 0] += r_lo
        valid_indices[:, 1] += c_lo
        dists = (valid_indices[:, 0] - src_r) ** 2 + (valid_indices[:, 1] - src_c) ** 2
        best = int(np.argmin(dists))
        return int(valid_indices[best, 0]), int(valid_indices[best, 1])

    def _in_bounds(r: int, c: int) -> bool:
        """Check whether a grid cell index lies inside grid bounds.

        Args:
            r: Candidate row index.
            c: Candidate column index.

        Returns:
            True when ``(r, c)`` is inside the occupancy grid, otherwise False.
        """
        return 0 <= r < rows and 0 <= c < cols

    if not _in_bounds(sr, sc) or not _in_bounds(gr, gc):
        return None
    if not traversable[sr, sc] or not traversable[gr, gc]:
        if not project_start_goal_to_valid:
            return None
        start_proj = _nearest_valid_cell(sr, sc)
        goal_proj = _nearest_valid_cell(gr, gc)
        if start_proj is None or goal_proj is None:
            return None
        sr, sc = start_proj
        gr, gc = goal_proj

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

    # Track the explored cell closest to the goal (by Euclidean distance)
    # so we can return a partial path when the goal is unreachable.
    best_partial_node: Optional[_PrioritizedNode] = None
    best_partial_dist_sq: float = float("inf")

    while open_heap:
        current = heapq.heappop(open_heap)
        cr, cc = current.pos
        if current.pos in closed:
            continue
        closed.add(current.pos)
        expanded += 1

        # Update best partial candidate (closest Euclidean to goal).
        dist_sq = float((cr - gr) ** 2 + (cc - gc) ** 2)
        if dist_sq < best_partial_dist_sq:
            best_partial_dist_sq = dist_sq
            best_partial_node = current

        if cr == gr and cc == gc:
            path = _reconstruct_path(came_from, current.pos)
            return AStarResult(path=path, cost=current.g, expanded=expanded, partial=False)

        for dr, dc in dirs:
            nr, nc = cr + dr, cc + dc
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            if not traversable[nr, nc]:
                continue
            move = _move_cost(dr, dc)
            cell_c = float(cost_arr[nr, nc]) if cost_arr is not None else 0.0
            tentative_g = current.g + move * (1.0 + cost_weight * cell_c)
            if tentative_g >= g_score.get((nr, nc), float("inf")):
                continue
            came_from[(nr, nc)] = current.pos
            g_score[(nr, nc)] = tentative_g
            h = _heuristic(nr, nc, gr, gc, allow_diagonal)
            f = tentative_g + h
            heapq.heappush(open_heap, _PrioritizedNode(f, tentative_g, h, (nr, nc), current.pos))

    # Goal unreachable.  Return best-effort partial path if requested.
    if allow_partial and best_partial_node is not None:
        # Only return a partial path if the best cell is not the start
        # itself (otherwise the path would be a single point and useless).
        if best_partial_node.pos != (sr, sc):
            path = _reconstruct_path(came_from, best_partial_node.pos)
            return AStarResult(
                path=path,
                cost=best_partial_node.g,
                expanded=expanded,
                partial=True,
            )
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
