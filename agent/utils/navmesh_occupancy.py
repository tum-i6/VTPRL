"""
Helpers to convert Unity navmesh data into occupancy grids for planners.

Inputs are plain vertices/indices from Unity (meters) plus optional resolution and
padding. Outputs are numpy arrays with occupancy values: 0 = free, 100 = occupied,
-1 = unknown.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class OccupancyGridResult:
    """Container for occupancy grid outputs.

    Attributes:
        grid: int8 array with values in {0, 100, -1} for free/occupied/unknown.
        resolution: Cell size in metres.
        origin: (x, y) origin of the grid in world coordinates.
        width: Number of columns in the grid.
        height: Number of rows in the grid.
        costmap: Optional float32 array with continuous obstacle proximity costs.
    """

    grid: np.ndarray
    resolution: float
    origin: Tuple[float, float]
    width: int
    height: int
    costmap: Optional[np.ndarray] = None


def _coerce_vertices(vertices: Sequence[Sequence[float]]) -> np.ndarray:
    """Convert vertices to a float XY array with shape (N, 2).

    Accepts flat or nested sequences of 2D or 3D points. If 3D, drops Y-up height
    to work in the XZ plane, which matches Unity's navmesh export conventions.

    Raises:
        ValueError: If the input cannot produce at least two columns or is empty.

    Returns:
        np.ndarray: Float array shaped (N, 2) containing XY coordinates.
    """
    array = np.asarray(vertices, dtype=float)
    if array.ndim == 1:
        # Prefer 2D interpretation (X, Z pairs) over 3D to match the current
        # Unity export format and avoid misrealigning columns.
        if array.size % 2 == 0:
            array = array.reshape(-1, 2)
        elif array.size % 3 == 0:
            array = array.reshape(-1, 3)
    if array.shape[1] >= 3:
        array = array[:, (0, 2)]
    elif array.shape[1] >= 2:
        array = array[:, :2]
    else:
        raise ValueError("vertices must have at least 2 columns")
    if array.size == 0:
        raise ValueError("vertices array is empty")
    return array


def _coerce_faces(faces: Iterable[Iterable[int]]) -> List[List[int]]:
    """Normalize faces to triangle index triplets.

    Faces with fewer than three indices are discarded. The function keeps only the
    first three indices of larger polygons, as rasterization expects triangles.

    Raises:
        ValueError: If no valid triangle faces are provided.

    Returns:
        List[List[int]]: Triangle faces as lists of three indices.
    """
    face_list: List[List[int]] = []
    for face in faces:
        entries = list(face)
        if len(entries) < 3:
            continue
        face_list.append(entries[:3])
    if not face_list:
        raise ValueError("no triangle faces provided")
    return face_list


def navmesh_to_occupancy_grid(
    vertices: Sequence[Sequence[float]],
    faces: Sequence[Sequence[int]],
    resolution: float = 0.05,
    padding_cells: int = 2,
    rotation_deg: float = 0.0,
    build_costmap: bool = True,
    inflate_radius: float = 2.5,
    cost_decay_power: float = 2.5,
    obstacle_cost: float = 1.0,
    unknown_cost: float = 1.0,
    free_cost: float = 0.0,
) -> OccupancyGridResult:
    """Convert a navmesh into a 2D occupancy grid and optional costmap.

    Args:
        vertices: Iterable of (x, y[, z]) coordinates in metres. Flat or nested
            inputs are accepted; 3D coordinates are projected to the XZ plane.
        faces: Iterable of triangle index triplets. Both 0-based and 1-based
            indices are supported.
        resolution: Metres per grid cell.
        padding_cells: Extra dilation (in cells) applied to obstacles.
        rotation_deg: Rotation (degrees) applied after rasterization.
        build_costmap: Whether to compute a float costmap near obstacles.
        inflate_radius: Radius in metres over which costs decay from obstacles.
        cost_decay_power: Exponent shaping the decay curve of costs.
        obstacle_cost: Cost assigned to occupied cells.
        unknown_cost: Cost assigned to unknown cells.
        free_cost: Base cost assigned to free cells.

    Returns:
        OccupancyGridResult: Occupancy grid metadata and optional costmap.
    """

    verts_xy = _coerce_vertices(vertices)
    faces_tri = _coerce_faces(faces)

    res = max(float(resolution), 1e-4)

    min_x = float(np.min(verts_xy[:, 0]))
    min_y = float(np.min(verts_xy[:, 1]))
    max_x = float(np.max(verts_xy[:, 0]))
    max_y = float(np.max(verts_xy[:, 1]))

    # Always leave a one-cell outer border so areas beyond the navmesh remain occupied.
    border_cells = 1
    total_pad = int(padding_cells) + border_cells
    scale = 1.0 / res
    width = int(math.ceil((max_x - min_x) * scale)) + 2 * total_pad
    height = int(math.ceil((max_y - min_y) * scale)) + 2 * total_pad
    width = max(width, 1)
    height = max(height, 1)

    # Start as fully occupied; navmesh triangles carve free space (0 = free, 100 = occupied)
    canvas = np.full((height, width), fill_value=100, dtype=np.uint8)

    # Determine if faces are 1-based; if min index > 0, subtract 1.
    min_index = min(min(face) for face in faces_tri)
    index_offset = 1 if min_index >= 1 else 0

    origin_x = min_x - total_pad * res
    origin_y = min_y - total_pad * res
    shifted = verts_xy - np.array([[origin_x, origin_y]], dtype=float)
    pix = shifted * scale - 0.5

    # Rasterise each triangle using cv2.fillConvexPoly (overwrite mode).
    # We intentionally avoid cv2.fillPoly with multiple contours because
    # it uses the *even-odd fill rule*: a point covered by an even number
    # of overlapping polygons is treated as "outside" and left unfilled.
    # The NavMeshProvider patches add extra triangles that overlap the
    # base mesh near the robot, so the even-odd rule causes those cells
    # to stay occupied.  fillConvexPoly writes unconditionally (no XOR)
    # and is optimised for convex shapes — triangles are always convex.
    face_arr = np.asarray(faces_tri, dtype=np.intp) - index_offset
    # Filter out faces with out-of-bounds vertex indices.
    num_verts = len(pix)
    valid_mask = np.all((face_arr >= 0) & (face_arr < num_verts), axis=1)
    face_arr = face_arr[valid_mask]
    if face_arr.size > 0:
        all_pts = pix[face_arr]  # shape (num_faces, 3, 2)
        all_pts = np.round(all_pts).astype(np.int32)
        all_pts[:, :, 0] = np.clip(all_pts[:, :, 0], 0, width - 1)
        all_pts[:, :, 1] = np.clip(all_pts[:, :, 1], 0, height - 1)
        for tri_pts in all_pts:
            cv2.fillConvexPoly(canvas, tri_pts, color=0)

    if padding_cells > 0:
        kernel_size = max(1, int(padding_cells))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        canvas = cv2.dilate(canvas, kernel, iterations=1)

    if rotation_deg:
        canvas = _rotate_image(canvas, rotation_deg, background=0)

    grid_int8 = canvas.astype(np.int8)

    costmap = None
    if build_costmap:
        costmap = _build_costmap(
            grid_int8,
            res,
            inflate_radius=inflate_radius,
            decay_power=cost_decay_power,
            obstacle_cost=obstacle_cost,
            unknown_cost=unknown_cost,
            free_cost=free_cost,
        )

    return OccupancyGridResult(
        grid=grid_int8,
        resolution=res,
        origin=(origin_x, origin_y),
        width=grid_int8.shape[1],
        height=grid_int8.shape[0],
        costmap=costmap,
    )


def _rotate_image(image: np.ndarray, angle_deg: float, background: int = 0) -> np.ndarray:
    """Rotate a binary image around its center using nearest-neighbor sampling.

    Args:
        image: 2D array representing the occupancy raster to rotate.
        angle_deg: Rotation angle in degrees, counter-clockwise.
        background: Fill value for regions introduced by rotation.

    Returns:
        np.ndarray: Rotated image with the same dtype as the input.
    """
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos_val = abs(rot_mat[0, 0])
    sin_val = abs(rot_mat[0, 1])
    new_w = int((height * sin_val) + (width * cos_val))
    new_h = int((height * cos_val) + (width * sin_val))
    rot_mat[0, 2] += (new_w / 2) - center[0]
    rot_mat[1, 2] += (new_h / 2) - center[1]
    rotated = cv2.warpAffine(image, rot_mat, (new_w, new_h), borderValue=background, flags=cv2.INTER_NEAREST)
    return rotated


def _build_costmap(
    grid: np.ndarray,
    resolution: float,
    inflate_radius: float,
    decay_power: float,
    obstacle_cost: float,
    unknown_cost: float,
    free_cost: float,
) -> np.ndarray:
    """Construct a continuous costmap that rises near obstacles.

    Args:
        grid: Occupancy grid with free (0), occupied (100), and unknown (-1) cells.
        resolution: Cell size in metres.
        inflate_radius: Distance over which obstacle proximity costs are applied.
        decay_power: Exponent controlling how quickly costs decay with distance.
        obstacle_cost: Cost value assigned to occupied cells.
        unknown_cost: Cost value assigned to unknown cells.
        free_cost: Base cost value assigned to free cells.

    Returns:
        np.ndarray: Float32 costmap aligned with the input grid.
    """

    grid = np.asarray(grid)
    costmap = np.full(grid.shape, free_cost, dtype=np.float32)

    obstacle_mask = grid > 0
    unknown_mask = grid < 0
    free_mask = grid == 0

    if inflate_radius > 0.0 and decay_power > 0.0 and np.any(free_mask):
        free_uint8 = free_mask.astype(np.uint8)
        # Distance to nearest obstacle/unknown in metres.
        dist_pix = cv2.distanceTransform(free_uint8, cv2.DIST_L2, 3)
        dist_m = dist_pix * float(resolution)

        normalized = np.clip((inflate_radius - dist_m) / max(inflate_radius, 1e-6), 0.0, 1.0)
        inflated = np.power(normalized, decay_power, dtype=np.float32)
        costmap[free_mask] = np.maximum(costmap[free_mask], inflated[free_mask])

    costmap[obstacle_mask] = float(obstacle_cost)
    costmap[unknown_mask] = float(unknown_cost)
    return costmap.astype(np.float32, copy=False)
