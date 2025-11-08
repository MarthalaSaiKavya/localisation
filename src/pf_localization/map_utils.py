import math
import os
from dataclasses import dataclass

import cv2
import numpy as np
import yaml


def resolve_map_image(map_yaml_path, image_path):
    """Resolve the image path stored in the YAML."""
    if os.path.isabs(image_path):
        if os.path.exists(image_path):
            return image_path
        basename_candidate = os.path.join(
            os.path.dirname(map_yaml_path), os.path.basename(image_path)
        )
        if os.path.exists(basename_candidate):
            return basename_candidate
    candidate = os.path.join(os.path.dirname(map_yaml_path), image_path)
    if os.path.exists(candidate):
        return candidate
    # Fall back to the original path; downstream code will raise if invalid.
    return image_path


@dataclass
class LikelihoodFieldMap:
    resolution: float
    origin: tuple  # (x, y, theta)
    width: int
    height: int
    distance_field: np.ndarray  # meters
    occupancy: np.ndarray  # bool array, True = obstacle/unknown
    free_indices: np.ndarray

    def world_to_grid(self, x, y):
        """Convert world (meters) to grid indices (col, row-from-top)."""
        mx = int(math.floor((x - self.origin[0]) / self.resolution))
        my = int(math.floor((y - self.origin[1]) / self.resolution))
        row = self.height - my - 1
        if mx < 0 or mx >= self.width or row < 0 or row >= self.height:
            return None
        return mx, row

    def grid_to_world(self, col, row):
        """Convert grid indices (col, row-from-top) to world meters."""
        wx = self.origin[0] + (col + 0.5) * self.resolution
        wy = self.origin[1] + ((self.height - row - 0.5) * self.resolution)
        return wx, wy

    def lookup_distance(self, x, y):
        idx = self.world_to_grid(x, y)
        if idx is None:
            return math.inf
        col, row = idx
        return float(self.distance_field[row, col])

    def is_occupied(self, x, y):
        idx = self.world_to_grid(x, y)
        if idx is None:
            return True
        col, row = idx
        return bool(self.occupancy[row, col])

    def lookup_distances(self, xs, ys):
        """Vectorized lookup for arrays of x/y coordinates."""
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        cols = np.floor((xs - self.origin[0]) / self.resolution).astype(np.int32)
        rows_from_bottom = np.floor((ys - self.origin[1]) / self.resolution).astype(
            np.int32
        )
        rows = self.height - rows_from_bottom - 1
        mask = (
            (cols >= 0)
            & (cols < self.width)
            & (rows >= 0)
            & (rows < self.height)
        )
        dists = np.full(xs.shape, np.inf, dtype=np.float32)
        valid_cols = cols[mask]
        valid_rows = rows[mask]
        dists[mask] = self.distance_field[valid_rows, valid_cols]
        return dists

    def sample_free_states(self, count, rng):
        """Sample (x,y,theta) states uniformly from free space."""
        if self.free_indices.size == 0:
            raise RuntimeError("No free cells available for sampling in the map.")
        indices = rng.choice(len(self.free_indices), size=count, replace=True)
        rows_cols = self.free_indices[indices]
        rows = rows_cols[:, 0]
        cols = rows_cols[:, 1]
        xs = self.origin[0] + (cols + 0.5) * self.resolution
        ys = self.origin[1] + ((self.height - rows - 0.5) * self.resolution)
        thetas = rng.uniform(-math.pi, math.pi, size=count)
        return np.stack([xs, ys, thetas], axis=1)


def load_likelihood_field(map_yaml_path):
    """Load the map metadata, occupancy, and distance transform."""
    with open(map_yaml_path, "r") as f:
        meta = yaml.safe_load(f)

    resolution = float(meta["resolution"])
    origin = tuple(meta.get("origin", [0.0, 0.0, 0.0]))
    image_path = resolve_map_image(map_yaml_path, meta["image"])

    raw = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise RuntimeError(f"Failed to read map image at {image_path}")
    if raw.ndim == 3:
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)

    negate = int(meta.get("negate", 0)) != 0
    occ_thresh = float(meta.get("occupied_thresh", 0.65))
    free_thresh = float(meta.get("free_thresh", 0.196))

    raw = raw.astype(np.float32) / 255.0
    if negate:
        occ_prob = raw
    else:
        occ_prob = 1.0 - raw

    occ_mask = occ_prob > occ_thresh
    free_mask = occ_prob < free_thresh
    unknown_mask = ~(occ_mask | free_mask)
    occ_mask = occ_mask | unknown_mask

    inv = (~occ_mask).astype(np.uint8)
    distance_px = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
    distance_m = distance_px * resolution

    free_indices = np.column_stack(np.where(inv > 0))

    height, width = raw.shape
    return LikelihoodFieldMap(
        resolution=resolution,
        origin=origin,
        width=width,
        height=height,
        distance_field=distance_m,
        occupancy=occ_mask,
        free_indices=free_indices,
    )
