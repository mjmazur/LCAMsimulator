"""Housing footprints for fixed-pointing camera modules.

Each camera module is a rectangular box (default 50 x 50 x 100 mm) whose long
axis lies along the optical axis, lens on the front face. Three packaging
options are compared for a set of az/alt pointings:

- one camera per housing,
- several cameras (default 4) per housing, grouped by neighbouring azimuth,
- all cameras in one housing.

Multi-camera layout: camera orientations are fixed by the pointings, their
positions are free. Starting from a collision-free flat ring, a compaction search
slides cameras inward to minimize the plan-view footprint while keeping

- at least `gap_mm` (default 20 mm) between any two camera bodies, and
- every camera's field of view clear of every other camera's body.

Cameras may sit at different heights above the base plane (one can tuck
behind and above another), up to a total body height of `max_height_mm`.
The layout is a heuristic (greedy descent with restarts), not a proven
optimum.

The footprint is the plan (top-down) outline on the mounting surface: the
convex hull of the camera bodies' ground projections, grown by `wall_mm` for
the enclosure wall. The lens window/hood is not modelled.

Coordinates are ENU millimetres (x = East, y = North, z = Up); azimuth is
measured from North towards East, matching the rest of Luciole.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

from .geometry import _boresight_frame

CAMERA_MM = (50.0, 50.0, 100.0)  # cross-section u, v; length along boresight


@dataclass
class Box:
    center: np.ndarray  # (3,)
    axes: np.ndarray  # (3, 3) rows: u, v, boresight
    half: np.ndarray  # (3,)
    index: int = -1
    az_deg: float = 0.0
    alt_deg: float = 0.0

    def corners(self) -> np.ndarray:
        signs = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], dtype=float)
        return self.center + (signs * self.half) @ self.axes

    def front_corners(self) -> np.ndarray:
        """The four corners of the lens face."""
        signs = np.array([[a, b, 1] for a in (-1, 1) for b in (-1, 1)], dtype=float)
        return self.center + (signs * self.half) @ self.axes

    def copy(self) -> "Box":
        return Box(self.center.copy(), self.axes, self.half, self.index, self.az_deg, self.alt_deg)


def camera_box(az_deg, alt_deg, radius_mm: float | None = None, size_mm=CAMERA_MM, index: int = -1) -> Box:
    """Camera box pointing at (az, alt), centred on the origin. With
    `radius_mm`, it is instead placed with its lens-face centre at that
    distance from the origin along the boresight."""
    u, v, w = _boresight_frame(az_deg, alt_deg)
    half = np.array(size_mm, dtype=float) / 2.0
    center = np.zeros(3) if radius_mm is None else (radius_mm - half[2]) * w
    return Box(center, np.stack([u, v, w]), half, index, float(az_deg), float(alt_deg))


def boxes_overlap(a: Box, b: Box, gap_mm: float = 0.0) -> bool:
    """Separating-axis test for two oriented boxes, each grown by half of
    `gap_mm` on every side. False guarantees the bodies are at least `gap_mm`
    apart (separation along one axis is a lower bound on the true distance)."""
    pad = gap_mm / 2.0
    ha, hb = a.half + pad, b.half + pad
    d = b.center - a.center
    if np.linalg.norm(d) > np.linalg.norm(ha) + np.linalg.norm(hb):
        return False  # bounding spheres don't touch
    cross = np.cross(a.axes[:, None, :], b.axes[None, :, :]).reshape(9, 3)
    norm = np.linalg.norm(cross, axis=1)
    keep = norm > 1e-9
    axes = np.vstack([a.axes, b.axes, cross[keep] / norm[keep, None]])  # (3 + 3 + <=9, 3)
    ra = (np.abs(a.axes @ axes.T) * ha[:, None]).sum(axis=0)
    rb = (np.abs(b.axes @ axes.T) * hb[:, None]).sum(axis=0)
    return not bool(np.any(np.abs(axes @ d) > ra + rb))


def _fov_ray_coeffs(fov_deg, nx: int = 5, ny: int = 9) -> np.ndarray:
    """(R, 3) coefficients (a, b, 1) of FOV sample rays a*u + b*v + w, covering
    the rectangular (gnomonic) field of view including its edges and corners."""
    hw = math.tan(math.radians(fov_deg[0]) / 2.0)
    hh = math.tan(math.radians(fov_deg[1]) / 2.0)
    xs, ys = np.meshgrid(np.linspace(-hw, hw, nx), np.linspace(-hh, hh, ny))
    return np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)], axis=-1)


def fov_rays(box: Box, coeffs: np.ndarray):
    """Origins (R,3) and unit directions (R,3) of the FOV sample rays, from the
    centre of the lens face."""
    d = coeffs @ box.axes
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    origin = box.center + box.half[2] * box.axes[2]
    return np.broadcast_to(origin, d.shape), d


def rays_hit_box(origins: np.ndarray, dirs: np.ndarray, box: Box, length_mm: float = 2000.0) -> bool:
    """Whether any ray segment (slab test) passes through `box`."""
    o = (origins - box.center) @ box.axes.T
    d = dirs @ box.axes.T
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (-box.half - o) / d
        t2 = (box.half - o) / d
    parallel = np.abs(d) < 1e-12
    inside = np.abs(o) <= box.half
    t_lo = np.where(parallel, np.where(inside, -np.inf, np.inf), np.minimum(t1, t2))
    t_hi = np.where(parallel, np.where(inside, np.inf, -np.inf), np.maximum(t1, t2))
    near = np.maximum(t_lo.max(axis=1), 0.0)
    far = np.minimum(t_hi.min(axis=1), length_mm)
    return bool(np.any(far >= near))


def camera_fits(i: int, boxes, gap_mm: float, coeffs: np.ndarray, rays=None) -> bool:
    """Whether box `i` keeps `gap_mm` from every other body and neither blocks
    nor is blocked by another camera's field of view. `rays`, if given, holds
    the other cameras' precomputed `fov_rays` (index-aligned with `boxes`)."""
    bi = boxes[i]
    oi, di = fov_rays(bi, coeffs)
    for j, bj in enumerate(boxes):
        if j == i:
            continue
        if boxes_overlap(bi, bj, gap_mm):
            return False
        if rays_hit_box(oi, di, bj):
            return False
        oj, dj = rays[j] if rays is not None else fov_rays(bj, coeffs)
        if rays_hit_box(oj, dj, bi):
            return False
    return True


def all_fit(boxes, gap_mm: float, coeffs: np.ndarray) -> bool:
    return all(camera_fits(i, boxes, gap_mm, coeffs) for i in range(len(boxes)))


def _ring_layout(pts, members, rho_mm: float, size_mm) -> list:
    """Flat (common-height) ring start: every camera's centre at radius rho
    along its own azimuth."""
    boxes = []
    for (az, alt), m in zip(pts, members):
        b = camera_box(az, alt, None, size_mm, m)
        a = math.radians(az)
        b.center = np.array([rho_mm * math.sin(a), rho_mm * math.cos(a), 0.0])
        boxes.append(b)
    return boxes


def _height(boxes) -> float:
    z = np.concatenate([b.corners()[:, 2] for b in boxes])
    return float(z.max() - z.min())


def _objective(boxes, max_height_mm: float = math.inf) -> float:
    """Plan hull area, plus a small pull toward the centroid (so interior
    cameras keep compacting when the hull area is unchanged) and a small
    height cost (so cameras only rise when it shrinks the footprint). Infinite
    if the stack exceeds `max_height_mm`."""
    height = _height(boxes)
    if height > max_height_mm:
        return math.inf
    xy = np.concatenate([b.corners()[:, :2] for b in boxes])
    area = polygon_area(convex_hull(xy))
    c = np.array([b.center[:2] for b in boxes])
    return area + 0.05 * float(np.sum((c - c.mean(axis=0)) ** 2)) + 5.0 * height


def pack_cameras(
    pointings, members, gap_mm: float, size_mm=CAMERA_MM, fov_deg=(10.0, 18.0), restarts: int = 3, seed: int = 0,
    max_height_mm: float = 200.0,
):
    """Compact camera boxes (fixed orientations, free x/y/z) to a small plan
    footprint subject to `gap_mm` between bodies, unobstructed fields of view,
    and a total body height of at most `max_height_mm`. Cameras may sit at
    different heights, so one can tuck behind and above another."""
    pts = [pointings[m] for m in members]
    coeffs = _fov_ray_coeffs(fov_deg)
    rng = np.random.default_rng(seed)

    rho = 20.0
    while True:  # smallest feasible flat starting ring
        if all_fit(_ring_layout(pts, members, rho, size_mm), gap_mm, coeffs):
            break
        rho += 10.0
        if rho > 5000.0:
            raise ValueError("no collision-free starting layout found")

    steps = [64.0, 32.0, 16.0, 8.0, 4.0, 2.0, 1.0]
    dirs8 = [np.array([math.cos(t), math.sin(t)]) for t in np.linspace(0, 2 * math.pi, 8, endpoint=False)]

    best, best_j = None, math.inf
    for r in range(restarts):
        boxes = _ring_layout(pts, members, rho * (1.0 + 0.15 * r), size_mm)
        if not all_fit(boxes, gap_mm, coeffs) or _height(boxes) > max_height_mm:
            continue
        rays = [fov_rays(b, coeffs) for b in boxes]
        cur = _objective(boxes, max_height_mm)
        for step in steps:
            improved = True
            while improved:
                improved = False
                for i in rng.permutation(len(boxes)):
                    centroid = np.mean([b.center[:2] for b in boxes], axis=0)
                    inward = centroid - boxes[i].center[:2]
                    n = np.linalg.norm(inward)
                    xy_dirs = ([inward / n] if n > 1e-9 else []) + dirs8 + [np.zeros(2)]
                    base = boxes[i].center.copy()
                    best_move, best_val = None, cur - 1e-6
                    for d in xy_dirs:
                        for dz in (-step, 0.0, step):
                            if not d.any() and dz == 0.0:
                                continue
                            boxes[i].center = base + np.array([d[0] * step, d[1] * step, dz])
                            val = _objective(boxes, max_height_mm)
                            if val < best_val and camera_fits(i, boxes, gap_mm, coeffs, rays):
                                best_move, best_val = boxes[i].center.copy(), val
                    boxes[i].center = base if best_move is None else best_move
                    if best_move is not None:
                        rays[i] = fov_rays(boxes[i], coeffs)
                        cur = best_val
                        improved = True
        if cur < best_j:
            best, best_j = [b.copy() for b in boxes], cur
    return best


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Counter-clockwise convex hull of 2-D points (Andrew's monotone chain)."""
    pts = sorted({(round(float(x), 9), round(float(y), 9)) for x, y in points})
    if len(pts) <= 2:
        return np.array(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def grow_hull(hull: np.ndarray, margin_mm: float, n_arc: int = 16) -> np.ndarray:
    """Hull offset outward by `margin_mm` (Minkowski sum with a disc)."""
    if margin_mm <= 0:
        return hull
    t = np.linspace(0, 2 * np.pi, n_arc, endpoint=False)
    disc = margin_mm * np.stack([np.cos(t), np.sin(t)], axis=-1)
    return convex_hull((hull[:, None, :] + disc[None, :, :]).reshape(-1, 2))


def polygon_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def min_area_rect(hull: np.ndarray):
    """Minimum-area bounding rectangle of a convex polygon. Returns
    (width, length, corners (4,2)) with width <= length."""
    best = None
    for i in range(len(hull)):
        e = hull[(i + 1) % len(hull)] - hull[i]
        n = np.hypot(*e)
        if n < 1e-9:
            continue
        c, s = e / n
        rot = np.array([[c, s], [-s, c]])  # rotates edge onto +x
        p = hull @ rot.T
        lo, hi = p.min(axis=0), p.max(axis=0)
        area = (hi[0] - lo[0]) * (hi[1] - lo[1])
        if best is None or area < best[0] - 1e-9:
            corners = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]]) @ rot
            best = (area, hi[0] - lo[0], hi[1] - lo[1], corners)
    _, w, l, corners = best
    return (min(w, l), max(w, l), corners)


@dataclass
class Housing:
    label: str
    boxes: list  # list[Box], lifted so the lowest point sits at z=0
    footprint: np.ndarray  # (N,2) convex outline incl. wall clearance
    rect_w_mm: float
    rect_l_mm: float
    rect_corners: np.ndarray
    area_mm2: float
    height_mm: float
    gap_mm: float | None = None
    members: list = field(default_factory=list)  # camera indices (0-based)

    def bed_fit_rotation(self, side_mm: float):
        """Smallest rotation (deg, 0.5 deg steps) at which the footprint fits
        inside a `side_mm` square (e.g. a print bed), or None if it never does."""
        for deg in np.arange(0.0, 90.0, 0.5):
            a = math.radians(deg)
            rot = np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]])
            p = self.footprint @ rot.T
            span = p.max(axis=0) - p.min(axis=0)
            if span[0] <= side_mm and span[1] <= side_mm:
                return float(deg)
        return None

    def fits_square(self, side_mm: float) -> bool:
        return self.bed_fit_rotation(side_mm) is not None

    def summary(self, bed_mm: float | None = None) -> dict:
        d = self._summary()
        if bed_mm is not None:
            d[f"fits_{bed_mm:g}mm_bed"] = self.fits_square(bed_mm)
            d[f"bed_rotation_deg"] = self.bed_fit_rotation(bed_mm)
        return d

    def _summary(self) -> dict:
        return {
            "label": self.label,
            "cameras": [m + 1 for m in self.members],
            "footprint_rect_mm": [round(self.rect_w_mm, 1), round(self.rect_l_mm, 1)],
            "footprint_hull_area_cm2": round(self.area_mm2 / 100.0, 1),
            "height_mm": round(self.height_mm, 1),
            "camera_gap_mm": self.gap_mm,
            "camera_raise_mm": [round(float(b.center[2] - min(x.center[2] for x in self.boxes)), 1) for b in self.boxes],
        }


def build_housing(
    pointings, members, label: str, gap_mm: float = 20.0, wall_mm: float = 5.0, size_mm=CAMERA_MM,
    fov_deg=(10.0, 18.0), max_height_mm: float = 200.0,
) -> Housing:
    """One housing holding the cameras `members` (indices into `pointings`)."""
    if len(members) == 1:
        pt = pointings[members[0]]
        boxes = [camera_box(pt[0], pt[1], None, size_mm, members[0])]
        gap = None
    else:
        boxes = pack_cameras(pointings, members, gap_mm, size_mm, fov_deg, max_height_mm=max_height_mm)
        gap = gap_mm

    zmin = min(b.corners()[:, 2].min() for b in boxes)
    for b in boxes:
        b.center = b.center + np.array([0.0, 0.0, -zmin])
    zmax = max(b.corners()[:, 2].max() for b in boxes)

    xy = np.concatenate([b.corners()[:, :2] for b in boxes])
    outline = grow_hull(convex_hull(xy), wall_mm)
    w, l, corners = min_area_rect(outline)
    return Housing(
        label=label, boxes=boxes, footprint=outline, rect_w_mm=w, rect_l_mm=l, rect_corners=corners,
        area_mm2=polygon_area(outline), height_mm=zmax + wall_mm, gap_mm=gap, members=list(members),
    )


def _az_span(azs) -> float:
    """Azimuth extent (deg) of a contiguous run of sorted azimuths."""
    return (azs[-1] - azs[0]) % 360.0


def group_by_azimuth(pointings, size: int):
    """Partition cameras into groups of `size` (last group may be smaller) of
    neighbouring azimuth, choosing the circular start offset that minimizes the
    summed azimuth extent of the groups. Returns lists of 0-based indices."""
    n = len(pointings)
    order = sorted(range(n), key=lambda i: pointings[i][0])
    best = None
    for start in range(n):
        ring = order[start:] + order[:start]
        groups = [ring[i : i + size] for i in range(0, n, size)]
        cost = sum(_az_span([pointings[i][0] for i in g]) for g in groups)
        if best is None or cost < best[0] - 1e-9:
            best = (cost, groups)
    return best[1]


def build_cases(
    pointings, group_size: int = 4, gap_mm: float = 20.0, wall_mm: float = 5.0, size_mm=CAMERA_MM, fov_deg=(10.0, 18.0),
    max_height_mm: float = 200.0,
) -> dict:
    n = len(pointings)
    kw = dict(gap_mm=gap_mm, wall_mm=wall_mm, size_mm=size_mm, fov_deg=fov_deg, max_height_mm=max_height_mm)
    single = [build_housing(pointings, [i], f"Cam {i + 1}", **kw) for i in range(n)]
    grouped = [
        build_housing(pointings, g, f"Enclosure {k + 1} (cams {', '.join(str(i + 1) for i in sorted(g))})", **kw)
        for k, g in enumerate(group_by_azimuth(pointings, group_size))
    ]
    everything = [build_housing(pointings, list(range(n)), f"All {n} cameras", **kw)]
    return {"single": single, f"group{group_size}": grouped, "all": everything}


def load_pointings(path: str):
    with open(path, encoding="utf-8") as f:
        cams = json.load(f)["cameras"]
    return [(c["az_deg"], c["alt_deg"]) for c in cams]


def main(argv=None):
    p = argparse.ArgumentParser(prog="luciole.housing", description="Plot housing footprints for camera pointings.")
    p.add_argument("--pointing-dir", action="append", help="Run output dir holding pointings_leo.json (repeatable)")
    p.add_argument("--pointings-name", default="pointings_leo.json")
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--max-height-mm", type=float, default=200.0,
                   help="Cap on total camera-body height; cameras may be raised/stacked up to this (default 200)")
    p.add_argument("--gap-mm", type=float, default=20.0, help="Minimum gap between camera bodies (default 20)")
    p.add_argument("--wall-mm", type=float, default=5.0, help="Wall clearance around the cameras (default 5)")
    p.add_argument("--fov-width", type=float, default=10.0, help="Camera FOV width, deg")
    p.add_argument("--fov-height", type=float, default=18.0, help="Camera FOV height, deg")
    p.add_argument("--camera-mm", type=float, nargs=3, default=CAMERA_MM, metavar=("U", "V", "LEN"))
    p.add_argument("--bed-mm", type=float, default=300.0, help="Print-bed square drawn around each option (default 300)")
    p.add_argument("--output-dir", default=os.path.join("output", "housing"))
    args = p.parse_args(argv)

    from .housing_plot import plot_housing_cases

    dirs = args.pointing_dir or [os.path.join("output", d) for d in ("8cam", "10cam", "12cam-3")]
    os.makedirs(args.output_dir, exist_ok=True)
    summary = {}
    for d in dirs:
        name = os.path.basename(os.path.normpath(d))
        path = os.path.join(d, args.pointings_name)
        if not os.path.exists(path):  # location runs name files pointings_<group>_<city>.json
            found = sorted(glob.glob(os.path.join(d, "pointings_*.json")))
            if len(found) != 1:
                raise SystemExit(f"{path} not found and {len(found)} pointings_*.json candidates in {d}; use --pointings-name")
            path = found[0]
        pointings = load_pointings(path)
        cases = build_cases(
            pointings, args.group_size, args.gap_mm, args.wall_mm, tuple(args.camera_mm), (args.fov_width, args.fov_height),
            args.max_height_mm,
        )
        plot_housing_cases(name, pointings, cases, os.path.join(args.output_dir, f"housing_{name}.png"), args.group_size, args.gap_mm, args.bed_mm)
        summary[name] = {k: [h.summary(args.bed_mm) for h in v] for k, v in cases.items()}
        print(f"\n{name} ({len(pointings)} cameras)", flush=True)
        for case, housings in cases.items():
            for h in housings:
                s = h.summary(args.bed_mm)
                print(f"  {case:>7} | {s['label']:<32} {s['footprint_rect_mm'][0]:>6.0f} x {s['footprint_rect_mm'][1]:>6.0f} mm"
                      f"  hull {s['footprint_hull_area_cm2']:>8.0f} cm2  h {s['height_mm']:>5.0f} mm  fits bed: {s[f'fits_{args.bed_mm:g}mm_bed']}", flush=True)
    with open(os.path.join(args.output_dir, "housing_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
