"""Plan-view footprint figures for the three housing options (see `housing`)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle

from .housing import Housing, convex_hull
from .plotting import BASELINE, CATEGORICAL, GRID, MUTED, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, _savefig

_GAP_MM = 40.0  # space between housings drawn side by side
_LABEL_MM = 38.0  # room under each housing for its size label
_MIN_TILE_MM = 140.0  # tile width so neighbouring labels don't collide


def _rot(deg: float) -> np.ndarray:
    """Counter-clockwise rotation matrix (applied as pts @ R.T)."""
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def _placed(h: Housing, rotate_deg: float):
    """Housing geometry rotated about the origin and re-centred on the origin
    (by footprint bounding-box centre). Returns (outline, rect, [(box, hull, front)])."""
    R = _rot(rotate_deg)
    outline = h.footprint @ R.T
    centre = (outline.min(axis=0) + outline.max(axis=0)) / 2.0
    outline = outline - centre
    rect = h.rect_corners @ R.T - centre
    cams = []
    for b in h.boxes:
        xy = b.corners()[:, :2] @ R.T - centre
        front = convex_hull(b.front_corners()[:, :2] @ R.T - centre)
        cams.append((b, convex_hull(xy), front))
    return outline, rect, cams


def _extent(outline: np.ndarray):
    return outline[:, 0].max() - outline[:, 0].min(), outline[:, 1].max() - outline[:, 1].min()


def _draw(ax, h: Housing, rotate_deg: float, offset, bed_mm: "float | None", show_rect: bool = False):
    outline, rect, cams = _placed(h, rotate_deg)
    off = np.asarray(offset, dtype=float)
    # Print-bed reference square, centred on the housing footprint.
    if bed_mm:
        ax.add_patch(Rectangle(off - bed_mm / 2.0, bed_mm, bed_mm, fill=False, edgecolor=TEXT_PRIMARY,
                               linewidth=1.1, linestyle=(0, (6, 4)), zorder=6))
    ax.add_patch(Polygon(outline + off, closed=True, facecolor="#f1f0ea", edgecolor=TEXT_PRIMARY, linewidth=1.4, zorder=1))
    if show_rect:
        ax.add_patch(Polygon(rect + off, closed=True, fill=False, edgecolor=MUTED, linewidth=0.9, linestyle=(0, (4, 3)), zorder=2))
    zmin_centre = min(b.center[2] for b in h.boxes)
    # Draw lowest-mounted bodies first so higher ones sit on top where they overlap in plan.
    for box, hull, front in sorted(cams, key=lambda c: c[0].center[2]):
        colour = CATEGORICAL[box.index % len(CATEGORICAL)]
        ax.add_patch(Polygon(hull + off, closed=True, facecolor=colour, edgecolor=TEXT_PRIMARY, linewidth=0.8, alpha=0.6, zorder=3))
        ax.add_patch(Polygon(front + off, closed=True, fill=False, edgecolor=TEXT_PRIMARY, linewidth=2.2, zorder=4))
        c = hull.mean(axis=0) + off
        raise_mm = box.center[2] - zmin_centre
        tag = str(box.index + 1) + (f"\n+{raise_mm:.0f}" if raise_mm >= 1.0 else "")
        ax.text(c[0], c[1], tag, ha="center", va="center", fontsize=7, color=TEXT_PRIMARY, zorder=5, fontweight="bold", linespacing=1.1)
    return outline


def _label(ax, text: str, x: float, y: float):
    ax.text(x, y, text, ha="center", va="top", fontsize=7, color=TEXT_SECONDARY, linespacing=1.3)


def _single_layout(housings, ncols, bed_mm):
    """Tile positions (centres) for one-camera housings, boresight pointing up.
    Each tile is at least as large as the print-bed square drawn around it."""
    placed = [_extent(_placed(h, h.boxes[0].az_deg)[0]) for h in housings]
    cw = max(max(w for w, _ in placed) + _GAP_MM, _MIN_TILE_MM, bed_mm + _GAP_MM)
    ch = max(max(h for _, h in placed), bed_mm) + _GAP_MM + _LABEL_MM
    nrows = -(-len(housings) // ncols)
    pos = [((i % ncols) * cw + cw / 2, -(i // ncols) * ch - ch / 2 + _LABEL_MM / 2) for i in range(len(housings))]
    return pos, (ncols * cw, nrows * ch)


def _row_layout(housings, rotate_deg, bed_mm):
    """Side-by-side positions (north up) for multi-camera housings; each cell is
    at least as large as the print-bed square drawn around it."""
    ext = [_extent(_placed(h, rotate_deg)[0]) for h in housings]
    x = 0.0
    pos = []
    for w, _ in ext:
        w = max(w, bed_mm)
        pos.append((x + w / 2 + _GAP_MM / 2, 0.0))
        x += w + _GAP_MM
    return pos, (x, max(max(h for _, h in ext), bed_mm) + _GAP_MM + _LABEL_MM)


def _bed_label(h: Housing, bed_mm: float) -> str:
    rot = h.bed_fit_rotation(bed_mm)
    if rot is None:
        return f"exceeds {bed_mm:g} mm bed"
    return f"fits {bed_mm:g} mm bed" + (f" if rotated {rot:g}°" if rot > 0 else "")


def _finish(ax, w, cy, hmax, title, note):
    """Common view: width w, height hmax (same mm scale on every panel),
    vertically centred on `cy`."""
    ax.set_xlim(0, w)
    ax.set_ylim(cy - hmax / 2, cy + hmax / 2)
    ax.set_aspect("equal")
    ax.set_facecolor(SURFACE)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(BASELINE)
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=10, loc="left", pad=8)
    ax.text(0.01, 0.01, note, transform=ax.transAxes, fontsize=7, color=MUTED, va="bottom", ha="left")


def plot_housing_cases(name: str, pointings, cases: dict, output_path: str, group_size: int = 4, gap_mm: float = 20.0, bed_mm: float = 300.0):
    single = cases["single"]
    grouped = cases[f"group{group_size}"]
    whole = cases["all"]
    n = len(pointings)

    ncols = min(n, 4)
    pos_a, (wa, ha) = _single_layout(single, ncols, 0.0)
    pos_b, (wb, hb) = _row_layout(grouped, 0.0, bed_mm)
    pos_c, (wc, hc) = _row_layout(whole, 0.0, bed_mm)

    hmax = max(ha, hb, hc) + 60.0  # breathing room for labels and corner notes
    mm_in = 22.0 / (wa + wb + wc)  # one common mm -> inch scale for all three panels
    fig = plt.figure(figsize=(22.0 + 0.6, hmax * mm_in + 1.5), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 3, width_ratios=[wa, wb, wc], wspace=0.05, left=0.01, right=0.99, top=0.88, bottom=0.04)

    # --- A: one camera per housing (each drawn with its lens pointing up) ---
    ax = fig.add_subplot(gs[0])
    for h, (cx, cy) in zip(single, pos_a):
        outline = _draw(ax, h, h.boxes[0].az_deg, (cx, cy), None)
        b = h.boxes[0]
        _label(ax, f"az {b.az_deg:g}° alt {b.alt_deg:g}°\n{h.rect_w_mm:.0f} × {h.rect_l_mm:.0f} mm", cx, cy - _extent(outline)[1] / 2 - 4)
    tot_a = sum(h.area_mm2 for h in single) / 100.0
    _finish(ax, wa, _LABEL_MM / 2 - ha / 2, hmax, f"A · One camera per housing ({n} housings)",
            f"lens toward top of each tile · Σ footprint {tot_a:,.0f} cm²")

    # --- B: grouped housings ---
    ax = fig.add_subplot(gs[1])
    for h, (cx, cy) in zip(grouped, pos_b):
        outline = _draw(ax, h, 0.0, (cx, cy), bed_mm)
        _label(ax, f"{h.label}\n{h.rect_w_mm:.0f} × {h.rect_l_mm:.0f} mm · h {h.height_mm:.0f} mm\n{h.area_mm2 / 100.0:,.0f} cm²\n{_bed_label(h, bed_mm)}",
               cx, cy - max(_extent(outline)[1], bed_mm) / 2 - 4)
    tot_b = sum(h.area_mm2 for h in grouped) / 100.0
    _finish(ax, wb, -_LABEL_MM / 2, hmax, f"B · {group_size} cameras per enclosure ({len(grouped)} enclosures)",
            f"north up · Σ footprint {tot_b:,.0f} cm² · +N = camera raised N mm")

    # --- C: everything in one enclosure ---
    ax = fig.add_subplot(gs[2])
    h = whole[0]
    (cx, cy) = pos_c[0]
    outline = _draw(ax, h, 0.0, (cx, cy), bed_mm)
    _label(ax, f"{h.rect_w_mm:.0f} × {h.rect_l_mm:.0f} mm · h {h.height_mm:.0f} mm\n{h.area_mm2 / 100.0:,.0f} cm²\n{_bed_label(h, bed_mm)}",
           cx, cy - max(_extent(outline)[1], bed_mm) / 2 - 4)
    cyc = -_LABEL_MM / 2
    _finish(ax, wc, cyc, hmax, f"C · All {n} cameras in one enclosure", f"north up · footprint {h.area_mm2 / 100.0:,.0f} cm² · +N = camera raised N mm")

    # North arrow + 100 mm scale bar (common scale on all panels).
    top, bot = cyc + hmax / 2, cyc - hmax / 2
    ax.annotate("", xy=(wc - 25, top - 8), xytext=(wc - 25, top - 58), arrowprops={"arrowstyle": "-|>", "color": TEXT_SECONDARY})
    ax.text(wc - 25, top - 6, "N", ha="center", va="bottom", fontsize=8, color=TEXT_SECONDARY)
    ax.plot([wc - 125, wc - 25], [bot + 20] * 2, color=TEXT_PRIMARY, linewidth=2)
    ax.text(wc - 75, bot + 28, "100 mm", ha="center", va="bottom", fontsize=7, color=TEXT_SECONDARY)

    fig.suptitle(
        f"Housing footprints (plan view) — {name}: {n} cameras, 50×50×100 mm modules, thick edge = lens face, ≥{gap_mm:g} mm between cameras, dashed square = {bed_mm:g} × {bed_mm:g} mm print bed",
        color=TEXT_PRIMARY, fontsize=11, x=0.01, ha="left", y=0.97,
    )
    _savefig(fig, output_path)
    plt.close(fig)
