"""All matplotlib figures: sky score heatmap, sky tracks, timeline, sun
altitude, and per-camera summary."""

from __future__ import annotations

import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .geometry import fov_boundary_azalt
from .models import CameraSpec, Pointing
from .passes import contiguous_true_runs
from .timewindow import sun_altitude_deg

# Fixed categorical order (never cycled/re-sorted) -- see dataviz skill.
CATEGORICAL = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"]
)
# Second sequential context (satellite-count density, alongside the
# arc-length-weighted score) takes the next categorical slot's hue, per the
# dataviz palette's rule for two simultaneous magnitude encodings.
SEQUENTIAL_ORANGE = LinearSegmentedColormap.from_list(
    "seq_orange", ["#fbe0d0", "#f2a97e", "#eb6834", "#8a3212"]
)


def _savefig(fig, output_path: str, retries: int = 3, delay_sec: float = 0.5):
    """`fig.savefig` with a few retries -- on Windows, a file that was just
    written/read by another process (antivirus scan, a previous run's own
    output, an image viewer) can transiently fail to open with
    OSError/PermissionError; a short retry avoids losing an expensive
    multi-minute propagation run to that flakiness."""
    for attempt in range(retries):
        try:
            fig.savefig(output_path, dpi=150, facecolor=SURFACE)
            return
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(delay_sec)


def _camera_color(idx: int) -> str:
    return CATEGORICAL[idx % len(CATEGORICAL)]


def _style_axes(ax, polar=False):
    ax.set_facecolor(SURFACE)
    ax.title.set_color(TEXT_PRIMARY)
    if polar:
        ax.grid(color=GRID, linewidth=0.7)
        ax.spines["polar"].set_color(BASELINE)
        ax.tick_params(colors=MUTED)
    else:
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(BASELINE)
        ax.tick_params(colors=MUTED, labelcolor=TEXT_SECONDARY)
        ax.grid(color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)


def _new_polar_fig():
    fig = plt.figure(figsize=(7.5, 7.5), facecolor=SURFACE)
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    _style_axes(ax, polar=True)
    return fig, ax


def _plot_fov_outline(ax, pointing: Pointing, color, label=None, linestyle="-", linewidth=1.6):
    az_deg, alt_deg = fov_boundary_azalt(
        pointing.az_deg, pointing.alt_deg, pointing.fov.fov_w_deg, pointing.fov.fov_h_deg
    )
    theta = np.unwrap(np.radians(az_deg))
    theta = np.append(theta, theta[0])
    r = np.append(90.0 - alt_deg, 90.0 - alt_deg[0])
    ax.plot(theta, r, color=color, linewidth=linewidth, linestyle=linestyle, label=label, zorder=5)


def _draw_horizon_and_floor(ax, min_elevation_deg: float):
    """Always show the full sky down to the horizon (r=90), with a dashed
    reference circle marking the minimum camera-pointing altitude -- that
    floor is no longer the plot's edge, so it needs its own marker."""
    ax.set_ylim(0, 90)
    ax.set_yticklabels([])
    if min_elevation_deg > 0:
        floor_r = 90.0 - min_elevation_deg
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(theta, np.full_like(theta, floor_r), color=MUTED, linewidth=1.0, linestyle=":", zorder=3)
        # Placed in the SE quadrant, off any 45-degree tick angle and clear
        # of the camera legend anchored at the plot's bottom-left.
        ax.text(
            np.radians(157.5), floor_r, f"  {min_elevation_deg:g}° min pointing alt",
            color=MUTED, fontsize=7, va="center", ha="left",
            bbox={"facecolor": SURFACE, "edgecolor": "none", "alpha": 0.75, "pad": 1.0},
        )


def plot_sky_scores(
    grid,
    cell_contributions,
    selected: list,
    min_elevation_deg: float,
    az_step_deg: float,
    alt_step_deg: float,
    output_path: str,
    pointing_floor_deg: "float | None" = None,
):
    """`grid`/`cell_contributions` are expected to span the full sky (down
    to `min_elevation_deg`, typically 0 -- the horizon) so satellite traffic
    is shown everywhere, even below where a camera could actually be
    pointed. `pointing_floor_deg` (defaults to `min_elevation_deg` if not
    given separately) draws that camera-placement floor as its own
    reference circle rather than cropping the plot to it.
    """
    azs = np.arange(0.0, 360.0, az_step_deg)
    alts = np.arange(min_elevation_deg, 90.0, alt_step_deg)
    scores = np.array([sum(c.values()) for c in cell_contributions]).reshape(
        len(alts), len(azs)
    )

    fig, ax = _new_polar_fig()
    theta_edges = np.radians(np.append(azs, azs[-1] + az_step_deg))
    r_edges = 90.0 - np.append(alts, alts[-1] + alt_step_deg)
    mesh = ax.pcolormesh(
        theta_edges, r_edges, scores, cmap=SEQUENTIAL_BLUE, shading="flat", zorder=1
    )
    cbar = fig.colorbar(mesh, ax=ax, pad=0.1, shrink=0.75)
    cbar.set_label("Total in-FOV arc-length (deg)", color=TEXT_SECONDARY)
    cbar.ax.tick_params(colors=MUTED)

    for i, pointing in enumerate(selected):
        _plot_fov_outline(ax, pointing, _camera_color(i), label=f"Camera {i + 1}")

    _draw_horizon_and_floor(ax, pointing_floor_deg if pointing_floor_deg is not None else min_elevation_deg)
    ax.set_title("Candidate pointing score & selected cameras", color=TEXT_PRIMARY, pad=20)
    if ax.get_legend_handles_labels()[0]:  # none when no camera could be placed
        ax.legend(loc="lower left", bbox_to_anchor=(-0.15, -0.1), frameon=False, labelcolor=TEXT_SECONDARY)
    _savefig(fig, output_path)
    plt.close(fig)


def plot_sky_density(
    grid,
    cell_contributions,
    selected: list,
    min_elevation_deg: float,
    az_step_deg: float,
    alt_step_deg: float,
    output_path: str,
    pointing_floor_deg: "float | None" = None,
):
    """Sky map of raw satellite-count density (distinct trackable satellites
    per candidate cell) -- unlike `plot_sky_scores`, this ignores arc-length
    weighting entirely, so it isolates "how much satellite traffic passes
    through this patch of sky" from the optimizer's combined objective.
    See `plot_sky_scores` for the `min_elevation_deg` vs. `pointing_floor_deg`
    distinction.
    """
    azs = np.arange(0.0, 360.0, az_step_deg)
    alts = np.arange(min_elevation_deg, 90.0, alt_step_deg)
    counts = np.array([len(c) for c in cell_contributions]).reshape(len(alts), len(azs))

    fig, ax = _new_polar_fig()
    theta_edges = np.radians(np.append(azs, azs[-1] + az_step_deg))
    r_edges = 90.0 - np.append(alts, alts[-1] + alt_step_deg)
    mesh = ax.pcolormesh(
        theta_edges, r_edges, counts, cmap=SEQUENTIAL_ORANGE, shading="flat", zorder=1
    )
    cbar = fig.colorbar(mesh, ax=ax, pad=0.1, shrink=0.75)
    cbar.set_label("Distinct trackable satellites", color=TEXT_SECONDARY)
    cbar.ax.tick_params(colors=MUTED)

    for i, pointing in enumerate(selected):
        _plot_fov_outline(ax, pointing, _camera_color(i), label=f"Camera {i + 1}")

    _draw_horizon_and_floor(ax, pointing_floor_deg if pointing_floor_deg is not None else min_elevation_deg)
    ax.set_title("Satellite number density & selected cameras", color=TEXT_PRIMARY, pad=20)
    if ax.get_legend_handles_labels()[0]:  # none when no camera could be placed
        ax.legend(loc="lower left", bbox_to_anchor=(-0.15, -0.1), frameon=False, labelcolor=TEXT_SECONDARY)
    _savefig(fig, output_path)
    plt.close(fig)


def plot_sky_tracks(camera_reports, min_elevation_deg: float, output_path: str):
    fig, ax = _new_polar_fig()

    # Selected pointings are often only a few degrees apart (deliberately, to
    # chain a satellite's track across adjacent cameras), so the same track
    # segment can sit inside several cameras' FOVs at once. A small per-camera
    # radial offset keeps every camera's contribution visible as a distinct
    # parallel line instead of later, opaque draws hiding earlier ones.
    n_cams = max(len(camera_reports), 1)
    jitter_deg = 0.35

    for i, report in enumerate(camera_reports):
        color = _camera_color(i)
        _plot_fov_outline(ax, report.pointing, color, linestyle="--", linewidth=1.2)
        offset = (i - (n_cams - 1) / 2.0) * jitter_deg
        for capture in report.captures:
            for start, end in contiguous_true_runs(capture.in_fov_mask):
                az = capture.sat_pass.az_deg[start:end]
                alt = capture.sat_pass.alt_deg[start:end]
                theta = np.unwrap(np.radians(az))
                r = 90.0 - alt + offset
                ax.plot(theta, r, color=color, linewidth=1.8, alpha=0.9, zorder=4)

    handles = [
        plt.Line2D([0], [0], color=_camera_color(i), linewidth=2, label=f"Camera {i + 1}")
        for i in range(len(camera_reports))
    ]
    _draw_horizon_and_floor(ax, min_elevation_deg)
    ax.set_title("Captured satellite tracks by camera", color=TEXT_PRIMARY, pad=20)
    if handles:
        ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.15, -0.1), frameon=False, labelcolor=TEXT_SECONDARY)
    _savefig(fig, output_path)
    plt.close(fig)


MAX_TIMELINE_ROWS = 60


def plot_timeline(camera_reports, t_start, output_path: str):
    # One row per distinct captured satellite pass-set, ordered by first
    # appearance. Each camera gets its own thin sub-lane within the row so
    # that simultaneous coverage by several overlapping cameras (common,
    # since chaining relies on adjacent/overlapping FOVs) stays visible as
    # parallel bars instead of later cameras opaquely occluding earlier ones.
    n_cams = len(camera_reports)
    rows: dict[int, dict] = {}
    for i, report in enumerate(camera_reports):
        for capture in report.captures:
            key = id(capture.sat_pass)
            row = rows.setdefault(
                key,
                {"name": capture.sat_pass.sat_name, "first_t": capture.sat_pass.times_jd[0], "segments": [], "cams": set()},
            )
            row["cams"].add(i)
            hours = (capture.sat_pass.times_jd - t_start.tt) * 24.0
            for start, end in contiguous_true_runs(capture.in_fov_mask):
                row["segments"].append((hours[start], hours[end - 1] - hours[start], i))

    n_total = len(rows)
    # With a large catalog there can be thousands of captured satellites --
    # cap what's drawn (both for legibility and so the figure doesn't become
    # enormous) by keeping the rows that best demonstrate chaining: most
    # distinct cameras crossed, then most total time in-FOV.
    all_rows = list(rows.values())
    if n_total > MAX_TIMELINE_ROWS:
        all_rows.sort(
            key=lambda r: (len(r["cams"]), sum(w for _, w, _ in r["segments"])), reverse=True
        )
        all_rows = all_rows[:MAX_TIMELINE_ROWS]

    ordered = sorted(all_rows, key=lambda r: r["first_t"])

    lane_h = 0.8 / max(n_cams, 1)
    fig, ax = plt.subplots(figsize=(10, max(3, 0.35 * len(ordered) + 1)), facecolor=SURFACE)
    for y, row in enumerate(ordered):
        for start_h, width_h, cam_idx in row["segments"]:
            lane_y = y - 0.4 + cam_idx * lane_h
            ax.broken_barh(
                [(start_h, max(width_h, 0.002))],
                (lane_y, lane_h * 0.85),
                facecolors=_camera_color(cam_idx),
                edgecolors="none",
            )

    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels([r["name"] for r in ordered], fontsize=8, color=TEXT_SECONDARY)
    ax.set_xlabel("Hours since window start", color=TEXT_SECONDARY)
    title = "Satellite passes through selected cameras"
    if n_total > len(ordered):
        title += f" (top {len(ordered)} of {n_total}, by cameras chained then duration)"
    ax.set_title(title, color=TEXT_PRIMARY)
    _style_axes(ax)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=_camera_color(i), label=f"Camera {i + 1}")
        for i in range(len(camera_reports))
    ]
    if handles:
        ax.legend(handles=handles, loc="upper right", frameon=False, labelcolor=TEXT_SECONDARY, ncol=min(len(camera_reports), 6))
    fig.tight_layout()
    _savefig(fig, output_path)
    plt.close(fig)


def plot_sun_altitude(eph, topos, ts, t_start, t_end, dark_intervals, threshold_deg, output_path: str):
    n = 400
    jd = np.linspace(t_start.tt, t_end.tt, n)
    times = ts.tt_jd(jd)
    hours = (jd - t_start.tt) * 24.0
    alt = sun_altitude_deg(eph, topos, times)

    fig, ax = plt.subplots(figsize=(9, 4), facecolor=SURFACE)
    for d0, d1 in dark_intervals:
        h0 = (d0.tt - t_start.tt) * 24.0
        h1 = (d1.tt - t_start.tt) * 24.0
        ax.axvspan(h0, h1, color=CATEGORICAL[0], alpha=0.12, zorder=0)

    ax.plot(hours, alt, color=TEXT_PRIMARY, linewidth=1.8, zorder=2)
    ax.axhline(threshold_deg, color=CATEGORICAL[1], linestyle="--", linewidth=1.2, label=f"Dark threshold ({threshold_deg:g} deg)")
    ax.axhline(0.0, color=MUTED, linestyle=":", linewidth=1.0, label="Horizon")

    ax.set_xlabel("Hours since window start", color=TEXT_SECONDARY)
    ax.set_ylabel("Sun altitude (deg)", color=TEXT_SECONDARY)
    ax.set_title("Sun altitude & dark-sky windows used", color=TEXT_PRIMARY)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper right", frameon=False, labelcolor=TEXT_SECONDARY)
    _style_axes(ax)
    fig.tight_layout()
    _savefig(fig, output_path)
    plt.close(fig)


def _camera_ticks(ax, ks):
    """Camera-number x ticks: small font, thinned so labels never touch."""
    n = len(ks)
    step = 1 if n <= 16 else 2 if n <= 32 else 5
    ax.set_xticks([k for k in ks if (k - ks[0]) % step == 0])
    ax.tick_params(axis="both", labelsize=8)


def plot_summary(camera_reports, output_path: str):
    n = len(camera_reports)
    labels = [str(i + 1) for i in range(n)]
    counts = [len(r.satellite_indices) for r in camera_reports]
    arclengths = [r.total_arclength_deg for r in camera_reports]
    colors = [_camera_color(i) for i in range(len(camera_reports))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(10, 0.4 * n * 2 + 2), 4), facecolor=SURFACE)

    ax1.bar(labels, counts, color=colors)
    ax1.set_ylabel("Distinct satellites captured", color=TEXT_SECONDARY)
    ax1.set_title("Satellites per camera", color=TEXT_PRIMARY)
    _style_axes(ax1)

    ax2.bar(labels, arclengths, color=colors)
    ax2.set_ylabel("Total arc-length (deg)", color=TEXT_SECONDARY)
    ax2.set_title("Arc-length per camera", color=TEXT_PRIMARY)
    _style_axes(ax2)
    for ax in (ax1, ax2):
        ax.set_xlabel("Camera", color=TEXT_SECONDARY, fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
        if n > 16:
            ax.set_xticks(range(0, n, 2 if n <= 32 else 5))

    fig.tight_layout()
    _savefig(fig, output_path)
    plt.close(fig)


def plot_camera_sweep(sweep, output_path: str):
    """Value vs number of cameras (left) and each camera's marginal gain (right)."""
    ks = [r.cameras for r in sweep.rows]
    sats = [r.satellites for r in sweep.rows]
    gains = [r.gain for r in sweep.rows]
    label = {"satellites": "Distinct satellites", "objective": "Pass-utility objective", "arc": "Arc-length (deg)"}[sweep.metric]
    curve = {"satellites": sats, "objective": [r.objective for r in sweep.rows], "arc": [r.arc_deg for r in sweep.rows]}[sweep.metric]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), facecolor=SURFACE)
    ax1.plot(ks, curve, marker="o", color=CATEGORICAL[0], linewidth=1.8)
    ax1.set_xlabel("Cameras", color=TEXT_SECONDARY)
    ax1.set_ylabel(label + " observed", color=TEXT_SECONDARY)
    ax1.set_title("Observed vs number of cameras", color=TEXT_PRIMARY)
    ax2.bar(ks, gains, color=CATEGORICAL[0], alpha=0.85)
    thresh = sweep.min_gain_fraction * (gains[0] if gains else 0.0)
    ax2.axhline(thresh, color=CATEGORICAL[1], linestyle="--", linewidth=1.2, label=f"threshold ({sweep.min_gain_fraction:.0%} of camera 1, {sweep.patience} in a row)")
    ax2.set_xlabel("Camera added", color=TEXT_SECONDARY)
    ax2.set_ylabel("Marginal gain", color=TEXT_SECONDARY)
    ax2.set_title("What each added camera contributes", color=TEXT_PRIMARY)
    ax2.legend(frameon=False, labelcolor=TEXT_SECONDARY, fontsize=8)
    for ax in (ax1, ax2):
        _style_axes(ax)
        _camera_ticks(ax, ks)
        ax.set_xlabel(ax.get_xlabel(), fontsize=9, color=TEXT_SECONDARY)
        ax.set_ylabel(ax.get_ylabel(), fontsize=9, color=TEXT_SECONDARY)
    ax1.axvline(sweep.recommended, color=CATEGORICAL[1], linewidth=1.2, alpha=0.7)
    ax2.axvline(sweep.recommended + 0.5, color=CATEGORICAL[1], linewidth=1.2, alpha=0.7)  # between kept and dropped cameras
    ax1.annotate(f"recommended: {sweep.recommended}", (sweep.recommended, curve[sweep.recommended - 1]),
                 textcoords="offset points", xytext=(8, -14), color=TEXT_PRIMARY, fontsize=9)
    fig.tight_layout()
    _savefig(fig, output_path)
    plt.close(fig)
