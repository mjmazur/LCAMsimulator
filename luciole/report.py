"""Build the final per-camera satellite assignment used for plots and JSON
output, from the selected pointings and the full pass list."""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import CameraSpec, Pointing, SatPass
from .scoring import detectable_samples, pass_arclength_in_fov, pass_in_fov_mask


@dataclass
class CameraCapture:
    sat_pass: SatPass
    arclength_deg: float
    in_fov_mask: "object"  # numpy bool array aligned with sat_pass times


@dataclass
class CameraReport:
    pointing: Pointing
    captures: list = field(default_factory=list)  # list[CameraCapture]

    @property
    def satellite_indices(self):
        return {c.sat_pass.sat_index for c in self.captures}

    @property
    def total_arclength_deg(self) -> float:
        return sum(c.arclength_deg for c in self.captures)


def build_camera_reports(
    passes: list[SatPass],
    selected: list[Pointing],
    fov: CameraSpec,
    std_magnitude: "float | None" = None,
    limiting_magnitude: "float | None" = None,
) -> list[CameraReport]:
    """Only samples detectable at `limiting_magnitude` count as captured,
    matching what the optimizer scored (see `scoring.detectable_samples`)."""
    reports = [CameraReport(pointing=p) for p in selected]
    for sat_pass in passes:
        detect = detectable_samples(sat_pass, std_magnitude, limiting_magnitude)
        for report in reports:
            p = report.pointing
            mask = pass_in_fov_mask(sat_pass, p.az_deg, p.alt_deg, fov, detect)
            if not mask.any():
                continue
            length = pass_arclength_in_fov(sat_pass, p.az_deg, p.alt_deg, fov, detect)
            if length <= 0.0:
                continue
            report.captures.append(
                CameraCapture(sat_pass=sat_pass, arclength_deg=length, in_fov_mask=mask)
            )
    return reports
