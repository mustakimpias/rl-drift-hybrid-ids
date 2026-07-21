"""Concept-drift detection over a stream of per-sample error/loss signals.

Which detector(s) run is controlled by `detector_type`: "adwin" (default),
"page_hinkley", or "both". Both consume a scalar signal per sample — here,
the 0/1 prediction-error indicator — and flag a drift point when their
internal statistics diverge significantly.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from river import drift as river_drift

logger = logging.getLogger("rl_drift_ids")

VALID_DETECTOR_TYPES = ("adwin", "page_hinkley", "both")


@dataclass
class DriftEvent:
    index: int
    detector: str


class DriftDetector:
    """Wraps one or more river drift detectors over an error-indicator stream."""

    def __init__(
        self,
        detector_type: str = "adwin",
        adwin_delta: float = 0.002,
        page_hinkley_threshold: float = 50,
        page_hinkley_min_instances: int = 30,
    ) -> None:
        detector_type = detector_type.lower()
        if detector_type not in VALID_DETECTOR_TYPES:
            raise ValueError(
                f"Unknown drift_detector '{detector_type}'; expected one of {VALID_DETECTOR_TYPES}."
            )
        self.detector_type = detector_type
        self.adwin = (
            river_drift.ADWIN(delta=adwin_delta) if detector_type in ("adwin", "both") else None
        )
        self.page_hinkley = (
            river_drift.PageHinkley(threshold=page_hinkley_threshold, min_instances=page_hinkley_min_instances)
            if detector_type in ("page_hinkley", "both")
            else None
        )
        self.events: list[DriftEvent] = []
        self._index = -1

    def update(self, error_indicator: float) -> dict[str, bool]:
        """Feed one new error value (0=correct, 1=incorrect). Returns which
        detector(s) fired on this step (an inactive detector simply never
        fires, so callers don't need to special-case detector_type).
        """
        self._index += 1
        fired = {"adwin": False, "page_hinkley": False}

        if self.adwin is not None:
            self.adwin.update(error_indicator)
            if self.adwin.drift_detected:
                fired["adwin"] = True
                self.events.append(DriftEvent(index=self._index, detector="adwin"))

        if self.page_hinkley is not None:
            self.page_hinkley.update(error_indicator)
            if self.page_hinkley.drift_detected:
                fired["page_hinkley"] = True
                self.events.append(DriftEvent(index=self._index, detector="page_hinkley"))

        return fired

    @property
    def any_drift_detected_last_step(self) -> bool:
        return bool(self.events) and self.events[-1].index == self._index

    def drift_points(self) -> list[dict[str, Any]]:
        return [{"index": e.index, "detector": e.detector} for e in self.events]

    def total_drift_count(self) -> int:
        return len(self.events)
