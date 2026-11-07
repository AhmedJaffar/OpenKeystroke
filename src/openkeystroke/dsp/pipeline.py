"""Configurable preprocessing pipeline that records lineage for every transformation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import PreprocessConfig
from ..errors import DSPError
from . import ops


@dataclass
class Processed:
    waveforms: np.ndarray
    sample_rate: int
    lineage: list[dict[str, Any]] = field(default_factory=list)


class PreprocessPipeline:
    def __init__(self, cfg: PreprocessConfig) -> None:
        self.cfg = cfg

    def __call__(self, x: np.ndarray, sample_rate: int) -> Processed:
        x = ops.as_batch(x)
        sr = ops.check_sr(sample_rate)
        lineage: list[dict[str, Any]] = [{"op": "input", "sample_rate": sr, "shape": list(x.shape)}]
        c = self.cfg
        if c.target_sample_rate and c.target_sample_rate != sr:
            x = ops.resample(x, sr, c.target_sample_rate)
            lineage.append({"op": "resample", "from": sr, "to": c.target_sample_rate})
            sr = c.target_sample_rate
        if c.remove_dc:
            x = ops.remove_dc(x)
            lineage.append({"op": "remove_dc"})
        if c.bandpass_hz is not None:
            x = ops.bandpass(x, sr, *c.bandpass_hz)
            lineage.append({"op": "bandpass", "hz": list(c.bandpass_hz)})
        if c.lowpass_hz is not None:
            x = ops.lowpass(x, sr, c.lowpass_hz)
            lineage.append({"op": "lowpass", "hz": c.lowpass_hz})
        if c.align_transient:
            x = ops.align_to_transient(x, sr, c.align_pre_s)
            lineage.append({"op": "align_transient", "pre_s": c.align_pre_s})
        if c.normalize != "none":
            x = ops.normalize(x, c.normalize)
            lineage.append({"op": "normalize", "mode": c.normalize})
        if not np.isfinite(x).all():
            raise DSPError("preprocessing produced non-finite values")
        return Processed(x.astype(np.float32), sr, lineage)
