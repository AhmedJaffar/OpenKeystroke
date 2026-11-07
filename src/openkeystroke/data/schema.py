"""Dataset schema. Metadata is deliberately free of personal information."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SampleRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_id: str
    dataset_id: str
    dataset_version: str
    waveform_index: int = Field(ge=0)  # row in waveforms.npy (the audio reference)
    sample_rate: int
    duration_s: float
    label: str  # predefined, non-sensitive acoustic research category
    session_id: str
    device_id: str
    source_category: str  # keyboard / acoustic source category
    mic_category: str
    environment: dict[str, Any] = Field(default_factory=dict)
    noise: dict[str, Any] = Field(default_factory=dict)
    provenance: str = "unknown"
    partition: str = "unassigned"  # train / val / test once a split is applied (not part of the fingerprint)
    lineage: list[str] = Field(default_factory=list)

    @property
    def environment_name(self) -> str:
        return str(self.environment.get("name", "unknown"))


class DatasetManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    version: str
    name: str = ""
    is_synthetic: bool
    sample_rate: int
    n_samples: int
    n_waveform_samples: int
    class_names: list[str]
    fingerprint: str = ""
    provenance: dict[str, Any] = Field(default_factory=dict)
    consent: dict[str, Any] | None = None
    created_at: str = ""
    format_version: int = 1
