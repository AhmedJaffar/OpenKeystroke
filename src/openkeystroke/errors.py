"""Structured, predictable errors. Every error carries a stable code and CLI exit code."""

from __future__ import annotations

from typing import Any


class OpenKeystrokeError(Exception):
    code = "error"
    exit_code = 1

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {"error": self.code, "message": self.message, "details": self.details}


class ConfigError(OpenKeystrokeError):
    code = "config_invalid"
    exit_code = 2


class DatasetError(OpenKeystrokeError):
    code = "dataset_invalid"
    exit_code = 2


class ConsentError(DatasetError):
    code = "consent_required"


class LeakageError(DatasetError):
    code = "data_leakage"


class DSPError(OpenKeystrokeError):
    code = "dsp_invalid"
    exit_code = 2


class ModelError(OpenKeystrokeError):
    code = "model_invalid"
    exit_code = 2


class TrainingError(OpenKeystrokeError):
    code = "training_failed"


class RegistryError(OpenKeystrokeError):
    code = "registry_error"


class LifecycleError(RegistryError):
    code = "invalid_transition"


class NotFoundError(OpenKeystrokeError):
    code = "not_found"
    exit_code = 3


class CancelledError(OpenKeystrokeError):
    code = "cancelled"
    exit_code = 4
