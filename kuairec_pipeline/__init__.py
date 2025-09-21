"""Utilities to prepare and train KuaiRec sequential recommenders."""

from __future__ import annotations

from typing import Any

from .data_prep import build_interaction_file, prepare_kuairec_dataset  # noqa: F401

__all__ = ["build_interaction_file", "prepare_kuairec_dataset", "train_model"]


def __getattr__(name: str) -> Any:
    """Lazily expose training helpers to avoid importing RecBole eagerly."""

    if name == "train_model":
        from .train_sasrec import train_model as _train_model

        return _train_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
