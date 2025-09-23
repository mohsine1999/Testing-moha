"""Shared helpers for launching RecBole training jobs on KuaiRec."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional

try:  # pragma: no cover - exercised only when RecBole is missing.
    import numpy as np

    _NUMPY_COMPAT_ALIASES = {
        # NumPy<1.24 exposes bool8 as bool_, whereas newer Ray builds expect
        # the alias to exist. NumPy>=2 removed a couple of other dtype aliases
        # (``float_``, ``complex_``, ``unicode_``) that the RecBole compatibility
        # shim still references. We restore them here so older RecBole releases
        # keep working across NumPy versions.
        "bool8": ("bool_", bool),
        "float_": ("float64", float),
        "complex_": ("complex128", complex),
        "unicode_": ("str_", str),
    }
    for alias, candidates in _NUMPY_COMPAT_ALIASES.items():
        if hasattr(np, alias):
            continue
        if not isinstance(candidates, tuple):
            candidates = (candidates,)
        for candidate in candidates:
            target = getattr(np, candidate, None) if isinstance(candidate, str) else candidate
            if target is not None:
                setattr(np, alias, target)
                break

    from recbole.quick_start import run_recbole
except ImportError as import_error:  # pragma: no cover - raised only if RecBole missing.
    run_recbole = None  # type: ignore[assignment]
    _IMPORT_ERROR = import_error
else:  # pragma: no branch - executed when the import succeeds.
    _IMPORT_ERROR = None


def train_model(
    *,
    dataset: str,
    data_path: Path,
    config_file: Optional[Path],
    model: str,
    checkpoint_dir: Optional[Path] = None,
    learning_rate: Optional[float] = None,
    epochs: Optional[int] = None,
    train_batch_size: Optional[int] = None,
    eval_batch_size: Optional[int] = None,
    seed: Optional[int] = None,
    device: Optional[str] = None,
    max_seq_length: Optional[int] = None,
    neg_samples: Optional[int] = None,
    extra_config: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Launch a RecBole training run and return its reported metrics."""

    if run_recbole is None:  # pragma: no cover - triggered only when dependency missing.
        raise RuntimeError(
            "RecBole is required to train the model. Install it with `pip install recbole`."
        ) from _IMPORT_ERROR

    overrides: Dict[str, object] = {
        "data_path": str(data_path),
        "dataset": dataset,
    }
    if checkpoint_dir is not None:
        overrides["checkpoint_dir"] = str(checkpoint_dir)
    if learning_rate is not None:
        overrides["learning_rate"] = learning_rate
    if epochs is not None:
        overrides["epochs"] = epochs
    if train_batch_size is not None:
        overrides["train_batch_size"] = train_batch_size
    if eval_batch_size is not None:
        overrides["eval_batch_size"] = eval_batch_size
    if seed is not None:
        overrides["seed"] = seed
    if device is not None:
        overrides["device"] = device
    if max_seq_length is not None:
        overrides["MAX_ITEM_LIST_LENGTH"] = max_seq_length

    neg_sample_count = neg_samples
    if neg_sample_count is not None and neg_sample_count <= 0:
        logging.warning(
            "Ignoring non-positive neg_samples=%s; training without negative sampling.",
            neg_sample_count,
        )
        neg_sample_count = None

    if neg_sample_count is not None:
        overrides["train_neg_sample_args"] = {
            "distribution": "uniform",
            "sample_num": neg_sample_count,
            "alpha": 1.0,
            "dynamic": False,
            "candidate_num": 0,
        }
        # RecBole expects the BCE loss when negative sampling is enabled.
        if overrides.get("loss_type") not in {"BCE", "BPR"}:
            logging.info(
                "Switching loss_type to BCE to support %s uniform negative samples",
                neg_sample_count,
            )
            overrides["loss_type"] = "BCE"
    elif config_file is None:
        overrides["train_neg_sample_args"] = None
    if extra_config:
        overrides.update(extra_config)

    config_files = [str(config_file)] if config_file else None
    logging.info(
        "Launching RecBole run with model=%s dataset=%s overrides=%s",
        model,
        dataset,
        json.dumps(overrides, indent=2, sort_keys=True),
    )
    result = run_recbole(  # type: ignore[misc]
        model=model,
        dataset=dataset,
        config_file_list=config_files,
        config_dict=overrides,
    )
    logging.info("Training finished. Best validation metric: %s", result.get("best_valid_score"))
    logging.info("Validation result: %s", result.get("valid_result"))
    logging.info("Test result: %s", result.get("test_result"))
    return result


def build_parser(
    description: str,
    *,
    default_config: Optional[Path],
    default_model: str,
) -> argparse.ArgumentParser:
    """Create an ``argparse`` parser shared by the training entry-points."""

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data"),
        help="Directory containing RecBole atomic datasets (default: ./data).",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="kuairec_small",
        help="Dataset name to load from the data path (default: kuairec_small).",
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        default=default_config,
        help="RecBole YAML config file to use.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=default_model,
        help=f"RecBole model to train (default: {default_model}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("saved"),
        help="Directory where checkpoints and logs will be stored.",
    )
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=None,
        help="Override MAX_ITEM_LIST_LENGTH for sequential recommenders.",
    )
    parser.add_argument(
        "--neg-samples",
        type=int,
        default=None,
        help="Number of negative samples per positive instance.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (default: INFO).",
    )
    return parser


__all__ = ["build_parser", "train_model"]

