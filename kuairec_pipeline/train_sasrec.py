"""Training entry-point for running RecBole's SASRec on KuaiRec."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np


def _ensure_numpy_alias(name: str, target: object) -> None:
    """Expose deprecated NumPy aliases expected by RecBole/Ray.

    NumPy 2.0 removed a collection of legacy attribute names (``np.float_``,
    ``np.complex_``, ``np.unicode_``, ``np.bool8`` …) that downstream packages
    such as RecBole and Ray Tune still access at import time. When those
    attributes are missing, importing the training entry point immediately fails
    before any of our code runs. We defensively recreate the aliases the first
    time this module is imported so the rest of the pipeline remains usable
    regardless of the NumPy major version installed by the user.
    """

    if not hasattr(np, name):
        setattr(np, name, target)


# RecBole's quick-start module pulls in Ray Tune, whose logger expects the alias
# ``np.bool8`` that only exists in newer NumPy releases. Some Python environments
# (including the one reported by users) ship an older NumPy where this alias is
# missing, causing an ``AttributeError`` during import. We provide the alias
# manually when absent so the training entry-point works regardless of the NumPy
# version that happens to be installed.
_ensure_numpy_alias("bool8", np.bool_)

# RecBole also references the NumPy 1.x scalar aliases that vanished in NumPy
# 2.0 when applying its internal backward-compatibility shim. Expose them so
# their assignments succeed instead of raising ``AttributeError``.
_ensure_numpy_alias("float_", np.float64)
_ensure_numpy_alias("complex_", np.complex128)
_ensure_numpy_alias("unicode_", np.str_)

try:
    from recbole.quick_start import run_recbole
except ImportError as import_error:  # pragma: no cover - raised only if RecBole missing.
    run_recbole = None  # type: ignore[assignment]
    _IMPORT_ERROR = import_error
else:
    _IMPORT_ERROR = None

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "kuairec_sasrec.yaml"


def train_model(
    *,
    dataset: str,
    data_path: Path,
    config_file: Optional[Path] = DEFAULT_CONFIG,
    model: str = "SASRec",
    checkpoint_dir: Optional[Path] = None,
    learning_rate: Optional[float] = None,
    epochs: Optional[int] = None,
    train_batch_size: Optional[int] = None,
    eval_batch_size: Optional[int] = None,
    seed: Optional[int] = None,
    device: Optional[str] = None,
    max_seq_length: Optional[int] = None,
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


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
        default=DEFAULT_CONFIG,
        help="RecBole YAML config file to use (default: configs/kuairec_sasrec.yaml).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="SASRec",
        help="RecBole model to train (default: SASRec).",
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
        help="Override MAX_ITEM_LIST_LENGTH for SASRec.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (default: INFO).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, object]:
    args = _parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="[%(levelname)s] %(message)s")

    config_file = args.config_file if args.config_file and args.config_file.exists() else None
    if args.config_file and not args.config_file.exists():
        logging.warning("Config file %s does not exist; relying solely on CLI overrides", args.config_file)

    result = train_model(
        dataset=args.dataset_name,
        data_path=args.data_path,
        config_file=config_file,
        model=args.model,
        checkpoint_dir=args.output_dir,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        seed=args.seed,
        device=args.device,
        max_seq_length=args.max_seq_length,
    )
    return result


if __name__ == "__main__":
    main()
