"""Training entry-point for running RecBole's SASRec on KuaiRec."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
import torch.distributed as dist


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

try:  # pragma: no cover - exercised only when RecBole is available.
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.data.transform import construct_transform
    from recbole.utils import (
        get_environment,
        get_flops,
        get_model,
        get_trainer,
        init_logger,
        init_seed,
        set_color,
    )
    from recbole.utils.case_study import full_sort_topk
except ImportError as import_error:  # pragma: no cover - raised only if RecBole missing.
    Config = None  # type: ignore[assignment]
    create_dataset = None  # type: ignore[assignment]
    data_preparation = None  # type: ignore[assignment]
    construct_transform = None  # type: ignore[assignment]
    get_environment = None  # type: ignore[assignment]
    get_flops = None  # type: ignore[assignment]
    get_model = None  # type: ignore[assignment]
    get_trainer = None  # type: ignore[assignment]
    init_logger = None  # type: ignore[assignment]
    init_seed = None  # type: ignore[assignment]
    set_color = None  # type: ignore[assignment]
    full_sort_topk = None  # type: ignore[assignment]
    _IMPORT_ERROR = import_error
else:
    _IMPORT_ERROR = None

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "kuairec_sasrec.yaml"


@dataclass
class TrainingRun:
    """Container bundling the RecBole artifacts produced during training."""

    result: Dict[str, object]
    config: Any
    dataset: Any
    train_data: Any
    valid_data: Any
    test_data: Any
    trainer: Any


def _is_rank_zero() -> bool:
    """Return ``True`` when the current process should handle I/O side effects."""

    if not dist.is_available():  # pragma: no cover - PyTorch without distributed support.
        return True
    if not dist.is_initialized():
        return True
    try:
        return dist.get_rank() == 0
    except RuntimeError:  # pragma: no cover - defensive for partially initialised process groups.
        return True


def _load_user_tokens(csv_path: Path) -> Sequence[str]:
    """Parse ``user_id`` tokens from a CSV file."""

    tokens: list[str] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            first_row = next(reader)
        except StopIteration:
            return tokens

        header = [cell.strip() for cell in first_row]
        header_lower = [cell.lower() for cell in header]
        if "user_id" in header_lower:
            user_idx = header_lower.index("user_id")
        else:
            user_idx = 0
            if header and header[0]:
                tokens.append(header[0])

        for row in reader:
            if len(row) <= user_idx:
                continue
            token = row[user_idx].strip()
            if token:
                tokens.append(token)

    return tokens


def _candidate_token_values(raw_token: Any) -> Sequence[Any]:
    """Yield potential representations for a raw token value."""

    candidates: list[Any] = []
    if isinstance(raw_token, str):
        stripped = raw_token.strip()
        if not stripped:
            return candidates
        candidates.append(stripped)
        try:
            candidates.append(int(stripped))
        except ValueError:
            pass
    else:
        candidates.append(raw_token)
        candidates.append(str(raw_token))

    # Deduplicate while preserving order.
    return list(dict.fromkeys(candidates))


def _resolve_user_token(dataset: Any, token: Any) -> Optional[int]:
    """Convert an external user token into its internal RecBole id."""

    uid_field = dataset.uid_field
    for candidate in _candidate_token_values(token):
        try:
            internal_id = dataset.token2id(uid_field, candidate)
        except (TypeError, ValueError):
            continue
        if isinstance(internal_id, np.ndarray):
            if internal_id.size == 0:
                continue
            return int(internal_id.item())
        return int(internal_id)
    return None


def _resolve_user_ids(dataset: Any, test_data: Any, user_tokens: Optional[Iterable[str]] = None) -> np.ndarray:
    """Determine the internal user ids to evaluate for top-k export."""

    if user_tokens:
        resolved = []
        missing = []
        for token in user_tokens:
            internal = _resolve_user_token(dataset, token)
            if internal is None:
                missing.append(token)
            else:
                resolved.append(internal)
        if missing:
            preview = ", ".join(map(str, missing[:5]))
            logging.warning(
                "Skipping %d unknown user_id token(s) from subset: %s%s",
                len(missing),
                preview,
                "…" if len(missing) > 5 else "",
            )
        if not resolved:
            return np.array([], dtype=np.int64)
        return np.unique(np.asarray(resolved, dtype=np.int64))

    uid_field = dataset.uid_field
    eval_dataset = getattr(test_data, "dataset", None) or getattr(test_data, "_dataset", None)
    if eval_dataset is None:
        raise ValueError("Unable to derive evaluation dataset for top-k export.")
    user_array = eval_dataset.inter_feat[uid_field].numpy()
    return np.unique(user_array.astype(np.int64, copy=False))


def export_topk_recs(
    *,
    trainer: Any,
    dataset: Any,
    test_data: Any,
    k: int,
    output_file: Path,
    user_tokens: Optional[Iterable[str]] = None,
) -> Optional[Path]:
    """Write top-``k`` recommendations per user to ``output_file``.

    Returns the path when export succeeds and ``None`` when no users were available.
    """

    if full_sort_topk is None:  # pragma: no cover - triggered only when RecBole missing.
        raise RuntimeError(
            "RecBole utilities are required for top-k export. Install recbole to proceed."
        )
    if k <= 0:
        return None

    user_ids = _resolve_user_ids(dataset, test_data, user_tokens)
    if user_ids.size == 0:
        logging.warning("Top-%d export skipped because no users were available", k)
        return None

    max_items = max(int(dataset.item_num) - 1, 1)
    effective_k = min(k, max_items)
    if effective_k != k:
        logging.warning(
            "Requested top-%d recommendations but only %d items are available; using top-%d instead",
            k,
            max_items,
            effective_k,
        )

    device = getattr(trainer, "device", None)
    scores, indices = full_sort_topk(user_ids, trainer.model, test_data, effective_k, device=device)
    scores = scores.cpu().numpy()
    indices = indices.cpu().numpy()

    uid_field = dataset.uid_field
    iid_field = dataset.iid_field
    user_tokens_resolved = np.asarray(dataset.id2token(uid_field, user_ids)).tolist()
    item_tokens = dataset.id2token(iid_field, indices)
    item_tokens = np.asarray(item_tokens)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["user_id", "item_id", "rank", "score"])
        for user_token, user_items, user_scores in zip(
            user_tokens_resolved, item_tokens, scores, strict=True
        ):
            for rank, (item_token, score) in enumerate(zip(user_items, user_scores, strict=True), start=1):
                writer.writerow([str(user_token), str(item_token), rank, float(score)])

    logging.info("Wrote %s (users=%d, k=%d)", output_file, len(user_tokens_resolved), effective_k)
    return output_file


def _run_recbole_training(
    *,
    model: str,
    dataset: str,
    config_file_list: Optional[Sequence[str]],
    config_dict: Dict[str, object],
) -> TrainingRun:
    """Execute the RecBole training loop and capture its artifacts."""

    if Config is None:  # pragma: no cover - triggered only when RecBole missing.
        raise RuntimeError(
            "RecBole is required to train the model. Install it with `pip install recbole`."
        ) from _IMPORT_ERROR

    config = Config(
        model=model,
        dataset=dataset,
        config_file_list=config_file_list,
        config_dict=config_dict,
    )
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = logging.getLogger()
    logger.info(sys.argv)
    logger.info(config)

    dataset_obj = create_dataset(config)
    logger.info(dataset_obj)

    train_data, valid_data, test_data = data_preparation(config, dataset_obj)

    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model_class = get_model(config["model"])
    model_obj = model_class(config, train_data._dataset).to(config["device"])
    logger.info(model_obj)

    transform = construct_transform(config)
    flops = get_flops(model_obj, dataset_obj, config["device"], logger, transform)
    logger.info(set_color("FLOPs", "blue") + f": {flops}")

    trainer_class = get_trainer(config["MODEL_TYPE"], config["model"])
    trainer = trainer_class(config, model_obj)

    best_valid_score, best_valid_result = trainer.fit(
        train_data,
        valid_data,
        saved=True,
        show_progress=config["show_progress"],
    )

    test_result = trainer.evaluate(
        test_data,
        load_best_model=True,
        show_progress=config["show_progress"],
    )

    environment_tb = get_environment(config)
    logger.info(
        "The running environment of this training is as follows:\n" + environment_tb.draw()
    )

    logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result", "yellow") + f": {test_result}")

    result = {
        "best_valid_score": best_valid_score,
        "valid_score_bigger": config["valid_metric_bigger"],
        "best_valid_result": best_valid_result,
        "test_result": test_result,
    }

    if not config["single_spec"]:
        dist.destroy_process_group()

    return TrainingRun(
        result=result,
        config=config,
        dataset=dataset_obj,
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data,
        trainer=trainer,
    )


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
) -> TrainingRun:
    """Launch a RecBole training run and return the resulting artifacts."""

    if Config is None:  # pragma: no cover - triggered only when dependency missing.
        raise RuntimeError(
            "RecBole is required to train the model. Install it with `pip install recbole`."
        ) from _IMPORT_ERROR

    overrides: Dict[str, object] = {
        "data_path": str(data_path),
        "dataset": dataset,
        # RecBole's SASRec defaults to one negative sample per positive pair when
        # ``loss_type`` is ``CE``. That combination is invalid in recent
        # RecBole versions, so we always disable negative sampling unless a
        # caller explicitly supplies a different setting via ``extra_config``.
        "train_neg_sample_args": None,
    }

    dataset_lower = dataset.lower()
    auto_memory_guard = any(token in dataset_lower for token in ("big", "full"))
    if auto_memory_guard:
        # Training on the full KuaiRec interaction log requires substantially
        # more host memory than the default quick-start hyperparameters demand.
        # We proactively dial down the most memory-hungry knobs when we detect a
        # "big" dataset name so the run has a better chance of finishing on
        # commodity laptops. Users can still override these choices explicitly
        # via the CLI/``extra_config`` if they have access to more resources.
        if train_batch_size is None:
            overrides["train_batch_size"] = 256
        if eval_batch_size is None:
            overrides["eval_batch_size"] = 512
        if max_seq_length is None:
            overrides["MAX_ITEM_LIST_LENGTH"] = 100
        logging.info(
            "Dataset '%s' detected as large; applying memory-friendly defaults %s",
            dataset,
            {
                "train_batch_size": overrides.get("train_batch_size"),
                "eval_batch_size": overrides.get("eval_batch_size"),
                "MAX_ITEM_LIST_LENGTH": overrides.get("MAX_ITEM_LIST_LENGTH"),
            },
        )
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
    training_run = _run_recbole_training(
        model=model,
        dataset=dataset,
        config_file_list=config_files,
        config_dict=overrides,
    )
    result = training_run.result
    logging.info(
        "Training finished. Best validation metric: %s",
        result.get("best_valid_score"),
    )
    logging.info("Validation result: %s", result.get("valid_result"))
    logging.info("Test result: %s", result.get("test_result"))
    return training_run


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
        "--export-topk",
        type=int,
        default=0,
        help="Export top-K recommendations after training (0 disables).",
    )
    parser.add_argument(
        "--export-topk-users",
        type=Path,
        default=None,
        help="Optional CSV listing user_id tokens to restrict the export.",
    )
    parser.add_argument(
        "--export-topk-file",
        type=Path,
        default=Path("saved/topk_recs.csv"),
        help="Destination CSV for exported recommendations (default: saved/topk_recs.csv).",
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

    training_run = train_model(
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
    if args.export_topk > 0 and _is_rank_zero():
        user_tokens: Optional[Sequence[str]] = None
        if args.export_topk_users:
            if args.export_topk_users.exists():
                user_tokens = _load_user_tokens(args.export_topk_users)
                logging.info(
                    "Loaded %d user_id token(s) from %s for top-%d export",
                    len(user_tokens),
                    args.export_topk_users,
                    args.export_topk,
                )
            else:
                logging.warning(
                    "User subset file %s not found; exporting recommendations for all users",
                    args.export_topk_users,
                )

        export_topk_recs(
            trainer=training_run.trainer,
            dataset=training_run.dataset,
            test_data=training_run.test_data,
            k=args.export_topk,
            output_file=args.export_topk_file,
            user_tokens=user_tokens,
        )
    return training_run.result


if __name__ == "__main__":
    main()
