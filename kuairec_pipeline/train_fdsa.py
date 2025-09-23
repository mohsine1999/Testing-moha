"""Training entry-point for RecBole's feature-level attention model FDSA."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional, Sequence

from .train_common import build_parser, train_model

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "kuairec_fdsa.yaml"


def _ensure_item_features(data_path: Path, dataset_name: str) -> None:
    """Raise if the dataset directory is missing the RecBole `.item` file."""

    item_file = data_path / dataset_name / f"{dataset_name}.item"
    if not item_file.exists():
        raise FileNotFoundError(
            f"Item side-information file not found at {item_file}. "
            "Re-run `kuairec_pipeline.data_prep` with `--with-side-info`."
        )


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = build_parser(
        __doc__ or "Train FDSA on KuaiRec.",
        default_config=DEFAULT_CONFIG,
        default_model="FDSA",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, object]:
    args = _parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="[%(levelname)s] %(message)s")

    _ensure_item_features(args.data_path, args.dataset_name)

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
        neg_samples=args.neg_samples,
    )
    return result


if __name__ == "__main__":
    main()
