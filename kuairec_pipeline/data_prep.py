"""Utilities to convert the KuaiRec dataset into RecBole atomic files."""

from __future__ import annotations

import argparse
import ast
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import pandas as pd


@dataclass(frozen=True)
class FieldSpec:
    """Schema definition for RecBole atomic files."""

    name: str
    dtype: str
    kind: str
    required: bool = False


INTERACTION_SCHEMA: Sequence[FieldSpec] = (
    FieldSpec("user_id", "token", "token", required=True),
    FieldSpec("item_id", "token", "token", required=True),
    FieldSpec("timestamp", "float", "numeric", required=True),
    FieldSpec("label", "float", "numeric"),
    FieldSpec("watch_ratio", "float", "numeric"),
    FieldSpec("play_duration", "float", "numeric"),
    FieldSpec("video_duration", "float", "numeric"),
)

USER_SCHEMA: Sequence[FieldSpec] = (
    FieldSpec("user_id", "token", "token", required=True),
    FieldSpec("user_active_degree", "token", "token"),
    FieldSpec("is_lowactive_period", "float", "numeric"),
    FieldSpec("is_live_streamer", "float", "numeric"),
    FieldSpec("is_video_author", "float", "numeric"),
    FieldSpec("follow_user_num", "float", "numeric"),
    FieldSpec("fans_user_num", "float", "numeric"),
    FieldSpec("friend_user_num", "float", "numeric"),
    FieldSpec("register_days", "float", "numeric"),
    FieldSpec("follow_user_num_range", "token", "token"),
    FieldSpec("fans_user_num_range", "token", "token"),
    FieldSpec("friend_user_num_range", "token", "token"),
    FieldSpec("register_days_range", "token", "token"),
    *[
        FieldSpec(f"onehot_feat{idx}", "token", "token")
        for idx in range(18)
    ],
)

ITEM_SCHEMA: Sequence[FieldSpec] = (
    FieldSpec("item_id", "token", "token", required=True),
    FieldSpec("author_id", "token", "token"),
    FieldSpec("video_type", "token", "token"),
    FieldSpec("upload_type", "token", "token"),
    FieldSpec("video_duration", "float", "numeric"),
    FieldSpec("video_width", "float", "numeric"),
    FieldSpec("video_height", "float", "numeric"),
    FieldSpec("music_id", "token", "token"),
    FieldSpec("video_tag_id", "token", "token"),
    FieldSpec("show_cnt", "float", "numeric"),
    FieldSpec("show_user_num", "float", "numeric"),
    FieldSpec("play_cnt", "float", "numeric"),
    FieldSpec("play_user_num", "float", "numeric"),
    FieldSpec("play_duration", "float", "numeric"),
    FieldSpec("complete_play_cnt", "float", "numeric"),
    FieldSpec("valid_play_cnt", "float", "numeric"),
    FieldSpec("long_time_play_cnt", "float", "numeric"),
    FieldSpec("short_time_play_cnt", "float", "numeric"),
    FieldSpec("play_progress", "float", "numeric"),
    FieldSpec("like_cnt", "float", "numeric"),
    FieldSpec("comment_cnt", "float", "numeric"),
    FieldSpec("follow_cnt", "float", "numeric"),
    FieldSpec("share_cnt", "float", "numeric"),
    FieldSpec("download_cnt", "float", "numeric"),
    FieldSpec("tags", "token_seq", "token_seq"),
)

DEFAULT_MATRIX = "small"


def _sanitize_numeric(series: pd.Series, fill_value: float = 0.0) -> pd.Series:
    coerced = pd.to_numeric(series, errors="coerce")
    return coerced.fillna(fill_value).astype(float)


def _sanitize_token(series: pd.Series, fill_value: str = "UNKNOWN") -> pd.Series:
    return series.fillna(fill_value).astype(str)


def _sanitize_token_seq(series: pd.Series) -> pd.Series:
    def _convert(value: object) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ""
        if isinstance(value, list):
            parts = value
        else:
            text = str(value).strip()
            if not text:
                return ""
            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                parsed = [token.strip() for token in text.replace("[", "").replace("]", "").split(",") if token.strip()]
            parts = parsed if isinstance(parsed, list) else [parsed]
        return " ".join(str(part) for part in parts if str(part))

    return series.map(_convert)


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_atomic_file(df: pd.DataFrame, schema: Sequence[FieldSpec], destination: Path) -> int:
    available_fields = [field for field in schema if field.name in df.columns]
    missing_required = [field.name for field in schema if field.required and field.name not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")
    if not available_fields:
        raise ValueError("No columns available to write to atomic file.")

    processors = {
        "numeric": _sanitize_numeric,
        "token": _sanitize_token,
        "token_seq": _sanitize_token_seq,
    }

    processed = {}
    for field in available_fields:
        processor = processors.get(field.kind)
        if processor is None:
            raise KeyError(f"Unsupported field kind: {field.kind}")
        processed[field.name] = processor(df[field.name])

    ordered_columns = [field.name for field in available_fields]
    header = "\t".join(f"{field.name}:{field.dtype}" for field in available_fields)

    _ensure_parent_dir(destination)
    with destination.open("w", encoding="utf-8") as fp:
        fp.write(header + "\n")
        processed_df = pd.DataFrame(processed)
        processed_df.to_csv(fp, sep="\t", index=False, header=False)
    logging.info("Wrote %d rows to %s", len(processed_df), destination)
    return len(processed_df)


def build_interaction_file(
    matrix_path: Path,
    output_path: Path,
    *,
    chunk_size: int = 1_000_000,
    min_watch_ratio: Optional[float] = None,
) -> Dict[str, int]:
    """Convert a KuaiRec interaction matrix into a RecBole ``.inter`` file."""

    if not matrix_path.exists():
        raise FileNotFoundError(matrix_path)

    logging.info("Building interaction file from %s", matrix_path)
    _ensure_parent_dir(output_path)

    header = "\t".join(f"{field.name}:{field.dtype}" for field in INTERACTION_SCHEMA)
    with output_path.open("w", encoding="utf-8") as fp:
        fp.write(header + "\n")

    total_rows = 0
    users: set[int] = set()
    items: set[int] = set()

    for chunk in pd.read_csv(matrix_path, chunksize=chunk_size):
        chunk = chunk.rename(columns={"video_id": "item_id"})
        columns = {
            "user_id": pd.to_numeric(chunk["user_id"], errors="coerce"),
            "item_id": pd.to_numeric(chunk["item_id"], errors="coerce"),
            "timestamp": pd.to_numeric(chunk["timestamp"], errors="coerce"),
            "watch_ratio": pd.to_numeric(chunk.get("watch_ratio"), errors="coerce"),
            "play_duration": pd.to_numeric(chunk.get("play_duration"), errors="coerce"),
            "video_duration": pd.to_numeric(chunk.get("video_duration"), errors="coerce"),
        }
        frame = pd.DataFrame(columns)
        frame = frame.dropna(subset=["user_id", "item_id", "timestamp"])
        if min_watch_ratio is not None:
            frame = frame[frame["watch_ratio"].fillna(0.0) >= min_watch_ratio]
        if frame.empty:
            continue
        frame["user_id"] = frame["user_id"].astype(int)
        frame["item_id"] = frame["item_id"].astype(int)
        frame["timestamp"] = frame["timestamp"].astype(float)
        frame["label"] = 1.0
        frame["watch_ratio"] = frame["watch_ratio"].fillna(0.0).astype(float)
        frame["play_duration"] = frame["play_duration"].fillna(0.0).astype(float)
        frame["video_duration"] = frame["video_duration"].fillna(0.0).astype(float)

        frame = frame.sort_values(["user_id", "timestamp"], kind="mergesort")

        users.update(frame["user_id"].unique().tolist())
        items.update(frame["item_id"].unique().tolist())
        total_rows += len(frame)

        frame = frame[[field.name for field in INTERACTION_SCHEMA]]
        frame.to_csv(
            output_path,
            mode="a",
            sep="\t",
            header=False,
            index=False,
            float_format="%.6f",
        )

    logging.info(
        "Finished writing %d interactions covering %d users and %d items",
        total_rows,
        len(users),
        len(items),
    )
    return {"n_interactions": total_rows, "n_users": len(users), "n_items": len(items)}


def _load_latest_item_daily_features(path: Path) -> pd.DataFrame:
    daily = pd.read_csv(path)
    if "date" in daily.columns:
        daily = daily.sort_values(["video_id", "date"], ascending=[True, False])
        daily = daily.drop_duplicates("video_id", keep="first")
    return daily


def _build_user_file(source_dir: Path, dataset_dir: Path, dataset_name: str) -> Optional[int]:
    user_path = source_dir / "user_features.csv"
    if not user_path.exists():
        logging.warning("User feature file not found at %s; skipping", user_path)
        return None
    df = pd.read_csv(user_path)
    destination = dataset_dir / f"{dataset_name}.user"
    return _write_atomic_file(df, USER_SCHEMA, destination)


def _build_item_file(source_dir: Path, dataset_dir: Path, dataset_name: str) -> Optional[int]:
    daily_path = source_dir / "item_daily_features.csv"
    categories_path = source_dir / "item_categories.csv"

    if not daily_path.exists():
        logging.warning("Item daily features not found at %s; skipping", daily_path)
        return None

    daily = _load_latest_item_daily_features(daily_path)
    daily = daily.rename(columns={"video_id": "item_id"})

    if categories_path.exists():
        categories = pd.read_csv(categories_path)
        categories = categories.rename(columns={"video_id": "item_id", "feat": "tags"})
        categories["tags"] = _sanitize_token_seq(categories["tags"])
        daily = daily.merge(categories[["item_id", "tags"]], on="item_id", how="left")
    else:
        logging.info("Item category file not found at %s; tags will be empty", categories_path)

    destination = dataset_dir / f"{dataset_name}.item"
    return _write_atomic_file(daily, ITEM_SCHEMA, destination)


def prepare_kuairec_dataset(
    source_data: Path,
    output_dir: Path,
    *,
    dataset_name: str = "kuairec_small",
    matrix_size: str = DEFAULT_MATRIX,
    chunk_size: int = 1_000_000,
    min_watch_ratio: Optional[float] = None,
    include_side_info: bool = False,
) -> Dict[str, Optional[int]]:
    """Prepare RecBole atomic files for KuaiRec."""

    matrix_filename = f"{matrix_size}_matrix.csv"
    matrix_path = source_data / matrix_filename
    dataset_dir = output_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    stats = build_interaction_file(
        matrix_path,
        dataset_dir / f"{dataset_name}.inter",
        chunk_size=chunk_size,
        min_watch_ratio=min_watch_ratio,
    )

    results: Dict[str, Optional[int]] = dict(stats)
    if include_side_info:
        results["n_user_rows"] = _build_user_file(source_data, dataset_dir, dataset_name)
        results["n_item_rows"] = _build_item_file(source_data, dataset_dir, dataset_name)

    logging.info("Dataset preparation summary: %s", results)
    return results


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source_data",
        type=Path,
        help="Path to the extracted KuaiRec `data` directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory to place the RecBole-formatted dataset (default: ./data)",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="kuairec_small",
        help="Name of the dataset folder/filename prefix to create.",
    )
    parser.add_argument(
        "--matrix-size",
        choices=["small", "big"],
        default=DEFAULT_MATRIX,
        help="Which interaction matrix to use (default: small).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1_000_000,
        help="Number of rows to process per chunk when streaming the CSV.",
    )
    parser.add_argument(
        "--min-watch-ratio",
        type=float,
        default=None,
        help="Drop interactions whose watch_ratio is below this threshold.",
    )
    parser.add_argument(
        "--with-side-info",
        action="store_true",
        help="Generate `.user` and `.item` side information files as well.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (default: INFO).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Optional[int]]:
    args = _parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="[%(levelname)s] %(message)s")

    return prepare_kuairec_dataset(
        args.source_data,
        args.output_dir,
        dataset_name=args.dataset_name,
        matrix_size=args.matrix_size,
        chunk_size=args.chunk_size,
        min_watch_ratio=args.min_watch_ratio,
        include_side_info=args.with_side_info,
    )


if __name__ == "__main__":
    main()
