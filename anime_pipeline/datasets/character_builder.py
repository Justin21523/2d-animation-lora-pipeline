"""
Stub character dataset builder: copies fg metadata into a training-ready table.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from anime_pipeline.utils.logging_utils import setup_logging
from anime_pipeline.utils.path_utils import ensure_dir


@dataclass
class CharacterDatasetConfig:
    fg_metadata_path: str = "metadata/fg.parquet"
    output_dir: str = "lora_datasets/characters"
    output_metadata_path: str = "lora_datasets/characters/metadata.parquet"
    class_name: str = "character"
    use_stub: bool = True
    log_dir: Optional[str] = "logs"


def build_character_dataset(config: CharacterDatasetConfig, logger=None) -> List[Dict]:
    """
    Prepare character dataset entries from fg metadata.
    """
    logger = logger or setup_logging("build_character_dataset", config.log_dir)
    rows = _load_rows(config.fg_metadata_path, logger)
    if not rows:
        logger.warning("No fg metadata found at %s", config.fg_metadata_path)
        return []

    ensure_dir(config.output_dir)
    records: List[Dict] = []
    for row in rows:
        fg_path = row.get("rgba_path") or row.get("image_path")
        if not fg_path:
            continue
        sample_id = row.get("fg_id") or row.get("det_id") or Path(fg_path).stem
        records.append(
            {
                "sample_id": sample_id,
                "image_path": fg_path,
                "mask_path": row.get("mask_path"),
                "class_name": config.class_name,
                "video_id": row.get("video_id"),
            }
        )

    meta_path = _write_records(records, config.output_metadata_path, logger)
    logger.info("Wrote %d character samples -> %s", len(records), meta_path)
    return records


def _load_rows(path: str | Path, logger) -> List[Dict]:
    path = Path(path)
    if not path.exists() and path.suffix == ".parquet":
        alt = path.with_suffix(".csv")
        if alt.exists():
            path = alt
    if not path.exists():
        return []
    try:
        import pandas as pd

        df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        return df.to_dict(orient="records")
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to load rows from %s due to %s", path, exc)
        return []


def _write_records(records: List[Dict], output_path: str | Path, logger) -> Path:
    target_path = Path(output_path)
    ensure_dir(target_path.parent)
    if not records:
        target_path.touch()
        return target_path
    try:
        import pandas as pd

        df = pd.DataFrame(records)
        if target_path.suffix == ".parquet":
            df.to_parquet(target_path, index=False)
        else:
            df.to_csv(target_path, index=False)
        return target_path
    except Exception as exc:  # pragma: no cover
        logger.warning("Falling back to CSV due to %s", exc)
        import csv

        csv_path = target_path.with_suffix(".csv")
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
        return csv_path

