"""
Stub ControlNet pose dataset builder: pairs pose images with frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from anime_pipeline.utils.logging_utils import setup_logging
from anime_pipeline.utils.path_utils import ensure_dir


@dataclass
class ControlNetPoseDatasetConfig:
    poses_metadata_path: str = "metadata/poses.parquet"
    output_dir: str = "controlnet_datasets/pose"
    output_metadata_path: str = "controlnet_datasets/pose/metadata.parquet"
    use_stub: bool = True
    log_dir: Optional[str] = "logs"


def build_controlnet_pose_dataset(config: ControlNetPoseDatasetConfig, logger=None) -> List[Dict]:
    """
    Prepare pose-condition dataset entries.
    """
    logger = logger or setup_logging("build_controlnet_pose_dataset", config.log_dir)
    rows = _load_rows(config.poses_metadata_path, logger)
    if not rows:
        logger.warning("No poses metadata found at %s", config.poses_metadata_path)
        return []

    ensure_dir(config.output_dir)
    records: List[Dict] = []
    for row in rows:
        pose_img = row.get("pose_image_path")
        if not pose_img:
            continue
        records.append(
            {
                "sample_id": row.get("pose_id"),
                "pose_image_path": pose_img,
                "target_frame_id": row.get("frame_id"),
                "video_id": row.get("video_id"),
                "pose_type": row.get("pose_type"),
            }
        )

    meta_path = _write_records(records, config.output_metadata_path, logger)
    logger.info("Wrote %d controlnet pose samples -> %s", len(records), meta_path)
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

