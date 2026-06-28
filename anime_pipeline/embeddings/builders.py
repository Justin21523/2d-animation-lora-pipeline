"""
Stub embedding builders for frames/foregrounds/etc.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from anime_pipeline.utils.logging_utils import setup_logging
from anime_pipeline.utils.path_utils import ensure_dir


@dataclass
class EmbeddingConfig:
    target_type: str = "frame"  # frame | det | fg
    input_metadata_path: str = "metadata/frames.parquet"
    output_dir: str = "embeddings/clip"
    output_metadata_path: str = "metadata/embeddings.parquet"
    dim: int = 16
    use_stub: bool = True
    log_dir: Optional[str] = "logs"


def build_embeddings(config: EmbeddingConfig, logger=None) -> List[Dict]:
    """
    Generate deterministic random embeddings for each input row.
    """
    logger = logger or setup_logging("build_embeddings", config.log_dir)
    rows = _load_rows(config.input_metadata_path, logger)
    if not rows:
        logger.warning("No input metadata found at %s", config.input_metadata_path)
        return []

    out_dir = ensure_dir(config.output_dir)
    records: List[Dict] = []
    for row in rows:
        target_id = row.get("frame_id") or row.get("det_id") or row.get("fg_id")
        if target_id is None:
            continue
        vector = _make_stub_vector(target_id, config.dim)
        vector_path = out_dir / f"{target_id}.json"
        vector_path.write_text(json.dumps(vector), encoding="utf-8")
        records.append(
            {
                "embed_id": f"{config.target_type}_{target_id}",
                "type": "stub",
                "target_type": config.target_type,
                "target_id": target_id,
                "vector_path": str(vector_path),
                "dim": config.dim,
                "extra": None,
            }
        )

    meta_path = _write_records(records, config.output_metadata_path, logger)
    logger.info("Wrote %d embeddings -> %s", len(records), meta_path)
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


def _make_stub_vector(target_id: str, dim: int) -> List[float]:
    seed_int = int(hashlib.sha1(target_id.encode("utf-8")).hexdigest(), 16) % (2**32)
    rng = random.Random(seed_int)
    return [rng.uniform(-1.0, 1.0) for _ in range(dim)]


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

