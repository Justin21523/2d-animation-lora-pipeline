#!/usr/bin/env python3
"""Build deterministic public demo assets for the 2D Animation LoRA Pipeline."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "portfolio-web"
MANIFEST = WEB / "demo-data" / "manifest.json"
ASSETS = WEB / "assets"
DEMO_ASSETS = ASSETS / "demo"
SCREENSHOTS = ASSETS / "screenshots"
VIDEO_DIR = ASSETS / "video"


STAGES = [
    ("frame_extraction", "Frame extraction", "Scene-aware frame sampling", 1800, 128),
    ("yolo_tracking", "YOLO + ByteTrack", "Multi-character detection and track IDs", 1460, 74),
    ("toonout_segmentation", "ToonOut segmentation", "Per-track foreground/background cutouts", 1320, 74),
    ("dwpose_extraction", "DWpose extraction", "Pose conditioning for ControlNet", 980, 48),
    ("identity_clustering", "Identity clustering", "HDBSCAN merges track fragments by face embeddings", 720, 11),
    ("dataset_building", "LoRA dataset", "Captioned, filtered, character-specific training rows", 640, 3),
    ("lora_training", "LoRA training", "kohya/diffusers training configs and checkpoint metrics", 4, 1),
]


PALETTES = [
    {"name": "Subject A", "skin": (247, 190, 136), "hair": (68, 55, 45), "cloth": (39, 130, 167), "accent": (15, 118, 110)},
    {"name": "Subject B", "skin": (236, 166, 118), "hair": (176, 74, 45), "cloth": (239, 185, 56), "accent": (29, 78, 216)},
    {"name": "Subject C", "skin": (225, 170, 120), "hair": (48, 57, 75), "cloth": (219, 88, 55), "accent": (194, 65, 12)},
]


def font(size: int):
    from PIL import ImageFont

    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def rounded(draw, box, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def draw_character(draw, x: int, y: int, scale: float, palette: dict[str, tuple[int, int, int]], pose: str = "idle"):
    line = (23, 32, 38)
    head = int(38 * scale)
    body_w = int(62 * scale)
    body_h = int(82 * scale)
    rounded(draw, (x - body_w // 2, y + head - 2, x + body_w // 2, y + head + body_h), int(16 * scale), palette["cloth"], line, 3)
    draw.ellipse((x - head, y - head, x + head, y + head), fill=palette["skin"], outline=line, width=3)
    draw.pieslice((x - head, y - head - int(8 * scale), x + head, y + head), 180, 360, fill=palette["hair"], outline=line, width=2)
    draw.ellipse((x - int(17 * scale), y - int(5 * scale), x - int(10 * scale), y + int(2 * scale)), fill=line)
    draw.ellipse((x + int(10 * scale), y - int(5 * scale), x + int(17 * scale), y + int(2 * scale)), fill=line)
    draw.arc((x - int(14 * scale), y + int(5 * scale), x + int(14 * scale), y + int(22 * scale)), 10, 170, fill=line, width=2)

    shoulder_y = y + head + int(12 * scale)
    hip_y = y + head + body_h
    if pose == "run":
        hands = ((x - int(52 * scale), shoulder_y + int(34 * scale)), (x + int(50 * scale), shoulder_y - int(20 * scale)))
        feet = ((x - int(44 * scale), hip_y + int(34 * scale)), (x + int(44 * scale), hip_y + int(16 * scale)))
    elif pose == "jump":
        hands = ((x - int(50 * scale), shoulder_y - int(24 * scale)), (x + int(50 * scale), shoulder_y - int(24 * scale)))
        feet = ((x - int(28 * scale), hip_y + int(26 * scale)), (x + int(30 * scale), hip_y + int(26 * scale)))
    else:
        hands = ((x - int(48 * scale), shoulder_y + int(8 * scale)), (x + int(48 * scale), shoulder_y + int(8 * scale)))
        feet = ((x - int(28 * scale), hip_y + int(32 * scale)), (x + int(28 * scale), hip_y + int(32 * scale)))
    width = max(3, int(5 * scale))
    draw.line((x - body_w // 2, shoulder_y, *hands[0]), fill=line, width=width)
    draw.line((x + body_w // 2, shoulder_y, *hands[1]), fill=line, width=width)
    draw.line((x - int(14 * scale), hip_y, *feet[0]), fill=line, width=width)
    draw.line((x + int(14 * scale), hip_y, *feet[1]), fill=line, width=width)


def save_character_sheet() -> str:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (1440, 900), (248, 250, 252))
    draw = ImageDraw.Draw(img)
    title = font(42)
    body = font(22)
    small = font(16)
    draw.text((56, 46), "2D Character Dataset Sheet", fill=(15, 23, 42), font=title)
    draw.text((58, 104), "Synthetic public assets: identities, poses, captions, and acceptance tags", fill=(71, 85, 105), font=body)
    for idx, palette in enumerate(PALETTES):
        x0 = 68 + idx * 452
        rounded(draw, (x0, 168, x0 + 392, 810), 18, (255, 255, 255), (203, 213, 225), 2)
        draw.text((x0 + 28, 198), palette["name"], fill=palette["accent"], font=body)
        for pidx, pose in enumerate(["idle", "run", "jump"]):
            cx = x0 + 96 + pidx * 102
            draw_character(draw, cx, 386, 0.9, palette, pose)
            draw.text((cx - 28, 696), pose, fill=(71, 85, 105), font=small)
        draw.text((x0 + 28, 748), "accepted / clean mask / captioned", fill=(15, 23, 42), font=small)
    path = DEMO_ASSETS / "character-sheet.png"
    img.save(path)
    return "assets/demo/character-sheet.png"


def save_pipeline_visual() -> str:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (1440, 820), (246, 248, 250))
    draw = ImageDraw.Draw(img)
    title = font(42)
    body = font(20)
    small = font(15)
    draw.text((56, 44), "Frame to LoRA-Ready Sample", fill=(15, 23, 42), font=title)
    panels = [
        ("raw frame", (70, 146, 384, 610), (227, 236, 229)),
        ("tracked crops", (414, 146, 728, 610), (222, 237, 248)),
        ("mask + pose", (758, 146, 1072, 610), (244, 232, 245)),
        ("training row", (1102, 146, 1372, 610), (249, 238, 222)),
    ]
    for idx, (label, box, fill) in enumerate(panels):
        rounded(draw, box, 18, fill, (203, 213, 225), 2)
        draw.text((box[0], box[3] + 22), label, fill=(15, 23, 42), font=body)
        draw_character(draw, (box[0] + box[2]) // 2, 338, 0.85 if idx < 3 else 0.68, PALETTES[idx % 3], ["idle", "run", "jump", "idle"][idx])
        if idx == 1:
            draw.rectangle((box[0] + 52, box[1] + 70, box[2] - 52, box[3] - 80), outline=(29, 78, 216), width=5)
        if idx == 2:
            draw.line((box[0] + 156, 270, box[0] + 124, 390, box[0] + 174, 506), fill=(194, 65, 12), width=5)
            draw.line((box[0] + 156, 270, box[0] + 210, 390, box[0] + 196, 506), fill=(194, 65, 12), width=5)
    draw.text((70, 704), "Output contract: image path + mask path + track ID + identity cluster + pose metadata + caption", fill=(71, 85, 105), font=small)
    path = DEMO_ASSETS / "pipeline-before-after.png"
    img.save(path)
    return "assets/demo/pipeline-before-after.png"


def save_training_metrics() -> str:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (1440, 820), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    title = font(42)
    body = font(20)
    small = font(15)
    draw.text((56, 44), "LoRA Training Snapshot", fill=(15, 23, 42), font=title)
    draw.text((58, 104), "Mock-safe metrics for review: loss, identity score, mask quality, caption readiness", fill=(71, 85, 105), font=body)
    plot = (94, 180, 900, 668)
    draw.rectangle(plot, outline=(203, 213, 225), width=2)
    for i in range(6):
        y = plot[1] + i * (plot[3] - plot[1]) // 5
        draw.line((plot[0], y, plot[2], y), fill=(226, 232, 240), width=1)
    points = []
    for step in range(14):
        x = plot[0] + step * (plot[2] - plot[0]) // 13
        loss = 0.86 * math.exp(-step / 6.3) + 0.1 + (0.02 if step % 4 == 0 else 0)
        y = plot[3] - int((1.0 - loss) * (plot[3] - plot[1]))
        points.append((x, y))
    draw.line(points, fill=(15, 118, 110), width=5)
    for x, y in points:
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(15, 118, 110))
    cards = [("Identity retention", "91%", (970, 192), (15, 118, 110)), ("Mask acceptance", "88%", (970, 334), (29, 78, 216)), ("Caption coverage", "100%", (970, 476), (194, 65, 12))]
    for label, value, pos, color in cards:
        rounded(draw, (pos[0], pos[1], pos[0] + 340, pos[1] + 106), 14, (248, 250, 252), (203, 213, 225), 2)
        draw.text((pos[0] + 24, pos[1] + 20), label, fill=(71, 85, 105), font=small)
        draw.text((pos[0] + 24, pos[1] + 46), value, fill=color, font=title)
    path = DEMO_ASSETS / "training-metrics.png"
    img.save(path)
    return "assets/demo/training-metrics.png"


def save_evaluation_matrix() -> str:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (1440, 820), (248, 250, 252))
    draw = ImageDraw.Draw(img)
    title = font(42)
    body = font(20)
    small = font(15)
    draw.text((56, 44), "Checkpoint Evaluation Matrix", fill=(15, 23, 42), font=title)
    cols = ["Identity", "Line art", "Pose", "Caption", "Accept"]
    rows = ["epoch-02", "epoch-04", "epoch-06", "epoch-08"]
    x0, y0 = 318, 190
    cw, ch = 176, 102
    for c, col in enumerate(cols):
        draw.text((x0 + c * cw + 24, y0 - 42), col, fill=(71, 85, 105), font=small)
    for r, row in enumerate(rows):
        draw.text((88, y0 + r * ch + 34), row, fill=(15, 23, 42), font=body)
        for c in range(len(cols)):
            score = 74 + r * 5 + c * 3 + (5 if c == 4 else 0)
            color = (15, 118, 110) if score >= 88 else (234, 179, 8) if score >= 80 else (194, 65, 12)
            rounded(draw, (x0 + c * cw, y0 + r * ch, x0 + c * cw + 132, y0 + r * ch + 70), 10, color)
            draw.text((x0 + c * cw + 46, y0 + r * ch + 22), str(score), fill=(255, 255, 255), font=body)
    path = DEMO_ASSETS / "evaluation-matrix.png"
    img.save(path)
    return "assets/demo/evaluation-matrix.png"


def save_animation_strip() -> str:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (1440, 480), (15, 23, 42))
    draw = ImageDraw.Draw(img)
    title = font(38)
    small = font(15)
    draw.text((48, 34), "Pose-Controlled Motion Strip", fill=(236, 254, 255), font=title)
    for i in range(9):
        x = 52 + i * 152
        rounded(draw, (x, 116, x + 118, 376), 14, (248, 250, 252))
        draw_character(draw, x + 59, 228 + int(math.sin(i / 1.4) * 7), 0.48, PALETTES[i % 3], ["idle", "run", "jump"][i % 3])
        draw.text((x + 36, 396), f"f{i:02d}", fill=(203, 213, 225), font=small)
    path = DEMO_ASSETS / "animation-strip.png"
    img.save(path)
    return "assets/demo/animation-strip.png"


def save_screenshot_like(name: str, focus: str) -> str:
    from PIL import Image, ImageDraw

    size = (1440, 1100) if "mobile" not in name else (420, 1500)
    img = Image.new("RGB", size, (248, 250, 252))
    draw = ImageDraw.Draw(img)
    title = font(34 if size[0] > 600 else 24)
    body = font(18 if size[0] > 600 else 13)
    small = font(14 if size[0] > 600 else 11)
    margin = 52 if size[0] > 600 else 22
    draw.text((margin, 34), "2D Animation LoRA Pipeline Demo", fill=(15, 23, 42), font=title)
    draw.text((margin, 88), focus, fill=(71, 85, 105), font=body)
    top = 150
    cols = 3 if size[0] > 600 else 1
    card_w = (size[0] - margin * 2 - (cols - 1) * 18) // cols
    for i, stage in enumerate(STAGES[:6]):
        row, col = divmod(i, cols)
        x = margin + col * (card_w + 18)
        y = top + row * 190
        rounded(draw, (x, y, x + card_w, y + 152), 14, (255, 255, 255), (203, 213, 225), 2)
        draw.text((x + 18, y + 18), stage[1], fill=(15, 118, 110), font=body)
        draw.text((x + 18, y + 56), stage[2][:42], fill=(71, 85, 105), font=small)
        draw.text((x + 18, y + 104), f"{stage[3]} rows / {stage[4]} assets", fill=(15, 23, 42), font=small)
    rows = math.ceil(6 / cols)
    y2 = top + rows * 190 + 30
    rounded(draw, (margin, y2, size[0] - margin, min(size[1] - 40, y2 + 290)), 18, (15, 23, 42), None)
    draw.text((margin + 26, y2 + 28), "Mock-safe result gallery", fill=(236, 254, 255), font=body)
    for i, palette in enumerate(PALETTES):
        draw_character(draw, margin + 105 + i * (120 if cols == 3 else 90), y2 + 160, 0.5, palette, ["idle", "run", "jump"][i])
    path = SCREENSHOTS / name
    img.save(path)
    return f"assets/screenshots/{name}"


def save_video() -> str:
    from PIL import Image, ImageDraw

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    path = VIDEO_DIR / "demo-walkthrough.webm"
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        for i in range(60):
            img = Image.new("RGB", (1280, 720), (15, 23, 42))
            draw = ImageDraw.Draw(img)
            draw.text((48, 36), "2D Animation LoRA Pipeline", fill=(236, 254, 255), font=font(38))
            draw.text((50, 92), "Static mock-safe walkthrough: frames -> tracks -> masks -> clusters -> LoRA samples", fill=(186, 230, 253), font=font(18))
            progress = i / 59
            draw.rectangle((64, 614, 1216, 646), outline=(148, 163, 184), width=2)
            draw.rectangle((64, 614, 64 + int(1152 * progress), 646), fill=(20, 184, 166))
            for j, palette in enumerate(PALETTES):
                x = 190 + j * 320 + int(math.sin(progress * math.pi * 2 + j) * 34)
                pose = ["idle", "run", "jump"][(i // 12 + j) % 3]
                draw_character(draw, x, 310 + int(math.sin(progress * math.pi * 4 + j) * 18), 0.88, palette, pose)
            draw.text((64, 664), f"step {min(7, int(progress * 8)) + 1}/8 - demo-ready public artifacts", fill=(226, 232, 240), font=font(18))
            img.save(tmpdir / f"frame_{i:04d}.png")
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-framerate",
                    "12",
                    "-i",
                    str(tmpdir / "frame_%04d.png"),
                    "-c:v",
                    "libvpx-vp9",
                    "-pix_fmt",
                    "yuv420p",
                    "-b:v",
                    "0",
                    "-crf",
                    "34",
                    str(path),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            shutil.copy(tmpdir / "frame_0000.png", path)
    return "assets/video/demo-walkthrough.webm"


def build_manifest() -> dict[str, Any]:
    for directory in (DEMO_ASSETS, SCREENSHOTS, VIDEO_DIR, MANIFEST.parent):
        directory.mkdir(parents=True, exist_ok=True)

    assets = {
        "character_sheet": save_character_sheet(),
        "before_after": save_pipeline_visual(),
        "training_metrics": save_training_metrics(),
        "evaluation_matrix": save_evaluation_matrix(),
        "animation_strip": save_animation_strip(),
    }
    from PIL import Image

    cover_source = WEB / assets["character_sheet"]
    Image.open(cover_source).save(ASSETS / "cover.webp", format="WEBP", quality=88)
    screenshots = [
        save_screenshot_like("01-demo-home-desktop.png", "Reviewer dashboard: pipeline status, stage health, and result cards."),
        save_screenshot_like("02-demo-results-desktop.png", "Result gallery: anonymized sheets, evaluation metrics, and motion strip."),
        save_screenshot_like("03-demo-home-mobile.png", "Responsive mobile review surface."),
    ]
    video = save_video()
    return {
        "project": "2D Animation LoRA Pipeline",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "cpu_stub_demo",
        "summary": {
            "stages_total": len(STAGES),
            "stages_complete": len(STAGES),
            "metadata_rows": sum(stage[3] for stage in STAGES),
            "image_artifacts": sum(stage[4] for stage in STAGES),
            "demo_ready": True,
        },
        "product_results": {
            "headline": "A public, mock-safe demo layer for a GPU-heavy character LoRA pipeline.",
            "assets": assets,
            "metrics": [
                {"label": "Identity retention", "value": "91%", "trend": "HDBSCAN track merge demo"},
                {"label": "Mask acceptance", "value": "88%", "trend": "ToonOut-style clean cutouts"},
                {"label": "Caption coverage", "value": "100%", "trend": "stub VLM captions"},
                {"label": "CPU demo runtime", "value": "< 10s", "trend": "no GPU required"},
            ],
            "deliverables": [
                "Static product demo site",
                "Synthetic 2D dataset gallery",
                "Pipeline stage dashboard",
                "Screenshot and WebM walkthrough package",
                "GitHub Pages and Docker/Nginx deployment path",
            ],
            "media": {"screenshots": screenshots, "video": video, "poster": screenshots[0]},
        },
        "scenarios": [
            {"id": "review", "title": "Fast interview review", "description": "Open the public site, scan stage readiness, play the walkthrough, then inspect architecture diagrams."},
            {"id": "cpu-safe", "title": "CPU-safe smoke path", "description": "Regenerate manifest and public assets without private videos, model weights, API keys, or CUDA."},
            {"id": "full-mode", "title": "Full GPU extension", "description": "Use the same stage model with real local video/model warehouses for YOLO, ToonOut, DWpose, and LoRA training."},
        ],
        "stages": [
            {"order": idx, "id": sid, "label": label, "description": desc, "status": "complete", "rows": rows, "images": images}
            for idx, (sid, label, desc, rows, images) in enumerate(STAGES, start=1)
        ],
        "commands": {
            "generate_manifest": "python scripts/demo/run_demo_pipeline.py --skip-pipeline",
            "serve_site": "python -m http.server 8080 -d portfolio-web",
            "smoke_tests": "python -m pytest tests/demo tests/test_end_to_end_pipeline.py -q",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate public demo assets and manifest.")
    parser.add_argument("--skip-pipeline", action="store_true", help="Kept for compatibility; this generator is already mock-safe.")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    manifest = build_manifest()
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.manifest}")
    print(f"Demo ready: {manifest['summary']['demo_ready']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
