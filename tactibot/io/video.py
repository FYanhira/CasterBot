"""Utilidades de video (frames para SAM 3)."""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2


def get_video_meta(video_path: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir el video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return {
        "fps": float(fps),
        "width": width,
        "height": height,
        "frame_count": frame_count,
    }


def extract_frames(
    video_path: Path,
    output_dir: Path,
    max_frames: int | None = None,
) -> tuple[int, float]:
    """Extrae frames JPEG numerados para SAM 3."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir el video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    idx = 0
    while True:
        if max_frames is not None and idx >= max_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(str(output_dir / f"{idx:05d}.jpg"), frame)
        idx += 1
    cap.release()
    return idx, float(fps)
