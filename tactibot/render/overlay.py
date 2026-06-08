"""Render de overlay y video lado a lado (Fase 1)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

CLASS_COLORS = {
    "field": (0, 180, 0),
    "ball": (0, 255, 255),
    "ally": (255, 120, 0),
    "opponent": (0, 80, 255),
    "robot": (200, 200, 200),
}


def _draw_objects(frame: np.ndarray, objects: list[dict], alpha: float) -> np.ndarray:
    out = frame.copy()
    overlay = frame.copy()
    for obj in objects:
        if obj["class"] == "field":
            continue
        bbox = obj.get("bbox_xyxy")
        if not bbox:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        color = CLASS_COLORS.get(obj["class"], (255, 255, 255))
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        label = obj["class"][:3]
        cv2.putText(
            overlay,
            label,
            (x1, max(y1 - 4, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)


def _events_at_frame(events: list[dict], frame_idx: int) -> list[str]:
    labels = []
    for ev in events:
        if ev.get("frame_idx") == frame_idx:
            labels.append(ev["type"])
    return labels


def render_overlay_video(
    video_path: Path,
    tracks_path: Path,
    events_path: Path | None,
    output_path: Path,
    render_cfg: dict[str, Any] | None = None,
    max_frames: int | None = None,
) -> Path:
    """Escribe MP4: original | overlay con cajas y eventos en pantalla."""
    render_cfg = render_cfg or {}
    alpha = float(render_cfg.get("overlay_alpha", 0.45))
    out_fps = render_cfg.get("output_fps")

    with tracks_path.open(encoding="utf-8") as f:
        tracks = json.load(f)
    events_list: list[dict] = []
    if events_path and events_path.is_file():
        with events_path.open(encoding="utf-8") as f:
            events_list = json.load(f).get("events", [])

    fps = out_fps or tracks.get("fps", 30.0)
    frames_data = {fr["frame_idx"]: fr for fr in tracks.get("frames", [])}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir video: {video_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if out_fps is None:
        fps = cap.get(cv2.CAP_PROP_FPS) or fps

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w * 2, h),
    )

    idx = 0
    while True:
        if max_frames is not None and idx >= max_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break
        fr = frames_data.get(idx, {"objects": []})
        overlay = _draw_objects(frame, fr.get("objects", []), alpha)
        ev_labels = _events_at_frame(events_list, idx)
        if ev_labels:
            cv2.putText(
                overlay,
                " | ".join(ev_labels),
                (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        combined = np.hstack([frame, overlay])
        writer.write(combined)
        idx += 1

    cap.release()
    writer.release()
    return output_path
