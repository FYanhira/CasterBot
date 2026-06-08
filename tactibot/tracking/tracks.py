"""Construcción de tracks.json desde salidas SAM 3."""

from __future__ import annotations

from typing import Any

import numpy as np

from tactibot.perception.sam3_wrapper import mask_to_bbox


def _centroid(bbox: list[float]) -> list[float]:
    x1, y1, x2, y2 = bbox
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]


def _classify_robot_team(
    centroid: list[float],
    width: int,
    height: int,
    axis: str,
    ally_side: str,
) -> str:
    """Divide detecciones 'robot' en ally / opponent por mitad del campo."""
    cx, cy = centroid
    if axis == "y":
        mid = height / 2.0
        on_ally_side = cy < mid
        if ally_side in ("top", "left"):
            return "ally" if on_ally_side else "opponent"
        return "opponent" if on_ally_side else "ally"
    mid_x = width / 2.0
    on_left = cx < mid_x
    if ally_side == "left":
        return "ally" if on_left else "opponent"
    return "opponent" if on_left else "ally"


def masks_to_frame_detections(
    perception_by_concept: dict[str, dict[int, dict[int, np.ndarray]]],
    width: int,
    height: int,
    team_split: dict[str, str] | None = None,
) -> dict[int, list[dict[str, Any]]]:
    """
    Fusiona máscaras por frame en lista de objetos con clase semántica.
    track_id estable: "{semantic}_{sam_obj_id}" por pasada de prompt.
    """
    team_split = team_split or {"axis": "x", "ally_side": "left"}
    frame_indices: set[int] = set()
    for by_frame in perception_by_concept.values():
        frame_indices.update(by_frame.keys())

    frames_out: dict[int, list[dict[str, Any]]] = {}
    for frame_idx in sorted(frame_indices):
        objects: list[dict[str, Any]] = []
        for concept, by_frame in perception_by_concept.items():
            masks = by_frame.get(frame_idx, {})
            for obj_id, mask in masks.items():
                bbox = mask_to_bbox(mask)
                if bbox is None:
                    continue
                centroid = _centroid(bbox)
                if concept == "robot":
                    semantic = _classify_robot_team(
                        centroid,
                        width,
                        height,
                        team_split.get("axis", "x"),
                        team_split.get("ally_side", "left"),
                    )
                elif concept == "field":
                    semantic = "field"
                elif concept == "ball":
                    semantic = "ball"
                else:
                    semantic = concept

                track_id = f"{semantic}_{concept}_{obj_id}"
                objects.append(
                    {
                        "track_id": track_id,
                        "class": semantic,
                        "sam_concept": concept,
                        "sam_obj_id": int(obj_id),
                        "bbox_xyxy": bbox,
                        "centroid": centroid,
                    }
                )
        frames_out[frame_idx] = objects
    return frames_out


def build_tracks_json(
    video_id: str,
    fps: float,
    width: int,
    height: int,
    frame_detections: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    frames = []
    for frame_idx in sorted(frame_detections.keys()):
        frames.append(
            {
                "frame_idx": frame_idx,
                "time_sec": round(frame_idx / fps, 4) if fps > 0 else 0.0,
                "objects": frame_detections[frame_idx],
            }
        )
    return {
        "video_id": video_id,
        "fps": fps,
        "width": width,
        "height": height,
        "frames": frames,
    }
