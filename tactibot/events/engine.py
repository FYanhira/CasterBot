"""Event Engine heurístico (Fase 1)."""

from __future__ import annotations

import math
from typing import Any


def _bbox_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _get_ball_carrier(
    objects: list[dict],
    d_attach: float,
) -> dict[str, Any] | None:
    balls = [o for o in objects if o["class"] == "ball"]
    robots = [o for o in objects if o["class"] in ("ally", "opponent")]
    if not balls or not robots:
        return None
    ball = balls[0]
    bc = ball["centroid"]
    best, best_d = None, float("inf")
    for r in robots:
        rc = r["centroid"]
        d = math.hypot(bc[0] - rc[0], bc[1] - rc[1])
        if d < best_d:
            best_d, best = d, r
    if best is None or best_d > d_attach:
        return None
    return {"ball": ball, "carrier": best, "distance": best_d}


def detect_events(
    tracks: dict[str, Any],
    events_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Genera events.json desde tracks.json."""
    fps = tracks.get("fps", 30.0)
    height = tracks.get("height", 720)
    frames = tracks.get("frames", [])

    d_attach = float(events_cfg.get("d_attach_px", 80))
    iou_collision = float(events_cfg.get("iou_collision", 0.15))
    goal_y_frac = float(events_cfg.get("goal_y_fraction", 0.08))
    n_goal_frames = int(events_cfg.get("n_goal_frames", 5))
    smooth_n = int(events_cfg.get("possession_smooth_frames", 3))

    events: list[dict[str, Any]] = []
    event_id = 0

    prev_team: str | None = None
    team_history: list[str | None] = []
    goal_streak = 0

    for frame in frames:
        frame_idx = frame["frame_idx"]
        time_sec = frame["time_sec"]
        objects = frame["objects"]
        robots = [o for o in objects if o["class"] in ("ally", "opponent")]

        # Colisiones robot-robot
        for i in range(len(robots)):
            for j in range(i + 1, len(robots)):
                iou = _bbox_iou(robots[i]["bbox_xyxy"], robots[j]["bbox_xyxy"])
                if iou >= iou_collision:
                    event_id += 1
                    events.append(
                        {
                            "id": event_id,
                            "time_sec": time_sec,
                            "frame_idx": frame_idx,
                            "type": "collision",
                            "importance": 0.6,
                            "confidence": min(1.0, iou / max(iou_collision, 1e-6)),
                            "actors": {
                                "track_a": robots[i]["track_id"],
                                "track_b": robots[j]["track_id"],
                            },
                            "metadata": {"iou": round(iou, 4)},
                        }
                    )

        carrier_info = _get_ball_carrier(objects, d_attach)
        team = None
        if carrier_info:
            team = carrier_info["carrier"]["class"]

        team_history.append(team)
        if len(team_history) > smooth_n:
            team_history.pop(0)
        stable_team = team
        if len(team_history) >= smooth_n:
            teams = [t for t in team_history if t is not None]
            if teams and all(t == teams[0] for t in teams):
                stable_team = teams[0]

        if stable_team and prev_team and stable_team != prev_team:
            event_id += 1
            events.append(
                {
                    "id": event_id,
                    "time_sec": time_sec,
                    "frame_idx": frame_idx,
                    "type": "possession_change",
                    "importance": 0.75,
                    "confidence": 0.85,
                    "actors": {"from_team": prev_team, "to_team": stable_team},
                    "metadata": {},
                }
            )
        if stable_team:
            prev_team = stable_team

        # Gol heurístico: balón cerca de borde superior/inferior del frame
        balls = [o for o in objects if o["class"] == "ball"]
        if balls:
            _, by = balls[0]["centroid"]
            margin = height * goal_y_frac
            near_goal = by < margin or by > height - margin
            if near_goal:
                goal_streak += 1
                if goal_streak >= n_goal_frames:
                    event_id += 1
                    events.append(
                        {
                            "id": event_id,
                            "time_sec": time_sec,
                            "frame_idx": frame_idx,
                            "type": "goal",
                            "importance": 1.0,
                            "confidence": 0.7,
                            "actors": {"team": prev_team or "unknown"},
                            "metadata": {"y": by},
                        }
                    )
                    goal_streak = 0
            else:
                goal_streak = 0

            # Tiro heurístico: balón con velocidad alta (entre frames)
            if frame_idx > 0 and len(frames) > 1:
                pass  # shot opcional en Fase 1.1

    # Deduplicar colisiones consecutivas en mismo par
    filtered: list[dict[str, Any]] = []
    last_collision: dict[str, int] = {}
    for ev in sorted(events, key=lambda e: e["frame_idx"]):
        if ev["type"] == "collision":
            key = tuple(sorted(ev["actors"].values()))
            if key in last_collision and ev["frame_idx"] - last_collision[key] < fps * 0.5:
                continue
            last_collision[key] = ev["frame_idx"]
        filtered.append(ev)

    return {
        "video_id": tracks.get("video_id"),
        "fps": fps,
        "events": filtered,
    }
