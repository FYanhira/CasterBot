"""Orquestación del pipeline CASTERBOT Fase 1."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import yaml

from tactibot.config.loader import REPO_ROOT, load_config, snapshot_config
from tactibot.events.engine import detect_events
from tactibot.io.video import extract_frames, get_video_meta
from tactibot.narrative.templates import build_script_json
from tactibot.perception.sam3_wrapper import Sam3Perception
from tactibot.render.overlay import render_overlay_video
from tactibot.tracking.tracks import build_tracks_json, masks_to_frame_detections


def run_pipeline(cfg: dict[str, Any]) -> Path:
    """Ejecuta perception → tracks → events → script → render. Retorna output dir."""
    video_path = Path(cfg["video_path"])
    if not video_path.is_file():
        raise FileNotFoundError(
            f"Video no encontrado: {video_path}\n"
            "Copia un .mp4 a data/videos/ y actualiza video_path en tu config."
        )

    run_name = cfg.get("run_name", "run")
    out_dir = REPO_ROOT / "outputs" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "config.snapshot.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(snapshot_config(cfg), f, allow_unicode=True)

    meta = get_video_meta(video_path)
    max_frames = cfg.get("max_frames")
    video_id = video_path.stem

    frame_dir = Path(tempfile.mkdtemp(prefix="tactibot_frames_"))
    try:
        n_frames, fps = extract_frames(
            video_path,
            frame_dir,
            max_frames=max_frames,
            max_short_side=cfg.get("frame_max_short_side"),
        )
        if n_frames == 0:
            raise RuntimeError("El video no produjo frames.")

        sam3_cfg = cfg.get("sam3", {})
        perception = Sam3Perception(sam3_cfg)
        prompts = sam3_cfg.get("prompts", {})
        frame_index = int(sam3_cfg.get("frame_index", 0))

        perception_results = perception.run_all_prompts(
            frame_dir,
            prompts=prompts,
            frame_index=frame_index,
        )

        frame_dets = masks_to_frame_detections(
            perception_results,
            width=meta["width"],
            height=meta["height"],
            team_split=cfg.get("team_split"),
        )
        tracks = build_tracks_json(
            video_id=video_id,
            fps=fps,
            width=meta["width"],
            height=meta["height"],
            frame_detections=frame_dets,
        )
        tracks_path = out_dir / "tracks.json"
        with tracks_path.open("w", encoding="utf-8") as f:
            json.dump(tracks, f, indent=2)

        events_data = detect_events(tracks, cfg.get("events", {}))
        events_path = out_dir / "events.json"
        with events_path.open("w", encoding="utf-8") as f:
            json.dump(events_data, f, indent=2)

        script = build_script_json(events_data)
        script_path = out_dir / "script.json"
        with script_path.open("w", encoding="utf-8") as f:
            json.dump(script, f, indent=2, ensure_ascii=False)

        render_path = out_dir / "renders" / "demo_side_by_side.mp4"
        render_overlay_video(
            video_path,
            tracks_path,
            events_path,
            render_path,
            render_cfg=cfg.get("render"),
            max_frames=max_frames,
        )

        summary = {
            "video_path": str(video_path),
            "frames_processed": n_frames,
            "fps": fps,
            "tracks": str(tracks_path),
            "events": str(events_path),
            "events_count": len(events_data.get("events", [])),
            "render": str(render_path),
        }
        with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return out_dir
    finally:
        shutil.rmtree(frame_dir, ignore_errors=True)
