"""Narración por templates (Fase 1)."""

from __future__ import annotations

from typing import Any

TEMPLATES_ES: dict[str, str] = {
    "goal": "¡Gol! El balón cruza la línea de meta.",
    "possession_change": "Cambio de posesión en el campo.",
    "shot": "Disparo hacia la portería.",
    "collision": "Choque entre robots.",
    "pressure": "Alta presión sobre el portador.",
    "pass": "Pase entre compañeros.",
}


def build_script_json(events_data: dict[str, Any]) -> dict[str, Any]:
    lines = []
    for ev in events_data.get("events", []):
        etype = ev.get("type", "event")
        text = TEMPLATES_ES.get(etype, f"Evento: {etype}")
        lines.append(
            {
                "time_sec": ev.get("time_sec"),
                "text": text,
                "highlight": ev.get("importance", 0) >= 0.7,
                "event_ref": ev.get("id"),
            }
        )
    return {"lines": lines}
