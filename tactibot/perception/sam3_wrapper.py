"""Wrapper SAM 3 — usa el paquete Meta instalado vía pip (no modifica upstream)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

# SECURITY-REVIEW: HF_TOKEN debe venir del entorno; huggingface_hub lo lee al descargar checkpoints.


def mask_to_bbox(mask: np.ndarray) -> list[float] | None:
    """bbox [x1,y1,x2,y2] desde máscara booleana."""
    m = mask.astype(bool)
    if not m.any():
        return None
    ys, xs = np.where(m)
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def _masks_from_propagate_response(response: dict) -> dict[int, np.ndarray]:
    outputs = response.get("outputs", {})
    obj_ids = outputs.get("out_obj_ids", [])
    binary_masks = outputs.get("out_binary_masks")
    if binary_masks is None:
        return {}
    if isinstance(obj_ids, torch.Tensor):
        obj_ids = obj_ids.cpu().numpy()
    if isinstance(binary_masks, torch.Tensor):
        binary_masks = binary_masks.cpu().numpy()
    masks: dict[int, np.ndarray] = {}
    for i, oid in enumerate(obj_ids):
        m = binary_masks[i]
        if m.ndim == 3:
            m = m[0]
        masks[int(oid)] = np.asarray(m).astype(bool)
    return masks


class Sam3Perception:
    """Segmentación y propagación en video vía API handle_request de SAM 3."""

    def __init__(self, sam3_cfg: dict[str, Any]):
        self.cfg = sam3_cfg
        self._predictor = None

    def _build_predictor(self):
        if self._predictor is not None:
            return self._predictor
        try:
            from sam3 import build_sam3_predictor
        except ImportError as e:
            raise ImportError(
                "Falló la importación de 'sam3' o una dependencia transitiva.\n"
                "  pip install git+https://github.com/facebookresearch/sam3.git\n"
                "  pip install einops\n"
                "Requiere PyTorch con CUDA y acceso Hugging Face (HF_TOKEN o hf auth login).\n"
                f"Causa original: {e}"
            ) from e

        version = self.cfg.get("version", "sam3")
        kwargs: dict[str, Any] = {
            "version": version,
            "compile": bool(self.cfg.get("compile", False)),
            "async_loading_frames": bool(self.cfg.get("async_loading_frames", False)),
        }
        ckpt = self.cfg.get("checkpoint_path")
        if ckpt:
            kwargs["checkpoint_path"] = ckpt
        self._predictor = build_sam3_predictor(**kwargs)
        return self._predictor

    def propagate_text_prompt(
        self,
        frame_dir: Path,
        text: str,
        frame_index: int = 0,
    ) -> dict[int, dict[int, np.ndarray]]:
        """
        Una pasada: prompt de texto en frame_index y propagación a todo el clip.
        Retorna {frame_idx: {obj_id: mask}}.
        """
        predictor = self._build_predictor()
        if torch.cuda.is_available():
            torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

        resp = predictor.handle_request(
            {
                "type": "start_session",
                "resource_path": str(frame_dir),
            }
        )
        session_id = resp["session_id"]

        predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": frame_index,
                "text": text,
            }
        )

        mask_dict: dict[int, dict[int, np.ndarray]] = {}
        for stream_resp in predictor.handle_stream_request(
            {"type": "propagate_in_video", "session_id": session_id}
        ):
            frame_idx = stream_resp.get("frame_index")
            if frame_idx is None:
                continue
            mask_dict[int(frame_idx)] = _masks_from_propagate_response(stream_resp)

        try:
            predictor.handle_request(
                {"type": "close_session", "session_id": session_id}
            )
        except Exception:
            pass

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return mask_dict

    def run_all_prompts(
        self,
        frame_dir: Path,
        prompts: dict[str, str],
        frame_index: int = 0,
    ) -> dict[str, dict[int, dict[int, np.ndarray]]]:
        """Una sesión SAM 3 por concepto (field, ball, robot, ...)."""
        username = os.environ.get("USER", os.environ.get("USERNAME", "user"))
        os.environ.setdefault(
            "TORCHINDUCTOR_CACHE_DIR", f"/tmp/torchinductor_cache_{username}"
        )
        results: dict[str, dict[int, dict[int, np.ndarray]]] = {}
        for concept, text in prompts.items():
            if not text:
                continue
            results[concept] = self.propagate_text_prompt(
                frame_dir, text=text, frame_index=frame_index
            )
        return results
