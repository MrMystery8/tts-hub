from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class WatermarkRun:
    id: str  # workspace-relative path (from hub_root)
    label: str
    path: Path
    status: str | None = None
    updated_at: float | None = None


class WatermarkService:
    """
    Lightweight watermarking + attribution wrapper for the TTS Hub web UI.

    Notes:
    - Watermarking models operate at 16kHz (watermark.config.SAMPLE_RATE).
    - For now, we only map 2 TTS models to attribution IDs:
        - index-tts2 -> pred_model_id 0 (class_id 1)
        - chatterbox-multilingual -> pred_model_id 1 (class_id 2)
    """

    # Mapping: TTS model id -> pred_model_id (0..K-1). Encoder class_id is pred_model_id + 1.
    _TTS_TO_ATTR_ID: dict[str, int] = {
        "index-tts2": 0,
        "chatterbox-multilingual": 1,
    }

    def __init__(self, *, hub_root: Path):
        self.hub_root = hub_root
        self.outputs_root = hub_root / "outputs"
        self._lock = threading.Lock()
        self._cache: dict[str, dict[str, Any]] = {}

    def list_runs(self) -> list[WatermarkRun]:
        if not self.outputs_root.exists():
            return []

        runs: list[WatermarkRun] = []
        candidates: list[Path] = []
        for p in self.outputs_root.iterdir():
            if not p.is_dir():
                continue
            candidates.append(p)

            # Live dashboard stores sessions at outputs/dashboard_runs/<sid>/...
            if p.name.startswith("dashboard_runs"):
                for c in p.iterdir():
                    if c.is_dir():
                        candidates.append(c)

        for p in candidates:
            enc = p / "encoder.pt"
            dec = p / "decoder.pt"
            if not enc.exists() or not dec.exists():
                continue

            status: str | None = None
            label = p.name
            updated_at: float | None = None

            session_path = p / "session.json"
            if session_path.exists():
                try:
                    sess = json.loads(session_path.read_text(encoding="utf-8"))
                    status = str(sess.get("status") or "").strip() or None
                    label = str(sess.get("name") or "").strip() or label
                except Exception:
                    status = status

            try:
                updated_at = max(enc.stat().st_mtime, dec.stat().st_mtime)
            except Exception:
                updated_at = None

            run_id = str(p.relative_to(self.hub_root))
            runs.append(WatermarkRun(id=run_id, label=label, path=p, status=status, updated_at=updated_at))

        def sort_key(r: WatermarkRun):
            # Prefer completed dashboard runs, then newest.
            is_completed = 1 if (r.status or "").lower() == "completed" else 0
            t = r.updated_at or 0.0
            return (is_completed, t)

        runs.sort(key=sort_key, reverse=True)
        return runs

    def get_default_run_id(self) -> str | None:
        runs = self.list_runs()
        if not runs:
            return None
        # Prefer completed if present (list_runs sorts that way).
        return runs[0].id

    def get_run_details(self, *, run_id: str | None) -> dict[str, Any]:
        """
        Return small, UI-friendly details for a given run.

        If run_id is None/empty, uses the default run (latest completed if available).
        """
        run_id_used = (run_id or "").strip() or self.get_default_run_id()
        if not run_id_used:
            raise RuntimeError("No watermark runs available")

        run_dir = self._resolve_run_dir(run_id_used)

        status: str | None = None
        label: str | None = None
        session_path = run_dir / "session.json"
        if session_path.exists():
            try:
                sess = json.loads(session_path.read_text(encoding="utf-8"))
                status = str(sess.get("status") or "").strip() or None
                label = str(sess.get("name") or "").strip() or None
            except Exception:
                status = status

        enc = run_dir / "encoder.pt"
        dec = run_dir / "decoder.pt"
        updated_at: float | None = None
        try:
            updated_at = max(enc.stat().st_mtime, dec.stat().st_mtime)
        except Exception:
            updated_at = None

        report_excerpt = None
        report_path = run_dir / "report.md"
        if report_path.exists():
            report_excerpt = self._read_text_head(report_path, max_chars=1400)

        metrics = None
        metrics_path = run_dir / "metrics.jsonl"
        if metrics_path.exists():
            metrics = self._read_latest_probe_metrics(metrics_path)

        run_config = None
        config_path = run_dir / "config.json"
        if config_path.exists():
            try:
                run_config = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:
                run_config = None

        return {
            "id": str(run_id_used),
            "label": label or run_dir.name,
            "status": status,
            "updated_at": updated_at,
            "report_excerpt": report_excerpt,
            "metrics": metrics,
            "config": run_config,
        }

    def embed_into_wav(
        self,
        *,
        input_wav_path: Path,
        output_wav_path: Path,
        tts_model_id: str,
        run_id: str | None,
    ) -> None:
        attr_id = self._TTS_TO_ATTR_ID.get(tts_model_id)
        if attr_id is None:
            raise ValueError(f"Watermark model mapping not configured for model_id={tts_model_id!r}")
        class_id = int(attr_id + 1)

        encoder, _decoder, run_id_used = self._load_models(run_id=run_id)
        audio_16k, orig_sr = self._load_audio_any_sr(input_wav_path)

        wm_audio_16k = self._encode_audio(encoder=encoder, audio_16k=audio_16k, class_id=class_id)
        wm_audio_out = self._resample_back(wm_audio_16k, orig_sr=orig_sr)

        self._save_audio(output_wav_path, wm_audio_out, sr=orig_sr)

        # Touch cache entry so callers can surface which run was used (if desired).
        _ = run_id_used

    def detect_from_audio_file(
        self,
        *,
        audio_path: Path,
        run_id: str | None,
        wm_threshold: float = 0.8,
    ) -> dict[str, Any]:
        _encoder, decoder, run_id_used = self._load_models(run_id=run_id)
        audio_16k, _orig_sr = self._load_audio_any_sr(audio_path)
        outputs = self._decode_audio(decoder=decoder, audio_16k=audio_16k)
        decision = self._decide(outputs, wm_threshold=wm_threshold)

        pred_attr_id: int | None = decision.get("pred_model_id")
        tts_model_id: str | None = None
        for k, v in self._TTS_TO_ATTR_ID.items():
            if pred_attr_id == v:
                tts_model_id = k
                break

        return {
            "detected": bool(decision.get("positive")),
            "wm_prob": float(decision.get("clip_wm_prob", 0.0)),
            "pred_attr_id": pred_attr_id,
            "tts_model_id": tts_model_id,
            "run_id": run_id_used,
        }

    # -----------------
    # Internals
    # -----------------
    def _resolve_run_dir(self, run_id: str) -> Path:
        run_id = str(run_id or "").strip()
        if not run_id:
            raise ValueError("run_id is required")

        p = (self.hub_root / run_id).resolve()
        if self.hub_root.resolve() not in p.parents and p != self.hub_root.resolve():
            raise ValueError("Invalid run_id (must resolve under hub_root)")
        if not p.exists() or not p.is_dir():
            raise FileNotFoundError(f"Run not found: {run_id}")
        if not (p / "encoder.pt").exists() or not (p / "decoder.pt").exists():
            raise FileNotFoundError(f"Run missing encoder.pt/decoder.pt: {run_id}")
        return p

    def _load_models(self, *, run_id: str | None):
        run_id_used = (run_id or "").strip() or self.get_default_run_id()
        if not run_id_used:
            raise RuntimeError("No watermark runs available (expected outputs/*/encoder.pt + decoder.pt)")

        with self._lock:
            cached = self._cache.get(run_id_used)
            if cached is not None:
                return cached["encoder"], cached["decoder"], run_id_used

            run_dir = self._resolve_run_dir(run_id_used)

            # Heavy imports are deferred so non-watermark use stays light.
            import torch

            from watermark.config import DEVICE, N_CLASSES
            from watermark.models.decoder import SlidingWindowDecoder, WatermarkDecoder
            from watermark.models.encoder import OverlapAddEncoder, WatermarkEncoder

            num_classes = int(N_CLASSES)
            cfg_path = run_dir / "config.json"
            if cfg_path.exists():
                try:
                    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                    if cfg.get("num_classes", None) is not None:
                        num_classes = int(cfg.get("num_classes"))
                    elif cfg.get("n_classes", None) is not None:
                        num_classes = int(cfg.get("n_classes"))
                    elif cfg.get("n_models", None) is not None:
                        num_classes = int(cfg.get("n_models")) + 1
                except Exception:
                    num_classes = int(N_CLASSES)

            if int(num_classes) < 3:
                raise ValueError(
                    f"Watermark run has num_classes={num_classes} (<3). "
                    "Need at least clean + 2 model IDs for the hub mapping."
                )

            encoder = OverlapAddEncoder(WatermarkEncoder(num_classes=num_classes)).to(DEVICE)
            decoder = SlidingWindowDecoder(WatermarkDecoder(num_classes=num_classes)).to(DEVICE)

            encoder_state = torch.load(run_dir / "encoder.pt", map_location="cpu")
            decoder_state = torch.load(run_dir / "decoder.pt", map_location="cpu")
            encoder.load_state_dict(encoder_state, strict=True)
            decoder.load_state_dict(decoder_state, strict=True)

            encoder.eval()
            decoder.eval()

            self._cache[run_id_used] = {"encoder": encoder, "decoder": decoder}
            return encoder, decoder, run_id_used

    def _load_audio_any_sr(self, path: Path):
        import numpy as np
        import soundfile as sf
        import torch
        import torchaudio

        from watermark.config import SAMPLE_RATE

        audio_np, sr = sf.read(str(path), dtype="float32", always_2d=True)
        # audio_np: (T, C)
        if audio_np.shape[1] > 1:
            audio_np = audio_np.mean(axis=1, keepdims=True)

        audio = torch.from_numpy(audio_np.T)  # (1, T)
        if int(sr) != int(SAMPLE_RATE):
            audio = torchaudio.functional.resample(audio, int(sr), int(SAMPLE_RATE))

        return audio, int(sr)

    def _save_audio(self, path: Path, audio: Any, *, sr: int) -> None:
        import numpy as np
        import soundfile as sf

        path.parent.mkdir(parents=True, exist_ok=True)

        # Accept torch tensor or numpy.
        if hasattr(audio, "detach"):
            audio = audio.detach().cpu().numpy()
        audio = np.asarray(audio)
        if audio.ndim == 2 and audio.shape[0] == 1:
            audio = audio[0]
        audio = np.clip(audio, -1.0, 1.0)
        sf.write(str(path), audio, int(sr))

    def _encode_audio(self, *, encoder: Any, audio_16k: Any, class_id: int):
        import torch

        from watermark.config import DEVICE

        audio = audio_16k
        if hasattr(audio, "detach"):
            audio = audio.detach()
        if audio.dim() != 2 or audio.shape[0] != 1:
            raise ValueError(f"Expected audio shape (1, T), got {tuple(audio.shape)}")

        x = audio.unsqueeze(0).to(DEVICE)  # (B=1, C=1, T)
        cls = torch.tensor([int(class_id)], dtype=torch.long, device=DEVICE)

        with torch.inference_mode():
            y = encoder(x, cls)  # (1, 1, T)
        return y.squeeze(0).detach().to("cpu")

    def _decode_audio(self, *, decoder: Any, audio_16k: Any) -> dict[str, Any]:
        import torch

        from watermark.config import DEVICE

        audio = audio_16k
        if hasattr(audio, "detach"):
            audio = audio.detach()
        if audio.dim() != 2 or audio.shape[0] != 1:
            raise ValueError(f"Expected audio shape (1, T), got {tuple(audio.shape)}")

        x = audio.unsqueeze(0).to(DEVICE)  # (1, 1, T)
        with torch.inference_mode():
            return decoder(x)

    def _decide(self, outputs: dict[str, Any], *, wm_threshold: float) -> dict[str, Any]:
        from watermark.models.decoder import AttributionDecisionRule

        rule = AttributionDecisionRule(wm_threshold=float(wm_threshold))
        return rule.decide(outputs)

    def _resample_back(self, audio_16k: Any, *, orig_sr: int):
        import torchaudio

        from watermark.config import SAMPLE_RATE

        if int(orig_sr) == int(SAMPLE_RATE):
            return audio_16k
        if hasattr(audio_16k, "detach"):
            audio_16k = audio_16k.detach()
        # (1, T)
        return torchaudio.functional.resample(audio_16k, int(SAMPLE_RATE), int(orig_sr))

    def _read_text_head(self, path: Path, *, max_chars: int) -> str:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return f.read(max(0, int(max_chars)))
        except Exception:
            return ""

    def _read_latest_probe_metrics(self, path: Path) -> dict[str, Any] | None:
        """
        Best-effort extraction of the latest metrics from a JSONL log.
        Prefers `type == "test_probe"` (end-of-run held-out test), otherwise falls back to the most
        recent `type == "probe"` (validation probe during training).
        """
        import os

        max_bytes = 96 * 1024
        try:
            size = os.path.getsize(path)
            start = max(0, size - max_bytes)
            with open(path, "rb") as f:
                f.seek(start)
                chunk = f.read()
        except Exception:
            return None

        try:
            text = chunk.decode("utf-8", errors="replace")
        except Exception:
            return None

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return None

        best: dict[str, Any] | None = None
        fallback: dict[str, Any] | None = None
        for ln in reversed(lines):
            try:
                obj = json.loads(ln)
            except Exception:
                continue
            if isinstance(obj, dict):
                fallback = obj
                if obj.get("type") == "test_probe":
                    best = obj
                    break
                if best is None and obj.get("type") == "probe":
                    best = obj

        obj = best or fallback
        if not isinstance(obj, dict):
            return None

        out: dict[str, Any] = {}
        for k in (
            "type",
            "stage",
            "epoch",
            "mini_auc",
            "mini_auc_reverb",
            "detect_pos_mean",
            "detect_neg_mean",
            "thr_at_fpr_1pct",
            "tpr_at_fpr_1pct",
            "tpr_at_fpr_1pct_reverb",
            "id_acc_pos",
            "id_acc_pos_reverb",
            "attr_acc",
            "wm_acc",
            "wm_acc_reverb",
            "wm_snr_db_mean",
            "wm_budget_ok_frac",
        ):
            if k in obj:
                out[k] = obj[k]
        return out or None
