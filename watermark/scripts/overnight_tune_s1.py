#!/usr/bin/env python3
"""
Overnight Auto-Tuning for S1 weights (detect/id/neg).

This script:
1. Generates a fixed shared manifest (deterministic seed).
2. Launches 'quick_voice_smoke_train.py' in successive halving phases.
3. Scores trials using composite_score (tpr_at_fpr_1pct * id_acc_pos).
4. Handles resumes/promotions via checkpoints.
5. Respects hard floors and time budgets.

Usage:
    python -m watermark.scripts.overnight_tune_s1 \
        --source_dir "mini_benchmark_data" \
        --out_root "outputs/dashboard_runs/overnight_001" \
        --max_hours 7.5
"""

import argparse
import csv
import hashlib
import json
import logging
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[Tuner] %(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class TrialConfig:
    trial_id: str
    detect_weight: float
    id_weight: float
    neg_weight: float
    # These track state
    status: str = "pending"  # pending, running, completed, dead, pruned
    current_epoch: int = 0
    score: Optional[float] = None
    run_dir: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TunerState:
    trials: List[TrialConfig]
    shared_manifest_path: str
    seed: int
    start_time_iso: str
    manifest_params: Dict[str, Any]
    best_trial_id: Optional[str] = None


class OvernightTuner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.root_dir = Path(args.out_root).resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)
        
        self.state_path = self.root_dir / "tuner_state.json"
        
        # Initialize or load state
        if self.state_path.exists():
            logger.info(f"Resuming tuner from {self.state_path}")
            self.state = self._load_state()
        else:
            logger.info("Initializing new tuner state")
            self.state = self._init_state()

        self.start_time = datetime.fromisoformat(self.state.start_time_iso)
        self.max_duration = timedelta(hours=args.max_hours)
        
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        def handler(signum, frame):
            logger.warning(f"Received signal {signum}. Exiting cleanly.")
            sys.exit(1)
            
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    def _init_state(self) -> TunerState:
        # 1. Create shared manifest
        logger.info("Generating shared manifest...")
        manifest_path = self.root_dir / "shared_manifest.json"
        
        # We shell out to quick_voice_smoke_train just to reuse its manifest logic? 
        # Or better: use the logic directly if importable, or just run a dummy setup.
        # Actually, quick_voice_smoke_train can build a manifest if we just call it with --num_clips.
        # But we need it to just generate manifest and exit.
        # Let's write a small helper to generate it consistently.
        
        # Note: We can reuse the build_manifest logic if we import it, but imports might be messy if 
        # running as script. Let's assume we invoke the script to generate it.
        # A trick: run quick_voice_smoke_train with dry_run or just 0 epochs?
        # Re-reading quick_voice: it builds manifest before training.
        # So we can run it with --epochs_s1 0 --save_only_manifest? No such flag.
        
        # Alternative: We just carry the manifest generation logic here or import it.
        # Let's try importing, as we are in the same package structure usually.
        # If import fails, we fallback to subprocess? No, let's keep it simple.
        # We will assume this script is run as module: python -m watermark.scripts.overnight_tune_s1
        
        try:
            from watermark.scripts.quick_voice_smoke_train import build_manifest, collect_audio_files
        except ImportError as e:
            # Fallback if path issues, though -m should fix it.
            logger.error(f"Could not import build_manifest: {e}. Run as python -m watermark.scripts.overnight_tune_s1")
            sys.exit(1)

        rng = random.Random(self.args.seed)
        source_dir = Path(self.args.source_dir)
        audio_files = collect_audio_files(source_dir)
        if not audio_files:
            raise ValueError(f"No audio files in {source_dir}")
            
        num_clips = self.args.num_clips
        if num_clips < len(audio_files):
            selected = rng.sample(audio_files, num_clips)
        else:
            selected = list(audio_files)
            while len(selected) < num_clips:
                selected.append(rng.choice(audio_files))
                
        # Manually build manifest so we don't depend on script side-effects
        # But wait, build_manifest writes to out_dir.
        # We can just use the function.
        # The function `build_manifest` signature: (paths: list[Path], out_dir: Path) -> Path
        
        # Logic from quick_voice_smoke_train:
        # manifest: list[dict[str, object]] = []
        # ... logic ...
        # out_dir.mkdir(...)
        # json.dump(...)
        
        # We will use the imported function to be 100% consistent with the training script logic.
        real_manifest_path = build_manifest(selected, self.root_dir)
        # Rename/Move to shared name if needed, but build_manifest saves as manifest.json
        if real_manifest_path.name != "shared_manifest.json":
            shutil.move(real_manifest_path, manifest_path)
            
        manifest_params = {
            "num_clips": num_clips,
            "source_dir": str(source_dir),
            "files_count": len(selected)
        }
        
        return TunerState(
            trials=[],
            shared_manifest_path=str(manifest_path.resolve()),
            seed=self.args.seed,
            start_time_iso=datetime.now().isoformat(),
            manifest_params=manifest_params
        )

    def _load_state(self) -> TunerState:
        with open(self.state_path, "r") as f:
            data = json.load(f)
        # Reconstruct dataclasses
        trials = [TrialConfig(**t) for t in data["trials"]]
        del data["trials"]
        return TunerState(trials=trials, **data)

    def _save_state(self):
        data = asdict(self.state)
        # Atomic save
        tmp = self.state_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, self.state_path)

    def generate_trials(self):
        if self.state.trials:
            logger.info("Trials already generated.")
            return

        rng = random.Random(self.args.seed + 1) # distinct from data seed
        
        # Grid/Range parsing
        def parse_range(s):
            parts = [float(x) for x in s.split(",")]
            if len(parts) == 1: return parts[0], parts[0]
            return min(parts), max(parts)

        d_min, d_max = parse_range(self.args.detect_range)
        i_min, i_max = parse_range(self.args.id_range)
        n_min, n_max = parse_range(self.args.neg_range)

        logger.info(f"Generating {self.args.num_initial} trials...")
        logger.info(f"Ranges: detect=[{d_min}, {d_max}], id=[{i_min}, {i_max}], neg=[{n_min}, {n_max}]")

        for i in range(self.args.num_initial):
            # Strategy: Random Uniform for now (simple and robust for small batch)
            # Could do grid if requested, but random covers space better overnight usually.
            
            # If ranges are single points, we just use them
            d_val = round(rng.uniform(d_min, d_max), 2)
            i_val = round(rng.uniform(i_min, i_max), 2)
            n_val = round(rng.uniform(n_min, n_max), 2)
            
            tid = f"trial_{i+1:03d}_d{d_val}_i{i_val}_n{n_val}"
            t = TrialConfig(
                trial_id=tid,
                detect_weight=d_val,
                id_weight=i_val,
                neg_weight=n_val,
                run_dir=str(self.root_dir / tid)
            )
            self.state.trials.append(t)
        
        self._save_state()

    def run_phase(self, trials: List[TrialConfig], target_epochs: int, phase_name: str):
        logger.info(f"\n=== Starting Phase {phase_name} (Target Epochs: {target_epochs}) ===")
        
        for trial in trials:
            # STOP File Check
            if (self.root_dir / "STOP").exists():
                logger.warning("STOP file detected. Saving state and exiting.")
                return

            # Check budget (Hard stop)
            elapsed = datetime.now() - self.start_time
            if elapsed > self.max_duration:
                logger.warning(f"Time budget exceeded ({elapsed} > {self.max_duration}). Stopping.")
                return

            if trial.status == "dead":
                continue

            # Need to run?
            if trial.current_epoch >= target_epochs:
                logger.info(f"Trial {trial.trial_id} already at epoch {trial.current_epoch}, skipping.")
                continue

            # Pre-flight Time Estimation
            # Calculate global average time per epoch to predict if we can finish this trial
            total_epochs_done = sum(t.current_epoch for t in self.state.trials)
            if total_epochs_done > 0:
                avg_sec_per_epoch = elapsed.total_seconds() / total_epochs_done
                epochs_needed = target_epochs - trial.current_epoch
                est_seconds = epochs_needed * avg_sec_per_epoch
                est_remaining = timedelta(seconds=est_seconds)
                
                if elapsed + est_remaining > self.max_duration:
                     logger.warning(f"Skipping {trial.trial_id}: Est. time {est_remaining} would exceed budget (Elapsed: {elapsed}, Budget: {self.max_duration})")
                     trial.status = "skipped_time"
                     self._save_state()
                     continue

            logger.info(f"Running trial {trial.trial_id} -> {target_epochs} epochs")
            
            # Prepare command
            # If resuming, we need to point to existing run_dir and use --resume
            # If new, we start fresh.
            
            # Actually, we always point to the same run_dir per trial. 
            # If checkpoint exists, we resume.
            # quick_voice_smoke_train automatically handles Resume ONLY IF --resume is passed.
            # So we check if checkpoint exists.
            
            run_dir = Path(trial.run_dir)
            ckpt_path = run_dir / "checkpoints" / "last.pt"
            
            cmd = [
                sys.executable, "-m", "watermark.scripts.quick_voice_smoke_train",
                "--out", str(run_dir),
                "--manifest", self.state.shared_manifest_path,
                "--seed", str(self.state.seed),
                # Weights
                "--detect_weight", str(trial.detect_weight),
                "--id_weight", str(trial.id_weight),
                "--neg_weight", str(trial.neg_weight),
                # Fixed params
                "--reverb_prob", str(self.args.reverb_prob),
                "--probe_every", str(self.args.probe_every),
                "--probe_reverb_every", str(self.args.probe_reverb_every),
                "--probe_clips", str(self.args.probe_clips),
                "--log_steps_every", "25",
                # Metrics
                "--best_metric", "composite_score",
                "--best_mode", "max",
                # Checkpointing
                "--save_last", "--save_best",
                "--save_every", "1",
            ]
            
            # Forward dashboard logging if enabled
            if self.args.log_metrics:
                # We append to the main dashboard log so all trials show up in the timeline
                cmd.extend(["--log_metrics", self.args.log_metrics])
            
            # Epochs logic
            # We want total epochs = target_epochs
            # We use --extend_to_epochs_s1 to enforce this cleanly
            
            # If just starting (Phase A, epoch 0), we set epochs_s1 = target
            # If extending (Phase B/C), we set --resume ... --extend_to_epochs_s1 target
            
            if ckpt_path.exists():
                cmd.extend(["--resume", str(ckpt_path)])
                cmd.extend(["--extend_to_epochs_s1", str(target_epochs)])
                # Pass original epochs args just to be safe/consistent, 
                # but run logic is handled by extend.
                # Actually quick_voice expects some epochs arg usually. 
                # extend overrides s1, so let's just pass reasonable defaults for others (0)
                cmd.extend(["--epochs_s1", "0", "--epochs_s2", "0", "--epochs_s1b_post", "0"])
            else:
                # Fresh start
                cmd.extend(["--epochs_s1", str(target_epochs)])
                cmd.extend(["--epochs_s2", "0"]) # S1 only for tuning usually? 
                # Wait, user prompt said "overnight tune s1 weights". 
                # quick_voice has s1, s2, s3.
                # We assume we are tuning S1 only for now (decoder pretrain).
                # If the user wants full pipeline tuning, that's different.
                # "Spec sheet: Overnight Auto-Tuning for S1 weights" -> Yes, Stage 1.
                cmd.extend(["--epochs_s1b_post", "0"])

            # Execute
            try:
                subprocess.run(cmd, check=True)
                trial.current_epoch = target_epochs
                trial.status = "completed" if phase_name == "C" else "pending_promotion"
            except subprocess.CalledProcessError as e:
                logger.error(f"Trial {trial.trial_id} failed: {e}")
                trial.status = "failed"
            
            # Update score
            self._score_trial(trial)
            self._save_state()
            self._save_summary()

    def _save_summary(self):
        # CSV Summary
        csv_path = self.root_dir / "summary.csv"
        headers = ["trial_id", "status", "epoch", "score", "detect_w", "id_w", "neg_w", 
                   "det_acc_1pct", "id_acc", "composite", "mini_auc"]
        
        # Sort by score (descending) for readability, dead/pending at bottom
        sorted_trials = sorted(self.state.trials, key=lambda t: (t.score if t.score is not None else -1.0), reverse=True)
        
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for t in sorted_trials:
                m = t.metrics
                writer.writerow([
                    t.trial_id, t.status, t.current_epoch, 
                    f"{t.score:.4f}" if t.score is not None else "",
                    t.detect_weight, t.id_weight, t.neg_weight,
                    f"{float(m.get('tpr_at_fpr_1pct', 0)):.4f}", 
                    f"{float(m.get('id_acc_pos', 0)):.4f}",
                    f"{float(m.get('composite_score', 0)):.4f}",
                    f"{float(m.get('mini_auc', 0)):.4f}"
                ])

        # Markdown Summary
        md_path = self.root_dir / "summary.md"
        top_5 = sorted_trials[:5]
        
        md_content = f"# Tuning Summary\n\n"
        md_content += f"**Start Time:** {self.state.start_time_iso}\n"
        md_content += f"**Best Trial:** {self.state.best_trial_id}\n\n"
        
        md_content += "## Top 5 Trials\n\n"
        md_content += "| Trial ID | Score | Status | Epochs | Det (1%) | ID Acc | Weights (D/I/N) |\n"
        md_content += "|---|---|---|---|---|---|---|\n"
        
        for t in top_5:
            score_str = f"{t.score:.4f}" if t.score is not None else "-"
            m = t.metrics
            det = f"{float(m.get('tpr_at_fpr_1pct', 0)):.2f}"
            id_acc = f"{float(m.get('id_acc_pos', 0)):.2f}"
            weights = f"{t.detect_weight}/{t.id_weight}/{t.neg_weight}"
            md_content += f"| {t.trial_id} | {score_str} | {t.status} | {t.current_epoch} | {det} | {id_acc} | {weights} |\n"
            
        md_path.write_text(md_content, encoding="utf-8")

    def _score_trial(self, trial: TrialConfig):
        # Read best_meta.json
        meta_path = Path(trial.run_dir) / "checkpoints" / "best_meta.json"
        if not meta_path.exists():
            logger.warning(f"No best_meta.json for {trial.trial_id}, marking dead.")
            trial.score = None
            trial.status = "dead"
            return

        with open(meta_path, "r") as f:
            meta = json.load(f)
        
        metrics = meta.get("probe_metrics", {})
        trial.metrics = metrics
        
        # Guardrails
        det = float(metrics.get("tpr_at_fpr_1pct", -1.0))
        id_acc = float(metrics.get("id_acc_pos", -1.0))
        comp = float(metrics.get("composite_score", -1.0))
        
        # Floors
        if det < self.args.det_floor or id_acc < self.args.id_floor:
            logger.info(f"Trial {trial.trial_id} DEAD (det={det:.2f}<{self.args.det_floor} or id={id_acc:.2f}<{self.args.id_floor})")
            trial.score = None
            trial.status = "dead"
        else:
            trial.score = comp
            logger.info(f"Trial {trial.trial_id} SCORED {comp:.4f} (det={det:.2f}, id={id_acc:.2f})")

    def run(self):
        self.generate_trials()
        
        # Phase A: All trials
        active_trials = [t for t in self.state.trials if t.status != "dead" and t.status != "failed"]
        self.run_phase(active_trials, self.args.base_epochs, "A")
        
        # Promote for Phase B
        valid_trials = [t for t in self.state.trials if t.score is not None]
        valid_trials.sort(key=lambda x: x.score, reverse=True)
        
        if not valid_trials:
            logger.error("No valid trials after Phase A (all died or failed). Stopping.")
            return

        promoted_b = valid_trials[:2] # Top 2
        logger.info(f"Promoting to Phase B: {[t.trial_id for t in promoted_b]}")
        
        self.run_phase(promoted_b, self.args.promote_epochs, "B")
        
        # Promote for Phase C
        # Re-sort because scores might have changed in Phase B
        valid_b = [t for t in promoted_b if t.score is not None]
        valid_b.sort(key=lambda x: x.score, reverse=True)
        
        if not valid_b:
            logger.error("No valid trials after Phase B. Stopping.")
            return

        winner = valid_b[0] # Top 1
        logger.info(f"Promoting Winner to Phase C: {winner.trial_id}")
        
        self.run_phase([winner], self.args.final_epochs, "C")
        
        self.state.best_trial_id = winner.trial_id
        self._save_state()
        logger.info(f"Tuning Complete. Winner: {winner.trial_id} (Score: {winner.score:.4f})")
        if self.args.log_metrics:
            try:
                # Log completion event for dashboard
                with open(self.args.log_metrics, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "type": "epoch", "stage": "tuning", "epoch": 1, 
                        "best_score": float(winner.score) if winner.score else 0.0,
                        "best_trial": winner.trial_id
                    }) + "\n")
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="Overnight S1 Tuner")
    
    # Dataset
    parser.add_argument("--source_dir", type=str, required=True)
    parser.add_argument("--num_clips", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=1337)
    
    # Search Space
    parser.add_argument("--detect_range", type=str, default="6,12")
    parser.add_argument("--id_range", type=str, default="3,7")
    parser.add_argument("--neg_range", type=str, default="2,5")
    parser.add_argument("--num_initial", type=int, default=8)
    
    # Schedule
    parser.add_argument("--base_epochs", type=int, default=6)
    parser.add_argument("--promote_epochs", type=int, default=18)
    parser.add_argument("--final_epochs", type=int, default=36)
    
    # Probing / Config
    parser.add_argument("--reverb_prob", type=float, default=0.0)
    parser.add_argument("--probe_every", type=int, default=2)
    parser.add_argument("--probe_reverb_every", type=int, default=0)
    parser.add_argument("--probe_clips", type=int, default=1024)
    
    # Objective
    parser.add_argument("--det_floor", type=float, default=0.60)
    parser.add_argument("--id_floor", type=float, default=0.50)
    
    # Constraints
    parser.add_argument("--max_hours", type=float, default=7.5)
    parser.add_argument("--out_root", type=str, default="outputs/dashboard_runs/overnight_tune")
    
    # Dashboard Compatibility
    parser.add_argument("--output", type=str, default=None, help="Alias for out_root (injected by dashboard)")
    parser.add_argument("--log_metrics", type=str, default=None, help="Path to metrics.jsonl (injected by dashboard)")
    
    args = parser.parse_args()
    
    # Handle Dashboard Integration
    if args.output:
        logger.info(f"Using dashboard injected output: {args.output}")
        args.out_root = args.output
        
    if args.log_metrics:
        # Initialize the session log with meta info if it doesn't exist or is empty
        p = Path(args.log_metrics)
        if not p.exists() or p.stat().st_size == 0:
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                json.dump({
                    "type": "meta",
                    "run_name": "overnight_tune_s1",
                    "config": vars(args),
                    "targets": {"mini_auc": 0.95, "tpr_at_fpr_1pct": 0.85}
                }, f)
                f.write("\n")
    
    tuner = OvernightTuner(args)
    tuner.run()


if __name__ == "__main__":
    main()
