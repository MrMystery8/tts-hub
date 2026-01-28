#!/usr/bin/env python3
"""
Watermark training live dashboard.

This serves a local, novice-friendly dashboard for a JSONL metrics log written
by training scripts (e.g. `quick_voice_smoke_train.py` or `train_full.py`).

Usage:
  .venv/bin/python -m watermark.scripts.live_dashboard --log outputs/run/metrics.jsonl

Then open:
  http://127.0.0.1:8765
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]


def _preferred_python_executable() -> str:
    """
    Prefer the repo-local venv python if present, otherwise fall back to the
    interpreter running the dashboard.
    """
    import sys

    venv_py = REPO_ROOT / ".venv" / "bin" / "python"
    if venv_py.exists():
        # IMPORTANT: do NOT resolve symlinks here.
        # On macOS, `.venv/bin/python` is often a symlink to the base interpreter.
        # If we resolve to the base interpreter path, Python won't find the venv's
        # `pyvenv.cfg`, and imports (e.g. torch) will fail.
        return str(venv_py)
    return str(Path(sys.executable))


def _read_first_meta(path: Path, max_lines: int = 200) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            for _ in range(max_lines):
                line = f.readline()
                if not line:
                    break
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict) and obj.get("type") == "meta":
                    return obj
    except Exception:
        return None
    return None


def _read_tail_jsonl(path: Path, max_lines: int = 800) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    # Read tail efficiently (works well for append-only JSONL logs).
    chunk_size = 64 * 1024
    data = b""
    lines: list[bytes] = []
    with path.open("rb") as f:
        f.seek(0, 2)
        pos = f.tell()
        while pos > 0 and len(lines) <= max_lines:
            read_size = min(chunk_size, pos)
            pos -= read_size
            f.seek(pos)
            data = f.read(read_size) + data
            lines = data.splitlines()

    out: list[dict[str, Any]] = []
    for raw in lines[-max_lines:]:
        try:
            obj = json.loads(raw.decode("utf-8"))
        except Exception:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def _make_single_log_app(log_path: Path):
    from fastapi import FastAPI, Query
    from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse

    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def index():
        return """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Watermark Training Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
    <style>
      :root {
        --bg: #0b1220;
        --panel: rgba(255,255,255,0.06);
        --panel2: rgba(255,255,255,0.08);
        --border: rgba(255,255,255,0.10);
        --text: rgba(255,255,255,0.92);
        --muted: rgba(255,255,255,0.65);
        --muted2: rgba(255,255,255,0.50);
        --good: #34d399;
        --warn: #fbbf24;
        --bad: #fb7185;
        --accent: #60a5fa;
        --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      }

      body {
        background: radial-gradient(1200px 800px at 20% 0%, rgba(96,165,250,0.22), transparent 55%),
                    radial-gradient(900px 700px at 80% 0%, rgba(217,70,239,0.18), transparent 55%),
                    var(--bg);
        color: var(--text);
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
        margin: 0;
      }

      .wrap { max-width: 1320px; margin: 0 auto; padding: 18px 16px 40px; }
      header { display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; margin-bottom: 14px; }
      h1 { font-size: 18px; margin: 0; letter-spacing: 0.2px; }
      .sub { font-size: 13px; color: var(--muted); margin-top: 2px; }

      .controls { display: flex; gap: 10px; flex-wrap: wrap; justify-content: flex-end; align-items: center; }
      .pill { background: rgba(255,255,255,0.08); border: 1px solid var(--border); color: var(--text); border-radius: 999px; padding: 8px 10px; font-size: 12px; }
      .pill input, .pill select { background: transparent; border: none; color: var(--text); outline: none; font-size: 12px; }
      .pill input[type="number"] { width: 90px; }
      a { color: var(--accent); text-decoration: none; }
      a:hover { text-decoration: underline; }

      .grid { display: grid; grid-template-columns: repeat(12, 1fr); gap: 12px; }
      .card { background: var(--panel); border: 1px solid var(--border); border-radius: 14px; padding: 12px; }
      .card h2 { margin: 0 0 8px; font-size: 13px; letter-spacing: 0.2px; color: rgba(255,255,255,0.92); }
      .card h3 { margin: 0 0 6px; font-size: 12px; color: rgba(255,255,255,0.88); }
      .tiny { font-size: 11px; color: var(--muted2); }
      .small { font-size: 12px; }
      .mono { font-family: var(--mono); }
      code { font-family: var(--mono); font-size: 12px; background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.10); padding: 2px 6px; border-radius: 10px; }
      pre { font-family: var(--mono); font-size: 12px; white-space: pre-wrap; background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.10); border-radius: 14px; padding: 10px; margin: 0; }

      .kvs { display: grid; grid-template-columns: 1fr 1fr; gap: 6px 12px; font-size: 12px; }
      .kv { display: flex; justify-content: space-between; gap: 10px; }
      .kv span:first-child { color: var(--muted); }
      .kv span:last-child { font-variant-numeric: tabular-nums; }

      .ok { color: var(--good); }
      .warn { color: var(--warn); }
      .bad { color: var(--bad); }
      .dot { display:inline-block; width: 8px; height: 8px; border-radius: 999px; margin-right: 6px; vertical-align: -1px; background: rgba(255,255,255,0.35); }
      .dot.ok { background: var(--good); }
      .dot.warn { background: var(--warn); }
      .dot.bad { background: var(--bad); }

      .badge { border: 1px solid var(--border); border-radius: 999px; padding: 3px 8px; font-size: 11px; color: var(--muted); background: rgba(255,255,255,0.04); }
      .row { display: flex; gap: 10px; flex-wrap: wrap; }

      .chartWrap { height: 280px; }
      canvas { width: 100%; height: 260px; }
      .split { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }

      table { width: 100%; border-collapse: collapse; font-size: 12px; }
      th, td { border-bottom: 1px solid rgba(255,255,255,0.10); padding: 8px 8px; text-align: left; vertical-align: top; }
      th { color: rgba(255,255,255,0.75); font-weight: 600; }
      tr:hover td { background: rgba(255,255,255,0.04); }
      ul { margin: 8px 0 0 18px; padding: 0; }
      li { margin: 6px 0; color: var(--muted); }

      .help { background: rgba(96,165,250,0.10); border: 1px solid rgba(96,165,250,0.20); border-radius: 14px; padding: 10px; }
      .matrix { overflow-x: auto; }
      .matrix table { width: auto; min-width: 100%; }
      .cell { text-align: center; font-variant-numeric: tabular-nums; border-radius: 8px; padding: 4px 6px; display: inline-block; min-width: 28px; }
      .foot { margin-top: 10px; color: var(--muted2); font-size: 11px; }
    </style>
  </head>
  <body>
    <div class="wrap">
      <header>
        <div>
          <h1>Watermark Training Dashboard</h1>
          <div class="sub">Log: <span class="mono" id="logPath">—</span></div>
          <div class="sub" id="serverInfo"></div>
          <div class="sub" id="clientInfo"></div>
        </div>
        <div class="controls">
          <div class="pill">
            Auto-refresh
            <input id="auto" type="checkbox" checked />
          </div>
          <div class="pill">
            Interval (ms)
            <input id="interval" type="number" value="1000" min="250" step="250" />
          </div>
          <div class="pill">
            Tail lines
            <input id="tail" type="number" value="5000" min="200" step="200" />
          </div>
          <div class="pill">
            <a href="/download/metrics" target="_blank" rel="noreferrer">Download metrics.jsonl</a>
          </div>
        </div>
      </header>

      <div class="grid">
        <div class="card" style="grid-column: span 6;">
          <h2>Run Overview</h2>
          <div class="row" id="runBadges"></div>
          <div style="height: 8px;"></div>
          <div class="kvs" id="runInfo"></div>
          <div class="foot">
            Key idea: “Detection/presence” metrics tell you if audio is watermarked; “Attribution/identity” metrics tell you which model/version produced it.
          </div>
        </div>

        <div class="card" style="grid-column: span 6;">
          <h2>Is Training On Track?</h2>
          <div class="help small">
            Each metric below compares the latest probe value to a target line.
            <span class="ok">Green</span> = meets target, <span class="warn">yellow</span> = close, <span class="bad">red</span> = off target.
          </div>
          <div style="height: 10px;"></div>
          <div class="kvs" id="healthKVs"></div>
        </div>

        <div class="card" style="grid-column: span 12;">
          <h2>Metric Guide (What To Look For)</h2>
          <div class="tiny">This table explains each headline metric, whether it should be above/below the target line, and why it matters.</div>
          <div style="height: 8px;"></div>
          <div style="overflow-x:auto;">
            <table>
              <thead>
                <tr><th>metric</th><th>goal</th><th>meaning</th></tr>
              </thead>
              <tbody id="metricGuide"></tbody>
            </table>
          </div>
        </div>

        <div class="card" style="grid-column: span 12;">
          <h2>Recommendations (Auto)</h2>
          <div class="tiny">Simple rule-based suggestions based on the latest probe values.</div>
          <div id="recs" class="small"></div>
        </div>

        <div class="card" style="grid-column: span 12;">
          <h2>Attribution & Calibration (Probe)</h2>
          <div class="split">
             <div class="chartWrap"><canvas id="chart_crc"></canvas></div>
             <div class="chartWrap"><canvas id="chart_bits"></canvas></div>
          </div>
        </div>

        <div class="card" style="grid-column: span 12;">
          <h2>Detection AUC (Presence)</h2>
          <div class="tiny">Goal: AUC lines above target (random baseline = 0.5). Reverb AUC measures robustness.</div>
          <div class="chartWrap"><canvas id="chart_auc"></canvas></div>
        </div>

        <div class="card" style="grid-column: span 12;">
          <h2>Operating Point: TPR @ 1% FPR</h2>
          <div class="tiny">Goal: high TPR while allowing only 1% false positives. This is stricter than AUC.</div>
          <div class="chartWrap"><canvas id="chart_tpr"></canvas></div>
        </div>

        <div class="card" style="grid-column: span 12;">
          <h2>Detection Separation (Score Summaries)</h2>
          <div class="tiny">Goal: positives higher than negatives. Displays means and p10/p50/p90 when available.</div>
          <div class="split">
            <div>
              <h3>Clean vs Watermarked</h3>
              <div class="chartWrap"><canvas id="chart_sep_clean"></canvas></div>
            </div>
            <div>
              <h3>Reverb Attack</h3>
              <div class="chartWrap"><canvas id="chart_sep_reverb"></canvas></div>
            </div>
          </div>
        </div>

        <div class="card" style="grid-column: span 12;">
          <h2>Detection Calibration (Localization Pooling)</h2>
          <div class="tiny">Goal: localization-pooled watermark score high on watermarked clips, low on clean clips.</div>
          <div class="chartWrap"><canvas id="chart_preamble"></canvas></div>
        </div>

        <div class="card" style="grid-column: span 12;">
          <h2>Attribution (Multiclass)</h2>
          <div class="tiny">Goal: overall accuracy and watermarked-only accuracy rise above chance.</div>
          <div class="chartWrap"><canvas id="chart_id"></canvas></div>
        </div>

        <div class="card" style="grid-column: span 12;">
          <h2>Training Losses</h2>
          <div class="tiny">Goal: losses trend down; compare trends within each chart (loss scales differ).</div>
          <div class="split">
            <div>
              <h3>Stage 1 (Detection)</h3>
              <div class="chartWrap"><canvas id="chart_s1"></canvas></div>
            </div>
            <div>
              <h3>Stage 1B (Payload / Decoder)</h3>
              <div class="chartWrap"><canvas id="chart_s1b"></canvas></div>
            </div>
          </div>
          <div style="height: 10px;"></div>
          <h3>Stage 2 (Encoder)</h3>
          <div class="chartWrap"><canvas id="chart_s2"></canvas></div>
          <div style="height: 10px;"></div>
          <h3>Energy Budget</h3>
          <div class="chartWrap"><canvas id="chart_budget"></canvas></div>
        </div>

        <div class="card" style="grid-column: span 12;">
          <h2>Attribution Confusion Matrix (Probe)</h2>
          <div class="tiny">Diagonal dominance means correct attribution.</div>
          <div class="row" style="margin-top:8px; gap:10px; align-items:center;">
            <div class="pill">
              Matrix
              <select id="matrixMode">
                <option value="attr" selected>Attribution-only (K×K, watermarked-only)</option>
                <option value="full">Thresholded full (K+1×K+1, includes clean)</option>
              </select>
            </div>
          </div>
          <div class="split">
            <div>
              <h3>Classes</h3>
              <div class="matrix" id="matrix_model"></div>
            </div>
            <div>
              <h3>Legacy</h3>
              <div class="matrix" id="matrix_version"></div>
            </div>
          </div>
        </div>

        <div class="card" style="grid-column: span 12;">
          <h2>Recent Events</h2>
          <div class="tiny">Last ~50 events (epoch + probe) from the log.</div>
          <div style="height: 8px;"></div>
          <div style="overflow-x:auto;">
            <table>
              <thead>
                <tr>
                  <th>time</th><th>type</th><th>stage</th><th>epoch</th>
                  <th>loss</th><th>mini_auc</th><th>auc_reverb</th><th>tpr@1%fpr</th><th>probe_wm_acc</th><th>attr_acc</th><th>wm_acc</th>
                </tr>
              </thead>
              <tbody id="eventsTable"></tbody>
            </table>
          </div>
        </div>

        <div class="card" style="grid-column: span 12;">
          <h2>Decode Report (Latest)</h2>
          <div class="tiny">End-of-run snapshot saved by the training script (if present next to the metrics log).</div>
          <div style="height: 8px;"></div>
          <pre id="decodeReport">Loading…</pre>
        </div>
      </div>
    </div>

    <script>
      const DEFAULT_TARGETS = {
        mini_auc: 0.95,
        mini_auc_reverb: 0.85,
        tpr_at_fpr_1pct: 0.90,
        tpr_at_fpr_1pct_reverb: 0.70,
        attr_acc: 0.70,
        wm_acc: 0.70,
        attr_acc_reverb: 0.60,
        wm_acc_reverb: 0.60,
        id_acc_pos: 0.40,
        wm_prob_loc_pos_mean: 0.70,
        wm_prob_loc_neg_mean: 0.30,
        p_clean_pos_mean: 0.30, // legacy calibration signal (derived)
        p_clean_neg_mean: 0.90, // legacy calibration signal (derived)
        loss_budget: 0.05,
      };

      const DIRECTIONS = {
        mini_auc: "gte",
        mini_auc_reverb: "gte",
        tpr_at_fpr_1pct: "gte",
        tpr_at_fpr_1pct_reverb: "gte",
        attr_acc: "gte",
        wm_acc: "gte",
        attr_acc_reverb: "gte",
        wm_acc_reverb: "gte",
        id_acc_pos: "gte",
        wm_prob_loc_pos_mean: "gte",
        wm_prob_loc_neg_mean: "lte",
        p_clean_pos_mean: "lte",
        p_clean_neg_mean: "gte",
        loss_budget: "lte",
      };

      const EXPLAIN = {
        mini_auc: "AUC clean vs watermarked (0.5=random, 1.0=perfect). Higher is better.",
        mini_auc_reverb: "AUC under reverb. Higher is better.",
        tpr_at_fpr_1pct: "TPR when we allow only 1% false positives (strict). Higher is better.",
        tpr_at_fpr_1pct_reverb: "Same strict operating point under reverb. Higher is better.",
        attr_acc: "Overall multiclass accuracy across probe items (clean + model classes).",
        wm_acc: "Accuracy on watermarked-only probe items (model attribution).",
        attr_acc_reverb: "Overall accuracy under reverb attack.",
        wm_acc_reverb: "Watermarked-only attribution under reverb attack.",
        id_acc_pos: "ID-head accuracy on watermarked probe items (ignores detection threshold).",
        wm_prob_loc_pos_mean: "Mean localization-pooled watermark score on watermarked probe items. Higher is better.",
        wm_prob_loc_neg_mean: "Mean localization-pooled watermark score on clean probe items. Lower is better.",
        p_clean_pos_mean: "Mean P(clean) on watermarked probe items (derived from class_probs). Lower is better.",
        p_clean_neg_mean: "Mean P(clean) on clean probe items (derived from class_probs). Higher is better.",
        loss_budget: "Penalty for exceeding energy budget (-30dB). Should be 0.0.",
      };

      function fmt(x) {
        if (x === null || x === undefined) return "—";
        if (typeof x === "number") return x.toFixed(4).replace(/0+$/,"").replace(/\\.$/,"");
        return String(x);
      }

      function timeAgo(ts) {
        if (!ts) return "—";
        const s = Math.max(0, (Date.now()/1000) - ts);
        if (s < 60) return `${s.toFixed(0)}s ago`;
        if (s < 3600) return `${(s/60).toFixed(0)}m ago`;
        return `${(s/3600).toFixed(1)}h ago`;
      }

      function kvRow(k, v, cls) {
        return `<div class="kv"><span>${k}</span><span class="${cls||""}">${v}</span></div>`;
      }

      function dot(cls) {
        const c = cls || "";
        return `<span class="dot ${c}"></span>`;
      }

      function metricStatus(key, value, targets) {
        if (value === null || value === undefined || typeof value !== "number") return {cls:"", label:"—"};
        const t = (targets && targets[key] !== undefined) ? targets[key] : DEFAULT_TARGETS[key];
        if (t === undefined) return {cls:"", label: fmt(value)};
        const dir = DIRECTIONS[key] || "gte";
        const eps = 1e-12;
        if (dir === "lte") {
          if (value <= t + eps) return {cls:"ok", label: fmt(value)};
          if (value <= t * 1.10 + eps) return {cls:"warn", label: fmt(value)};
          return {cls:"bad", label: fmt(value)};
        }
        if (value >= t - eps) return {cls:"ok", label: fmt(value)};
        if (value >= t * 0.90 - eps) return {cls:"warn", label: fmt(value)};
        return {cls:"bad", label: fmt(value)};
      }

      function maybeNumber(x) { return (typeof x === "number") ? x : null; }

      function makeMatrix(container, mat, rowLabelPrefix, colLabelPrefix) {
        if (!mat || !Array.isArray(mat) || !mat.length) {
          container.innerHTML = `<div class="tiny">No confusion matrix available yet (needs classification heads + a probe run).</div>`;
          return;
        }
        const rows = mat.length;
        const cols = mat[0].length;
        let maxVal = 0;
        for (const r of mat) for (const v of r) maxVal = Math.max(maxVal, (typeof v === "number") ? v : 0);
        const headerCols = Array.from({length: cols}, (_,j) => {
          const isUnknown = (cols === rows + 1) && (j === cols - 1);
          const name = isUnknown ? "pred unk" : (colLabelPrefix + String(j));
          return `<th>${name}</th>`;
        }).join("");
        const header = `<tr><th></th>${headerCols}</tr>`;
        const body = mat.map((r,i) => {
          const tds = r.map((v) => {
            const x = (typeof v === "number") ? v : 0;
            const a = maxVal > 0 ? (x / maxVal) : 0;
            const bg = `rgba(96,165,250,${0.08 + 0.55*a})`;
            return `<td><span class="cell" style="background:${bg}; border:1px solid rgba(255,255,255,0.10);">${x}</span></td>`;
          }).join("");
          return `<tr><th>${rowLabelPrefix}${i}</th>${tds}</tr>`;
        }).join("");
        container.innerHTML = `<table>${header}${body}</table>`;
      }

      let charts = {};
      const MATRIX_MODE_KEY = "wmDash:matrixMode";

      function initMatrixMode() {
        const el = document.getElementById("matrixMode");
        if (!el || el._wmInit) return;
        el._wmInit = true;
        const saved = localStorage.getItem(MATRIX_MODE_KEY);
        if (saved) el.value = saved;
        el.addEventListener("change", () => localStorage.setItem(MATRIX_MODE_KEY, el.value));
      }

      function getMatrixMode() {
        const el = document.getElementById("matrixMode");
        if (!el) return localStorage.getItem(MATRIX_MODE_KEY) || "attr";
        return el.value || "attr";
      }

      function ensureCharts() {
        if (!window.Chart) return false;
        if (charts._ready) return true;

        Chart.defaults.color = "rgba(255,255,255,0.80)";
        Chart.defaults.borderColor = "rgba(255,255,255,0.12)";
        Chart.defaults.font.family = 'ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial';
        Chart.defaults.plugins.legend.labels.boxWidth = 10;
        Chart.defaults.plugins.legend.labels.usePointStyle = true;
        Chart.defaults.plugins.tooltip.backgroundColor = "rgba(17,24,39,0.92)";
        Chart.defaults.plugins.tooltip.borderColor = "rgba(255,255,255,0.14)";
        Chart.defaults.plugins.tooltip.borderWidth = 1;

        function lineChart(id, yMin=null, yMax=null) {
          const ctx = document.getElementById(id).getContext("2d");
          return new Chart(ctx, {
            type: "line",
            data: { labels: [], datasets: [] },
            options: {
              responsive: true,
              maintainAspectRatio: false,
              interaction: { mode: "nearest", intersect: false },
              plugins: { legend: { display: true } },
              elements: { point: { radius: 0 }, line: { tension: 0.25 } },
              scales: {
                y: { beginAtZero: false, suggestedMin: yMin, suggestedMax: yMax },
                x: { ticks: { maxRotation: 0, autoSkip: true, maxTicksLimit: 14 } }
              }
            }
          });
        }

        charts.auc = lineChart("chart_auc", 0, 1);
        charts.tpr = lineChart("chart_tpr", 0, 1);
        charts.sepClean = lineChart("chart_sep_clean", 0, 1);
        charts.sepReverb = lineChart("chart_sep_reverb", 0, 1);
        charts.preamble = lineChart("chart_preamble", 0, 1);
        charts.id = lineChart("chart_id", 0, 1);
        charts.crc = lineChart("chart_crc", 0, 1);
        charts.bits = lineChart("chart_bits", 0.7, 1.0);
        charts.s1 = lineChart("chart_s1");
        charts.s1b = lineChart("chart_s1b");
        charts.s2 = lineChart("chart_s2");
        charts.budget = lineChart("chart_budget", 0, 1);

        charts._ready = true;
        return true;
      }

      function setDatasets(chart, labels, datasets) {
        chart.data.labels = labels;
        chart.data.datasets = datasets;
        chart.update("none");
      }

      function dashed(label, color, values) {
        return { label, data: values, borderColor: color, borderDash: [6,6], fill: false, pointRadius: 0 };
      }

      function solid(label, color, values) {
        return { label, data: values, borderColor: color, backgroundColor: color, fill: false, pointRadius: 0 };
      }

      async function refresh() {
        initMatrixMode();
        const tail = parseInt(document.getElementById("tail").value || "5000", 10);
        const startedAt = Date.now();
        let events = [];
        let server = null;
        try {
          const [resMetrics, resServer] = await Promise.all([
            fetch(`/api/metrics?tail=${encodeURIComponent(tail)}`, { cache: "no-store" }),
            fetch(`/api/server`, { cache: "no-store" }),
          ]);
          server = await resServer.json();
          events = await resMetrics.json();
        } catch (e) {
          // Keep the dashboard alive even if one refresh fails.
          const msg = (e && e.message) ? e.message : String(e);
          document.getElementById("serverInfo").textContent = `Error fetching updates: ${msg}`;
          return;
        } finally {
          const elapsed = Date.now() - startedAt;
          document.getElementById("clientInfo").textContent = `Refresh: ${elapsed}ms · tail=${tail}`;
        }

        const meta = events.find(e => e.type === "meta") || null;
        const targets = (meta && meta.targets) ? meta.targets : DEFAULT_TARGETS;
        const epochs = events.filter(e => e.type === "epoch");
        const probes = events.filter(e => e.type === "probe");
        const latestEpoch = epochs.length ? epochs[epochs.length - 1] : null;
        const latestProbe = probes.length ? probes[probes.length - 1] : null;

        // Header + file status
        document.getElementById("logPath").textContent = (meta && meta.metrics_path) ? meta.metrics_path : "metrics.jsonl";
        const parts = [];
        if (meta && meta.ts) parts.push(`Started: ${new Date(meta.ts*1000).toLocaleString()}`);
        if (server && server.exists) {
          parts.push(`File: ${server.bytes} bytes`);
          parts.push(`Modified: ${new Date(server.mtime*1000).toLocaleTimeString()} (${timeAgo(server.mtime)})`);
        } else if (server && server.exists === false) {
          parts.push(`File not found (check the --log path)`);
        }
        if (events && events.length) {
          const lastTs = events[events.length - 1].ts;
          if (lastTs) parts.push(`Last event: ${new Date(lastTs*1000).toLocaleTimeString()} (${timeAgo(lastTs)})`);
          parts.push(`Events loaded: ${events.length}`);
        }
        document.getElementById("serverInfo").textContent = parts.join(" · ");

        // Badges
        const badges = [];
        if (meta && meta.run_name) badges.push(`<span class="badge">run: <b>${meta.run_name}</b></span>`);
        if (meta && meta.device) badges.push(`<span class="badge">device: <b>${meta.device}</b></span>`);
        if (latestEpoch) badges.push(`<span class="badge">stage: <b>${latestEpoch.stage}</b></span>`);
        if (latestEpoch) badges.push(`<span class="badge">epoch: <b>${latestEpoch.epoch}</b></span>`);
        if (latestEpoch && latestEpoch.ts) badges.push(`<span class="badge">updated: <b>${timeAgo(latestEpoch.ts)}</b></span>`);
        document.getElementById("runBadges").innerHTML = badges.join("");

        // Run overview KVs
        const info = [];
        if (meta) {
          if (meta.out_dir) info.push(kvRow("out_dir", `<code>${meta.out_dir}</code>`));
          if (meta.output_dir) info.push(kvRow("output_dir", `<code>${meta.output_dir}</code>`));
          if (meta.manifest) info.push(kvRow("manifest", `<code>${meta.manifest}</code>`));
          if (meta.config && meta.config.num_clips !== undefined) info.push(kvRow("num_clips", fmt(meta.config.num_clips)));
          if (meta.config && meta.config.probe_n !== undefined) info.push(kvRow("probe_n", fmt(meta.config.probe_n)));
          if (meta.config && meta.config.reverb_prob !== undefined) info.push(kvRow("reverb_prob", fmt(meta.config.reverb_prob)));
          if (meta.config && meta.config.model_ce_weight !== undefined) info.push(kvRow("model_ce_weight", fmt(meta.config.model_ce_weight)));
          if (meta.config && meta.config.version_ce_weight !== undefined) info.push(kvRow("version_ce_weight", fmt(meta.config.version_ce_weight)));
        }
        if (latestEpoch) {
          info.push(kvRow("latest_loss", fmt(latestEpoch.loss)));
          if (latestEpoch.lr !== undefined) info.push(kvRow("lr", fmt(latestEpoch.lr)));
        }
        document.getElementById("runInfo").innerHTML = info.join("");

        // Health panel
        const hk = [];
        const keys = [
          "mini_auc",
          "mini_auc_reverb",
          "tpr_at_fpr_1pct",
          "tpr_at_fpr_1pct_reverb",
          "wm_prob_loc_pos_mean",
          "wm_prob_loc_neg_mean",
          "attr_acc",
          "wm_acc",
          "id_acc_pos",
          "attr_acc_reverb",
          "wm_acc_reverb",
          "p_clean_pos_mean",
          "p_clean_neg_mean",
        ];
        if (latestProbe) {
          for (const k of keys) {
            const v = latestProbe[k];
            if (v === undefined) continue;
            const st = metricStatus(k, v, targets);
            const target = (targets && targets[k] !== undefined) ? targets[k] : DEFAULT_TARGETS[k];
            const dir = DIRECTIONS[k] || "gte";
            const goalTxt = (target !== undefined) ? `${dir === "lte" ? "≤" : "≥"} ${fmt(target)}` : "—";
            const expl = EXPLAIN[k] || "";
            hk.push(`<div class="kv"><span title="${expl}">${k} <span class="tiny">(goal ${goalTxt})</span></span><span class="${st.cls}">${dot(st.cls)}${st.label}</span></div>`);
          }
        } else {
          hk.push(`<div class="tiny">Waiting for first probe event… (probes usually start in Stage 2)</div>`);
        }
        document.getElementById("healthKVs").innerHTML = hk.join("");

        // Metric guide table
        const guideRows = [];
        for (const k of keys) {
          const target = (targets && targets[k] !== undefined) ? targets[k] : DEFAULT_TARGETS[k];
          const dir = DIRECTIONS[k] || "gte";
          const goal = (target !== undefined) ? `${dir === "lte" ? "≤" : "≥"} ${fmt(target)}` : "—";
          const meaning = EXPLAIN[k] || "";
          guideRows.push(`<tr><td><code>${k}</code></td><td class="mono">${goal}</td><td>${meaning}</td></tr>`);
        }
        guideRows.push(`<tr><td><code>chance baselines</code></td><td class="mono">overall≈1/(K+1) · watermarked≈1/K</td><td>Use these to check if attribution is better than random guessing (K = number of model classes).</td></tr>`);
        document.getElementById("metricGuide").innerHTML = guideRows.join("");

        // Recommendations (rule-based)
        const recs = [];
        function addRec(title, body) {
          recs.push(`<li><b>${title}:</b> ${body}</li>`);
        }
        if (latestProbe) {
          const stAuc = metricStatus("mini_auc", latestProbe.mini_auc, targets).cls;
          const stAucR = metricStatus("mini_auc_reverb", latestProbe.mini_auc_reverb, targets).cls;
          const stTpr = metricStatus("tpr_at_fpr_1pct", latestProbe.tpr_at_fpr_1pct, targets).cls;
          const stTprR = metricStatus("tpr_at_fpr_1pct_reverb", latestProbe.tpr_at_fpr_1pct_reverb, targets).cls;
          const stAttr = metricStatus("wm_acc", latestProbe.wm_acc, targets).cls;
          const stLocNeg = metricStatus("wm_prob_loc_neg_mean", latestProbe.wm_prob_loc_neg_mean, targets).cls;
          const stLocPos = metricStatus("wm_prob_loc_pos_mean", latestProbe.wm_prob_loc_pos_mean, targets).cls;

          if (stAuc === "bad") {
            addRec("Detection AUC is low", "Increase Stage 1/2 training, confirm mixed pos/neg is correct, and verify the probe set contains both classes.");
          }
          if (stAucR === "bad") {
            addRec("Reverb robustness is low", "Reduce `reverb_prob` early (schedule it up later), increase Stage 2 epochs, or add post-Stage2 decoder fine-tune (`--epochs_s1b_post`).");
          }
          if (stTpr === "bad") {
            addRec("TPR@1%FPR is low", "AUC might be high but confidence is low. Consider more Stage 1 training and/or longer Stage 2 (encoder) while monitoring quality.");
          }
          if (stTprR === "bad") {
            addRec("TPR@1%FPR under reverb is low", "The watermark is fragile at strict thresholds. Try lowering `reverb_prob` early then ramping it up, and/or increase Stage 2 epochs.");
          }
          if (stLocNeg === "bad") {
            addRec("Clean audio looks watermarked", "If the localization-pooled watermark score is high on clean clips, increase clean regularization (Stage 3 `--neg_weight`) and ensure you have enough clean negatives.");
          }
          if (stLocPos === "bad") {
            addRec("Watermarked audio looks clean", "If the localization-pooled watermark score is low on watermarked clips, the encoder watermark is too weak/undertrained. Increase Stage 2 epochs and/or `detect_weight`.");
          }
          if (stAttr === "bad") {
            addRec("Attribution is low", "If detection is good but wm_acc is near chance, increase Stage 2 (encoder) epochs and consider Stage 3 finetuning to align decision boundaries.");
          }
        }
        document.getElementById("recs").innerHTML = recs.length ? `<ul>${recs.join("")}</ul>` : `<div class="tiny">No recommendations right now (either no probe yet, or metrics look on-track).</div>`;

        // Charts
        if (!ensureCharts()) return;
        const probeLabels = probes.map((p, idx) => `${idx+1}:${p.stage || "?"}/e${p.epoch || "?"}`);

        // Detection AUC
        const aucVals = probes.map(p => maybeNumber(p.mini_auc));
        const aucRVals = probes.map(p => maybeNumber(p.mini_auc_reverb));
        const aucTarget = probeLabels.map(_ => (targets.mini_auc !== undefined ? targets.mini_auc : DEFAULT_TARGETS.mini_auc));
        const aucRTarget = probeLabels.map(_ => (targets.mini_auc_reverb !== undefined ? targets.mini_auc_reverb : DEFAULT_TARGETS.mini_auc_reverb));
        const aucBase = probeLabels.map(_ => 0.5);
        setDatasets(charts.auc, probeLabels, [
          solid("mini_auc", "#60a5fa", aucVals),
          solid("mini_auc_reverb", "#c084fc", aucRVals),
          dashed("target mini_auc", "rgba(96,165,250,0.65)", aucTarget),
          dashed("target reverb", "rgba(192,132,252,0.65)", aucRTarget),
          dashed("random=0.5", "rgba(255,255,255,0.35)", aucBase),
        ]);

        // TPR @ 1% FPR
        const tprVals = probes.map(p => maybeNumber(p.tpr_at_fpr_1pct));
        const tprRVals = probes.map(p => maybeNumber(p.tpr_at_fpr_1pct_reverb));
        const tprTarget = probeLabels.map(_ => (targets.tpr_at_fpr_1pct !== undefined ? targets.tpr_at_fpr_1pct : DEFAULT_TARGETS.tpr_at_fpr_1pct));
        const tprRTarget = probeLabels.map(_ => (targets.tpr_at_fpr_1pct_reverb !== undefined ? targets.tpr_at_fpr_1pct_reverb : DEFAULT_TARGETS.tpr_at_fpr_1pct_reverb));
        setDatasets(charts.tpr, probeLabels, [
          solid("tpr_at_fpr_1pct", "#34d399", tprVals),
          solid("tpr_at_fpr_1pct_reverb", "#fb7185", tprRVals),
          dashed("target tpr", "rgba(52,211,153,0.65)", tprTarget),
          dashed("target reverb tpr", "rgba(251,113,133,0.65)", tprRTarget),
        ]);

        // Separation charts
        function bandDatasets(prefix) {
          const posMean = probes.map(p => maybeNumber(p[`${prefix}_pos_mean`]));
          const negMean = probes.map(p => maybeNumber(p[`${prefix}_neg_mean`]));
          const posP10 = probes.map(p => maybeNumber(p[`${prefix}_pos_p10`]));
          const posP50 = probes.map(p => maybeNumber(p[`${prefix}_pos_p50`]));
          const posP90 = probes.map(p => maybeNumber(p[`${prefix}_pos_p90`]));
          const negP10 = probes.map(p => maybeNumber(p[`${prefix}_neg_p10`]));
          const negP50 = probes.map(p => maybeNumber(p[`${prefix}_neg_p50`]));
          const negP90 = probes.map(p => maybeNumber(p[`${prefix}_neg_p90`]));
          return [
            solid("pos_mean", "#34d399", posMean),
            solid("neg_mean", "#fb7185", negMean),
            dashed("pos_p10", "rgba(52,211,153,0.40)", posP10),
            dashed("pos_p50", "rgba(52,211,153,0.65)", posP50),
            dashed("pos_p90", "rgba(52,211,153,0.40)", posP90),
            dashed("neg_p10", "rgba(251,113,133,0.40)", negP10),
            dashed("neg_p50", "rgba(251,113,133,0.65)", negP50),
            dashed("neg_p90", "rgba(251,113,133,0.40)", negP90),
          ];
        }
        setDatasets(charts.sepClean, probeLabels, bandDatasets("detect"));
        setDatasets(charts.sepReverb, probeLabels, bandDatasets("reverb"));

        // Localization-pooled calibration
        const locPos = probes.map(p => maybeNumber(p.wm_prob_loc_pos_mean));
        const locNeg = probes.map(p => maybeNumber(p.wm_prob_loc_neg_mean));
        const locPosT = probeLabels.map(_ => (targets.wm_prob_loc_pos_mean !== undefined ? targets.wm_prob_loc_pos_mean : DEFAULT_TARGETS.wm_prob_loc_pos_mean));
        const locNegT = probeLabels.map(_ => (targets.wm_prob_loc_neg_mean !== undefined ? targets.wm_prob_loc_neg_mean : DEFAULT_TARGETS.wm_prob_loc_neg_mean));
        setDatasets(charts.preamble, probeLabels, [
          solid("wm_prob_loc_pos_mean", "#34d399", locPos),
          solid("wm_prob_loc_neg_mean", "#fb7185", locNeg),
          dashed("target pos", "rgba(52,211,153,0.65)", locPosT),
          dashed("target neg", "rgba(251,113,133,0.65)", locNegT),
        ]);

        // Attribution accuracy (probe)
        const attrAcc = probes.map(p => maybeNumber(p.attr_acc));
        const wmAcc = probes.map(p => maybeNumber(p.wm_acc));
        const idAccPos = probes.map(p => maybeNumber(p.id_acc_pos));
        const attrAccR = probes.map(p => maybeNumber(p.attr_acc_reverb));
        const wmAccR = probes.map(p => maybeNumber(p.wm_acc_reverb));
        const idAccPosR = probes.map(p => maybeNumber(p.id_acc_pos_reverb));
        const attrT = probeLabels.map(_ => (targets.attr_acc !== undefined ? targets.attr_acc : DEFAULT_TARGETS.attr_acc));
        const wmT = probeLabels.map(_ => (targets.wm_acc !== undefined ? targets.wm_acc : DEFAULT_TARGETS.wm_acc));
        setDatasets(charts.id, probeLabels, [
          solid("attr_acc", "#60a5fa", attrAcc),
          solid("wm_acc", "#34d399", wmAcc),
          solid("id_acc_pos", "#fbbf24", idAccPos),
          solid("attr_acc_reverb", "#c084fc", attrAccR),
          solid("wm_acc_reverb", "#fb7185", wmAccR),
          solid("id_acc_pos_reverb", "#94a3b8", idAccPosR),
          dashed("target attr", "rgba(96,165,250,0.65)", attrT),
          dashed("target wm", "rgba(52,211,153,0.65)", wmT),
        ]);

        // Extra charts (repurpose old CRC/bits slots)
        const predClean = probes.map(p => maybeNumber(p.pred_clean_rate));
        const pPos = probes.map(p => maybeNumber(p.p_clean_pos_mean));
        const pNeg = probes.map(p => maybeNumber(p.p_clean_neg_mean));
        setDatasets(charts.crc, probeLabels, [
             solid("pred_clean_rate", "#94a3b8", predClean),
             solid("p_clean_pos_mean", "#fbbf24", pPos),
             solid("p_clean_neg_mean", "#60a5fa", pNeg),
        ]);

        setDatasets(charts.bits, probeLabels, [
             solid("attr_acc", "#60a5fa", attrAcc),
             solid("wm_acc", "#34d399", wmAcc),
        ]);

        // Loss charts
        const s1 = epochs.filter(e => e.stage === "s1");
        setDatasets(charts.s1, s1.map(e => `e${e.epoch}`), [
          solid("loss", "#cbd5e1", s1.map(e => maybeNumber(e.loss))),
          solid("loss_detect", "#60a5fa", s1.map(e => maybeNumber(e.loss_detect))),
          solid("loss_id", "#c084fc", s1.map(e => maybeNumber(e.loss_id))),
        ]);

        const s1b = epochs.filter(e => e.stage === "s1b" || e.stage === "s1b_post");
        setDatasets(charts.s1b, s1b.map(e => `${e.stage}/e${e.epoch}`), [
          solid("loss", "#cbd5e1", s1b.map(e => maybeNumber(e.loss))),
        ]);

        const s2 = epochs.filter(e => e.stage === "s2_encoder" || e.stage === "s3_finetune");
        setDatasets(charts.s2, s2.map(e => `${e.stage}/e${e.epoch}`), [
          solid("loss", "#cbd5e1", s2.map(e => maybeNumber(e.loss))),
          solid("loss_detect", "#60a5fa", s2.map(e => maybeNumber(e.loss_detect))),
          solid("loss_id", "#34d399", s2.map(e => maybeNumber(e.loss_id))),
          solid("loss_clean_detect", "#94a3b8", s2.map(e => maybeNumber(e.loss_clean_detect))),
          solid("loss_qual", "rgba(255,255,255,0.70)", s2.map(e => maybeNumber(e.loss_qual))),
          solid("loss_budget", "#ef4444", s2.map(e => maybeNumber(e.loss_budget))),
        ]);

        const s2bud = epochs.filter(e => e.stage === "s2" || e.stage === "s2_encoder" || e.stage === "s3_finetune");
        setDatasets(charts.budget, s2bud.map(e => `e${e.epoch}`), [
             solid("loss_budget", "#ef4444", s2bud.map(e => maybeNumber(e.loss_budget))),
             dashed("target (0.0)", "rgba(255,255,255,0.3)", s2bud.map(_ => 0.0))
        ]);

        // Confusion matrix (toggle: attribution-only vs full)
        const mm = getMatrixMode();
        const cm = latestProbe ? ((mm === "full") ? latestProbe.confusion : (latestProbe.confusion_attr || latestProbe.confusion)) : null;
        const isAttrOnly = (mm !== "full") && latestProbe && latestProbe.confusion_attr;
        makeMatrix(
          document.getElementById("matrix_model"),
          cm,
          isAttrOnly ? "true_id " : "true ",
          isAttrOnly ? "pred_id " : "pred "
        );
        makeMatrix(document.getElementById("matrix_version"), null, "true ", "pred ");

        // Recent events table
        const rows = [];
        const recent = events.slice(-50);
        for (const e of recent.reverse()) {
          const dt = e.ts ? new Date(e.ts*1000).toLocaleTimeString() : "—";
          const tpr = (e.type === "probe") ? (e.tpr_at_fpr_1pct !== undefined ? e.tpr_at_fpr_1pct : e.tpr_at_fpr_1pct_reverb) : null;
          const wmAccEv = (e.type === "probe") ? e.wm_acc : null;
          rows.push(`<tr>
            <td class="mono">${dt}</td>
            <td>${e.type || ""}</td>
            <td>${e.stage || ""}</td>
            <td>${e.epoch !== undefined ? e.epoch : ""}</td>
            <td class="mono">${e.loss !== undefined ? fmt(e.loss) : ""}</td>
            <td class="mono">${e.mini_auc !== undefined ? fmt(e.mini_auc) : ""}</td>
            <td class="mono">${e.mini_auc_reverb !== undefined ? fmt(e.mini_auc_reverb) : ""}</td>
            <td class="mono">${tpr !== null && tpr !== undefined ? fmt(tpr) : ""}</td>
            <td class="mono">${wmAccEv !== null && wmAccEv !== undefined ? fmt(wmAccEv) : ""}</td>
            <td class="mono">${e.attr_acc !== undefined ? fmt(e.attr_acc) : ""}</td>
            <td class="mono">${e.wm_acc !== undefined ? fmt(e.wm_acc) : ""}</td>
          </tr>`);
        }
        document.getElementById("eventsTable").innerHTML = rows.join("");

        // Decode report (best-effort)
        try {
          const r = await fetch("/api/decode_report");
          const text = await r.text();
          document.getElementById("decodeReport").textContent = text.trim() ? text : "No decode_report.txt found next to this metrics log.";
        } catch {
          document.getElementById("decodeReport").textContent = "Could not load decode_report.txt";
        }
      }

      let timer = null;
      function schedule() {
        if (timer) clearInterval(timer);
        const ms = parseInt(document.getElementById("interval").value || "1000", 10);
        if (document.getElementById("auto").checked) {
          timer = setInterval(refresh, Math.max(250, ms));
        }
      }

      document.getElementById("auto").addEventListener("change", schedule);
      document.getElementById("interval").addEventListener("change", schedule);
      document.getElementById("tail").addEventListener("change", refresh);

      refresh();
      schedule();
    </script>
  </body>
</html>
        """

    @app.get("/api/metrics")
    def metrics(tail: int = Query(5000, ge=200, le=200000)):
        events = _read_tail_jsonl(log_path, max_lines=tail)
        if not any(e.get("type") == "meta" for e in events):
            meta = _read_first_meta(log_path)
            if meta is not None:
                events = [meta, *events]
        return events

    @app.get("/download/metrics")
    def download_metrics():
        if not log_path.exists():
            return PlainTextResponse("metrics.jsonl not found", status_code=404)
        return FileResponse(log_path, filename=log_path.name)

    @app.get("/api/decode_report")
    def decode_report():
        report = log_path.parent / "audio" / "decode_report.txt"
        if not report.exists():
            return PlainTextResponse("", status_code=200)
        try:
            return PlainTextResponse(report.read_text(encoding="utf-8"), status_code=200)
        except Exception:
            return PlainTextResponse("", status_code=200)

    @app.get("/api/server")
    def server_info():
        try:
            st = log_path.stat()
            return {
                "path": str(log_path),
                "exists": True,
                "bytes": int(st.st_size),
                "mtime": float(st.st_mtime),
                "now": time.time(),
            }
        except Exception:
            return {"path": str(log_path), "exists": False, "now": time.time()}

    return app


def _tail_text(path: Path, *, max_lines: int = 400) -> str:
    if not path.exists():
        return ""
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    lines = raw.splitlines()
    return "\n".join(lines[-max_lines:])


def _is_pid_alive(pid: int) -> bool:
    import os

    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except Exception:
        return False
    return True


def _json_dump(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _json_load(path: Path) -> Optional[dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _median(xs: list[float]) -> Optional[float]:
    xs = [float(x) for x in xs if x is not None]
    if not xs:
        return None
    xs.sort()
    mid = len(xs) // 2
    if len(xs) % 2 == 1:
        return xs[mid]
    return 0.5 * (xs[mid - 1] + xs[mid])


def _summarize_metrics(events: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Extract a compact summary from a metrics.jsonl tail.
    """
    meta = None
    for e in events:
        if e.get("type") == "meta":
            meta = e
            break
    epochs = [e for e in events if e.get("type") == "epoch"]
    probes = [e for e in events if e.get("type") == "probe"]
    latest_epoch = epochs[-1] if epochs else None
    latest_probe = probes[-1] if probes else None

    best_probe = None
    best_val = None
    best_key_candidates = ["wm_acc", "attr_acc", "tpr_at_fpr_1pct", "mini_auc"]
    best_key = None
    for k in best_key_candidates:
        if any(isinstance(p.get(k), (int, float)) for p in probes):
            best_key = k
            break

    if best_key is not None:
        for p in probes:
            v = p.get(best_key)
            if isinstance(v, (int, float)):
                if best_val is None or float(v) > float(best_val):
                    best_val = float(v)
                    best_probe = p

    return {
        "meta": meta,
        "latest_epoch": latest_epoch,
        "latest_probe": latest_probe,
        "best_probe": best_probe,
    }


def _render_run_report_md(*, sess: dict[str, Any], summary: dict[str, Any]) -> str:
    meta = summary.get("meta") if isinstance(summary.get("meta"), dict) else {}
    cfg = meta.get("config") if isinstance(meta.get("config"), dict) else {}
    targets = meta.get("targets") if isinstance(meta.get("targets"), dict) else {}
    best_probe = summary.get("best_probe") if isinstance(summary.get("best_probe"), dict) else {}
    latest_probe = summary.get("latest_probe") if isinstance(summary.get("latest_probe"), dict) else {}

    def fmt(v: Any) -> str:
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return f"{float(v):.6g}"
        if v is None:
            return "—"
        return str(v)

    def metric_line(key: str, src: dict[str, Any]) -> str:
        v = src.get(key)
        t = targets.get(key)
        if t is None:
            return f"- `{key}`: `{fmt(v)}`"
        return f"- `{key}`: `{fmt(v)}` (target `{fmt(t)}`)"

    started = sess.get("started_ts") or sess.get("created_ts")
    ended = sess.get("ended_ts")
    lines: list[str] = []
    lines.append("# Watermark run report")
    lines.append("")
    lines.append("## Run")
    lines.append(f"- `id`: `{fmt(sess.get('id'))}`")
    lines.append(f"- `kind`: `{fmt(sess.get('kind'))}`")
    lines.append(f"- `status`: `{fmt(sess.get('status'))}`")
    lines.append(f"- `returncode`: `{fmt(sess.get('returncode'))}`")
    lines.append(f"- `run_dir`: `{fmt(sess.get('run_dir'))}`")
    lines.append(f"- `metrics_path`: `{fmt(sess.get('metrics_path'))}`")
    lines.append(f"- `started_ts`: `{fmt(started)}`")
    lines.append(f"- `ended_ts`: `{fmt(ended)}`")
    lines.append("")
    if cfg:
        lines.append("## Config (selected)")
        for k in (
            "num_clips",
            "epochs_s1",
            "epochs_s1b",
            "epochs_s2",
            "epochs_s1b_post",
            "neg_weight",
            "neg_preamble_target",
            "unknown_ce_weight",
            "model_ce_weight",
            "version_ce_weight",
            "pair_ce_weight",
            "msg_weight",
            "reverb_prob",
            "stage2_payload_on_all",
            "log_steps_every",
            "batch_size",
        ):
            if k in cfg:
                lines.append(f"- `{k}`: `{fmt(cfg.get(k))}`")
        lines.append("")

    if best_probe:
        lines.append("## Best probe (by `wm_acc`)")
        lines.append(f"- `stage`: `{fmt(best_probe.get('stage'))}`")
        lines.append(f"- `epoch`: `{fmt(best_probe.get('epoch'))}`")
        for k in (
            "mini_auc",
            "tpr_at_fpr_1pct",
            "mini_auc_reverb",
            "tpr_at_fpr_1pct_reverb",
            "attr_acc",
            "wm_acc",
            "attr_acc_reverb",
            "wm_acc_reverb",
            "p_clean_pos_mean",
            "p_clean_neg_mean",
        ):
            if k in best_probe:
                lines.append(metric_line(k, best_probe))
        lines.append("")

    if latest_probe:
        lines.append("## Latest probe")
        lines.append(f"- `stage`: `{fmt(latest_probe.get('stage'))}`")
        lines.append(f"- `epoch`: `{fmt(latest_probe.get('epoch'))}`")
        for k in (
            "mini_auc",
            "tpr_at_fpr_1pct",
            "attr_acc",
            "wm_acc",
            "p_clean_pos_mean",
            "p_clean_neg_mean",
        ):
            if k in latest_probe:
                lines.append(metric_line(k, latest_probe))
        lines.append("")

    if targets:
        lines.append("## Targets")
        lines.append("- These are the dashboard’s “expected results” guide-lines for this run.")
        for k, v in sorted(targets.items(), key=lambda kv: kv[0]):
            lines.append(f"- `{k}`: `{fmt(v)}`")
        lines.append("")

    lines.append("## Notes")
    lines.append("- If `mini_auc` is high but `wm_acc` is near chance, the system is doing presence-only. Increase encoder training (Stage 2) and consider Stage 3 finetuning.")
    lines.append("- If clean clips are often misclassified, increase clean negatives and (in finetune) increase `neg_weight` so the decoder stays calibrated.")
    lines.append("- Use the confusion matrix to spot class collapse (predicting only a few classes).")
    lines.append("")
    return "\n".join(lines)


def _estimate_seconds_for_quick_voice(
    *,
    runs_dir: Path,
    num_clips: int,
    epochs_s1: int,
    epochs_s1b: int,
    epochs_s2: int,
    epochs_s1b_post: int,
) -> Optional[dict[str, Any]]:
    """
    Heuristic ETA based on prior quick_voice_smoke_train runs in runs_dir.

    Model: per-epoch time scales linearly with num_clips; we take a robust median
    of (seconds / clip / epoch) per stage across past runs.
    """
    per_clip_per_epoch: dict[str, list[float]] = {"s1": [], "s1b": [], "s2": [], "s1b_post": []}

    # Sources:
    # - Sessions created by controller (runs_dir/*/session.json)
    # - Legacy runs under outputs/**/metrics.jsonl
    candidate_logs: list[Path] = []

    for sess_file in runs_dir.glob("*/session.json"):
        sess = _json_load(sess_file)
        if not sess or sess.get("kind") != "quick_voice_smoke_train":
            continue
        mp = Path(sess.get("metrics_path", ""))
        if mp.exists():
            candidate_logs.append(mp)

    for mp in (REPO_ROOT / "outputs").glob("**/metrics.jsonl"):
        candidate_logs.append(mp)

    seen: set[str] = set()
    for metrics_path in candidate_logs:
        key = str(metrics_path.resolve())
        if key in seen:
            continue
        seen.add(key)
        if not metrics_path.exists():
            continue
        meta = _read_first_meta(metrics_path) or {}
        if not isinstance(meta, dict):
            continue
        if meta.get("run_name") != "quick_voice_smoke_train":
            continue
        events = _read_tail_jsonl(metrics_path, max_lines=20000)
        cfg = (meta.get("config") or {}) if isinstance(meta.get("config"), dict) else {}
        n = cfg.get("num_clips")
        if not isinstance(n, (int, float)) or int(n) <= 0:
            continue
        n = int(n)
        epochs = [e for e in events if e.get("type") == "epoch" and isinstance(e.get("ts"), (int, float))]
        by_stage: dict[str, list[dict[str, Any]]] = {}
        for e in epochs:
            st = e.get("stage")
            if not isinstance(st, str):
                continue
            by_stage.setdefault(st, []).append(e)
        for st, evs in by_stage.items():
            evs.sort(key=lambda x: float(x["ts"]))
            dts = []
            prev_ts = None
            for e in evs:
                ts = float(e["ts"])
                if prev_ts is not None:
                    dts.append(max(0.0, ts - prev_ts))
                prev_ts = ts
            med = _median(dts)
            if med is None or med <= 0:
                continue
            per_clip_per_epoch.setdefault(st, []).append(float(med) / float(n))

    stage_rate = {st: _median(vs) for st, vs in per_clip_per_epoch.items()}
    if not any(stage_rate.values()):
        return None

    def stage_secs(stage: str, epochs: int) -> float:
        r = stage_rate.get(stage)
        if r is None:
            return 0.0
        return float(r) * float(num_clips) * float(epochs)

    s = {
        "s1": stage_secs("s1", epochs_s1),
        "s1b": stage_secs("s1b", epochs_s1b),
        "s2": stage_secs("s2", epochs_s2),
        "s1b_post": stage_secs("s1b_post", epochs_s1b_post),
    }
    total = sum(s.values())
    return {"total_seconds": total, "by_stage_seconds": s, "stage_rate_sec_per_clip_epoch": stage_rate}


def _make_controller_app(runs_dir: Path):
    """
    Multi-run controller + dashboard.

    - Create sessions with configs.
    - Launch training runs in the background.
    - Render a dashboard for any session without restarting the server.
    """
    import os
    import shlex
    import secrets
    import subprocess
    import threading
    import sys
    import signal

    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse

    runs_dir.mkdir(parents=True, exist_ok=True)

    app = FastAPI()

    lock = threading.Lock()
    procs: dict[str, subprocess.Popen] = {}
    proc_fhs: dict[str, Any] = {}

    def load_sessions() -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for p in runs_dir.glob("*/session.json"):
            sess = _json_load(p)
            if not sess:
                continue
            out[str(sess.get("id"))] = sess
        return out

    def save_session(sess: dict[str, Any]) -> None:
        sid = str(sess["id"])
        sdir = runs_dir / sid
        sdir.mkdir(parents=True, exist_ok=True)
        sess["session_path"] = str((sdir / "session.json").resolve())
        _json_dump(sdir / "session.json", sess)

    def start_process(sess: dict[str, Any]) -> None:
        sid = str(sess["id"])
        if sess.get("status") in {"running"}:
            return
        cmd = sess.get("cmd")
        if not isinstance(cmd, list) or not cmd:
            raise ValueError("session cmd missing")
        sdir = Path(sess["run_dir"])
        stdout_path = Path(sess["stdout_path"])
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        fh = stdout_path.open("a", encoding="utf-8")
        p = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=fh,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ},
            start_new_session=True,
        )
        with lock:
            procs[sid] = p
            proc_fhs[sid] = fh
        sess["pid"] = int(p.pid)
        sess["status"] = "running"
        sess["started_ts"] = time.time()
        save_session(sess)

    def _filter_flag_args(argv: list[str], *, forbidden: set[str]) -> list[str]:
        """
        Remove forbidden flags and their values from an argv list.
        Handles:
          --flag value
          --flag=value
        """
        out: list[str] = []
        i = 0
        while i < len(argv):
            a = argv[i]
            if a in forbidden:
                i += 2
                continue
            eq = a.split("=", 1)
            if len(eq) == 2 and eq[0] in forbidden:
                i += 1
                continue
            out.append(a)
            i += 1
        return out

    def _strip_stray_positionals(argv: list[str]) -> list[str]:
        """
        Our training CLIs don't take positional args; if the pasted command contains
        stray tokens (often from line-wrapped paths), drop them.

        Heuristic: keep `--flag value` pairs (treat next token as a value unless it
        starts with `--`) and keep `--flag=value`.
        """
        out: list[str] = []
        i = 0
        while i < len(argv):
            a = argv[i]
            if not a.startswith("-"):
                i += 1
                continue
            out.append(a)
            if a.startswith("--") and "=" not in a and (i + 1) < len(argv):
                nxt = argv[i + 1]
                # Some paste sources inject literal newline tokens like "\n".
                # Treat whitespace-only tokens as stray positionals and drop them,
                # otherwise they get incorrectly consumed as a flag value.
                if not str(nxt).strip():
                    i += 2
                    continue
                if not nxt.startswith("--"):
                    out.append(nxt)
                    i += 2
                    continue
            i += 1
        return out

    def _parse_pasted_command(cmd_text: str) -> dict[str, Any]:
        """
        Parse a pasted CLI command and return a sanitized session spec.

        Safety: only allows running our supported python modules.
        """
        tokens = shlex.split(cmd_text)
        if not tokens:
            raise ValueError("empty command")

        # Expect: python ... -m watermark.scripts.<tool> [args...]
        if "-m" not in tokens:
            raise ValueError("command must use `-m watermark.scripts.<...>`")
        m_idx = tokens.index("-m")
        if m_idx + 1 >= len(tokens):
            raise ValueError("missing module after -m")

        module = tokens[m_idx + 1]
        allow = {
            "watermark.scripts.quick_voice_smoke_train": "quick_voice_smoke_train",
            "watermark.scripts.train_full": "train_full",
        }
        if module not in allow:
            raise ValueError(f"unsupported module: {module}")
        kind = allow[module]

        # Keep only args after module, but strip any user-provided output/metrics paths.
        args = tokens[m_idx + 2 :]
        if kind == "quick_voice_smoke_train":
            args = _filter_flag_args(args, forbidden={"--out", "--log_metrics"})
        else:
            args = _filter_flag_args(args, forbidden={"--output", "--log_metrics"})
        args = _strip_stray_positionals(args)

        return {"kind": kind, "module": module, "args": args}

    def try_finalize(sess: dict[str, Any]) -> None:
        sid = str(sess["id"])
        with lock:
            p = procs.get(sid)
        if not p:
            pid = sess.get("pid")
            if isinstance(pid, int) and _is_pid_alive(pid):
                sess["status"] = "running"
                save_session(sess)
                return
            if sess.get("status") == "stopping":
                sess["returncode"] = sess.get("returncode")
                sess["ended_ts"] = time.time()
                sess["status"] = "stopped"
                save_session(sess)
                return
            if sess.get("status") == "running":
                sess["status"] = "unknown"
                save_session(sess)
            return
        rc = p.poll()
        if rc is None:
            return
        sess["returncode"] = int(rc)
        sess["ended_ts"] = time.time()
        if sess.get("status") == "stopping":
            sess["status"] = "stopped"
        else:
            sess["status"] = "completed" if rc == 0 else "failed"
        save_session(sess)
        with lock:
            procs.pop(sid, None)
            fh = proc_fhs.pop(sid, None)
        try:
            if fh is not None:
                fh.close()
        except Exception:
            pass

        # Append a summary event to metrics.jsonl (best-effort).
        mp = Path(sess["metrics_path"])
        if mp.exists():
            events = _read_tail_jsonl(mp, max_lines=200000)
            summ = _summarize_metrics(events)
            payload = {
                "type": "summary",
                "stage": "final",
                "epoch": 0,
                "status": sess.get("status"),
                "returncode": sess.get("returncode"),
                "run_seconds": float(sess.get("ended_ts", time.time()) - float(sess.get("started_ts", sess.get("created_ts", time.time())))),
                "best_probe": summ.get("best_probe"),
                "latest_probe": summ.get("latest_probe"),
                "latest_epoch": summ.get("latest_epoch"),
            }
            try:
                with mp.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            except Exception:
                pass
            try:
                report_path = Path(sess.get("run_dir", "")) / "report.md"
                report_path.write_text(_render_run_report_md(sess=sess, summary=summ), encoding="utf-8")
            except Exception:
                pass

    def monitor_loop() -> None:
        while True:
            try:
                sessions = load_sessions()
                for sess in sessions.values():
                    if sess.get("status") in {"running", "stopping"}:
                        try_finalize(sess)
            except Exception:
                pass
            time.sleep(1.0)

    threading.Thread(target=monitor_loop, daemon=True).start()

    @app.get("/", response_class=HTMLResponse)
    def index():
        return """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Watermark Controller Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
	    <style>
	      :root {
	        --bg: #0b1220;
	        --panel: rgba(255,255,255,0.06);
	        --panel2: rgba(255,255,255,0.08);
	        --border: rgba(255,255,255,0.10);
	        --text: rgba(255,255,255,0.92);
	        --muted: rgba(255,255,255,0.65);
	        --muted2: rgba(255,255,255,0.50);
	        --good: #34d399;
	        --warn: #fbbf24;
	        --bad: #fb7185;
	        --accent: #60a5fa;
	        --shadow: 0 18px 60px rgba(0,0,0,0.35);
	        --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
	      }
	      body {
	        background: radial-gradient(1200px 800px at 20% 0%, rgba(96,165,250,0.22), transparent 55%),
	                    radial-gradient(900px 700px at 80% 0%, rgba(217,70,239,0.18), transparent 55%),
	                    var(--bg);
	        color: var(--text);
	        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
	        margin: 0;
	      }
	      .wrap { display: grid; grid-template-columns: 340px 1fr; min-height: 100vh; }
	      .side { border-right: 1px solid var(--border); background: rgba(0,0,0,0.18); padding: 14px; height: 100vh; overflow: auto; }
	      .main { padding: 18px 18px 44px; }
	      h1 { font-size: 16px; margin: 0 0 8px; letter-spacing: 0.2px; }
	      h2 { font-size: 13px; margin: 18px 0 8px; color: rgba(255,255,255,0.86); }
	      .sub { font-size: 12px; color: var(--muted); margin: 0 0 10px; }
	      .panel { background: var(--panel); border: 1px solid var(--border); border-radius: 16px; padding: 12px; box-shadow: var(--shadow); backdrop-filter: blur(10px); }
	      .row { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
	      .btn { cursor: pointer; background: rgba(255,255,255,0.08); border: 1px solid var(--border); color: var(--text); border-radius: 12px; padding: 8px 10px; font-size: 12px; transition: transform 120ms ease, background 120ms ease, border-color 120ms ease; }
	      .btn:hover { background: rgba(255,255,255,0.10); border-color: rgba(255,255,255,0.14); transform: translateY(-1px); }
	      .btn:disabled { cursor: not-allowed; opacity: 0.55; transform: none; }
	      .btnPrimary { background: rgba(96,165,250,0.18); border-color: rgba(96,165,250,0.35); }
	      .btnPrimary:hover { background: rgba(96,165,250,0.22); border-color: rgba(96,165,250,0.42); }
	      .btnBad { background: rgba(251,113,133,0.10); border-color: rgba(251,113,133,0.35); }
	      .btnBad:hover { background: rgba(251,113,133,0.14); border-color: rgba(251,113,133,0.42); }
	      .input, select, textarea { background: rgba(255,255,255,0.06); border: 1px solid var(--border); color: var(--text); border-radius: 12px; padding: 8px 10px; font-size: 12px; width: 100%; outline: none; }
	      .input:focus, select:focus, textarea:focus { border-color: rgba(96,165,250,0.55); box-shadow: 0 0 0 4px rgba(96,165,250,0.12); }
	      label { font-size: 11px; color: var(--muted); display: block; margin-bottom: 6px; }
	      .mono { font-family: var(--mono); font-size: 11px; color: rgba(255,255,255,0.85); }
	      code { font-family: var(--mono); font-size: 11px; background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.10); padding: 2px 7px; border-radius: 999px; }
	      .sessions { display: grid; gap: 8px; max-height: 52vh; overflow: auto; padding-right: 4px; }
	      .sess { padding: 10px; border-radius: 14px; border: 1px solid var(--border); background: rgba(255,255,255,0.05); cursor: pointer; transition: background 120ms ease, border-color 120ms ease, transform 120ms ease; }
	      .sess:hover { background: rgba(255,255,255,0.07); border-color: rgba(255,255,255,0.14); transform: translateY(-1px); }
	      .sessActive { outline: 2px solid rgba(96,165,250,0.35); }
	      .badge { display: inline-flex; align-items: center; gap: 6px; padding: 4px 8px; border-radius: 999px; border: 1px solid var(--border); background: rgba(255,255,255,0.06); font-size: 11px; color: rgba(255,255,255,0.86); }
	      .dot { width: 8px; height: 8px; border-radius: 999px; display: inline-block; }
	      .ok { color: var(--good); }
	      .warn { color: var(--warn); }
	      .bad { color: var(--bad); }
	      .dot.ok { background: var(--good); }
	      .dot.warn { background: var(--warn); }
	      .dot.bad { background: var(--bad); }
	      .grid2 { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
	      .grid3 { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }
	      .chartWrap { height: 280px; border-radius: 14px; overflow: hidden; background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); }
	      canvas { width: 100% !important; height: 100% !important; }
	      .matrixWrap { overflow: auto; border-radius: 14px; border: 1px solid rgba(255,255,255,0.08); background: rgba(0,0,0,0.10); }
	      .matrix { width: 100%; border-collapse: collapse; font-size: 11px; }
	      .matrix th, .matrix td { padding: 8px 8px; border-bottom: 1px solid rgba(255,255,255,0.08); text-align: center; vertical-align: middle; }
	      .matrix th { color: rgba(255,255,255,0.70); font-weight: 600; }
	      .matrix th:first-child { text-align: left; position: sticky; left: 0; background: rgba(0,0,0,0.35); backdrop-filter: blur(10px); }
	      .cell { display: inline-block; padding: 4px 7px; border-radius: 10px; font-family: var(--mono); font-size: 11px; min-width: 28px; text-align: center; }
	      pre { white-space: pre-wrap; margin: 0; }
	      .tiny { font-size: 11px; color: var(--muted2); }
	      ::-webkit-scrollbar { height: 10px; width: 10px; }
	      ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.12); border: 2px solid rgba(0,0,0,0); border-radius: 999px; background-clip: padding-box; }
	      ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.18); border: 2px solid rgba(0,0,0,0); background-clip: padding-box; }
	      @media (max-width: 1050px) {
	        .wrap { grid-template-columns: 1fr; }
	        .side { height: auto; border-right: 0; border-bottom: 1px solid var(--border); }
	        .sessions { max-height: 260px; }
	      }
	      @media (max-width: 820px) {
	        .grid3 { grid-template-columns: 1fr; }
	        .grid2 { grid-template-columns: 1fr; }
	        .chartWrap { height: 240px; }
	      }
	    </style>
  </head>
  <body>
    <div class="wrap">
      <div class="side">
        <h1>Controller</h1>
        <div class="sub">Create runs, monitor progress, and browse past sessions.</div>

        <div class="panel">
          <div class="row">
            <button class="btn btnPrimary" id="newRunBtn">New run</button>
            <button class="btn" id="refreshBtn">Refresh</button>
          </div>
          <div style="height:10px"></div>
          <div class="sessions" id="sessions"></div>
        </div>

        <h2>Run config</h2>
        <div class="panel" id="runForm" style="display:none;">
          <div class="grid2">
            <div>
              <label>Kind</label>
              <select id="kind">
                <option value="quick_voice_smoke_train">quick_voice_smoke_train</option>
                <option value="train_full">train_full</option>
              </select>
            </div>
            <div>
              <label>Name (optional)</label>
              <input class="input" id="name" placeholder="e.g. 2k_attr_v1" />
            </div>
          </div>
          <div style="height:10px"></div>

	          <div id="formQuickVoice">
              <div class="grid3">
                <div><label>manifest (optional)</label><input class="input" id="q_manifest" placeholder="e.g. outputs/.../manifest.json" /></div>
                <div><label>load_encoder (optional)</label><input class="input" id="q_load_encoder" placeholder="e.g. outputs/.../encoder.pt" /></div>
                <div><label>load_decoder (optional)</label><input class="input" id="q_load_decoder" placeholder="e.g. outputs/.../decoder.pt" /></div>
              </div>
              <div style="height:10px"></div>
	            <div class="grid2">
	              <div><label>source_dir</label><input class="input" id="q_source_dir" value="mini_benchmark_data" /></div>
	              <div><label>num_clips</label><input class="input" id="q_num_clips" value="512" /></div>
	            </div>
	            <div style="height:10px"></div>
	            <div class="grid3">
	              <div><label>epochs_s1</label><input class="input" id="q_epochs_s1" value="6" /></div>
	              <div><label>epochs_s1b</label><input class="input" id="q_epochs_s1b" value="1" /></div>
	              <div><label>epochs_s2</label><input class="input" id="q_epochs_s2" value="12" /></div>
	            </div>
	            <div style="height:10px"></div>
	            <div class="grid3">
	              <div><label>epochs_s1b_post</label><input class="input" id="q_epochs_s1b_post" value="12" /></div>
	              <div><label>probe_clips</label><input class="input" id="q_probe_clips" value="1024" /></div>
	              <div><label>reverb_prob</label><input class="input" id="q_reverb_prob" value="0.0" /></div>
	            </div>
	            <div style="height:10px"></div>
	            <div class="grid3">
	              <div><label>msg_weight</label><input class="input" id="q_msg_weight" value="1.0" /></div>
	              <div>
	                <label>stage2_payload_on_all</label>
	                <select id="q_stage2_payload_on_all">
	                  <option value="0" selected>0</option>
	                  <option value="1">1</option>
	                </select>
	              </div>
	              <div></div>
	            </div>
	            <div style="height:10px"></div>
	            <div class="grid3">
	              <div><label>unknown_ce_weight</label><input class="input" id="q_unknown_ce_weight" value="1" /></div>
	              <div><label>model_ce_weight</label><input class="input" id="q_model_ce_weight" value="4" /></div>
	              <div><label>pair_ce_weight</label><input class="input" id="q_pair_ce_weight" value="4" /></div>
	            </div>
            <div style="height:10px"></div>
            <div class="grid3">
              <div><label>version_ce_weight</label><input class="input" id="q_version_ce_weight" value="2" /></div>
              <div><label>neg_weight</label><input class="input" id="q_neg_weight" value="5" /></div>
              <div><label>neg_preamble_target</label><input class="input" id="q_neg_preamble_target" value="0.5" /></div>
            </div>
	            <div style="height:10px"></div>
	            <div class="grid3">
	              <div><label>seed</label><input class="input" id="q_seed" value="1337" /></div>
	              <div><label>probe_every</label><input class="input" id="q_probe_every" value="1" /></div>
	              <div><label>probe_reverb_every</label><input class="input" id="q_probe_reverb_every" value="999999" /></div>
	            </div>
	            <div style="height:10px"></div>
              <div class="grid3">
                <div><label>detect_weight</label><input class="input" id="q_detect_weight" value="1.0" /></div>
                <div><label>id_weight</label><input class="input" id="q_id_weight" value="2.0" /></div>
                <div>
                  <label>freeze_detect_head_in_s3</label>
                  <select id="q_freeze_detect_head_in_s3">
                    <option value="0" selected>0</option>
                    <option value="1">1</option>
                  </select>
                </div>
              </div>
              <div style="height:10px"></div>
	            <div class="grid3">
	              <div><label>log_steps_every</label><input class="input" id="q_log_steps_every" value="25" /></div>
	              <div></div>
	              <div></div>
	            </div>
	            <div style="height:10px"></div>
	            <label>extra_args (optional)</label>
	            <input class="input" id="q_extra_args" placeholder="e.g. --profile large --msg_weight 1" />
	          </div>

	          <div id="formTrainFull" style="display:none;">
	            <label>manifest</label>
	            <input class="input" id="t_manifest" placeholder="e.g. mini_benchmark_train.json" />
	            <div style="height:10px"></div>
              <div class="grid2">
                <div><label>load_encoder (optional)</label><input class="input" id="t_load_encoder" placeholder="e.g. outputs/.../encoder.pt" /></div>
                <div><label>load_decoder (optional)</label><input class="input" id="t_load_decoder" placeholder="e.g. outputs/.../decoder.pt" /></div>
              </div>
              <div style="height:10px"></div>
	            <div class="grid3">
	              <div><label>epochs_s1</label><input class="input" id="t_epochs_s1" value="20" /></div>
	              <div><label>epochs_s1b</label><input class="input" id="t_epochs_s1b" value="10" /></div>
	              <div><label>warmup_s1b</label><input class="input" id="t_warmup_s1b" value="3" /></div>
	            </div>
	            <div style="height:10px"></div>
	            <div class="grid3">
	              <div><label>epochs_s2</label><input class="input" id="t_epochs_s2" value="20" /></div>
	              <div><label>epochs_s1b_post</label><input class="input" id="t_epochs_s1b_post" value="0" /></div>
	              <div><label>reverb_prob</label><input class="input" id="t_reverb_prob" value="0.25" /></div>
	            </div>
	            <div style="height:10px"></div>
              <div class="grid3">
                <div><label>detect_weight</label><input class="input" id="t_detect_weight" value="1.0" /></div>
                <div><label>id_weight</label><input class="input" id="t_id_weight" value="2.0" /></div>
                <div>
                  <label>freeze_detect_head_in_s3</label>
                  <select id="t_freeze_detect_head_in_s3">
                    <option value="0" selected>0</option>
                    <option value="1">1</option>
                  </select>
                </div>
              </div>
              <div style="height:10px"></div>
	            <div class="grid3">
	              <div><label>msg_weight</label><input class="input" id="t_msg_weight" value="1.0" /></div>
	              <div>
	                <label>stage2_payload_on_all</label>
	                <select id="t_stage2_payload_on_all">
	                  <option value="0" selected>0</option>
	                  <option value="1">1</option>
	                </select>
	              </div>
	              <div></div>
	            </div>
	            <div style="height:10px"></div>
	            <div class="grid3">
	              <div><label>unknown_ce_weight</label><input class="input" id="t_unknown_ce_weight" value="0.0" /></div>
	              <div><label>model_ce_weight</label><input class="input" id="t_model_ce_weight" value="1.0" /></div>
	              <div><label>pair_ce_weight</label><input class="input" id="t_pair_ce_weight" value="2.0" /></div>
	            </div>
            <div style="height:10px"></div>
            <div class="grid3">
              <div><label>version_ce_weight</label><input class="input" id="t_version_ce_weight" value="1.0" /></div>
              <div><label>probe_clips</label><input class="input" id="t_probe_clips" value="1024" /></div>
              <div><label>probe_every</label><input class="input" id="t_probe_every" value="1" /></div>
            </div>
	            <div style="height:10px"></div>
	            <label>extra_args (optional)</label>
	            <input class="input" id="t_extra_args" placeholder="e.g. --msg_weight 1 --neg_weight 0.4" />
	            <div style="height:10px"></div>
	            <div class="grid3">
	              <div><label>log_steps_every</label><input class="input" id="t_log_steps_every" value="100" /></div>
	              <div></div>
	              <div></div>
	            </div>
	          </div>

          <div style="height:12px"></div>
          <div class="tiny" id="etaBox"></div>
          <div style="height:8px"></div>
          <div class="row">
            <button class="btn" id="estimateBtn">Estimate time</button>
            <button class="btn btnPrimary" id="startBtn">Start</button>
          </div>
          <div style="height:8px"></div>
          <div class="mono tiny" id="cmdPreview"></div>
        </div>

	        <h2>Paste command</h2>
	        <div class="panel" id="pastePanel">
	          <div class="tiny">Paste a full command for <code class="mono">quick_voice_smoke_train</code> or <code class="mono">train_full</code>. Output/log paths are auto-overridden to the session folder.</div>
	          <div style="height:8px"></div>
	          <textarea class="input mono" id="rawCmd" rows="5" style="width:100%; resize:vertical;"
	            placeholder="./.venv/bin/python -m watermark.scripts.quick_voice_smoke_train --source_dir mini_benchmark_data --num_clips 2048 --epochs_s1 6 --epochs_s1b 1 --epochs_s2 12 --epochs_s1b_post 12 --model_ce_weight 4 --version_ce_weight 2 --pair_ce_weight 4 --unknown_ce_weight 1 --neg_weight 5 --neg_preamble_target 0.5 --reverb_prob 0.0 --probe_clips 1024 --probe_every 1 --probe_reverb_every 999999 --out outputs/ignored"></textarea>
	          <div style="height:8px"></div>
	          <div class="row">
	            <button class="btn btnPrimary" id="runRawBtn">Run pasted command</button>
	          </div>
	          <div style="height:8px"></div>
	          <div class="mono tiny" id="rawCmdNote"></div>
	        </div>

	        <h2>Attach log</h2>
	        <div class="panel" id="attachPanel">
	          <div class="tiny">Attach an existing run by pointing at its <code class="mono">metrics.jsonl</code> (terminal runs included). No rerun required.</div>
	          <div style="height:8px"></div>
	          <label>metrics.jsonl path</label>
	          <input class="input mono" id="attachPath" placeholder="outputs/quick_voice_smoke_2k_dashboard/metrics.jsonl" />
	          <div style="height:8px"></div>
	          <label>Name (optional)</label>
	          <input class="input" id="attachName" placeholder="e.g. terminal_2k_run" />
	          <div style="height:8px"></div>
	          <div class="row">
	            <button class="btn btnPrimary" id="attachBtn">Attach</button>
	          </div>
	          <div style="height:8px"></div>
	          <div class="mono tiny" id="attachNote"></div>
	        </div>
	      </div>

      <div class="main">
        <div class="row" style="justify-content:space-between; align-items:flex-start;">
	          <div>
	            <h1>Watermark Dashboard</h1>
	            <div class="sub" id="selectedInfo">Select a session on the left.</div>
	            <div class="mono tiny" id="cmdShown"></div>
	            <div class="row" style="gap:8px; margin-top:8px;">
	              <span class="badge" id="progressInfo">—</span>
	              <span class="badge" id="etaInfo">—</span>
	            </div>
	          </div>
	          <div class="row">
	            <span class="badge" id="statusBadge">—</span>
	            <button class="btn btnBad" id="stopBtn" disabled>Stop</button>
            <button class="btn btnPrimary" id="resumeBtn" disabled>Resume</button>
            <a class="btn" id="dlBtn" href="#" target="_blank" rel="noreferrer">Download metrics</a>
          </div>
        </div>

        <div style="height:12px"></div>
        <div class="panel">
	          <div class="grid3">
	            <div class="chartWrap"><canvas id="chart_auc"></canvas></div>
	            <div class="chartWrap"><canvas id="chart_tpr"></canvas></div>
	            <div class="chartWrap"><canvas id="chart_preamble"></canvas></div>
	          </div>
          <div style="height:10px"></div>
          <div class="grid3">
             <div class="chartWrap"><canvas id="chart_id"></canvas></div>
             <div class="chartWrap"><canvas id="chart_crc"></canvas></div>
             <div class="chartWrap"><canvas id="chart_bits"></canvas></div>
          </div>
          <div style="height:10px"></div>
          <div class="grid2">
            <div class="chartWrap"><canvas id="chart_s2"></canvas></div>
            <div class="chartWrap"><canvas id="chart_budget"></canvas></div>
          </div>
        </div>

	        <div style="height:12px"></div>
	        <div class="grid2">
	          <div class="panel">
	            <h2 style="margin:0 0 8px;">Confusion · model</h2>
	            <div class="tiny">Latest probe event confusion matrix (if available).</div>
              <div class="row" style="gap:10px; align-items:center; margin-top:8px;">
                <div class="badge">Matrix</div>
                <select id="matrixMode" class="input mono" style="max-width: 100%;">
                  <option value="attr" selected>Attribution-only (K×K)</option>
                  <option value="full">Thresholded full (K+1×K+1)</option>
                </select>
              </div>
	            <div style="height:8px"></div>
	            <div id="matrixModel" class="matrixWrap">—</div>
	          </div>
	          <div class="panel">
	            <h2 style="margin:0 0 8px;">Confusion · version</h2>
	            <div class="tiny">Latest probe event confusion matrix (if available).</div>
	            <div style="height:8px"></div>
	            <div id="matrixVersion" class="matrixWrap">—</div>
	          </div>
	        </div>

        <div style="height:12px"></div>
	        <div class="grid2">
	          <div class="panel">
	            <h2 style="margin:0 0 8px;">Stdout (tail)</h2>
	            <pre id="stdout" class="mono tiny">—</pre>
	          </div>
	          <div class="panel">
	            <h2 style="margin:0 0 8px;">Decode report</h2>
	            <pre id="decode" class="mono tiny">—</pre>
	          </div>
	        </div>
	        <div style="height:12px"></div>
	        <div class="panel">
	          <h2 style="margin:0 0 8px;">Run report</h2>
	          <pre id="report" class="mono tiny">—</pre>
	        </div>
	      </div>
    </div>

    <script>
      let selected = null;
      let charts = {};
      const MATRIX_MODE_KEY = "wmDash:matrixMode";

      function initMatrixMode() {
        const el = document.getElementById("matrixMode");
        if (!el || el._wmInit) return;
        el._wmInit = true;
        const saved = localStorage.getItem(MATRIX_MODE_KEY);
        if (saved) el.value = saved;
        el.addEventListener("change", () => localStorage.setItem(MATRIX_MODE_KEY, el.value));
      }

      function getMatrixMode() {
        const el = document.getElementById("matrixMode");
        if (!el) return localStorage.getItem(MATRIX_MODE_KEY) || "attr";
        return el.value || "attr";
      }
      let refreshing = false;

      function fmt(x) {
        if (x === null || x === undefined) return "—";
        if (typeof x === "number") return x.toFixed(4).replace(/0+$/,"").replace(/\\.$/,"");
        return String(x);
      }

      function dot(cls) { return `<span class="dot ${cls}"></span>`; }

      function statusCls(st) {
        if (st === "running") return "warn";
        if (st === "completed") return "ok";
        if (st === "external") return "ok";
        if (st === "failed" || st === "stopped") return "bad";
        return "";
      }

      function lineChart(id, ymin=null, ymax=null) {
        const ctx = document.getElementById(id);
        return new Chart(ctx, {
          type: "line",
          data: { labels: [], datasets: [] },
          options: {
            animation: false,
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: "index", intersect: false },
            scales: {
              x: { ticks: { color: "rgba(255,255,255,0.55)" }, grid: { color: "rgba(255,255,255,0.06)" } },
              y: { ticks: { color: "rgba(255,255,255,0.55)" }, grid: { color: "rgba(255,255,255,0.06)" }, min: ymin, max: ymax },
            },
            plugins: { legend: { labels: { color: "rgba(255,255,255,0.70)" } } },
          }
        });
      }

      function ensureCharts() {
        if (charts._ready) return;
        charts.auc = lineChart("chart_auc", 0, 1);
        charts.tpr = lineChart("chart_tpr", 0, 1);
        charts.id = lineChart("chart_id", 0, 1);
        charts.crc = lineChart("chart_crc", 0, 1);
        charts.bits = lineChart("chart_bits", 0.7, 1.0);
        charts.preamble = lineChart("chart_preamble", 0, 1);
        charts.s2 = lineChart("chart_s2");
        charts.budget = lineChart("chart_budget", 0, 0.5);
        charts._ready = true;
      }

      function setDatasets(chart, labels, datasets) {
        chart.data.labels = labels;
        chart.data.datasets = datasets;
        chart.update("none");
      }

	      function solid(label, color, data) {
	        return { label, data, borderColor: color, backgroundColor: color, tension: 0.25, fill: false, pointRadius: 0 };
	      }

	      function dashed(label, color, data) {
	        return { label, data, borderColor: color, backgroundColor: color, tension: 0.25, fill: false, pointRadius: 0, borderDash: [6,6] };
	      }

	      function makeMatrix(containerId, mat, opts={}) {
	        const el = document.getElementById(containerId);
	        if (!mat || !Array.isArray(mat) || mat.length === 0) { el.textContent = "—"; return; }
	        const rows = mat.length;
	        const cols = Array.isArray(mat[0]) ? mat[0].length : 0;
	        if (cols === 0) { el.textContent = "—"; return; }

	        const rowPrefix = opts.rowPrefix || "";
	        const colPrefix = opts.colPrefix || "";
	        const rowNames = opts.rowNames || Array.from({length: rows}, (_, i) => `${rowPrefix}${i}`);
	        const colNames = opts.colNames || Array.from({length: cols}, (_, i) => (i === cols-1 && opts.lastColName ? opts.lastColName : `${colPrefix}${i}`));

	        let maxVal = 0;
	        for (let r=0; r<rows; r++) for (let c=0; c<cols; c++) maxVal = Math.max(maxVal, mat[r][c] || 0);
	        if (maxVal <= 0) maxVal = 1;

	        let html = `<table class="matrix"><thead><tr><th>t\\p</th>`;
	        for (const cn of colNames) html += `<th class="mono">${cn}</th>`;
	        html += `</tr></thead><tbody>`;

	        for (let r=0; r<rows; r++) {
	          html += `<tr><th class="mono">${rowNames[r] ?? r}</th>`;
	          for (let c=0; c<cols; c++) {
	            const v = mat[r][c] || 0;
	            const a = 0.06 + 0.34*(v/maxVal);
	            const isDiag = (r === c);
	            const border = isDiag ? "rgba(52,211,153,0.45)" : "rgba(255,255,255,0.12)";
	            const bg = `rgba(96,165,250,${a.toFixed(3)})`;
	            html += `<td><span class="cell" style="background:${bg}; border:1px solid ${border};">${v}</span></td>`;
	          }
	          html += `</tr>`;
	        }
	        html += `</tbody></table>`;
	        el.innerHTML = html;
	      }

	      async function api(path, opts={}) {
	        const r = await fetch(path, { cache: "no-store", ...opts });
	        if (!r.ok) throw new Error(await r.text());
	        return r;
	      }

      async function refreshSessions() {
        const r = await api("/api/sessions");
        const sessions = await r.json();
        const el = document.getElementById("sessions");
        el.innerHTML = "";
        for (const s of sessions) {
          const st = s.status || "—";
          const cls = statusCls(st);
          const active = (selected && selected.id === s.id) ? "sessActive" : "";
          const name = s.name ? ` · ${s.name}` : "";
          const kind = s.kind || "";
          const created = s.created_ts ? new Date(s.created_ts*1000).toLocaleTimeString() : "";
          const html = `<div class="sess ${active}" data-id="${s.id}">
              <div style="display:flex; justify-content:space-between; gap:10px;">
                <div style="font-size:12px;"><b>${kind}</b>${name}</div>
                <div class="tiny mono">${created}</div>
              </div>
              <div class="tiny" style="display:flex; justify-content:space-between; margin-top:6px;">
                <span>${dot(cls)}<span class="${cls}">${st}</span></span>
                <span class="mono">${s.pid ? "pid "+s.pid : ""}</span>
              </div>
            </div>`;
          el.insertAdjacentHTML("beforeend", html);
        }
        for (const node of el.querySelectorAll(".sess")) {
          node.addEventListener("click", () => selectSession(node.getAttribute("data-id")));
        }
      }

      async function selectSession(id) {
        const r = await api(`/api/sessions/${id}`);
        selected = await r.json();
        document.getElementById("selectedInfo").textContent = selected.run_dir || id;
        document.getElementById("cmdShown").textContent = selected.cmd ? selected.cmd.join(" ") : "";
        document.getElementById("statusBadge").innerHTML = `${dot(statusCls(selected.status))}<b>${selected.status || "—"}</b>`;
        document.getElementById("stopBtn").disabled = (selected.status !== "running");
        document.getElementById("resumeBtn").disabled = (selected.status === "running" || selected.status === "created");
        document.getElementById("dlBtn").href = `/api/sessions/${id}/download/metrics`;
        await refreshSelected();
        await refreshSessions();
      }

      function showForm(show) {
        document.getElementById("runForm").style.display = show ? "block" : "none";
      }

      function formKind() { return document.getElementById("kind").value; }

	      function readQuickVoiceArgs() {
	        const args = {
            manifest: (document.getElementById("q_manifest").value || "").trim() || null,
            load_encoder: (document.getElementById("q_load_encoder").value || "").trim() || null,
            load_decoder: (document.getElementById("q_load_decoder").value || "").trim() || null,
	          source_dir: document.getElementById("q_source_dir").value,
	          num_clips: parseInt(document.getElementById("q_num_clips").value, 10),
	          epochs_s1: parseInt(document.getElementById("q_epochs_s1").value, 10),
	          epochs_s1b: parseInt(document.getElementById("q_epochs_s1b").value, 10),
	          epochs_s2: parseInt(document.getElementById("q_epochs_s2").value, 10),
	          epochs_s1b_post: parseInt(document.getElementById("q_epochs_s1b_post").value, 10),
            detect_weight: parseFloat(document.getElementById("q_detect_weight").value),
            id_weight: parseFloat(document.getElementById("q_id_weight").value),
            freeze_detect_head_in_s3: (document.getElementById("q_freeze_detect_head_in_s3").value === "1"),
	          msg_weight: parseFloat(document.getElementById("q_msg_weight").value),
	          stage2_payload_on_all: (document.getElementById("q_stage2_payload_on_all").value === "1"),
		          probe_clips: parseInt(document.getElementById("q_probe_clips").value, 10),
		          probe_every: parseInt(document.getElementById("q_probe_every").value, 10),
		          probe_reverb_every: parseInt(document.getElementById("q_probe_reverb_every").value, 10),
		          log_steps_every: parseInt(document.getElementById("q_log_steps_every").value, 10),
		          reverb_prob: parseFloat(document.getElementById("q_reverb_prob").value),
	          unknown_ce_weight: parseFloat(document.getElementById("q_unknown_ce_weight").value),
	          model_ce_weight: parseFloat(document.getElementById("q_model_ce_weight").value),
          version_ce_weight: parseFloat(document.getElementById("q_version_ce_weight").value),
          pair_ce_weight: parseFloat(document.getElementById("q_pair_ce_weight").value),
          neg_weight: parseFloat(document.getElementById("q_neg_weight").value),
          neg_preamble_target: parseFloat(document.getElementById("q_neg_preamble_target").value),
          seed: parseInt(document.getElementById("q_seed").value, 10),
          extra_args: document.getElementById("q_extra_args").value || null,
        };
        return args;
      }

	      function readTrainFullArgs() {
	        const args = {
	          manifest: document.getElementById("t_manifest").value,
            load_encoder: (document.getElementById("t_load_encoder").value || "").trim() || null,
            load_decoder: (document.getElementById("t_load_decoder").value || "").trim() || null,
	          epochs_s1: parseInt(document.getElementById("t_epochs_s1").value, 10),
	          epochs_s1b: parseInt(document.getElementById("t_epochs_s1b").value, 10),
	          warmup_s1b: parseInt(document.getElementById("t_warmup_s1b").value, 10),
	          epochs_s2: parseInt(document.getElementById("t_epochs_s2").value, 10),
	          epochs_s1b_post: parseInt(document.getElementById("t_epochs_s1b_post").value, 10),
	          reverb_prob: parseFloat(document.getElementById("t_reverb_prob").value),
            detect_weight: parseFloat(document.getElementById("t_detect_weight").value),
            id_weight: parseFloat(document.getElementById("t_id_weight").value),
            freeze_detect_head_in_s3: (document.getElementById("t_freeze_detect_head_in_s3").value === "1"),
	          msg_weight: parseFloat(document.getElementById("t_msg_weight").value),
	          stage2_payload_on_all: (document.getElementById("t_stage2_payload_on_all").value === "1"),
	          unknown_ce_weight: parseFloat(document.getElementById("t_unknown_ce_weight").value),
	          model_ce_weight: parseFloat(document.getElementById("t_model_ce_weight").value),
	          version_ce_weight: parseFloat(document.getElementById("t_version_ce_weight").value),
	          pair_ce_weight: parseFloat(document.getElementById("t_pair_ce_weight").value),
		          probe_clips: parseInt(document.getElementById("t_probe_clips").value, 10),
		          probe_every: parseInt(document.getElementById("t_probe_every").value, 10),
		          extra_args: document.getElementById("t_extra_args").value || null,
		          log_steps_every: parseInt(document.getElementById("t_log_steps_every").value, 10),
		        };
	        return args;
	      }

      function updateFormVisibility() {
        const k = formKind();
        document.getElementById("formQuickVoice").style.display = (k === "quick_voice_smoke_train") ? "block" : "none";
        document.getElementById("formTrainFull").style.display = (k === "train_full") ? "block" : "none";
      }

      async function estimate() {
        const k = formKind();
        const args = (k === "quick_voice_smoke_train") ? readQuickVoiceArgs() : readTrainFullArgs();
        const r = await api("/api/estimate", { method:"POST", headers:{\"Content-Type\":\"application/json\"}, body: JSON.stringify({ kind:k, args }) });
        const res = await r.json();
        const box = document.getElementById("etaBox");
        if (!res || !res.total_seconds) { box.textContent = "ETA: not enough historical data yet."; return; }
        const mins = (res.total_seconds/60).toFixed(1);
        box.textContent = `ETA: ~${mins} min (heuristic; based on prior runs).`;
      }

      async function startRun() {
        const k = formKind();
        const name = document.getElementById("name").value || null;
        const args = (k === "quick_voice_smoke_train") ? readQuickVoiceArgs() : readTrainFullArgs();
        const r = await api("/api/sessions", { method:"POST", headers:{\"Content-Type\":\"application/json\"}, body: JSON.stringify({ kind:k, name, args }) });
        const sess = await r.json();
        showForm(false);
        await refreshSessions();
        await selectSession(sess.id);
      }

      async function resumeSelected() {
        if (!selected) return;
        try {
          const r = await api(`/api/sessions/${selected.id}/resume`, { method:"POST" });
          const sess = await r.json();
          await refreshSessions();
          await selectSession(sess.id);
        } catch (e) {
          alert("Failed to resume: " + e.message);
        }
      }

	      async function runPasted() {
	        const command = document.getElementById("rawCmd").value || "";
	        const name = document.getElementById("name").value || null;
	        const r = await api("/api/sessions/raw", { method:"POST", headers:{\"Content-Type\":\"application/json\"}, body: JSON.stringify({ name, command }) });
	        const sess = await r.json();
	        document.getElementById("rawCmdNote").textContent = "Started.";
	        await refreshSessions();
	        await selectSession(sess.id);
	      }

	      async function attachLog() {
	        const metrics_path = document.getElementById("attachPath").value || "";
	        const name = document.getElementById("attachName").value || null;
	        if (!metrics_path.trim()) { document.getElementById("attachNote").textContent = "Missing path."; return; }
	        const r = await api("/api/sessions/attach", { method:"POST", headers:{\"Content-Type\":\"application/json\"}, body: JSON.stringify({ name, metrics_path }) });
	        const sess = await r.json();
	        document.getElementById("attachNote").textContent = "Attached.";
	        await refreshSessions();
	        await selectSession(sess.id);
	      }

	      async function stopSelected() {
	        if (!selected) return;
	        await api(`/api/sessions/${selected.id}/stop`, { method:"POST" });
	        await refreshSelected();
        await refreshSessions();
      }

	      async function refreshSelected() {
	        if (!selected) return;
	        if (refreshing) return;
	        refreshing = true;
	        initMatrixMode();
	        ensureCharts();
	        const id = selected.id;
	        try {
	          const sessR = await api(`/api/sessions/${id}`);
	          selected = await sessR.json();
	          document.getElementById("statusBadge").innerHTML = `${dot(statusCls(selected.status))}<b>${selected.status || "—"}</b>`;
          document.getElementById("stopBtn").disabled = (selected.status !== "running");
          document.getElementById("resumeBtn").disabled = (selected.status === "running" || selected.status === "created");

		          try {
		            const etaR = await api(`/api/sessions/${id}/eta`);
		            const eta = await etaR.json();
		            document.getElementById("progressInfo").textContent = eta.progress_text || "—";
		            document.getElementById("etaInfo").textContent = eta.eta_text || "—";
		          } catch (e) {
		            document.getElementById("progressInfo").textContent = "—";
		            document.getElementById("etaInfo").textContent = "—";
		          }

	          const eventsR = await api(`/api/sessions/${id}/metrics?tail=20000`);
	          const events = await eventsR.json();
	          const meta = events.find(e => e.type === "meta") || {};
	          const targets = (meta && meta.targets) ? meta.targets : {};
	          const probes = events.filter(e => e.type === "probe");
	          const s2 = events.filter(e => e.type === "epoch" && (e.stage === "s2_encoder" || e.stage === "s3_finetune"));

	          const latestProbe = probes.length ? probes[probes.length - 1] : null;
	          if (latestProbe) {
	            const mm = getMatrixMode();
	            const cm = (mm === "full") ? latestProbe.confusion : (latestProbe.confusion_attr || latestProbe.confusion);
	            const prefix = (mm !== "full" && latestProbe.confusion_attr) ? "id" : "class";
	            makeMatrix("matrixModel", cm, { lastColName: null, rowPrefix: `true_${prefix} `, colPrefix: `pred_${prefix} ` });
	            document.getElementById("matrixVersion").textContent = "—";
	          } else {
	            document.getElementById("matrixModel").textContent = "—";
	            document.getElementById("matrixVersion").textContent = "—";
	          }

	          const labels = probes.map(p => `${p.stage}/e${p.epoch}`);
	          const auc = probes.map(p => p.mini_auc ?? null);
	          const aucR = probes.map(p => p.mini_auc_reverb ?? null);
	          const aucT = labels.map(_ => (targets.mini_auc !== undefined ? targets.mini_auc : 0.95));
	          const aucRT = labels.map(_ => (targets.mini_auc_reverb !== undefined ? targets.mini_auc_reverb : 0.85));
	          setDatasets(charts.auc, labels, [
	            solid("mini_auc", "#60a5fa", auc),
	            solid("mini_auc_reverb", "#c084fc", aucR),
	            dashed("target auc", "rgba(96,165,250,0.35)", aucT),
	            dashed("target auc (reverb)", "rgba(192,132,252,0.35)", aucRT),
	          ]);

	          const tpr = probes.map(p => p.tpr_at_fpr_1pct ?? null);
	          const tprR = probes.map(p => p.tpr_at_fpr_1pct_reverb ?? null);
	          const tprT = labels.map(_ => (targets.tpr_at_fpr_1pct !== undefined ? targets.tpr_at_fpr_1pct : 0.85));
	          const tprRT = labels.map(_ => (targets.tpr_at_fpr_1pct_reverb !== undefined ? targets.tpr_at_fpr_1pct_reverb : 0.70));
	          setDatasets(charts.tpr, labels, [
	            solid("tpr@1%FPR", "#34d399", tpr),
	            solid("tpr@1%FPR (reverb)", "#fbbf24", tprR),
	            dashed("target tpr", "rgba(52,211,153,0.35)", tprT),
	            dashed("target tpr (reverb)", "rgba(251,191,36,0.35)", tprRT),
	          ]);

	          const attr = probes.map(p => p.attr_acc ?? null);
	          const wm = probes.map(p => p.wm_acc ?? null);
	          const idPos = probes.map(p => p.id_acc_pos ?? null);
	          const attrR = probes.map(p => p.attr_acc_reverb ?? null);
	          const wmR = probes.map(p => p.wm_acc_reverb ?? null);
	          const idPosR = probes.map(p => p.id_acc_pos_reverb ?? null);
	          setDatasets(charts.id, labels, [
	            solid("attr_acc", "#60a5fa", attr),
	            solid("wm_acc", "#34d399", wm),
	            solid("id_acc_pos", "#fbbf24", idPos),
	            solid("attr_acc_reverb", "#c084fc", attrR),
	            solid("wm_acc_reverb", "#fb7185", wmR),
	            solid("id_acc_pos_reverb", "#94a3b8", idPosR),
	          ]);

	          const predClean = probes.map(p => p.pred_clean_rate ?? null);
	          const pCleanPos = probes.map(p => p.p_clean_pos_mean ?? null);
	          const pCleanNeg = probes.map(p => p.p_clean_neg_mean ?? null);
	          setDatasets(charts.crc, labels, [
	               solid("pred_clean_rate", "#94a3b8", predClean),
	               solid("p_clean_pos_mean", "#fbbf24", pCleanPos),
	               solid("p_clean_neg_mean", "#60a5fa", pCleanNeg),
	          ]);
	          setDatasets(charts.bits, labels, [
	               solid("attr_acc", "#60a5fa", attr),
	               solid("wm_acc", "#34d399", wm),
	          ]);

	          const locPos = probes.map(p => p.wm_prob_loc_pos_mean ?? null);
	          const locNeg = probes.map(p => p.wm_prob_loc_neg_mean ?? null);
	          setDatasets(charts.preamble, labels, [
	            solid("wm_prob_loc_pos_mean", "#34d399", locPos),
	            solid("wm_prob_loc_neg_mean", "#fb7185", locNeg),
	          ]);

	          const s2labels = s2.map(e => `${e.stage}/e${e.epoch}`);
	          setDatasets(charts.s2, s2labels, [
	            solid("loss", "#cbd5e1", s2.map(e => e.loss ?? null)),
	            solid("loss_detect", "#60a5fa", s2.map(e => e.loss_detect ?? null)),
	            solid("loss_id", "#34d399", s2.map(e => e.loss_id ?? null)),
	            solid("loss_clean_detect", "#94a3b8", s2.map(e => e.loss_clean_detect ?? null)),
	            solid("loss_qual", "#fbbf24", s2.map(e => e.loss_qual ?? null)),
	            solid("loss_budget", "#ef4444", s2.map(e => e.loss_budget ?? null)),
	          ]);

	          // Budget chart
	          setDatasets(charts.budget, s2labels, [
	             solid("loss_budget", "#ef4444", s2.map(e => e.loss_budget ?? null)),
	             dashed("target (0.0)", "rgba(255,255,255,0.3)", s2labels.map(_ => 0.0))
	          ]);

		          const outR = await api(`/api/sessions/${id}/stdout?tail=120`);
		          document.getElementById("stdout").textContent = await outR.text();
		          const decR = await api(`/api/sessions/${id}/decode_report`);
		          document.getElementById("decode").textContent = (await decR.text()) || "—";
		          const repR = await api(`/api/sessions/${id}/report`);
		          document.getElementById("report").textContent = (await repR.text()) || "—";
		        } catch (e) {
		          document.getElementById("stdout").textContent = `Dashboard refresh error: ${e}`;
		        } finally {
	          refreshing = false;
	        }
	      }

      document.getElementById("newRunBtn").addEventListener("click", () => { showForm(true); updateFormVisibility(); });
      document.getElementById("refreshBtn").addEventListener("click", () => refreshSessions());
      document.getElementById("estimateBtn").addEventListener("click", () => estimate());
      document.getElementById("startBtn").addEventListener("click", () => startRun());
      document.getElementById("runRawBtn").addEventListener("click", () => runPasted());
      document.getElementById("attachBtn").addEventListener("click", () => attachLog());
      document.getElementById("stopBtn").addEventListener("click", () => stopSelected());
      document.getElementById("resumeBtn").addEventListener("click", () => resumeSelected());
      document.getElementById("kind").addEventListener("change", updateFormVisibility);

      refreshSessions();
      setInterval(() => { refreshSessions(); refreshSelected(); }, 1000);
    </script>
  </body>
</html>
        """

    @app.get("/api/sessions")
    def sessions():
        sessions = list(load_sessions().values())
        sessions.sort(key=lambda s: float(s.get("created_ts", 0.0)), reverse=True)
        # Compact list view
        out = []
        for s in sessions:
            out.append(
                {
                    "id": s.get("id"),
                    "kind": s.get("kind"),
                    "name": s.get("name"),
                    "status": s.get("status"),
                    "pid": s.get("pid"),
                    "created_ts": s.get("created_ts"),
                }
            )
        return out

    @app.get("/api/sessions/{sid}")
    def session(sid: str):
        sess = load_sessions().get(sid)
        if not sess:
            raise HTTPException(status_code=404, detail="session not found")
        # Update runtime status if needed.
        if sess.get("status") in {"running", "stopping"}:
            try_finalize(sess)
        return sess

    @app.post("/api/estimate")
    def estimate(payload: dict[str, Any]):
        kind = payload.get("kind")
        args = payload.get("args") or {}
        if kind != "quick_voice_smoke_train":
            return {}
        try:
            est = _estimate_seconds_for_quick_voice(
                runs_dir=runs_dir,
                num_clips=int(args.get("num_clips", 0)),
                epochs_s1=int(args.get("epochs_s1", 0)),
                epochs_s1b=int(args.get("epochs_s1b", 0)),
                epochs_s2=int(args.get("epochs_s2", 0)),
                epochs_s1b_post=int(args.get("epochs_s1b_post", 0)),
            )
            return est or {}
        except Exception:
            return {}

    def _extract_flag_value(cmd: list[str], flag: str) -> Optional[str]:
        try:
            idx = cmd.index(flag)
        except ValueError:
            idx = -1
        if idx >= 0 and (idx + 1) < len(cmd):
            return str(cmd[idx + 1])
        prefix = f"{flag}="
        for t in cmd:
            if isinstance(t, str) and t.startswith(prefix):
                return t.split("=", 1)[1]
        return None

    def _planned_epochs_from_meta_or_cmd(sess: dict[str, Any], meta: dict[str, Any]) -> dict[str, int]:
        cfg = meta.get("config") if isinstance(meta.get("config"), dict) else {}
        cmd = sess.get("cmd") if isinstance(sess.get("cmd"), list) else []
        # Multiclass training emits stages: s1, s2_encoder, s3_finetune
        # We fold legacy `epochs_s1b` into stage `s1` for backward-compatible configs.
        def _read_int(cfg_key: str, flag: str) -> int:
            v = cfg.get(cfg_key)
            if isinstance(v, (int, float)):
                return int(v)
            raw = _extract_flag_value(cmd, flag)
            if raw is None:
                return 0
            try:
                return int(float(raw))
            except Exception:
                return 0

        s1 = _read_int("epochs_s1", "--epochs_s1") + _read_int("epochs_s1b", "--epochs_s1b")
        s2 = _read_int("epochs_s2", "--epochs_s2")
        s3 = _read_int("epochs_s1b_post", "--epochs_s1b_post")
        return {"s1": int(s1), "s2_encoder": int(s2), "s3_finetune": int(s3)}

    def _num_clips_from_meta_or_cmd(sess: dict[str, Any], meta: dict[str, Any]) -> int:
        cfg = meta.get("config") if isinstance(meta.get("config"), dict) else {}
        v = cfg.get("num_clips")
        if isinstance(v, (int, float)) and int(v) > 0:
            return int(v)
        cmd = sess.get("cmd") if isinstance(sess.get("cmd"), list) else []
        raw = _extract_flag_value(cmd, "--num_clips")
        if raw is None:
            return 0
        try:
            return int(float(raw))
        except Exception:
            return 0

    @app.get("/api/sessions/{sid}/eta")
    def eta(sid: str):
        sess = load_sessions().get(sid)
        if not sess:
            raise HTTPException(status_code=404, detail="session not found")

        def _fmt_dur(sec: float) -> str:
            sec = max(0.0, float(sec))
            s = int(round(sec))
            h = s // 3600
            m = (s % 3600) // 60
            r = s % 60
            if h > 0:
                return f"{h}h {m}m"
            if m > 0:
                return f"{m}m {r}s"
            return f"{r}s"

        now = time.time()
        started = float(sess.get("started_ts") or sess.get("created_ts") or now)
        ended = float(sess.get("ended_ts") or now)
        is_done = bool(sess.get("ended_ts"))
        elapsed = max(0.0, (ended - started) if is_done else (now - started))

        mp = Path(sess.get("metrics_path", ""))
        meta = _read_first_meta(mp) or {}
        planned = _planned_epochs_from_meta_or_cmd(sess, meta)
        planned_total = sum(int(v) for v in planned.values() if int(v) > 0)

        current_stage = None
        current_epoch = None
        current_batch = None
        current_n_batches = None
        done_by_stage: dict[str, int] = {}
        done_by_stage_float: dict[str, float] = {}
        if mp.exists():
            events = _read_tail_jsonl(mp, max_lines=20000)

            def _ts(ev: dict[str, Any]) -> float:
                try:
                    return float(ev.get("ts") or 0.0)
                except Exception:
                    return 0.0

            epochs = [
                e
                for e in events
                if e.get("type") == "epoch"
                and isinstance(e.get("stage"), str)
                and isinstance(e.get("epoch"), (int, float))
            ]
            epochs.sort(key=_ts)
            for e in epochs:
                st = str(e.get("stage"))
                ep = int(float(e.get("epoch")))
                done_by_stage[st] = max(done_by_stage.get(st, 0), ep)

            for st, ep in done_by_stage.items():
                done_by_stage_float[st] = float(ep)

            steps = [
                e
                for e in events
                if e.get("type") == "step"
                and isinstance(e.get("stage"), str)
                and isinstance(e.get("epoch"), (int, float))
                and isinstance(e.get("batch"), (int, float))
            ]
            steps.sort(key=_ts)

            last_epoch = epochs[-1] if epochs else None
            last_step = steps[-1] if steps else None

            use_step = False
            if last_step is not None and (last_epoch is None or _ts(last_step) > _ts(last_epoch)):
                use_step = True

            if use_step:
                st = str(last_step.get("stage"))
                ep = int(float(last_step.get("epoch")))
                b = int(float(last_step.get("batch")))
                nb = last_step.get("n_batches")
                nb_i = int(nb) if isinstance(nb, (int, float)) and int(nb) > 0 else None
                current_stage = st
                current_epoch = ep
                current_batch = b
                current_n_batches = nb_i

                # Compute fractional epoch progress if we know the batch count.
                completed = int(done_by_stage.get(st, 0))
                base = completed
                if ep > completed:
                    base = ep - 1
                frac = 0.0
                if nb_i is not None and nb_i > 0:
                    frac = min(1.0, max(0.0, float(b + 1) / float(nb_i)))
                done_by_stage_float[st] = max(done_by_stage_float.get(st, float(completed)), float(base) + frac)
            elif last_epoch is not None:
                current_stage = str(last_epoch.get("stage"))
                current_epoch = int(float(last_epoch.get("epoch")))

        stage_order = ["s1", "s2_encoder", "s3_finetune"]
        done_epochs = 0.0
        for st in stage_order:
            pe = int(planned.get(st, 0))
            if pe <= 0:
                continue
            de = float(done_by_stage_float.get(st, float(done_by_stage.get(st, 0))))
            done_epochs += min(float(pe), max(0.0, de))
        progress = (float(done_epochs) / float(planned_total)) if planned_total > 0 else 0.0

        est_total = None
        est_remaining = None
        eta_text = f"Elapsed: {_fmt_dur(elapsed)}"
        if sess.get("kind") == "quick_voice_smoke_train" and planned_total > 0:
            nclips = _num_clips_from_meta_or_cmd(sess, meta)
            try:
                est = _estimate_seconds_for_quick_voice(
                    runs_dir=runs_dir,
                    num_clips=int(nclips),
                    epochs_s1=int(planned.get("s1", 0)),
                    epochs_s1b=0,
                    epochs_s2=int(planned.get("s2_encoder", 0)),
                    epochs_s1b_post=int(planned.get("s3_finetune", 0)),
                )
                if est and isinstance(est.get("total_seconds"), (int, float)):
                    est_total = float(est["total_seconds"])
                    est_remaining = max(0.0, est_total - elapsed)
            except Exception:
                pass

        if est_total is not None:
            eta_text = f"ETA remaining: ~{_fmt_dur(est_remaining)} (est total ~{_fmt_dur(est_total)}) · elapsed {_fmt_dur(elapsed)}"
        else:
            eta_text = f"Elapsed: {_fmt_dur(elapsed)}"

        if is_done:
            eta_text = f"Finished in {_fmt_dur(elapsed)}"

        if current_stage and current_epoch is not None and planned_total > 0:
            if current_batch is not None and current_n_batches is not None:
                prog_text = (
                    f"Progress: {current_stage} e{current_epoch} b{current_batch + 1}/{current_n_batches}"
                    f" · epochs {done_epochs:.2f}/{planned_total} ({progress*100:.1f}%)"
                )
            else:
                prog_text = f"Progress: {current_stage} e{current_epoch} · epochs {done_epochs:.2f}/{planned_total} ({progress*100:.1f}%)"
        elif planned_total > 0:
            prog_text = f"Progress: epochs {done_epochs:.2f}/{planned_total} ({progress*100:.1f}%)"
        else:
            prog_text = "Progress: —"

        return {
            "elapsed_seconds": elapsed,
            "estimated_total_seconds": est_total,
            "estimated_remaining_seconds": est_remaining,
            "progress_frac": progress,
            "current_stage": current_stage,
            "current_epoch": current_epoch,
            "done_epochs_by_stage": done_by_stage,
            "planned_epochs_by_stage": planned,
            "progress_text": prog_text,
            "eta_text": eta_text,
        }

    @app.post("/api/sessions")
    def create_session(payload: dict[str, Any]):
        kind = payload.get("kind")
        name = payload.get("name")
        args = payload.get("args") or {}
        if kind not in {"quick_voice_smoke_train", "train_full"}:
            raise HTTPException(status_code=400, detail="unsupported kind")

        sid = f"{int(time.time())}_{secrets.token_hex(3)}"
        sdir = (runs_dir / sid).resolve()
        sdir.mkdir(parents=True, exist_ok=True)
        metrics_path = sdir / "metrics.jsonl"
        stdout_path = sdir / "stdout.log"

        if kind == "quick_voice_smoke_train":
            cmd = [
                _preferred_python_executable(),
                "-m",
                "watermark.scripts.quick_voice_smoke_train",
                "--source_dir",
                str(args.get("source_dir", "mini_benchmark_data")),
                "--num_clips",
                str(int(args.get("num_clips", 512))),
                "--seed",
                str(int(args.get("seed", 1337))),
                "--epochs_s1",
                str(int(args.get("epochs_s1", 6))),
                "--epochs_s1b",
                str(int(args.get("epochs_s1b", 1))),
                "--epochs_s2",
                str(int(args.get("epochs_s2", 12))),
                "--epochs_s1b_post",
                str(int(args.get("epochs_s1b_post", 12))),
                "--msg_weight",
                str(float(args.get("msg_weight", 1.0))),
                "--model_ce_weight",
                str(float(args.get("model_ce_weight", 4.0))),
                "--version_ce_weight",
                str(float(args.get("version_ce_weight", 2.0))),
                "--pair_ce_weight",
                str(float(args.get("pair_ce_weight", 4.0))),
                "--unknown_ce_weight",
                str(float(args.get("unknown_ce_weight", 1.0))),
                "--neg_weight",
                str(float(args.get("neg_weight", 5.0))),
                "--neg_preamble_target",
                str(float(args.get("neg_preamble_target", 0.5))),
                "--reverb_prob",
                str(float(args.get("reverb_prob", 0.0))),
                "--probe_clips",
                str(int(args.get("probe_clips", 1024))),
                "--probe_every",
                str(int(args.get("probe_every", 1))),
                "--probe_reverb_every",
                str(int(args.get("probe_reverb_every", 999999))),
                "--detect_weight",
                str(float(args.get("detect_weight", 1.0))),
                "--id_weight",
                str(float(args.get("id_weight", 2.0))),
                "--log_steps_every",
                str(int(args.get("log_steps_every", 25))),
                "--log_metrics",
                str(metrics_path),
                "--out",
                str(sdir),
            ]
            manifest = str(args.get("manifest") or "").strip()
            if manifest:
                cmd.extend(["--manifest", manifest])
            load_encoder = str(args.get("load_encoder") or "").strip()
            if load_encoder:
                cmd.extend(["--load_encoder", load_encoder])
            load_decoder = str(args.get("load_decoder") or "").strip()
            if load_decoder:
                cmd.extend(["--load_decoder", load_decoder])
            if bool(args.get("freeze_detect_head_in_s3")):
                cmd.append("--freeze_detect_head_in_s3")
            if bool(args.get("stage2_payload_on_all")):
                cmd.append("--stage2_payload_on_all")
            extra = args.get("extra_args")
            if isinstance(extra, str) and extra.strip():
                cmd.extend(shlex.split(extra))
        else:
            manifest = str(args.get("manifest", "")).strip()
            if not manifest:
                raise HTTPException(status_code=400, detail="train_full requires manifest")
            cmd = [
                _preferred_python_executable(),
                "-m",
                "watermark.scripts.train_full",
                "--manifest",
                manifest,
                "--output",
                str(sdir),
                "--epochs_s1",
                str(int(args.get("epochs_s1", 20))),
                "--epochs_s1b",
                str(int(args.get("epochs_s1b", 10))),
                "--warmup_s1b",
                str(int(args.get("warmup_s1b", 3))),
                "--epochs_s2",
                str(int(args.get("epochs_s2", 20))),
                "--epochs_s1b_post",
                str(int(args.get("epochs_s1b_post", 0))),
                "--neg_weight",
                str(float(args.get("neg_weight", 0.4))),
                "--neg_preamble_target",
                str(float(args.get("neg_preamble_target", 0.5))),
                "--unknown_ce_weight",
                str(float(args.get("unknown_ce_weight", 0.0))),
                "--model_ce_weight",
                str(float(args.get("model_ce_weight", 1.0))),
                "--version_ce_weight",
                str(float(args.get("version_ce_weight", 1.0))),
                "--pair_ce_weight",
                str(float(args.get("pair_ce_weight", 2.0))),
                "--msg_weight",
                str(float(args.get("msg_weight", 1.0))),
                "--reverb_prob",
                str(float(args.get("reverb_prob", 0.25))),
                "--detect_weight",
                str(float(args.get("detect_weight", 1.0))),
                "--id_weight",
                str(float(args.get("id_weight", 2.0))),
                "--probe_clips",
                str(int(args.get("probe_clips", 1024))),
                "--probe_every",
                str(int(args.get("probe_every", 1))),
                "--probe_reverb_every",
                "999999",
                "--log_steps_every",
                str(int(args.get("log_steps_every", 100))),
                "--log_metrics",
                str(metrics_path),
            ]
            load_encoder = str(args.get("load_encoder") or "").strip()
            if load_encoder:
                cmd.extend(["--load_encoder", load_encoder])
            load_decoder = str(args.get("load_decoder") or "").strip()
            if load_decoder:
                cmd.extend(["--load_decoder", load_decoder])
            if bool(args.get("freeze_detect_head_in_s3")):
                cmd.append("--freeze_detect_head_in_s3")
            if bool(args.get("stage2_payload_on_all")):
                cmd.append("--stage2_payload_on_all")
            extra = args.get("extra_args")
            if isinstance(extra, str) and extra.strip():
                cmd.extend(shlex.split(extra))

        sess = {
            "id": sid,
            "kind": kind,
            "name": name,
            "created_ts": time.time(),
            "status": "created",
            "run_dir": str(sdir),
            "metrics_path": str(metrics_path),
            "stdout_path": str(stdout_path),
            "cmd": cmd,
            "pid": None,
            "returncode": None,
        }
        save_session(sess)
        try:
            start_process(sess)
        except Exception as e:
            sess["status"] = "failed"
            sess["error"] = str(e)
            save_session(sess)
            raise HTTPException(status_code=500, detail=str(e))
        return {"id": sid}

    @app.post("/api/sessions/raw")
    def create_session_raw(payload: dict[str, Any]):
        name = payload.get("name")
        cmd_text = payload.get("command")
        if not isinstance(cmd_text, str) or not cmd_text.strip():
            raise HTTPException(status_code=400, detail="missing command")

        try:
            spec = _parse_pasted_command(cmd_text)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

        sid = f"{int(time.time())}_{secrets.token_hex(3)}"
        sdir = (runs_dir / sid).resolve()
        sdir.mkdir(parents=True, exist_ok=True)
        metrics_path = sdir / "metrics.jsonl"
        stdout_path = sdir / "stdout.log"

        kind = spec["kind"]
        module = spec["module"]
        args = list(spec["args"])

        if kind == "quick_voice_smoke_train":
            # Force controller-owned paths.
            args.extend(["--log_metrics", str(metrics_path), "--out", str(sdir)])
        else:
            args.extend(["--log_metrics", str(metrics_path), "--output", str(sdir)])

        cmd = [os.fspath(Path(sys.executable).resolve()), "-m", module, *args]
        cmd = [_preferred_python_executable(), "-m", module, *args]

        sess = {
            "id": sid,
            "kind": kind,
            "name": name,
            "created_ts": time.time(),
            "status": "created",
            "run_dir": str(sdir),
            "metrics_path": str(metrics_path),
            "stdout_path": str(stdout_path),
            "cmd": cmd,
            "cmd_raw": cmd_text,
            "pid": None,
            "returncode": None,
        }
        save_session(sess)
        try:
            start_process(sess)
        except Exception as e:
            sess["status"] = "failed"
            sess["error"] = str(e)
            save_session(sess)
            raise HTTPException(status_code=500, detail=str(e))
        return {"id": sid}

    @app.post("/api/sessions/attach")
    def attach_session(payload: dict[str, Any]):
        name = payload.get("name")
        metrics_path = payload.get("metrics_path")
        if not isinstance(metrics_path, str) or not metrics_path.strip():
            raise HTTPException(status_code=400, detail="missing metrics_path")

        mp = Path(metrics_path.strip()).expanduser()
        if not mp.is_absolute():
            mp = (REPO_ROOT / mp).resolve()
        if not mp.exists():
            raise HTTPException(status_code=404, detail=f"metrics.jsonl not found: {mp}")

        meta = _read_first_meta(mp) or {}
        run_name = meta.get("run_name")
        kind = "attached"
        if run_name == "quick_voice_smoke_train":
            kind = "quick_voice_smoke_train"
        elif run_name == "train_full":
            kind = "train_full"

        run_dir_raw = payload.get("run_dir")
        if isinstance(run_dir_raw, str) and run_dir_raw.strip():
            rd = Path(run_dir_raw.strip()).expanduser()
            if not rd.is_absolute():
                rd = (REPO_ROOT / rd).resolve()
            run_dir = rd
        else:
            run_dir = mp.parent

        sid = f"{int(time.time())}_{secrets.token_hex(3)}"
        sdir = (runs_dir / sid).resolve()
        sdir.mkdir(parents=True, exist_ok=True)

        sess = {
            "id": sid,
            "kind": kind,
            "name": name,
            "created_ts": time.time(),
            "status": "external",
            "run_dir": str(run_dir),
            "metrics_path": str(mp),
            "stdout_path": str((run_dir / "stdout.log").resolve()),
            "cmd": [],
            "pid": None,
            "returncode": None,
        }
        save_session(sess)
        return {"id": sid}

    @app.post("/api/sessions/{sid}/stop")
    def stop(sid: str):
        sess = load_sessions().get(sid)
        if not sess:
            raise HTTPException(status_code=404, detail="session not found")
        pid = sess.get("pid")
        if sess.get("status") not in {"running", "stopping"} or not isinstance(pid, int):
            return {"ok": True}
        sess["status"] = "stopping"
        save_session(sess)
        try:
            mp = Path(sess.get("metrics_path", ""))
            if mp:
                mp.parent.mkdir(parents=True, exist_ok=True)
                with mp.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"type": "control", "action": "stop_requested", "ts": time.time()}, ensure_ascii=False) + "\n")
        except Exception:
            pass
        with lock:
            p = procs.get(sid)
        if p:
            try:
                # Prefer Ctrl-C semantics for Python training loops.
                try:
                    os.killpg(int(p.pid), signal.SIGINT)
                except Exception:
                    p.send_signal(signal.SIGINT)
            except Exception:
                pass
            # Escalate if it doesn't exit soon.
            def _escalate() -> None:
                time.sleep(10.0)
                with lock:
                    pp = procs.get(sid)
                if not pp:
                    return
                try:
                    if pp.poll() is None:
                        try:
                            os.killpg(int(pp.pid), signal.SIGKILL)
                        except Exception:
                            pp.kill()
                except Exception:
                    pass

            threading.Thread(target=_escalate, daemon=True).start()
        return {"ok": True}

    @app.post("/api/sessions/{sid}/resume")
    def resume(sid: str):
        sess = load_sessions().get(sid)
        if not sess:
            raise HTTPException(status_code=404, detail="session not found")
        
        # Determine checkpoint path
        rd = Path(sess["run_dir"])
        ckpt = rd / "checkpoints" / "last.pt"
        if not ckpt.exists():
            raise HTTPException(status_code=400, detail=f"No checkpoint found at {ckpt}")
            
        # Construct new command
        old_cmd = sess.get("cmd") or []
        if not old_cmd:
            raise HTTPException(status_code=400, detail="Original command not found")
            
        new_cmd = []
        skip_next = False
        # Copy args but filter out old resume
        for i, token in enumerate(old_cmd):
            if skip_next:
                skip_next = False
                continue
            if token == "--resume":
                skip_next = True
                continue
            if token.startswith("--resume="):
                continue
            new_cmd.append(token)
            
        # Verify the command is for our known scripts (sanity check)
        is_valid = False
        for token in new_cmd:
            if "watermark.scripts" in token:
                is_valid = True
                break
        if not is_valid:
             raise HTTPException(status_code=400, detail="Cannot resume unidentified script type")

        # Append new resume flag
        new_cmd.extend(["--resume", str(ckpt.resolve())])
        
        # Create new session
        new_sid = f"{int(time.time())}_{secrets.token_hex(3)}"
        new_sdir = (runs_dir / new_sid).resolve()
        new_sdir.mkdir(parents=True, exist_ok=True)
        metrics_path = new_sdir / "metrics.jsonl"
        stdout_path = new_sdir / "stdout.log"
        
        # We need to update the output paths in the command if they were explicit
        # But _parse_pasted_command logic in create_session usually handles this by appending.
        # Here we are working with the raw list.
        # Strategy: filter out --out / --output / --log_metrics and re-append correctly
        final_cmd = []
        skip_next = False
        for token in new_cmd:
            if skip_next:
                skip_next = False
                continue
            if token in {"--out", "--output", "--log_metrics"}:
                skip_next = True
                continue
            if any(token.startswith(p) for p in ["--out=", "--output=", "--log_metrics="]):
                continue
            final_cmd.append(token)
            
        # Re-add our managed paths
        # Detect kind to know which flag to use
        kind = sess.get("kind", "unknown")
        if kind == "quick_voice_smoke_train":
             final_cmd.extend(["--log_metrics", str(metrics_path), "--out", str(new_sdir)])
        else:
             final_cmd.extend(["--log_metrics", str(metrics_path), "--output", str(new_sdir)])

        new_sess = {
            "id": new_sid,
            "kind": kind,
            "name": f"Resume {sess.get('name') or sid}",
            "created_ts": time.time(),
            "status": "created",
            "run_dir": str(new_sdir),
            "metrics_path": str(metrics_path),
            "stdout_path": str(stdout_path),
            "cmd": final_cmd,
            "pid": None,
            "returncode": None,
        }
        save_session(new_sess)
        try:
            start_process(new_sess)
        except Exception as e:
            new_sess["status"] = "failed"
            new_sess["error"] = str(e)
            save_session(new_sess)
            raise HTTPException(status_code=500, detail=str(e))
            
        return {"id": new_sid}

    @app.get("/api/sessions/{sid}/metrics")
    def metrics(sid: str, tail: int = Query(5000, ge=200, le=200000)):
        sess = load_sessions().get(sid)
        if not sess:
            raise HTTPException(status_code=404, detail="session not found")
        mp = Path(sess["metrics_path"])
        events = _read_tail_jsonl(mp, max_lines=tail)
        if not any(e.get("type") == "meta" for e in events):
            meta = _read_first_meta(mp)
            if meta is not None:
                events = [meta, *events]
        return events

    @app.get("/api/sessions/{sid}/download/metrics")
    def download_metrics(sid: str):
        sess = load_sessions().get(sid)
        if not sess:
            raise HTTPException(status_code=404, detail="session not found")
        mp = Path(sess["metrics_path"])
        if not mp.exists():
            return PlainTextResponse("metrics.jsonl not found", status_code=404)
        return FileResponse(mp, filename=mp.name)

    @app.get("/api/sessions/{sid}/stdout")
    def stdout(sid: str, tail: int = Query(200, ge=20, le=5000)):
        sess = load_sessions().get(sid)
        if not sess:
            raise HTTPException(status_code=404, detail="session not found")
        sp = Path(sess["stdout_path"])
        return PlainTextResponse(_tail_text(sp, max_lines=tail), status_code=200)

    @app.get("/api/sessions/{sid}/report")
    def report(sid: str):
        sess = load_sessions().get(sid)
        if not sess:
            raise HTTPException(status_code=404, detail="session not found")
        sdir = Path(sess["run_dir"])
        rp = sdir / "report.md"
        if not rp.exists():
            try:
                mp = Path(sess.get("metrics_path", ""))
                events = _read_tail_jsonl(mp, max_lines=200000)
                if not any(e.get("type") == "meta" for e in events):
                    meta = _read_first_meta(mp)
                    if meta is not None:
                        events = [meta, *events]
                summ = _summarize_metrics(events)
                text = _render_run_report_md(sess=sess, summary=summ)
                try:
                    rp.write_text(text, encoding="utf-8")
                except Exception:
                    pass
                return PlainTextResponse(text, status_code=200)
            except Exception:
                return PlainTextResponse("", status_code=200)
        try:
            return PlainTextResponse(rp.read_text(encoding="utf-8"), status_code=200)
        except Exception:
            return PlainTextResponse("", status_code=200)

    @app.get("/api/sessions/{sid}/decode_report")
    def decode_report(sid: str):
        sess = load_sessions().get(sid)
        if not sess:
            raise HTTPException(status_code=404, detail="session not found")
        sdir = Path(sess["run_dir"])
        report = sdir / "audio" / "decode_report.txt"
        if not report.exists():
            return PlainTextResponse("", status_code=200)
        try:
            return PlainTextResponse(report.read_text(encoding="utf-8"), status_code=200)
        except Exception:
            return PlainTextResponse("", status_code=200)

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="Live dashboard for watermark metrics.jsonl")
    parser.add_argument("--log", type=str, default=None, help="Path to metrics.jsonl (single-log mode)")
    parser.add_argument("--metrics", dest="log", type=str, default=None, help="Alias for --log")
    parser.add_argument(
        "--runs_dir",
        type=str,
        default="outputs/dashboard_runs",
        help="Directory to store/manage sessions (controller mode)",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8765, help="Bind port")
    args = parser.parse_args()

    if args.log:
        log_path = Path(args.log)
        app = _make_single_log_app(log_path)
        print(f"[Dashboard] Mode: single-log")
        print(f"[Dashboard] Log: {log_path}")
    else:
        runs_dir = (REPO_ROOT / Path(args.runs_dir)).resolve()
        app = _make_controller_app(runs_dir)
        print(f"[Dashboard] Mode: controller")
        print(f"[Dashboard] Runs: {runs_dir}")

    try:
        import uvicorn
    except Exception:
        print("uvicorn not installed. Install `uvicorn[standard]` and retry.")
        return 1

    print(f"[Dashboard] Open: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
