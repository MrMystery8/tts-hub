# FYP Final Report — Project Handoff Context

**Project:** "Designing A Local First Voice Preservation System With Audio Watermark Detection For Assistive Communication" (title LOCKED from IR — do not change)
**Student:** Ayaan Minhas, TP077859, APD3F2511CS(AI) — BSc Computer Science (Hons) (Artificial Intelligence), APU
**Supervisor:** Mr. Justin Gilbert · **2nd Marker:** Dr. Murugananthan Velayutham
**SDG:** Goal 3 (Good Health and Well-Being)

## Deliverable file (edit THIS one)
`/Users/ayaanminhas/Desktop/Personal_Work/tts-hub/Ayaan Minhas-TP077859-APD3F2511CS(AI)-FYP Final Report (DRAFT).docx`
- **~191 pages** as of the latest edit. Built by copying the IR docx and editing it (inherits styles: Times New Roman 12; Heading 1–4; "Table Grid"; "Caption"; "table of figures" for the lists).
- Diagrams live in the working outputs folder under `dg/` (Graphviz sources + PNGs).

## What the system actually is (for factual accuracy)
TTS Hub is a local-first, offline-by-default voice cloning system on Apple Silicon (MacBook Air M4, 24 GB).
- FastAPI backend (`webui.py`) + static web UI; a **HubManager** runs **one isolated worker subprocess per model** over newline-delimited JSON stdio.
- **Three frozen backends:** IndexTTS2 (PyTorch/MPS), Chatterbox Multilingual (PyTorch/MPS), Qwen3-TTS (MLX).
- **Multiclass watermark:** encoder embeds bounded `x_wm = x + alpha*tanh(delta)`; decoder detects + attributes to source model; (K+1) classes, K=3.
- **Companion Mobile PWA** at `/mobile` (phone→laptop delegation over a private Tailscale network — still local-first, no cloud) + opt-in onboarding tours.

## Key REAL results in the report (watermark "Run B")
- Trained on LibriSpeech-derived `medium_benchmark_data`; 5,120 clips (4,096 train / 512 val / 512 test); Stage-1 only; 100 epochs; detect_weight=12 / id_weight=2.
- Held-out clean test: detection **AUC 0.988, TPR@1%FPR 0.795, attribution acc 0.875**, composite 0.696.
- Tiered robustness: T1 AUC 0.986 / attr 0.860 · T2 AUC 0.974 / attr 0.482 · T3 AUC 0.882 / attr 0.366.
- Perceptual: **PESQ 4.64, STOI 1.00, SNR 21.5 dB** (watermark effectively transparent).
- Unit/integration tests are real (pytest + Playwright, all pass).

### Run B provenance (confirmed from run logs)
Run B = `outputs/dashboard_runs/sweep3_B_static_12_2` (config.json + metrics.jsonl + tiered_eval + quality_eval on disk).
Siblings `sweep3_A_static_8_3` and `sweep3_C_adaptive` are the controlled ablation arms — identical data/splits/epochs/seed, only the loss-weighting differs. `composite_score = TPR@1%FPR × id_acc_pos` (0.795 × 0.875 = 0.696 ✓).
~113 logged runs in `outputs/dashboard_runs/`, each with `session.json` (exact command line) — the tuning-history evidence base behind 4.6.6.

## Report structure as built (current)
- **Ch1–3:** carried from IR and improved. Ch1 has 1.6.1 Tangible / 1.6.2 Intangible benefits, 1.7 Overview, 1.8 Project Plan.
- **Ch4:** 4.1 Intro · 4.2 System Design · 4.3 Architecture · 4.4 Design Diagrams (use case + **6 specs**; 5 activity; sequence; class [NOT ERD — NoSQL/local JSON]; 2 DFDs) · 4.5 Interface Design (+4.5.1 Mobile; screenshot placeholders Figs 4.12–4.20) · **4.6 Model Implementation (4.6.1–4.6.6, incl. tuning history + rejected codebook alternative)** · 4.7 Key Features (6) · 4.8 Code Implementation (deliberately minimal) · 4.9 Summary.
- **Ch5:** Test Plan (unit 5.1a–f; integration; **system + Table 5.3**; UAT design) · Model Evaluation (cross-backend — cells empty) · Watermark Evaluation (real Run B numbers) · UAT Results scaffold · Discussion **(+ 5.6.1 literature comparison)** · Summary.
- **Ch6:** 6.1 Critical Evaluation (+ Table 6.1, 6.1.1 Strengths, 6.1.2 Contribution) · 6.2 Limitations · 6.3 Recommendations.
- Appendices A, B, C, D1, D2, E, F, G, H, I, J (ordering fixed).

**Current table numbering (post-renumber — use these):** Ch4 → 4.1–4.3 UC specs (original), 4.4–4.6 UC specs (new), 4.7 backends, 4.8 backend controls, 4.9 preprocessing, 4.10 hyperparameters, 4.11 tuning phases, 4.12 codebook perceptual cost, 4.13 ablation. Figures → 4.1–4.11 diagrams, 4.12–4.13 desktop screens, **4.14 reference-intake panel**, 4.15–4.17 desktop screens, **4.18–4.20 guided-tour steps**, 4.21–4.24 mobile, **4.25 codebook spectrograms**. Sequence 4.12–4.25 contiguous; all in-text refs resolve. Ch5 → 5.1a–f unit, 5.2 integration, **5.3 system**, 5.4 UAT profile, 5.5a–f UAT instrument, 5.6 cross-backend, 5.7 detection, 5.8 tiered, 5.9 perceptual, 5.10 UAT respondents, 5.11 UAT ratings, **5.12 capability positioning, 5.13 tier mapping**. Ch6 → 6.1 objectives.

## Diagrams (Graphviz; SVG for sequence) — Figs 4.1–4.11
arch · usecase · 5 activity · sequence · classes · dfd0 · dfd1.

---

# TODO — what's left

## A. Blocked on your data/assets (cannot be written for you)
- [x] ~~**UI screenshots**~~ — **DONE by you** (9 screens, ~120–140 words of prose each), plus 4 figures added from live captures: reference-intake modes (4.14) and three guided-tour steps (4.18–4.20). Section 4.5 now has no unshown claimed behaviour.
- [ ] **Cross-backend benchmark numbers** — **Table 5.6**, all cells empty. This is the CSAI model-comparison table the briefing explicitly asks for.
- [ ] **UAT sessions + results** — **Tables 5.10 / 5.11**, Appendix H. 63 empty `[ ]` cells across the instrument.
- [ ] **Semester-2 Gantt** (Appendix D2) · **poster** (Appendix I) · **Turnitin report** (Appendix F).
- [ ] **Front-matter forms — absent entirely, not merely unfilled:** Declaration of Confidentiality, Library form, 6 log sheets.

## B. Gaps found by comparing against the CSAI sample report — CLOSED
Sample = "An Online Marketplace for Surplus Food" (189 pp). All four items resolved: one fixed, three deliberately declined with reasons. **Nothing outstanding in this section.**

- [x] ~~**Appendix ordering**~~ — **DONE.** "Appendix D (Update)" had been sitting after Appendix G. Renamed to **D1 / D2** and moved into place; sequence now reads A, B, C, D1, D2, E, F, G, H, I, J. Pure move, no content changed.
- [ ] ~~**Data Dictionary**~~ — **decided against, deliberately.** Initially recommended because the sample has one (its 4.3.2). On inspection that was over-weighting the sample: it is a CRUD marketplace whose data model *is* the system (users, listings, orders, carts, reviews), whereas this project persists **one document type** — `outputs/voices/<id>/meta.json` (id, name, created_at, prompt_text, prompt_text_source, audio{duration_s, path, sr, sha256, profile{rms, peak_abs, clipped_samples, clipped_ratio}}, caches{<backend>{source_sha256, model_fingerprint, …}}). The briefing's actual data-modelling requirement was "class diagram, NOT ERD", which **4.4.5 already satisfies** with `«local NoSQL store»` / `«document»` stereotypes and a justification for rejecting an ERD. A generic field dictionary would be padding — the same objection raised against the sample's "Execution" section.
  - **If it is ever revisited**, the version worth writing is narrow: a 6–8 row table of the voice-record document added to **4.4.5**, justified by the `caches.source_sha256` + `model_fingerprint` **cache-invalidation and integrity** mechanism. That behaviour is claimed in 4.6.3 and in UC-10's postcondition but is not currently evidenced anywhere.
- [ ] ~~Sample 4.5 "Execution" (6 pp of library-install / config / API-wiring screenshots)~~ — **deliberately skipped.** It is padding; our 4.8 is intentionally minimal and 4.6 carries the technical weight.
- [ ] ~~Sample 1.6.3 Target User~~ — **not a gap.** Already covered by **1.5.4 Target Users**; the sample just files it under Potential Benefits instead of Scope. Do not add a duplicate.

### Where we already exceed the sample (do NOT over-correct toward it)
- **Ch5**: sample has only unit testing + UAT. We have unit, integration, system, UAT design, model evaluation, watermark evaluation, discussion.
- **Ch4 model depth**: the sample has **no model implementation section at all** — its CV algorithm appears only in the literature review and in screenshots, never as a training regime. Our 4.6 (~1,800w, 5 tables, documented tuning history, controlled ablation) has no counterpart. This is the CSAI differentiator.
- **Ch6**: sample is 2 pages; ours is ~4 with the objective-evidence table.

## C. End-stage mechanics
- [ ] **Refresh TOC / List of Figures / List of Tables fields in Word.** Now substantially stale: the figure renumbering (4.14–4.25) and the new tables mean most entries and every page number are out of date. Report is now ~191 pp. Do this LAST.
- [ ] **Commit the untracked work.** `docs/WATERMARK_FINAL_FINDINGS.md`, `docs/DETERMINISTIC_CODEBOOK_FINDINGS.md`, `docs/DETERMINISTIC_CODEBOOK_PLAN.md`, `report_assets/`, and the .docx itself are all untracked. The two findings docs were recovered from git history after being deleted — don't lose them again.
- [ ] Optional: trim UAT (~26 rating items) if too long for fatigue-prone testers.
- [ ] Optional: produce a **redlined copy** (tracked changes) of this session's edits for the supervisor — see tooling note below.

---

## Marking guidance (teacher Briefing 3 — CSAI)
- Abstract must state ACHIEVED results (done).
- Ch1–3 improved; markers compare against IR comments.
- **Class diagram, NOT ERD** for NoSQL (done).
- **UAT participants MUST match target users** (voice-loss users, caregiver, speech/AAC expert).
- **CSAI needs a model-comparison table as the AI contribution** — present as Table 5.6; **still needs numbers**.
- Keep AI-generated content ≤30% (rewrite in own academic voice).
- 3 supervisor meetings + log sheets; A/A+ needs evidence + expert/user validation.

## Tooling notes for whoever edits next
- Edits so far were made with **python-docx** (good for styled table/paragraph insertion). Verify two ways: the docx skill's `scripts/office/validate.py --original <backup>` **and** a LibreOffice PDF render you actually look at.
- ⚠️ **The front matter contains List of Tables / List of Figures entries whose text is identical to real body captions** (style `table of figures`). A naive `find first paragraph starting with "Table 4.11:"` matches the *front-matter* entry, and a whole block of content silently lands in the front matter. This happened once this session. **Any body-paragraph lookup must exclude styles `table of figures`, `toc 1/2/3`** — see `find_para()` in the session's `final_pass.py` (copied to `report_assets/codebook_ab/`).
- ⚠️ **`validate.py` will NOT catch misplaced content** — misplacement is schema-valid. It passed the front-matter bug cleanly. Only a render check or a paragraph-index check catches that class of error.
- `validate.py` needs `defusedxml` (installed into `/opt/homebrew/bin/python3`).
- python-docx **cannot** produce tracked changes. For a redlined supervisor copy, use the docx skill's unzip → edit `word/document.xml` → rezip route with `<w:ins>`/`<w:del>`, then `validate.py --author`.
- Caution: `paragraph.text = "..."` flattens a paragraph's runs into one. Fine for plain body text (all that was touched), but it would destroy mixed bold/italic formatting.
- **Renumbering gotcha:** a plain `Table N.M` regex misses plural forms ("Tables 5.5a to 5.5f"). Sweep for `Tables ` separately — this was missed once and caught on render.
- **Honour forward references.** Twice this session a new passage promised something "discussed in Chapter 6" that Chapter 6 did not yet contain. After adding any cross-chapter promise, grep the target section.
- Backups of each edit stage are in the session scratchpad (`backup_before_46.docx`, `backup_before_442.docx`, `backup_before_ch6.docx`, `backup_before_final.docx`) — **temporary; copy them somewhere durable if you want them.**

## Known quirk
LibreOffice PDF export mis-renders some wide tables (crushes narrow columns) even though the XML is correct for Word. All new tables were kept to 4–5 columns for this reason.

---

## Change log — session of 2026-07-18
**Ch4:**
- 4.4.2: 3 → 6 use case specifications (UC-05 backend comparison, UC-10 delete voice + local data, UC-12 mobile delegation). Lead-in rewritten to justify six of twelve.
- 4.6: ~710w → ~1,800w + 5 tables. Dataset provenance paragraph; Table 4.8 per-backend controls/defaults read from the worker adapters; 4.6.3 rewritten + Table 4.9 (6-stage preprocessing); **new 4.6.5** training regime + Table 4.10 (20-row hyperparameters); **new 4.6.6** tuning process + Table 4.11 (6 phases) + Table 4.12 (A/B/C ablation) + limitations paragraph.
- Corrected a factual error: 4.6.3 claimed the quality gate checks *silence fraction*; `_wav_profile` in `hub/voice_library.py:39` actually computes peak_abs, rms, clipped_samples, clipped_ratio.
- Renumbered old Tables 4.4–4.9 → 4.7–4.12.

**Ch5:**
- 5.2.3: 79w → ~120w + **Table 5.3** (12-step end-to-end scenario S1–S12). Added a clean-audio control step (S9) so the detector isn't only shown succeeding on positives. Prose names the hardware and the "all interfaces disabled" condition.
- Renumbered 5.3→5.4, 5.4a–f→5.5a–f, 5.5→5.6 … 5.10→5.11, including plural refs ("Tables 5.5a to 5.5f").

**Ch6:** ~865w → ~1,600w.
- Fixed a real contradiction: para 1 claimed "all four objectives were met" while para 3 said Objective 4 was only partial. Now consistent.
- New **Table 6.1** (objective → status → evidence). Contributions paragraph rewritten around the tuning process and ablation. New **6.1.1 Strengths** and **6.1.2 Contribution to community and industry**. 6.2 gained the single-seed / one-factor limitation promised by the forward reference in 4.6.6.

---

## Change log — correction + literature-comparison pass (same session)

### Eight factual errors corrected in Ch4 (found on self-audit; all had been introduced earlier that session)
1. **Phase 5 conflation.** "Attribution improved to 0.87 on the largest run" merged two different experiments — the 0.87 run is `1770060443_21312a`, mini corpus at **K=2**; the medium-corpus 20k run reached **0.75**. Rewritten.
2. **Class count was not constant and was never disclosed.** Phases 2–6 ran at the implementation default **K=8 (9 classes)**; only the final sweep uses K=3. Attribution is therefore *not comparable across phases* (chance 0.125 vs 0.33), meaning **Phase 4's 0.855 is a harder result than the delivered 0.875**. Table 4.11 gained an "Attribution classes" column plus a caveat paragraph.
3. "Roughly one hundred and twenty" runs → **113** (actual directory count).
4. **Batch size was never unrecoverable** — hardcoded `batch_size=16` at `watermark/scripts/quick_voice_smoke_train.py:486`, corroborated by `n_batches: 256` × 16 = 4,096. Added to Table 4.10.
5. **Stage-2 claim overstated.** "Actively hurting" softened to the matched-pair evidence (0.566→0.639 in one pair, unchanged in another) and to `WATERMARK_FINAL_FINDINGS.md`'s own wording ("often hurts attribution").
6. **Energy budget presented as achieved.** `tiered_eval` reports `budget_ok=0.008` — only 0.8% of clips are inside budget. Table 4.10 now says so and points to §5.4.3.
7. **4.6.5 / 4.6.6 contradiction.** Loss weights were called "the most consequential tuning decision" while 4.6.6 showed the Stage-1-only regime change mattered more. Resolved.
8. Minor: Table 4.10's "Device: MPS" is **inferred** from `config.py get_device()`, not recorded in the sweep3 logs.

### New: deterministic codebook as an evidenced rejected alternative
- New **Phase 6** row in Table 4.11, **Figure 4.21** (spectrogram comparison) and **Table 4.12** (SNR / PESQ / STOI / attribution).
- Generated by re-running the real `codebook.json` files and the real Run B `encoder.pt` on a common carrier clip. Assets + reproduction script in **`report_assets/codebook_ab/`**.
- **Measured result:** codebook variants are 10–20 dB *quieter* yet sound far worse — multitone PESQ **2.89–3.25** vs learned **4.63**. The only perceptually acceptable variant (subband-noise, PESQ 3.95) is the one whose attribution **collapsed to 0.51**, because per-class magnitude spectra were ~0.996 cosine-similar and the mel front end could not separate them.
- **The argument:** the codebook is accurate only while it is audible, and inaudible only while it is inaccurate. Plus a keyed scheme weakens provenance (leaked key ⇒ forgery/removal) and stable tones are notch-filterable.
- Sanity check on the whole pipeline: PESQ measured here **4.63** vs the reported 200-clip mean **4.638**.

### New: Ch5 now answers Ch2 (the "B + C" comparison)
- Previously **5.6 Discussion cited none of the Ch2 baselines** — the results floated free of the literature. This was a genuine gap missed in the first sample comparison.
- New **5.6.1 Comparison with existing watermarking systems**, **Table 5.12** (capability positioning vs AudioSeal / WavMark / C2PA, with an explicit "published figures, not a common test set" caveat) and **Table 5.13** (T1/T2/T3 mapped onto AudioMarkBench / SoK categories).
- States plainly that **on detection alone the project claims no advance over AudioSeal**; the contribution is attribution plus local-first operation.
- 6.3 gained the matching future-work item (a like-for-like AudioSeal/WavMark benchmark on the same 512 clips), honouring the forward reference made in 5.6.1.

### Recovered from git history (had been deleted from the working tree)
- `docs/WATERMARK_FINAL_FINDINGS.md` (commit `dc88392`) — the author's own selection rationale, tier findings, and the SNR-vs-PESQ explanation. **Primary evidence for several Ch4/Ch5 claims.**
- `docs/DETERMINISTIC_CODEBOOK_FINDINGS.md` + `_PLAN.md` (commit `b832afc`) — records the "ringing/whistling" audibility problem, the ~0.996 spectral-similarity cause of the attribution collapse, and the forgery/notch-filter weaknesses.

---

## Change log — accuracy pass on unimplemented claims (same session)

### ⚠️ The "quality gate" did not exist. Removed from 8 places.
`hub/voice_library.py::_wav_profile()` computes duration / rms / peak_abs / clipped_ratio and `create_voice()` stores them in `meta.json` — **but nothing ever reads them back.** No threshold, no rejection, no warning. A search for `clipped_ratio|peak_abs|rms` across `hub/`, `webui.py`, `desktop/` and `mobile/` returns nothing outside the function that computes them. The docstring's "for quality gating" was aspirational. There is also no silence measurement at all.

Corrected to *profiling* language in: 4.4.3 activity prose · the Key Features claim · 6.1 Objective 1 · Table 4.1 (UC-02) main flow · Table 4.1 alternative flow · Table 4.9 stage 3 name and rationale · Table 5.3 step S4 · Table 6.1 evidence. Verified zero remaining matches for `quality gate|gating|fails the gate|prompts a re-record`.

**Figure 4.3 was redrawn.** The old diagram *drew* a "Quality gate" node, a "Passes quality?" decision diamond and a `no →` "Warn user & re-record" loop. The original Graphviz sources were never committed and are not in git history, so the diagram was rebuilt from scratch matching the sibling diagrams' style. **Source now lives in `report_assets/diagrams/*.dot`** — do not let it go missing again. Render: `dot -Tpng -Gdpi=150 fig_4_3_activity_reference_intake.dot -o fig_4_3.png`.

6.3 gained a matching future-work item: turning the recorded profile into an intake warning is a contained change with a direct usability benefit.

### UC-10 delete dialog corrected
The spec claimed "the system states plainly what will be removed: the reference clip, the preprocessing metadata, and any cached model state". The actual UI is a native `confirm('Delete this saved voice?')` in `desktop/index.html`. The *behaviour* is right (`shutil.rmtree(voice_dir)` removes all three), but the dialog does not say so. Step 3 now reads "The system asks the user to confirm the deletion." Chosen over changing the code.

Note: a native `confirm()` is browser chrome, not page content — **it cannot be screenshotted** by any tool. If a delete-confirmation figure is ever wanted, the dialog must first become an in-page modal.

### New UI assets (`report_assets/ui/`)
Captured by driving the real app with Playwright at 2× DPI: three tour-spotlight steps (the tour renders a dimmed backdrop + ring + explanatory card, so the figures are self-annotating), and the reference-intake panel in Record and Upload modes. Two voice-library captures were taken but **not committed** — they show first names of people who provided voice samples, and this repo is public. Decide before adding them.

### Process errors made and caught during this pass
- **Content inserted into the front matter.** `find_para` matched a List-of-Tables entry instead of the body caption; the entire codebook block landed in the front matter. `validate.py` passed it (schema was valid). Caught by a caption-sequence check, reverted to backup, re-run with a body-only lookup. See the tooling notes.
- **Unhonoured forward reference**, second occurrence: 5.6.1 promised a Chapter 6 item that did not exist until added.
- **Overwrote the wrong image, then misdiagnosed it.** Swapping Figure 4.3's PNG used "find the caption, take the next image" — but **captions sit BELOW figures in this document**, so that targeted Figure 4.4 and destroyed its onboarding diagram. The render then still showed the old Figure 4.3, which was misread as "the write silently failed"; in fact it had succeeded on the wrong part. Caught by md5-comparing every media entry against a backup, and restored from `backup_before_gate.docx`.
  - **Rule:** to target a figure's image, anchor on the **prose paragraph that precedes it** (e.g. "Figure 4.3 describes…"), never on its caption.
  - **Rule:** `part._blob = ...` in python-docx **does** persist. Prefer a zip-level rewrite of `word/media/imageNN.png`, and always md5-verify every media entry against a backup afterwards — `validate.py` passes a corrupted-but-well-formed image without complaint.

---

## UI feature and evidence refresh — 2026-07-19

### Completed scope
- Finished persistent job favourites and labels across the service, PATCH API, desktop and mobile clients. Legacy records default safely; labels normalise to 80 characters and can be cleared with `null`; favourite timestamps are created and removed with star state.
- Completed Quick Phrases and Saved filtering. Replaying a saved phrase requests the existing job audio and does not submit a generation job.
- The Run control now shows `Generating...` and remains unavailable while any job is active. Long 32-character job IDs and the added star column remain inside the desktop and mobile layouts.
- Mobile Generate now places Quick Phrases immediately after the page heading, before model and reference selection. Both the desktop output dock and the mobile lower player expose a direct Save/star control for the currently loaded job.
- Playing a completed mobile job stays on Jobs and opens the same lower player above the navigation; it no longer redirects to Generate. The repaired player layout keeps metadata, waveform, star and download controls fully visible at 390 x 844.
- Saved jobs remain deletable. Desktop and mobile use itemised in-page dialogs that identify local audio and metadata, with an additional Quick Phrase warning for favourited jobs. Voice deletion remains API-backed and warns when saved phrases depend on the voice.
- Removed the obsolete state-only desktop `deleteVoice` path while retaining the real API deletion flow. Mobile sheets refresh immediately after rename and star actions. The mobile service-worker shell cache is `tts-hub-mobile-v11`.

### Automated and browser evidence
- Backend coverage lives in `tests/test_generation_jobs.py` and `tests/test_generation_job_api.py`.
- Desktop coverage lives in `tests/e2e/desktop_saved_phrases.spec.ts`; mobile coverage is in `tests/e2e/mobile_app.spec.ts`.
- Final verification passed: **13/13 backend unittests** and **11/11 scoped Playwright tests**. `node --check` passed for the mobile app, service worker and capture script; Python compilation passed for the job service, API and DOCX refresh script.
- Stored-audio network replay is automated; audible sound remains the only manual acceptance check.
- Current 2x-DPI captures are Figures 4.12--4.24 in `report_assets/ui_screenshots/`. The authoritative record is `report_assets/ui_screenshots/capture_manifest.json` (captured `2026-07-18T17:22:28.007Z`); it records timestamp, git identity, dirty state, viewports, record IDs, favourite/label state, player state and the real Run B detector result.
- Reproduction: `node report_assets/ui_screenshots/capture_report_screenshots.cjs`; DOCX replacement: `.venv/bin/python3 report_assets/refresh_fyp_ui_figures.py`.

### Private report outputs
- Updated report: `Ayaan Minhas-TP077859-APD3F2511CS(AI)-FYP Final Report (DRAFT).docx` (ignored/private).
- Latest pre-refresh backup: `report_assets/backups/Ayaan Minhas-TP077859-APD3F2511CS(AI)-FYP Final Report (DRAFT)-pre-ui-refresh-2026-07-19-012432.docx`.
- Latest verified render: `tmp/docs/fyp-ui-player-refresh/Ayaan Minhas-TP077859-APD3F2511CS(AI)-FYP Final Report (DRAFT).pdf`, with page renders and contact sheets beside it.
- All 13 approved UI figures were recaptured. Eleven embedded media files changed; Figures 4.14 and 4.15 were byte-identical to their deterministic recaptures. No media outside the approved 13 relationships changed. ZIP validation, python-docx loading, PDF rendering and a 191-page visual sweep passed; Section 4.5 was inspected closely on rendered pages 108--119.
- TOC, List of Figures and List of Tables fields were deliberately not refreshed. Their text remains a later final-report task.

### Remaining manual check
- Play one Quick Phrase through speakers/headphones on desktop and mobile and confirm it is audible. Automation already verifies the audio GET and absence of a generation POST.
