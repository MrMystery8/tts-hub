# Detailed Completion Task Plan for `tts-hub` FYP

## Summary
- Goal: take the current repo from “working prototype with active experiments” to a report-ready FYP deliverable that satisfies the Investigation Report commitments for local-first voice cloning, AAC-oriented usability, and bounded watermark provenance.
- Recommended completion target: a frozen final system using 3 primary backends, 1 final watermark run, 1 reproducible evaluation pack, and 1 clean documentation/reporting trail.
- Definition of complete: every task below is done, the final benchmark results are reproducible from repo commands, and the report can truthfully claim implementation and evaluation rather than only planning.

## Completion Standard
- Final backend comparison set is fixed and documented.
- Final watermark attribution class set is fixed and documented.
- Prompt intake, generation, watermarking, and deletion flows all work locally.
- Evaluation covers quality/intelligibility proxy, similarity, latency, memory/stability, usability, and watermark robustness.
- One final run/config is pinned as the default artifact for demo and report screenshots.
- Documentation reflects actual final behavior, not exploratory or outdated states.

## Tracker

### Phase 1 Tracker: Scope Lock and Final Project Definition
| Done | ID | Task | Priority | Status | Effort |
|---|---|---|---|---|---|
| [ ] | 1.1 | Freeze the final backend set to the primary report models | High | Not Started | S |
| [ ] | 1.2 | Freeze the final watermark attribution class set | High | Not Started | S |
| [ ] | 1.3 | Define and document the final claim boundary and non-claims | High | Not Started | S |
| [ ] | 1.4 | Separate final deliverable artifacts from exploratory leftovers | High | Not Started | M |

### Phase 2 Tracker: Prompt Intake and Local-First Voice Pipeline
| Done | ID | Task | Priority | Status | Effort |
|---|---|---|---|---|---|
| [ ] | 2.1 | Finish the report-grade prompt intake workflow | High | Not Started | L |
| [ ] | 2.2 | Add quality-gating rules for prompt audio | High | Not Started | M |
| [ ] | 2.3 | Persist preprocessing metadata for saved voices | High | Not Started | M |
| [ ] | 2.4 | Finalize transcript policy per primary backend | Medium | Not Started | M |
| [ ] | 2.5 | Finish real local deletion semantics for prompts, caches, and related artifacts | High | Not Started | M |

### Phase 3 Tracker: AAC Workflow and Product Hardening
| Done | ID | Task | Priority | Status | Effort |
|---|---|---|---|---|---|
| [ ] | 3.1 | Implement onboarding flow | High | Not Started | M |
| [ ] | 3.2 | Implement saved phrases | High | Not Started | M |
| [ ] | 3.3 | Finalize generation workflow UX across desktop and mobile | High | Not Started | M |
| [ ] | 3.4 | Expose accurate model requirements in the UI | High | Not Started | S |
| [ ] | 3.5 | Add output and history management with cleanup controls | Medium | Not Started | M |
| [ ] | 3.6 | Keep React as the single final UI surface | Medium | Not Started | S |

### Phase 4 Tracker: Runtime, Config, and Stability Hardening
| Done | ID | Task | Priority | Status | Effort |
|---|---|---|---|---|---|
| [ ] | 4.1 | Replace hardcoded sibling-repo assumptions with config | High | Not Started | M |
| [ ] | 4.2 | Standardize worker status telemetry | High | Not Started | M |
| [ ] | 4.3 | Generalize worker recycling policy where appropriate | Medium | Not Started | M |
| [ ] | 4.4 | Add structured error reporting | High | Not Started | M |
| [ ] | 4.5 | Freeze runtime defaults for final evaluation | High | Not Started | S |

### Phase 5 Tracker: Watermark System Finalization
| Done | ID | Task | Priority | Status | Effort |
|---|---|---|---|---|---|
| [ ] | 5.1 | Select the final watermark training regime | High | Not Started | M |
| [ ] | 5.2 | Train the final watermark run against the chosen class set | High | Not Started | L |
| [ ] | 5.3 | Wire the final watermark run as the default runtime artifact | High | Not Started | S |
| [ ] | 5.4 | Update watermark model mapping to final scope | High | Not Started | S |
| [ ] | 5.5 | Finalize threshold policy | Medium | Not Started | S |
| [ ] | 5.6 | Bound and document the final robustness suite | High | Not Started | M |

### Phase 6 Tracker: Benchmark and Evaluation Pack
| Done | ID | Task | Priority | Status | Effort |
|---|---|---|---|---|---|
| [ ] | 6.1 | Build the final benchmark protocol | High | Not Started | M |
| [ ] | 6.2 | Measure latency and runtime feasibility | High | Not Started | L |
| [ ] | 6.3 | Measure intelligibility proxies | High | Not Started | L |
| [ ] | 6.4 | Measure speaker similarity | High | Not Started | L |
| [ ] | 6.5 | Record qualitative acceptability notes | Medium | Not Started | M |
| [ ] | 6.6 | Run watermark robustness evaluation | High | Not Started | L |
| [ ] | 6.7 | Run usability task evaluation | High | Not Started | M |

### Phase 7 Tracker: Testing and Verification
| Done | ID | Task | Priority | Status | Effort |
|---|---|---|---|---|---|
| [ ] | 7.1 | Keep the current Python regression suite green | High | Not Started | S |
| [ ] | 7.2 | Run and stabilize Playwright smoke coverage | High | Not Started | M |
| [ ] | 7.3 | Add or stabilize backend generation e2e tests for the final backend set | Medium | Not Started | L |
| [ ] | 7.4 | Add tests for new AAC and product features | Medium | Not Started | M |
| [ ] | 7.5 | Define the final acceptance checklist and validation commands | High | Not Started | S |

### Phase 8 Tracker: Documentation and Report Alignment
| Done | ID | Task | Priority | Status | Effort |
|---|---|---|---|---|---|
| [ ] | 8.1 | Rewrite the main README around the final deliverable | High | Not Started | M |
| [ ] | 8.2 | Consolidate spec, progress, and roadmap docs | High | Not Started | M |
| [ ] | 8.3 | Produce a final reproducibility guide | High | Not Started | M |
| [ ] | 8.4 | Produce report-ready evidence artifacts | High | Not Started | L |
| [ ] | 8.5 | Cross-check the repo against the Investigation Report objectives | High | Not Started | M |

## Phase 1: Scope Lock and Final Project Definition
- Task 1.1: Freeze the final backend set.
  - Use `Qwen3-TTS MLX`, `IndexTTS2`, and `Chatterbox Multilingual` as the primary report set.
  - Keep `F5`, `CosyVoice3`, `PocketTTS`, and `VoxCPM-ANE` as optional/bonus integrations unless they are needed for a specific chapter or appendix.
  - Done when all docs, evaluation commands, and UI wording clearly identify the primary set.
- Task 1.2: Freeze the final watermark class set.
  - Decide the attribution classes to match the final backend set.
  - Recommended final classes: `qwen3-tts-mlx`, `index-tts2`, `chatterbox-multilingual`.
  - Done when watermark training config, runtime mapping, and report tables all use the same class mapping.
- Task 1.3: Define the final claim boundary.
  - Keep “offline-by-default, bounded non-adversarial provenance, not clinically deployed, not robust to adaptive attacker” as explicit non-claims.
  - Done when the README, report notes, and evaluation write-up use the same claim language.
- Task 1.4: Cleanly separate “primary deliverable” from “experimental leftovers”.
  - Mark which runs, docs, and outputs are exploratory only.
  - Done when one person can open the repo and identify the final path without guessing.

## Phase 2: Prompt Intake and Local-First Voice Pipeline
- Task 2.1: Finish the report-grade prompt intake workflow.
  - Current state is mainly file upload plus ffmpeg conversion.
  - Add explicit intake stages: mono conversion, canonical sample-rate handling, optional segmentation, optional denoising, and quality screening.
  - Done when every saved voice passes through one visible, logged preprocessing path.
- Task 2.2: Add quality-gating rules for prompt audio.
  - Use measurable checks already hinted at in the repo: clipping ratio, duration, silence proportion, loudness/RMS proxy, and transcript presence/provenance.
  - Define pass, warning, and fail conditions.
  - Done when low-quality prompts trigger actionable user-facing feedback instead of silent acceptance.
- Task 2.3: Persist preprocessing metadata.
  - Record what happened to each saved prompt: source format, converted format, duration, sample rate, quality flags, whether transcript was user-provided or auto-transcribed, and which caches were built.
  - Done when saved voice metadata is sufficient for reproducibility and deletion auditing.
- Task 2.4: Decide and implement transcript policy for each primary backend.
  - Qwen: allow auto-transcribe or manual transcript.
  - IndexTTS2: transcript optional unless required by future workflow.
  - Chatterbox: transcript optional unless cloning mode needs it.
  - Done when the UI and validation logic match backend reality exactly.
- Task 2.5: Finish local deletion semantics.
  - Deleting a saved voice must remove prompt audio, metadata, derived caches, and any related preprocessing artifacts.
  - Decide retention behavior for generated outputs and temporary uploads.
  - Done when deletion is real, local, and auditable rather than symbolic.

## Phase 3: AAC Workflow and Product Hardening
- Task 3.1: Implement onboarding flow.
  - The report expects “basic onboarding”; define it as a short guided flow covering model choice, prompt upload/save, transcript entry if needed, and first successful generation.
  - Done when a first-time user can complete setup without reading code or raw docs.
- Task 3.2: Implement saved phrases.
  - Add a persistent phrase bank for common AAC utterances.
  - Minimum capability: create, list, edit if needed, select, delete.
  - Done when “save a phrase” is a real feature rather than a future idea.
- Task 3.3: Finalize generation workflow UX.
  - Ensure text entry, prompt selection, output generation, playback, and download/export are stable on desktop and mobile.
  - Done when the main generation surface supports the four core user tasks without dead ends.
- Task 3.4: Expose accurate model requirements in UI.
  - Each model should clearly state whether it needs reference audio, transcript, optional emotion prompt, or cached voice.
  - Done when model-specific input mistakes are prevented before submission wherever possible.
- Task 3.5: Add meaningful output/history management.
  - Decide whether history is session-only or persistent.
  - Add clear output naming, timestamps, model labels, watermark-used state, and cleanup controls.
  - Done when outputs can be reviewed and removed intentionally.
- Task 3.6: Keep the React UI as the final surface.
  - Treat the archived React prototype as historical unless there is a reason to restore it.
  - Done when there is one clearly supported frontend for final demos and testing.

## Phase 4: Runtime, Config, and Stability Hardening
- Task 4.1: Replace hardcoded sibling-repo assumptions with config.
  - Current runtime path resolution is workspace-specific.
  - Add a central config for repo roots, Python paths, output roots, default model variants, and runtime toggles.
  - Done when the final setup is explicit and not dependent on one machine layout.
- Task 4.2: Standardize worker status telemetry.
  - Extend status reporting to include load state, device, generation count, last duration, PID, memory stats where available, and last error.
  - Done when the system-status surface is useful for evaluation evidence.
- Task 4.3: Generalize worker recycling policy.
  - IndexTTS2 already has recycle recommendation logic; decide whether primary backends should all support recycle or only expose what is meaningful per runtime.
  - Done when memory-growth handling is deliberate and documented.
- Task 4.4: Add structured error reporting.
  - Return stable error codes and user-facing hints for missing transcript, missing prompt audio, unavailable weights, worker crash, ffmpeg failure, and watermark incompatibility.
  - Done when common failures are diagnosable without reading logs.
- Task 4.5: Freeze runtime defaults for final evaluation.
  - Pick the exact model variant and default settings used in the report for each primary backend.
  - Done when benchmarking and screenshots are using fixed defaults rather than drifting settings.

## Phase 5: Watermark System Finalization
- Task 5.1: Select the final training regime.
  - Use the existing training framework, but stop treating the repo as an open-ended experiment space.
  - Choose one final schedule, one final class count, one final data source, and one final threshold strategy.
  - Done when the final run can be named as the official artifact.
- Task 5.2: Train the final watermark run against the chosen class set.
  - Use the public benchmark data and the final attribution scope.
  - Keep the full artifact pack: `encoder.pt`, `decoder.pt`, config, metrics, session/report files, and threshold note.
  - Done when the run is complete and intentionally pinned for the hub.
- Task 5.3: Wire the final run as the default runtime watermark.
  - Runtime embed/detect should default to the chosen final run, not a moving target.
  - Done when the UI and API resolve to the same default run unless manually overridden.
- Task 5.4: Update watermark model mapping to final scope.
  - Current runtime mapping is narrower than the report ambition.
  - Extend or intentionally freeze the mapping according to the final backend set.
  - Done when runtime attribution IDs, training labels, and report tables are identical.
- Task 5.5: Finalize threshold policy.
  - Decide whether the hub uses recommended per-run threshold automatically by default.
  - Document manual override behavior.
  - Done when detection results are reproducible and not dependent on ad hoc threshold changes.
- Task 5.6: Bound the robustness suite.
  - Final evaluation should cover the agreed T1/T2/T3-style transforms only.
  - Keep the explicit non-claim that robustness against adaptive removal is out of scope.
  - Done when robustness reporting is disciplined and consistent.

## Phase 6: Benchmark and Evaluation Pack
- Task 6.1: Build the final benchmark protocol.
  - For each primary backend, define prompt set, generation texts, repetitions, output settings, and logging fields.
  - Recommended rule from report: each model-condition combination runs 5 times.
  - Done when benchmark commands are frozen and repeatable.
- Task 6.2: Measure latency and runtime feasibility.
  - Capture median and p95 latency, time-to-first-audio if observable, maximum memory, and failure categories.
  - Done when feasibility claims on Apple Silicon are backed by saved results.
- Task 6.3: Measure intelligibility proxies.
  - Add WER or CER proxy evaluation for final generated outputs using a consistent ASR path.
  - Document limitations for accents, multilingual cases, and unusual speech.
  - Done when intelligibility is reported for the primary backend comparison set.
- Task 6.4: Measure speaker similarity.
  - Add embedding-based similarity statistics for cloned outputs against reference prompts.
  - Use one fixed methodology and report limitations honestly.
  - Done when “voice identity preservation” is evidenced rather than assumed.
- Task 6.5: Record qualitative acceptability notes.
  - For each backend, note artifacts, pronunciation failures, instability, or strengths that objective metrics may miss.
  - Done when the report has bounded listening/acceptability commentary.
- Task 6.6: Run watermark robustness evaluation.
  - Report AUC and TPR at fixed FPR for clean and each chosen transform tier.
  - Include quality-impact diagnostics where already supported.
  - Done when one final table summarizes watermark behavior under the bounded suite.
- Task 6.7: Run usability task evaluation.
  - Required tasks: onboarding, saving a phrase, generating speech, deleting a recording.
  - Record completion status, failure points, and brief acceptability notes.
  - Done when the repo can support the report’s usability claims directly.

## Phase 7: Testing and Verification
- Task 7.1: Keep the current Python regression suite green.
  - This is the baseline engineering contract.
  - Done when the targeted unit/smoke suite passes after all final changes.
- Task 7.2: Run and stabilize Playwright smoke coverage.
  - Keep the existing render-smoke tests and ensure the React UI renders cleanly across surfaces.
  - Done when frontend smoke is part of the final verification routine.
- Task 7.3: Add or stabilize backend generation e2e tests for the final backend set.
  - Qwen already has stronger e2e coverage.
  - Add at least one successful end-to-end flow per primary backend if practical.
  - Done when final-demo backends are not verified only by manual use.
- Task 7.4: Add tests for new AAC/product features.
  - Cover phrase CRUD, deletion semantics, status payload shape, and watermark default-run selection.
  - Done when the new final-scope features have regression protection.
- Task 7.5: Define final acceptance checklist.
  - Include one command or script path for backend smoke, frontend smoke, watermark smoke, and benchmark reproduction.
  - Done when another person can validate the final build without tribal knowledge.

## Phase 8: Documentation and Report Alignment
- Task 8.1: Rewrite the main README around the final deliverable.
  - Remove ambiguity between prototype, experiments, and final supported path.
  - Done when README matches the real final workflow.
- Task 8.2: Consolidate spec/progress/roadmap docs.
  - Keep one source-of-truth technical description and clearly mark archival or exploratory notes.
  - Done when documentation no longer competes with itself.
- Task 8.3: Produce a final reproducibility guide.
  - Include environment setup, model repo expectations, run commands, benchmark commands, and final output locations.
  - Done when the final results can be reproduced from the guide.
- Task 8.4: Produce report-ready evidence artifacts.
  - Save tables, figures, metrics summaries, and screenshots needed for the final write-up.
  - Done when the report can be assembled from real outputs already generated by the repo.
- Task 8.5: Cross-check the repo against the Investigation Report objectives.
  - Verify each original objective has corresponding implemented evidence.
  - Done when there is no remaining objective that is only “planned”.

## Public APIs, Interfaces, and Types to Add or Finalize
- Add phrase management endpoints and corresponding frontend state/types.
- Extend status API payload to include memory/error/recycle telemetry.
- Add cleanup/output-management endpoints if outputs remain user-visible.
- Extend saved-voice metadata schema to include preprocessing and transcript provenance.
- Add central app/runtime config for model repo paths, defaults, and final watermark run selection.

## Final Acceptance Criteria
- A new user can onboard, save a reusable voice, save a phrase, generate speech, and delete stored material locally.
- The primary backend set is fixed and benchmarked.
- The final watermark run is fixed, selectable, and used by default.
- Quality/intelligibility proxy, similarity, latency, memory/stability, usability, and watermark robustness are all reported.
- The repo has one clear “final system” path and one clear “experimental/archive” boundary.
- The final report can claim implementation and evaluation with direct evidence from the repo.

## Assumptions and Defaults
- Default final comparison set: `Qwen3-TTS MLX`, `IndexTTS2`, `Chatterbox Multilingual`.
- Default final watermark attribution set: the same three backends.
- Default final UI: `desktop/`; archived React variants are not part of the delivered runtime.
- Default benchmark datasets: `mini_benchmark_data` for fast iteration, `medium_benchmark_data` for main public evaluation.
- Existing exploratory runs remain useful as references, but do not count as the final artifact unless explicitly pinned and documented.
