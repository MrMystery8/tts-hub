# Post-UAT Improvements

Date: 21 July 2026

## Purpose

This document records the interface improvements implemented after reviewing the User Acceptance Testing (UAT) evidence in the FYP report. The work focuses on the recurring usability issues reported in Tables 5.9-5.11: difficult watermark interpretation, technical model choice, incomplete low-effort operation, saved-phrase discoverability, and model selection resetting across mobile reconnection.

The changes apply to both the desktop hub and mobile Progressive Web Application (PWA) unless stated otherwise.

## 1. Plain-language watermark verification

### UAT pain point

- Three of four participants required help with desktop verification.
- Three of four participants required help with mobile verification.
- Watermark verification was the lowest-rated questionnaire item, with a mean of 3.25/5.
- Participants asked for a concise conclusion before probability, threshold, detector-run, and attribution evidence.

### Improvements

- Verification results now lead with either `Watermark detected` or `No watermark detected`.
- A short explanation immediately follows the verdict.
- Raw score, threshold, detector run, and source-model evidence are collapsed under `View technical details`.
- A source model is shown only when a watermark is detected.
- Clean audio no longer displays a misleading `unknown source model` message.
- The redundant `Clear result` pill was removed.
- Desktop detector-run configuration and metrics remain available under advanced details instead of preceding the verdict.

### Result

The default result is readable without watermarking expertise while the evidence required for evaluation and auditing remains available on demand.

## 2. Recommended model and simpler model choice

### UAT pain point

- Multiple participants described model selection as technical or difficult.
- Low-effort use received a mean rating of 3.50/5.
- Participants requested one fast recommended model for routine communication.
- The mobile reconnection task showed that the selected model could reset.

### Improvements

- Qwen3-TTS MLX is the default and recommended model on desktop and mobile.
- Qwen is ordered first and carries a visible `Recommended` pill.
- The selected model is stored under the shared `ttshub-selected-model` preference and restored after reload or reconnection.
- The three evaluated models now have short, task-oriented descriptions:
  - IndexTTS2: `Best quality and expressive control.`
  - Qwen3-TTS MLX: `Fast and reliable for daily use.`
  - Chatterbox Multilingual: `Multilingual and long-form speech.`
- Desktop descriptions were shortened and tested to fit without truncation in the model rail.
- Unhelpful selector metadata such as `idle - unknown - 0 runs` was removed from both clients.
- The mobile Generate screen now includes a `Model` heading, a compact selector description, and the Recommended pill.
- The expanded mobile model list retains capability pills and descriptions but removes the ambiguous loaded/device text.
- Selection controls expose `aria-pressed` state for assistive technology and automated checks.

### Result

Users receive a safe default and plain-language trade-offs without losing access to alternative models or advanced configuration.

## 3. Quick Phrase discovery, saving, and playback

### UAT pain point

- Only two of four participants saved a phrase without assistance.
- Phrase creation was discoverable mainly through starring a completed job.
- Participants requested clearer phrase creation for routine communication.

### Improvements

- The desktop Generate tutorial now contains two distinct Quick Phrase steps:
  1. `Save a quick phrase` spotlights the exact Save button in the completed-output player.
  2. `Play a quick phrase` spotlights the Quick Phrases shelf and explains one-click playback.
- Mobile now has its own touch-first tutorial rather than a reduced desktop overlay.
- The mobile `?` button replays a guide tailored to the currently open tab:
  - Generate covers navigation, model and reference choice, script entry, options, generation, saving, and Quick Phrase playback.
  - Voices covers adding and managing reusable references.
  - Jobs covers filtering history and reopening completed work.
  - Verify covers choosing audio, optional technical controls, and running detection.
- The mobile tutorial uses safe-area-aware placement, large Back/Next controls, reduced-motion support, focus trapping, and a card that docks opposite the highlighted control.
- Before a completed output exists, the save lesson uses the output dock as its fallback target.
- The playback lesson is included once at least one phrase exists.
- Browser-native prompts were replaced with styled, product-native dialogs on desktop and mobile.
- The dialogs support saving, renaming, cancelling, click-outside dismissal, Escape dismissal, and Enter submission.
- Saving failures retain the dialog contents so the action can be retried.
- The redundant `plays instantly` label and prompt copy were removed from both interfaces.

### Result

Phrase creation is taught at the point where the action occurs, while naming and renaming use the same visual language as the rest of the product.

## 4. Mobile PWA update reliability

### Observed issue

The Tailscale URL served the latest application, but an installed mobile PWA could continue displaying an older application shell after reinstalling or reopening it.

### Improvements

- Mobile CSS, JavaScript, and service-worker URLs now use explicit build query versions.
- The service worker registers with `updateViaCache: "none"` and requests an update after registration.
- Old application caches are removed during service-worker activation.
- Network-first shell handling remains in place, with the cache retained only as an offline fallback.
- The current mobile application-shell cache is `tts-hub-mobile-v19`.
- The current desktop tutorial asset version is `v11`.

### Result

Mobile interface changes reach installed PWAs without relying on users to clear Safari website data manually.

## 5. Regression coverage

The post-UAT changes are covered by Playwright tests for:

- plain-language desktop verification and collapsed technical evidence;
- clean-audio results without false source-model wording;
- recommended-model ordering, descriptions, persistence, and selector accessibility;
- desktop descriptions fitting within the model rail;
- mobile selector descriptions and removal of loaded/device metadata;
- custom Quick Phrase save and rename dialogs on desktop and mobile;
- dialog cancellation and keyboard submission;
- separate tutorial spotlights for saving and playing Quick Phrases;
- touch-first, per-tab mobile tutorial content and spotlight geometry;
- mobile service-worker cache versioning; and
- supported viewport rendering without horizontal overflow.

Validation command:

```bash
npx playwright test
```

Latest result: 18 passed and 2 optional Qwen integration tests skipped.

## Remaining UAT backlog

The following suggestions were not part of this improvement pass:

- persistent, step-by-step mobile reconnection guidance;
- phrase folders or situation-based phrase packs;
- protected caregiver settings; and
- a remove-after-recovery workflow for temporary voice-loss users.

The latter three were individual participant suggestions rather than recurring findings and require broader product decisions than the low-risk improvements documented above.
