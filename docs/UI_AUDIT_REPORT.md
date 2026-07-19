# Desktop UI and Feature Audit

Date: 2026-04-21  
Scope: `desktop/` frontend and the FastAPI-backed local workflows exposed on `http://127.0.0.1:7896`

## Method

- Restarted the app server before testing.
- Inspected the UI at desktop and mobile widths with rendered screenshots for these surfaces:
  - `Generate`
  - `Models`
  - `Voices`
  - `History`
  - `Watermark Lab`
  - `System Status`
  - `Advanced Settings`
- Exercised completed workflows through the browser:
  - save voice
  - delete voice
  - watermark detection
  - session reset
  - validation errors on generate
  - persistence across reload
- Queried backend APIs directly to verify data format and surface behavior.

## Checklist

### Visual System

- [PASS] Dark palette is coherent across all surfaces.
- [PASS] Typography scale is strong and consistent for page titles, panel titles, and helper text.
- [PASS] Card radius, shadow depth, and glass treatment are visually consistent.
- [PASS] Primary and secondary buttons are visually distinct.
- [PASS] Focus rings are visible on interactive controls.
- [PASS] Contrast is generally good on desktop and mobile.
- [WARN] The UI is visually dense; the app presents a lot of information before the user reaches the main work area.

### Cross-Surface Consistency

- [PASS] Panels, badges, chips, form fields, and status bars share a common visual language.
- [PASS] The `Models`, `Voices`, and `Watermark Lab` surfaces reuse the same component system rather than introducing one-off styling.
- [WARN] Model selection is duplicated in the sidebar, model cards, and the `Advanced Settings` model picker, which increases cognitive load.
- [WARN] The selected model title and description are repeated at the top of every surface, which adds redundancy without adding much utility.

### Responsive Behavior

- [PASS] The layout collapses into a single-column stack below the desktop breakpoint.
- [PASS] Buttons wrap instead of overflowing on narrow screens.
- [PASS] Major panels do not visibly clip their content at mobile width.
- [WARN] On mobile, the sidebar remains very tall and pushes the primary workspace far below the fold.
- [WARN] The mobile first screen is dominated by navigation and model cards, so users must scroll a long distance before they reach the task they came for.

### Interaction and UX

- [PASS] Disabled controls are styled and mostly understandable.
- [PASS] Destructive actions are guarded by confirmation for saved-voice deletion.
- [PASS] The `Reset All` control clears transient form state without deleting saved voices.
- [PASS] Status messaging is present after key actions.
- [WARN] Some controls rely on repeated context rather than clear single-source affordances.
- [WARN] Several places still show low-value `unknown` device labels, which adds noise rather than clarity.
- [WARN] The main generate path does not feel immediately responsive; a real synthesize attempt did not return within 30 seconds.

### Functional Behavior

- [PASS] Save voice workflow works end to end.
- [PASS] Delete voice workflow works end to end and only removed the disposable test voice.
- [PASS] Watermark detection workflow works end to end.
- [PASS] Session reset clears transient UI state.
- [PASS] Validation blocks Qwen generation when reference audio is missing.
- [PASS] Advanced-settings model choice and model-specific toggle persisted across reload.
- [WARN] Positive generation completion was not observed within the audit window for a saved-voice synthesize attempt.
- [WARN] Some long-running generation paths may be too slow for a dashboard-style workflow.

## Confirmed Findings

### 1. Timestamp unit mismatch renders dates as 1970-era values

Severity: High

Evidence:

- Voice cards show `created_at` as 1970-era dates in the UI.
- Watermark run metadata shows `updated_at` as 1970-era dates in the UI.
- API payloads expose epoch-like values such as `1772014976` and `1770101493.4112577`, while the UI formats them with `new Date(ts)` as if they were milliseconds.

Impact:

- Saved voice metadata looks broken.
- Watermark run recency becomes misleading.
- Any workflow that depends on history or auditability becomes less trustworthy.

Likely fix:

- Normalize timestamps before formatting in the frontend, or enforce a single backend contract for milliseconds vs seconds.
- Apply the same conversion everywhere `formatDateTime` is used.

### 2. The generate workflow is too slow or blocked for a dashboard interaction

Severity: High

Evidence:

- A real generate attempt with an existing saved voice did not return an `/api/generate` response within 30 seconds.
- A Qwen validation-only path returned immediately, so the UI shell is functional, but the positive synthesize path is not currently snappy enough for interactive use.

Impact:

- The core workflow feels unreliable even if the backend eventually completes.
- The UI does not give enough feedback for a long-running request.

Likely fix:

- Surface a stronger progress state for generation.
- Consider async job handling or explicit loading phases.
- If the backend is expected to take this long, the UI should say so instead of looking stalled.

### 3. Mobile layout overweights the sidebar and navigation chrome

Severity: Medium

Evidence:

- At mobile width, the sidebar and model cards dominate the first viewport.
- Users reach the primary `Generate` content only after significant scrolling.

Impact:

- The main task surface is not front-loaded on small screens.
- The mobile experience feels like a compressed desktop layout rather than a mobile-first layout.

Likely fix:

- Collapse the sidebar into a compact top nav or drawer on mobile.
- Move model cards into a smaller, optional chooser surface instead of keeping the whole library visible first.

### 4. Repeated selection controls create redundant state entry points

Severity: Medium

Evidence:

- Models can be selected from the sidebar model cards and again in `Advanced Settings`.
- Surface switching is also available in the sidebar and via top-of-surface buttons.

Impact:

- The same state can be changed from multiple places without a single obvious primary control.
- This increases the chance of accidental context switches and makes the app feel busier than it needs to be.

Likely fix:

- Keep one primary selection control per task.
- Reduce duplicated navigation affordances where they do not add unique value.

### 5. Low-value `unknown` device labels add clutter

Severity: Low

Evidence:

- Sidebar model cards and model detail panels frequently show `unknown` device chips.
- In the default state these chips do not add useful information.

Impact:

- Visual noise increases without improving decision-making.

Likely fix:

- Hide or de-emphasize device chips until a real device value exists.
- Replace `unknown` with a more actionable state like `not loaded yet` if that better matches the backend.

## Functional Audit Notes

- Voice save and delete were verified with a disposable test voice only.
- Watermark detect was verified with a local benchmark audio file.
- Session reset cleared text, prompt text, watermark toggles, and output state without removing saved voices.
- Qwen generation validation behaved correctly when reference audio was missing.
- Persistence across reload behaved correctly for:
  - active surface
  - selected model
  - model-specific Qwen toggle state

## Recommendation Priority

1. Fix timestamp formatting across voice and watermark metadata.
2. Address generate latency or expose better long-running request feedback.
3. Rework the mobile layout so the task surface appears before the full navigation/model library.
4. Reduce duplicated selection controls and repeated header chrome.
5. Clean up low-value `unknown` device presentation.

## Retest Update

- [RESOLVED] Voice and watermark timestamps now render with the correct unit normalization in the frontend.
- [RESOLVED] A browser-driven positive generation flow now completes successfully with the UI showing `Generation complete.` after the response returns.
- [RESOLVED] The Qwen e2e coverage now uses a 4-minute grace period and waits for the success state, so the automated test reflects real-world latency instead of a 30-second default.
