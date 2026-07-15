# TTS Hub — Onboarding & User-Guidance Concepts

Research + 4 full walkthrough concepts, each with animation plans, implementation plans, and trade-offs.
Grounded in the actual `claude_exact/` stack: single-page `index.html` + `dc-runtime` (sc-if/sc-for → React),
all inline styles, CSS-var theming (dark/light), IBM Plex, existing keyframes (`fade`, `pulse`, `sweep`, `spin`),
views = Generate (model rail, editor, player), Voices, Jobs, Models, Watermark.

---

## 0. Research summary — what actually works in onboarding

**Pattern landscape** (Intro.js, Shepherd.js, Driver.js, React Joyride, Pendo/Appcues product tours):

1. **Passive tooltip tours get skipped.** Industry telemetry (Appcues, Chameleon benchmark reports) consistently
   shows multi-step "Next → Next → Next" tours have high abandonment — most users dismiss by step 2–3.
   NN/g's guidance: users don't read instructions upfront; they learn by doing and want help *in the moment of need*.
2. **Action-gated > click-through.** Tours that advance only when the user performs the real action
   ("type some text", "press Generate") have far better retention of the taught behavior, because the
   knowledge is encoded as motor memory, not reading.
3. **The "aha moment" should come first.** The single best activation predictor is time-to-first-value.
   For TTS Hub, value = *hearing your first generated audio*. Everything in onboarding should be a straight
   line to that, with detail (Jobs, Watermark, Models management) deferred.
4. **Contextual help beats upfront help.** A persistent, low-friction help affordance (`?` button / `Shift+?`)
   that users pull when confused outperforms anything pushed at them on first launch.
5. **Motion rules:** spotlight/dim overlays focus attention (dimming to ~50–65% is the sweet spot);
   animated transitions *between* steps preserve spatial context (the spotlight should travel, not teleport);
   everything must respect `prefers-reduced-motion`; keep durations in the app's existing 120–250 ms idiom.
6. **Escapability is non-negotiable.** Esc always exits, progress is persisted, tour is re-launchable from `?`.

**Technical constraint specific to this codebase:** there are no CSS classes or ids to target — everything is
inline-styled and rendered through `dc-runtime`. So any tour engine needs **explicit anchors**: add
`data-tour="<step-id>"` attributes to key elements in `index.html` (nav buttons, model rail, textarea,
generate button, player). This is the one invasive-but-tiny prerequisite shared by all four concepts
(~10 one-line edits). The tour engine itself can live in a separate `tour.js` loaded after `support.js`,
touching the app only through those anchors + a tiny event bridge (see §5).

---

## 1. Concept A — "Spotlight Voyage" (classic guided tour, done properly)

A cinematic spotlight tour on first launch: the whole UI dims under an SVG mask, a rounded-rect
spotlight *glides* from element to element, with a themed tooltip card (step dots, Next/Back/Skip,
keyboard ←/→/Esc).

**Step script (7 steps):**
1. Nav rail — "Five surfaces. You'll live in Generate."
2. Model rail — "Pick an engine; the dot shows load state."
3. Text editor — "Your script goes here. Saved transcripts land here too."
4. Voice picker — "Choose or clone a voice in Voices."
5. Generate button — pulse ring on it
6. Player/output — "Results play here and archive to Jobs."
7. `?` button — "Re-run this anytime from here." → done, confetti-less, quiet checkmark.

**Animation plan:**
- Overlay = single fixed `<svg>` covering viewport; `<mask>` = white full-screen rect + black rounded
  rect over the target. Animate the mask rect's `x/y/width/height/rx` with WAAPI (`element.animate`,
  260 ms, `cubic-bezier(.4,0,.2,1)`) so the spotlight **morphs and travels** between steps — this is the
  signature move; it keeps spatial continuity and looks expensive for ~40 lines of code.
- Tooltip card: `fade` keyframe you already have (translateY 4px + opacity), repositioned with
  FLIP so it slides rather than pops.
- Target gets a soft `box-shadow: 0 0 0 4px var(--accent-line)` breathing ring (reuse `pulse`).
- `prefers-reduced-motion`: skip travel animation, crossfade instead.

**Implementation plan (~1–1.5 days):**
1. Add `data-tour` anchors in `index.html` (10 edits).
2. New `tour.js`: step array `{anchor, title, body, placement}`; engine = position via
   `getBoundingClientRect` + `ResizeObserver` + scroll listener; SVG mask overlay; keyboard handling.
3. Trigger: `localStorage['tts.tour.v1']` absent → auto-start 600 ms after first paint; `?` button
   in nav footer re-launches.
4. Theming free via CSS vars (works in dark + light automatically).

**Strengths:** predictable, familiar UX; lowest engineering risk; fully decoupled from app logic
(read-only — never needs to know app state); easy to script/extend; great for a demo/FYP screenshot.
**Weaknesses:** it's still a passive tour — skip rates will be real; teaches locations, not workflow;
users may forget everything after dismissing; adds no ongoing help value beyond re-launch.

---

## 2. Concept B — "First Take" quest (action-gated, learn-by-doing) ★ recommended core

No lecture. On first launch a slim **mission card** docks bottom-right:
*"Make your first clip — 4 steps"* with a checklist (Pick a model → Write a line → Generate → Play it).
Only the **next required control** gets a pulsing beacon; each checklist item ticks off when the user
*actually does it*. Finale: when the first job completes, the checklist card flips into a mini
celebration — an animated waveform burst in accent green — then self-dismisses.

**Animation plan:**
- Beacon: 8 px accent dot with expanding sonar ring (`@keyframes` scale 1→2.6, opacity .6→0, 1.6 s loop)
  absolutely positioned on the current target's corner. One beacon at a time — calm, not a christmas tree.
- Checklist tick: SVG checkmark path with `stroke-dashoffset` draw-on (300 ms), row slides + fades to
  "done" state (reuse `fade`).
- Text pre-fill offer: "Try a sample line" chip that *types itself* into the textarea
  (typewriter, 18 ms/char) — thematically perfect for a TTS app: the app literally speaks first.
- Finale: 5 vertical bars (your logo motif!) doing a scaled-up `bar` keyframe dance for 1.2 s in the
  card — celebration that's on-brand instead of confetti.
- Reduced motion: beacons become static outlined badges; no typewriter.

**Implementation plan (~2–2.5 days):**
1. Same `data-tour` anchors.
2. Needs to *know app state* → add a tiny event bridge in dc-runtime state code: dispatch
  `window.dispatchEvent(new CustomEvent('tts:activity', {detail:{type:'model-selected'|'text-changed'|'generate-clicked'|'job-complete'|'played'}}))`
  at 5 call sites (few-line edits in dc-runtime/src, rebuild with bun). Fallback if you don't want to
  touch the runtime: infer via DOM listeners (`input` on textarea, click delegation on anchors,
  MutationObserver on player region) — dirtier but zero runtime edits.
3. `quest.js`: state machine (idx 0–3), persisted per-step in localStorage so a refresh resumes mid-quest.
4. Card is dismissible ("skip"); `?` re-offers it if never completed.

**Strengths:** teaches by doing → best retention; drives straight to the aha moment (first audio);
non-blocking — user keeps full control of the UI, power users just ignore the card; the completion
event is a real activation metric you can log.
**Weaknesses:** more coupling (needs app events or DOM inference); only covers the Generate flow —
Voices/Jobs/Watermark need a second mechanism; edge cases (user does steps out of order, generation
fails) need handling — define: out-of-order actions tick their boxes anyway; on job failure the card
says "that one hiccuped — try again" instead of celebrating.

---

## 3. Concept C — "Blueprint mode" (annotated x-ray on demand, powers the `?` button)

Press `?` (or click the help button) and the whole interface freezes into an **annotated schematic**:
UI dims, thin accent connector lines draw outward from ~8 key regions to floating labels in IBM Plex Mono
— like an exploded engineering diagram of the app. Press `?`/Esc again to snap back. No steps, no
sequence — the entire mental model in one glance, available forever, in any view (each view gets its own
annotation set: Generate has 8 labels, Jobs has 4, etc.).

**Animation plan:**
- Enter: backdrop dims (180 ms), then connector lines draw via `stroke-dashoffset` staggered 40 ms
  apart, labels `fade` in at line ends. Total reveal ~600 ms — feels like a blueprint materializing.
- Labels sit in whitespace, positions precomputed per breakpoint from anchor rects (simple placement
  solver: prefer nearest empty quadrant, nudge on overlap).
- Hover a label → its region gets the accent ring, others dim further (progressive focus).
- Optional flourish: mono labels "type on" (staggered), reinforcing the terminal aesthetic.
- Exit: reverse at 2× speed. Reduced-motion: instant show/hide.

**Implementation plan (~1.5 days):**
1. Same anchors + one JSON map per view: `{anchor, label, blurb}`.
2. `blueprint.js`: on toggle, snapshot anchor rects, render one SVG overlay (lines) + absolutely
   positioned label cards; `ResizeObserver` re-layouts; key handler for `?`/Esc.
3. Read-only — zero app-state coupling. Works on every view with just data additions.

**Strengths:** genuinely novel (nobody expects it — strong demo/FYP wow factor); *pull* not *push* —
respects the research that contextual help beats forced tours; covers all views cheaply; doubles as a
living feature map as the app grows; zero interruption on first launch.
**Weaknesses:** doesn't guide sequence — a truly lost first-time user still doesn't know what to do
*first*; label placement solver is the fiddly part (overlaps on small windows); discoverability of the
`?` affordance itself must be seeded (one-time tooltip on the help button).

---

## 4. Concept D — "Autopilot demo" (ghost cursor self-driving tour)

First launch offers: *"Watch a 30-second demo?"* Accept → a ghost cursor (soft accent circle with
trailing glow) pilots the real UI: glides to a model, clicks it, typewrites a sentence, presses
Generate, a **mocked** job animates through the progress states, and a pre-bundled sample clip plays.
Banner on top: "Demo — press any key to take over." Any input instantly aborts and hands control back.

**Animation plan:**
- Cursor: fixed div, moved along bezier paths with WAAPI (`offset-path` or JS-lerped), ease-in-out,
  ~500–800 ms per hop; click = scale-pulse + ripple ring at the point.
- Typewriter into the real textarea (dispatching real `input` events so the UI reacts authentically).
- Progress states reuse your `sweep` shimmer; player bars animate with `bar`.
- Ends with the ghost cursor parking on the Generate button and dissolving — "your turn."

**Implementation plan (~3–4 days, the heavy one):**
1. Anchors + a scripted timeline `[{move, click, type, wait-for}]` interpreter.
2. **Demo mode flag** in the app: generate must short-circuit to a mocked job + bundled `demo.wav`
   (never hit real workers) — this requires real edits in dc-runtime state logic, and cleanup on abort
   (restore textarea, clear mock job). This is the risk center.
3. Abort handling: any `keydown/mousedown` → cancel timeline, run cleanup, show "take over" toast.

**Strengths:** maximum wow; shows real *feel* of the flow including timing; great marketing/demo asset
(also screen-recordable for docs); zero reading required.
**Weaknesses:** highest effort and highest fragility — breaks whenever UI layout/flow changes; mocked
state can drift from real behavior; passive again (watching ≠ doing) so retention is worse than B;
abort/cleanup edge cases are a genuine bug farm. Best treated as a v2 luxury, not the foundation.

---

## 5. Shared tour-engine foundation (build once, all concepts ride on it)

```
claude_exact/
  tour/
    engine.js      # anchor registry, rect tracking (ResizeObserver+scroll), overlay layer, WAAPI helpers
    spotlight.js   # Concept A renderer (SVG mask)
    quest.js       # Concept B state machine
    blueprint.js   # Concept C renderer
    steps.js       # all copy/step data — pure data, easy to edit
```
- **Anchors:** `data-tour="nav-generate|model-rail|editor|voice|generate-btn|player|help"` in index.html.
- **Event bridge (B only):** 5 CustomEvent dispatches in dc-runtime/src → rebuild.
- **Persistence:** `localStorage` keys `tts.onboard.quest`, `tts.onboard.tour.v1` (version the key so a
  future redesign can re-trigger).
- **A11y:** overlay `role="dialog"` + focus trap for A; `aria-live="polite"` announcements for B ticks;
  every mode exits on Esc; all animations gated on `prefers-reduced-motion`.
- **Theming:** exclusively CSS vars → dark/light for free.

## 6. Recommendation

**Ship B + C as one system (~3.5 days):**
- **B ("First Take" quest)** on first launch — it's the shortest path to the aha moment and the research
  says action-gated onboarding is what actually sticks. It's quiet enough that it never insults a
  power user.
- **C ("Blueprint mode")** behind the `?` button + `Shift+?` — permanent, all-views contextual help, and
  it's the innovative signature piece.
- Skip A (subsumed by B+C), park D as a future demo/marketing mode.
- Seed discoverability: after the quest completes, the `?` button pulses once with a one-time tooltip
  "Blueprint mode lives here."
