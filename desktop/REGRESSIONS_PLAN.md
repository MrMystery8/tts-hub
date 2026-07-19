# Desktop Client - Backend Feature Regressions and Integration Plan

Comparison of backend capabilities (`webui.py` + `/api/*`) against what the
supported desktop design (`desktop/index.html`) actually wires up.

> **Architecture note (corrected).** `index.html` is a **single class**
> `Component extends DCLogic` (L419). It carries **duplicate methods**: an
> earlier mock definition and a later `fetch`-backed override (e.g.
> `runGenerate` L648 vs async L1521; `deleteVoice` L712 vs L1559; `downloadOut`
> L703 vs L1555). Later definitions **shadow** earlier ones at runtime.
> **But the renderers** `renderVoices` (L1032) and `renderJobs` (L1119) live in
> the *earlier* section and contain inline mock handlers (e.g. mock download at
> L1192). So fixes must touch **both** the renderers and the later override
> area — and stale shadowed mock methods should be removed.
>
> **Real method names to target** (avoid drift): `runGenerate()` (L1521),
> `validate()` (L636), `__appendParams(form, id)` (L1499). There is **no**
> `__buildForm` or `canGenerate`. **Theme tokens** are `var(--line)`,
> `var(--line-2)`, `var(--line-3)`, `var(--accent)`, `var(--accent-tx)`,
> `var(--accent-dim)`, `var(--accent-line)`, `var(--tx-2)`. There is **no**
> `var(--bd)`.

---

## Backend capabilities inventory (source of truth)

| Capability | Endpoint | Wired in design? |
|---|---|---|
| List/create/rename/delete voices | `/api/voices` (GET/POST/PATCH/DELETE) | ✅ |
| **Play a saved voice's reference audio** | `GET /api/voices/{id}/audio` | ❌ **missing** |
| Voice meta: `duration_s`, `has_transcript`, `compatible_models` | `/api/voices` | ✅ |
| Voice meta: `has_caches` (warm/cached) | `/api/voices` | ❌ not surfaced |
| Voice `prompt_text` (transcript) view/edit after create | create only | ⚠️ partial |
| Generate (sync + async jobs) | `/api/generate`, `/api/generation-jobs` | ✅ |
| **Play generated output in-browser** | `GET /api/generation-jobs/{id}/audio` | ❌ **mock only** |
| Download generated output | same | ✅ (L1555) |
| **Emotion reference audio upload** (`emo_audio`) | generate form field | ❌ **missing** |
| Output format wav/mp3/flac | form `output_format` | ✅ |
| Watermark embed / run select / detect / threshold | `/api/watermark/*` | ✅ |
| Model status / device / unload | `/api/status`, `/api/unload` | ✅ |
| Cancel / delete jobs, history, restore | `/api/generation-jobs/*` | ✅ |

---

## Regressions (ranked)

### 🔴 1. Playback controls are mock; only download is backend-backed
`togglePlay()` (L702, bound L1017) only flips an `isPlaying` boolean to swap the
play/pause icon. There is **no `<audio>` element, no `new Audio()`, no
`.play()`** in the file. Download *is* live (`downloadOut` L1555 →
`__audioUrl`), so the precise gap is: the generated result can be **downloaded
but not heard in-app**. Biggest functional regression.

### 🔴 2. Cannot listen to a saved / recorded voice
`GET /api/voices/{id}/audio` is never called. The Voice Library can Use /
Rename / Delete a voice but offers no Preview/Listen. (User's flagged example.)

### 🔴 3. Emotion reference audio (`emo_ref`) is a dead option
IndexTTS2 exposes `emoMode: 'emo_ref'` ("Emotion reference audio", L738) in the
dropdown, but `runGenerate()`'s form assembly (L1526–1530, via `__appendParams`
L1499) never appends an `emo_audio` file — there is no upload control for it.
Selecting the mode silently does nothing, even though the backend fully supports
`emo_audio` → `emo_audio_path`.

### 🟠 4. Reference audio can't be previewed before generating
Uploaded/recorded references (`refFileBlob`, L1472/L1491) are sent blind. No
"Listen" affordance to verify the take, and a recorded reference cannot be
**saved into the Voice Library** (it's one-shot per generation only).

### 🟠 5. Voice transcript (`prompt_text`) is write-once and hidden
`prompt_text` is sent on create (L1563) but the library never shows or lets you
edit it; PATCH only carries `name` (L1560). Models needing a transcript can't
have it inspected/fixed for an existing voice.

### 🟠 6. Watermark detection collapses structured attribution into a toast
`__detectWatermark` (L1597) flashes only "detected / p=…". The backend returns
`detected`, `wm_prob`, attributed model `{id,name,tts_model_id}`, and `run.id`.
All attribution is lost. The Watermark Lab should render a **persistent result
panel** (detected/not, probability, attributed model, run id) that stays until
the next detection.

### 🟡 7. Per-model cache state (`has_caches`) not surfaced
`has_caches` is a **per-model dict** `{model_id: bool}` in the list response
(voice_library.py:99,133) — a real latency signal. The design shows `duration_s`
+ `has_transcript` but not cache state. Surface it **per model** ("Index cache",
"Qwen cache", or "Warm for selected model"), not as one generic warm/cold chip.

> **Scoped out / already present:** Job *Restore settings* already works via
> `restoreJob()` (L711) — verify only, do not rebuild. Transcript **editing**
> (#5) needs a backend change (`PATCH /api/voices/{id}` is rename-only) and is
> **scoped OUT of the first implementation**; transcript **viewing** only needs
> a `GET /api/voices/{id}` call on expand/select since `prompt_text` is absent
> from the list response (webui.py:161). Everything else is pure frontend
> parity against existing APIs.

---

## Integration plan (keep theme & tokens intact)

Reuse existing primitives everywhere: the `el()` / `mk()` DOM helpers, tokens
`var(--accent)`, `var(--accent-tx)`, `var(--accent-dim)`, `var(--tx-2)`, and
borders `var(--line)` / `var(--line-2)` / `var(--line-3)`, 7px radii, ~12px
font, and the filled-vs-ghost button pattern already used on voice cards.

**Shared audio engine (foundation for #1, #2, #4)**
- Add one reusable `this._audio = new Audio()` (init in `componentDidMount`,
  L1346 area). Helpers: `playUrl(url)`, `pause()`, with `timeupdate`/`ended`/
  `play`/`pause` events wired into `setState` so the existing scrubber/time and
  `isPlaying` icon reflect **real** element state.
- Don't bind src to "whenever `outputJobId` changes" (awkward in this class
  style). Instead **compute the URL inside `playUrl()`/`togglePlay()`** and let
  audio events drive state.
- **Object-URL lifecycle:** for blob previews (uploaded/recorded/emo) call
  `URL.revokeObjectURL` on the prior URL before creating a new one and on
  clear/unmount, so repeated records/uploads don't leak URLs.

**#1 Output player** — template L370–400 + `togglePlay` (L1017 binding):
- `togglePlay()` computes `this.__audioUrl(outputJobId)` and passes it to
  `playUrl()` (or `pause()` when already playing) — no `src` binding on state
  change.
- Bind the existing scrub bar to `currentTime/duration`. No visual change — the
  circular accent play button (L380) stays as-is.

**#2 Voice Library preview** — render at `acts.appendChild` (L1108):
- Add a `mk('Preview', () => this.playUrl('/api/voices/'+id+'/audio'), false)`
  ghost button beside **Use** (L1108). Toggle label to "Stop" while that voice
  is the active source. Same `mk()` styling — zero new CSS.

**#3 Emotion reference upload** — fields block (L738) + `runGenerate` (L1521):
- When `index.emoMode === 'emo_ref'`, render an upload control (reuse the
  `__chooseVoiceFile`/file-input pattern, L1563) storing `state.emoFileBlob`
  + a "Listen" button via `URL.createObjectURL`.
- In `runGenerate`'s form assembly (the `form.append` block at L1526–1530,
  alongside `__appendParams` L1499), append `form.append('emo_audio',
  emoFileBlob, name)` when present. Gate it in `validate()` (L636).

**#4 Reference preview + save-to-library** — ref panel (L939–940, L1472/L1491):
- Add a "Listen" button next to the upload/record controls that plays
  `URL.createObjectURL(refFileBlob)`.
- Add "Save to Library" that POSTs the existing `refFileBlob` to `/api/voices`.
  Reuse the `__chooseVoiceFile` body (L1563) **but** it assumes
  `state.newVoiceName` — so this flow must first prompt for / show a voice-name
  input before POSTing (don't call it with an empty name).

**#5 Transcript view (edit scoped out)** — voice card (L1080–1110):
- **View:** fetch `GET /api/voices/{id}` on expand/select (`prompt_text` is not
  in the list response) and show it collapsibly on cards with `has_transcript`.
- **Edit:** scoped OUT of first pass — requires extending `PATCH
  /api/voices/{id}` (rename-only today) in `webui.py` + a test. Only do this if
  backend changes are explicitly wanted.

**#6 Watermark result panel** — `__detectWatermark` (L1597) + Watermark Lab:
- Store the full detect response in state (not just a toast) and render a
  persistent panel: detected/not, `wm_prob`, attributed model name, run id.
  Reuse the existing run-details box styling (L1585). Clear on next detection.

**#7 `has_caches` badge** — voice card badges (near `duration_s`):
- Show **per-model** cache chips: `Qwen cache`, `Index cache`, `Chatterbox
  cache`, or `Warm for selected model`, using the existing badge style next to
  the duration/transcript chips.

---

## Suggested sequencing
1. Shared audio engine + #1 output playback (unblocks the core UX).
2. #2 voice preview, #4 reference preview (same engine, high value, low risk).
3. #3 emotion reference upload (closes a misleading dead control).
4. #6 watermark result panel (recovers lost backend attribution).
5. #7 cache badge + #5 transcript view (polish; transcript **edit** is scoped
   out — it's the only item needing a `webui.py` PATCH change).

Also clean up stale shadowed mock methods and inline mock handlers (e.g. mock
download at L1192) as each surface is touched.

Each step is additive and reuses existing helpers/tokens — no layout or theme
changes required.
