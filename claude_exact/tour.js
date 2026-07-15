// TTS Hub — Spotlight onboarding tours (one per surface).
// Self-contained: no dependencies, reads theme from the app's [data-theme] root.
// Generate view anchors on [data-tour="…"] attributes in index.html; other
// surfaces are rendered at runtime, so their steps resolve elements live via
// data-fk/data-scroll-keep hooks and text lookup.
(() => {
  'use strict';

  const Z = 2147480000;
  const PAD = 8;          // spotlight padding around target
  const RADIUS = 12;      // spotlight corner radius
  const TRAVEL_MS = 340;  // spotlight travel duration between steps
  const GAP = 14;         // card distance from spotlight
  const MARGIN = 12;      // min distance from viewport edge

  // ------------------------------------------------------------- resolvers
  const anchorEl = (name) => document.querySelector(`[data-tour="${name}"]`);
  const fk = (name) => document.querySelector(`[data-fk="${name}"],[data-scroll-keep="${name}"]`);
  const surface = (name) => document.querySelector(`[data-surface="${name}"]`);
  function byText(scopeSel, sel, text) {
    const scope = document.querySelector(scopeSel);
    if (!scope) return null;
    const needle = text.toLowerCase();
    for (const el of scope.querySelectorAll(sel)) {
      if (el.textContent.trim().toLowerCase().includes(needle)) return el;
    }
    return null;
  }

  // ------------------------------------------------------------- tours
  const TOURS = {
    generate: [
      {
        anchor: 'nav', placement: 'right', title: 'Navigate the hub',
        body: 'Five surfaces live here — Generate for synthesis, Voices for your reference library, Jobs for history, Models for engine management, and Watermark for verification.',
      },
      {
        anchor: 'models', placement: 'right', title: 'Pick an engine',
        body: 'Each card is a TTS engine. The dot shows its load state and the chips tell you what it supports. Selecting one reshapes the whole console to match.',
      },
      {
        anchor: 'script', placement: 'bottom', title: 'Write your script',
        body: 'Whatever you type here is what gets spoken. The character counter on the right keeps an eye on model limits.',
      },
      {
        anchor: 'reference', placement: 'bottom', title: 'Give it a voice',
        body: 'Voice-cloning engines need a reference: pick a saved voice, upload a clip, or record one right here. A verbatim transcript makes the clone noticeably better.',
      },
      {
        anchor: 'settings', placement: 'left', title: 'Fine-tune the model',
        body: 'Every engine exposes its own knobs — pacing, temperature, emotion blends. Reset puts everything back to sane defaults.',
      },
      {
        anchor: 'actions', placement: 'top', title: 'Format, watermark, generate',
        body: 'Choose the output format, toggle audio watermarking, then hit Generate. If the button is disabled, the note beside it tells you exactly what is missing.',
      },
      {
        anchor: 'transport', placement: 'top', title: 'Your output lands here',
        body: 'Watch the pipeline phases while a job runs, then play, scrub and download the result. Every clip is archived under Jobs.',
      },
      {
        anchor: 'help', placement: 'right', title: 'Take the tour again',
        body: 'That is the whole loop. Each page has its own quick tour — press this ? (or Shift + /) on any page to replay it.',
      },
    ],
    voices: [
      {
        find: () => { const b = byText('[data-surface="voices"]', 'button', 'add voice'); return b && b.parentElement; },
        placement: 'bottom', title: 'Your voice library',
        body: 'Every reference voice you save lands here, ready to reuse across compatible engines.',
      },
      {
        find: () => byText('[data-surface="voices"]', 'button', 'add voice'),
        placement: 'bottom', title: 'Add a voice',
        body: 'Upload or record a short, clean clip. Adding a verbatim transcript makes cloning noticeably more faithful.',
      },
      {
        find: () => fk('voice-search'),
        placement: 'bottom', title: 'Search and sort',
        body: 'Filter by name, or sort by recency and duration when the library grows.',
      },
      {
        find: () => fk('voices-body'),
        placement: 'top', title: 'Audition and manage',
        body: 'Each card previews its audio. Use sends the voice straight to Generate; Edit and Delete keep the library tidy.',
      },
    ],
    jobs: [
      {
        find: () => { const b = byText('[data-surface="jobs"]', 'button', 'all'); return b && b.parentElement; },
        placement: 'bottom', title: 'Filter the archive',
        body: 'Jump between everything, still-running jobs, finished clips and failures.',
      },
      {
        find: () => fk('jobs-body'),
        placement: 'top', title: 'Every generation, logged',
        body: 'Status, script, engine, format and timing for each run — open a row to inspect it or replay the audio.',
      },
    ],
    models: [
      {
        find: () => fk('models-body'),
        placement: 'top', title: 'Engine roster',
        body: 'One row per engine: load state, device, run count and when it last generated.',
      },
      {
        find: () => byText('[data-surface="models"]', 'button', 'load'),
        placement: 'bottom', title: 'Load and unload',
        body: 'Engines load on demand, but you can warm one up ahead of time — or free its memory when you are done.',
      },
    ],
    watermark: [
      {
        find: () => fk('wm-runs'),
        placement: 'right', title: 'Evaluation runs',
        body: 'Every watermark embed and detection run is listed here, newest first.',
      },
      {
        find: () => fk('wm-detail'),
        placement: 'left', title: 'Run detail',
        body: 'Select a run to inspect its configuration, metrics and verdicts.',
      },
    ],
  };

  const resolveStep = (s) => (s.anchor ? anchorEl(s.anchor) : s.find());

  // ------------------------------------------------------------- state
  let open = false;
  let steps = [];          // resolved steps for this run
  let activeView = null;
  let idx = 0;
  let mode = 'steps';      // 'welcome' | 'steps' | 'done'
  let hole = null;         // current spotlight rect {x,y,w,h,r}
  let raf = 0;
  let watchTimer = 0;
  let lastFocus = null;
  let root, svg, holeRect, backdropRect, ring, card;

  const reducedMotion = () =>
    window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  const themeRoot = () => document.querySelector('div[data-theme]');
  const theme = () => (themeRoot() && themeRoot().getAttribute('data-theme')) === 'light' ? 'light' : 'dark';
  const dimColor = () => theme() === 'light' ? 'rgba(18,26,40,.48)' : 'rgba(4,6,10,.64)';

  function currentView() {
    const s = document.querySelector('[data-surface]');
    if (s) return s.getAttribute('data-surface');
    return anchorEl('script') ? 'generate' : null;
  }

  // ------------------------------------------------------------- css
  // Kept inside the overlay element (not document.head): the app's <helmet>
  // handling can rewrite the head at any time and would drop injected styles.
  const TOUR_CSS = `
      #tts-tour{position:fixed;inset:0;z-index:${Z};font-family:var(--sans,system-ui);
        opacity:0;transition:opacity .24s ease;overscroll-behavior:contain}
      #tts-tour.on{opacity:1}
      #tts-tour > svg{position:absolute;inset:0;width:100%;height:100%}
      .tts-tour-ring{position:absolute;border:1.5px solid var(--accent-line,rgba(61,220,151,.4));
        border-radius:${RADIUS}px;pointer-events:none;
        box-shadow:0 0 0 4px var(--accent-dim,rgba(61,220,151,.13));
        animation:ttsTourBreathe 2.2s ease-in-out infinite}
      @keyframes ttsTourBreathe{
        0%,100%{box-shadow:0 0 0 3px var(--accent-dim,rgba(61,220,151,.13))}
        50%{box-shadow:0 0 0 7px var(--accent-dim,rgba(61,220,151,.13))}}
      .tts-tour-card{position:absolute;left:0;top:0;width:316px;background:var(--bg-2,#171a20);
        border:1px solid var(--line-2,#333a45);border-radius:12px;box-shadow:var(--shadow,0 8px 30px rgba(0,0,0,.45));
        color:var(--tx,#e9ebef);padding:16px 16px 14px;
        transition:transform .32s cubic-bezier(.4,0,.2,1),opacity .2s ease;opacity:1}
      .tts-tour-card.centered{width:360px}
      .tts-tour-card.compact{width:280px;padding:13px 15px}
      .tts-tour-card.nudge{animation:ttsTourNudge .28s ease}
      @keyframes ttsTourNudge{0%,100%{scale:1}50%{scale:1.025}}
      @keyframes ttsTourIn{from{opacity:0;translate:0 6px}to{opacity:1;translate:0 0}}
      .tts-tour-inner{animation:ttsTourIn .22s ease both}
      .tts-tour-kicker{display:flex;align-items:center;gap:8px;margin-bottom:9px}
      .tts-tour-count{font-family:var(--mono,monospace);font-size:10px;color:var(--tx-3,#6b7280);letter-spacing:.06em}
      .tts-tour-x{margin-left:auto;width:22px;height:22px;border-radius:6px;display:flex;align-items:center;
        justify-content:center;color:var(--tx-3,#6b7280);font-size:13px;line-height:1}
      .tts-tour-x:hover{background:var(--bg-hover,#232832);color:var(--tx,#e9ebef)}
      .tts-tour-title{font-size:14px;font-weight:700;margin:0 0 6px}
      .tts-tour-body{font-size:12.5px;line-height:1.55;color:var(--tx-2,#9aa2ad);margin:0 0 14px}
      .tts-tour-foot{display:flex;align-items:center;gap:5px}
      .tts-tour-dot{width:6px;height:6px;border-radius:4px;background:var(--line-3,#414a57);
        padding:0;transition:width .25s cubic-bezier(.4,0,.2,1),background .2s}
      .tts-tour-dot:hover{background:var(--tx-3,#6b7280)}
      .tts-tour-dot.on{width:16px;background:var(--accent,#3ddc97)}
      .tts-tour-btn{font-size:12px;font-weight:600;padding:7px 13px;border-radius:7px;
        color:var(--tx-2,#9aa2ad);transition:background .12s,color .12s}
      .tts-tour-btn:hover{background:var(--bg-hover,#232832);color:var(--tx,#e9ebef)}
      .tts-tour-btn[disabled]{opacity:.35;pointer-events:none}
      .tts-tour-btn--go{background:var(--accent,#3ddc97);color:var(--accent-tx,#06140e);margin-left:6px}
      .tts-tour-btn--go:hover{background:var(--accent,#3ddc97);color:var(--accent-tx,#06140e);filter:brightness(1.08)}
      .tts-tour-spacer{margin-left:auto}
      .tts-tour-done{display:flex;align-items:center;gap:11px;text-align:left}
      .tts-tour-done h2{font-size:13px;font-weight:700;margin:0 0 2px}
      .tts-tour-done p{font-size:11.5px;line-height:1.45;color:var(--tx-3,#6b7280);margin:0}
      .tts-tour-check{width:26px;height:26px;flex:0 0 auto;border-radius:50%;
        background:var(--accent-dim,rgba(61,220,151,.13));border:1px solid var(--accent-line,rgba(61,220,151,.4));
        display:flex;align-items:center;justify-content:center}
      .tts-tour-check path{stroke:var(--accent,#3ddc97);stroke-width:1.8;fill:none;
        stroke-linecap:round;stroke-linejoin:round;stroke-dasharray:14;stroke-dashoffset:14;
        animation:ttsTourDraw .35s .12s ease forwards}
      @keyframes ttsTourDraw{to{stroke-dashoffset:0}}
      @media (prefers-reduced-motion: reduce){
        #tts-tour,.tts-tour-card,.tts-tour-dot{transition:none}
        .tts-tour-ring,.tts-tour-inner,.tts-tour-card.nudge,.tts-tour-check path{animation:none}
        .tts-tour-check path{stroke-dashoffset:0}}
    `;

  // ------------------------------------------------------------- dom
  function build() {
    // defensive: never allow two overlays (a stale one can survive an
    // interrupted close if start/close interleave)
    document.querySelectorAll('#tts-tour').forEach((n) => n.remove());
    root = document.createElement('div');
    root.id = 'tts-tour';
    root.setAttribute('data-theme', theme()); // so CSS-var overrides resolve inside
    // critical positioning inline as well, in case the <style> is ever delayed
    root.style.cssText = `position:fixed;inset:0;z-index:${Z}`;
    root.innerHTML = `
      <style>${TOUR_CSS}</style>
      <svg aria-hidden="true">
        <defs><mask id="tts-tour-mask">
          <rect x="0" y="0" width="100%" height="100%" fill="#fff"/>
          <rect class="hole" fill="#000" rx="${RADIUS}"/>
        </mask></defs>
        <rect class="dim" x="0" y="0" width="100%" height="100%" mask="url(#tts-tour-mask)"/>
      </svg>
      <div class="tts-tour-ring" hidden></div>
      <div class="tts-tour-card" role="dialog" aria-modal="true" aria-labelledby="tts-tour-title"></div>
    `;
    svg = root.querySelector('svg');
    holeRect = svg.querySelector('.hole');
    backdropRect = svg.querySelector('.dim');
    backdropRect.setAttribute('fill', dimColor());
    ring = root.querySelector('.tts-tour-ring');
    card = root.querySelector('.tts-tour-card');
    root.addEventListener('mousedown', (e) => {
      if (!card.contains(e.target)) {
        card.classList.remove('nudge');
        void card.offsetWidth; // restart animation
        card.classList.add('nudge');
      }
    });
    // mount on <html>, not <body>: the dc-runtime React root owns body's
    // children and can drop or re-render over foreign nodes there
    document.documentElement.appendChild(root);
  }

  function destroy() {
    cancelAnimationFrame(raf);
    clearInterval(watchTimer);
    window.removeEventListener('resize', onReflow);
    window.removeEventListener('scroll', onReflow, true);
    if (root) { root.remove(); root = null; }
    if (lastFocus && document.contains(lastFocus)) { try { lastFocus.focus(); } catch (_) {} }
    lastFocus = null;
    open = false;
    activeView = null;
  }

  // ------------------------------------------------------------- geometry
  function measure(el) {
    const r = el.getBoundingClientRect();
    return { x: r.left - PAD, y: r.top - PAD, w: r.width + PAD * 2, h: r.height + PAD * 2, r: RADIUS };
  }

  const centerHole = () => ({ x: innerWidth / 2, y: innerHeight / 2, w: 0, h: 0, r: 0 });

  function applyHole(h) {
    hole = h;
    holeRect.setAttribute('x', h.x); holeRect.setAttribute('y', h.y);
    holeRect.setAttribute('width', Math.max(0, h.w)); holeRect.setAttribute('height', Math.max(0, h.h));
    holeRect.setAttribute('rx', h.r);
    const vis = h.w > 1 && h.h > 1;
    ring.hidden = !vis;
    if (vis) {
      ring.style.left = h.x + 'px'; ring.style.top = h.y + 'px';
      ring.style.width = h.w + 'px'; ring.style.height = h.h + 'px';
    }
  }

  let tweenFallback = 0;
  function tweenHole(to, dur, done) {
    cancelAnimationFrame(raf);
    clearTimeout(tweenFallback);
    const from = hole || centerHole();
    if (reducedMotion() || dur <= 0) { applyHole(to); done && done(); return; }
    const t0 = performance.now();
    const ease = (t) => (t < .5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
    let finished = false;
    const finish = () => {
      if (finished) return;
      finished = true;
      clearTimeout(tweenFallback);
      cancelAnimationFrame(raf);
      applyHole(to);
      done && done();
    };
    const frame = (now) => {
      if (finished) return;
      const p = Math.min(1, (now - t0) / dur), e = ease(p), cur = {};
      for (const k of ['x', 'y', 'w', 'h', 'r']) cur[k] = from[k] + (to[k] - from[k]) * e;
      applyHole(cur);
      if (p < 1) raf = requestAnimationFrame(frame);
      else finish();
    };
    raf = requestAnimationFrame(frame);
    // rAF stalls while the window is occluded — never leave the spotlight
    // stranded mid-travel
    tweenFallback = setTimeout(finish, dur + 120);
  }

  function placeCard(target, pref) {
    const cw = card.offsetWidth, ch = card.offsetHeight;
    const vw = innerWidth, vh = innerHeight;
    if (!target) { // centered (welcome / done)
      setCardXY((vw - cw) / 2, (vh - ch) / 2);
      return;
    }
    const fits = {
      right: target.x + target.w + GAP + cw <= vw - MARGIN,
      left: target.x - GAP - cw >= MARGIN,
      bottom: target.y + target.h + GAP + ch <= vh - MARGIN,
      top: target.y - GAP - ch >= MARGIN,
    };
    const order = [pref, 'bottom', 'right', 'top', 'left'].filter((v, i, a) => a.indexOf(v) === i);
    const side = order.find((s) => fits[s]);
    let x, y;
    if (side === 'right') { x = target.x + target.w + GAP; y = target.y + target.h / 2 - ch / 2; }
    else if (side === 'left') { x = target.x - GAP - cw; y = target.y + target.h / 2 - ch / 2; }
    else if (side === 'top') { x = target.x + target.w / 2 - cw / 2; y = target.y - GAP - ch; }
    else if (side === 'bottom') { x = target.x + target.w / 2 - cw / 2; y = target.y + target.h + GAP; }
    else { x = (vw - cw) / 2; y = (vh - ch) / 2; }
    setCardXY(
      Math.min(Math.max(x, MARGIN), vw - MARGIN - cw),
      Math.min(Math.max(y, MARGIN), vh - MARGIN - ch),
    );
  }

  function setCardXY(x, y) {
    card.style.transform = `translate(${Math.round(x)}px, ${Math.round(y)}px)`;
  }

  // ------------------------------------------------------------- rendering
  function dotsHtml() {
    return steps.map((_, i) =>
      `<button class="tts-tour-dot${i === idx ? ' on' : ''}" data-jump="${i}" aria-label="Step ${i + 1}"></button>`,
    ).join('');
  }

  function renderStep() {
    const step = steps[idx];
    const last = idx === steps.length - 1;
    card.classList.remove('centered', 'compact');
    card.innerHTML = `
      <div class="tts-tour-inner">
        <div class="tts-tour-kicker">
          <span class="tts-tour-count">${String(idx + 1).padStart(2, '0')} / ${String(steps.length).padStart(2, '0')}</span>
          <button class="tts-tour-x" data-act="skip" aria-label="Exit tour" title="Exit (Esc)">✕</button>
        </div>
        <h2 class="tts-tour-title" id="tts-tour-title">${step.title}</h2>
        <p class="tts-tour-body">${step.body}</p>
        <div class="tts-tour-foot">
          ${dotsHtml()}
          <span class="tts-tour-spacer"></span>
          <button class="tts-tour-btn" data-act="back" ${idx === 0 ? 'disabled' : ''}>Back</button>
          <button class="tts-tour-btn tts-tour-btn--go" data-act="next">${last ? 'Finish' : 'Next'}</button>
        </div>
      </div>`;
    card.querySelector('[data-act="next"]').focus({ preventScroll: true });
  }

  function renderDone() {
    card.classList.add('centered', 'compact');
    card.innerHTML = `
      <div class="tts-tour-inner tts-tour-done">
        <span class="tts-tour-check">
          <svg width="11" height="11" viewBox="0 0 12 12"><path d="M2.5 6.5 L5 9 L9.5 3.5"/></svg>
        </span>
        <div>
          <h2 id="tts-tour-title">You're all set</h2>
          <p>Replay anytime with ? or Shift&thinsp;+&thinsp;/</p>
        </div>
      </div>`;
  }

  // ------------------------------------------------------------- flow
  function goTo(i, instant) {
    // skip over steps whose element vanished since the tour started
    let j = i;
    const dir = i >= idx ? 1 : -1;
    while (j >= 0 && j < steps.length && !resolveStep(steps[j])) j += dir;
    if (j < 0 || j >= steps.length) { endQuiet(); return; }
    idx = j;
    mode = 'steps';
    const el = resolveStep(steps[idx]);
    el.scrollIntoView({ block: 'nearest', behavior: 'auto' });
    // synchronous on purpose: rAF callbacks stall entirely while the window
    // is occluded/backgrounded, which would leave an empty invisible card
    const target = measure(el);
    ring.hidden = true; // hide during travel, reappear on arrival
    tweenHole(target, instant ? 0 : TRAVEL_MS, () => { ring.hidden = false; });
    renderStep();
    placeCard(target, steps[idx].placement);
  }

  function finish() {
    mode = 'done';
    tweenHole(centerHole(), 280);
    renderDone();
    placeCard(null);
    setTimeout(close, 2400);
  }

  function skip() {
    close();
  }

  // tour lost its targets (e.g. user switched pages mid-tour) — bow out silently
  function endQuiet() {
    close();
  }

  function close() {
    if (!root) return;
    root.classList.remove('on');
    setTimeout(destroy, reducedMotion() ? 0 : 250);
  }

  function start(view) {
    if (open) return;
    if (root) destroy(); // a close animation may still be in flight
    const tour = TOURS[view];
    if (!tour) return;
    steps = tour.filter((s) => resolveStep(s));
    if (!steps.length) return;
    open = true;
    activeView = view;
    idx = 0;
    hole = null;
    lastFocus = document.activeElement;
    build();
    applyHole(centerHole());
    void root.offsetWidth; // commit initial opacity:0 so the fade-in transitions
    root.classList.add('on');
    window.addEventListener('resize', onReflow);
    window.addEventListener('scroll', onReflow, true);
    // safety net: app re-renders (clock ticks, job updates) can move targets
    watchTimer = setInterval(() => { if (mode === 'steps') onReflow(); }, 900);
    goTo(0, true);
  }

  function onReflow() {
    if (!open) return;
    if (mode !== 'steps') { placeCard(null); return; }
    const el = resolveStep(steps[idx]);
    if (!el) { goTo(idx + 1, true); return; }
    const target = measure(el);
    if (hole && Math.abs(target.x - hole.x) < 1 && Math.abs(target.y - hole.y) < 1 &&
        Math.abs(target.w - hole.w) < 1 && Math.abs(target.h - hole.h) < 1) return;
    cancelAnimationFrame(raf);
    applyHole(target);
    ring.hidden = false;
    placeCard(target, steps[idx].placement);
  }

  // ------------------------------------------------------------- events
  document.addEventListener('click', (e) => {
    const launch = e.target.closest && e.target.closest('[data-tour-launch]');
    if (launch) { start(currentView() || 'generate'); return; }
    if (!root) return;
    const btn = e.target.closest && e.target.closest('[data-act],[data-jump]');
    if (!btn || !root.contains(btn)) return;
    const act = btn.getAttribute('data-act');
    if (act === 'skip') skip();
    else if (act === 'back') goTo(idx - 1);
    else if (act === 'next') { if (idx === steps.length - 1) finish(); else goTo(idx + 1); }
    else if (btn.hasAttribute('data-jump')) goTo(+btn.getAttribute('data-jump'));
  });

  document.addEventListener('keydown', (e) => {
    if (open) {
      if (e.key === 'Escape') { e.preventDefault(); skip(); return; }
      if (mode !== 'steps') return;
      if (e.key === 'ArrowRight' || e.key === 'Enter') {
        if (e.key === 'Enter' && e.target.closest && e.target.closest('.tts-tour-card')) return; // let buttons work
        e.preventDefault();
        if (idx === steps.length - 1) finish(); else goTo(idx + 1);
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault(); goTo(idx - 1);
      } else if (e.key === 'Tab') {
        // focus trap inside the card
        const focusables = card.querySelectorAll('button:not([disabled])');
        if (!focusables.length) return;
        const first = focusables[0], last = focusables[focusables.length - 1];
        if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
        else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
        else if (!card.contains(document.activeElement)) { e.preventDefault(); first.focus(); }
      }
      return;
    }
    // Shift+/ opens the current page's tour (unless typing in a field)
    if (e.key === '?' && !e.metaKey && !e.ctrlKey && !e.altKey) {
      const t = e.target;
      const typing = t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable);
      if (!typing) { e.preventDefault(); start(currentView() || 'generate'); }
    }
  });

  // keep overlay theme + dim color in sync with the app's theme toggle
  const themeWatcher = new MutationObserver(() => {
    if (!root) return;
    root.setAttribute('data-theme', theme());
    backdropRect.setAttribute('fill', dimColor());
  });

  // ------------------------------------------------------------- boot
  // Tours are strictly opt-in: only the ? button or Shift+/ starts one.
  function boot() {
    const tr = themeRoot();
    if (tr) themeWatcher.observe(tr, { attributes: true, attributeFilter: ['data-theme'] });
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
