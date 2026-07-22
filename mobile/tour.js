/* TTS Hub mobile - touch-first, per-surface guided tutorial. */
(() => {
  "use strict";

  const $ = (selector) => document.querySelector(selector);
  const PAD = 7;
  const surfaceName = () => {
    const visible = document.querySelector("main.surface:not(.hidden)");
    return visible ? visible.id.replace("surface-", "") : "generate";
  };
  const visible = (element) => {
    if (!element) return false;
    const rect = element.getBoundingClientRect();
    const style = getComputedStyle(element);
    return rect.width > 0 && rect.height > 0 && style.display !== "none" && style.visibility !== "hidden";
  };
  const pick = (...selectors) => {
    for (const selector of selectors) {
      const element = $(selector);
      if (visible(element)) return element;
    }
    return null;
  };

  const TOURS = {
    generate: [
      {
        find: () => pick("#tabbar"),
        title: "Move around the mobile hub",
        body: "Generate speech, manage reference voices, review jobs, and verify watermarks from these four tabs.",
      },
      {
        find: () => pick("#model-row"),
        title: "Choose a model",
        body: "Qwen is recommended for fast daily use. Open this selector whenever you need a different quality, language, or expressiveness trade-off.",
      },
      {
        find: () => pick("#voice-section"),
        title: "Choose the reference voice",
        body: "Select a saved voice, upload audio, or record a reference directly from the phone. Required items are labelled before generation.",
      },
      {
        find: () => pick("#script"),
        title: "Write what you want to say",
        body: "Type or paste the message here. Your text stays in place while a generation is running.",
      },
      {
        find: () => pick("#options-row"),
        title: "Keep advanced controls optional",
        body: "Output format, watermarking, and model-specific settings live here. The recommended defaults are suitable for routine use.",
      },
      {
        find: () => pick("#runbar"),
        title: "Run the generation",
        body: "This button explains anything still required. Once ready, start the job and you can move to another tab while the laptop works.",
      },
      {
        find: () => pick("#mp-favorite", "#runbar"),
        title: "Save a Quick Phrase",
        body: "After a result is ready, use the star in the mobile player and give it a short name. The finished audio is kept for easy reuse.",
      },
      {
        find: () => pick("#phrase-section"),
        title: "Play a Quick Phrase",
        body: "Saved phrases appear at the top of Generate. Tap one to replay it without starting another generation.",
      },
      {
        find: () => pick("#mobile-tour-launch"),
        title: "Replay help anytime",
        body: "Open the tutorial again from this question-mark button. Each mobile tab has its own short guide.",
      },
    ],
    voices: [
      {
        find: () => pick("#add-voice-btn"),
        title: "Add a reference voice",
        body: "Save a short, clean recording and its transcript so compatible models can reuse it without repeating setup.",
      },
      {
        find: () => pick("#voice-list"),
        title: "Manage saved voices",
        body: "Preview, use, edit, or permanently delete references here. Use sends a voice directly back to Generate.",
      },
    ],
    jobs: [
      {
        find: () => pick("#job-filters"),
        title: "Filter generation history",
        body: "Separate saved phrases, active work, completed results, and failures to find the run you need quickly.",
      },
      {
        find: () => pick("#job-list"),
        title: "Open a run",
        body: "Select a job to replay audio, restore its settings, rename it, or save a completed result as a Quick Phrase.",
      },
    ],
    verify: [
      {
        find: () => pick("#verify-tabs"),
        title: "Choose verification audio",
        body: "Upload a clip or record one on the phone to check for a TTS Hub watermark.",
      },
      {
        find: () => pick("#verify-advanced-toggle"),
        title: "Technical evidence stays optional",
        body: "The result leads with a plain conclusion. Detector run and threshold controls remain here when you need them.",
      },
      {
        find: () => pick("#verify-btn"),
        title: "Check the watermark",
        body: "Run verification after choosing audio. A detected result can also identify the likely source model.",
      },
    ],
  };

  let root = null;
  let spotlight = null;
  let card = null;
  let steps = [];
  let index = 0;
  let previousFocus = null;

  function resolveSteps(name) {
    return (TOURS[name] || []).filter((step) => step.find());
  }

  function build() {
    root = document.createElement("div");
    root.className = "mobile-tour";
    root.innerHTML = `
      <div class="mobile-tour-spotlight" aria-hidden="true"></div>
      <section class="mobile-tour-card" role="dialog" aria-modal="true" aria-labelledby="mobile-tour-title">
        <div class="mobile-tour-kicker">
          <span class="mobile-tour-count"></span>
          <button class="mobile-tour-close" type="button" aria-label="Exit tutorial">✕</button>
        </div>
        <h2 class="mobile-tour-title" id="mobile-tour-title"></h2>
        <p class="mobile-tour-copy"></p>
        <div class="mobile-tour-progress" aria-hidden="true"></div>
        <div class="mobile-tour-actions">
          <button class="mobile-tour-back" type="button">Back</button>
          <button class="mobile-tour-next" type="button">Next</button>
        </div>
      </section>`;
    spotlight = root.querySelector(".mobile-tour-spotlight");
    card = root.querySelector(".mobile-tour-card");
    root.querySelector(".mobile-tour-close").addEventListener("click", close);
    root.querySelector(".mobile-tour-back").addEventListener("click", () => show(index - 1));
    root.querySelector(".mobile-tour-next").addEventListener("click", () => {
      if (index === steps.length - 1) close();
      else show(index + 1);
    });
    root.addEventListener("click", (event) => {
      if (event.target === root) card.animate(
        [{ transform: "scale(1)" }, { transform: "scale(1.015)" }, { transform: "scale(1)" }],
        { duration: 180 }
      );
    });
    $("#app").appendChild(root);
  }

  function position(element) {
    element.scrollIntoView({ block: "center", inline: "nearest", behavior: "auto" });
    const rect = element.getBoundingClientRect();
    const left = Math.max(4, rect.left - PAD);
    const top = Math.max(4, rect.top - PAD);
    spotlight.style.left = `${left}px`;
    spotlight.style.top = `${top}px`;
    spotlight.style.width = `${Math.min(innerWidth - left - 4, rect.width + PAD * 2)}px`;
    spotlight.style.height = `${Math.min(innerHeight - top - 4, rect.height + PAD * 2)}px`;
    card.classList.toggle("top", rect.top + rect.height / 2 > innerHeight / 2);
  }

  function show(nextIndex) {
    if (!root || nextIndex < 0 || nextIndex >= steps.length) return;
    index = nextIndex;
    const step = steps[index];
    const element = step.find();
    if (!element) {
      if (index < steps.length - 1) show(index + 1);
      else close();
      return;
    }
    root.querySelector(".mobile-tour-count").textContent = `${index + 1} / ${steps.length}`;
    root.querySelector(".mobile-tour-title").textContent = step.title;
    root.querySelector(".mobile-tour-copy").textContent = step.body;
    root.querySelector(".mobile-tour-progress").innerHTML = steps.map((_, dot) =>
      `<span class="mobile-tour-dot${dot === index ? " active" : ""}"></span>`
    ).join("");
    const back = root.querySelector(".mobile-tour-back");
    back.disabled = index === 0;
    root.querySelector(".mobile-tour-next").textContent = index === steps.length - 1 ? "Finish" : "Next";
    requestAnimationFrame(() => {
      position(element);
      root.querySelector(".mobile-tour-next").focus({ preventScroll: true });
    });
  }

  function start() {
    close();
    steps = resolveSteps(surfaceName());
    if (!steps.length) return;
    previousFocus = document.activeElement;
    index = 0;
    build();
    show(0);
    addEventListener("resize", reflow);
  }

  function close() {
    removeEventListener("resize", reflow);
    if (root) root.remove();
    root = spotlight = card = null;
    if (previousFocus && document.contains(previousFocus)) previousFocus.focus({ preventScroll: true });
    previousFocus = null;
  }

  function reflow() {
    if (!root || !steps[index]) return;
    const element = steps[index].find();
    if (element) position(element);
  }

  document.addEventListener("click", (event) => {
    if (event.target.closest && event.target.closest("#mobile-tour-launch")) start();
  });
  document.addEventListener("keydown", (event) => {
    if (!root) return;
    if (event.key === "Escape") { event.preventDefault(); close(); return; }
    if (event.key !== "Tab") return;
    const controls = [...card.querySelectorAll("button:not([disabled])")];
    if (!controls.length) return;
    const first = controls[0];
    const last = controls[controls.length - 1];
    if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
    else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
  });
})();
