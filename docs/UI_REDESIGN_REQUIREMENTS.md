# TTS Hub UI Redesign Requirements

## 1. Why Redesign

The current UI is functional, but it behaves more like an internal control panel than a product:

- Too many controls compete at the same level.
- The primary task, generating speech, is visually buried inside technical panels.
- Advanced model parameters are exposed too early.
- Watermarking, voice management, generation, and history are all treated as equal-priority blocks.
- The interface is dark, dense, and technically descriptive, but not especially informative or easy to scan.

The redesign should make the app feel intentional, legible, and useful for both quick generation and deeper model experimentation.

## 2. Primary Product Goals

The new UI should optimize for these jobs:

1. Generate speech quickly with the currently selected model.
2. Understand what each model is good at before using it.
3. Add or reuse a voice reference without friction.
4. Adjust advanced settings only when needed.
5. Review, replay, download, and compare outputs.
6. Run watermark embed/detect workflows without cluttering the default generation flow.

## 3. User Modes

The redesign should explicitly support two modes of use:

### A. Quick Generate

For users who want:

- model selection
- text input
- voice input
- one-click generation
- output playback/download

This mode should hide most technical controls by default.

### B. Lab / Advanced

For users who want:

- model-specific tuning
- generation parameter control
- watermark embed/detect workflows
- repeat testing across models
- voice library reuse

This mode can expose technical controls, but with better grouping and hierarchy.

## 4. Core Information Architecture

The product should be reorganized into 5 top-level surfaces.

### 1. Model Selection

Purpose:

- choose a model
- understand its core capability
- see whether it is ready/loaded/supported for watermarking

Requirements:

- show model name, short human-readable purpose, and capability tags
- avoid exposing raw IDs as the primary label
- show state badges like `Ready`, `Loaded`, `Requires Transcript`, `Watermark Supported`
- allow comparison-level scanning, not just a vertical list of cards

### 2. Generation Workspace

Purpose:

- serve as the main default screen

Requirements:

- make this the visual center of the app
- include text input, voice input, model summary, and generate CTA
- surface required inputs before the user clicks generate
- show only the settings required for the selected model at first
- keep advanced settings collapsed or moved into a secondary panel/drawer

### 3. Voice Library / Voice Input

Purpose:

- manage uploaded, recorded, and saved voices

Requirements:

- unify upload, record, preview, transcript, and saved voices into one coherent module
- clearly distinguish temporary voice input vs saved reusable voice
- show duration, transcript availability, and source type
- support clear empty, loading, selected, recorded, and saved states

### 4. Output & History

Purpose:

- make generated results easy to review and reuse

Requirements:

- output must feel like a first-class result area, not an afterthought
- support playback, download, model label, generation timestamp, and key settings summary
- history should support quick replay and rerun context
- current history should be upgraded from a passive list to a usable session record

### 5. Watermark Lab

Purpose:

- separate provenance workflows from core speech generation

Requirements:

- watermarking should live in its own tab, drawer, or dedicated section
- embedding and detection should be treated as related but distinct tasks
- show run details, supported models, threshold mode, and analysis output clearly
- avoid making watermark controls part of the default generation path unless enabled

## 5. Required Interface Inventory

At minimum, the redesign should define these interfaces.

### A. Main Workspace

Must include:

- model picker
- model capability summary
- voice input area
- text input area
- generate action
- live status / error area
- output player

### B. Advanced Settings Interface

Must include:

- model-specific parameter groups
- collapsible advanced sections
- parameter presets like `Fast`, `Balanced`, `Quality`
- inline help for specialized controls

### C. Voice Manager Interface

Must include:

- upload state
- recording state
- saved voices list
- preview state
- save/delete flows

### D. History / Session Interface

Must include:

- recent generations
- replay/download actions
- basic metadata
- empty state

### E. Watermark Interface

Must include:

- watermark enable state for generation
- run selector
- run details
- test upload/dropzone
- threshold control
- analysis results

### F. System Status Interface

Must include:

- ffmpeg availability
- model load/unload status
- generation loading state
- actionable error messages

## 6. Functional Requirements

The redesign should preserve all current functionality and make it easier to discover.

### Global Functional Requirements

- Persist session state where it is already supported.
- Preserve model-specific validation rules.
- Keep upload, recording, saved voice, and transcript flows intact.
- Preserve generation history.
- Preserve watermark embed and detect flows.
- Keep unload / reset / clear session actions available, but reduce their visual dominance.

### Model-Specific Functional Requirements

The UI must continue to support:

- IndexTTS2 emotion modes and sampling controls
- Chatterbox language, chunking, and enhancement controls
- F5 Roman mode, override rules, speed, seed, and silence options
- CosyVoice mode switching, transcript requirement, and instruct text
- Qwen model variant, auto-transcribe, language, speed, temperature, max tokens
- Pocket voice URL or prompt audio, plus decoding controls
- VoxCPM cached voice or prompt audio + transcript, plus inference parameters

## 7. UX Requirements

### Hierarchy

- The primary action must always be obvious.
- Required fields must read as required before submission.
- Technical controls should appear below the main task, not beside it at equal weight.

### Progressive Disclosure

- Default view should show only the controls needed to get a valid output.
- Advanced controls should be grouped, named, and collapsible.
- Watermark tools should be separated from the standard create flow.

### Feedback

- Loading states must be prominent and specific.
- Errors should explain the fix, not just the problem.
- Success states should confirm what happened and where the result is.

### Empty States

- No model selected
- No voice selected
- No output yet
- No history yet
- No watermark runs found

Each should provide next-step guidance.

## 8. Visual Design Requirements

The redesign should move away from “generic dark dashboard” styling.

### Theme System

Create a semantic token system for:

- app background
- panel/surface levels
- text hierarchy
- accent colors
- border strengths
- status colors
- shadows/glow

This should support multiple themes without rewriting components.

### Visual Direction

Requirements:

- stronger brand identity than the current indigo/purple dark theme
- clearer separation between primary workspace and secondary utilities
- more purposeful typography
- more spacious rhythm and grouping
- less card spam and less equal-weight chrome

Suggested design directions worth exploring:

1. Studio / audio workstation
2. Research lab / model console
3. Minimal creation canvas with expandable technical drawers

### Typography

- Use a more expressive display treatment for headings.
- Keep UI copy highly legible.
- Differentiate technical metadata from user-facing labels.

### Motion

- Use subtle transitions for panel expansion, status changes, and output reveal.
- Respect `prefers-reduced-motion`.
- Avoid decorative motion that slows down generation workflows.

## 9. Accessibility Requirements

The redesign should fix current accessibility weaknesses and use them as hard requirements.

Required:

- icon-only controls need accessible labels
- form controls need labels or ARIA labels
- visible focus states on all interactive elements
- status/error updates should use polite live regions where appropriate
- strong contrast in all themes
- keyboard navigation for the core flow
- semantic headings and landmarks

## 10. Responsive Requirements

The current UI technically collapses on small screens, but the structure is still desktop-first.

The redesign should define:

- desktop layout for power use
- tablet layout with stacked panels
- mobile layout focused on quick generation only

On mobile:

- advanced settings should move into sheets/drawers
- history and watermarking should not compete with the main form
- recording/upload controls must remain easy to tap

## 11. Current UI Problems To Solve

These are the main design problems visible in the current implementation.

### Structural Problems

- Sidebar model list consumes a lot of space without giving enough decision support.
- Generation, settings, watermarking, and history compete in one long page.
- Advanced settings dominate too early.
- Watermark tools feel bolted on rather than designed into the product.

### Content Problems

- Labels are technically correct but not product-friendly.
- Raw model IDs leak into visible UI.
- Required inputs are explained in hints instead of enforced through layout.
- Status communication is too small and easy to miss.

### Interaction Problems

- Validation happens late, after clicking generate.
- History is mostly archival, not actionable.
- Voice workflows are split across several controls without strong hierarchy.
- Destructive actions are highly visible in the header even though they are secondary tasks.

### Style Problems

- The interface relies on a familiar dark-SaaS pattern without much product character.
- Accent color and glow are overused.
- Most panels have the same visual weight, so nothing feels primary.

## 12. Recommended Redesign Strategy

Do not start by restyling the existing layout 1:1.

Instead:

1. Redefine the product around a primary workspace.
2. Separate quick-generate from advanced experimentation.
3. Split watermarking into its own intentional surface.
4. Turn voice management into a proper reusable module.
5. Introduce theme tokens and layout primitives before exploring visual themes.

## 13. Deliverables For The Redesign Phase

Before implementation, produce:

1. Sitemap / surface map
2. Wireframes for desktop + mobile
3. Component inventory
4. Theme token system
5. 2 to 3 visual directions
6. Final high-fidelity screens for:
   - main workspace
   - advanced settings
   - voice manager
   - output/history
   - watermark lab

## 14. Success Criteria

The redesign is successful if:

- a new user can generate audio without reading dense instructions
- advanced users still have access to all model controls
- required fields are obvious before submission
- output review feels like a core workflow
- watermarking feels intentional rather than bolted on
- the app has a distinct visual identity instead of a generic dashboard look
