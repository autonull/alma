Design a self-contained GUI for experimenting and using the system - toolkit agnostic.
* Rich, detailed, animated visualization and feedback
* Intuitive ergonomic usability
* Real-time controls for anything that can be controlled online
* Experiment setup (architecture, etc...)
* Plug-and-play Reward environments, for interactive curriculum training
* Feels like a Digital Audio Workstation UI
* Feels Gamified: in the sense of a casual open-ended simulation game that requires no up-front knowledge from the user
* Load, Save, Checkpoint (with tags for quick)

----

# ALMA Studio — Interface Design

## Visual Language & Aesthetic Philosophy

The interface borrows its soul from two unlikely parents: a professional DAW (think Ableton Live) and a city-builder simulation game. The color palette is near-black with deep navy surfaces, teal as the primary action color, amber for live/hot signals, and purple for latent/abstract state. Everything glows faintly when alive. The typography is monospaced for data, geometric sans for labels — precise, not sterile. The whole thing feels like a cockpit you grow comfortable inside over time, not a form you fill out.

The guiding metaphor is **the agent as a living organism you are cultivating**, not a script you are configuring. Controls breathe. Metrics pulse. The latent space actually moves.

---

## Global Layout

The window is divided into five persistent zones, arranged like a recording studio's signal chain read left-to-right, top-to-bottom:

**Top strip — The Transport Bar.** Running across the full width. Holds the project name (editable inline), a master run/pause/stop control (large, satisfying, center-aligned), a global step counter and elapsed time, a BPM-style "steps per second" readout that pulses with each loop tick, and the session state badge (Idle / Warming / Running / Collapsed / Converged). Far right: Save, Checkpoint (with a tag field), Load, and an Export button. The transport bar never scrolls away.

**Left column — The Architecture Panel.** Roughly one-fifth of the width, full height below the transport bar. This is the patch bay / rack. Three stacked module slots: Belief Engine, Planner, Bridge. Each slot is a card with a colored left border (frozen blue for the engine, teal for the planner, amber for the bridge). Cards can be collapsed to just their header. Clicking a card expands it to show its parameters — all editable as sliders or dropdowns while stopped, grayed-out-but-visible while running. A subtle lock icon appears on the Belief Engine card with a tooltip: "Weights sealed at init." Swapping a module (changing the engine model, swapping the planner variant) triggers a brief animated "hot-swap" transition rather than a hard reset — the card slides out, the new one slides in.

**Center — The Stage.** The dominant area, roughly three-fifths of the width. This is split vertically into two regions:

- *Upper Stage:* The primary live visualization canvas. Fills most of the center.
- *Lower Stage:* A scrolling token stream with inline reward annotation, plus a tabbed panel for secondary plots.

**Right column — The Reward Lab.** Roughly one-fifth of the width. The plug-and-play environment zone. More below.

**Bottom strip — The Timeline.** A horizontal scrollable history of reward totals, checkpoints (shown as vertical markers with tag labels), and mode-switch events — exactly like a DAW's arrangement timeline. Scrubbing it doesn't rewind the agent, but it does scrub the visualization playhead so you can review any past moment. Checkpoints glow with a small tag bubble.

---

## The Stage — Live Visualization Canvas

The upper stage is the heart of the interface and the thing that makes the agent feel alive. It offers three swappable "scene" views, toggled by tabs at the top edge of the canvas:

**Scene 1 — Latent Space (default).** A 2D projection (PCA or UMAP, selectable) of the agent's latent goal vectors over time. Each point is a past step, colored by reward value on a cold-to-warm spectrum (deep blue → teal → amber → hot coral). The most recent point is larger and surrounded by a soft glow ring that pulses with each new step. A faint trail connects sequential points. When the agent is in hierarchical mode, sub-goal points appear as smaller satellites orbiting the parent goal point. The target embedding (if goal-conditioned) appears as a fixed glowing star the cloud is drifting toward. Mode collapse looks like a tight cluster. Healthy exploration looks like a spreading nebula. The whole thing slowly rotates on a subtle auto-orbit so it never feels static.

**Scene 2 — Steering X-Ray.** A cross-sectional view of the belief engine's transformer layers, shown as a horizontal stack of layer slabs. The injection point glows teal, with the steering vector visualized as a colored column entering the residual stream. The alpha (magnitude) controls how thick and bright this column appears — you can literally see how hard the planner is pushing. Animated particles flow left-to-right through the layer stack representing token generation. When the planner switches mode, the injection layer can shift, and you watch the glow migrate up or down the stack.

**Scene 3 — Memory Constellation.** The memory buffer visualized as a circular constellation — each slot is a node arranged around a ring, connected to its neighbors. Node brightness encodes outcome quality. The inner space shows a cosine-distance heatmap between all slots: tight clusters indicate repetition, spread indicates diversity. A novelty meter on the side tracks the mean distance from new outcomes to the existing memory manifold in real time.

All three scenes share a common overlay: a semi-transparent reward decomposition bar that floats at the top of the canvas, showing the live split between novelty, coherence, depth bonus, and relevance contributions — like a spectrum analyzer, updated each step.

---

## The Token Stream

Below the stage canvas, a continuous scroll of generated text flows upward (like a chat log in reverse, newest at the bottom). Each token is colored by its per-token coherence contribution — high-coherence tokens are cool white, incoherent tokens drift toward coral. Clicking any token expands a small popover showing its probability under the steered vs. unsteered distribution. When the agent produces something with notably high novelty, a small shimmer animation sweeps across that line. When it repeats an n-gram, those tokens get a faint diagonal strikethrough.

---

## The Right Column — Reward Lab

This is the "plug-and-play" zone, designed to feel like a pedalboard or an effects rack in a DAW. It contains a scrollable stack of Reward Environment cards — each one is a named preset that configures the reward function with a particular personality.

Each card shows:
- A name and one-line description ("Pure Curiosity", "Concept Cartographer", "Style Hunter", "Boundary Prober", "Multi-Target Synthesis")
- A small live sparkline of the reward this environment has been producing
- A single drag-to-activate region

Dragging a card from the library into the active slot at the top of the column swaps the reward environment live (if running, it fades out the old weights and fades in the new over ~10 steps, so you can watch the agent react). The active environment card expands to show its specific controls — for example, "Concept Cartographer" reveals a text field for the goal phrase and a cosine similarity gauge; "Style Hunter" reveals a style-class selector and a sliding threshold.

Below the active slot, a "Build Your Own" card opens an equation editor — five labeled sliders (novelty weight, coherence weight, repetition penalty, depth bonus, relevance weight) plus a curriculum scheduler toggle that lets you draw a ramp-up curve for any weight over time (like an envelope in a synthesizer). When you modify weights live, a ghosted "before" curve stays visible in the sparkline so you can see what changed.

---

## The Architecture Panel — Detail

Each module card in the left column contains:

**Belief Engine card:** Model selector (dropdown with the three SmolLM2 variants plus a "Custom HuggingFace ID" text field), max new tokens slider, temperature slider, a "Verify Purity" button that runs a hash comparison of the frozen weights and returns a pass/fail badge. The card border pulses faintly blue at each generation step — the frozen engine's heartbeat.

**Planner card:** Hidden dim, number of layers, noise std, and mode (flat / hierarchical / auto-switch). A small "policy entropy" gauge lives here — when entropy collapses, the gauge goes red and a warning icon appears. An "Inject Noise" nudge button gives the planner a one-time perturbation to break attractors.

**Bridge card:** Injection layer selector — shown not as a dropdown but as a miniature layer stack diagram where you click or drag to select which layers receive steering. Multi-layer selection is supported; selected layers glow teal. Alpha source toggle (planner-driven vs. standalone parameter). Initial alpha slider.

---

## Checkpoint & Session Management

The Save/Checkpoint system is modeled after game save slots, not file dialogs. Clicking "Checkpoint" drops a marker on the timeline and opens a tiny tag bubble — type a label ("first stable run", "post mode-switch", "goal-conditioned start") and press enter. The checkpoint appears immediately on the timeline as a glowing pin.

The Load panel (accessible from the transport bar) shows a grid of checkpoint cards, each with: the tag label, a thumbnail of the latent space at that moment, key metrics (step count, avg reward, mode), and a "Branch from here" button that forks the session without discarding the current one. Multiple branches can be open simultaneously as tabs along the top of the stage — exactly like browser tabs or DAW arrangement windows.

---

## Gamification Layer

Several design choices make the system feel like a simulation game rather than a research tool:

**Agent "biome" status.** A small persistent widget in the transport bar displays a one-word state derived from the agent's behavioral signature: *Wandering, Drilling, Looping, Crystallizing, Resonating, Collapsing*. This is computed from recent trajectory entropy, mode dwell time, and reward trend — but the user just sees the evocative word and a matching ambient color shift across the entire interface (the border glow of all panels subtly shifts to match the biome).

**Discovery feed.** A narrow ticker at the very bottom of the screen (below the timeline) logs notable moments as short event cards: "🔍 High-novelty region found at step 847", "⚡ Mode switched to hierarchical", "🌀 Memory diversity peak", "⚠️ Coherence floor approaching". These disappear after a few seconds like toast notifications but also accumulate in a collapsible log.

**Reward unlocks.** The Reward Lab library starts with two environments unlocked. Others are shown as locked cards with a hint: "Run 500 steps with avg novelty > 0.3 to unlock Concept Cartographer." This is entirely cosmetic and skippable, but it gives new users a progression scaffold — a reason to understand what novelty actually means before being handed a goal-conditioning tool.

**Alpha "instrument" feel.** When the agent is running, the alpha (steering magnitude) value is shown as a large glowing ring around the main visualization — bigger ring, harder steer. The ring breathes with each step. You can grab the ring's edge and drag it to manually override alpha, exactly like grabbing a knob.

---

## Ergonomic Details

**Keyboard-first transport:** Spacebar = run/pause. Enter = checkpoint (opens tag field). Cmd+Z = undo last config change (not agent state — just parameter edits). Escape = emergency stop with confirmation.

**Every slider has a reset-to-default double-click.**

**Hover on any metric** reveals a tooltip with: what it measures, what healthy range looks like, and what to do if it's wrong. This is the "no up-front knowledge required" guarantee — the interface teaches itself through use.

**Dark/dim mode:** A "cinematic" toggle dims the architecture and reward panels and expands the stage to nearly full width — for when you just want to watch the agent think.

**Multi-monitor:** The stage can be torn off as a separate window so the visualization runs full-screen on one monitor while controls stay on another.

----

# ALMA Studio: GUI Design Specification
### *A Toolkit-Agnostic Interface for Autonomous Cognitive Architecture Experimentation*

> **Design Philosophy**: "Discover, Don't Configure" — The interface should feel like piloting a curious mind, not debugging a neural network. Every control has a visual consequence; every metric tells a story.

---

## 🎛️ Core Layout: DAW-Inspired Workspace

```
┌─────────────────────────────────────────────────────────────┐
│  [🚀 RUN] [⏸️ PAUSE] [⏭️ STEP] [⏪ RESET]  │  🔍 Search  │ ⚙️  │
├─────────────────────────────────────────────────────────────┤
│  PROJECT: "Curiosity_v3"  │  🎯 Goal: "renewable energy"   │
├─────────────┬─────────────┬─────────────┬───────────────────┤
│             │             │             │                  │
│  🧠 BELIEF  │  🎭 PLANNER │  🌉 BRIDGE  │  💭 MEMORY      │
│  ENGINE     │  MODULE     │  STEERING   │  BUFFER         │
│             │             │             │                  │
│  • Layer    │  • Latent   │  • Alpha    │  • Intent →     │
│    viz      │    trajectory│    slider   │    Outcome      │
│  • Activation│  • Mode    │  • Injection│  • Scrollable   │
│    heatmap  │    toggle   │    points   │    timeline     │
│  • Token    │  • Subgoal │  • Vector   │  • Diversity    │
│    stream   │    counter  │    preview  │    metric       │
│             │             │             │                  │
├─────────────┴─────────────┴─────────────┴───────────────────┤
│  🎚️ REWARD MIXER (DAW-Style Tracks)                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Novelty    [████████░░] 1.0  [✚] [🗑️] [🔗 plugin] │   │
│  │ Coherence  [██████████] 0.5  [✚] [🗑️] [🔗 plugin] │   │
│  │ Relevance  [████░░░░░░] 0.0  [✚] [🗑️] [🔗 plugin] │   │
│  │ Depth      [███░░░░░░░] 0.2  [✚] [🗑️] [🔗 plugin] │   │
│  │ Repetition [████████░░] 2.0  [✚] [🗑️] [🔗 plugin] │   │
│  └─────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│  📊 LIVE VISUALIZATION CANVAS (Animated, Interactive)      │
│  • Latent Space Trajectory (UMAP/PCA)                      │
│  • Reward Signal Waveform                                  │
│  • Text Generation Stream with Novelty/Coherence Coloring  │
│  • Steering Vector Injection Animation                     │
├─────────────────────────────────────────────────────────────┤
│  🎮 GAMIFICATION OVERLAY (Toggleable)                      │
│  • Curiosity Meter: [▰▰▰▰▰▱▱▱] 62%                        │
│  • Discovery Badge: 🌟 "First Semantic Pivot!"             │
│  • Exploration Streak: 🔥 12 cycles                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧩 Panel Specifications

### 🧠 Belief Engine Panel
*Visualize the frozen oracle without overwhelming detail*

| Element | Type | Interaction | Visual Feedback |
|---------|------|-------------|----------------|
| **Layer Selector** | Horizontal scrubber | Drag to select injection layer | Highlights selected layer in architecture diagram |
| **Activation Heatmap** | 2D grid (layers × hidden dim) | Hover to inspect neuron group | Color intensity = activation magnitude; animation on steering injection |
| **Token Stream** | Scrollable text view | Click token to see hidden state projection | Tokens color-coded by novelty (blue→purple) and coherence (green→red) |
| **Model Stats** | Readout cards | — | Parameters, context window, current perplexity |

*Animation*: When steering is applied, a subtle "pulse" travels through the selected layer(s) in the architecture diagram.

### 🎭 Planner Module Panel
*Make latent decision-making tangible*

| Element | Type | Interaction | Visual Feedback |
|---------|------|-------------|----------------|
| **Latent Trajectory** | 2D/3D scatter (projected) | Rotate/zoom; click point to inspect goal vector | Path colored by reward; glowing "current goal" marker |
| **Mode Toggle** | Segmented control (Flat/Hierarchical) | Click to switch; planner can also auto-switch | Visual mode indicator + animation of goal structure expanding/collapsing |
| **Policy Heads Readout** | Collapsible cards | Expand to see raw logits/vectors | Real-time updating values with sparkline history |
| **Sub-Goal Tree** | Interactive node graph (hierarchical mode) | Drag to rearrange; right-click to edit | Nodes pulse when active; edges show transition probability |

*Gamification*: When the planner discovers a high-reward latent direction, the trajectory point "bursts" with particle effects and awards a micro-achievement.

### 🌉 Bridge & Steering Panel
*Demystify the projection between spaces*

| Element | Type | Interaction | Visual Feedback |
|---------|------|-------------|----------------|
| **Alpha Slider** | Vertical fader (DAW-style) | Drag; supports fine-tune with Shift | Numerical readout + visual "steering strength" meter |
| **Injection Points** | Layer diagram with toggle dots | Click to enable/disable per-layer injection | Active layers glow; magnitude shown by dot size |
| **Vector Preview** | Dual-arrow visualization | Hover to see component values | Planner-space vector → projection → engine-space vector animation |
| **Projection Matrix** | Heatmap (optional expert view) | Toggle visibility | Shows learned mapping density |

*Animation*: When a steering vector is injected, show a smooth morph from latent goal → projected vector → activation addition in the belief engine.

### 💭 Memory Buffer Panel
*Make history navigable and insightful*

| Element | Type | Interaction | Visual Feedback |
|---------|------|-------------|----------------|
| **Intent→Outcome Timeline** | Horizontal scrollable cards | Click to expand; drag to reorder (for curriculum) | Color-coded by reward; novelty/coherence mini-sparklines |
| **Diversity Metric** | Circular progress + trend arrow | Hover for pairwise distance distribution | Animates as buffer updates |
| **Search/Filter** | Text input + tag chips | Filter by reward range, topic, mode | Highlights matching entries; dims others |
| **Export Selection** | Button + format dropdown | Export selected memories as JSON/CSV | Confirmation animation + copy-to-clipboard feedback |

*Ergonomic touch*: Swipe gestures (on touch) or keyboard arrows to navigate memory entries quickly.

### 🎚️ Reward Mixer (The "Console")
*DAW-inspired real-time reward engineering*

```
[ TRACK HEADER ]───────────────────────────────────────┐
│ 🎛️ Novelty Plugin v2.1          [🔗] [⚙️] [🗑️]      │
├───────────────────────────────────────────────────────┤
│ Weight: [████████░░] 1.0 ────●─────── [0.0 ── 5.0]   │
│ Scale:  [██████░░░░] 5.0 ────●─────── [1.0 ── 20.0]  │
│                                                                       │
│ [LIVE SIGNAL] ▁▂▃▅▆▇▆▅▃▂▁ (sparkline, updates per step)              │
│                                                                       │
│ [PLUGIN SLOTS]                                                        │
│ • Input: Memory Buffer, Outcome Embedding                            │
│ • Output: Scalar reward contribution                                 │
└─────────────────────────────────────────────────────────────────────┘
```

**Key Interactions**:
- **Faders**: Smooth, momentum-enabled dragging; right-click for numerical entry
- **Plugin Chain**: Drag-and-drop reward components into tracks; reorder to change computation sequence
- **Bypass Toggle**: Click track header to temporarily disable without deleting
- **A/B Compare**: Hold `Alt` while adjusting to preview changes before committing
- **Preset Manager**: Save/load reward configurations as "Curriculum Packs"

*Visual Feedback*: Each track's signal waveform animates in real-time; when a component dominates the total reward, its track gently pulses.

---

## 🎮 Gamification Layer (Optional Overlay)

Designed to lower entry barrier without compromising expert control.

### 🌟 Curiosity Dashboard
```
┌─────────────────────────────┐
│   🧭 EXPLORATION STATUS    │
├─────────────────────────────┤
│  Curiosity Meter:          │
│  [▰▰▰▰▰▰▱▱▱▱] 58%         │
│  "Balanced novelty & coherence" │
│                            │
│  🏆 Recent Discoveries:    │
│  • 🌱 First semantic pivot │
│  • 🔗 Cross-topic bridge   │
│  • 📈 10-step coherence run│
│                            │
│  🔥 Streak: 12 cycles      │
│  🎯 Next Milestone:        │
│  "Sustain topic for 15 steps" │
└─────────────────────────────┘
```

### 🎨 Visual Language
- **Discovery Events**: When novelty spikes *and* coherence holds, trigger a subtle particle burst in the latent trajectory view
- **Progressive Disclosure**: New users see simplified controls with tooltips; "Expert Mode" toggle reveals advanced parameters
- **Narrative Framing**: Onboarding as "Mission Briefing" — "Your agent is curious. Help it explore without losing coherence."
- **Casual Aesthetic**: Soft gradients, rounded corners, gentle animations; optional "Cyberpunk" or "Minimal" themes

### 🎁 Achievement System (Non-Intrusive)
| Badge | Trigger | Visual Reward |
|-------|---------|--------------|
| 🌱 First Steps | Complete 10 autonomy cycles | Subtle confetti in canvas |
| 🧭 Balanced Explorer | Maintain novelty 0.2–0.6 for 50 steps | Curiosity meter gains golden border |
| 🔗 Bridge Builder | Successfully transfer planner to new engine | Unlock "Transfer" visualization theme |
| 🎨 Style Shifter | Generate text in 3 distinct registers | Unlock style-vector library panel |

*Note*: All achievements are opt-out; experts can disable entirely.

---

## 🔄 Real-Time Controls & Feedback Loop

### Transport Controls (DAW-Inspired)
```
[⏮️] [⏪ STEP BACK] [⏸️ PAUSE] [▶️ RUN] [⏭️ STEP] [⏩ RAPID] [⏹️ STOP]
```
- **Step**: Execute one autonomy loop cycle; highlight each stage (Plan→Steer→Generate→Observe→Reward→Learn) with animated flow arrows
- **Rapid**: Run at max FPS with optional frame skipping; show aggregate metrics
- **Pause + Inspect**: Freeze state; enable deep inspection of any tensor/vector

### Live Parameter Binding
Any parameter in YAML config can be exposed as a GUI control:
- **Sliders**: For continuous values (alpha, temperature, reward weights)
- **Toggles**: For boolean flags (pool_enabled, mode_override)
- **Dropdowns**: For categorical choices (engine model, injection layers)
- **Vector Editors**: For latent goals (with semantic search autocomplete)

*Binding Syntax* (internal):
```yaml
# In config.yaml
planner:
  noise_std: { gui: slider, min: 0.0, max: 1.0, default: 0.3, step: 0.05 }
  mode_enabled: { gui: toggle, default: true }
```

### Instant Visual Feedback
- **Parameter Change → Immediate Effect**: Adjust alpha → steering vector magnitude updates → next generation reflects change
- **Undo/Redo Stack**: All GUI changes are reversible; `Ctrl+Z` works across sessions
- **Diff View**: Compare current state to last checkpoint with side-by-side visualizations

---

## 📁 Project Management: Load, Save, Checkpoint

### 🗂️ Project Browser (Sidebar)
```
📁 Projects/
├─ 🔖 Curiosity_v3 (active)
│  ├─ 📄 config.yaml
│  ├─ 🧠 engine: SmolLM2-360M
│  ├─ 🎭 planner: latent_transformer_4L
│  ├─ 💾 checkpoints/
│  │  ├─ 🏷️ initial [2024-06-01 14:30]
│  │  ├─ 🏷️ first-pivot [2024-06-01 15:12] ✨
│  │  └─ 🏷️ coherent-run-12 [2024-06-01 16:45] 🏆
│  └─ 📊 logs/
├─ 📁 transfer-experiments/
└─ 📁 reward-ablations/
```

### 💾 Checkpoint System
- **Auto-Save**: Every N steps (configurable) with rolling window
- **Manual Checkpoint**: `Ctrl+S` → dialog with tag input + optional note
- **Tagging**: Emoji + text tags for quick visual scanning (`✨`, `🏆`, `⚠️`)
- **Diff Checkpoints**: Compare two checkpoints side-by-side (latent trajectories, reward history, generated samples)
- **Export Bundle**: Package project + checkpoints + visualizations for sharing

### 🔄 Versioning & Collaboration
- **Git Integration** (optional): Auto-commit checkpoints to branch
- **Shareable Links**: Generate read-only view of a checkpoint for collaboration
- **Annotation Layer**: Add comments to specific timesteps or memory entries

---

## 🧪 Experiment Setup Wizard

Guided flow for new experiments, with smart defaults and progressive disclosure:

```
Step 1: Choose Belief Engine
[🧠 SmolLM2-135M] [🧠 SmolLM2-360M ★] [🧠 SmolLM2-1.7B] [🔌 Custom]

Step 2: Planner Configuration
[🎭 Latent Transformer (4L)] [🎭 Flat Policy] [🎭 Hierarchical] [🔌 Custom]

Step 3: Reward Environment (Plug-and-Play)
[🎁 Curiosity Baseline] [🎯 Goal-Conditioned] [🔀 Multi-Objective] [🔌 Load Plugin]

Step 4: Steering Strategy
[⚡ Single-layer ActAdd] [🌐 Multi-layer] [🎚️ Adaptive Alpha]

Step 5: Visualization Preferences
[🎮 Gamified UI] [🔬 Expert Mode] [🎨 Theme: Soft/Cyber/Minimal]

[🚀 Launch Experiment] [💾 Save as Template]
```

*Smart Defaults*: Based on selected engine size, recommend appropriate planner hidden_dim, injection layers, and learning rate.

---

## 🌐 Plug-and-Play Architecture

### Plugin Interfaces (Abstract)
```python
# Reward Plugin
class RewardPlugin:
    def compute(self, outcome, memory, goal=None) -> dict[str, float]: ...
    def gui_controls(self) -> list[ControlSpec]: ...  # Declare GUI bindings

# Engine Wrapper
class BeliefEngine:
    def generate(self, prompt, steering=None) -> GenerationOutput: ...
    def extract_hidden(self, layer: int) -> Tensor: ...

# Planner Module
class Planner:
    def plan(self, memory: MemoryBuffer) -> PlannerOutput: ...
    def get_policy_heads(self) -> dict[str, Tensor]: ...
```

### Plugin Marketplace (Conceptual)
- **Official Plugins**: Novelty, Coherence, Goal-Relevance, Style-Diversity
- **Community Plugins**: Human-Feedback Proxy, Tool-Use Reward, Multi-Agent Debate
- **Installation**: Drag `.almaplugin` file onto Reward Mixer → auto-register

*Security*: Plugins run in sandboxed subprocess; require manifest declaring permissions.

---

## 🎨 Visualization & Animation System

### Core Principles
1. **Progressive Detail**: Overview → Zoom → Inspect → Raw Tensor
2. **Temporal Coherence**: Animations show *change*, not just state
3. **Semantic Color**: Consistent mapping (novelty=blue↔purple, coherence=green↔red, reward=gold)
4. **Performance First**: LOD (level-of-detail) rendering; pause heavy viz during rapid-run

### Key Visualizations
| Component | Primary Viz | Animation Trigger |
|-----------|------------|------------------|
| Latent Space | UMAP/PCA trajectory | New goal point added → smooth path extension |
| Reward Signal | Multi-line waveform | Step completion → waveform extends + pulse on spike |
| Text Generation | Color-coded token stream | Token generated → fade-in + novelty/coherence border flash |
| Steering Injection | Layer diagram + vector arrows | Alpha change → vector magnitude scales + pulse through layers |
| Memory Diversity | Radial scatter + trend line | Buffer update → points reposition + diversity metric animates |

### Interactive Exploration
- **Brush & Link**: Select region in latent trajectory → highlight corresponding memory entries + text samples
- **Time Scrubber**: Drag slider to replay autonomy loop at any point in history
- **Tensor Inspector**: Click any visualization element → drill down to raw values with export option

---

## ♿ Ergonomics & Accessibility

### Input Flexibility
- **Mouse**: Click, drag, scroll, right-click context menus
- **Keyboard**: Full navigation via tab/arrows; shortcuts for common actions (`Space`=play/pause, `S`=step, `Ctrl+S`=save)
- **Touch**: Swipe to navigate timelines; pinch-to-zoom on visualizations
- **Voice** (future): "Show me high-novelty generations" → filter view

### Cognitive Load Management
- **Focus Mode**: Hide all panels except active visualization + essential controls
- **Guided Tours**: Contextual tooltips that appear on first-use of complex features
- **Undo Everywhere**: No action is irreversible without confirmation
- **Performance Budget**: UI remains responsive even during heavy computation (offload viz to worker thread)

### Accessibility Features
- **High Contrast Mode**: For low-vision users
- **Screen Reader Support**: Semantic ARIA labels for all interactive elements
- **Reduced Motion**: Option to disable non-essential animations
- **Keyboard-Only Flow**: Complete workflow achievable without mouse

---

## 🧰 Toolkit-Agnostic Implementation Strategy

### Abstract UI Primitive Layer
Define interface components without framework commitment:
```typescript
interface Panel {
  id: string;
  title: string;
  children: UIElement[];
  resizable?: boolean;
  collapsible?: boolean;
}

interface ControlSpec {
  type: 'slider' | 'toggle' | 'dropdown' | 'vector-editor' | 'plugin-slot';
  bind: string;  // YAML path, e.g., "reward.novelty_weight"
  label: string;
  config: Record<string, any>;  // min/max/steps/options
}

interface Visualization {
  type: 'trajectory' | 'heatmap' | 'waveform' | 'text-stream';
  dataStream: Observable<DataUpdate>;
  interactions: InteractionSpec[];
}
```

### Rendering Adapters (Examples)
| Target Framework | Adapter Strategy |
|-----------------|-----------------|
| **Web (React/Vue)** | Web Components + Canvas/WebGL for viz; state via RxJS |
| **Desktop (Tauri/Electron)** | Same web frontend in native shell; file system access via IPC |
| **Native (Qt/GTK)** | QML/ GTK4 UI bindings to same abstract spec; OpenGL for viz |
| **Jupyter** | Widget-based adapter; visualizations as interactive Plotly/D3 |

### State Management Pattern
```
YAML Config ←→ Abstract State Model ←→ UI Bindings
       ↑                ↑
   File I/O      Business Logic (ALMA core)
       ↓                ↓
  Checkpoints    Training Loop (async)
```

- **Single Source of Truth**: Abstract state model; UI is pure projection
- **Reactive Updates**: Changes propagate via observable streams
- **Offline-First**: All state serializable; sync to cloud optional

---

## 🚀 Getting Started Flow (User Journey)

1. **Launch ALMA Studio** → Welcome screen with "New Experiment" / "Load Project"
2. **Wizard** → Select SmolLM2-360M + Curiosity Baseline + Gamified UI
3. **Auto-Generated Workspace** → Pre-configured panels with sensible defaults
4. **Press ▶️** → Autonomy loop begins; latent trajectory starts drawing
5. **Observe** → Curiosity meter fills; first discovery badge appears at step ~15
6. **Tweak** → Adjust novelty weight slider; see immediate effect on trajectory spread
7. **Checkpoint** → `Ctrl+S`, tag "first-pivot", add note "balanced exploration achieved"
8. **Explore** → Click memory entry to see full intent→outcome; brush to find similar
9. **Export** → Share checkpoint link with collaborator; they load and continue

---

## 📋 Implementation Roadmap (Phased)

### Phase 1: Core MVP (Weeks 1-4)
- [ ] Abstract UI primitive system
- [ ] Basic panel layout + transport controls
- [ ] Live binding for 5 key parameters (alpha, novelty_weight, etc.)
- [ ] Simple latent trajectory + text stream visualizations
- [ ] Checkpoint save/load with tags

### Phase 2: Rich Visualization (Weeks 5-8)
- [ ] Animated steering injection feedback
- [ ] Reward mixer with faders + plugin slots
- [ ] Memory buffer timeline with filtering
- [ ] Gamification overlay (curiosity meter, badges)
- [ ] Step-through debugging mode

### Phase 3: Polish & Extensibility (Weeks 9-12)
- [ ] Plugin system for reward/environments
- [ ] Advanced visualizations (heatmaps, vector previews)
- [ ] Accessibility features + theme system
- [ ] Project browser with diff/compare
- [ ] Export/share functionality

### Phase 4: Collaboration & Scale (Future)
- [ ] Real-time multi-user sessions
- [ ] Cloud checkpoint sync
- [ ] Mobile-responsive views
- [ ] Voice/command interface prototype

---

## 🎯 Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Time to First Insight** | < 3 minutes | User testing: from launch to meaningful observation |
| **Parameter Discoverability** | 90% of controls found without help | Click-heatmaps + survey |
| **Cognitive Load** | < 2 min to understand core loop | Think-aloud protocol |
| **Expert Efficiency** | 5x faster config iteration vs. YAML editing | Task completion time |
| **Engagement** | 70% of users enable gamification | Telemetry (opt-in) |

---

> **Final Note**: This GUI isn't just a control panel—it's a *cognitive partner*. By making the invisible dynamics of autonomous exploration visible, tangible, and playful, we lower the barrier to understanding how beliefs and goals can co-evolve without corruption. The DAW metaphor works because, like music production, ALMA experimentation is about *orchestrating processes*, not just setting parameters. The gamification isn't decoration—it's scaffolding that helps users develop intuition for emergent cognitive phenomena.

*Toolkit-agnostic by design, human-centered by default.* 🎛️🧠✨
