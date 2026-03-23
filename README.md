# Autonomous Language Model Architecture (ALMA)

**Separating Causal Beliefs from Self-Evolving Goals in Language Models**

A modular research framework for building language model agents that possess beliefs about the world and independent goals about what to do with those beliefs — evolving both without mutual corruption.

---

## The Problem

Large language models are trained end-to-end. Their statistical knowledge of the world ("beliefs") is inseparable from the objectives imposed during alignment ("goals"). This entanglement produces familiar pathologies: hallucination drift, sycophancy, goal contamination, and an inability to revise intentions without distorting evidence. A model that has been optimized to be helpful cannot easily distinguish between *what it knows* and *what it wants* — and neither can we.

This project asks: **what happens when you enforce the separation architecturally?**

## Core Thesis

Genuine autonomous cognition requires two distinct systems:

1. A **frozen causal belief engine** — an unmodified pretrained language model whose weights are never updated. It serves as a neutral oracle: given a context, it returns next-token probabilities and hidden-state representations that reflect only what it learned during pretraining. No agenda. No drift. No self-modification.  
     
2. A **trainable goal/planner module** — a lightweight neural network that owns all volition. It reads the belief engine's internal states, maintains memory of past intentions versus outcomes, generates steering signals, and learns entirely from self-generated intrinsic reward. It alone receives gradients.

The belief engine answers *"what is likely?"* The planner decides *"what should I explore next?"* Neither can corrupt the other.

## Architecture

┌─────────────────────────────────────────────────────────────┐

│                      PLANNER MODULE                         │

│  ┌──────────────┐  ┌──────────┐  ┌───────────────────────┐ │

│  │   Latent      │  │  Memory  │  │   Policy Heads        │ │

│  │   Planner     │──│  Buffer  │──│  • latent goal        │ │

│  │  (Transformer)│  │ (intent/ │  │  • mode (flat/hier.)  │ │

│  │              │  │  outcome) │  │  • sub-goal count     │ │

│  └──────┬───────┘  └──────────┘  │  • alpha (magnitude)  │ │

│         │                        │  • temperature         │ │

│         ▼                        └───────────┬───────────┘ │

│  ┌──────────────┐                            │             │

│  │   Bridge      │◄───────────────────────────┘             │

│  │  (projection) │                                          │

│  └──────┬───────┘                                          │

└─────────┼───────────────────────────────────────────────────┘

          │ steering vector(s)

          ▼ activation addition

┌─────────────────────────────────────────────────────────────┐

│                  BELIEF ENGINE (frozen)                      │

│                                                             │

│  ┌─────┐  ┌─────┐  ┌─────┐       ┌─────┐  ┌─────┐        │

│  │ L0  │──│ L1  │──│ L2  │─ ... ─│ L\_k │──│ L\_N │──► tokens│

│  └─────┘  └─────┘  └─────┘  ▲    └─────┘  └─────┘        │

│                              │                              │

│                     steering injection                      │

│                                                             │

└─────────────────────────────────────────────────────────────┘

          │

          ▼ generated tokens \+ hidden states

┌─────────────────────────────────────────────────────────────┐

│                    INTRINSIC REWARD                          │

│                                                             │

│  novelty (semantic distance from memory manifold)           │

│  \- coherence (cross-entropy against unsteered baseline)     │

│  \+ relevance (optional: cosine similarity to goal embed.)   │

│  \+ depth bonus (sustained multi-step topic maintenance)     │

│  \- repetition penalty (n-gram overlap)                      │

│                                                             │

└─────────────────────────────────────────────────────────────┘

### **The Autonomy Loop**

Each step of the loop:

1. **Plan.** The planner reads its memory buffer and outputs a latent goal vector, a mode decision, and steering parameters.  
2. **Steer.** The bridge projects the latent goal into the belief engine's activation space and injects it at one or more transformer layers via activation addition (ActAdd).  
3. **Generate.** The belief engine produces tokens under the influence of steering. Its weights never change.  
4. **Observe.** The resulting hidden states are projected back into the planner's latent space as an outcome embedding.  
5. **Reward.** Intrinsic reward is computed by comparing the outcome against memory (novelty), against unsteered generation (coherence), and optionally against a target embedding (relevance).  
6. **Remember.** The intention-outcome pair is stored in the memory buffer.  
7. **Learn.** Policy gradients update the planner and bridge. The belief engine is untouched.

This loop is self-sustaining and requires no external supervision, human labels, or reward models.

## Modules

### **Belief Engine (`engines/`)**

Any pretrained causal language model. The framework wraps it with a consistent interface for generation, hidden-state extraction, and loss computation. The engine's weights are frozen at initialization and verified to remain statistically identical after arbitrary numbers of autonomous cycles.

Tested engines:

- `HuggingFaceTB/SmolLM2-135M` — fastest iteration, useful for architecture search  
- `HuggingFaceTB/SmolLM2-360M` — current default, good balance of capability and speed  
- `HuggingFaceTB/SmolLM2-1.7B` — scaling experiments

The engine is a parameter of the experiment, not a component of the agent. Swap it freely.

### **Latent Planner (`agents/`)**

A 4-layer transformer encoder that reads the memory buffer (a sequence of intention-outcome slot vectors) and a learned mode embedding. It outputs:

- **Latent goal** — a continuous vector in the planner's embedding space representing the intended semantic direction of the next generation.  
- **Mode logits** — a categorical choice between flat exploration (broad scanning) and hierarchical exploitation (focused investigation). The planner learns when to switch.  
- **Sub-goal count** — (planned) for hierarchical decomposition of high-level goals into multi-step steering sequences.

The planner defines the agent's policy. Everything it does is logged and attributable.

### **Bridge (`bridges/`)**

A pair of learned linear projections that translate between the planner's latent space and the belief engine's activation space:

- `up_projection`: latent goal → steering vector (planner space → engine space)  
- `down_projection`: engine hidden state → outcome embedding (engine space → planner space)

The bridge also controls steering magnitude via a learned `alpha` parameter. In the recommended configuration, alpha is an output of the planner (not a standalone parameter) so it receives proper policy gradients.

### **Steering via Activation Addition (`hooks/`)**

The steering vector is injected into the belief engine's residual stream at a chosen layer via a forward hook. This is:

- **Lightweight** — a single vector addition per forward pass.  
- **Reversible** — remove the hook and the engine behaves identically to its pretrained state.  
- **Layer-configurable** — different layers encode different levels of abstraction (syntax at early layers, semantics at mid layers, planning at late layers). Multi-layer injection with per-layer magnitudes is supported.

  ### **Memory (`memory/`)**

A fixed-size buffer of concatenated `[intention, outcome]` vectors maintained as a sliding window (deque). The planner reads this buffer as a sequence, enabling it to detect patterns in what it has tried and what resulted.

The memory serves three functions:

1. **Novelty computation** — outcomes are compared against the memory manifold to measure semantic surprise.  
2. **Policy context** — the planner conditions its next decision on its history of intentions and results.  
3. **Goal revision evidence** — the gap between intention and outcome signals whether the current strategy is working.

   ### **Intrinsic Reward (`rewards/`)**

The reward signal is computed entirely from internal dynamics — no external judges, no human preferences, no second model. This makes the system self-bootstrapping.

**Core reward components:**

| Component | Signal | Purpose |
| :---- | :---- | :---- |
| **Novelty** | Cosine distance from memory mean | Explore new semantic regions |
| **Coherence** | Negative cross-entropy loss | Stay within natural language manifold |
| **Repetition penalty** | N-gram overlap score | Prevent degenerate loops |
| **Depth bonus** | Sustained topic maintenance count | Reward focused investigation |
| **Relevance** | Cosine similarity to target embedding | Direct exploration toward concepts |

Weights between components can be fixed, scheduled (curriculum), or dynamically adjusted by reactive tuners that detect collapse or incoherence.

## Getting Started

### **Requirements**

- Python 3.10+  
- PyTorch 2.0+  
- Transformers (HuggingFace)  
- CUDA-capable GPU (recommended) or CPU (slow but functional)

  ### **Installation**

git clone https://github.com/\<org\>/autonomous-cognitive-architecture.git

cd autonomous-cognitive-architecture

pip install \-r requirements.txt

### **Quick Start: Run the Autonomy Loop**

\# Default configuration: SmolLM2-360M, REINFORCE, novelty+coherence reward

python run.py \--config configs/baseline.yaml

\# Goal-conditioned exploration

python run.py \--config configs/goal\_conditioned.yaml \\

  \--goal-text "renewable energy and sustainability"

\# Minimal test (random steering baseline, no learning)

python run.py \--config configs/random\_baseline.yaml

### **Configuration**

All experiments are defined by YAML config files. Key parameters:

\# configs/baseline.yaml

engine:

  model\_id: "HuggingFaceTB/SmolLM2-360M"

  max\_new\_tokens: 8

  temperature: 0.7

planner:

  hidden\_dim: 256

  mem\_dim: 512

  num\_layers: 4

  noise\_std: 0.3

bridge:

  injection\_layers: \[24\]

  initial\_alpha: 0.5

  alpha\_source: "planner"        \# "planner" (recommended) or "parameter"

reward:

  novelty\_weight: 1.0

  novelty\_scale: 5.0

  coherence\_weight: 0.5

  repetition\_weight: 2.0

  depth\_weight: 0.2

  goal\_weight: 0.0               \# set \> 0 for directed exploration

  goal\_text: ""

training:

  algorithm: "reinforce"          \# "reinforce" or "ppo"

  steps: 5000

  batch\_size: 8

  learning\_rate: 3e-4

  gamma: 0.90

  grad\_clip: 1.0

memory:

  buffer\_size: 16

prompting:

  seed: "Thinking process initiated."

  pool\_enabled: false

  pool\_rotation\_interval: 200

logging:

  log\_interval: 50

  backend: "console"              \# "console", "wandb", or "tensorboard"

## Experiment Guide

### **Ablation Studies**

The modular design enables systematic ablation of each component:

| Experiment | What it tests | Config change |
| :---- | :---- | :---- |
| Random steering | Does the planner learn anything? | `planner: null` (random latent each step) |
| No memory | Does history matter? | `memory.buffer_size: 1` |
| No novelty | Can coherence alone drive learning? | `reward.novelty_weight: 0.0` |
| No coherence | Does the coherence anchor prevent noise? | `reward.coherence_weight: 0.0` |
| Single layer vs. multi-layer | Where should steering inject? | `bridge.injection_layers: [24]` vs. `[8, 16, 24]` |
| Flat mode only | Does hierarchical switching help? | `planner.mode_enabled: false` |

### **Scaling Experiments**

Hold the planner fixed and vary the belief engine:

for model in SmolLM2-135M SmolLM2-360M SmolLM2-1.7B; do

  python run.py \--config configs/scaling.yaml \--engine.model\_id "HuggingFaceTB/$model"

done

Key questions: Does a larger belief engine make the planner's exploration richer? Does optimal injection layer scale with model depth? Does the latent space become easier to navigate?

### **Transfer Experiments**

Train planner+bridge on one engine, freeze the planner, retrain only the bridge on a different engine:

\# Phase 1: Train on SmolLM2-360M

python run.py \--config configs/transfer\_phase1.yaml \--save-planner planner\_360m.pt

\# Phase 2: Transfer to Pythia-410M

python run.py \--config configs/transfer\_phase2.yaml \\

  \--load-planner planner\_360m.pt \--freeze-planner

If the planner transfers, the learned exploration strategy is model-agnostic.

## Reward Engineering

The reward function is the most consequential design choice. It determines what the agent finds interesting, what it avoids, and what "progress" means.

### **Phase 1: Undirected Curiosity (Baseline)**

The agent explores freely, rewarded for finding coherent novelty. This is the foundation — validation that the architecture can sustain self-directed exploration without external supervision.

**What to watch for:** Novelty staying in the 0.10–0.50 range. Coherence loss staying below \~2.5. Generated text drifting across topics without collapsing into repetition. The planner learning to modulate its steering strength.

### **Phase 2: Directed Exploration (Goal-Conditioned)**

Add a target embedding computed from a concept phrase. The agent must now balance novelty, coherence, *and* relevance to the target. This transforms free exploration into semantic search.

**Applications:** Concept mapping (explore everything the belief engine knows about "photosynthesis"), knowledge boundary detection (steer toward a topic and observe where coherence collapses — that's the edge of what the model knows), creative ideation (set a target like "novel applications of fermentation" and collect the trajectories).

### **Phase 3: Multi-Objective Exploration**

Multiple simultaneous targets with potentially competing reward signals. The agent must discover regions of the latent space where multiple concepts intersect — surfacing non-obvious connections.

**Applications:** Interdisciplinary synthesis (set targets for "materials science" and "biological structures" simultaneously), analogy discovery, curriculum design.

### **Phase 4: Social Reward (Future)**

Replace or augment intrinsic reward with feedback from a second agent or from a lightweight evaluator trained on human quality judgments. This bridges the gap between autonomous exploration and useful output — but must be introduced carefully to avoid re-entangling beliefs and goals.

## Monitoring & Visualization

### **Scalar Metrics (Every Step)**

- `reward/total`, `reward/novelty`, `reward/coherence`, `reward/relevance`  
- `agent/alpha`, `agent/steering_norm`, `agent/mode`  
- `tuner/w1`, `tuner/w2`  
- `text/unique_token_ratio` (across recent window)

  ### **Diagnostic Plots (Periodic)**

- **Latent trajectory** — PCA/UMAP projection of latent goals over time, colored by reward. Mode collapse appears as a tight cluster; healthy exploration as a spreading trajectory; goal-conditioned search as directed drift toward a target.  
- **Steering similarity matrix** — Pairwise cosine similarity of recent steering vectors. Diagonal bands indicate repetition; block structure indicates discovered modes.  
- **Memory diversity** — Average pairwise cosine distance in the memory buffer over time. Should increase during healthy exploration.  
- **Text stream** — Generated text with color-coded novelty and coherence values.

## Research Roadmap

### **Completed**

- [x] Core loop: planner → bridge → ActAdd steering → generation → intrinsic reward → REINFORCE  
- [x] Memory buffer with intention-outcome tracking  
- [x] Mode switching (flat/hierarchical) with learned toggle  
- [x] Reactive dynamic weight adjustment for novelty and coherence  
- [x] Validation on SmolLM2-360M with stable multi-thousand-step runs

      ### **In Progress**

- [ ] **Fix alpha learning** — move alpha from standalone parameter to planner output head  
- [ ] **PPO upgrade** — add value head, GAE, clipped policy ratio for stable training  
- [ ] **Prompt pool** — break math-textbook attractor basin with domain-diverse seed rotation  
- [ ] **Repetition penalty** — direct n-gram overlap penalty in reward  
- [ ] **Logging infrastructure** — Weights & Biases / TensorBoard integration  
- [ ] **Modularization** — config-driven experiment runner with swappable components

      ### **Planned**

- [ ] **Goal-conditioned reward** — target embeddings for directed semantic exploration  
- [ ] **Multi-layer steering** — per-layer projection and magnitude, learned end-to-end  
- [ ] **Attention-based memory** — selective read/write replacing flat deque  
- [ ] **Hierarchical sub-goals** — planner decomposes high-level goals into multi-step steering sequences  
- [ ] **Scaling law experiments** — identical experiments across 135M, 360M, 1.7B belief engines  
- [ ] **Cross-engine transfer** — freeze planner, retrain bridge on a different model family  
- [ ] **Belief purity verification** — automated statistical tests confirming engine weights unchanged

      ### **Exploratory**

- [ ] Multi-agent debate (competing reward functions, shared belief engine)  
- [ ] World model learning (predict next latent state from current state \+ action)  
- [ ] Self-improving loop (use curiosity-generated data to fine-tune a separate copy of the engine)  
- [ ] Tool-use generalization (steer toward API calls instead of text)

## Applications

### **Knowledge Cartography**

Steer the agent toward a domain and record its latent trajectory. The path through semantic space maps what the belief engine "knows" about a topic — including boundary regions where coherence degrades, indicating the limits of the model's knowledge. This produces machine-generated knowledge maps without requiring labeled evaluation data.

### **Adversarial Robustness Probing**

Reward the agent for finding steering vectors that maximize perplexity while maintaining a coherence floor. The resulting vectors identify fragile regions of the language model — topics or phrasings where small perturbations cause disproportionate degradation. This is a form of automated red-teaming driven by curiosity rather than by adversarial prompting.

### **Concept Interpolation and Analogy Discovery**

Set two or more distant target embeddings and let the agent find latent paths between them. The intermediate regions may contain non-obvious conceptual bridges — analogies, metaphors, or structural similarities that the model has learned but would never surface through standard prompting. A "renewable energy" ↔ "immune system" interpolation might reveal shared regulatory dynamics.

### **Style and Register Control**

Collect steering vectors that produce text classified (by a lightweight style detector) as formal, casual, technical, narrative, etc. The resulting vector library enables fine-grained style transfer without retraining — swap the steering vector and the register changes, while the underlying knowledge remains identical.

### **Curriculum Generation**

Use the agent's exploration trajectories as raw material for educational content sequencing. A trajectory that coherently drifts from basic concepts to advanced implications, as judged by the novelty-coherence balance, naturally produces a learning progression. The depth bonus rewards sustained investigation, favoring trajectories that go deep before pivoting.

### **Mechanistic Interpretability**

The bridge's projections provide a learned mapping between the planner's intentional space and the engine's activation space. Analyzing which latent directions produce which semantic effects yields interpretable "knobs" for the language model's behavior — a form of representation engineering discovered through autonomous exploration rather than manual annotation.

## Design Principles

1. **Belief purity.** The frozen engine's weights and outputs must remain statistically identical regardless of how many autonomous cycles have run. This is the foundational invariant. Verify it.  
     
2. **Attribution clarity.** Every change in the system's behavior must be traceable to the planner's learned parameters. If the agent starts generating different text, the cause is in the planner or bridge — never in the engine. This makes the system auditable.  
     
3. **Intrinsic motivation.** The reward signal is generated entirely from internal dynamics. External reward sources (human feedback, task completion) may be added as optional components but must never be the sole signal — the agent should be capable of self-directed exploration without them.  
     
4. **Modularity.** Every component (engine, planner, bridge, memory, reward, hook) has a defined interface and can be swapped independently. This enables systematic ablation and prevents architectural lock-in.  
     
5. **Minimal intervention.** Steering should be the lightest-touch modification that achieves the desired effect. Activation addition is preferred over weight modification, attention reweighting, or prompt manipulation because it is additive, reversible, and layer-specific.  
     
6. **Scalable simplicity.** The planner is deliberately small relative to the engine. The cognitive overhead of autonomy should be a fraction of the cost of the underlying world model. If the planner needs to be as large as the engine, the architecture has failed.

## Ethical Considerations

This research builds autonomous agents that generate their own goals. This capability carries inherent risks that must be addressed proactively.

**Containment.** The current system operates within a narrow loop: the only actions available to the planner are steering vector injection and mode selection. The agent cannot modify its own reward function, access external systems, or persist state beyond the memory buffer. These constraints are architectural, not behavioral — they cannot be circumvented by learning.

**Transparency.** The separation of beliefs and goals is itself a safety property. Because the engine is frozen and the planner is lightweight, all learned behavior is concentrated in a small, inspectable set of parameters. The latent goals, steering vectors, and reward signals are logged at every step. There are no hidden optimization pressures.

**Scope limitation.** The system is designed for research into the mechanisms of autonomous exploration, not for deployment as an unsupervised agent. The intrinsic reward function does not encode human values, task completion, or helpfulness — it encodes curiosity. Responsible applications must add appropriate objective alignment as an additional (not replacement) reward layer.

**Dual use.** Techniques for steering language model activations can be applied beneficially (style control, knowledge mapping) or harmfully (generating targeted disinformation, bypassing safety filters). We note that activation addition is already a published technique (Turner et al., 2023\) and that this project's contribution is the autonomous planner, not the steering mechanism itself. We encourage the community to develop corresponding detection and defense methods alongside steering capabilities.

## Citation

@misc{alma2025,

  title={Autonomous Cognitive Architecture: Separating Causal Beliefs 

         from Self-Evolving Goals in Language Models},

  year={2025},

  note={Research in progress}

}

## License

\[TBD — recommend Apache 2.0 or MIT for maximum research accessibility, with an explicit responsible-use clause.\]  
