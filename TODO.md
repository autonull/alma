# Curiosity Loop v3.2 — Strategic Roadmap

## Diagnosis of Current State

Before planning, it's worth naming what the logs reveal:

- **Alpha is frozen at 0.50.** The learned steering magnitude isn't moving, suggesting gradients through the bridge are weak or conflicting. The planner is doing all the work.  
- **Norm is constant at 0.50.** Because alpha is static and the steering vector is L2-normalized, every injection has identical magnitude. The system has no way to modulate *how hard* it steers.  
- **Novelty oscillates between near-zero and overcorrection.** Steps 150–200 show collapse (Nov 0.059–0.078), then step 400 spikes to 1.095 after W1 ramps up. This is a classic sign of reactive tuning with no lookahead — the agent oscillates between boredom and incoherence.  
- **Text is math-heavy.** SmolLM2-360M's training data skews mathematical; the initial prompt "Thinking process initiated." doesn't steer away from this attractor basin.  
- **Mode is almost always H.** The mode signal isn't providing meaningful behavioral differentiation yet.

---

## 1\. Prompting Patterns

The prompt is the initial condition of the dynamical system. Right now `"Thinking process initiated."` seeds math-textbook territory. Strategic prompting can open up the full semantic landscape.

### **Prompt Pool with Domain Diversity**

Instead of a single seed, maintain a rotating pool of 10–20 prompts spanning different registers:

PROMPT\_POOL \= \[

    "The fundamental question is whether",

    "In a small village by the sea,",

    "Consider the following experimental setup:",

    "Dear colleague, I wanted to share",

    "The data suggests an unexpected pattern:",

    "Once, there was a machine that could",

    "The three main arguments against this are",

    "Looking at the night sky, one might wonder",

    "Step 1: Gather the ingredients.",

    "According to recent findings in neuroscience,",

\]

**When to re-seed:** Every N steps (e.g., 200), or when novelty drops below a threshold for K consecutive steps, sample a new prompt from the pool. This breaks attractor basins without requiring the planner to do it alone.

### **Prompt as Observation**

A more powerful pattern: let the planner *choose* the prompt. Add a prompt-selection head to the planner that outputs a distribution over the pool. Now the agent controls both the steering vector and the initial conditions — a much richer action space.

### **Context Conditioning**

Currently `prompt_ids[:, -64:]` is a sliding window. Consider structured context:

\# Prepend a "task frame" that the planner selects

task\_frames \= \["\[EXPLORE\]", "\[EXPLAIN\]", "\[NARRATE\]", "\[ANALYZE\]"\]

frame\_idx \= planner.task\_head(pooled)  \# new head

context \= tokenizer.encode(task\_frames\[frame\_idx\] \+ " " \+ recent\_text)

This gives the LM a soft instruction signal alongside the activation-space steering.

---

## 2\. Reward Function Design

The current reward `R = w1 * novelty * 5.0 - w2 * loss + w3 * depth_bonus` optimizes for "say something new and coherent." This is necessary but not sufficient for useful behavior.

### **Reward Decomposition Framework**

Think of reward as a sum of orthogonal signals:

| Signal | What it measures | Current? | Priority |
| :---- | :---- | :---- | :---- |
| Novelty | Semantic distance from memory mean | ✅ | Keep |
| Coherence | Cross-entropy loss | ✅ | Keep |
| Topical relevance | Similarity to a target embedding | ❌ | High |
| Diversity over time | Entropy of topic distribution | ❌ | High |
| Self-consistency | Entailment between consecutive outputs | ❌ | Medium |
| Information density | Unique token ratio, no repetition | ❌ | Medium |

### **Goal-Conditioned Rewards (the big unlock)**

The system becomes *useful* when you can say "explore toward X." Implement this by:

1. Encode a target concept (e.g., "renewable energy") into a target embedding `z_target` using the LM's own encoder.  
2. Add a **relevance reward**: `cosine_similarity(outcome_downsampled, z_target)`.  
3. The agent now has to find novel, coherent text *in the neighborhood of a concept*.

\# Goal-conditioned reward

target\_text \= "renewable energy and sustainability"

with torch.no\_grad():

    target\_ids \= tokenizer.encode(target\_text, return\_tensors="pt").to(DEVICE)

    target\_hidden \= world\_engine(target\_ids, output\_hidden\_states=True)

    z\_target \= bridge.down\_projection(target\_hidden.hidden\_states\[-1\]\[:, \-1, :\].float())

\# In the loop:

relevance \= F.cosine\_similarity(outcome\_downsampled, z\_target, dim=-1).item()

reward \= (w1 \* novelty \* 5.0) \- (w2 \* coherence\_loss) \+ (w4 \* relevance \* 3.0)

This transforms the system from "random curious exploration" to "directed semantic search."

### **Anti-Repetition via N-gram Penalty**

The reactive W1 tuning is a blunt instrument. Add a direct repetition penalty:

from collections import Counter

def ngram\_repetition\_penalty(text, n=3):

    tokens \= text.split()

    if len(tokens) \< n:

        return 0.0

    ngrams \= \[tuple(tokens\[i:i+n\]) for i in range(len(tokens)-n+1)\]

    counts \= Counter(ngrams)

    repeated \= sum(1 for c in counts.values() if c \> 1\)

    return repeated / max(len(ngrams), 1\)

rep\_penalty \= ngram\_repetition\_penalty(generated\_text)

reward \-= 2.0 \* rep\_penalty

### **Curriculum: Phased Reward Schedules**

- **Phase 1 (steps 0–1000):** Pure novelty \+ coherence. Learn to steer without crashing.  
- **Phase 2 (steps 1000–3000):** Add goal-conditioning. Learn to steer *toward things*.  
- **Phase 3 (steps 3000+):** Add self-consistency and information density. Learn to steer toward *useful* things.

---

## 3\. Architectural Enhancements

### **Fix the Alpha Problem**

Alpha isn't learning because REINFORCE gradients don't flow through `torch.no_grad()` generation. The bridge parameters only get gradient through `log_prob`, which depends on the planner's output, not alpha. Fix:

**Option A — Treat alpha as part of the action:**

\# Make alpha an output of the planner

self.alpha\_head \= nn.Linear(mem\_dim, 1\)

\# ...

alpha \= torch.sigmoid(self.alpha\_head(pooled)) \* 3.0  \# range \[0, 3\]

Now alpha is part of the policy and gets REINFORCE gradients naturally.

**Option B — Auxiliary loss on alpha:**

\# Directly optimize alpha with a heuristic target

alpha\_target \= 0.5 \+ 0.5 \* (novelty \- 0.2)  \# want more steering when novelty is low

alpha\_loss \= (bridge.alpha \- alpha\_target) \*\* 2

total\_loss \+= 0.1 \* alpha\_loss

### **Multi-Layer Steering**

Currently injecting at layer 24 only. Different layers encode different information:

- **Early layers (0–8):** Token-level syntax, local patterns  
- **Mid layers (9–18):** Semantic composition, topic  
- **Late layers (19+):** High-level planning, next-token prediction

Add per-layer steering with learned magnitudes:

class MultiLayerBridge(nn.Module):

    def \_\_init\_\_(self, latent\_dim=256, phi\_dim=960, n\_layers=3):

        super().\_\_init\_\_()

        self.target\_layers \= \[8, 16, 24\]

        self.projections \= nn.ModuleList(\[

            nn.Linear(latent\_dim, phi\_dim, bias=False) for \_ in range(n\_layers)

        \])

        self.alphas \= nn.ParameterList(\[

            nn.Parameter(torch.tensor(0.3)) for \_ in range(n\_layers)

        \])

### **Memory Architecture Upgrade**

The current memory is a fixed-size deque of concatenated vectors. Upgrade to attention-based memory:

class AttentiveMemory(nn.Module):

    def \_\_init\_\_(self, size=32, dim=256):

        super().\_\_init\_\_()

        self.keys \= nn.Parameter(torch.randn(size, dim) \* 0.01)

        self.values \= nn.Parameter(torch.randn(size, dim) \* 0.01)

        self.query\_proj \= nn.Linear(dim, dim)

    

    def read(self, query):

        q \= self.query\_proj(query)

        attn \= F.softmax(q @ self.keys.T / (dim \*\* 0.5), dim=-1)

        return attn @ self.values

    

    def write(self, key, value):

        \# Least-recently-used replacement

        ...

This lets the planner selectively attend to relevant past experiences rather than getting a flat average.

### **PPO Instead of REINFORCE**

REINFORCE has high variance. Switch to PPO for more stable learning:

- Add a value head to the planner: `self.value_head = nn.Linear(mem_dim, 1)`  
- Use GAE (Generalized Advantage Estimation) instead of raw returns  
- Clip the policy ratio for stability

This is the single highest-impact change for training stability.

### **Bigger Action Space: Steering \+ Temperature \+ Length**

Let the planner control more than just the steering vector:

\# Planner outputs:

\# 1\. latent\_goal (steering direction)

\# 2\. alpha (steering magnitude) 

\# 3\. temperature (generation randomness)

\# 4\. max\_tokens (generation length)

\# 5\. mode (explore vs exploit)

The agent can now learn to turn down temperature when it finds something interesting (exploit) and crank it up when stuck (explore).

---

## 4\. Modularization & Experiment Infrastructure

### **Core Abstractions**

curiosity\_loop/

├── agents/

│   ├── base\_agent.py          \# Abstract: plan() \-\> action, update(reward)

│   ├── reinforce\_agent.py     \# Current REINFORCE planner

│   ├── ppo\_agent.py           \# PPO variant

│   └── random\_agent.py        \# Baseline: random steering vectors

├── bridges/

│   ├── base\_bridge.py         \# Abstract: latent \-\> steering\_vector

│   ├── linear\_bridge.py       \# Current single-layer bridge

│   └── multilayer\_bridge.py   \# Multi-layer injection

├── rewards/

│   ├── base\_reward.py         \# Abstract: compute(state) \-\> float

│   ├── novelty.py             \# Cosine novelty

│   ├── coherence.py           \# Cross-entropy loss

│   ├── goal\_conditioned.py    \# Target embedding similarity

│   └── composite.py           \# Weighted sum of reward components

├── memory/

│   ├── deque\_memory.py        \# Current fixed buffer

│   └── attentive\_memory.py    \# Attention-based

├── engines/

│   ├── base\_engine.py         \# Abstract: generate(input, steering) \-\> output

│   └── hf\_engine.py           \# HuggingFace model wrapper

├── hooks/

│   └── actadd\_hook.py         \# Activation addition injection

├── config.py                  \# Dataclass configs for all components

├── loop.py                    \# Main training loop (engine-agnostic)

├── logger.py                  \# Metrics logging (see section 5\)

└── experiments/

    ├── baseline.yaml          \# Current v3.2 config

    ├── goal\_conditioned.yaml  \# Directed exploration

    └── ablation\_no\_memory.yaml

### **Config-Driven Experiments**

@dataclass

class ExperimentConfig:

    \# Agent

    hidden\_dim: int \= 256

    mem\_dim: int \= 512

    planner\_layers: int \= 4

    algo: str \= "reinforce"  \# or "ppo"

    

    \# Bridge

    phi\_dim: int \= 960

    injection\_layers: list \= field(default\_factory=lambda: \[24\])

    initial\_alpha: float \= 0.5

    

    \# Reward

    novelty\_weight: float \= 1.0

    coherence\_weight: float \= 0.5

    goal\_weight: float \= 0.0

    goal\_text: str \= ""

    

    \# Training

    steps: int \= 5000

    batch\_size: int \= 8

    lr: float \= 3e-4

    noise\_std: float \= 0.3

    gamma: float \= 0.90

    

    \# Engine

    model\_id: str \= "HuggingFaceTB/SmolLM2-360M"

    max\_new\_tokens: int \= 8

    temperature: float \= 0.7

### **Simplified Variations for Quick Tests**

**Minimal version (no planner, no memory):**

\# Random steering baseline — useful for measuring planner's contribution

for step in range(STEPS):

    random\_latent \= torch.randn(1, 256, device=DEVICE) \* 0.3

    steering \= bridge.get\_steering\_vector(random\_latent)

    \# ... generate and measure novelty/coherence

**Static steering (no RL, fixed direction):**

\# Test what a single steering vector does over time

target \= encode("Tell me about biology")

fixed\_steering \= bridge.get\_steering\_vector(target)

\# ... generate 100 steps with same vector, observe drift

These baselines are essential for proving the planner actually learns something.

---

## 5\. Visualization & Monitoring

### **Real-Time Dashboard (recommended: Weights & Biases or TensorBoard)**

**Scalar metrics to log every step:**

- `reward/total`, `reward/novelty`, `reward/coherence`, `reward/goal`  
- `agent/alpha`, `agent/steering_norm`, `agent/mode`  
- `tuner/w1`, `tuner/w2`  
- `text/unique_token_ratio` (tokens / unique tokens in last 50 steps)

**Periodic (every 100 steps):**

- `text/sample` — the actual generated text  
- `latent/goal_pca` — PCA projection of latent goals (track trajectory through latent space)  
- `memory/diversity` — average pairwise cosine distance in memory buffer  
- `steering/direction_change` — cosine similarity between consecutive steering vectors

  ### **Latent Space Trajectory Plot**

This is the most revealing visualization. Every 10 steps, record the latent goal. Then project all goals into 2D via PCA or UMAP. Color by reward. You'll see:

- **Mode collapse** as a tight cluster  
- **Healthy exploration** as a spreading cloud  
- **Goal-conditioned search** as a trajectory toward a target point

  ### **Text Stream Viewer**

Log generated text with color-coded metadata:

\[Step 0050\] \[Nov: 0.194 | Coh: 1.54 | Mode: H\] "task at hand. Consider using"

\[Step 0100\] \[Nov: 0.249 | Coh: 0.93 | Mode: H\] "an edge length of 25 cm"

              ^^^ GREEN (good novelty)      ^^^ GREEN (low loss)

### **Steering Vector Similarity Matrix**

Every 100 steps, compute pairwise cosine similarity between the last 100 steering vectors. Display as a heatmap. Diagonal stripes \= repetitive steering. Uniform noise \= no structure. Block structure \= the agent has found distinct "modes" of steering.

---

## 6\. Applications & Research Directions

### **Near-Term Applications**

**Semantic Search via Directed Exploration:** Given a query, set it as the goal embedding and let the agent explore the LM's latent space toward it. The trajectory of generated text becomes a "chain of thought" toward the concept. Unlike standard prompting, this searches through activation space, not token space.

**Concept Interpolation:** Set two goal embeddings (e.g., "biology" and "music") and reward proximity to both. The agent finds regions of the LM's latent space where these concepts overlap — potentially surfacing novel connections the LM "knows" but wouldn't generate unprompted.

**Adversarial Probing:** Reward the agent for finding steering vectors that *maximize* perplexity while maintaining some coherence floor. This identifies fragile regions of the LM — useful for robustness testing and understanding failure modes.

**Style Transfer via Steering:** Collect steering vectors that produce formal vs. casual text (by filtering on a style classifier reward). The learned bridge becomes a style-transfer module: swap steering vectors to change register without retraining the LM.

### **Medium-Term Research Directions**

**Hierarchical Planning:** The `sub_goal_count` head exists but isn't used. Implement hierarchical RL: the planner sets a high-level goal, then a sub-planner decomposes it into a sequence of steering vectors executed over multiple generation steps. This enables paragraph-level coherence, not just 8-token fragments.

**Multi-Agent Debate:** Run two curiosity loops with the same world engine but different reward functions (e.g., one rewards scientific language, another rewards creative language). Feed each agent's output into the other's context. The resulting "debate" could surface more nuanced exploration than either alone.

**World Model Learning:** The bridge's `down_projection` already maps LM hidden states back to latent space. Train it as an explicit world model: given `(latent_goal, current_state)`, predict `next_state`. Then the planner can do *model-based* planning — imagining trajectories through latent space before committing to steering vectors. This is the path toward something like MuZero for language.

**Scaling Law Experiments:** Run identical experiments across SmolLM2-135M, 360M, and 1.7B. Key questions: Does a bigger world engine make the planner's job easier or harder? Does the optimal injection layer scale linearly with depth? Does the latent space become more structured (easier to navigate) with scale?

**Transfer Across Engines:** Train a planner+bridge on SmolLM2-360M, then freeze the planner and retrain only the bridge on a different LM (e.g., Pythia-410M). If the planner transfers, it means the learned "exploration strategy" is model-agnostic — a genuinely portable curiosity module.

### **Longer-Term Vision**

**Self-Improving Loop:** Use the curiosity loop to generate training data for fine-tuning the world engine itself. The agent identifies novel, coherent regions of latent space → generates text → filters for quality → fine-tunes the LM. The next iteration of the curiosity loop then explores a richer landscape. This is a self-improving cycle where exploration and capability grow together.

**Intrinsic Motivation for Tool Use:** Replace text generation with tool-call generation. The world engine outputs API calls; the reward measures whether the tool response is novel and useful. The agent learns to explore an API landscape the same way it currently explores semantic space.

---

## Priority Implementation Order

1. **Fix alpha learning** (Option A — planner-controlled alpha). 30 minutes. Immediate impact on steering expressiveness.  
2. **Add n-gram repetition penalty** to reward. 15 minutes. Directly addresses mode collapse.  
3. **Implement prompt pool with rotation.** 20 minutes. Breaks math-textbook attractor.  
4. **Add goal-conditioned reward.** 1 hour. Transforms the system from toy to tool.  
5. **Set up logging** (wandb or tensorboard). 1 hour. Can't improve what you can't see.  
6. **Modularize into config-driven experiment runner.** 3–4 hours. Enables everything else.  
7. **Implement PPO.** 2–3 hours. Major stability improvement.  
8. **Multi-layer steering.** 1–2 hours. Richer control.  
9. **Latent trajectory visualization.** 1 hour. The "aha" plot for understanding what the agent learns.  
10. **Hierarchical sub-goals.** 4–6 hours. The path to paragraph-level generation.

----

Yeah—there *are* benchmarks, but they're not perfect fits because ALMA's so... weirdly pure. No tasks, no labels, no human eval—just self-play. Still, here's what actually works (and why most won't):

**Best real ones to steal from (plug-and-play):**  
1. **BabyAI** (or MiniGrid) — tiny grid worlds, natural-language commands.  
   Why it fits: you can "steer" toward novelty (explore unseen rooms) + coherence (follow grammar). No reward needed—just let the planner chase "new tiles" and "logical sentences." Run 10k episodes; measure if it learns to navigate without ever seeing a reward. If it does? Boom—proof of unsupervised exploration.  

2. **Procgen** (like CoinRun or Maze) — procedural, no memorization.  
   Same trick: planner steers activations to "find coins" or "solve mazes" via novelty + consistency (does this path loop?). Track: does average path length drop over time? That's your "learning curve" without cheating.  

3. **TextWorld** — text adventure games (Zork-lite).  
   Planner generates commands; frozen LLM parses world state. Score: novelty (new rooms/items) + coherence (valid syntax). If it starts winning games on its own? That's gold—shows it builds mental models.  

**Quick hack to make 'em ALMA-friendly:**  
- Freeze the LLM on world knowledge (no fine-tune).  
- Planner outputs activation vectors or prompt prefixes.  
- Reward = novelty (embedding diff) + coherence (perplexity) + consistency (does action match last state?).  
- No ground-truth—just self-scores.  

**If you want zero-code benchmarks:**  
- **GSM8K** (math word problems) — but twist it: planner tries to "explore" solutions, scores novelty + correctness (via frozen LLM's own judgment).  
- **BigBench** subsets — pick "logical deduction" or "analogies." Let it generate chains, score internal consistency.  

**Why not everything?**  
Most benches (GLUE, MMLU, ARC) want *accuracy*—ALMA doesn't care about being right, just being... interesting and sane. So you get "creative but wrong" unless coherence wins out. That's why start with games: they reward structure without needing truth.  

**My pick for you:** BabyAI + PPO.  
- Colab-ready.  
- Visual (record GIFs of agent wandering).  
- Measurable: "steps to first goal" vs. random baseline.  
- If planner learns faster than random? Tweet it—"No rewards, no humans, still solves mazes."  

Want me to grab the BabyAI env link and a starter script?
