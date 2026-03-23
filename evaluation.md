# ALMA v2 — Brutal Evaluation

**Date:** 2026-03-23  
**Purpose:** Determine whether this architecture is worth building, or needs fundamental redesign.

---

## Executive Verdict

**Build it — but smaller.** The core insight is valuable, but v2.md is overengineered. Strip 60% of the features, ship the 40% that actually matters, and validate the core hypothesis before adding complexity.

**Recommended scope:** Concept Explorer + Style Shifter only. Nothing else until these two work.

---

## What's Actually Novel (vs. Rearranging Existing Parts)

### ✅ Genuinely Novel

| Idea | Why it's novel | Commercial/research value |
|------|----------------|---------------------------|
| **Steering vector libraries** | Pre-trained steering vectors for style, register, domain — reusable across sessions | High. This is "LoRA for inference" without weight changes |
| **Planner as model router** | Learning to route tasks between models based on predicted success | Medium. Similar to MoE routing, but applied at inference time across model sizes |
| **Skill packs** | Exporting trained planners as reusable modules | High. This is the "product" — composable intelligence |

### ⚠️ Derivative (But Still Useful)

| Idea | Prior art | Differentiation |
|------|-----------|-----------------|
| Model cascade | FrugalGPT, Cascade inference | Planner-learned routing (not heuristic) |
| Multi-tier memory | RETRO, RAG systems | Planner-controlled write gating |
| Hybrid reward | RLHF + intrinsic motivation literature | Real-time mixing, not sequential |

### ❌ Not Novel (Consider Cutting)

| Idea | Why cut it |
|------|------------|
| **LoRA generation from latent goals** | LoRA is well-understood; generating adapters on-the-fly adds complexity without clear benefit. Just fine-tune when needed. |
| **Multi-agent debate** | Already done extensively (LLM debate papers, 2023-24). Low marginal value. |
| **Six-channel reward console** | Overengineered. Start with 2: intrinsic + task. Add human feedback only after validation. |
| **Graph memory with NetworkX** | Just use RAG. Graph construction is hard; maintaining causal structure is harder. |

---

## Core Hypothesis — Stated Clearly

> **A lightweight planner that learns to steer LLM activations can produce task-specific behavior improvements without fine-tuning the base model — and these improvements are exportable as reusable "skills."**

**This is the only hypothesis that matters.** Everything else is decoration.

**Success looks like:**
- Train a planner for 500 steps on "generate technical explanations"
- Export the planner + bridge weights
- Load in a new session, apply to unseen topics
- Generated text is measurably more technical (per classifier) and coherent (per perplexity) than baseline
- No fine-tuning of the LLM required

**If this works:** You have a product (skill packs, steering libraries).  
**If this fails:** Nothing else in v2.md saves it.

---

## Technical Risks — Ranked by Severity

### 🔴 Critical (Could Kill the Project)

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Steering doesn't generalize** | High | Fatal | Steering vectors may be prompt/context-specific. A "technical style" vector trained on one topic may not transfer. **Test early:** Train on biology, test on economics. |
| **Planner learns nothing** | Medium | Fatal | REINFORCE/PPO may not converge if reward signal is too noisy. The original prototype shows oscillation. **Test early:** Run 1000 steps, check if reward trend is positive. |
| **Commodity GPU can't run it** | Low | Fatal | 1.7B model + planner + memory may exceed 12GB. **Test early:** Profile memory usage with all components loaded. |

### 🟡 Serious (Could Make It Useless)

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Skill packs don't transfer** | High | High | Planner may overfit to training context. **Mitigation:** Regularize heavily, test on held-out topics during training. |
| **Steering degrades coherence** | Medium | High | Activation addition can break language quality. **Mitigation:** Multi-layer steering with per-layer alpha; coherence penalty in reward. |
| **Training too slow for iteration** | Medium | Medium | If each experiment takes hours, development stalls. **Mitigation:** Use 135M model for dev, 360M for final runs. |

### 🟢 Manageable (Annoying, Not Fatal)

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Memory bloat | High | Low | LRU eviction, compression, disk storage. |
| GUI complexity | High | Low | Build CLI first, GUI later. |
| Config sprawl | Medium | Low | Start with 2 configs, add only when needed. |

---

## Competitive Landscape — Honest Assessment

### Direct Competitors (Same Problem, Different Approach)

| Project | What they do | Why ALMA v2 is different |
|---------|--------------|--------------------------|
| **LangChain / LlamaIndex** | RAG, agent orchestration | ALMA learns steering policies; they use hardcoded prompts + retrieval |
| **LoRA fine-tuning** | Weight adaptation for tasks | ALMA steers activations without weight changes; faster, reversible |
| **Prompt engineering tools** | Systematic prompt optimization | ALMA learns in activation space, not token space |
| **Inference-time steering** (Turner et al., 2023) | Manual activation addition | ALMA automates steering vector discovery via learning |

### Indirect Competitors (Different Problem, Same Budget)

| Project | What they do | Threat level |
|---------|--------------|--------------|
| **Fine-tuning APIs** (OpenAI, Anthropic) | Upload data, get custom model | High. Easier for most users. |
| **Prompt libraries** | Pre-built prompts for tasks | Medium. Good enough for many use cases. |
| **RAG as a service** (Pinecone, Weaviate) | Managed retrieval + generation | Medium. Solves "customization" without learning. |

### ALMA v2's Actual Competitive Advantage

> **Steering is faster than fine-tuning, more powerful than prompting, and produces exportable artifacts.**

This is the pitch. If it's not true, the project has no reason to exist.

---

## Feature Prioritization — What to Cut

### Must Build (Core Hypothesis)

- [ ] Planner with latent goal + alpha output
- [ ] Bridge with activation addition (single-layer to start)
- [ ] Vector memory for novelty computation
- [ ] Intrinsic + task reward (2 channels only)
- [ ] PPO training loop
- [ ] Skill pack export/load
- [ ] CLI runner with YAML config

### Should Build (If Core Works)

- [ ] Multi-layer steering
- [ ] Model router (2 models: 135M + 360M)
- [ ] Human feedback buffer
- [ ] Basic visualization (terminal-based metrics)

### Could Build (Later, Maybe Never)

- [ ] LoRA generation
- [ ] Graph memory
- [ ] Episodic memory with replay
- [ ] Multi-agent debate
- [ ] API model integration
- [ ] Full DAW-style GUI
- [ ] Curriculum learning
- [ ] Social reward channel

### Should Not Build (Distractions)

- [ ] Six-channel reward console
- [ ] Automatic difficulty adjustment
- [ ] Safety guardrails (use existing tools)
- [ ] Checkpoint tagging system
- [ ] Git integration

---

## Minimum Viable Product — Redefined

**ALMA v2 MVP:**

> A CLI tool that trains a planner to steer SmolLM2-360M toward a target style or topic, exports the trained weights as a skill pack, and applies it to new inputs — all in under 10 minutes on an RTX 3060.

**User journey:**
```bash
# 1. Train a skill
alma train --config style_technical.yaml --output skills/technical.pt

# 2. Apply the skill
alma generate --skill skills/technical.pt --input "Explain photosynthesis"

# 3. See the difference
# Baseline: "Photosynthesis is when plants make food..."
# Steered:  "Photosynthesis is a biochemical process wherein chlorophyll-containing organisms..."
```

**That's it.** Nothing else matters until this works.

---

## Design Flaws in v2.md

### 1. **Overengineered Memory**

Three-tier memory (vector + graph + episodic) is premature. Graph memory alone is a multi-month research project.

**Fix:** Start with vector memory only. Add episodic if replay learning proves necessary. Drop graph entirely unless a specific application demands it.

### 2. **Reward Console Complexity**

Six reward channels with learnable weights is absurd for an MVP. The original prototype barely manages 2 channels without oscillation.

**Fix:** Intrinsic (novelty + coherence) + task (goal relevance). That's it. Human feedback can be added after validation.

### 3. **Model Router Premature**

Routing between 3 models (135M, 1.7B, API) adds complexity without validating the core hypothesis.

**Fix:** Use 360M only for MVP. Add 135M for speed comparison after core works. API integration is a nice-to-have, not essential.

### 4. **LoRA Integration Confusing**

Generating LoRA weights from latent goals is clever but untested. It may not work, and it distracts from the core steering hypothesis.

**Fix:** Offer fine-tuning as a separate mode (not integrated with steering). Compare steering vs. fine-tuning empirically.

### 5. **GUI Before Validation**

Building a DAW-style interface before proving the core works is putting the cart before the horse.

**Fix:** Terminal-based metrics + optional TensorBoard. Build GUI only after MVP validation.

---

## Revised Architecture — Stripped Down

```
┌─────────────────────────────────────────────────────────────┐
│                      ALMA v2 LITE                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐ │
│  │   Planner    │────▶│    Bridge    │────▶│  SmolLM2-360M│ │
│  │  (PPO)       │     │  (ActAdd)    │     │  (frozen)    │ │
│  │              │     │              │     │              │ │
│  │ • latent     │     │ • up/down    │     │              │ │
│  │ • alpha      │     │ • alpha      │     │              │ │
│  │ • mode       │     │              │     │              │ │
│  └──────┬───────┘     └──────┬───────┘     └──────────────┘ │
│         │                    │                               │
│         ▼                    ▼                               │
│  ┌──────────────────────────────────────────────────────────┐│
│  │                   Vector Memory (FAISS)                   ││
│  │                   (novelty computation)                   ││
│  └──────────────────────────────────────────────────────────┘│
│                                                              │
│  ┌──────────────────────────────────────────────────────────┐│
│  │              Reward = w1*novelty + w2*coherence           ││
│  │                       + w3*relevance                      ││
│  └──────────────────────────────────────────────────────────┘│
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**Lines of code estimate:** ~800 (down from ~3000 in full v2.md)  
**Time to MVP:** 2-3 weeks (down from 8-10 weeks)  
**GPU requirement:** RTX 3060 (12GB) — unchanged

---

## Validation Experiments — Run These First

### Experiment 1: Steering Transfer (Week 1)

**Question:** Does a steering vector trained on one topic transfer to another?

**Method:**
1. Train planner on "technical biology text" (100 samples)
2. Apply to economics prompts (unseen domain)
3. Measure: technicality (classifier), coherence (perplexity)

**Success:** Technicality score > 0.7, perplexity < 3.0  
**Failure:** Technicality < 0.5 or perplexity > 5.0

**If this fails:** Stop. Steering doesn't generalize. Pivot to fine-tuning.

---

### Experiment 2: Planner Learning (Week 1)

**Question:** Does the planner actually learn, or is reward noise?

**Method:**
1. Run 1000 steps with fixed task
2. Plot reward vs. step
3. Compare to random steering baseline

**Success:** Positive reward trend, beats random by > 20%  
**Failure:** Flat or negative trend, no difference from random

**If this fails:** Reward function is broken. Debug or redesign.

---

### Experiment 3: Skill Pack Utility (Week 2)

**Question:** Are exported skill packs actually useful?

**Method:**
1. Train skill, export, reload in new session
2. Apply to 10 held-out prompts
3. Human evaluation: blind A/B test (steered vs. baseline)

**Success:** Humans prefer steered output > 70% of time  
**Failure:** < 60% preference

**If this fails:** Skills don't transfer. Add regularization or reduce scope.

---

### Experiment 4: Commodity GPU Feasibility (Week 1)

**Question:** Does it actually run on RTX 3060?

**Method:**
1. Load all components
2. Run 100 steps, measure memory and time

**Success:** < 10GB VRAM, > 10 steps/second  
**Failure:** OOM or < 1 step/second

**If this fails:** Reduce model size or batch size. May need to drop to 135M.

---

## Go/No-Go Decision Tree

```
Week 1: Run Experiments 1, 2, 4
│
├─ All pass → Continue to Week 2
│
├─ Experiment 1 fails → Steering doesn't generalize
│   └─ Pivot: Fine-tuning only (drop steering)
│
├─ Experiment 2 fails → Planner doesn't learn
│   └─ Debug reward function; if still fails, abandon
│
└─ Experiment 4 fails → Can't run on commodity GPU
    └─ Reduce model size; if still fails, abandon

Week 2: Run Experiment 3
│
├─ Pass → Build MVP (CLI + configs + export)
│
└─ Fail → Skills don't transfer
    └─ Add regularization; if still fails, abandon
```

**Abandon criteria:** If 2+ experiments fail after fixes, the core hypothesis is probably wrong. Cut losses.

---

## If Core Hypothesis Fails — Pivot Options

### Pivot 1: Fine-Tuning Orchestrator

**Idea:** ALMA becomes a tool for managing LoRA fine-tuning workflows, not steering.

**What changes:**
- Drop activation addition entirely
- Planner learns to trigger fine-tuning when needed
- Skill packs = LoRA adapters, not steering vectors

**What stays:**
- Autonomy loop
- Memory-based novelty
- Hybrid reward

**Value prop:** "Automated fine-tuning decision-making" (less novel, but still useful)

---

### Pivot 2: Prompt Optimization Tool

**Idea:** ALMA learns optimal prompts instead of steering vectors.

**What changes:**
- Drop activation addition
- Planner outputs prompt prefixes, not latent goals
- Search prompt space instead of activation space

**What stays:**
- Intrinsic reward
- Memory for prompt history
- PPO training

**Value prop:** "Automatic prompt engineering with learning" (crowded space, but simpler)

---

### Pivot 3: Research Tool Only

**Idea:** Abandon applications; focus on mechanistic interpretability.

**What changes:**
- Drop skill packs, applications
- Add visualization/analysis tools
- Focus on understanding what steering vectors do

**Value prop:** "Activation space cartography for LLMs" (niche, but academically valuable)

---

## Final Recommendation

### Build It — With These Conditions

1. **Strip to MVP:** Vector memory only, 2 reward channels, no GUI, no LoRA, no model router
2. **Validate early:** Run 4 validation experiments in Week 1-2
3. **Kill criteria:** If 2+ experiments fail, pivot or abandon
4. **Scope discipline:** No new features until Concept Explorer and Style Shifter work

### Timeline

| Week | Milestone |
|------|-----------|
| 1 | Core loop working, Experiments 1/2/4 complete |
| 2 | Experiment 3 complete, MVP decision |
| 3-4 | Build MVP (CLI, configs, export) |
| 5 | Demo: Concept Explorer |
| 6 | Demo: Style Shifter |
| 7-8 | Polish, documentation, release |

### Resources Required

- **Developer time:** 1 person, 8 weeks (or 2 people, 4 weeks)
- **GPU:** RTX 3060 or equivalent (already available)
- **Budget:** $0 (all open-source, local execution)

---

## Bottom Line

**The core insight is real:** Steering via activation addition + learned planner is a genuine alternative to fine-tuning and prompting.

**The v2.md design is bloated:** 60% of features are premature optimization.

**The risk is manageable:** 4 validation experiments will tell you within 2 weeks whether this is worth building.

**Build it — but build the smallest version first.** If the core works, you can always add complexity. If it fails, you'll know fast and can pivot without wasting months.

---

**Decision:** ✅ Proceed with stripped-down MVP. Re-evaluate after Week 2 validation experiments.
