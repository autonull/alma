# ALMA v5 — Design Specification
## The Version Worth Building

---

## Why This Document Is Different

Every prior design document opened by describing what the previous version did wrong, then proposed fixes. This document opens by asking a harder question: **what would make someone actually want to use this?**

The evaluation of v4 identified real problems — belief revision, ΔH underspecification, narrative synthesis gap, evaluation blindness, coherentism without calibration. But underneath all of them is a single meta-problem that none of v1 through v4 ever solved: **we have never asked whether the output is useful to a real person solving a real problem.**

v5 answers that question first, then designs backward from the answer.

---

## The Honest Scope

Language models are reliably good at some things and unreliable at others. v5 is honest about which is which.

**LLMs are reliably good at:**
- Mapping the conceptual structure of a domain (relationships, tensions, typologies, frameworks)
- Generating questions that experts recognize as the right questions
- Synthesizing across ideas that are rarely considered together
- Identifying what a domain takes for granted (surface hidden assumptions)
- Producing structured analytical distinctions

**LLMs are unreliable at:**
- Specific factual claims (dates, attributions, numerical values, recent events)
- Claims at the cutting edge of a field (training data doesn't include what isn't yet written)
- Claims requiring empirical observation rather than inference

v5 is a **conceptual synthesis engine**, not a knowledge base and not a research assistant. It maps the *shape* of a domain — the relationships, the tensions, the open questions, the assumed foundations — using an LLM's embedded knowledge. It does not establish facts. It establishes structure.

This scope is not a limitation to apologize for. It is a specific, valuable capability that no other tool provides. A product manager mapping a design space, a philosopher charting an argument's logical geography, a strategist identifying assumptions embedded in a plan — these users need conceptual structure, not facts. That is what v5 delivers.

Every artifact v5 produces explicitly declares this scope. Users know what they're getting and what to verify independently.

---

## What Changes from v4, and Why

v4 introduced several ideas that were right in spirit but wrong in execution. v5 is honest about which ones.

### Simplified: Beta distributions → scalar confidence + evidence count

Beta distributions are statistically principled but operationally opaque. A user looking at a claim node should be able to understand its confidence at a glance. The implementation of Beta updating is also more complex than the value it adds: the meaningful quantities (point estimate and how much evidence supports it) are better represented as two explicit values — `confidence` (0–1) and `support_count` (integer) — than as a distribution that users and developers alike must mentally unpack.

Beta distributions also imply a Bayesian update model that isn't actually implemented in v4. The confidence was updated by heuristic rules, not Bayes' theorem. Representing it as a Beta was misleading about the underlying mechanism.

### Replaced: ΔH "information gain" → Structural Contribution Score

The evaluation identified that "entropy of a Claim Graph" is undefined without a generative model over graphs. v4 used operational heuristics (new edges, resolved questions, connected components) that didn't actually compute ΔH. Calling it information gain was conceptually appealing but technically dishonest.

v5 replaces ΔH with a **Structural Contribution Score (SCS)**: a weighted sum of concrete, measurable graph deltas. The weights are not pretended to be information-theoretic — they are explicit design choices, declared as such, justified by the structural importance of each delta type. An honest heuristic is better than a misnamed formalism.

### Simplified: Continuous crystallization → paragraph-level crystallization

Running the Crystallizer every N tokens adds real-time latency and requires the LM to pause mid-thought. Paragraph-level crystallization (after each complete generation unit) is nearly as responsive, avoids the context interruption, and produces more coherent claims because it reads a complete unit of meaning rather than a fragment. The continuous version was solving a problem (lag between generation and graph update) that doesn't actually matter at the timescale of a human reading the artifact.

### Simplified: K-hop graph attention → relevance-gated top-K attention

K-hop neighborhood attention requires a graph traversal at every step and grows expensive as the graph densifies. The real requirement is that the prefix policy reads relevant graph context — and relevance is better captured by embedding similarity than by graph distance. Claims that are conceptually close to the current goal/role are the relevant ones, not claims that happen to be graph-adjacent to them. Top-K by embedding similarity is cheaper, more interpretable, and more correct.

### Added: Narrator (the component that makes the artifact useful)

The Claim Graph is a machine-readable structure. A human needs a narrative. v5 adds a **Narrator** component that converts the graph into a structured plain-language summary. This is not a cosmetic feature — it is the primary interface between the system and the user. Without it, v4's artifact requires the user to have graph analysis skills to extract value. With it, the system can be used by anyone.

### Added: Confidence Cascading (the component that keeps the graph correct)

v4 updated claim confidence when new evidence arrived but did not propagate those updates through the graph. v5 adds **confidence cascading**: when a claim's confidence changes significantly, the change propagates through `supports` and `challenges` edges with an attenuation factor. Downstream claims that were supported by a now-undermined claim have their confidence reduced proportionally. This is approximate belief revision, not AGM-complete, but it prevents zombie claims from accumulating.

### Added: Evaluation Infrastructure (the component that ends iteration blindness)

v5 is built with instrumentation from the start. Every claim is logged with its full context. Domain experts can annotate claim quality. Internal metrics (SCS, confidence distributions, graph topology) are tracked alongside external validity (expert agreement, user ratings). Without this, v6 would be designed as blindly as v1 through v4 were.

### Added: Claim Domain Typing (the component that establishes trustworthiness)

Every crystallized claim is typed by its epistemic domain: `conceptual`, `analytical`, `logical`, or `factual`. Factual claims are prominently marked as unverified. Conceptual and analytical claims are the system's core competence and are not marked with the same warning. Users see different visual treatment for different claim types. This is the mechanism by which v5 is honest about what it is and isn't doing.

---

## Architecture

```
SESSION START
│
├─ User provides macro goal (natural language)
│
├─ GRAPH SEEDER: Generate 5–7 seed claims via one-shot LM pass
│   "Given this goal, what are the foundational claims a structured
│    exploration would want to examine or challenge?"
│   → Seeds cold-start; gives prefix policy immediate context
│
└─ MAIN LOOP ─────────────────────────────────────────────────────
   │
   ├─ GOAL ENGINE: Select next exploration target
   │   Priority queue over: high-uncertainty nodes, orphaned questions,
   │   high-betweenness nodes, low-confidence high-degree nodes
   │
   ├─ ROLE SELECTOR: Choose generative role (curriculum-weighted)
   │   Propose | Challenge | Bridge | Question | Refine | Instantiate
   │   Role curriculum updated by SCS per role across session history
   │
   ├─ PREFIX POLICY: Generate prefix embeddings
   │   Inputs: top-K relevant claims (embedding similarity to goal+role)
   │           + role embedding + goal embedding
   │   → K soft-prompt tokens prepended to generation
   │
   ├─ GENERATION: Paragraph-scale output (direct gradient, v2/v4 mechanism)
   │
   ├─ CRYSTALLIZER: Extract claims from completed paragraph
   │   → coherence check, non-redundancy check, role consistency check
   │   → domain typing: conceptual | analytical | logical | factual
   │   → 1–3 claims extracted, embedded, ready for graph insertion
   │
   ├─ GRAPH UPDATE: Insert claims and edges
   │   → new nodes added with confidence=0.5, support_count=1
   │   → typed edges inferred from role + content
   │   → CONFIDENCE CASCADE: if any edge connects to existing high-degree nodes,
   │     propagate confidence update through graph (attenuation=0.6 per hop, max 3 hops)
   │
   ├─ SCS REWARD: Compute Structural Contribution Score
   │   = w_edge  × Δ(typed edges created)
   │   + w_qres  × Δ(open question nodes addressed)
   │   + w_bridge × Δ(previously disconnected components linked)
   │   + w_contra × Δ(contradictions surfaced, weighted by target confidence)
   │   + w_refine × Δ(vague claims replaced by specific ones)
   │   Subject to: coherence ≥ τ, non-redundant, role-consistent
   │
   ├─ POLICY UPDATE: (-SCS).backward() → prefix policy weights
   │
   └─ EPISTEMIC BOUNDARY CHECK: If target node has received ≥ N attempts
       with SCS < ε, mark as boundary. Do not target again.

SESSION END (or on-demand)
│
└─ NARRATOR: Convert Claim Graph to structured narrative
    → Top 5 findings (high-confidence, high-betweenness)
    → Top 3 tensions (contradicting claim pairs with high confidence on both sides)
    → Top 3 open questions (orphaned question nodes with highest connectivity)
    → Epistemic boundary map: "The system could not form reliable claims about X"
    → Domain breakdown: N conceptual claims, M analytical, K factual (unverified)
    → Recommended next session: goal suggestions from graph gaps
```

---

## The Four Components That Make This Worth Building

### 1. The Narrator

The most important component in v5 is not in the exploration loop. It is the thing the user actually reads.

The Narrator takes the Claim Graph and produces a structured document. It is not a flat summary — it has architecture:

**Top Findings:** The five highest-confidence, highest-betweenness claims, explained in plain language, with their primary supporting and challenging claims cited. These are what the session established most firmly.

**Key Tensions:** The three most significant contradiction pairs — both high-confidence, both high-connectivity, genuinely in conflict. These are not errors; they are the most intellectually productive part of the artifact. Real domains have real tensions. Surfacing them is the system's clearest value.

**Open Questions:** The three orphaned question nodes with the highest connectivity — questions that many other claims depend on or point toward, but that the session did not resolve. These tell the user where to look next, either in the next session or through external research.

**Epistemic Boundaries:** A short plain-language map of what the system consistently failed to reason about reliably. These are not gaps in the knowledge graph — they are honest declarations of the system's limits in this domain.

**Domain Summary:** How many claims are conceptual, analytical, logical, factual. The factual claims are listed separately with an explicit verification notice.

The Narrator uses the same base LM, conditioned on the graph structure. It is called with the graph serialized as structured input, and asked to produce each section. This is few-shot prompting, not a separately trained model.

The Narrator output is the deliverable. The graph is the substrate.

### 2. Confidence Cascading

When a claim's confidence changes, that change propagates through the graph.

The mechanism is simple: message passing with attenuation. When claim A's confidence is revised (Challenge role produces a strong objection; a Refine produces a narrower, better-specified version; an Instantiate reveals a concrete case that contradicts the abstract claim), every claim with a `supports` edge from A receives a confidence update proportional to `edge_weight × 0.6`. That update propagates one more hop with the same attenuation. Maximum three hops.

This is not AGM belief revision. It makes no claim to be logically complete. It is a practical approximation that prevents zombie claims — high-confidence nodes whose support has been quietly eroded — from accumulating indefinitely. The graph stays approximately consistent rather than precisely consistent, and approximate consistency is far better than no consistency.

The attenuation factor (0.6) and maximum hop count (3) are explicit design choices, tunable. They are not pretended to be information-theoretically derived.

### 3. Structural Contribution Score (replacing ΔH)

The reward function in v5 is honest. It does not claim to be information gain. It is a weighted sum of structural graph deltas that the designers believe reflect meaningful progress.

```
SCS = 3.0 × (open questions addressed)
    + 2.0 × (disconnected components bridged)
    + 1.5 × (high-confidence contradictions surfaced)
    + 1.0 × (new typed edges created)
    + 0.5 × (vague claims refined to specific ones)

subject to:
    coherence ≥ τ_coh           (LM perplexity below threshold)
    non-redundancy ≥ τ_dup      (cosine distance from all existing claims)
    role_consistency = True     (output matches declared role)
```

The weights encode a priority ordering: resolving open questions is more valuable than adding new edges, which is more valuable than refining vague claims. This ordering is a design choice that can be adjusted based on evaluation feedback.

Calling it SCS rather than ΔH is not a downgrade. It is honesty about what the reward actually computes.

### 4. Evaluation Infrastructure

v5 is built with instrumentation from the first run. The goal is to end iteration blindness — to be able to answer, after building v5, which design decisions actually mattered.

**Logged automatically per session:**
- All generated paragraphs with role, target, SCS
- All crystallized claims with domain type, confidence trajectory, provenance
- All graph structural metrics at each step: number of nodes, edges, components, mean confidence, claim type distribution
- All Narrator outputs

**Enabled by the log:**

*Internal consistency test:* For any two sessions with the same macro goal on the same LM, do the high-confidence findings converge? Agreement across sessions is evidence of reliability. Divergence is evidence of instability.

*Expert annotation protocol:* Present domain experts with the Narrator output (not the raw graph). Ask three questions per finding: (1) Is this claim true in your domain? (2) Is this a claim you would not have easily produced without this tool? (3) Is the tension or question surfaced genuinely important? Average scores across experts give a ground truth signal for whether the artifact has value.

*Baseline comparisons:* Run the same macro goal through three alternative processes: (a) direct LLM prompting ("generate 20 important claims about X"), (b) zero-shot chain-of-thought ("think systematically about X and list key claims"), (c) a human expert spending 30 minutes brainstorming. Compare Narrator output quality (via expert annotation) across all four. This tells us whether the ALMA architecture adds value over simpler approaches.

*Ablation tests:* Run v5 with individual components disabled — no confidence cascading, no generative roles, no exploration curriculum, flat attention instead of top-K — and measure whether expert ratings decline. This tells us which components actually matter.

Without this infrastructure, v6 would again be designed by hypothesis. The evaluation infrastructure is not optional — it is what makes v5 more than another design exercise.

---

## The Trustworthiness Model

v5 is trustworthy in specific ways and untrustworthy in others. Both are explicit.

**What v5 is trustworthy about:**
- The conceptual structure it produces (relationships, tensions, typologies) reflects patterns genuinely present in the LLM's training
- High-confidence claims have survived internal Challenge scrutiny and confidence cascading
- Epistemic boundary nodes reflect genuine repeated failure, not random underperformance
- The Narrator's "key tensions" reflect real contradictions in the graph, not fabricated conflict
- Open questions reflect structural gaps in the graph, not arbitrary choices

**What v5 is explicitly not trustworthy about:**
- Factual claims — all are marked unverified and should be independently checked
- Claims in domains near the training data cutoff — the system cannot know what it doesn't know about recent developments
- Claims in epistemic boundary zones — the system has declared these unreliable
- Causal claims — the graph captures association and logical implication, not causal mechanism
- The claim graph is not a consensus view — it reflects one LLM's training, not a survey of expert opinion

The Narrator output includes a boilerplate trustworthiness declaration that is not buried in fine print but appears as the first section of every artifact.

---

## A Concrete Use Case

*Elena is a product manager at a startup entering the "AI for legal research" space. She needs to understand the competitive and conceptual landscape quickly before a strategy meeting in two days. She has three hours.*

She gives v5 the goal: "Map the conceptual design space for AI-assisted legal research tools."

v5 runs for two hours, generating 180 paragraph-scale units across Propose, Challenge, Bridge, and Question roles. The Crystallizer extracts 240 claims. 190 are typed `conceptual` or `analytical`. 50 are typed `factual` (flagged).

The Narrator produces:

**Top 5 findings:** The tension between explainability requirements and model capability; the structural difference between search-augmentation tools and reasoning-augmentation tools; the liability question as a design constraint rather than a regulatory afterthought; the underexplored niche of transactional (vs. litigation) legal work; the assumption that the primary user is a lawyer rather than a paralegal or client.

**Key tensions:** Human-in-the-loop requirements vs. automation value proposition. Jurisdiction-specificity vs. generalizability. Auditability vs. competitive moat (if your reasoning is transparent, it's reproducible).

**Open questions:** What does "accuracy" mean for legal AI — citation accuracy, reasoning validity, or outcome prediction? Who owns liability when AI-assisted work contains errors? How does the regulatory landscape differ by practice area?

**Epistemic boundary:** The system could not form reliable claims about specific pricing models or recent regulatory enforcement actions (marked as factual, high uncertainty).

Elena reads this in 20 minutes. She enters her strategy meeting with a structured conceptual map she could not have produced in three hours of unaided work. She independently verifies the three factual claims she intends to cite. The tensions the Narrator surfaced become the organizing framework for the meeting agenda.

This is the use case. The system is worth building if it reliably produces artifacts like this.

---

## Evaluation Design

v5 is a success if it passes three tests within six months of first build:

**Test 1 — Internal consistency.**
The same macro goal run twice (different random seeds, same LM) should produce Narrator outputs whose top findings overlap by ≥ 60%. If the output is highly variable across identical inputs, the system is not reliable enough to use.

**Test 2 — Expert validity.**
Domain experts (N ≥ 3 per domain, at least 3 domains) should rate the Narrator's top findings as "accurate or productively challenging" in ≥ 70% of cases, and as "non-obvious to someone who hadn't done this analysis" in ≥ 50% of cases. If neither threshold is met, the architecture is not producing value over simpler baselines.

**Test 3 — Baseline superiority.**
The ALMA Narrator output should score higher on expert rating than direct LLM prompting in ≥ 2 of 3 tested domains. If direct prompting matches or exceeds ALMA, the architectural complexity is not justified.

If v5 fails Test 3, the honest conclusion is that the architecture adds engineering complexity without commensurate quality gain, and v6 should pursue a fundamentally different direction.

---

## Scope Boundaries

v5 does not:
- Retrieve or verify facts from external sources (the Ground role provides hooks; they are optional and not core)
- Reason causally or counterfactually
- Execute external actions or affect the world
- Support multi-user collaboration
- Learn to reason better across sessions (knowledge accumulates; reasoning capability does not)
- Guarantee accuracy of any claim
- Operate faster than ~2 minutes per meaningful generation unit on consumer hardware

These are not failures. They are the boundaries of honest scope. A system that knows what it is and isn't is more trustworthy than one that attempts everything and quietly fails at some of it.

---

## Evolution Summary

| Dimension | v4 | v5 |
|---|---|---|
| Core claim | Knowledge production system | Conceptual synthesis engine |
| Confidence model | Beta(α,β) distribution | Scalar + support_count (honest) |
| Reward | ΔH "information gain" (underspecified) | Structural Contribution Score (explicit heuristic) |
| Crystallization | Continuous (every N tokens) | Paragraph-level (per generation unit) |
| Graph attention | K-hop neighborhood | Top-K by embedding similarity |
| Belief revision | None | Confidence cascading (3-hop, attenuation 0.6) |
| User artifact | Queryable graph | Narrator + graph (narrative primary) |
| Factual claims | Unmarked | Explicitly typed, verification notice |
| Epistemic limits | Boundary detection | Boundary detection + prominent Narrator section |
| Evaluation | None | Built-in: consistency, expert validity, baseline comparison |
| Success criterion | Internal metrics | Expert agreement ≥ 70%, baseline superiority |
| Primary user action | Navigate graph | Read Narrator output |
