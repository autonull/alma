# ALMA v4 — Design Specification
## Brainstorm → Challenge → Reconstruct

---

## Part I: Assumption Audit

Before designing anything, interrogate what v1 through v3 took for granted.

---

### Assumption 1: Exploration and crystallization are separate phases

v3 generates text for a full thread, then extracts claims at the end. This is analogous to taking a long hike and only drawing the map afterwards. You inevitably forget details. You can't let the emerging map redirect where you walk.

**Challenge:** In real intellectual work, writing and thinking are simultaneous. The act of articulating a claim changes what you think next. The map shapes the territory.

**Implication:** Crystallization should be continuous and bidirectional. Newly extracted claims should feed immediately back into what gets explored next — within the same thread, not just the next one.

---

### Assumption 2: The knowledge store and the memory buffer are separate structures

v3 has two persistence layers: an episodic memory buffer (hidden states, for the prefix policy) and a knowledge store (crystallized claims, for user output). They serve different functions but hold overlapping information.

**Challenge:** This is redundant and creates a synchronization problem. The system's working memory and its knowledge artifact should be the same thing. The agent should think *in* its knowledge representation, not think separately and then transcribe.

**Implication:** Collapse both into a single structure that is simultaneously the agent's working memory and the user's readable artifact.

---

### Assumption 3: Claims are nodes in a flat list

v3's knowledge store is a list of claims with a contradiction map bolted on. The relationships between claims are an afterthought.

**Challenge:** Knowledge is inherently relational. A claim's meaning depends almost entirely on what it supports, what supports it, what it contradicts, and what it leaves open. A flat list destroys this structure.

**Implication:** Claims are nodes in a typed graph. Edges are first-class objects. The structure of the graph is itself a knowledge artifact — connectivity patterns reveal what's settled, what's contested, and what's unknown.

---

### Assumption 4: The reward function is a weighted sum of independent terms

v1 through v3 all compute reward as `Σ wᵢ · rᵢ`. The weights are either hand-tuned or reactively adjusted. The terms are treated as commensurate — you can trade off coherence against novelty with a scalar multiplier.

**Challenge:** These terms are not commensurate. Incoherence is a hard failure, not something to partially offset with high novelty. Coherence is a constraint, not an objective. Similarly, "goal alignment" below a threshold shouldn't be compensable by curiosity above some other threshold.

**Implication:** Separate hard constraints from optimization objectives. Constraint satisfaction first; then maximize information gain subject to those constraints. This is also more interpretable: you can explain why any given generation was rewarded.

---

### Assumption 5: Goals are stable within a session and user-defined

v3 takes a goal at session start and holds it fixed. If the user doesn't provide one, the system generates one — but it's fixed from that point.

**Challenge:** The most interesting discoveries change what you're looking for. Every significant research session ends with different questions than it started with. A system that can't revise its goals in response to what it finds is fundamentally limited.

**Implication:** Goals are hypotheses about what's worth exploring. The Claim Graph should reveal when a goal has been adequately answered, when a goal has proven intractable, and — most valuably — when exploration has surfaced a *better* goal than the one it started with.

---

### Assumption 6: One generation step produces one kind of output

The system alternates between generating text and evaluating it. Text is homogeneous — free-form output evaluated by the reward function.

**Challenge:** Not all generation steps should be the same. Sometimes the right move is to generate a candidate claim. Sometimes it's to generate an objection to an existing claim. Sometimes it's to generate a question. Sometimes it's to generate a bridge between two disconnected claims. These are qualitatively different cognitive acts.

**Implication:** Every generation step should have an explicit *generative role* — what kind of thing is it trying to produce? The role conditions the prefix, changes the reward function, and determines where in the Claim Graph the output lands.

---

### Assumption 7: The agent is unaware of what it doesn't know

v1 through v3 have no mechanism for recognizing the limits of the agent's competence. The system generates claims with varying confidence but doesn't distinguish between "I'm uncertain about this because it's genuinely contested" and "I'm uncertain about this because it's beyond what I can reason about reliably."

**Challenge:** A system that confabulates and a system that acknowledges its limits are not just different in accuracy — they are different in kind. Trust requires knowing when not to trust.

**Implication:** Design explicit **epistemic boundary detection**: regions of the Claim Graph where successive attempts to explore produce contradictory or incoherent claims despite strong pressure. These are marked as outside the system's reliable competence. This is part of the artifact.

---

### Assumption 8: The base unit of exploration is a "step"

v1 and v2 generate 8–32 tokens at a time. Each micro step is a tiny fragment. The reward evaluates fragments.

**Challenge:** Fragment-level evaluation is blind to argument structure. A claim might require 5 sentences to express properly. Rewarding sentence fragments for novelty produces sentence fragments, not claims.

**Implication:** The base unit should be a complete semantic unit — an argument, a claim with minimal supporting context. This is longer (a paragraph, not a sentence fragment) but evaluated holistically.

---

## Part II: The Reconstructed Design

From the assumption audit, three deep changes emerge:

1. Collapse memory and knowledge into a single structure: the **Claim Graph**
2. Collapse exploration and crystallization into a single continuous loop: **Generative Roles**
3. Replace the additive reward with a graph-theoretic one: **Information Gain**

Everything else adapts to serve these three changes.

---

## Architecture Overview

```
╔═════════════════════════════════════════════════════════════════╗
║                        CLAIM GRAPH                              ║
║  Nodes: claims (type, confidence, embedding, provenance)         ║
║  Edges: typed relationships (supports, contradicts, implies,     ║
║         questions, grounds, exemplifies, bridges)               ║
║                                                                  ║
║  This is simultaneously:                                         ║
║    • The agent's working memory                                  ║
║    • The session's knowledge artifact                            ║
║    • The source of the next exploration target                   ║
╚══════════════╦══════════════════════════╦═══════════════════════╝
               ║ graph attention readout  ║ gap / uncertainty signal
               ▼                          ▼
╔══════════════╧══════════╗  ╔════════════╧═══════════════════════╗
║   PREFIX POLICY         ║  ║   GOAL ENGINE                      ║
║   (v2 core, now reads   ║  ║   Watches graph for structural gaps ║
║   from Claim Graph via  ║  ║   High uncertainty → next goal     ║
║   graph attention)      ║  ║   High betweenness → priority node ║
╚══════════════╦══════════╝  ╚════════════╦═══════════════════════╝
               ║ prefix embeddings         ║ current target
               ▼                          ▼
╔═════════════════════════════════════════════════════════════════╗
║                  ROLE-CONDITIONED GENERATION                    ║
║                                                                  ║
║  Role selected by Thread Manager from:                           ║
║    Propose  — generate a new claim in target region             ║
║    Challenge — generate objection to an existing claim          ║
║    Bridge   — connect two claims in different graph regions      ║
║    Question — surface an open question implied by a claim        ║
║    Ground   — verify a claim against external structure          ║
║    Refine   — sharpen a vague claim into a falsifiable one       ║
╚══════════════╦══════════════════════════════════════════════════╝
               ║ paragraph-scale output with role metadata
               ▼
╔═════════════════════════════════════════════════════════════════╗
║                  CONTINUOUS CRYSTALLIZER                        ║
║                                                                  ║
║  Lightweight extraction running on completed paragraphs          ║
║  Produces: claim text, type, confidence, relationship targets    ║
║  Does NOT wait for thread end — updates graph in real-time       ║
║                                                                  ║
║  Quality gate: coherence check + duplicate suppression           ║
╚══════════════╦══════════════════════════════════════════════════╝
               ║ new nodes and edges
               ▼
╔═════════════════════════════════════════════════════════════════╗
║                  INFORMATION GAIN REWARD                        ║
║                                                                  ║
║  ΔH = entropy reduction in Claim Graph from new claim           ║
║      = f(new edges created, uncertainty reduced,                 ║
║           disconnected regions bridged,                          ║
║           open questions resolved)                               ║
║                                                                  ║
║  Hard constraints (must pass before ΔH is computed):            ║
║    1. Coherence ≥ threshold                                      ║
║    2. Not duplicate of existing claim (cosine < threshold)       ║
║    3. Role consistency (output matches declared role)            ║
╚══════════════╦══════════════════════════════════════════════════╝
               ║ reward signal
               ▼
         Policy update (prefix policy weights, via direct gradient)
         Thread Manager update (role/strategy curriculum)
         Goal Engine update (next target)
```

The graph is the center. Everything else reads from or writes to it.

---

## Innovation 1: The Claim Graph as Unified Substrate

### Structure

```
Node {
  id:          UUID
  text:        Natural language claim
  type:        hypothesis | observation | question | contradiction
               | analogy | boundary | open
  confidence:  Beta distribution (α, β) — not a point estimate
  novelty:     Distance from graph centroid at time of insertion
  role_origin: Which generative role produced this
  provenance:  [thread_id, strategy, step, parent_claim_ids]
  session:     Cross-session identifier
  grounded:    Boolean (verified against external source)
}

Edge {
  source, target: Node IDs
  type:           supports | contradicts | implies | questions
                  | exemplifies | bridges | refines | challenges
  weight:         0–1 (learned; stronger after repeated co-activation)
  confidence:     inherited from the weaker of its two nodes
}
```

### Why a Beta distribution for confidence?

A scalar confidence is a point estimate that erodes information. A Beta(α, β) distribution encodes both the estimate and how much evidence supports it. Low α + β = few observations, high uncertainty. High α + β = many consistent observations, high confidence. This enables:
- **Thompson sampling** for exploration target selection: sample from each node's confidence distribution, visit highest-variance samples more often
- **Credibility revision**: when a Challenge role produces a strong objection, the target node's Beta distribution is updated (decrease α or increase β), not reset
- **Principled aggregation**: two claims that support the same conclusion update a third claim's distribution via Bayes rule, not ad-hoc averaging

### Graph as Working Memory

When the prefix policy runs, it attends to a local subgraph (K-hop neighborhood of the current target) rather than a flat buffer. Graph attention is applied:

```
attention score(query, node_i) = softmax(query · embed(node_i) / √D)
weighted_context = Σᵢ attention_score(i) × embed(node_i)
prefix = prefix_policy(weighted_context, role_embedding, goal_embedding)
```

The structure of the graph determines what gets attended to — a claim with many connections will appear in many local neighborhoods and influence more generations than an isolated claim. Connection density = implicit relevance weighting.

### Graph as Artifact

At any moment, the Claim Graph is readable. A user can query it by:
- Similarity to a query string
- Claim type
- Confidence range
- Session range (what was found today vs. in prior sessions)
- Graph region (neighborhood of a specific claim)
- Structural role (gateway nodes — high betweenness centrality — are the most important bridging ideas)

The artifact is not produced at the end. It exists continuously, growing and revising in real-time. This fundamentally changes the user relationship: from "receive a report" to "watch a map being drawn."

---

## Innovation 2: Generative Roles

### The Problem with Homogeneous Generation

If every step generates "whatever seems interesting," the system cannot distinguish between a creative leap and a loose association, between a foundational claim and a peripheral detail, between an open question and a settled one. All output is undifferentiated text evaluated by the same reward.

### The Solution: Explicit Roles Before Generation

At the start of each paragraph-scale generation unit, the Thread Manager declares a **role** — a structured description of what kind of cognitive act this generation is attempting. The role is not a genre constraint (don't produce poetry) but a *relational constraint* (this output should create a specific kind of relationship to something already in the graph).

**Roles and their graph effects:**

| Role | Description | Graph Effect |
|---|---|---|
| **Propose** | Generate a new claim in the target region | New node, possibly new edges to similar claims |
| **Challenge** | Generate the strongest objection to a specific claim | New `contradicts` or `challenges` edge; updates target's confidence |
| **Bridge** | Synthesize two weakly-connected claims into a connecting claim | New node between two regions; new `bridges` edges |
| **Question** | Surface the most important open question implied by a claim | New `open` node with `questions` edge |
| **Refine** | Convert a vague claim into something falsifiable and specific | Replaces or supplements existing node with higher specificity |
| **Ground** | Test a claim against external structure or logical consistency | Adds or removes `grounds` edge; updates confidence |
| **Instantiate** | Generate a concrete example of an abstract claim | New `exemplifies` edge; helps operationalize abstract nodes |

### Role Curriculum

Which role to select is itself a learning problem. The Thread Manager maintains a distribution over roles, updated by the information gain each role produces. Early in a session: Propose dominates (building the initial graph). As the graph fills: Challenge, Bridge, and Question become more valuable (structural work). For underexplored goals: Propose again.

This is the same Thompson sampling mechanism as v3's exploration curriculum, but now operating over *types of cognitive acts* rather than topical strategies. The system learns its own thinking process, not just its topic preferences.

### Self-Play Without Two Models

Challenge and Refine roles implement adversarial self-play without requiring a second model or a second instance. The same LM plays both Proposer and Challenger by conditioning on different roles. The "debate" is asynchronous — a Propose generation adds a node, a Challenge generation later attacks it — but the graph captures the full dialectic.

The output is not just claims but a structured argument — claims with their challenges, counter-challenges, and eventual confidence updates. This is intellectually richer than any single-pass generation.

---

## Innovation 3: Information Gain as Reward

### Why Additive Weighted Rewards Are Wrong

The fundamental problem with `R = w₁ · novelty + w₂ · coherence + w₃ · curiosity` is that it assumes these are commensurate. They are not. A highly novel but incoherent output is not partially good — it is simply bad. Coherence is not a term to be traded off; it is a prerequisite for anything else to matter.

More importantly, none of these terms tell you whether the output *advanced the system's understanding*. A highly novel, coherent output that says something the graph already implied is not valuable. An obvious-seeming claim that connects two previously disconnected regions of the graph is enormously valuable.

### The Better Framing: What Does This Claim Do to the Graph?

Define reward as the **entropy reduction** caused by adding the new claim to the Claim Graph:

```
ΔH = H(Graph) − H(Graph ∪ {new_claim})
```

Graph entropy here has an operational definition based on:

1. **Edges created**: each new typed relationship reduces the relational uncertainty between claim pairs
2. **Questions resolved**: a new claim that directly addresses an `open` node reduces structural entropy substantially
3. **Contradiction revealed**: a Challenge that contradicts a high-confidence claim is highly informative (even though it doesn't "resolve" — it increases productive uncertainty that demands further exploration)
4. **Regions connected**: a Bridge claim connecting two previously isolated graph components has multiplicative impact (all claim pairs across the bridge are now relationally linked)
5. **Confidence updated**: a claim with strong `supports` edges to a low-confidence node increases that node's confidence, reducing distributional entropy

**Hard constraints (must pass before ΔH is evaluated):**
1. Coherence ≥ τ_coh (LM perplexity below threshold — same as v2)
2. Non-redundancy: cosine distance from all existing claims ≥ τ_dup
3. Role consistency: output actually performs the declared role (checked by a lightweight classifier)

If any constraint fails, reward is zero. ΔH is computed only on constraint-passing outputs.

### Why This Is Better

- **No weight tuning**: the reward is derived from graph structure, not from hand-chosen multipliers
- **Self-calibrating**: as the graph grows, the reward function automatically shifts — early sessions reward Propose (lots of novel claims), late sessions reward Bridge and Challenge (structural work)
- **Interpretable**: you can explain exactly why a generation earned high reward — it connected regions A and B, or it resolved question Q, or it updated confidence on claim C
- **Adversarial-robust**: cannot be gamed by generating superficially novel but structurally useless text, because novelty alone doesn't create edges

---

## Innovation 4: Dynamic Goal Evolution

### The Problem with Fixed Goals

A fixed session goal is an assumption that you know in advance what's worth finding. This is rarely true for genuine inquiry. The most valuable insight often appears in the periphery of what you were looking for.

### The Goal Engine

Rather than a user-provided goal that's fixed for the session, v4 has a **Goal Engine** that maintains a priority queue of exploration targets derived continuously from the Claim Graph:

**High-priority targets:**
- **High-betweenness nodes**: claims that sit between many other claims — gateway ideas whose elaboration would unlock many connections
- **High-uncertainty nodes**: Beta distributions with large variance — claims where the system has conflicting evidence
- **Orphaned open nodes**: `open` type claims with no incoming `answers` or `refines` edges — unanswered questions
- **Low-confidence high-degree nodes**: claims that many other claims depend on but which themselves have weak support — structural vulnerabilities

**Goal evolution:** the Goal Engine does not replace the user-provided macro goal — it *refines* it. As the graph fills in, the macro goal is interpreted through the lens of what's been found. "Explore the relationship between X and Y" shifts from "what is the relationship?" (early) to "why does this specific mechanism explain the relationship?" (middle) to "what are the boundary conditions where the relationship breaks down?" (late). These are all answering the same macro goal, but the specific target evolves.

**Emergent goal discovery:** occasionally the Goal Engine finds that the graph's most important structural feature has nothing to do with the original macro goal. A cluster of high-betweenness, high-uncertainty nodes has emerged in an unexpected region. The system surfaces this as a **goal suggestion**: "The exploration has revealed an important unaddressed area — would you like to shift focus?" This is the system telling the user something it found that the user didn't ask for. This is where genuine discovery happens.

---

## The Epistemic Boundary

### What No Prior Version Does

v1 through v3 have no mechanism for recognizing the limits of the model's reliable competence. If the model consistently generates contradictory or incoherent output about a topic, this is invisible — it just earns low reward and gets abandoned.

### The Boundary Map

v4 tracks a third type of graph region alongside settled knowledge (high-confidence clusters) and open questions (explicit `open` nodes): **epistemic boundaries** — regions where the system has repeatedly attempted to generate coherent claims and failed.

Detection: if a target node has received N or more Propose and Challenge attempts and its confidence distribution remains at maximum entropy (α ≈ β ≈ 1 — uninformative prior, no evidence either way), it is marked as an **epistemic boundary node** with type `boundary`.

Boundary nodes are:
- Not penalized — they represent honest failure, which is valuable information
- Explicitly surfaced in the artifact: "The system could not form reliable claims about X despite multiple attempts"
- Used to constrain future exploration — don't attempt to bridge from a boundary node to a well-settled region, as the bridge will be unreliable
- Presented to users as high-priority targets for external grounding or human expertise

A system that maps its own ignorance is more trustworthy than a system that doesn't. Boundary detection is a safety property as much as a capability property.

---

## Provenance as a First-Class Property

Every node and edge carries full provenance: which session, which thread, which role, which step, which parent claims were in the prefix context at generation time. This creates a complete audit chain.

Why this matters:
- **Reproducibility**: given the same prefix context and role, you can approximately reproduce any claim
- **Debugging**: if a high-confidence claim later proves wrong, you can trace back through all claims that `supports` it to find where the error was introduced
- **Attribution**: in multi-session use, you can see which session discovered which insight
- **Curriculum analysis**: which roles and strategies produce the highest-gain claims? Provenance enables post-hoc analysis that improves future sessions.

---

## What v4 Preserves from v1–v3

- **v2 prefix policy + direct gradient descent**: the micro-level generation mechanism is unchanged. Prefix policy reads from the Claim Graph instead of a flat buffer, but the training loop is the same.
- **v2 world model**: retained as a secondary curiosity signal, now specifically rewarding exploration of Claim Graph regions the world model cannot predict well.
- **v3 optional LoRA**: unchanged. The base model can slowly adapt.
- **v3 exploration curriculum**: retained, now operating over generative roles rather than abstract strategies.
- **v3 adversarial threads**: replaced by the Challenge role, which is more tightly integrated.
- **v3 cross-session persistence**: now the Claim Graph itself persists — richer than a claim list.

---

## The v4 Loop, End to End

```
Session begins
│
├─ User provides macro goal (optional: auto-generated from graph gaps)
│
├─ Goal Engine selects first target from Claim Graph
│   (empty graph → high-entropy region near goal embedding)
│
└─ For each generation unit:
   │
   ├─ Thread Manager selects generative role (curriculum-weighted)
   │
   ├─ Prefix Policy attends to local graph neighborhood of target
   │   + goal embedding + role embedding
   │   → generates prefix
   │
   ├─ LM generates paragraph-scale output under prefix
   │
   ├─ Continuous Crystallizer extracts claim(s) from output
   │   Checks: coherence, non-redundancy, role consistency
   │
   ├─ Valid claims added to Claim Graph as nodes + typed edges
   │
   ├─ Information Gain computed: ΔH from new nodes/edges
   │
   ├─ Direct gradient: (-ΔH).backward() → prefix policy update
   │
   ├─ Goal Engine recalculates priorities from updated graph
   │
   ├─ Thread Manager curriculum updated from role's ΔH contribution
   │
   └─ If boundary detection threshold reached → mark node, suggest
      If goal appears answered → surface goal evolution suggestion
      If unexpected high-value cluster found → surface discovery alert
```

---

## Application Domains

The generative roles are not domain-specific. Every domain benefits from Propose, Challenge, Bridge, Question, Refine, Ground, and Instantiate. What changes between domains:

| Domain | Dominant Early Roles | Dominant Later Roles | Most Valuable Graph Feature |
|---|---|---|---|
| Scientific research | Propose (hypotheses), Question | Challenge, Ground | Contradiction clusters — contested areas signal active research fronts |
| Strategic planning | Propose (opportunities), Question | Challenge (risks), Bridge (cross-domain) | Boundary nodes — what no one can predict reliably |
| Philosophy | Propose, Challenge | Bridge (synthesis), Refine | High-betweenness nodes — the crux claims everything hinges on |
| Engineering design | Propose, Instantiate | Challenge (failure modes), Ground | Open nodes — unresolved design questions blocking progress |
| Education design | Propose, Question | Bridge (prerequisites), Instantiate | Graph connectivity — prerequisite gaps in the dependency structure |
| Creative development | Propose, Bridge | Instantiate, Question | Isolated nodes — unexplored territories with no connections yet |

---

## Feasibility Analysis

The v4 design is more ambitious than v3, but every component has a clear implementation path:

| Component | Complexity | Path |
|---|---|---|
| Claim Graph | Low | NetworkX or pure Python dict; embed claims with sentence-transformers |
| Graph attention in prefix policy | Medium | Sparse local attention over K-hop neighborhood, not full graph |
| Beta confidence tracking | Low | Two scalars (α, β) per node, updated via simple rule |
| Generative roles | Low | Role embedding concatenated with prefix input; same architecture |
| Continuous Crystallizer | Medium | Lightweight few-shot extraction prompt, same LM, every N tokens |
| Role consistency classifier | Low | Same LM, binary classification prompt: "does this output match role X?" |
| ΔH reward computation | Medium | Graph delta: count new edges, resolved questions, updated confidences |
| Goal Engine | Low | Priority queue over nodes, weighted by betweenness + uncertainty |
| Epistemic boundary detection | Low | Count failed high-entropy attempts per node, threshold flag |
| Provenance tracking | Low | Metadata dict attached to every node/edge at creation time |

The one genuinely hard problem is **scalable graph attention** as the Claim Graph grows to thousands of nodes across sessions. Mitigation: limit attention to a constant-size local neighborhood (K=10–20 hops), keeping computation constant regardless of global graph size.

---

## What v4 Does Not Solve (and Why That's the Right Choice)

**Factual accuracy against the external world.** v4 knows when it doesn't know, and it knows when its internal beliefs are consistent. It does not know when its internal beliefs are systematically wrong. The `Ground` role provides hooks for external grounding, but they are optional and lightweight. A full grounding system — persistent retrieval, citation verification, formal fact-checking — is v5 territory.

**Multi-user collaboration.** The annotation surface allows one user to interact. Multiple users with potentially conflicting annotations, permissions, and perspectives require a social layer that v4 doesn't design.

**Causal interventional reasoning.** The Claim Graph captures associative and logical relationships. It does not capture counterfactuals ("what would change about Y if X were false?"). This requires an interventional framework beyond what graph structure provides.

**Real-time speed.** Continuous crystallization adds latency. For use cases requiring fast generation, the crystallizer can be batched or run asynchronously. But the tight loop design prioritizes quality over throughput.

---

## What v4 Opens

v4 produces something v1 through v3 never could: an artifact with full epistemic structure — not just what was found, but how confident, how challenged, how connected, and where the limits of the system's knowledge lie.

This opens the door to **v5's core question**: can the Claim Graph from one session meaningfully improve the starting point of the next — not just by carrying forward known claims, but by carrying forward the *relational structure* that makes those claims meaningful? Can the system learn, across many sessions in the same domain, to think better about that domain — not just know more, but reason with higher fidelity?

That question requires a theory of how graph structure transfers across contexts. It is the right next question.

---

## Evolution Summary

| Dimension | v1 | v2 | v3 | v4 |
|---|---|---|---|---|
| Core unit | Token sequence | Token sequence | Thread | Claim |
| Output | Ephemeral text | Ephemeral text | Claim list | Claim Graph |
| Memory | Deque | Attention buffer | Buffer + store | Unified graph |
| Reward | REINFORCE | Direct gradient | ΔReward (weighted sum) | ΔH (information gain) |
| Goal | None | None | Fixed session goal | Dynamic, graph-derived |
| Self-evaluation | None | None | Adversarial threads | Challenge role + boundary detection |
| Knowledge structure | None | None | Flat with contradiction map | Typed graph with Beta confidence |
| User relationship | Log reader | Log reader | Artifact consumer | Map watcher + annotator |
| Core question | Can it explore? | Can it explore well? | Can it produce artifacts? | Can it build knowledge? |
