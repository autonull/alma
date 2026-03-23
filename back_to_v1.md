# What Happened to ALMA?

**From honest research question → fake product → useless mess**

---

## v1 Was Honest

**The research question:**
> Can intrinsic reward (novelty + coherence) sustain autonomous exploration in language models?

**The architecture (261 lines, clean):**
```
Planner (reads memory) → Bridge (latent → steering) → ActAdd → LLM → Reward → Learn
```

**The memory:**
- Simple buffer of (goal, outcome) vectors
- Used for ONE thing: computing novelty
- No pretense of being a "knowledge graph"

**The reward:**
- Novelty: cosine distance from memory mean
- Coherence: negative cross-entropy loss
- That's it. Honest. Measurable.

**What it proved:**
- The loop can run autonomously
- Reward can drive learning
- Steering can affect output

**What it didn't claim:**
- To complete tasks
- To build useful knowledge
- To be a product

---

## v5 Is Dishonest

**The fake value prop:**
> "Autonomous goal-driven language agent"

**The reality:**
- "Goals" come from a fixed pool of 12 topics
- "Knowledge graph" stores text snippets as fake triples
- "Task completion" is just generating text and moving on

**The bloated architecture (1100+ lines):**
```
GoalGenerator (fake) → Planner → SteeringLibrary → LLM → Reward (fake) → KnowledgeGraph (fake) → Learn
```

**What we added (and why it's garbage):**

| Component | What It Does | Why It's Fake |
|-----------|--------------|---------------|
| **GoalGenerator** | Picks from 12 topics, 16 templates | Not autonomous, just selection |
| **KnowledgeGraph** | Stores `(subject, predicate, "text snippet")` | Never used, just logging |
| **GoalPlanner** | Selects from 4 actions | Overengineered action selection |
| **Reward** | progress + knowledge + coherence + efficiency | None of these are real |

---

## The Knowledge Graph Lie

**v5:**
```python
memory.add_knowledge("black holes", "has_property", "gravity so strong light cannot escape")
```

**What happens to this "knowledge"?**
- Never queried for reasoning
- Never used for inference
- Never synthesized into new insights
- Just... stored. Forever. Unused.

**v1 (honest):**
```python
memory.append(latent_goal, outcome_downsampled)
# Used for ONE thing: novelty computation
novelty = 1.0 - cosine_similarity(outcome, memory.mean())
```

No pretense. It's a buffer for computing novelty. That's it.

---

## The "Goal Generation" Lie

**v5 claims:** "Autonomous goal generation"

**v5 reality:**
```python
TOPIC_POOL = ["photosynthesis", "neural networks", "climate change", ...]  # 12 topics
GOAL_TEMPLATES = {"explore": ["Explore {topic}...", ...], ...}  # 16 templates

goal_type = sample(["explore", "explain", "create", "resolve"])
topic = sample(TOPIC_POOL)
description = random.choice(GOAL_TEMPLATES[goal_type]).format(topic=topic)
```

**This isn't autonomy. It's a menu.**

**v1 (honest):**
- No fake "goal generation"
- Latent goal is a continuous vector, learned by the planner
- The planner learns what to explore based on reward history

---

## The "Task Completion" Lie

**v5 claims:** "Goal-driven task completion"

**v5 reality:**
```python
def execute_action(action, goal, steering):
    output, _ = generate_with_steering(prompt, steering)
    memory.add_knowledge(goal.target, "has_property", output[:50])
    return output, success  # success = len(output) > 20
```

**"Success" is generating >20 characters.**

That's not task completion. That's text generation.

**v1 (honest):**
- No claim of task completion
- Reward is intrinsic (novelty + coherence)
- Success = sustained exploration without collapse

---

## What Got Lost

| v1 | v5 |
|----|-----|
| Honest research question | Fake product pitch |
| 261 lines, clean | 1100+ lines, bloated |
| Simple memory buffer | Fake "knowledge graph" |
| Intrinsic reward (novelty + coherence) | Fake multi-component "task reward" |
| Latent goals (learned vectors) | Fake "goal generation" (menu selection) |
| No pretense of tasks | Fake "task completion" |
| Research prototype | Pretends to be product |

---

## The Path Back

**Strip it down. Go back to v1's core.**

### What to Keep from v5
- Working steering vectors (actually computed from activations)
- Hook-based ActAdd injection
- Rich UI (it's genuinely useful for watching the agent think)

### What to Throw Away
- ❌ GoalGenerator (fake autonomy)
- ❌ KnowledgeGraph (fake knowledge)
- ❌ "Task completion" framing
- ❌ Multi-component fake reward

### What to Restore from v1
- Simple memory buffer (goal, outcome vectors)
- Intrinsic reward (novelty + coherence)
- Honest research question: "Can curiosity sustain exploration?"
- Clean architecture (<400 lines)

---

## The Real Question

**Not:** "How do we make ALMA useful?"

**But:** "What are we actually studying?"

**Honest answer:**
> We're studying whether curiosity-driven exploration can sustain itself in language models, and whether activation steering can guide that exploration.

**That's a real research question.** It doesn't pretend to be a product. It doesn't claim to complete tasks. It's basic research on autonomous cognition.

**If that's not interesting:** Kill the project.

**If it is interesting:** Go back to v1's design, add working steering, make it clean.

---

## prototype.v6.py — What It Should Be

```python
# ~400 lines, honest

class Planner: ...  # Reads memory, outputs latent goal
class Bridge: ...   # Projects latent → steering vector
class Memory: ...   # Buffer of (goal, outcome) for novelty
class Reward: ...   # novelty + coherence, nothing else

# The loop:
for step in range(STEPS):
    latent_goal = planner(memory)
    steering = bridge(latent_goal)
    output = generate_with_steering(prompt, steering)
    outcome = encode(output)
    novelty = compute_novelty(outcome, memory)
    coherence = compute_coherence(output)
    reward = w1 * novelty + w2 * coherence
    planner.update(reward)
    memory.append(latent_goal, outcome)
```

**No fake goals. No fake knowledge. No fake tasks.**

Just: Can curiosity drive learning?

---

## Decision

**Option 1: Go back to v1 + working steering**
- Strip out all the fake product stuff
- Honest research prototype
- ~400 lines, clean
- Research question: curiosity-driven exploration

**Option 2: Pivot to actual task completion**
- Pick ONE real use case (tutor, research assistant, etc.)
- Build proper task specification
- Real completion criteria
- Compare to baselines

**Option 3: Kill it**
- No clear user need
- No advantage over simpler methods
- Opportunity cost too high

**My recommendation:** Option 1 or 3. No more fake product pretending.
