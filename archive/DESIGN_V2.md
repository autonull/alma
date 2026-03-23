# ALMA v2 — Design Specification

## The Core Problem with v1

Every issue in v1 traces to a single root cause: **the pipeline has too many seams, and gradient flow breaks at each one.**

```
Memory → Planner → Bridge → ActAdd → (no_grad engine) → reward → REINFORCE
                     ↑ alpha never learns (stuck behind no_grad)
                                                          ↑ high variance
```

v2 collapses this into a single differentiable path:

```
Memory → PrefixPolicy → prefix_embeds → model forward → state → reward → .backward()
```

---

## Architecture Comparison

| Aspect | v1 | v2 |
|---|---|---|
| Steering interface | Activation injection at layer 24 | Soft prefix tokens (K × embed_dim) |
| Components for steering | Planner + Bridge + ActAdd (3 modules) | PrefixPolicy (1 MLP) |
| Learning algorithm | REINFORCE (high variance, discrete) | Direct gradient descent |
| Novelty signal | Hand-crafted cosine distance | World model prediction error |
| Reward differentiability | Non-differentiable (REINFORCE required) | Fully differentiable |
| Memory readout | `deque[-1]` (last element only) | Learnable attention query |
| Memory content | `[intent ‖ outcome]` concatenation | Raw hidden states |
| Gradient flow to alpha | Broken (behind `torch.no_grad`) | N/A — no alpha parameter |
| Engine adaptability | Fully frozen | Frozen + optional LoRA |
| Attractor mitigation | None | Rotating prompt pool + curiosity |
| Context management | Full sequence carried forward | Sliding window (max_ctx_tokens) |

---

## Components

### 1. EpisodicMemory

A circular buffer of hidden states with a **learnable query parameter** for attention-based readout.

- **Stores**: raw last-layer hidden states (detached), shape `[D]`
- **Reads**: `softmax(query @ stack.T / √D) @ stack` → `[1, D]`
- **Centroid**: mean of all stored states → used for diversity reward
- **Why better than v1**: The query learns (through policy gradients) which past experiences matter for deciding what to explore next. v1's deque just used the last element.

### 2. PrefixPolicy

A 3-layer MLP that maps `memory_summary [1, D]` → `K soft-prompt embeddings [1, K, D]`.

- **Replaces**: LatentPlanner + TheBridge + ActAddHook
- **Interface**: prefix tokens are prepended to every forward pass — no layer index, no hooks, no projection spaces
- **Initialization**: output layer zero-initialized → prefix starts neutral
- **Why better**: gradients flow directly through the model's own attention mechanism from reward to policy parameters. No intermediate projection spaces to lose information through, no layer selection hyperparameter.

### 3. WorldModel

A 3-layer MLP trained to predict `state_{t+1}` from `state_t`.

```
curiosity(t) = MSE(world_model(state_{t-1}), state_t)
```

- **Trained**: minimize prediction error on observed transitions (supervised, separate optimizer)
- **Used for reward**: prediction error evaluated with `no_grad` so gradient flows only to `state_t` (and back through the forward pass to the prefix)
- **Why better than hand-crafted novelty**: adaptive — as the model explores a region repeatedly, the world model learns to predict it and curiosity drops, creating implicit curriculum. v1's cosine-distance novelty had no such self-regulation.

### 4. Differentiable Reward

Three terms, all computable from the teacher-forcing forward pass:

```
R = w_coh · coherence  +  w_div · diversity  +  w_cur · curiosity

coherence  = −log P(generated_tokens | prefix)       # standard LM cross-entropy
diversity  = 1 − cosine_sim(state, memory_centroid)  # distance from seen regions
curiosity  = MSE(world_model(s_{t-1}), s_t)          # prediction error
```

**All three are differentiable w.r.t. `state`**, which is differentiable w.r.t. `prefix_embeds` through the model's forward pass. This is why REINFORCE is unnecessary.

---

## Training Algorithm

Each step:

```
1. READ MEMORY
   summary = memory.read()               # [1, D], uses learnable query

2. GENERATE PREFIX
   prefix = policy(summary)              # [1, K, D], in computation graph

3. GENERATE TOKENS  [no grad]
   gen_ids = model.generate(
       inputs_embeds = cat(prefix.detach(), embed(ctx_ids))
   )

4. TEACHER-FORCING PASS  [with grad]
   state, logits = model(
       inputs_embeds = cat(prefix, embed(gen_ids))
   )
   # gradient flows: state → forward_pass → prefix → policy

5. COMPUTE REWARD  [differentiable]
   R = coherence(logits, gen_ids)
     + diversity(state, memory.centroid())
     + curiosity(world_model.no_grad(prev_state), state)

6. UPDATE POLICY
   opt_policy.zero_grad()
   (−R).backward()
   clip_grad_norm_(policy + memory.query, 1.0)
   opt_policy.step()

7. UPDATE WORLD MODEL  [separate, supervised]
   wm_loss = MSE(world_model(prev_state), state.detach())
   wm_loss.backward()
   opt_wm.step()

8. UPDATE MEMORY
   memory.push(state.detach())
   prev_state = state.detach()
   ctx_ids = rolling_window(ctx_ids + gen_ids)
```

---

## Why Direct Gradient Descent Works Here

The trick is the **teacher-forcing forward pass** (step 4):

- `gen_ids` is sampled with `prefix.detach()` → discrete, no grad
- The teacher-forcing pass runs `model(inputs_embeds=cat(prefix, embed(gen_ids)))` where `prefix` IS in the computation graph
- Attention mixes prefix embeddings into every token's representation
- Therefore `state = last_hidden` depends on `prefix` through attention
- Therefore `R` depends on `prefix` → `.backward()` works

This is the same principle as prompt tuning / prefix tuning. The discrete generation step is not differentiated through; instead, the prefix is optimized to produce high-reward *representations* given any text the model generates.

---

## Optional LoRA

With `use_lora = True` (requires `pip install peft`):

- Rank-4 LoRA applied to Q and V projections of all attention layers
- Trainable parameters: ~0.1% of total (e.g., ~360K out of 360M)
- Updated every `lora_update_every` steps using the LM coherence loss
- Separate optimizer, separate backward pass
- Effect: the model slowly adapts its own language modeling to the domain it has been exploring

The "frozen beliefs" invariant of v1 was philosophically motivated but practically limiting. LoRA preserves 99.9% of pretrained knowledge while allowing genuine adaptation — a better tradeoff.

---

## Attractor Mitigation

v1's math attractor came from:
1. A fixed seed: `"Thinking process initiated."` biased all subsequent generations
2. No explicit escape mechanism when novelty collapsed

v2 addresses both:
1. **Prompt pool**: every `context_reset_every` steps, a new seed is sampled from a diverse pool
2. **Curiosity reward**: world model prediction error drops in familiar territory, naturally pushing the system toward unexplored regions without manual weight tuning
3. **Diversity reward**: explicit penalty for staying near the memory centroid

---

## Hyperparameter Sensitivity Analysis

| Parameter | v1 sensitivity | v2 sensitivity |
|---|---|---|
| Injection layer (24) | High — wrong layer = silent failure | N/A |
| alpha (bridge magnitude) | High — doesn't learn anyway | N/A |
| w1/w2 (novelty/coherence) | High — manual reactive tuning needed | Lower — curiosity self-regulates |
| n_prefix | Low — 4–16 all work | Low |
| REINFORCE batch size | High — affects variance directly | N/A |
| Learning rate | Moderate | Moderate |

---

## Scaling Notes

| Model | D (hidden_size) | PrefixPolicy output | Memory per step |
|---|---|---|---|
| SmolLM2-135M | 576 | 8 × 576 = 4,608 | Low |
| SmolLM2-360M | 960 | 8 × 960 = 7,680 | Moderate |
| SmolLM2-1.7B | 2,048 | 8 × 2,048 = 16,384 | Higher |

The prefix policy and world model scale trivially. The teacher-forcing pass requires storing activations for backprop — this is the main memory cost, proportional to sequence length × hidden size × number of layers.

For larger models, reduce `n_prefix` (4 is sufficient) or reduce `max_new_tokens` to manage memory.
