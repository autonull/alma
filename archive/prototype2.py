"""
ALMA v2 — Adaptive Language Model Architecture
===============================================

Core insight: every problem in v1 stems from the same root cause —
a lossy, multi-step pipeline with broken gradient flow.

v1 pipeline: Memory → Planner → Bridge → ActAdd → (frozen engine) → reward → REINFORCE
Problems:
  • Activation injection at layer 24 is brittle; wrong layer = silent failure
  • Bridge alpha never learns (REINFORCE can't reach it through torch.no_grad)
  • REINFORCE on a continuous action space has catastrophically high variance
  • Frozen engine cannot adapt; all learning pressure lands on a tiny planner
  • Deque of [intent|outcome] pairs discards relational structure
  • Single seed prompt creates math attractor with no escape mechanism

v2 architecture:
  • One trainable interface: soft prefix tokens (replaces Planner + Bridge + ActAdd)
  • Direct gradient descent through differentiable reward (replaces REINFORCE)
  • World-model curiosity: learned predictor, not hand-tuned novelty weights
  • Attention-based episodic memory with a learnable query
  • Optional LoRA adapters: beliefs can now update slowly
  • Rotating prompt pool: no attractor basins

Gradient path (v2):
  prefix_policy.parameters()
      → prefix_embeds (in computation graph)
          → model forward pass [teacher-forcing on generated tokens]
              → last hidden state = s_t
                  → reward = coherence(logits, tokens)
                            + diversity(s_t, memory_centroid)
                            + curiosity(world_model, s_{t-1}, s_t)
                  → (-reward).backward()   ← single clean backprop

No REINFORCE. No activation injection. No projection spaces. No broken alpha.
"""

import random
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # Model
    model_id: str = "HuggingFaceTB/SmolLM2-360M"

    # Soft prefix policy
    n_prefix: int = 8            # number of soft-prompt tokens prepended to each forward pass
    prefix_hidden: int = 256     # hidden dim of the prefix MLP

    # Episodic memory
    mem_size: int = 64           # buffer capacity (hidden states)

    # World model (curiosity)
    wm_hidden: int = 256

    # Generation
    max_new_tokens: int = 32
    temperature: float = 0.8
    max_ctx_tokens: int = 64     # rolling context window kept between steps

    # Differentiable reward weights
    w_coherence: float = 1.0    # –log P(tokens | prefix): stay on manifold
    w_diversity: float = 1.5    # cosine distance from memory centroid: explore
    w_curiosity: float = 2.0    # world-model prediction error: seek surprise

    # Optimizers
    policy_lr: float = 3e-3
    wm_lr: float = 1e-3

    # Optional LoRA (pip install peft)
    use_lora: bool = False
    lora_rank: int = 4
    lora_alpha: int = 8          # typically 2× rank
    lora_lr: float = 5e-5
    lora_update_every: int = 20  # LoRA updates less frequently than prefix

    # Training
    n_steps: int = 1000
    log_every: int = 20
    context_reset_every: int = 100  # hard reset to a new seed (escape hatch)

    # Prompt pool — prevents single-seed attractor basins
    prompt_pool: List[str] = field(default_factory=lambda: [
        "The concept of emergence suggests",
        "One surprising property of language is",
        "In the history of mathematics,",
        "The relationship between mind and matter",
        "From an evolutionary perspective,",
        "Consider the paradox of",
        "An underappreciated aspect of consciousness is",
        "The fundamental tension between order and chaos",
        "What remains unexplained in physics is",
        "The boundary between life and non-life",
    ])

    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Episodic Memory
# ─────────────────────────────────────────────────────────────────────────────

class EpisodicMemory(nn.Module):
    """
    Circular buffer of hidden states with attention-based readout.

    Unlike v1's deque of [intent|outcome] pairs, this stores the model's
    own representational states and uses a learnable query for
    context-sensitive retrieval. The query parameter learns — through
    the policy gradient — which past states are most useful for deciding
    where to explore next.
    """

    def __init__(self, mem_size: int, state_dim: int):
        super().__init__()
        self.mem_size = mem_size
        self.state_dim = state_dim

        # Learned query: what part of memory is most useful right now?
        self.query = nn.Parameter(torch.randn(1, state_dim) * 0.02)

        self._buf: deque = deque(maxlen=mem_size)

    def push(self, state: torch.Tensor) -> None:
        """Detach and store a hidden state."""
        self._buf.append(state.detach().squeeze(0).cpu())

    def read(self, device: str) -> torch.Tensor:
        """Softmax-attention over buffer → [1, state_dim] summary."""
        if not self._buf:
            return torch.zeros(1, self.state_dim, device=device)
        stack = torch.stack(list(self._buf)).to(device)          # [N, D]
        scores = (self.query @ stack.T) / (self.state_dim ** 0.5)  # [1, N]
        return torch.softmax(scores, dim=-1) @ stack              # [1, D]

    def centroid(self, device: str) -> torch.Tensor:
        """Mean of all stored states — used for the diversity reward."""
        if not self._buf:
            return torch.zeros(self.state_dim, device=device)
        return torch.stack(list(self._buf)).to(device).mean(0)   # [D]

    def __len__(self) -> int:
        return len(self._buf)


# ─────────────────────────────────────────────────────────────────────────────
# Prefix Policy
# ─────────────────────────────────────────────────────────────────────────────

class PrefixPolicy(nn.Module):
    """
    memory_summary → K soft-prompt embeddings.

    Replaces v1's Planner + Bridge + ActAdd with a single, end-to-end
    differentiable component.

    Why soft prompts instead of activation injection?
      • No layer index to tune — works identically regardless of model depth
      • Gradients flow cleanly from reward → hidden states → prefix embeddings
      • Interpretable: prefix embeddings live in the same space as word embeddings
      • Reversible: remove prefix → baseline behavior, no hooks needed
    """

    def __init__(self, state_dim: int, n_prefix: int, embed_dim: int, hidden: int):
        super().__init__()
        self.n_prefix = n_prefix
        self.embed_dim = embed_dim

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_prefix * embed_dim),
        )

        # Zero-init output layer: prefix starts neutral, learns from scratch
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, memory_summary: torch.Tensor) -> torch.Tensor:
        """[1, D] → [1, n_prefix, embed_dim]"""
        return self.net(memory_summary).view(1, self.n_prefix, self.embed_dim)


# ─────────────────────────────────────────────────────────────────────────────
# World Model
# ─────────────────────────────────────────────────────────────────────────────

class WorldModel(nn.Module):
    """
    Predicts next latent state from current state.

    Trained to minimize prediction error on observed transitions.
    Used as a curiosity signal: states that the world model cannot predict
    well are genuinely novel and earn higher reward. This replaces v1's
    hand-crafted cosine-novelty computation with a learned, adaptive measure.

    Prediction error as curiosity is well-grounded:
      • Familiar territory → low error → low curiosity reward
      • Novel territory → high error → high curiosity reward
      • As more of state-space is explored, world model catches up → curriculum
    """

    def __init__(self, state_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, state_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)  # [1, D] → [1, D]


# ─────────────────────────────────────────────────────────────────────────────
# ALMA v2
# ─────────────────────────────────────────────────────────────────────────────

class ALMAv2:

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.dev = cfg.device

        print(f"Loading {cfg.model_id}…")
        self.tok = AutoTokenizer.from_pretrained(cfg.model_id)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id, torch_dtype=torch.float32
        ).to(self.dev)

        D = self.model.config.hidden_size  # e.g. 960 for SmolLM2-360M
        self.D = D

        # Freeze base weights; LoRA re-enables a small subset if requested
        for p in self.model.parameters():
            p.requires_grad_(False)

        if cfg.use_lora:
            self._attach_lora()

        # Learnable components
        self.memory = EpisodicMemory(cfg.mem_size, D).to(self.dev)
        self.policy = PrefixPolicy(D, cfg.n_prefix, D, cfg.prefix_hidden).to(self.dev)
        self.world_model = WorldModel(D, cfg.wm_hidden).to(self.dev)

        # Policy optimizer: prefix MLP + memory query (learnable readout)
        policy_params = list(self.policy.parameters()) + list(self.memory.parameters())
        self.opt_policy = AdamW(policy_params, lr=cfg.policy_lr, weight_decay=0.01)
        self.opt_wm = AdamW(self.world_model.parameters(), lr=cfg.wm_lr)

        if cfg.use_lora:
            lora_params = [p for p in self.model.parameters() if p.requires_grad]
            self.opt_lora = AdamW(lora_params, lr=cfg.lora_lr)
        else:
            self.opt_lora = None

        # State
        self.prev_state: Optional[torch.Tensor] = None
        self.ctx_ids: Optional[torch.Tensor] = None
        self.step_idx = 0
        self.logs: List[dict] = []

    # ─────────────────────────────────────────────────────────────────────────
    # LoRA setup (requires: pip install peft)
    # ─────────────────────────────────────────────────────────────────────────

    def _attach_lora(self) -> None:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError:
            print("  peft not installed → falling back to frozen engine")
            print("  (pip install peft to enable LoRA)")
            self.cfg.use_lora = False
            return

        lora_cfg = LoraConfig(
            r=self.cfg.lora_rank,
            lora_alpha=self.cfg.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        self.model = get_peft_model(self.model, lora_cfg)
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"  LoRA attached: {trainable:,} / {total:,} params trainable "
              f"({100 * trainable / total:.2f}%)")

    # ─────────────────────────────────────────────────────────────────────────
    # Generation (no gradient)
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _generate(self, prefix_embeds: torch.Tensor,
                  ctx_ids: torch.Tensor) -> torch.Tensor:
        """
        Generate tokens conditioned on soft prefix + context.
        prefix_embeds is detached here; we re-attach it with grad in
        the teacher-forcing pass below.
        Returns generated token IDs: [1, T]
        """
        embed = self.model.get_input_embeddings()
        ctx_embeds = embed(ctx_ids)                              # [1, ctx, D]
        input_embeds = torch.cat([prefix_embeds, ctx_embeds], dim=1)

        out = self.model.generate(
            inputs_embeds=input_embeds,
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=True,
            temperature=self.cfg.temperature,
            pad_token_id=self.tok.eos_token_id,
            use_cache=True,
        )
        return out  # [1, T]  (generated tokens only, not input)

    # ─────────────────────────────────────────────────────────────────────────
    # Teacher-forcing forward pass (with gradient)
    # ─────────────────────────────────────────────────────────────────────────

    def _forward_with_grad(self, prefix_embeds: torch.Tensor,
                           gen_ids: torch.Tensor) -> tuple:
        """
        Run a forward pass over [prefix_embeds | embed(gen_ids)] with
        gradient tracking enabled.

        Because prefix_embeds IS in the computation graph (output of
        self.policy), gradients flow:
            reward → last_hidden → forward_pass → prefix_embeds → policy

        Returns:
            state:  [1, D]   last hidden state (the "outcome" in v1 terms)
            logits: [1, T, V] token logits for the coherence reward
        """
        embed = self.model.get_input_embeddings()
        tok_embeds = embed(gen_ids)                              # [1, T, D]
        full_embeds = torch.cat([prefix_embeds, tok_embeds], dim=1)  # [1, n+T, D]

        out = self.model(inputs_embeds=full_embeds, output_hidden_states=True)

        last_hidden = out.hidden_states[-1][:, -1, :]            # [1, D]
        return last_hidden, out.logits                           # [1, n+T, V]

    # ─────────────────────────────────────────────────────────────────────────
    # Differentiable reward
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_reward(self, state: torch.Tensor,
                        logits: torch.Tensor,
                        gen_ids: torch.Tensor) -> dict:
        """
        Three reward components, all differentiable w.r.t. `state`
        (and thus w.r.t. prefix_embeds through the forward pass).

        coherence:
            –log P(generated tokens | prefix).  High coherence = low perplexity.
            Gradient pushes prefix toward token sequences the model considers likely.

        diversity:
            1 – cosine_sim(state, memory_centroid).
            Gradient pushes prefix toward underexplored regions of state space.

        curiosity:
            MSE(world_model_prediction, state).
            The world model is evaluated with no_grad so gradient flows only
            to `state` (and back to prefix), not to world model parameters.
            World model is trained separately below.
        """
        cfg = self.cfg
        n = cfg.n_prefix
        T = gen_ids.shape[1]

        # 1. Coherence: log P(tokens | prefix)
        if T > 1:
            # logits[:, n-1 : n+T-1, :] predicts gen_ids[:, 0 : T]
            gen_logits = logits[:, n - 1: n + T - 1, :]       # [1, T, V]
            coherence = -F.cross_entropy(
                gen_logits.reshape(-1, gen_logits.shape[-1]),
                gen_ids.reshape(-1),
            )
        else:
            coherence = torch.zeros(1, device=self.dev).squeeze()

        # 2. Diversity: distance from memory centroid
        centroid = self.memory.centroid(self.dev)               # [D]
        cos_sim = F.cosine_similarity(state, centroid.unsqueeze(0), dim=-1).squeeze()
        diversity = (1.0 - cos_sim).clamp(0.0, 2.0)

        # 3. Curiosity: prediction error of world model on current state
        #    world model evaluated with no_grad so gradient flows only to state
        if self.prev_state is not None:
            with torch.no_grad():
                predicted = self.world_model(self.prev_state)   # [1, D]
            curiosity = F.mse_loss(state, predicted)
        else:
            curiosity = torch.zeros(1, device=self.dev).squeeze()

        total = (cfg.w_coherence * coherence
                 + cfg.w_diversity * diversity
                 + cfg.w_curiosity * curiosity)

        return dict(total=total, coherence=coherence,
                    diversity=diversity, curiosity=curiosity)

    # ─────────────────────────────────────────────────────────────────────────
    # Single autonomy step
    # ─────────────────────────────────────────────────────────────────────────

    def step(self) -> dict:
        cfg = self.cfg

        # ── Context management ──────────────────────────────────────────────
        if self.ctx_ids is None or self.step_idx % cfg.context_reset_every == 0:
            seed = random.choice(cfg.prompt_pool)
            self.ctx_ids = self.tok(seed, return_tensors="pt").input_ids.to(self.dev)

        # ── 1. Generate prefix from current memory state ─────────────────────
        summary = self.memory.read(self.dev)                    # [1, D]
        prefix = self.policy(summary)                           # [1, n_prefix, D]

        # ── 2. Generate tokens (no grad; uses detached prefix) ───────────────
        gen_ids = self._generate(prefix.detach(), self.ctx_ids)
        text = self.tok.decode(gen_ids[0], skip_special_tokens=True)

        # ── 3. Teacher-forcing forward pass (prefix in computation graph) ────
        state, logits = self._forward_with_grad(prefix, gen_ids)

        # ── 4. Differentiable reward ─────────────────────────────────────────
        reward_dict = self._compute_reward(state, logits, gen_ids)
        total_reward = reward_dict["total"]

        # ── 5. Policy gradient (direct, no REINFORCE) ────────────────────────
        self.opt_policy.zero_grad()
        (-total_reward).backward()
        nn.utils.clip_grad_norm_(
            list(self.policy.parameters()) + list(self.memory.parameters()),
            max_norm=1.0,
        )
        self.opt_policy.step()

        # ── 6. World model: supervised on (prev → curr) transition ───────────
        if self.prev_state is not None:
            self.opt_wm.zero_grad()
            wm_loss = F.mse_loss(
                self.world_model(self.prev_state),
                state.detach(),
            )
            wm_loss.backward()
            self.opt_wm.step()

        # ── 7. Optional LoRA update (less frequent, separate pass) ───────────
        if (self.opt_lora is not None
                and self.step_idx % cfg.lora_update_every == 0
                and gen_ids.shape[1] > 1):
            self.opt_lora.zero_grad()
            _, logits2 = self._forward_with_grad(prefix.detach(), gen_ids)
            n = cfg.n_prefix
            T = gen_ids.shape[1]
            lm_loss = F.cross_entropy(
                logits2[:, n - 1: n + T - 1].reshape(-1, logits2.shape[-1]),
                gen_ids.reshape(-1),
            )
            lm_loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            self.opt_lora.step()

        # ── 8. Bookkeeping ───────────────────────────────────────────────────
        self.memory.push(state)
        self.prev_state = state.detach()

        # Roll context forward (sliding window)
        full = torch.cat([self.ctx_ids, gen_ids], dim=1)
        self.ctx_ids = full[:, -cfg.max_ctx_tokens:]

        self.step_idx += 1

        log = dict(
            step=self.step_idx,
            reward=total_reward.item(),
            coherence=reward_dict["coherence"].item(),
            diversity=reward_dict["diversity"].item(),
            curiosity=reward_dict["curiosity"].item(),
            text=text,
        )
        self.logs.append(log)
        return log

    # ─────────────────────────────────────────────────────────────────────────
    # Main loop
    # ─────────────────────────────────────────────────────────────────────────

    def run(self) -> List[dict]:
        cfg = self.cfg
        print(f"\nALMA v2  |  {cfg.model_id}  |  {cfg.n_steps} steps")
        print(f"  prefix: {cfg.n_prefix} tokens  "
              f"mem: {cfg.mem_size}  "
              f"lora: {cfg.use_lora}")
        print("─" * 72)

        for _ in range(cfg.n_steps):
            log = self.step()

            if log["step"] % cfg.log_every == 0:
                snippet = log["text"].replace("\n", " ")[-100:]
                print(
                    f"Step {log['step']:5d} | "
                    f"R={log['reward']:+.3f}  "
                    f"Coh={log['coherence']:+.3f}  "
                    f"Div={log['diversity']:.3f}  "
                    f"Cur={log['curiosity']:.4f}"
                )
                print(f"  ↳ …{snippet}\n")

        print("Done.")
        return self.logs


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = Config()
    # cfg.use_lora = True    # uncomment for adaptive beliefs
    # cfg.model_id = "HuggingFaceTB/SmolLM2-135M"  # faster iteration
    alma = ALMAv2(cfg)
    alma.run()
