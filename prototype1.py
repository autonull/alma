# !pip install -U "bitsandbytes>=0.46.1" transformers accelerate

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import deque
import warnings

warnings.filterwarnings("ignore")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================================================================
# 1. ARCHITECTURE & TENSOR SPECS
# ==============================================================================

class LatentPlanner(nn.Module):
    def __init__(self, hidden_dim=256, mem_dim=512):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=mem_dim, nhead=8, dim_feedforward=1024, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.mode_emb = nn.Embedding(2, mem_dim)
        
        self.latent_goal_head = nn.Linear(mem_dim, hidden_dim)
        self.mode_logits_head = nn.Linear(mem_dim, 2)
        self.sub_goal_head = nn.Linear(mem_dim, 1)

    def forward(self, memory_buffer, mode):
        mode_vec = self.mode_emb(mode).unsqueeze(1) 
        x = torch.cat([memory_buffer, mode_vec], dim=1)
        out = self.transformer(x)
        pooled = out[:, -1, :] 
        
        latent_mean = self.latent_goal_head(pooled)
        mode_logits = self.mode_logits_head(pooled)
        sub_goal_count = self.sub_goal_head(pooled)
        
        return latent_mean, mode_logits, sub_goal_count

class TheBridge(nn.Module):
    def __init__(self, latent_dim=256, phi_dim=960):
        super().__init__()
        self.up_projection = nn.Linear(latent_dim, phi_dim, bias=False)
        self.down_projection = nn.Linear(phi_dim, latent_dim, bias=False)
        # CHANGED: Start Alpha higher (0.5) to force early steering impact
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def get_steering_vector(self, latent_goal):
        raw_steering = self.up_projection(latent_goal)
        normed = raw_steering / (raw_steering.norm(dim=-1, keepdim=True) + 1e-8)
        clamped_alpha = torch.clamp(self.alpha, min=0.01, max=3.0)
        return normed * clamped_alpha

class MemoryBuffer:
    def __init__(self, size=16, dim=512):
        self.size = size
        self.buffer = deque(
            [torch.zeros(dim, device=DEVICE, dtype=torch.float32) for _ in range(size)], 
            maxlen=size
        )

    def append(self, latent_goal, outcome_downsampled):
        slot = torch.cat([latent_goal.squeeze(0), outcome_downsampled.squeeze(0)])
        self.buffer.append(slot.detach())

    def get_tensor(self):
        return torch.stack(list(self.buffer)).unsqueeze(0)

# ==============================================================================
# 2. SETUP WORLD ENGINE (SmolLM2-360M)
# ==============================================================================
# (Assuming it is already loaded from the previous cell, but re-fetching pointers)
print("Linking World Engine (SmolLM2-360M)...")
model_id = "HuggingFaceTB/SmolLM2-360M"
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ==============================================================================
# 3. INITIALIZE AGENT & OPTIMIZER (Fresh Brain)
# ==============================================================================

planner = LatentPlanner().to(DEVICE)
bridge = TheBridge(phi_dim=960).to(DEVICE) 
optimizer = optim.AdamW(list(planner.parameters()) + list(bridge.parameters()), lr=3e-4)

memory = MemoryBuffer()
current_mode = torch.tensor([0], device=DEVICE)
prompt_ids = tokenizer.encode("Thinking process initiated.", return_tensors="pt").to(DEVICE)

# TUNERS: Start with a heavier focus on Novelty
w1, w2, w3 = 1.0, 0.5, 0.2
STEPS = 400
BATCH_SIZE = 8

# ==============================================================================
# 4. ACTADD HOOK MANAGER
# ==============================================================================

class ActAddHook:
    def __init__(self, model, layer_idx=24):
        self.module = model.model.layers[layer_idx]
        self.handle = None
        self.steering_vector = None

    def attach(self, steering_vector):
        self.steering_vector = steering_vector
        def hook_fn(module, inputs, output):
            if isinstance(output, tuple):
                h = output[0]
                modified_hidden = h + self.steering_vector.to(h.dtype)
                return (modified_hidden,) + output[1:]
            else:
                h = output
                modified_hidden = h + self.steering_vector.to(h.dtype)
                return modified_hidden
                
        self.handle = self.module.register_forward_hook(hook_fn)

    def remove(self):
        if self.handle:
            self.handle.remove()

hook = ActAddHook(world_engine, layer_idx=24)

# ==============================================================================
# 5. THE TUNED AUTONOMY LOOP
# ==============================================================================

print("\nStarting Continuous Curiosity Loop v3.2 (Anti-Mode Collapse)...")

log_probs_buffer = []
rewards_buffer = []
depth_counter = 0

for step in range(1, STEPS + 1):
    planner.train()
    bridge.train()
    
    mem_tensor = memory.get_tensor()
    latent_mean, mode_logits, sub_goal_count = planner(mem_tensor, current_mode)
    
    # CHANGED: Increase exploration noise from 0.1 to 0.3
    std = 0.3
    latent_dist = Normal(latent_mean, std)
    latent_goal = latent_dist.sample()
    
    mode_dist = Categorical(logits=mode_logits)
    sampled_mode = mode_dist.sample()
    
    log_prob = mode_dist.log_prob(sampled_mode) + latent_dist.log_prob(latent_goal).sum(dim=-1)
    log_probs_buffer.append(log_prob)
    
    steering_vector = bridge.get_steering_vector(latent_goal)
    current_alpha = bridge.alpha.item()
    
    hook.attach(steering_vector)
    
    # Expand context window slightly to help it remember train of thought
    input_ids = prompt_ids[:, -64:] 
    attention_mask = torch.ones_like(input_ids) 
    
    with torch.no_grad():
        outputs = world_engine.generate(
            input_ids, 
            attention_mask=attention_mask,
            max_new_tokens=8, 
            output_hidden_states=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True, # Added slight temperature sampling to prevent hard loops
            temperature=0.7 
        )
    
    generated_tokens = outputs.sequences
    generated_text = tokenizer.decode(generated_tokens[0][-8:], skip_special_tokens=True)
    final_hidden_state = outputs.hidden_states[-1][-1][:, -1, :].to(torch.float32) 
    hook.remove()
    
    with torch.no_grad():
        full_attention_mask = torch.ones_like(generated_tokens)
        loss_output = world_engine(
            input_ids=generated_tokens, 
            attention_mask=full_attention_mask, 
            labels=generated_tokens
        )
        # CHANGED: Use raw Cross-Entropy Loss instead of Perplexity for linear scaling
        coherence_loss = loss_output.loss.item() 
        
    outcome_downsampled = bridge.down_projection(final_hidden_state)
    
    mem_outcomes = torch.stack([m[256:] for m in memory.buffer])
    mean_outcome = mem_outcomes.mean(dim=0, keepdim=True)
    novelty = 1.0 - F.cosine_similarity(outcome_downsampled, mean_outcome, dim=-1).item()
    
    memory.append(latent_goal, outcome_downsampled)
    current_mode = sampled_mode
    prompt_ids = generated_tokens
    
    if current_mode.item() == 1 and novelty < 0.2: 
        depth_counter += 1
    else:
        depth_counter = 0
    depth_bonus = 1.0 if depth_counter > 3 else 0.0
    
    # CHANGED: Multiply novelty by 5.0 so the agent actually cares about it
    reward = (w1 * novelty * 5.0) - (w2 * coherence_loss) + (w3 * depth_bonus)
    rewards_buffer.append(reward)
    
    # CHANGED: Aggressive anti-repetition tuners
    if novelty < 0.05:
        w1 = min(w1 * 1.1, 3.0) # Panic! I'm bored. Boost curiosity.
    if coherence_loss > 3.0: # (equivalent to PPL > 20)
        w2 = min(w2 * 1.1, 2.0) # Panic! I'm speaking gibberish. Boost grammar.
    elif novelty > 0.15 and coherence_loss < 2.0:
        # I am doing well! Relax weights back to baseline
        w1 = max(w1 * 0.98, 1.0)
        w2 = max(w2 * 0.98, 0.5)
        
    # REINFORCE POLICY UPDATE
    if step % BATCH_SIZE == 0:
        optimizer.zero_grad()
        
        R = 0
        returns = []
        for r in rewards_buffer[::-1]:
            R = r + 0.90 * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns, device=DEVICE, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        loss = []
        for lp, R_t in zip(log_probs_buffer, returns):
            loss.append(-lp * R_t)
        
        total_loss = torch.stack(loss).mean()
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(planner.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
        
        optimizer.step()
        
        log_probs_buffer.clear()
        rewards_buffer.clear()

    if step % 50 == 0:
        mode_str = "H" if current_mode.item() == 1 else "F"
        norm_val = (steering_vector.norm(dim=-1).mean()).item()
        clean_text = generated_text.replace('\n', ' ').strip()
        
        # Note: Log now shows 'Loss' instead of 'PPL'
        print(f"Step {step:04d} | Loss: {coherence_loss:.2f} | Nov: {novelty:.3f} | "
              f"Norm: {norm_val:.2f} | Alpha: {current_alpha:.2f} | Mode: {mode_str} | "
              f"W1:{w1:.2f} W2:{w2:.2f} | Text: \"{clean_text}\"")
