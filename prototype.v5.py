#!/usr/bin/env python3
"""
ALMA v5 — Autonomous Goal-Driven Language Agent
Proof-of-Concept with Rich Terminal UI

A self-contained trainer/demonstration showing an autonomous agent that:
- Generates its own goals
- Pursues goals over multiple steps
- Tracks progress in structured memory
- Revises strategy based on feedback
- Produces useful artifacts

Run: python prototype.v5.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import json
import os
import sys
import time
import math

# Rich terminal UI
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.syntax import Syntax
from rich.style import Style
from rich.color import Color
import random

# Check for GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================================================================
# RICH TERMINAL UI COMPONENTS
# ==============================================================================

class ALMAUI:
    """Badass terminal UI for ALMA v5 - inspired by DAW + city-builder games."""
    
    # Color palette - using valid Rich color names
    COLORS = {
        'teal': 'cyan',
        'purple': 'magenta', 
        'amber': 'yellow',
        'coral': 'red',
        'navy': 'blue',
        'green': 'green',
        'blue': 'blue',
    }
    
    EMOJIS = {
        'goal': '🎯',
        'plan': '📋',
        'execute': '⚡',
        'memory': '💾',
        'reward': '💎',
        'success': '✅',
        'failure': '❌',
        'thinking': '🤔',
        'explore': '🔍',
        'explain': '📖',
        'create': '🎨',
        'resolve': '✓',
        'fire': '🔥',
        'star': '⭐',
        'rocket': '🚀',
        'brain': '🧠',
        'spark': '✨',
    }
    
    def __init__(self):
        self.console = Console()
        self.step = 0
        self.current_goal = None
        self.current_action = None
        self.reward_history = []
        self.goals_completed = 0
        self.knowledge_graph_size = 0
        self.agent_state = "Initializing"
        self.discoveries = []
        self.biome = "Wandering"
        
    def make_layout(self) -> Layout:
        """Create the main UI layout."""
        layout = Layout()
        
        # Split into header, body, footer
        layout.split(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=5),
        )
        
        # Split body into left (stats), center (main), right (memory)
        layout["body"].split_row(
            Layout(name="left", size=40),
            Layout(name="center"),
            Layout(name="right", size=50),
        )
        
        return layout
    
    def render_header(self) -> Panel:
        """Render the transport bar / header."""
        state_colors = {
            "Initializing": "[bright_black]",
            "Running": "[green]",
            "Paused": "[yellow]",
            "Goal Pursuit": "[cyan]",
            "Reflection": "[magenta]",
            "Complete": "[green bold]",
        }
        
        # Biome status with color
        biome_colors = {
            "Wandering": "cyan",
            "Exploring": "magenta", 
            "Focused": "yellow",
            "Productive": "green",
            "Stuck": "red",
        }
        biome_color = biome_colors.get(self.biome, "cyan")
        
        header_text = Text()
        header_text.append(f"{self.EMOJIS['rocket']} ALMA v5 ", style="bold cyan")
        header_text.append(f"| Step: {self.step:04d} ", style="white")
        header_text.append(f"| State: {state_colors.get(self.agent_state, '[white]')}{self.agent_state}[/] ", style="")
        header_text.append(f"| Biome: [{biome_color}]{self.biome}[/] ", style="")
        header_text.append(f"| Goals: {self.goals_completed} ", style="white")
        header_text.append(f"| Knowledge: {self.knowledge_graph_size} facts ", style="white")
        header_text.append(f"| Time: {datetime.now().strftime('%H:%M:%S')}", style="bright_black")
        
        return Panel(header_text, style="dim blue", border_style="cyan")
    
    def render_goal_panel(self) -> Panel:
        """Render current goal and status."""
        if self.current_goal:
            goal_type_emoji = {
                "explore": self.EMOJIS['explore'],
                "explain": self.EMOJIS['explain'],
                "create": self.EMOJIS['create'],
                "resolve": self.EMOJIS['resolve'],
            }
            emoji = goal_type_emoji.get(self.current_goal.get('type', 'explore'), self.EMOJIS['goal'])
            
            goal_text = Text()
            goal_text.append(f"{emoji} Current Goal\n\n", style="bold cyan")
            goal_text.append(f"Type: {self.current_goal.get('type', 'unknown')}\n", style="white")
            goal_text.append(f"Target: {self.current_goal.get('target', 'N/A')}\n", style="bright_black")
            goal_text.append(f"Progress: ", style="white")
            
            # Progress bar
            progress = self.current_goal.get('progress', 0)
            bar_length = 20
            filled = int(progress * bar_length)
            goal_text.append("█" * filled + "░" * (bar_length - filled), style="green")
            goal_text.append(f" {progress*100:.0f}%\n", style="green")
            
            if self.current_action:
                goal_text.append(f"\n{self.EMOJIS['spark']} Action: {self.current_action}", style="yellow")
        else:
            goal_text = Text()
            goal_text.append("No active goal", style="bright_black")
            goal_text.append("\n\nWaiting for goal generator...", style="bright_black italic")
        
        return Panel(goal_text, title="🎯 Goal Status", border_style="cyan")
    
    def render_stats_panel(self) -> Panel:
        """Render key metrics."""
        # Calculate average reward
        avg_reward = sum(self.reward_history[-50:]) / max(len(self.reward_history[-50:]), 1)
        
        # Reward trend
        if len(self.reward_history) >= 10:
            recent = self.reward_history[-10:]
            trend = "↑" if recent[-1] > recent[0] else "↓" if recent[-1] < recent[0] else "→"
            trend_style = "green" if trend == "↑" else "red" if trend == "↓" else "yellow"
        else:
            trend, trend_style = "→", "yellow"
        
        stats_table = Table.grid(padding=1)
        stats_table.add_column(style="bright_black", justify="left")
        stats_table.add_column(style="white", justify="right")
        
        stats_table.add_row("Total Steps:", f"{self.step}")
        stats_table.add_row("Goals Completed:", f"{self.goals_completed} {self.EMOJIS['star']}")
        stats_table.add_row("Knowledge Facts:", f"{self.knowledge_graph_size}")
        stats_table.add_row("Avg Reward:", f"{avg_reward:.3f} {trend}", style=trend_style)
        stats_table.add_row("Memory Size:", f"{len(self.reward_history)}")
        
        # Reward sparkline
        if len(self.reward_history) >= 20:
            spark = self._make_sparkline(self.reward_history[-20:])
            stats_table.add_row("Reward Trend:", spark)
        
        return Panel(stats_table, title="📊 Metrics", border_style="magenta")
    
    def _make_sparkline(self, values: List[float]) -> str:
        """Create a text sparkline."""
        if not values:
            return ""
        min_v, max_v = min(values), max(values)
        range_v = max_v - min_v + 0.001
        bars = ["_", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
        return "".join(bars[int((v - min_v) / range_v * (len(bars) - 1))] for v in values)
    
    def render_memory_panel(self, memory: 'GoalMemory') -> Panel:
        """Render memory status."""
        mem_text = Text()
        mem_text.append(f"{self.EMOJIS['memory']} Knowledge Graph\n", style="bold magenta")
        mem_text.append(f"  Facts: {memory.knowledge_graph_size()} triples\n", style="white")
        
        mem_text.append(f"\n{self.EMOJIS['memory']} Attempt Log\n", style="bold yellow")
        recent_attempts = memory.get_recent_attempts(5)
        for attempt in recent_attempts:
            status = self.EMOJIS['success'] if attempt['success'] else self.EMOJIS['failure']
            mem_text.append(f"  {status} {attempt['action']}: {attempt['goal_type']}\n", 
                          style="green" if attempt['success'] else "red")
        
        return Panel(mem_text, title="💾 Memory", border_style="yellow")
    
    def render_output_panel(self, recent_output: str) -> Panel:
        """Render the most recent generated output."""
        output_text = Text()
        output_text.append("📝 Latest Output\n\n", style="bold cyan")
        
        # Truncate if too long
        if len(recent_output) > 500:
            recent_output = recent_output[:497] + "..."
        
        output_text.append(recent_output, style="white")
        
        return Panel(output_text, title="📄 Output", border_style="blue")
    
    def render_discoveries_panel(self) -> Panel:
        """Render recent discoveries/achievements."""
        disc_text = Text()
        disc_text.append(f"{self.EMOJIS['star']} Recent Discoveries\n\n", style="bold yellow")
        
        if not self.discoveries:
            disc_text.append("Making discoveries...", style="bright_black italic")
        else:
            for disc in self.discoveries[-5:]:
                disc_text.append(f"  {self.EMOJIS['spark']} {disc}\n", style="cyan")
        
        return Panel(disc_text, title="🏆 Discoveries", border_style="red")
    
    def render_log_panel(self, log_entries: List[str]) -> Panel:
        """Render action log."""
        log_text = Text()
        log_text.append("⚡ Action Log\n\n", style="bold green")
        
        for entry in log_entries[-10:]:
            log_text.append(f"  {entry}\n", style="bright_black")
        
        return Panel(log_text, title="📜 Log", border_style="green")
    
    def update_state(self, **kwargs):
        """Update UI state."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def add_discovery(self, discovery: str):
        """Add a discovery to the feed."""
        self.discoveries.append(discovery)
        if len(self.discoveries) > 20:
            self.discoveries.pop(0)


# ==============================================================================
# ALMA v5 CORE ARCHITECTURE
# ==============================================================================

@dataclass
class Goal:
    """Represents an autonomous goal."""
    goal_type: str  # explore, explain, create, resolve
    target: str
    description: str
    priority: float = 0.5
    progress: float = 0.0
    sub_goals: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'type': self.goal_type,
            'target': self.target,
            'description': self.description,
            'progress': self.progress,
            'metadata': self.metadata
        }


class KnowledgeGraph:
    """Structured knowledge storage."""
    
    def __init__(self):
        self.triples = []  # (subject, predicate, object, confidence)
        self.entities = set()
        self.relations = defaultdict(list)
        
    def add(self, subject: str, predicate: str, obj: str, confidence: float = 0.8):
        """Add a fact to the knowledge graph."""
        self.triples.append((subject, predicate, obj, confidence))
        self.entities.add(subject)
        self.entities.add(obj)
        self.relations[subject].append((predicate, obj, confidence))
        
    def query(self, entity: str) -> List[Tuple]:
        """Query facts about an entity."""
        return self.relations.get(entity, [])
    
    def size(self) -> int:
        return len(self.triples)
    
    def to_text(self) -> str:
        """Export knowledge as text."""
        lines = []
        for s, p, o, c in self.triples[-10:]:  # Recent 10
            lines.append(f"  • {s} --{p}--> {o} (conf: {c:.2f})")
        return "\n".join(lines)


class GoalMemory:
    """Structured memory for goal-directed behavior."""
    
    def __init__(self, max_attempts: int = 100):
        self.knowledge_graph = KnowledgeGraph()
        self.attempt_log = deque(maxlen=max_attempts)
        self.goal_history = deque(maxlen=50)
        
    def add_knowledge(self, subject: str, predicate: str, obj: str, confidence: float = 0.8):
        """Add a fact."""
        self.knowledge_graph.add(subject, predicate, obj, confidence)
        
    def log_attempt(self, goal_type: str, action: str, output: str, success: bool):
        """Log an action attempt."""
        self.attempt_log.append({
            'goal_type': goal_type,
            'action': action,
            'output': output[:200],  # Truncate
            'success': success,
            'timestamp': time.time()
        })
        
    def add_goal(self, goal: Goal):
        """Record a goal in history."""
        self.goal_history.append({
            'type': goal.goal_type,
            'target': goal.target,
            'completed': goal.progress >= 0.95,
            'progress': goal.progress
        })
        
    def get_recent_attempts(self, n: int = 5) -> List[Dict]:
        """Get recent attempts."""
        return list(self.attempt_log)[-n:]
    
    def knowledge_graph_size(self) -> int:
        return self.knowledge_graph.size()
    
    def get_context(self) -> str:
        """Get memory context for planner."""
        recent = self.get_recent_attempts(3)
        if not recent:
            return "No prior attempts."
        
        context = "Recent attempts:\n"
        for att in recent:
            status = "✓" if att['success'] else "✗"
            context += f"  {status} {att['action']}: {att['goal_type']}\n"
        return context


class GoalGenerator(nn.Module):
    """Generates autonomous goals based on knowledge state."""
    
    GOAL_TEMPLATES = {
        'explore': [
            "Explore {topic} to discover new information",
            "Find out what's known about {topic}",
            "Investigate the properties of {topic}",
        ],
        'explain': [
            "Explain how {topic} works",
            "Understand the mechanism behind {topic}",
            "Clarify the relationship between {topic} and related concepts",
        ],
        'create': [
            "Create a summary of {topic}",
            "Generate examples of {topic}",
            "Produce a diagram explaining {topic}",
        ],
        'resolve': [
            "Answer: What is {topic}?",
            "Resolve: How does {topic} relate to broader concepts?",
            "Determine the key facts about {topic}",
        ],
    }
    
    TOPIC_POOL = [
        "photosynthesis", "neural networks", "climate change",
        "quantum mechanics", "evolution", "machine learning",
        "black holes", "genetics", "renewable energy",
        "consciousness", "blockchain", "immunology",
    ]
    
    def __init__(self, model_dim: int = 576):  # SmolLM2-135M hidden size
        super().__init__()
        self.encoder = nn.Linear(model_dim, 128)
        self.goal_type_head = nn.Linear(128, 4)  # 4 goal types
        self.topic_head = nn.Linear(128, len(self.TOPIC_POOL))
        self.priority_head = nn.Linear(128, 1)
        
    def generate(self, knowledge_state: torch.Tensor, memory: GoalMemory) -> Goal:
        """Generate a new goal based on current state."""
        # Encode knowledge state
        encoded = F.relu(self.encoder(knowledge_state))
        
        # Sample goal type
        type_logits = self.goal_type_head(encoded)
        type_dist = Categorical(logits=type_logits)
        goal_type_idx = type_dist.sample().item()
        goal_types = ['explore', 'explain', 'create', 'resolve']
        goal_type = goal_types[goal_type_idx]
        
        # Sample topic
        topic_logits = self.topic_head(encoded)
        topic_dist = Categorical(logits=topic_logits)
        topic_idx = topic_dist.sample().item()
        topic = self.TOPIC_POOL[topic_idx]
        
        # Check if we already know about this topic
        known_topics = set()
        for s, p, o, c in memory.knowledge_graph.triples:
            if topic.lower() in s.lower() or topic.lower() in o.lower():
                known_topics.add(topic)
        
        # Adjust priority based on knowledge gaps
        priority = torch.sigmoid(self.priority_head(encoded)).item()
        if topic in known_topics:
            priority *= 0.7  # Lower priority for known topics
        
        # Create goal from template
        template = random.choice(self.GOAL_TEMPLATES[goal_type])
        description = template.format(topic=topic)
        
        goal = Goal(
            goal_type=goal_type,
            target=topic,
            description=description,
            priority=priority,
        )
        
        return goal


class GoalPlanner(nn.Module):
    """Plans actions to achieve goals."""
    
    ACTIONS = ['query', 'generate', 'reflect', 'synthesize']
    
    def __init__(self, mem_dim: int = 576, hidden_dim: int = 256):  # SmolLM2-135M hidden size
        super().__init__()
        
        # Encoder for goal + memory context
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=mem_dim, nhead=8, dim_feedforward=1024,
                batch_first=True, norm_first=False  # Changed for nested tensor compatibility
            ),
            num_layers=2
        )
        
        # Action selection
        self.action_head = nn.Linear(mem_dim, len(self.ACTIONS))
        
        # Steering selection (behavior to apply)
        self.steering_head = nn.Linear(mem_dim, 5)  # 5 steering types
        
        # Continuation decision
        self.continue_head = nn.Linear(mem_dim, 2)
        
        # Value head for PPO
        self.value_head = nn.Linear(mem_dim, 1)
        
    def forward(self, goal_embedding: torch.Tensor, memory_context: torch.Tensor):
        """Plan next action given goal and memory."""
        # Combine goal and memory
        combined = torch.cat([goal_embedding.unsqueeze(1), memory_context], dim=1)
        
        # Encode
        encoded = self.encoder(combined)
        pooled = encoded[:, 0, :]  # Use goal position
        
        # Output heads
        action_logits = self.action_head(pooled)
        steering_logits = self.steering_head(pooled)
        continue_logits = self.continue_head(pooled)
        value = self.value_head(pooled)
        
        return action_logits, steering_logits, continue_logits, value
    
    def select_action(self, goal_embedding: torch.Tensor, memory_context: torch.Tensor, 
                      temperature: float = 1.0) -> Tuple[str, str, bool, torch.Tensor]:
        """Select action with optional temperature scaling."""
        action_logits, steering_logits, continue_logits, value = self.forward(
            goal_embedding, memory_context
        )
        
        # Sample actions
        action_dist = Categorical(logits=action_logits / temperature)
        action_idx = action_dist.sample()
        action = self.ACTIONS[action_idx.item()]
        
        steering_dist = Categorical(logits=steering_logits)
        steering_idx = steering_dist.sample()
        steering_types = ['neutral', 'technical', 'casual', 'explanatory', 'creative']
        steering = steering_types[steering_idx.item()]
        
        continue_dist = Categorical(logits=continue_logits)
        should_continue = continue_dist.sample().item() == 0
        
        log_prob = action_dist.log_prob(action_idx) + steering_dist.log_prob(steering_idx)
        
        return action, steering, should_continue, log_prob


class SteeringLibrary:
    """Pre-computed steering vectors for different behaviors.
    
    These are trained by finding directions in activation space that
    increase the probability of specific style tokens.
    """
    
    STEERING_TYPES = ['neutral', 'technical', 'casual', 'explanatory', 'creative']
    
    # Style-specific prompt templates for training
    STYLE_PROMPTS = {
        'technical': [
            "The mechanism underlying this phenomenon involves",
            "Empirical evidence suggests that",
            "The theoretical framework posits",
            "Quantitative analysis reveals",
            "The biochemical pathway entails",
        ],
        'casual': [
            "So basically what happens is",
            "You know how like",
            "It's kind of like when",
            "The cool thing about this is",
            "Think of it this way",
        ],
        'explanatory': [
            "Let me break this down step by step",
            "Here's how it works",
            "The key concept to understand is",
            "This happens because",
            "In other words",
        ],
        'creative': [
            "Imagine a world where",
            "Picture this",
            "What if I told you",
            "Like a symphony of",
            "Dancing through the",
        ],
    }
    
    def __init__(self, model, tokenizer, model_dim: int = 576, phi_dim: int = 576):
        self.model = model
        self.tokenizer = tokenizer
        self.model_dim = model_dim
        self.device = next(model.parameters()).device
        
        # Compute steering vectors from style contrasts
        print("  Computing steering vectors from style contrasts...")
        self.vectors = self._compute_steering_vectors(model_dim)
        
    def _compute_steering_vectors(self, model_dim: int) -> Dict[str, torch.Tensor]:
        """Compute steering vectors by contrasting style activations."""
        vectors = {'neutral': torch.zeros(model_dim, device=self.device)}
        
        # Get base activation (neutral prompt)
        neutral_prompts = [
            "The answer is",
            "This is about",
            "What we know is",
        ]
        base_activation = self._get_mean_activation(neutral_prompts)
        
        # Compute style-specific vectors
        for style, prompts in self.STYLE_PROMPTS.items():
            # Get style activation
            style_activation = self._get_mean_activation(prompts)
            
            # Steering vector = style activation - neutral activation
            steering = style_activation - base_activation
            
            # Normalize to unit norm, scale by 0.5 for reasonable magnitude
            steering = steering / (steering.norm() + 1e-8) * 0.5
            
            vectors[style] = steering
            print(f"    {style}: norm={steering.norm():.3f}")
        
        return vectors
    
    def _get_mean_activation(self, prompts: List[str]) -> torch.Tensor:
        """Get mean hidden state activation for a set of prompts."""
        activations = []
        
        with torch.no_grad():
            for prompt in prompts:
                inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
                outputs = self.model(inputs, output_hidden_states=True)
                # Use mean of last hidden state
                hidden = outputs.hidden_states[-1].mean(dim=1).squeeze(0)
                activations.append(hidden)
        
        # Mean across prompts
        return torch.stack(activations).mean(dim=0)
        
    def get_vector(self, name: str, alpha: float = 0.5) -> torch.Tensor:
        """Get steering vector scaled by alpha."""
        base = self.vectors.get(name, self.vectors['neutral'])
        return base * alpha


class GoalReward:
    """Computes reward based on goal progress and knowledge gain."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.weights = {
            'goal_progress': self.config.get('goal_progress_weight', 1.0),
            'knowledge_gain': self.config.get('knowledge_gain_weight', 0.5),
            'coherence': self.config.get('coherence_weight', 0.3),
            'efficiency': self.config.get('efficiency_weight', 0.2),
        }
        self.previous_knowledge_size = 0
        
    def compute(self, goal: Goal, memory: GoalMemory, output: str, 
                coherence_loss: float, steps_taken: int) -> Tuple[float, Dict]:
        """Compute reward and breakdown."""
        rewards = {}
        
        # Goal progress
        rewards['progress'] = goal.progress * self.weights['goal_progress']
        
        # Knowledge gain
        current_knowledge = memory.knowledge_graph_size()
        knowledge_gain = max(0, current_knowledge - self.previous_knowledge_size)
        rewards['knowledge'] = min(knowledge_gain * 0.1, 1.0) * self.weights['knowledge_gain']
        self.previous_knowledge_size = current_knowledge
        
        # Coherence (negative loss)
        coherence = max(0, 1.0 - coherence_loss / 5.0)
        rewards['coherence'] = coherence * self.weights['coherence']
        
        # Efficiency (fewer steps = better)
        efficiency = max(0, 1.0 - steps_taken / 20.0)
        rewards['efficiency'] = efficiency * self.weights['efficiency']
        
        # Total
        total = sum(rewards.values())
        
        return total, rewards


class ALMAv5:
    """Main ALMA v5 Agent - Autonomous Goal-Driven Language Agent."""
    
    def __init__(self, model_id: str = "HuggingFaceTB/SmolLM2-135M"):
        print("🚀 Initializing ALMA v5...")
        
        # Load language model
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Try to use accelerate if available, otherwise load without device_map
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                dtype=torch.float32 if DEVICE == "cpu" else torch.float16,
                device_map=DEVICE if DEVICE != "cpu" else None
            )
        except ValueError:
            # accelerate not available, load without device_map
            print("⚠️  accelerate not available, loading without device_map...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=torch.float32
            )
            self.model.to(DEVICE)
            
        self.model.eval()
        
        # Core components
        self.goal_generator = GoalGenerator().to(DEVICE)
        self.planner = GoalPlanner().to(DEVICE)
        self.memory = GoalMemory()
        self.steering_library = SteeringLibrary(self.model, self.tokenizer, model_dim=576)
        self.reward_fn = GoalReward()
        
        # Training
        self.optimizer = torch.optim.Adam(self.planner.parameters(), lr=3e-4)
        
        # State
        self.current_goal: Optional[Goal] = None
        self.current_action: Optional[str] = None
        self.step = 0
        self.goals_completed = 0
        self.episode_log = []
        
        # PPO buffers
        self.log_probs_buffer = []
        self.rewards_buffer = []
        self.values_buffer = []
        
        print(f"✅ ALMA v5 ready on {DEVICE}")
        
    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text to embedding."""
        with torch.no_grad():
            inputs = self.tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=64)
            inputs = inputs.to(DEVICE)
            outputs = self.model(inputs, output_hidden_states=True)
            # Use last hidden state, mean pooled
            embedding = outputs.hidden_states[-1].mean(dim=1).squeeze(0)
            # Ensure float32 dtype for compatibility with planner/goal_generator
            return embedding.float()
    
    def generate_with_steering(self, prompt: str, steering_type: str = 'neutral',
                                alpha: float = 0.5, max_tokens: int = 50) -> Tuple[str, float]:
        """Generate text with actual activation steering."""
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        
        # Get steering vector
        steering_vector = self.steering_library.get_vector(steering_type, alpha)
        
        # Create hook for activation addition
        hook = self._add_steering_hook(steering_vector)
        
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
        finally:
            # Remove hook after generation
            hook.remove()
            
        generated = self.tokenizer.decode(outputs.sequences[0][inputs.shape[1]:], 
                                          skip_special_tokens=True)
        
        # Compute coherence (average log probability)
        scores = outputs.scores
        if scores:
            avg_log_prob = sum(s.max(dim=-1).values.log().mean() for s in scores) / len(scores)
            coherence_loss = -avg_log_prob.item()
        else:
            coherence_loss = 0.0
            
        return generated, coherence_loss
    
    def _add_steering_hook(self, steering_vector: torch.Tensor):
        """Add forward hook for activation addition at layer 12."""
        layer_idx = 12  # Middle layer for semantic steering
        
        def hook_fn(module, inputs, output):
            if isinstance(output, tuple):
                h = output[0]
                # Add steering vector to last token position
                h[:, -1, :] = h[:, -1, :] + steering_vector.to(h.dtype)
                return (h,) + output[1:]
            else:
                h = output
                h[:, -1, :] = h[:, -1, :] + steering_vector.to(h.dtype)
                return h
        
        handle = self.model.model.layers[layer_idx].register_forward_hook(hook_fn)
        return handle
    
    def execute_action(self, action: str, goal: Goal, steering: str) -> Tuple[str, bool]:
        """Execute an action and return output + success."""
        prompt = f"Q: {goal.description}\nA:"
        
        if action == 'query':
            prompt = f"Let me think about {goal.target} step by step.\n"
        elif action == 'generate':
            prompt = f"Explain: {goal.target}\n"
        elif action == 'reflect':
            prompt = f"Let me verify my understanding of {goal.target}.\n"
        elif action == 'synthesize':
            prompt = f"Summary of {goal.target}:\n"
            
        output, coherence = self.generate_with_steering(prompt, steering)
        
        # Extract knowledge from output (simplified)
        if len(output) > 20 and coherence < 3.0:
            # Add a fact to knowledge graph
            self.memory.add_knowledge(
                goal.target, "has_property", output[:50], 
                confidence=max(0.5, 1.0 - coherence / 5.0)
            )
            success = True
        else:
            success = False
            
        return output, success
    
    def update_goal_progress(self, goal: Goal, output: str, success: bool):
        """Update goal progress based on outcome."""
        if success:
            goal.progress = min(1.0, goal.progress + 0.25)
        else:
            goal.progress = max(0, goal.progress - 0.05)
            
        # Check completion
        if goal.progress >= 0.95:
            self.goals_completed += 1
            return True
        return False
    
    def run_step(self, ui: ALMAUI) -> Dict:
        """Run one step of the autonomy loop."""
        self.step += 1
        result = {'step': self.step}
        
        # Get knowledge state embedding
        knowledge_text = self.memory.knowledge_graph.to_text()
        if not knowledge_text:
            knowledge_text = "Starting fresh exploration."
        knowledge_state = self.encode_text(knowledge_text).to(DEVICE)
        
        # Generate or continue goal
        if self.current_goal is None or self.current_goal.progress >= 0.95:
            self.current_goal = self.goal_generator.generate(knowledge_state, self.memory)
            self.memory.add_goal(self.current_goal)
            ui.add_discovery(f"New goal: {self.current_goal.goal_type} '{self.current_goal.target}'")
            result['new_goal'] = True
        else:
            result['new_goal'] = False
            
        # Plan action
        memory_context = self.encode_text(self.memory.get_context()).to(DEVICE)
        memory_context = memory_context.unsqueeze(0).repeat(1, 1, 1)  # Match expected shape
        
        action, steering, should_continue, log_prob = self.planner.select_action(
            knowledge_state.unsqueeze(0), memory_context
        )
        self.current_action = action
        
        # Execute
        output, success = self.execute_action(action, self.current_goal, steering)
        
        # Log attempt
        self.memory.log_attempt(
            self.current_goal.goal_type, action, output, success
        )
        
        # Update progress
        completed = self.update_goal_progress(self.current_goal, output, success)
        if completed:
            ui.add_discovery(f"✅ Completed: {self.current_goal.target}")
            
        # Compute reward
        coherence_loss = 1.0  # Placeholder
        reward, reward_breakdown = self.reward_fn.compute(
            self.current_goal, self.memory, output, coherence_loss, 1
        )
        
        # Store for training
        self.log_probs_buffer.append(log_prob)
        self.rewards_buffer.append(reward)
        
        # PPO update every 8 steps
        if self.step % 8 == 0:
            self._ppo_update()
            
        # Update UI
        ui.update_state(
            step=self.step,
            current_goal=self.current_goal.to_dict(),
            current_action=f"{action} ({steering})",
            reward_history=self.rewards_buffer[-50:],
            goals_completed=self.goals_completed,
            knowledge_graph_size=self.memory.knowledge_graph_size(),
            agent_state="Goal Pursuit" if not completed else "Complete",
        )
        
        result.update({
            'action': action,
            'steering': steering,
            'output': output[:200],
            'success': success,
            'reward': reward,
            'reward_breakdown': reward_breakdown,
            'goal_progress': self.current_goal.progress,
        })
        
        return result
    
    def _ppo_update(self):
        """PPO policy update."""
        if len(self.log_probs_buffer) < 2:
            return
            
        self.optimizer.zero_grad()
        
        # Compute returns (simple Monte Carlo)
        returns = []
        R = 0
        for r in reversed(self.rewards_buffer[-8:]):
            R = r + 0.9 * R
            returns.insert(0, R)
            
        if not returns:
            return
            
        returns = torch.tensor(returns, device=DEVICE)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        log_probs = torch.stack(self.log_probs_buffer[-8:]).to(DEVICE)
        
        # Policy loss
        loss = -(log_probs * returns).mean()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.planner.parameters(), 1.0)
        self.optimizer.step()
        
        # Clear buffers
        self.log_probs_buffer.clear()
        self.rewards_buffer.clear()
        
    def save_checkpoint(self, path: str):
        """Save agent state."""
        checkpoint = {
            'planner': self.planner.state_dict(),
            'goal_generator': self.goal_generator.state_dict(),
            'memory_triples': self.memory.knowledge_graph.triples,
            'step': self.step,
            'goals_completed': self.goals_completed,
        }
        torch.save(checkpoint, path)
        print(f"💾 Checkpoint saved to {path}")
        
    def load_checkpoint(self, path: str):
        """Load agent state."""
        if not os.path.exists(path):
            print(f"⚠️  Checkpoint not found: {path}")
            return False
            
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=True)
        self.planner.load_state_dict(checkpoint['planner'])
        self.goal_generator.load_state_dict(checkpoint['goal_generator'])
        self.memory.knowledge_graph.triples = checkpoint.get('memory_triples', [])
        self.step = checkpoint.get('step', 0)
        self.goals_completed = checkpoint.get('goals_completed', 0)
        print(f"✅ Checkpoint loaded from {path}")
        return True


# ==============================================================================
# MAIN TRAINING LOOP WITH LIVE UI
# ==============================================================================

def run_training(steps: int = 200, checkpoint_path: str = "alma_checkpoint.pt"):
    """Run training with live UI updates."""
    
    console = Console()
    console.print("\n[bold teal]╔═══════════════════════════════════════════════════════════╗[/]")
    console.print("[bold teal]║[/]  [bold white]🚀 ALMA v5 — Autonomous Goal-Driven Language Agent[/]  [bold teal]║[/]")
    console.print("[bold teal]╚═══════════════════════════════════════════════════════════╝[/]\n")
    
    # Initialize
    ui = ALMAUI()
    
    with console.status("[teal]Loading model...[/]"):
        agent = ALMAv5(model_id="HuggingFaceTB/SmolLM2-135M")
        
    # Try to load checkpoint
    if os.path.exists(checkpoint_path):
        agent.load_checkpoint(checkpoint_path)
        
    # Create layout
    layout = ui.make_layout()
    log_entries = []
    recent_output = "Waiting for first generation..."
    
    console.print("\n[green]✅ Ready! Starting autonomy loop...[/]\n")
    console.print("[gray]Press Ctrl+C to stop and save checkpoint[/]\n")
    
    try:
        with Live(layout, console=console, refresh_per_second=4) as live:
            for step in range(steps):
                # Run step
                result = agent.run_step(ui)
                
                # Update log
                status_emoji = "✅" if result['success'] else "❌"
                log_entry = f"Step {result['step']:04d}: {result['action']} → {status_emoji} (R: {result['reward']:.3f})"
                log_entries.append(log_entry)
                
                # Update output
                recent_output = result.get('output', recent_output)
                
                # Update layout
                layout["header"].update(ui.render_header())
                layout["left"].split_column(
                    ui.render_goal_panel(),
                    ui.render_stats_panel(),
                )
                layout["center"].update(ui.render_output_panel(recent_output))
                layout["right"].split_column(
                    ui.render_memory_panel(agent.memory),
                    ui.render_discoveries_panel(),
                    ui.render_log_panel(log_entries),
                )
                layout["footer"].update(
                    Panel(
                        f"[gray]Reward breakdown: Progress={result['reward_breakdown'].get('progress', 0):.3f} | "
                        f"Knowledge={result['reward_breakdown'].get('knowledge', 0):.3f} | "
                        f"Coherence={result['reward_breakdown'].get('coherence', 0):.3f} | "
                        f"Efficiency={result['reward_breakdown'].get('efficiency', 0):.3f}[/]",
                        style="dim blue",
                        border_style="purple",
                    )
                )
                
                # Auto-save every 50 steps
                if step % 50 == 0 and step > 0:
                    agent.save_checkpoint(checkpoint_path)
                    
                live.update(layout)
                time.sleep(0.1)  # Small delay for visibility
                
    except KeyboardInterrupt:
        console.print("\n\n[yellow]⏸️  Training interrupted by user[/]")
        
    # Final save
    agent.save_checkpoint(checkpoint_path)
    
    # Summary
    console.print("\n[bold teal]═══════════════════════════════════════════════════════════[/]")
    console.print(f"[bold white]📊 Training Summary[/]")
    console.print(f"[teal]  Total steps:[/] {agent.step}")
    console.print(f"[teal]  Goals completed:[/] {agent.goals_completed}")
    console.print(f"[teal]  Knowledge facts:[/] {agent.memory.knowledge_graph_size()}")
    console.print(f"[teal]  Checkpoint saved to:[/] {checkpoint_path}")
    console.print("[bold teal]═══════════════════════════════════════════════════════════[/]\n")
    
    return agent


def demo_mode():
    """Run a quick demo without full training."""
    console = Console()
    
    console.print("\n[bold cyan]🎬 ALMA v5 Demo Mode[/]\n")
    console.print("This demonstrates the agent's capabilities without full training.\n")
    
    ui = ALMAUI()
    agent = ALMAv5()
    
    # Run 20 demo steps
    for i in range(20):
        result = agent.run_step(ui)
        
        console.print(f"\n[cyan]Step {result['step']:04d}[/]")
        console.print(f"  Goal: [{result.get('new_goal', False) and 'green' or 'bright_black'}]{agent.current_goal.description}[/]")
        console.print(f"  Action: {result['action']} (steering: {result['steering']})")
        console.print(f"  Output: {result['output'][:100]}...")
        console.print(f"  Reward: {result['reward']:.3f}")
        
        time.sleep(0.5)
        
    console.print(f"\n[green]✅ Demo complete! {agent.goals_completed} goals achieved.[/]")


def test_steering():
    """Test that steering vectors actually affect output style."""
    console = Console()
    
    console.print("\n[bold cyan]🧪 ALMA v5 Steering Validation Test[/]\n")
    console.print("Testing if steering vectors actually change output style...\n")
    
    agent = ALMAv5()
    
    test_prompt = "Explain how photosynthesis works"
    
    console.print(f"[cyan]Prompt:[/] {test_prompt}\n")
    
    results = {}
    
    for style in ['technical', 'casual', 'explanatory', 'creative']:
        console.print(f"  Generating with [yellow]{style}[/] steering...")
        output, coherence = agent.generate_with_steering(
            test_prompt, 
            steering_type=style, 
            alpha=0.5,
            max_tokens=40
        )
        results[style] = output
        console.print(f"  [grey]{output[:80]}...[/]\n")
    
    console.print("\n[bold green]✅ Steering Validation Complete[/]\n")
    console.print("[bold]Comparison:[/]\n")
    
    for style, output in results.items():
        console.print(f"[yellow]{style.upper()}:[/] {output}\n")
        console.print("-" * 80 + "\n")
    
    console.print("\n[cyan]Do the outputs show distinct styles?[/]")
    console.print("- [green]technical[/]: Should use formal, scientific language\n")
    console.print("- [green]casual[/]: Should use conversational, informal language\n")
    console.print("- [green]explanatory[/]: Should use clear, step-by-step explanations\n")
    console.print("- [green]creative[/]: Should use imaginative, metaphorical language\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ALMA v5 — Autonomous Goal-Driven Language Agent")
    parser.add_argument("--mode", choices=["train", "demo", "test_steering"], default="train",
                       help="Run mode: train (full), demo (quick), or test_steering (validation)")
    parser.add_argument("--steps", type=int, default=200,
                       help="Number of training steps")
    parser.add_argument("--checkpoint", type=str, default="alma_checkpoint.pt",
                       help="Checkpoint path")
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM2-135M",
                       help="Model ID to use")
    
    args = parser.parse_args()
    
    if args.mode == "demo":
        demo_mode()
    elif args.mode == "test_steering":
        test_steering()
    else:
        run_training(steps=args.steps, checkpoint_path=args.checkpoint)
