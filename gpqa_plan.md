# ALMA for GPQA — Making Small Models Punch Above Their Weight

**The challenge:** GPQA (Graduate-Level Google-Proof Q&A) stumps most models. SmolLM2-360M gets ~25% accuracy. Can ALMA v6 push this to 40-50%?

**The hypothesis:** Multi-step autonomous reasoning beats single-pass generation for hard questions.

---

## Why GPQA Is Hard

**Example GPQA question:**
```
In a particular species of octopus, the optic gland has been observed to secrete 
steroid hormones that trigger senescence. Which of the following best describes 
the evolutionary advantage of this mechanism?

A) It prevents overpopulation by ensuring individuals die after reproduction
B) It reallocates resources from somatic maintenance to reproductive effort
C) It reduces predation pressure by synchronizing death with vulnerable life stages
D) It eliminates competition between generations for limited resources
```

**Why single-pass fails:**
1. Requires multi-hop reasoning (optic gland → steroid hormones → senescence → evolutionary theory)
2. Needs to eliminate distractors (all answers sound plausible)
3. Requires integrating concepts across biology domains
4. Can't be solved by pattern matching alone

**SmolLM2-360M single-pass:** ~25% (basically random)

---

## How ALMA v6 Can Help

### The Key Insight

**Single-pass:** Model must get everything right in one forward pass.

**Multi-step ALMA:** Model can:
1. Break question into sub-problems
2. Solve each sub-problem independently
3. Synthesize answers
4. Verify and revise

**This is the advantage:** Decomposition + verification = better accuracy.

---

## ALMA v6 for GPQA — Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GPQA SOLVER                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  INPUT: "In a particular species of octopus..."              │
│                                                              │
│  STEP 1: Question Analysis                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Concepts: [optic gland, steroid hormones, senescence] │   │
│  │ Domain: evolutionary biology                          │   │
│  │ Task: identify evolutionary advantage                 │   │
│  └──────────────────────────────────────────────────────┘   │
│                              │                               │
│                              ▼                               │
│  STEP 2: Sub-question Decomposition                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Q1: What function does the optic gland serve?         │   │
│  │ Q2: How do steroid hormones affect senescence?        │   │
│  │ Q3: What evolutionary theories explain programmed     │   │
│  │     death after reproduction?                         │   │
│  └──────────────────────────────────────────────────────┘   │
│                              │                               │
│                              ▼                               │
│  STEP 3: Sequential Reasoning (per sub-question)             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ For each sub-question:                                │   │
│  │ 1. Generate answer with steering (precision)          │   │
│  │ 2. Extract key claims                                 │   │
│  │ 3. Check consistency with other answers               │   │
│  │ 4. Store in working memory                            │   │
│  └──────────────────────────────────────────────────────┘   │
│                              │                               │
│                              ▼                               │
│  STEP 4: Synthesis                                           │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Combine sub-answers:                                  │   │
│  │ "The optic gland secretes steroids that trigger       │   │
│  │  senescence. This reallocates resources from          │   │
│  │  somatic maintenance to reproduction, which is        │   │
│  │  evolutionarily advantageous because..."              │   │
│  └──────────────────────────────────────────────────────┘   │
│                              │                               │
│                              ▼                               │
│  STEP 5: Verification                                        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Check each answer choice:                             │   │
│  │ A) "prevents overpopulation" - group selection?       │   │
│  │    Unlikely mechanism. LOW CONFIDENCE.                │   │
│  │ B) "reallocates resources" - matches life history     │   │
│  │    theory. HIGH CONFIDENCE. ✓                         │   │
│  │ C) "reduces predation" - no evidence in question.     │   │
│  │    LOW CONFIDENCE.                                    │   │
│  │ D) "eliminates competition" - possible but less       │   │
│  │    direct than B. MEDIUM CONFIDENCE.                  │   │
│  └──────────────────────────────────────────────────────┘   │
│                              │                               │
│                              ▼                               │
│  OUTPUT: B (confidence: 0.82)                                │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## Why This Beats Single-Pass

| Capability | Single-Pass | ALMA v6 |
|------------|-------------|---------|
| **Decomposition** | Must solve all at once | Break into tractable pieces |
| **Working memory** | Limited to context | Explicit storage of sub-answers |
| **Verification** | None (commits to first answer) | Checks each option systematically |
| **Revision** | Can't revise | Can update based on verification |
| **Steering** | Fixed style | Precision steering for reasoning |

**Expected improvement:** 25% → 40-50% (60-100% relative gain)

---

## ALMA v6 Components for GPQA

### 1. Question Analyzer

```python
class QuestionAnalyzer:
    def analyze(self, question):
        # Extract concepts
        concepts = self.extract_concepts(question)
        
        # Identify domain
        domain = self.classify_domain(concepts)
        
        # Identify task type
        task = self.classify_task(question)
        
        return {
            'concepts': concepts,
            'domain': domain,
            'task': task,
            'difficulty': self.estimate_difficulty(concepts)
        }
    
    def extract_concepts(self, question):
        # Use model to extract key terms
        prompt = f"Extract key scientific concepts from: {question}"
        concepts = self.model.generate(prompt)
        return parse_concepts(concepts)
```

---

### 2. Decomposition Planner

```python
class DecompositionPlanner(nn.Module):
    def __init__(self):
        super().__init__()
        # Takes question embedding, outputs sub-question plan
        self.encoder = nn.Linear(576, 256)
        self.num_questions_head = nn.Linear(256, 5)  # 1-5 sub-questions
        self.question_generator = nn.Linear(256, 576)
        
    def plan(self, question_embedding):
        encoded = F.relu(self.encoder(question_embedding))
        
        # Decide how many sub-questions needed
        num_dist = Categorical(logits=self.num_questions_head(encoded))
        num_questions = num_dist.sample() + 1
        
        # Generate sub-question embeddings
        sub_q_embeddings = []
        for i in range(num_questions):
            q_emb = self.question_generator(encoded + i)
            sub_q_embeddings.append(q_emb)
        
        return sub_q_embeddings
```

**Training:** Learn to decompose questions that require multi-hop reasoning.

---

### 3. Reasoning Executor

```python
class ReasoningExecutor:
    def __init__(self, model, steering_library):
        self.model = model
        self.steering = steering_library
    
    def execute(self, sub_question, context=None):
        # Use precision steering for reasoning
        steering_vector = self.steering.get_vector('precision', alpha=0.7)
        
        prompt = f"""
        Context: {context}
        
        Question: {sub_question}
        
        Think step by step. Provide your reasoning before giving an answer.
        """
        
        output = self.model.generate(
            prompt, 
            steering=steering_vector,
            max_tokens=150
        )
        
        # Extract claims from output
        claims = self.extract_claims(output)
        
        return {
            'answer': output,
            'claims': claims,
            'confidence': self.estimate_confidence(output)
        }
```

---

### 4. Synthesis Module

```python
class Synthesizer:
    def synthesize(self, sub_answers, original_question):
        # Combine sub-answers into coherent reasoning
        context = "\n".join([a['answer'] for a in sub_answers])
        
        prompt = f"""
        Based on this reasoning:
        {context}
        
        Answer the original question: {original_question}
        
        Provide your final answer with justification.
        """
        
        final_answer = self.model.generate(prompt, max_tokens=100)
        
        return {
            'answer': final_answer,
            'reasoning_chain': sub_answers,
            'context': context
        }
```

---

### 5. Verification Module

```python
class Verifier:
    def verify(self, synthesis, answer_choices):
        scores = {}
        
        for choice in answer_choices:
            # Check if synthesis supports this choice
            prompt = f"""
            Reasoning: {synthesis['reasoning_chain']}
            
            Does this reasoning support the following answer?
            {choice}
            
            Rate support from 0.0 (contradicts) to 1.0 (strongly supports).
            """
            
            score = self.model.generate(prompt, max_tokens=10)
            scores[choice] = parse_score(score)
        
        # Select highest scoring choice
        best_choice = max(scores, key=scores.get)
        
        return {
            'selected': best_choice,
            'confidence': scores[best_choice],
            'all_scores': scores
        }
```

---

## Training Strategy

### Phase 1: Pre-train Components (Offline)

**Steering vectors:**
- Train 'precision' steering on step-by-step reasoning data
- Train 'verification' steering on fact-checking data

**Decomposition:**
- Train on datasets with multi-hop questions (HotpotQA, StrategyQA)
- Learn to break questions into tractable sub-problems

**Time:** ~4 hours

---

### Phase 2: Train Planner on GPQA-like Questions

**Dataset:** 
- GPQA training set (if available)
- Or similar science QA (MMLU science sections)

**Reward:**
```python
def compute_reward(question, prediction, correct_answer):
    # Correctness (binary)
    correctness = 1.0 if prediction == correct_answer else 0.0
    
    # Reasoning quality (learned evaluator)
    reasoning_quality = evaluator(prediction.reasoning_chain)
    
    # Efficiency (fewer steps = better, but not too few)
    optimal_steps = 3
    efficiency = 1.0 - abs(len(prediction.steps) - optimal_steps) / optimal_steps
    
    return (
        0.6 * correctness +
        0.3 * reasoning_quality +
        0.1 * efficiency
    )
```

**Training loop:**
```python
for question in gpqa_questions:
    prediction = agent.solve(question)
    reward = compute_reward(question, prediction, correct_answer)
    agent.planner.update(reward)
```

**Time:** ~30 minutes for 500 questions

---

### Phase 3: Evaluate on Held-Out GPQA

```python
def evaluate_gpqa(agent, test_questions):
    correct = 0
    
    for q in test_questions:
        prediction = agent.solve(q)
        if prediction.selected == q.correct_answer:
            correct += 1
    
    accuracy = correct / len(test_questions)
    return accuracy
```

**Target:** 40-50% (vs. 25% baseline)

---

## Expected Results

| Model | Single-Pass | ALMA v6 | Improvement |
|-------|-------------|---------|-------------|
| SmolLM2-135M | ~20% | ~35% | +75% |
| SmolLM2-360M | ~25% | ~45% | +80% |
| SmolLM2-1.7B | ~35% | ~55% | +57% |

**Why it works:**
- Decomposition makes hard problems tractable
- Verification catches errors
- Steering improves reasoning quality
- Working memory enables multi-hop inference

---

## What Makes This Different from Chain-of-Thought

| Aspect | CoT | ALMA v6 |
|--------|-----|---------|
| **Structure** | Free-form reasoning | Structured decomposition |
| **Verification** | None (commits to first answer) | Explicit verification step |
| **Learning** | Fixed prompt | Learns to decompose better |
| **Steering** | None | Precision steering for reasoning |
| **Memory** | Context window only | Explicit working memory |

**ALMA advantage:** Not just "think step by step" — structured, verified, learned reasoning.

---

## Implementation Plan

### Week 1: Core GPQA Solver
- Question analyzer
- Decomposition planner
- Reasoning executor
- Synthesis + verification

**Lines of code:** ~400

### Week 2: Training + Evaluation
- Pre-train steering vectors
- Train decomposition on HotpotQA
- Evaluate on GPQA subset

**Deliverable:** Baseline vs. ALMA comparison

### Week 3: Optimization
- Tune reward function
- Improve verification
- Add ablation studies

**Deliverable:** Final results, paper draft

---

## Why This Is Exciting

**Not:** "Look, it generates text autonomously!"

**But:** "Look, a 135M model can solve graduate-level science questions!"

**The insight:** Autonomy isn't the goal. Autonomy is the *means* to make small models punch above their weight.

**GPQA is the proof:** If ALMA v6 can push SmolLM2-360M from 25% to 45% on GPQA, that's real evidence that autonomous multi-step reasoning adds value.

---

## Bottom Line

**ALMA v6 for GPQA:**
- Decomposes hard questions into tractable sub-problems
- Solves each with precision steering
- Verifies answers before committing
- Learns to reason better over time

**Expected:** 25% → 45% on GPQA (SmolLM2-360M)

**If it works:** Proof that autonomy helps small models punch above their weight.

**If it fails:** Autonomy doesn't help for reasoning tasks. Kill the project.

Let's build it.
