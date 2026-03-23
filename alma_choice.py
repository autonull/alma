#!/usr/bin/env python3
"""
ALMA v6 — Multiple Choice Benchmark

Even simpler: just pick A, B, C, or D.

Run: python alma_choice.py --num 30
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import time

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================================================================
# MULTIPLE CHOICE QUESTIONS
# ==============================================================================

QUESTIONS = [
    {"q": "What color is the sky on a clear day?", "options": "A) Green B) Blue C) Red D) Purple", "a": "B"},
    {"q": "How many legs does a dog have?", "options": "A) 2 B) 3 C) 4 D) 5", "a": "C"},
    {"q": "What is the opposite of hot?", "options": "A) Warm B) Cold C) Cool D) Freezing", "a": "B"},
    {"q": "Which animal says 'meow'?", "options": "A) Dog B) Cat C) Cow D) Bird", "a": "B"},
    {"q": "What do you drink from a cup?", "options": "A) Food B) Water C) Air D) Sand", "a": "B"},
    {"q": "How many fingers on one hand?", "options": "A) 4 B) 5 C) 6 D) 10", "a": "B"},
    {"q": "What comes after Monday?", "options": "A) Sunday B) Tuesday C) Wednesday D) Friday", "a": "B"},
    {"q": "Which is a fruit?", "options": "A) Carrot B) Apple C) Potato D) Lettuce", "a": "B"},
    {"q": "What color is grass?", "options": "A) Blue B) Yellow C) Green D) Red", "a": "C"},
    {"q": "Which is bigger?", "options": "A) Mouse B) Elephant C) Cat D) Bird", "a": "B"},
    {"q": "What do fish live in?", "options": "A) Trees B) Water C) Sand D) Air", "a": "B"},
    {"q": "How many eyes do humans have?", "options": "A) 1 B) 2 C) 3 D) 4", "a": "B"},
    {"q": "What do bees make?", "options": "A) Milk B) Honey C) Eggs D) Wool", "a": "B"},
    {"q": "Which season is coldest?", "options": "A) Summer B) Fall C) Winter D) Spring", "a": "C"},
    {"q": "What shape is a ball?", "options": "A) Square B) Triangle C) Round D) Flat", "a": "C"},
    {"q": "Which flies?", "options": "A) Fish B) Bird C) Snake D) Frog", "a": "B"},
    {"q": "What color is snow?", "options": "A) Black B) Blue C) White D) Gray", "a": "C"},
    {"q": "How many wheels on a bike?", "options": "A) 1 B) 2 C) 3 D) 4", "a": "B"},
    {"q": "Which swims?", "options": "A) Cat B) Fish C) Dog D) Bird", "a": "B"},
    {"q": "What do you sleep on?", "options": "A) Floor B) Bed C) Table D) Chair", "a": "B"},
    {"q": "Which is fastest?", "options": "A) Turtle B) Car C) Snail D) Worm", "a": "B"},
    {"q": "What color is the sun?", "options": "A) Yellow B) Blue C) Green D) Purple", "a": "A"},
    {"q": "How many days in a week?", "options": "A) 5 B) 6 C) 7 D) 10", "a": "C"},
    {"q": "Which barks?", "options": "A) Cat B) Dog C) Bird D) Fish", "a": "B"},
    {"q": "What do trees grow on?", "options": "A) Water B) Ground C) Air D) Rocks", "a": "B"},
    {"q": "Which is sweet?", "options": "A) Salt B) Sugar C) Lemon D) Vinegar", "a": "B"},
    {"q": "What color is chocolate?", "options": "A) White B) Brown C) Green D) Blue", "a": "B"},
    {"q": "Which hops?", "options": "A) Fish B) Rabbit C) Snake D) Bird", "a": "B"},
    {"q": "What do birds have?", "options": "A) Fins B) Wings C) Hooves D) Paws", "a": "B"},
    {"q": "Which is tallest?", "options": "A) Grass B) Tree C) Bush D) Flower", "a": "B"},
]

# ==============================================================================
# ALMA v6
# ==============================================================================

class ALMA:
    def __init__(self, model_id="HuggingFaceTB/SmolLM2-360M"):
        print(f"\n🔧 Loading {model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).to(DEVICE)
        self.model.eval()
        print(f"✓ Ready on {DEVICE}\n")
    
    def generate(self, prompt, max_tokens=30):
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        attention_mask = torch.ones_like(inputs)
        with torch.no_grad():
            out = self.model.generate(inputs, attention_mask=attention_mask, max_new_tokens=max_tokens, pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip()
    
    def baseline(self, q, options):
        """Single-pass baseline."""
        prompt = f"{q}\n{options}\nAnswer:"
        resp = self.generate(prompt, max_tokens=10)
        match = re.search(r'\b([ABCD])\b', resp.upper())
        return match.group(1) if match else "X"
    
    def alma_solve(self, q, options):
        """ALMA v6: Eliminate wrong answers first."""
        prompt = f"""{q}
{options}

Which options are definitely wrong? Eliminate them, then pick the best answer.
Final answer:"""
        resp = self.generate(prompt, max_tokens=50)
        # Find last ABCD in response (the final answer)
        matches = re.findall(r'\b([ABCD])\b', resp.upper())
        return matches[-1] if matches else "X"
    
    def check(self, pred, expected):
        return pred.upper() == expected.upper()
    
    def benchmark(self, num=30):
        """Run benchmark."""
        qs = QUESTIONS[:num]
        base_correct = 0
        alma_correct = 0
        
        print("=" * 75)
        print(f"{'Q':<3} {'Expected':<10} {'Baseline':<10} {'ALMA':<10} {'Base OK':<8} {'ALMA OK':<8}")
        print("=" * 75)
        
        total_start = time.time()
        
        for i, item in enumerate(qs):
            q, opts, a = item['q'], item['options'], item['a']
            start = time.time()
            
            # Baseline
            base_ans = self.baseline(q, opts)
            base_ok = self.check(base_ans, a)
            if base_ok: base_correct += 1
            
            # ALMA
            alma_ans = self.alma_solve(q, opts)
            alma_ok = self.check(alma_ans, a)
            if alma_ok: alma_correct += 1
            
            elapsed = time.time() - start
            
            status_base = "✓" if base_ok else "✗"
            status_alma = "✓" if alma_ok else "✗"
            
            print(f"{i+1:<3} {a:<10} {base_ans:<10} {alma_ans:<10} {status_base:<8} {status_alma:<8} ({elapsed:.2f}s)")
        
        total_time = time.time() - total_start
        
        # Summary
        print("=" * 75)
        print(f"BASELINE: {base_correct}/{num} = {base_correct/num*100:.1f}%")
        print(f"ALMA v6:  {alma_correct}/{num} = {alma_correct/num*100:.1f}%")
        print(f"IMPROVEMENT: +{(alma_correct - base_correct)/num*100:.1f} pp")
        if base_correct > 0:
            print(f"RELATIVE GAIN: +{(alma_correct - base_correct)/base_correct*100:.1f}%")
        print(f"TIME: {total_time:.1f}s ({total_time/num:.2f}s/question)")
        print("=" * 75)
        
        return {
            'baseline': base_correct/num,
            'alma': alma_correct/num,
            'gain': (alma_correct - base_correct)/num
        }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=30)
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM2-360M")
    args = parser.parse_args()
    
    print("\n" + "═" * 75)
    print("  ALMA v6 — Multiple Choice Benchmark")
    print("═" * 75 + "\n")
    
    agent = ALMA(model_id=args.model)
    agent.benchmark(num=args.num)


if __name__ == "__main__":
    main()
