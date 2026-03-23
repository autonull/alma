#!/usr/bin/env python3
"""
ALMA v6 — Simple Arithmetic Benchmark

Easier than GSM8K. Tests basic multi-step calculation.

Run: python alma_simple.py --num 20
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import time

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================================================================
# SIMPLE ARITHMETIC QUESTIONS (Easier than GSM8K)
# ==============================================================================

QUESTIONS = [
    {"q": "What is 5 + 3?", "a": "8"},
    {"q": "What is 10 - 4?", "a": "6"},
    {"q": "What is 6 × 7?", "a": "42"},
    {"q": "What is 100 ÷ 4?", "a": "25"},
    {"q": "What is 12 + 8 - 5?", "a": "15"},
    {"q": "What is 3 × 4 + 2?", "a": "14"},
    {"q": "What is 20 - 8 + 3?", "a": "15"},
    {"q": "What is 9 × 9?", "a": "81"},
    {"q": "What is 48 ÷ 6?", "a": "8"},
    {"q": "What is 7 + 8 + 9?", "a": "24"},
    {"q": "What is 15 - 7?", "a": "8"},
    {"q": "What is 11 × 2?", "a": "22"},
    {"q": "What is 30 ÷ 5?", "a": "6"},
    {"q": "What is 25 + 25?", "a": "50"},
    {"q": "What is 100 - 37?", "a": "63"},
    {"q": "What is 8 × 8?", "a": "64"},
    {"q": "What is 72 ÷ 8?", "a": "9"},
    {"q": "What is 13 + 17?", "a": "30"},
    {"q": "What is 50 - 23?", "a": "27"},
    {"q": "What is 4 × 5 × 2?", "a": "40"},
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
    
    def generate(self, prompt, max_tokens=50):
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        attention_mask = torch.ones_like(inputs)
        with torch.no_grad():
            out = self.model.generate(inputs, attention_mask=attention_mask, max_new_tokens=max_tokens, pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip()
    
    def baseline(self, question):
        """Single-pass baseline."""
        prompt = f"{question}\nAnswer:"
        resp = self.generate(prompt, max_tokens=20)
        nums = re.findall(r'-?\d+\.?\d*', resp)
        return nums[-1] if nums else "0"
    
    def alma_solve(self, question):
        """ALMA v6: Chain of Thought."""
        prompt = f"""{question}

Let me solve this step by step:
"""
        resp = self.generate(prompt, max_tokens=60)
        nums = re.findall(r'-?\d+\.?\d*', resp)
        return nums[-1] if nums else "0"
    
    def check(self, pred, expected):
        try:
            return abs(float(pred) - float(expected)) < 0.01
        except:
            return pred.strip() == expected.strip()
    
    def benchmark(self, num=20):
        """Run benchmark."""
        qs = QUESTIONS[:num]
        base_correct = 0
        alma_correct = 0
        
        print("=" * 70)
        print(f"{'Q':<3} {'Expected':<10} {'Baseline':<10} {'ALMA':<10} {'Base OK':<8} {'ALMA OK':<8}")
        print("=" * 70)
        
        total_start = time.time()
        
        for i, item in enumerate(qs):
            q, a = item['q'], item['a']
            start = time.time()
            
            # Baseline
            base_ans = self.baseline(q)
            base_ok = self.check(base_ans, a)
            if base_ok: base_correct += 1
            
            # ALMA
            alma_ans = self.alma_solve(q)
            alma_ok = self.check(alma_ans, a)
            if alma_ok: alma_correct += 1
            
            elapsed = time.time() - start
            
            status_base = "✓" if base_ok else "✗"
            status_alma = "✓" if alma_ok else "✗"
            
            print(f"{i+1:<3} {a:<10} {base_ans:<10} {alma_ans:<10} {status_base:<8} {status_alma:<8} ({elapsed:.1f}s)")
        
        total_time = time.time() - total_start
        
        # Summary
        print("=" * 70)
        print(f"BASELINE: {base_correct}/{num} = {base_correct/num*100:.1f}%")
        print(f"ALMA v6:  {alma_correct}/{num} = {alma_correct/num*100:.1f}%")
        print(f"IMPROVEMENT: +{(alma_correct - base_correct)/num*100:.1f} pp")
        if base_correct > 0:
            print(f"RELATIVE GAIN: +{(alma_correct - base_correct)/base_correct*100:.1f}%")
        print(f"TIME: {total_time:.1f}s ({total_time/num:.1f}s/question)")
        print("=" * 70)
        
        return {
            'baseline': base_correct/num,
            'alma': alma_correct/num,
            'gain': (alma_correct - base_correct)/num
        }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=20)
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM2-360M")
    args = parser.parse_args()
    
    print("\n" + "═" * 70)
    print("  ALMA v6 — Simple Arithmetic Benchmark")
    print("═" * 70 + "\n")
    
    agent = ALMA(model_id=args.model)
    agent.benchmark(num=args.num)


if __name__ == "__main__":
    main()
