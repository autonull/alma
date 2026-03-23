#!/usr/bin/env python3
"""
ALMA v6 for GSM8K — Fast, Verbose Benchmark

Run: python alma_gsm8k.py --num 20
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import time

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================================================================
# GSM8K TEST QUESTIONS
# ==============================================================================

QUESTIONS = [
    {"q": "John has 12 apples. He gives 5 to Mary. How many apples does John have now?", "a": "7"},
    {"q": "A bakery sells cupcakes for $3 each. Sarah buys 4. How much change from $20?", "a": "8"},
    {"q": "There are 24 students divided into 4 equal groups. How many per group?", "a": "6"},
    {"q": "Mike runs 5 miles per day. How many miles in 2 weeks?", "a": "70"},
    {"q": "A book costs $15. Price increases 20%. What is the new price?", "a": "18"},
    {"q": "3 boxes, each with 8 red and 6 blue balls. Total balls?", "a": "42"},
    {"q": "Tom is 12. His sister is half his age. How old in 5 years?", "a": "11"},
    {"q": "A train travels 60 mph. How far in 2 hours 30 minutes?", "a": "150"},
    {"q": "5 workers build a wall in 8 hours. How long for 10 workers?", "a": "4"},
    {"q": "Pizza cut into 8 slices. 3 people eat 2 each. Slices left?", "a": "2"},
]

# ==============================================================================
# ALMA v6 — FAST VERSION
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
    
    def generate(self, prompt, max_tokens=100):
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        attention_mask = torch.ones_like(inputs)
        with torch.no_grad():
            out = self.model.generate(inputs, attention_mask=attention_mask, max_new_tokens=max_tokens, pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip()
    
    def baseline(self, question):
        """Single-pass baseline."""
        prompt = f"Solve: {question}\nAnswer (number only):"
        resp = self.generate(prompt, max_tokens=30)
        nums = re.findall(r'-?\d+\.?\d*', resp)
        return nums[-1] if nums else "0"
    
    def alma_solve(self, question):
        """ALMA v6 multi-step."""
        # Single consolidated reasoning prompt
        prompt = f"""{question}

Solve step by step:
1. Identify the numbers
2. Identify the operation
3. Calculate
4. Verify

Answer:"""
        resp = self.generate(prompt, max_tokens=80)
        nums = re.findall(r'-?\d+\.?\d*', resp)
        return nums[-1] if nums else "0"
    
    def check(self, pred, expected):
        try:
            return abs(float(pred) - float(expected)) < 0.01
        except:
            return pred.strip() == expected.strip()
    
    def benchmark(self, num=10, verbose=True):
        """Run benchmark with live feedback."""
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
            
            if verbose:
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
    parser.add_argument("--num", type=int, default=10, help="Number of questions")
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM2-360M")
    args = parser.parse_args()
    
    print("\n" + "═" * 70)
    print("  ALMA v6 — GSM8K Benchmark")
    print("═" * 70 + "\n")
    
    agent = ALMA(model_id=args.model)
    agent.benchmark(num=args.num, verbose=True)


if __name__ == "__main__":
    main()
