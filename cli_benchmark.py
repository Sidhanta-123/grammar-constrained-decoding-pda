import time
import numpy as np
from src.automata.math_pda import MathPDAEngine
from src.automata.json_pda import JSONPDAEngine
from src.automata.sql_pda import SQLPDAEngine
from src.decoder.llm_simulator import LLMSimulator
from src.decoder.logit_masker import LogitMasker

def run_constrained_episode(engine, max_tokens=25, bias_invalid=True) -> bool:
    engine.reset()
    steps = 0
    while not (engine.is_accepted() and steps >= 5) and steps < max_tokens:
        tokens_left = max_tokens - steps
        valid_tokens = engine.get_budget_safe_lookahead(tokens_left)
        if not valid_tokens:
            break

        raw_logits = LLMSimulator.generate_logits(engine.Sigma, valid_tokens, bias_invalid)
        result = LogitMasker.apply_pda_mask(engine.Sigma, raw_logits, valid_tokens)
        engine.transition(result["constrained_pick"])
        steps += 1

    return engine.is_accepted()

def run_unconstrained_episode(engine, max_tokens=25, bias_invalid=True) -> bool:
    engine.reset()
    steps = 0
    while not engine.is_accepted() and steps < max_tokens:
        valid_tokens = engine.get_valid_lookahead()
        if not valid_tokens:
            break
        raw_logits = LLMSimulator.generate_logits(engine.Sigma, valid_tokens, bias_invalid)
        chosen_token = engine.Sigma[int(np.argmax(raw_logits))]
        if not engine.transition(chosen_token):
            return False
        steps += 1
    return engine.is_accepted()

def benchmark_all_grammars(iterations=500, max_tokens=25):
    engines = [MathPDAEngine(), JSONPDAEngine(), SQLPDAEngine()]
    print("=" * 78)
    print(f"🚀 MULTI-GRAMMAR BENCHMARK SUITE ({iterations} runs/grammar, max_tokens={max_tokens})")
    print("=" * 78)
    print(f"{'Grammar Engine':<26} | {'Unconstrained':<15} | {'PDA-Constrained':<15} | {'Latency/tok'}")
    print("-" * 78)

    for eng in engines:
        u_wins = sum(run_unconstrained_episode(eng, max_tokens) for _ in range(iterations))

        t0 = time.time()
        c_wins = sum(run_constrained_episode(eng, max_tokens) for _ in range(iterations))
        elapsed = time.time() - t0

        u_pct = (u_wins / iterations) * 100
        c_pct = (c_wins / iterations) * 100
        ms_per_tok = (elapsed / (iterations * (max_tokens / 2))) * 1000

        print(f"{eng.name:<26} | {u_pct:>13.1f}% | {c_pct:>14.1f}% | {ms_per_tok:>8.3f} ms")
    print("=" * 78)

if __name__ == "__main__":
    np.random.seed(int(time.time()))
    benchmark_all_grammars(iterations=500, max_tokens=25)