"""The markets of the persistent environment: a market is the contract and the two suppliers' cost types, drawn once
per seed and repeated at every event. Derived from the code only (orgsim.sequence.pre_draw_event_sequence and
orgsim.primitives.get_contract_scenarios); it reads no run.

1  every (contract, type of supplier A, type of supplier B) combination and its probability under the draw's rules:
   the contract uniform over the contracts, A's type 1 in 2, B's type opposite to A's with probability P_DIFFERENT.
2  the combinations that the plan draws for the seeds of the recorded runs (3 to 62), and, as a check on the
   probabilities, for seeds 0 to 9999, drawn as scripts/run_demo.py draws a plan.
Usage: python market_combinations.py [output file]
"""
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

import pilot17_analysis as pa
from orgsim.primitives import get_contract_scenarios
from orgsim.sequence import P_DIFFERENT, pre_draw_event_sequence

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else pa.RESULTS / "market_combinations.txt"
TYPES = ("speed", "quality")


def draw(seed):
    """The persistent plan's one market, as run_demo draws the plan: (contract name, type of A, type of B)."""
    plan = pre_draw_event_sequence(6, get_contract_scenarios(), seed=seed, environment="persistent",
                                   prompt_set="primitives")
    sc, type_a, type_b = plan[0]
    return sc.name, type_a, type_b


def main():
    scenarios = get_contract_scenarios()
    p_diff = Fraction(P_DIFFERENT)
    lines = []
    w = lines.append
    probs = {}
    for sc in scenarios:
        for ta in TYPES:
            for tb in TYPES:
                probs[(sc.name, ta, tb)] = (Fraction(1, len(scenarios)) * Fraction(1, len(TYPES))
                                            * (p_diff if ta != tb else 1 - p_diff))
    w(f"1. Combinations (contract, type of A, type of B) under the draw's rules: {len(probs)}; probabilities sum to "
      f"{sum(probs.values())}")
    for key, p in sorted(probs.items(), key=lambda kv: (-kv[1], kv[0])):
        w(f"   {key[0]:<17} A {key[1]:<8} B {key[2]:<8} probability {p}")
    w("   probability by value: " + ", ".join(f"{p} x {n}" for p, n in sorted(Counter(probs.values()).items(),
                                                                                   reverse=True)))
    recorded = Counter(draw(s) for s in range(3, 63))
    w(f"\n2. Combinations drawn for seeds 3 to 62 (the seeds of the recorded runs): {len(recorded)} distinct, "
      f"{sum(recorded.values())} seeds")
    large = Counter(draw(s) for s in range(10000))
    w(f"   Combinations drawn for seeds 0 to 9999: {len(large)} distinct; all among the {len(probs)} above: "
      f"{set(large) <= set(probs)}")
    for key in sorted(probs, key=lambda k: (-probs[k], k)):
        w(f"   {key[0]:<17} A {key[1]:<8} B {key[2]:<8} drawn {large[key] / 10000:.4f}  probability {float(probs[key]):.4f}")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
