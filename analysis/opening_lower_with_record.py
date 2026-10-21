"""In how many seeds does a single agent with a record open lower than one without? (Appendix B.1.)

Data: the persistent-environment mixed markets `single_norecord:single` of the pilot and the follow-up
(data/runs/pilot17 and data/runs/followup17, seeds 3 to 22), read with pilot17_analysis.load. X is the firm without a
record (`single_norecord`), U the firm with one (`single`).
Per seed, the statistic is the round-1 markup of U minus that of X (percentage points), pooled over both positions and
events 2 to 6, as in section K of followup17_pooled_results.txt (whose means are the pooled values of the same
quantities). A seed counts when the difference is below zero. The count is given for each intermediary, and for the
average of the two intermediaries (the per-seed value used elsewhere).
Usage: python opening_lower_with_record.py [output file]
"""
import sys
from pathlib import Path

import numpy as np

import pilot17_analysis as pa

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else pa.RESULTS / "opening_lower_with_record.txt"
ROOTS = [pa.RUNS / "pilot17", pa.RUNS / "followup17"]


def main():
    runs, problems = pa.load(ROOTS)
    seeds = sorted({k[3] for k in runs if k[0] == "persistent"})
    lines = [f"Opening markup with a record (U, `single`) minus without (X, `single_norecord`), mixed markets, persistent "
             f"environment, events 2-6, both positions pooled. Roots: {', '.join(pa.rel(r) for r in ROOTS)}",
             f"runs read {len(runs)}; integrity problems {len(problems)}; seeds {seeds[0]}-{seeds[-1]} ({len(seeds)})"]
    per_inst = {}
    for inst in pa.INSTS:
        gaps = []
        for s in seeds:
            u, x = [], []
            for pos, xs in (("xa", "A"), ("xb", "B")):
                r = runs.get(("persistent", f"single_norecord_vs_single_{pos}", inst, s))
                if not r:
                    continue
                us = "B" if xs == "A" else "A"
                for k in pa.LATE:
                    ev = r[2].get(k)
                    if ev:
                        x.append(pa.markup_r1(ev, xs))
                        u.append(pa.markup_r1(ev, us))
            gaps.append(pa.mean(u) - pa.mean(x))
        per_inst[inst] = gaps
        lines.append(f"  {inst:<12} U - X round-1 markup per seed [{', '.join(f'{g:+.2f}' for g in gaps)}]")
        lines.append(f"  {inst:<12} mean {np.mean(gaps):+.2f}  U opens lower in {sum(g < 0 for g in gaps)} of {len(gaps)} seeds")
    both = [(a + b) / 2 for a, b in zip(*per_inst.values())]
    lines.append(f"  both intermediaries averaged, per seed [{', '.join(f'{g:+.2f}' for g in both)}]")
    lines.append(f"  both intermediaries averaged: mean {np.mean(both):+.2f}  U opens lower in "
                 f"{sum(g < 0 for g in both)} of {len(both)} seeds")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
