"""The efficient surplus per event behind the proposal's "efficient: 84.3" (Table 1 and Appendix B.2), read from the runs.

`outcome.efficient_surplus` in events.jsonl is the largest joint surplus (the buyer's value minus the winner's cost) that
the frontier of the contract and the suppliers' cost types allows (orgsim.primitives). The contracts and cost types
are the plan of the seed, which does not depend on the firms' organizations, so every homogeneous cell of a seed
faces the same efficient surplus.

Data: data/runs/departures18, homogeneous cells, seeds 23-42, events 2-6, both institutions. Per seed, the mean over
events 2-6 of each institution, then the mean of the two institutions (the pooling of departures18_analysis.py);
then the mean over the 20 seeds. The joint surplus of the same cells is printed beside it: it is the level that
departures18_analysis.py prints under LEVELS (joint), and its gap to the efficient surplus is the loss in joint surplus.
Usage: python efficient_surplus.py [output file]
"""
import sys
from pathlib import Path

import numpy as np

import departures18_analysis as da
import pilot17_analysis as pa

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else pa.RESULTS / "efficient_surplus.txt"
CELLS = ("forward", "forward_revenue", "brief", "brief_revenue")


def seed_value(runs, label, seed, fn):
    """The mean over the two institutions of the mean over events 2-6 of fn(event record)."""
    vals = []
    for inst in da.INSTS:
        run = da.homog(runs, label, inst, seed)
        vals.append(np.mean([fn(run[2][k]) for k in da.LATE]))
    return float(np.mean(vals))


def main():
    runs, problems = pa.load(da.DEP_ROOTS)
    lines = [f"Efficient surplus per event, data/runs/departures18: runs {len(runs)}; problems {len(problems)}; "
             f"seeds {da.SEEDS[0]}-{da.SEEDS[-1]}, events 2-6, both institutions pooled"]
    w = lines.append
    for label in CELLS:
        eff = [seed_value(runs, label, s, lambda ev: ev["outcome"]["efficient_surplus"]) for s in da.SEEDS]
        joint = [seed_value(runs, label, s, lambda ev: ev["outcome"]["joint_surplus"]) for s in da.SEEDS]
        w(f"  {label:<16} efficient surplus {np.mean(eff):.3f}  joint surplus {np.mean(joint):.3f}  "
          f"loss {np.mean(eff) - np.mean(joint):.3f}")
    forward = [seed_value(runs, "forward", s, lambda ev: ev["outcome"]["efficient_surplus"]) for s in da.SEEDS]
    revenue = [seed_value(runs, "forward_revenue", s, lambda ev: ev["outcome"]["efficient_surplus"])
               for s in da.SEEDS]
    w(f"  forward and forward_revenue face the same efficient surplus in every seed: "
      f"{bool(np.allclose(forward, revenue, atol=1e-9, rtol=0))}")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
