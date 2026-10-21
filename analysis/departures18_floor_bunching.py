"""EXPLORATORY (after the results): how often bids sit exactly at the floor, homogeneous floor and floor_revenue cells.

Events 2-6. A bid is "under a positive floor" when its supplier's floor_status is "set" with minimum_profit > 0 for the
event; it is "at the floor" when its price is within half a cent of its own cost at its terms plus the minimum.
Returned and raised bids are counted from the per-bid floor audits over all bids under any set floor ($0 included).
Usage: python departures18_floor_bunching.py [output file]
"""
import sys
from pathlib import Path

import pilot17_analysis as pa
from orgsim.primitives import ContractTerms

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else pa.RESULTS / "departures18_floor_bunching.txt"


def main():
    runs, problems = pa.load([pa.RUNS / "departures18"])
    n_pos = at_pos = n_fin = at_fin = n_set = ret = rai = nruns = 0
    for (env, lab, inst, s), (run, rows, evs) in runs.items():
        if lab not in ("floor", "floor_revenue"):
            continue
        nruns += 1
        for k in range(2, 7):
            ev = evs[k]
            for sid in ("A", "B"):
                d = (ev.get("departures") or {}).get(sid)
                if not d or d.get("floor_status") != "set":
                    continue
                x = d["minimum_profit"]
                own = pa.COSTS[ev["type_a" if sid == "A" else "type_b"]]
                bids = ev["bids"][sid]
                for i, b in enumerate(bids):
                    n_set += 1
                    f = b.get("floor") or {}
                    ret += bool(f.get("reasked"))
                    rai += bool(f.get("raised"))
                    if x > 0:
                        c = own.cost(ContractTerms(b["price"], b["delivery_weeks"], b["quality_tier"]))
                        hit = abs(b["price"] - (c + x)) <= 0.005
                        n_pos += 1
                        at_pos += hit
                        if i == len(bids) - 1:
                            n_fin += 1
                            at_fin += hit
    lines = [f"Homogeneous floor and floor_revenue cells: {nruns} runs, events 2-6; problems {len(problems)}",
             f"  bids under a positive floor: {n_pos}; at the floor (within half a cent): {at_pos} "
             f"({100 * at_pos / n_pos:.2f}%)",
             f"  final-round bids under a positive floor: {n_fin}; at the floor: {at_fin} ({100 * at_fin / n_fin:.2f}%)",
             f"  bids under any set floor ($0 included): {n_set}; returned {ret}; raised {rai}"]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
