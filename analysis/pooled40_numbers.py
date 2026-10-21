"""Numbers for the proposal pooled over seeds 23-62 (40 seeds), from the raw runs.

Same cells, events and definitions as panel_reanalysis.py and replication18_analysis.py; only the seed list changes.
Mixed cells: conflict brief_revenue:forward_revenue, team brief:forward; both positions, both institutions.
A  final bids below own cost, events 2-6, counts for the B and F firm in each cell (Figure 2A, Table 1).
B  conflict cell, B firm's round-1 price equal to a dollar figure in its own briefing, against the briefing of the
   next seed in the same 20-seed batch (the placebo of panel_reanalysis.py), events 2-6 (Figure 2B).
C  the F firm's final-bid cost and price, conflict - team, per seed (events 2-6).
D  per seed, the below-cost share B - F under conflict (events 2-6): seeds below, at and above zero.
E  win share B - F under conflict, per seed (events 2-6).
F  H1 (B firm's profit minus F firm's under conflict) at event 1 and over events 2-6, per seed.
   H: per seed, the events 2-6 gap minus the event-1 gap.
G  final bids below own cost at event 1, conflict cell, B and F firm (counts of 160) and per seed B - F.
Per-seed values are pooled over positions event by event within an institution, then averaged over the two
institutions, as in the source scripts. Estimates only; no tests.
Usage: python pooled40_numbers.py [output file]
"""
import sys
from pathlib import Path

import numpy as np

import panel_reanalysis as pr
import pilot17_analysis as pa
import replication18_analysis as ra

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else pa.RESULTS / "pooled40_results.txt"
ROOTS = [pa.RUNS / "departures18", pa.RUNS / "replication18", pa.RUNS / "replication18_team"]
BATCHES = (list(range(23, 43)), list(range(43, 63)))
SEEDS = [s for b in BATCHES for s in b]
T975_39 = 2.023


def describe(vals):
    d = np.array(vals, dtype=float)
    se = float(np.std(d, ddof=1)) / np.sqrt(len(d))
    r = np.round(d, 9)
    return (f"n={len(d)}  mean {d.mean():+.4f}  se {se:.4f}  95% CI [{d.mean() - T975_39 * se:+.4f}, "
            f"{d.mean() + T975_39 * se:+.4f}]  >0: {int((r > 0).sum())}  <0: {int((r < 0).sum())}  "
            f"=0: {int((r == 0).sum())}")


def main():
    runs, problems = pa.load(ROOTS)
    lines = [f"Pooled seeds 23-62 ({len(SEEDS)} seeds): runs loaded {len(runs)}; problems {len(problems)}"]
    lines += [f"  {p}" for p in problems]
    w = lines.append

    w("\nA. Final bids below own cost, events 2-6, both positions, both institutions")
    for obj in ("team", "conflict"):
        for firm in ("F", "B"):
            fn, vals = pr.below(firm), []
            for s in SEEDS:
                for inst in pa.INSTS:
                    cell = pr.cell_runs(runs, obj, inst, s)
                    if cell is None:
                        raise SystemExit(f"missing cell {obj} {inst} seed {s}")
                    vals += fn(cell, inst)
            w(f"  {obj:8s} {firm} firm: {int(sum(vals))} of {len(vals)}  ({sum(vals) / len(vals):.4%})")

    w("\nB. Conflict cell, B firm round-1 price equals a dollar figure in a briefing, events 2-6")
    own_n = plc_n = n_all = 0
    for b in BATCHES:
        _, own, plc, n = pr.anchoring(runs, b, "conflict")
        own_c, plc_c = round(own * n), round(plc * n)
        w(f"  batch {b[0]}-{b[-1]}: own {own_c} of {n}, next seed's {plc_c} of {n}")
        own_n, plc_n, n_all = own_n + own_c, plc_n + plc_c, n_all + n
    w(f"  pooled: own {own_n} of {n_all} ({own_n / n_all:.4%}); next seed's {plc_n} of {n_all} "
      f"({plc_n / n_all:.4%})")

    w("\nC. F firm's final bids, conflict - team, per seed, events 2-6")
    for what in ("cost", "price"):
        v = {o: pr.per_seed(runs, SEEDS, o, pr.f_terms(what)) for o in pr.CELLS}
        w(f"  {what}: conflict {np.mean(list(v['conflict'].values())):.4f}  team "
          f"{np.mean(list(v['team'].values())):.4f}")
        w(f"    conflict - team: {describe([v['conflict'][s] - v['team'][s] for s in SEEDS])}")

    w("\nD. Below-cost share B - F under conflict, per seed, events 2-6")
    vb = pr.per_seed(runs, SEEDS, "conflict", pr.below("B"))
    vf = pr.per_seed(runs, SEEDS, "conflict", pr.below("F"))
    w(f"  {describe([vb[s] - vf[s] for s in SEEDS])}")

    w("\nE. Win share B - F under conflict, per seed, events 2-6")
    ve = pr.per_seed(runs, SEEDS, "conflict", pr.win_share)
    w(f"  {describe([ve[s] for s in SEEDS])}")

    w("\nF. H1, the B firm's profit minus the F firm's under conflict, per seed")
    h = {}
    for tag, ev in (("event 1", ra.E1), ("events 2-6", ra.LATE)):
        h[tag] = dict(ra.gap(runs, ra.B, ra.F, ev, seeds=SEEDS))
        w(f"  {tag}: {describe([h[tag][s] for s in SEEDS])}")
    w(f"  events 2-6 minus event 1 (H): {describe([h['events 2-6'][s] - h['event 1'][s] for s in SEEDS])}")

    w("\nG. Final bids below own cost at event 1, conflict cell")
    e1 = {}
    for firm in ("B", "F"):
        cnt, n, per = 0, 0, {}
        for s in SEEDS:
            inst_vals = []
            for inst in pa.INSTS:
                vals = []
                for pos, (run, bs) in pr.cell_runs(runs, "conflict", inst, s).items():
                    sid = bs if firm == "B" else ("B" if bs == "A" else "A")
                    vals.append(float(run[2][1]["outcome"]["all_inflation_rates"][sid] < 0))
                cnt, n = cnt + int(sum(vals)), n + len(vals)
                inst_vals.append(np.mean(vals))
            per[s] = float(np.mean(inst_vals))
        e1[firm] = per
        w(f"  {firm} firm: {cnt} of {n}")
    w(f"  per seed, B - F: {describe([e1['B'][s] - e1['F'][s] for s in SEEDS])}")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
