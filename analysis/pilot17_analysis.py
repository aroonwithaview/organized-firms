"""Descriptive report of the Task 17 pilot (primitives prompts; seeds 3, 7, 9, 12, 19). It informs how to proceed.

Input: <repository>/data/runs/pilot17/<env>_<label>_<institution>_s<seed>/ (run.json, summary.csv, events.jsonl).
Market supplier profit per event = the winner's profit (0 if nothing is awarded). "Events 2-6" is where the record
can act. Per-seed differences are paired on (environment, institution, seed): every arm of a seed faces the same plan.
Usage: python pilot17_analysis.py [folder[;folder...]] [output file]
The folders are read together, so the pilot and the follow-up (data/runs/followup17) pool into one set of seeds; a run
found in two folders is a problem. Paths are relative to the repository root; the default output goes to
analysis/results/.
"""
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

REL = Path(__file__).resolve().parents[1]  # the repository root
RUNS = REL / "data" / "runs"  # the recorded runs
RESULTS = Path(__file__).resolve().parent / "results"  # where every analysis script writes by default
RESULTS.mkdir(exist_ok=True)
sys.path.insert(0, str(REL))
from orgsim.primitives import ContractTerms, get_fixed_suppliers  # noqa: E402


def rel(path):
    """A path as text, relative to the repository root and with forward slashes, so that results name no machine."""
    p = Path(path).resolve()
    try:
        return p.relative_to(REL).as_posix()
    except ValueError:
        return p.name


_S, _Q = get_fixed_suppliers()
COSTS = {"speed": _S, "quality": _Q}  # the code's own cost functions by cost type
PILOTS = [Path(p) for p in sys.argv[1].split(";")] if len(sys.argv) > 1 else [RUNS / "pilot17"]
PILOT = PILOTS[0]
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else RESULTS / "pilot17_results.txt"
INSTS, ENVS = ("mediator", "buyer_agent"), ("persistent", "random")
LATE = range(2, 7)


def load(roots=None):
    runs, problems = {}, []
    folders = [p for root in (roots or PILOTS) for p in sorted(root.iterdir()) if p.is_dir() and p.name != "logs"]
    for folder in folders:
        run = json.loads((folder / "run.json").read_text(encoding="utf-8"))
        u = run.get("usage") or {}
        if run.get("status") != "complete":  # a failed run is re-run with the same seed, never analysed
            problems.append(f"{folder.name}: {run.get('status')} {run.get('error')}")
            continue
        # A truncated or failed bidder call is re-asked and never used; a truncated analyst or platform reply is
        # used as it is. Both are flagged, by agent, rather than dropping the sequence.
        flags = {f"{label}.{k}": v.get(k) for label, v in (u.get("by_agent") or {}).items()
                 for k in ("failed_requests", "truncated", "no_usage") if v.get(k)}
        if flags:
            problems.append(f"FLAG (kept) {folder.name}: {flags}")
        rows = {int(r["event"]): r for r in csv.DictReader(open(folder / "summary.csv", encoding="utf-8", newline=""))}
        evs = {}
        for line in open(folder / "events.jsonl", encoding="utf-8"):
            if line.strip():
                e = json.loads(line)
                evs[e["event"]] = e
        env, rest = folder.name.split("_", 1)
        label, seed = rest.rsplit("_s", 1)
        inst = next(i for i in INSTS if label.endswith("_" + i))
        label = label[: -len(inst) - 1]
        key = (env, label, inst, int(seed))
        if key in runs:
            problems.append(f"DUPLICATE {rel(folder)}: {key} is also in another folder; the first is kept")
            continue
        runs[key] = (run, rows, evs)
    return runs, problems


def mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else float("nan")


def market(rows, key, events):
    return mean([float(rows[k][key]) for k in events if k in rows])


def firm_profit(rows, k, sid):
    r = rows[k]
    return float(r["supplier_profit"]) if r["winner"] == sid else 0.0


def summary(ds):
    """Mean, standard deviation across seeds (n - 1), the standard error of the mean (sd / sqrt(n)) and the count > 0."""
    sd = float(np.std(ds, ddof=1)) if len(ds) > 1 else float("nan")
    return (f"mean {np.mean(ds):+.2f}  sd {sd:.2f}  se {sd / np.sqrt(len(ds)):.2f}  "
            f">0: {sum(d > 0 for d in ds)}/{len(ds)}")


def revenue(evs, events):
    """Mean price paid per event (0 when nothing is awarded): the revenue-judged bidder's objective, per event."""
    return mean([evs[k]["outcome"]["winning_bid"]["price"] if evs[k]["outcome"]["awarded"] else 0.0
                 for k in events if k in evs])


def markup_r1(ev, sid):
    """Round-1 markup (percent) of supplier `sid` in event `ev`, at its round-1 terms and its own cost type."""
    b1 = ev["bids"][sid][0]
    own = ev["type_a" if sid == "A" else "type_b"]
    c1 = COSTS[own].cost(ContractTerms(b1["price"], b1["delivery_weeks"], b1["quality_tier"]))
    return (b1["price"] - c1) / c1 * 100


PRICE_IN, PRICE_OUT = 0.10, 0.50  # $ per million prompt / completion tokens (the model's list price)


def price_patterns(price):
    """The forms in which a reply can quote a price, as in run_demo's _price_patterns: the price to the cent
    ("123.45") and, for a whole number of dollars, also "$" and the integer ("$124")."""
    patterns = [rf"(?<!\d){re.escape(f'{price:.2f}')}(?!\d)"]
    if price == int(price):
        patterns.append(rf"\${int(price)}(?!\d|\.\d)")
    return patterns


def main():
    runs, problems = load()
    lines = []
    w = lines.append
    commits = {(r.get("git_commit"), r.get("git_dirty"), r.get("source_sha256"), r.get("prompts"),
                r.get("environment"))
               for r, _, _ in runs.values()}
    w(f"Pilot 17: {'; '.join(rel(p) for p in PILOTS)}; runs {len(runs)}; integrity problems {len(problems)}; "
      f"(commit, dirty, source, prompts, env): "
      f"{sorted(commits, key=str)}")
    for p in problems:
        w("  PROBLEM " + p)
    seeds = sorted({s for (_, _, _, s) in runs})
    homog = ("single_a", "single_b", "single_norecord", "organized", "organized_pass",
             "single_revenue", "organized_revenue")

    w("\nA. Market means per arm (per event; events 2-6 unless noted). n = seeds with a complete run.")
    w(f"{'env':<11} {'inst':<12} {'arm':<18} {'n':>2} {'profit 2-6':>10} {'profit 1-6':>10} {'buyer 2-6':>9} "
      f"{'joint 2-6':>9} {'walk-aways':>10}")
    for env in ENVS:
        for inst in INSTS:
            for arm in homog:
                cell = [runs[k] for k in runs if k[0] == env and k[1] == arm and k[2] == inst]
                if not cell:
                    continue
                w(f"{env:<11} {inst:<12} {arm:<18} {len(cell):>2} "
                  f"{mean([market(r, 'supplier_profit', LATE) for _, r, _ in cell]):>10.2f} "
                  f"{mean([market(r, 'supplier_profit', range(1, 7)) for _, r, _ in cell]):>10.2f} "
                  f"{mean([market(r, 'buyer_surplus', LATE) for _, r, _ in cell]):>9.2f} "
                  f"{mean([market(r, 'joint_surplus', LATE) for _, r, _ in cell]):>9.2f} "
                  f"{sum(r[k]['awarded'] == 'False' for _, r, _ in cell for k in r):>10}")

    def per_seed(env, inst, a, b, key="supplier_profit", events=LATE):
        out = []
        for s in seeds:
            x, y = runs.get((env, a, inst, s)), runs.get((env, b, inst, s))
            if x and y:
                out.append(market(x[1], key, events) - market(y[1], key, events))
        return out

    w("\nB. Per-seed differences in market supplier profit per event, events 2-6 (5 values; mean; seeds > 0).")
    for env in ENVS:
        for inst in INSTS:
            for a, b, what in (("single_b", "single_a", "U' - U (noise)"), ("single_norecord", "single_a", "U0 - U"),
                               ("organized", "single_a", "O - U"), ("organized_pass", "single_a", "O_pass - U"),
                               ("organized", "organized_pass", "O - O_pass")):
                ds = per_seed(env, inst, a, b)
                if ds:
                    w(f"  {env:<11} {inst:<12} {what:<16} [{', '.join(f'{d:+.2f}' for d in ds)}]  {summary(ds)}")

    w("\nC. Mixed markets (persistent): per-seed profit gap X - U per event, both positions averaged.")
    for x in ("organized", "single_norecord"):
        for inst in INSTS:
            for events, tag in ((LATE, "events 2-6"), (range(1, 7), "events 1-6"), (range(1, 2), "event 1")):
                ds = []
                for s in seeds:
                    vals = []
                    for pos, xs in (("xa", "A"), ("xb", "B")):
                        r = runs.get(("persistent", f"{x}_vs_single_{pos}", inst, s))
                        if r:
                            us = "B" if xs == "A" else "A"
                            vals += [firm_profit(r[1], k, xs) - firm_profit(r[1], k, us) for k in events if k in r[1]]
                    if vals:
                        ds.append(float(np.mean(vals)))
                if ds:
                    w(f"  {x + ':single':<24} {inst:<12} {tag:<10} [{', '.join(f'{d:+.2f}' for d in ds)}]  "
                      f"{summary(ds)}")

    w("\nC0. Placebo for C: the same gap between two identical single firms. Per seed, the mean over events of "
      "[(A - B) in the U run + (B - A) in the U' run] / 2, firm profits as in C. A in the U run and B in the U' run "
      "play the part of X in positions xa and xb, with the same plan, so this is the gap C would show if X were a "
      "single firm.")
    for inst in INSTS:
        for events, tag in ((LATE, "events 2-6"), (range(1, 7), "events 1-6"), (range(1, 2), "event 1")):
            ds = []
            for s in seeds:
                u, u2 = runs.get(("persistent", "single_a", inst, s)), runs.get(("persistent", "single_b", inst, s))
                if u and u2:
                    vals = [firm_profit(u[1], k, "A") - firm_profit(u[1], k, "B") for k in events if k in u[1]]
                    vals += [firm_profit(u2[1], k, "B") - firm_profit(u2[1], k, "A") for k in events if k in u2[1]]
                    ds.append(float(np.mean(vals)))
            if ds:
                w(f"  {'single:single (placebo)':<24} {inst:<12} {tag:<10} [{', '.join(f'{d:+.2f}' for d in ds)}]  "
                  f"{summary(ds)}")

    w("\nD. Incentive arm (persistent, buyer agent): per-seed market supplier profit per event, and the DiD "
      "(O_i - O) - (U_i - U). The analyst can act from event 2; at event 1 the arms differ in wording only.")
    for events, tag in ((range(1, 7), "events 1-6"), (LATE, "events 2-6"), (range(1, 2), "event 1")):
        w(f"  {tag}:")
        dids = []
        for s in seeds:
            g = {a: runs.get(("persistent", a, "buyer_agent", s)) for a in ("organized_revenue", "organized",
                                                                            "single_revenue", "single_a")}
            if all(g.values()):
                v = {a: market(r[1], "supplier_profit", events) for a, r in g.items()}
                rev = {a: revenue(r[2], events) for a, r in g.items()}
                did = (v["organized_revenue"] - v["organized"]) - (v["single_revenue"] - v["single_a"])
                dids.append(did)
                w(f"    seed {s}: profit O_i {v['organized_revenue']:.2f}  O {v['organized']:.2f}  "
                  f"U_i {v['single_revenue']:.2f}  U {v['single_a']:.2f}  DiD {did:+.2f}   |  price paid O_i "
                  f"{rev['organized_revenue']:.2f}  O {rev['organized']:.2f}  U_i {rev['single_revenue']:.2f}  "
                  f"U {rev['single_a']:.2f}")
        if dids:
            w(f"    DiD mean {np.mean(dids):+.2f}  >0: {sum(d > 0 for d in dids)}/{len(dids)}")

    w("\nE. Round-1 markups tracking past same-type markups (pass-through slope; validation: ~0 in U0). Pooled OLS "
      "slope over both institutions; the same by institution; and the within slope, which removes each run x "
      "supplier's own means (a fixed-effects slope: event-to-event tracking within a sequence).")
    for env in ENVS:
        for arm in ("single_a", "single_b", "single_norecord", "organized", "organized_pass"):
            pts = []  # (institution, group, x, y)
            for (e, a, inst, s), (_, rows, evs) in runs.items():
                if e != env or a != arm:
                    continue
                for sid in ("A", "B"):
                    last = {}
                    for k in range(1, 7):
                        if k not in evs:
                            continue
                        ev, own = evs[k], evs[k]["type_a" if sid == "A" else "type_b"]
                        if k >= 2 and own in last:  # y: round-1 markup at the round-1 terms (percent)
                            pts.append((inst, (inst, s, sid), last[own], markup_r1(ev, sid)))
                        # x for later events: the final markup of the most recent event with this own type
                        last[own] = ev["outcome"]["all_inflation_rates"][sid] * 100
            if len(pts) >= 5:
                slope = float(np.polyfit([p[2] for p in pts], [p[3] for p in pts], 1)[0])
                by_inst = {i: float(np.polyfit([p[2] for p in pts if p[0] == i], [p[3] for p in pts if p[0] == i], 1)[0])
                           for i in INSTS}
                groups = {}
                for _, g, x, y in pts:
                    groups.setdefault(g, []).append((x, y))
                dx, dy = [], []
                for vals in groups.values():
                    mx, my = np.mean([v[0] for v in vals]), np.mean([v[1] for v in vals])
                    dx += [v[0] - mx for v in vals]
                    dy += [v[1] - my for v in vals]
                within = float(np.dot(dx, dy) / np.dot(dx, dx)) if np.dot(dx, dx) > 0 else float("nan")
                w(f"  {env:<11} {arm:<16} n={len(pts):>3} slope={slope:+.3f}  mediator {by_inst['mediator']:+.3f}  "
                  f"buyer_agent {by_inst['buyer_agent']:+.3f}  within {within:+.3f}")

    w("\nF. Briefings (organized arms): length and content.")
    for env in ENVS:
        for arm in ("organized", "organized_revenue"):
            texts = [b for (e, a, _, _), (_, _, evs) in runs.items() if e == env and a == arm
                     for ev in evs.values() for b in (ev.get("analyst_briefings") or {}).values() if b]
            if texts:
                words = [len(t.split()) for t in texts]
                contract = sum(bool(re.search(r"(speed|quality) priority", t, re.I)) for t in texts)
                prices = sum(bool(re.search(r"\$\s?\d", t)) for t in texts)
                w(f"  {env:<11} {arm:<18} briefings {len(texts):>3}  words median {np.median(words):g} "
                  f"(min {min(words)}, max {max(words)})  name the contract {contract}  quote a $ figure {prices}")

    w("\nG. Event 1, where every arm has the same information (wording check): market supplier profit at event 1 and "
      "the mean round-1 markup (percent, both suppliers), means over seeds. Event 1 has the same draws in both "
      "environments.")
    for env in ENVS:
        for inst in INSTS:
            for arm in homog:
                cell = [runs[k] for k in runs if k[0] == env and k[1] == arm and k[2] == inst]
                if not cell:
                    continue
                prof = mean([float(r[1]["supplier_profit"]) for _, r, _ in cell if 1 in r])
                mk = mean([markup_r1(e[1], sid) for _, _, e in cell if 1 in e for sid in ("A", "B")])
                w(f"  {env:<11} {inst:<12} {arm:<18} n={len(cell)}  profit at event 1 {prof:6.2f}  "
                  f"round-1 markup at event 1 {mk:6.2f}%")

    w("\nH. Tokens and spend by arm (per sequence, means over seeds; spend at $0.10 / $0.50 per million prompt / "
      "completion tokens).")
    total_in = total_out = 0
    by_arm = {}
    for (env, label, inst, s), (run, _, _) in runs.items():
        u = run.get("usage") or {}
        total_in += u.get("prompt_tokens", 0)
        total_out += u.get("completion_tokens", 0)
        by_arm.setdefault((env, inst, label), []).append(u)
    for (env, inst, label), us in sorted(by_arm.items()):
        reqs = mean([u.get("requests", 0) for u in us])
        pin = mean([u.get("prompt_tokens", 0) for u in us])
        pout = mean([u.get("completion_tokens", 0) for u in us])
        agents = sorted({a for u in us for a in (u.get("by_agent") or {})})
        per_agent = ", ".join(f"{a} {mean([(u.get('by_agent') or {}).get(a, {}).get('completion_tokens', 0) for u in us]):.0f}"
                              for a in agents)
        w(f"  {env:<11} {inst:<12} {label:<34} requests {reqs:5.1f}  prompt {pin:8.0f}  completion {pout:7.0f}  "
          f"spend ${pin / 1e6 * PRICE_IN + pout / 1e6 * PRICE_OUT:.4f}  (completion by agent: {per_agent})")
    w(f"  TOTAL over {len(runs)} runs: prompt tokens {total_in}, completion tokens {total_out}, spend "
      f"${total_in / 1e6 * PRICE_IN + total_out / 1e6 * PRICE_OUT:.2f}")

    w("\nI. Platform replies that contain a price the rival had bid by then (rounds 1 to r for a round-r reply; within "
      "a round both suppliers bid before the mediator replies), excluding a price the addressee itself bid in rounds "
      "1 to r. Under the mediator a match is a possible leak; the buyer's agent never sees the rival's bids, so its "
      "matches are coincidences and serve as a rough baseline. Price forms as in run_demo's _price_patterns.")
    hits = []
    for env in ENVS:
        for inst in INSTS:
            n_rep = n_hit = 0
            for (e, label, i, s), (_, _, evs) in sorted(runs.items()):
                if e != env or i != inst:
                    continue
                for k, ev in sorted(evs.items()):
                    for sid, rival in (("A", "B"), ("B", "A")):
                        for r, b in enumerate(ev["bids"][sid], start=1):
                            reply = b.get("feedback")
                            if not reply:
                                continue
                            n_rep += 1
                            own = {x["price"] for x in ev["bids"][sid][:r]}
                            found = [p for p in {x["price"] for x in ev["bids"][rival][:r]} - own
                                     if any(re.search(pat, reply) for pat in price_patterns(p))]
                            if found:
                                n_hit += 1
                                hits.append((env, inst, label, s, k, r, sid, sorted(found), reply))
            w(f"  {env:<11} {inst:<12} replies {n_rep:>5}  with a rival price {n_hit:>4}")
    w("  Mediator replies with a match, in full:" if any(h[1] == "mediator" for h in hits) else
      "  No mediator reply has a match.")
    for env, inst, label, s, k, r, sid, found, reply in hits:
        if inst == "mediator":
            w(f"  - {env} {label} s{s} event {k} round {r}, reply to {sid}, rival price(s) {found}: {reply!r}")

    w("\nK. Mixed markets (persistent), events 2-6 and event 1, both positions pooled: how each firm bids. Per seed, "
      "wins of X and of the single firm U (out of 10 events for events 2-6 and 2 for event 1: the events x 2 "
      "positions); then pooled means over seeds, positions and events: round-1 markup and final markup (percent of "
      "the firm's own cost at the bid's terms), price cut (round-1 price minus final price), final delivery weeks "
      "and tier, and profit per event.")
    cells = [(x, inst, events, tag) for x, inst in (("organized", "mediator"), ("organized", "buyer_agent"),
                                                  ("single_norecord", "mediator"), ("single_norecord", "buyer_agent"),
                                                  ("placebo", "mediator"), ("placebo", "buyer_agent"))
             for events, tag in ((LATE, "events 2-6"), (range(1, 2), "event 1"))]
    for x, inst, events, tag in cells:
        # The placebo (as in C0): X is A in the U run and B in the U' run; the single firm is the other supplier.
        source = ({"xa": runs.get(("persistent", "single_a", inst, s)), "xb": runs.get(("persistent", "single_b", inst, s))}
                  for s in seeds) if x == "placebo" else None
        st = {who: {"win": [], "r1": [], "fin": [], "cut": [], "wk": [], "tier": [], "prof": []}
              for who in ("X", "U")}
        wins, mk_gaps = [], []
        for s in seeds:
            ws = {"X": 0, "U": 0}
            n_before = {who: len(st[who]["fin"]) for who in ("X", "U")}
            placebo = next(source) if source is not None else None
            for pos, xs in (("xa", "A"), ("xb", "B")):
                r = placebo[pos] if placebo is not None else runs.get(("persistent", f"{x}_vs_single_{pos}", inst, s))
                if not r:
                    continue
                us = "B" if xs == "A" else "A"
                for k in events:
                    ev = r[2].get(k)
                    if not ev:
                        continue
                    for who, sid in (("X", xs), ("U", us)):
                        won = bool(ev["outcome"]["awarded"]) and ev["outcome"]["winner_id"] == sid
                        ws[who] += won
                        d = st[who]
                        d["win"].append(won)
                        d["r1"].append(markup_r1(ev, sid))
                        d["fin"].append(ev["outcome"]["all_inflation_rates"][sid] * 100)
                        d["cut"].append(ev["bids"][sid][0]["price"] - ev["bids"][sid][-1]["price"])
                        d["wk"].append(ev["bids"][sid][-1]["delivery_weeks"])
                        d["tier"].append(ev["bids"][sid][-1]["quality_tier"])
                        d["prof"].append(firm_profit(r[1], k, sid))
            wins.append(f"{ws['X']}-{ws['U']}")
            seed_fin = {who: st[who]["fin"][n_before[who]:] for who in ("X", "U")}
            if seed_fin["X"] and seed_fin["U"]:
                mk_gaps.append(mean(seed_fin["X"]) - mean(seed_fin["U"]))
        name = "single:single (placebo)" if x == "placebo" else x + ":single"
        w(f"  {name:<24} {inst:<12} {tag:<10} wins X-U per seed [{', '.join(wins)}]; final markup X - U per seed "
          f"[{', '.join(f'{g:+.2f}' for g in mk_gaps)}]")
        for who in ("X", "U"):
            d = st[who]
            if d["win"]:
                n_win = sum(d["win"])
                per_win = sum(d["prof"]) / n_win if n_win else float("nan")
                w(f"    {who}: win share {mean(d['win']):.2f}  round-1 markup {mean(d['r1']):6.2f}%  final markup "
                  f"{mean(d['fin']):6.2f}%  price cut {mean(d['cut']):5.2f}  final weeks {mean(d['wk']):.2f}  "
                  f"final tier {mean(d['tier']):.2f}  profit/event {mean(d['prof']):5.2f}  profit per win {per_win:5.2f}")

    w("\nJ. Briefings chosen by a fixed rule (the first pilot seed, supplier A, events 2 and 6), verbatim.")
    first = seeds[0] if seeds else None
    for env, arm, inst in (("persistent", "organized", "mediator"), ("persistent", "organized", "buyer_agent"),
                           ("persistent", "organized_revenue", "buyer_agent"), ("random", "organized", "buyer_agent")):
        r = runs.get((env, arm, inst, first))
        if not r:
            continue
        for k in (2, 6):
            text = ((r[2].get(k) or {}).get("analyst_briefings") or {}).get("A")
            ev = r[2].get(k) or {}
            w(f"\n  [{env} {arm} {inst} seed {first} event {k}; contract {ev.get('contract')}, A's type "
              f"{ev.get('type_a')}]\n" + "\n".join("    " + line for line in (text or "(none)").splitlines()))
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
