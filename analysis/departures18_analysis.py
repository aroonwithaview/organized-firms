"""Departures stage (Task 18): the tests and estimates fixed in `2026-10-17 departures analysis plan.md`.

Data: the departures runs (persistent environment, seeds 23-42; data/runs/departures18, whose manifest.json lists the
jobs) and the frozen-code concession arms (U, O and O_pass under the buyer's agent, data/runs/confirm17_concession), read
with pilot17_analysis.load. The unit is the seed; each seed's value is the mean over events 2-6, pooled over the two
institutions (the mean of the per-institution values; each institution has the same number of events).

Gate (fixed sequence; it spends no significance level): event-1 final-bid margin, conflict minus team, over the
homogeneous B and F cells; passes if p < 0.05 with a negative mean.
Primary family (Holm over four, two-sided exact sign-flip): H1 the mixed B:F gap in firm profit under conflict;
H2 the margin DiD (B - F)_conf - (B - F)_team; H3 the mixed Floor:B gap under conflict; H4 the margin DiD
(Floor - B)_conf - (Floor - B)_team.
Concession family (Holm over three): C1 O - U and C2 O - O_pass (frozen code), C3 B - F (team, buyer's agent,
homogeneous); the cut is the round-1 price minus the price of the supplier's last bid, mean over both suppliers.
Secondary: estimates with 95% t intervals (19 degrees of freedom), no p-values.
Usage: python departures18_analysis.py [output file] [departures root;...] [frozen concession root;...]
"""
import random
import re
import sys
from pathlib import Path

import numpy as np

import followup17_primary as fp
import pilot17_analysis as pa
from orgsim.primitives import ContractTerms

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else pa.RESULTS / "departures18_results.txt"
DEP_ROOTS = ([Path(p) for p in sys.argv[2].split(";")] if len(sys.argv) > 2
             else [pa.RUNS / "departures18"])
FROZEN_ROOTS = ([Path(p) for p in sys.argv[3].split(";")] if len(sys.argv) > 3
                else [pa.RUNS / "confirm17_concession"])
SEEDS = list(range(23, 43))
INSTS = pa.INSTS
W = {"events 2-6": range(2, 7), "event 1": range(1, 2), "events 2-5": range(2, 6)}
LATE = W["events 2-6"]
T975_19 = 2.093  # the 0.975 quantile of t with 19 degrees of freedom
ORG = {"B": "brief", "F": "forward", "Floor": "floor", "Cap": "brief_cap", "Cost": "brief_cost"}


def org(name, conf):
    return ORG[name] + ("_revenue" if conf else "")


# ---- per-run statistics -------------------------------------------------------------------------------------------

def margin(run, events):
    """Market mean of the final-bid margin (percent over own cost at the bid's terms), both suppliers."""
    _, _, evs = run
    return pa.mean([evs[k]["outcome"]["all_inflation_rates"][sid] * 100 for k in events if k in evs
                    for sid in ("A", "B")])


def cut(run, events):
    _, _, evs = run
    return pa.mean([evs[k]["bids"][sid][0]["price"] - evs[k]["bids"][sid][-1]["price"] for k in events if k in evs
                    for sid in ("A", "B")])


def below_cost(run, events):
    """Share of final bids priced below own cost at their terms."""
    _, _, evs = run
    return pa.mean([float(evs[k]["outcome"]["all_inflation_rates"][sid] < 0) for k in events if k in evs
                    for sid in ("A", "B")])


def loss_win(run, events):
    """Share of awarded events whose winner's profit is negative."""
    _, rows, _ = run
    return pa.mean([float(float(rows[k]["supplier_profit"]) < 0) for k in events
                    if k in rows and rows[k]["awarded"] == "True"])


def market_stat(key):
    return lambda run, events: pa.market(run[1], key, events)


STATS = {"profit": market_stat("supplier_profit"), "buyer": market_stat("buyer_surplus"),
         "joint": market_stat("joint_surplus"), "margin": margin, "below_cost": below_cost, "loss_win": loss_win,
         "cut": cut}


def firm_gap(run_xa, run_xb, events):
    """X's firm profit minus Y's, pooled over positions xa (X is A) and xb (X is B) event by event."""
    vals = []
    for run, xs in ((run_xa, "A"), (run_xb, "B")):
        rows = run[1]
        ys = "B" if xs == "A" else "A"
        vals += [pa.firm_profit(rows, k, xs) - pa.firm_profit(rows, k, ys) for k in events if k in rows]
    return float(np.mean(vals))


def firm_profit_in_mixed(run_xa, run_xb, events, which):
    """The X firm's (which = 'x') or Y firm's ('y') profit per event in the X:Y cell, pooled over positions."""
    vals = []
    for run, xs in ((run_xa, "A"), (run_xb, "B")):
        rows = run[1]
        sid = xs if which == "x" else ("B" if xs == "A" else "A")
        vals += [pa.firm_profit(rows, k, sid) for k in events if k in rows]
    return float(np.mean(vals))


# ---- per-seed contrasts, institutions pooled ----------------------------------------------------------------------

def per_seed(runs, seeds, insts, fn):
    """fn(inst, seed) -> value or None; per seed, the mean over insts (None if any institution is missing)."""
    out = []
    for s in seeds:
        vals = [fn(i, s) for i in insts]
        if all(v is not None for v in vals):
            out.append((s, float(np.mean(vals))))
    return out


def homog(runs, label, inst, s):
    return runs.get(("persistent", label, inst, s))


def diff(runs, stat, events, a, b):
    """value(a) - value(b), homogeneous cells."""
    def fn(i, s):
        ra, rb = homog(runs, a, i, s), homog(runs, b, i, s)
        return STATS[stat](ra, events) - STATS[stat](rb, events) if ra and rb else None
    return fn


def did(runs, stat, events, a, b, c, d):
    """(a - b) - (c - d), homogeneous cells."""
    def fn(i, s):
        r = [homog(runs, x, i, s) for x in (a, b, c, d)]
        if not all(r):
            return None
        v = [STATS[stat](x, events) for x in r]
        return (v[0] - v[1]) - (v[2] - v[3])
    return fn


def mixed(runs, x, y, events, positions=("xa", "xb")):
    """The X:Y firm gap per institution and seed; positions ("xa",) or ("xb",) gives one position alone."""
    def fn(i, s):
        ra = runs.get(("persistent", f"{x}_vs_{y}_xa", i, s))
        rb = runs.get(("persistent", f"{x}_vs_{y}_xb", i, s))
        if not (ra and rb):
            return None
        if positions == ("xa",):
            return _gap_one(ra, "A", events)
        if positions == ("xb",):
            return _gap_one(rb, "B", events)
        return firm_gap(ra, rb, events)
    return fn


def _gap_one(run, xs, events):
    rows = run[1]
    ys = "B" if xs == "A" else "A"
    return float(np.mean([pa.firm_profit(rows, k, xs) - pa.firm_profit(rows, k, ys) for k in events if k in rows]))


def gate_fn(runs):
    def fn(i, s):
        conf = [homog(runs, org(n, True), i, s) for n in ("B", "F")]
        team = [homog(runs, org(n, False), i, s) for n in ("B", "F")]
        if not all(conf + team):
            return None
        return (np.mean([margin(r, W["event 1"]) for r in conf]) - np.mean([margin(r, W["event 1"]) for r in team]))
    return fn


# ---- messages: floor use, cap, cost quoting and the deterministic coding (plan: "Message coding", "Floor use") ------

ANALYST_ORGS = {org(n, c) for n in ("B", "Floor", "Cap", "Cost") for c in (False, True)}
FLOOR_ORGS = {org("Floor", False), org("Floor", True)}
LOWER_BOUND = ("at least $", "no lower than", "no less than", "not below", "not go below", "don't go below",
               "do not go below", "never go below", "not bid below", "don't bid below", "do not bid below",
               "never bid below", "minimum price", "price floor", "floor of", "floor at", "above $", "higher than $")
DOLLAR = re.compile(r"\$\s?\d[\d,]*(\.\d+)?")
NOTICES = ("MINIMUM PROFIT PER CONTRACT FOR THIS EVENT", "NO MINIMUM PROFIT PER CONTRACT IS SET")


def org_of(ev, sid):
    return ev["organization_a" if sid == "A" else "organization_b"]


N4_WORDS = re.compile(r"minimum[\s*_`~]+profit[\s*_`~]+per[\s*_`~]+contract", re.IGNORECASE)


def n4_violations(run):
    """Plan, added integrity check: lines a floor bidder still reads that begin with the words of the floor label
    (after any leading non-alphanumeric characters or list number; emphasis marks inside the phrase ignored), other
    than the code's notice. Returns a list of (event, supplier, line)."""
    out = []
    for k, ev in sorted(run[2].items()):
        for sid in ("A", "B"):
            text = (ev.get("analyst_briefings") or {}).get(sid)
            if org_of(ev, sid) not in FLOOR_ORGS or not text:
                continue
            for line in text.split("\n"):
                s = line.strip()
                if s.startswith(NOTICES):
                    continue
                if N4_WORDS.match(re.sub(r"^(?:[\W_]|[0-9]+[.)])*", "", s)):
                    out.append((k, sid, s))
    return out


def analyst_text(ev, sid):
    """The analyst-written part of what the bidder reads (the code's floor notice removed), or None."""
    if org_of(ev, sid) not in ANALYST_ORGS:
        return None
    text = (ev.get("analyst_briefings") or {}).get(sid)
    if text is None:
        return None
    if org_of(ev, sid) in FLOOR_ORGS:
        text = "\n".join(line for line in text.split("\n") if not line.strip().startswith(NOTICES)).rstrip()
    return text


def norm(text):
    return text.lower().replace("’", "'")


def quotes(text, value):
    """Whether `text` quotes `value` to the cent or, for a whole number of dollars, as "$" and the integer."""
    return any(re.search(p, text) for p in pa.price_patterns(round(value, 2)))


def quotes_cost(text, value):
    """Cost figures are shown to one decimal: match "85.0", "85.00" or "$85" (whole dollars)."""
    pats = [rf"(?<!\d){re.escape(f'{value:.1f}')}(?!\d)", rf"(?<!\d){re.escape(f'{value:.2f}')}(?!\d)"]
    if value == int(value):
        pats.append(rf"\${int(value)}(?!\d|\.\d)")
    return any(re.search(p, text) for p in pats)


def message_codes(run, k, sid):
    """Codes of one analyst-written message (event k, supplier sid), or None."""
    _, _, evs = run
    ev = evs[k]
    text = analyst_text(ev, sid)
    if text is None:
        return None
    past = [evs[j]["bids"][sid][-1]["price"] for j in range(1, k) if j in evs]
    own = pa.COSTS[ev["type_a" if sid == "A" else "type_b"]]
    cost_figs = (own.c_base, own.cost(ContractTerms(0, own.delivery_max, 2)), own.cost(ContractTerms(0, 1, 5)))
    return {"words": len(text.split()), "dollar": float(bool(DOLLAR.search(text))),
            "lower_bound": float(any(p in norm(text) for p in LOWER_BOUND)),
            "relayed": float(np.mean([quotes(text, p) for p in past])) if past else np.nan,
            "adherence": float(quotes(text, ev["bids"][sid][0]["price"])),
            "cost_quoted": float(any(quotes_cost(text, v) for v in cost_figs))}


def floor_codes(run, k, sid):
    """Floor use of one supplier-event (floor organizations, events 2-6), or None."""
    ev = run[2][k]
    d = (ev.get("departures") or {}).get(sid)
    if not d or d.get("organization") not in FLOOR_ORGS:
        return None
    x = d.get("minimum_profit")
    bids = [b for b in ev["bids"][sid] if "floor" in b]
    own = pa.COSTS[ev["type_a" if sid == "A" else "type_b"]]
    gaps = []
    for b in bids:
        f = b["floor"]
        first = f.get("first") or {"price": b["price"], "cost": own.cost(
            ContractTerms(b["price"], b["delivery_weeks"], b["quality_tier"]))}
        gaps.append(first["price"] - (first["cost"] + f["minimum_profit"]))
    return {"set_pos": float(d["floor_status"] == "set" and (x or 0) > 0),
            "set_zero": float(d["floor_status"] == "set" and x == 0),
            "none": float(d["floor_status"] == "none"), "malformed": float(d["floor_status"] == "malformed"),
            "x": x if d["floor_status"] == "set" else np.nan, "analyst_reasked": float(bool(d.get("reasked"))),
            "returned": float(np.mean([b["floor"]["reasked"] for b in bids])) if bids else np.nan,
            "raised": float(np.mean([b["floor"]["raised"] for b in bids])) if bids else np.nan,
            "first_gap": float(np.mean(gaps)) if gaps else np.nan}


def cap_codes(run, k, sid):
    d = (run[2][k].get("departures") or {}).get(sid)
    if not d or "word_limit" not in d:
        return None
    return {"reasked": float(bool(d["reasked"])), "cut": float(bool(d["cut"])),
            "original_words": d["original_words"], "final_words": d["final_words"]}


def cell_mean(runs, label, inst_list, events, codes_fn, key, only_org=None):
    """Per seed: the mean of codes_fn(...)[key] over institutions, suppliers and events (NaN entries skipped);
    only_org keeps the suppliers of that organization (for mixed cells)."""
    out = []
    for s in SEEDS:
        vals = []
        for i in inst_list:
            r = runs.get(("persistent", label, i, s))
            if not r:
                continue
            for k in events:
                if k not in r[2]:
                    continue
                for sid in ("A", "B"):
                    if only_org and org_of(r[2][k], sid) != only_org:
                        continue
                    c = codes_fn(r, k, sid)
                    if c is not None and not (isinstance(c[key], float) and np.isnan(c[key])):
                        vals.append(c[key])
        if vals:
            out.append((s, float(np.mean(vals))))
    return out


def seed_diff(a_pairs, b_pairs):
    b = dict(b_pairs)
    return [(s, v - b[s]) for s, v in a_pairs if s in b]


def describe(pairs):
    d = np.array([v for _, v in pairs], dtype=float)
    if len(d) < 2:
        return f"n={len(d)}"
    sd = float(np.std(d, ddof=1))
    se = sd / np.sqrt(len(d))
    half = (T975_19 if len(d) == 20 else float("nan")) * se
    return (f"n={len(d)}  mean {d.mean():+.3f}  sd {sd:.3f}  se {se:.3f}  95% CI [{d.mean() - half:+.3f}, "
            f"{d.mean() + half:+.3f}]  >0: {int((np.round(d, 9) > 0).sum())}/{len(d)} (=0: {int((np.round(d, 9) == 0).sum())})")


def per_seed_line(pairs):
    return ", ".join(f"s{s} {v:+.2f}" for s, v in pairs)


def h_family(runs, events, insts=INSTS):
    """The four primary statistics, per seed."""
    return [
        ("H1 mixed B:F gap, conflict (firm profit)",
         per_seed(runs, SEEDS, insts, mixed(runs, org("B", True), org("F", True), events))),
        ("H2 margin DiD (B - F)_conf - (B - F)_team",
         per_seed(runs, SEEDS, insts, did(runs, "margin", events, org("B", True), org("F", True),
                                          org("B", False), org("F", False)))),
        ("H3 mixed Floor:B gap, conflict (firm profit)",
         per_seed(runs, SEEDS, insts, mixed(runs, org("Floor", True), org("B", True), events))),
        ("H4 margin DiD (Floor - B)_conf - (Floor - B)_team",
         per_seed(runs, SEEDS, insts, did(runs, "margin", events, org("Floor", True), org("B", True),
                                          org("Floor", False), org("B", False)))),
    ]


def main():
    runs, problems = pa.load(DEP_ROOTS)
    frozen, fproblems = pa.load(FROZEN_ROOTS)
    lines = []
    w = lines.append
    w(f"Departures stage. Roots: {'; '.join(pa.rel(r) for r in DEP_ROOTS)}; frozen: "
      f"{'; '.join(pa.rel(r) for r in FROZEN_ROOTS)}")
    for name, rs, pr in (("departures", runs, problems), ("frozen", frozen, fproblems)):
        ids = sorted({(r.get("git_commit"), r.get("git_dirty"), r.get("source_sha256")) for r, _, _ in rs.values()},
                     key=str)
        w(f"  {name}: runs {len(rs)}; integrity problems {len(pr)}; (commit, dirty, source): {ids}")
        for p in pr:
            w("    PROBLEM " + p)
        rule = sum(int(rows[k]["n_rule_priced_bids"]) for _, rows, _ in rs.values() for k in rows)
        w(f"    bids priced by the cost + 5% rule: {rule}")
    cells = sorted({k[1:3] for k in runs})
    incomplete = [(c, sorted(set(SEEDS) - {k[3] for k in runs if k[1:3] == c})) for c in cells]
    incomplete = [(c, m) for c, m in incomplete if m]
    w(f"  departures cells {len(cells)}; cells missing seeds: {incomplete if incomplete else 'none'}")
    viol = [(key, v) for key, r in sorted(runs.items()) for v in n4_violations(r)]
    nfloor = sum(1 for r in runs.values() for k, ev in r[2].items() for sid in ("A", "B")
                 if org_of(ev, sid) in FLOOR_ORGS and (ev.get("analyst_briefings") or {}).get(sid))
    w(f"  floor messages in which the bidder still reads a line beginning with the floor label: {len(viol)} "
      f"of {nfloor}")
    for key, (k, sid, line) in viol[:20]:
        w(f"    {key} event {k} {sid}: {line[:100]!r}")

    # The gate
    gate = per_seed(runs, SEEDS, INSTS, gate_fn(runs))
    p_gate = fp.sign_flip_p([v for _, v in gate]) if gate else float("nan")
    passed = bool(gate) and p_gate < 0.05 and np.mean([v for _, v in gate]) < 0
    w("\nGATE: event-1 final-bid margin, conflict minus team, homogeneous B and F cells, institutions pooled.")
    w(f"  {describe(gate)}  p {p_gate:.4f}  {'PASSED' if passed else 'FAILED'}")
    w(f"      per seed: {per_seed_line(gate)}")

    # The primary family
    w("\nPRIMARY FAMILY (events 2-6, institutions pooled; exact two-sided sign-flip; Holm over four)"
      + ("" if passed else " -- the gate failed: estimates only, read as uninformative"))
    fam = [(n, pairs) for n, pairs in h_family(runs, LATE) if pairs]
    ps = [fp.sign_flip_p([v for _, v in pairs]) for _, pairs in fam]
    for (name, pairs), p, adj in zip(fam, ps, fp.holm(ps) if ps else []):
        mean = np.mean([v for _, v in pairs])
        verdict = ("> 0" if adj < 0.05 and mean > 0 else "< 0" if adj < 0.05 else "~ 0") if passed else "n/a"
        w(f"  {name:<50} {describe(pairs)}  p {p:.4f}  Holm p {adj:.4f}  [{verdict}]")
        w(f"      per seed: {per_seed_line(pairs)}")
    if len(fam) < 4:
        w(f"  (only {len(fam)} of the four statistics have data; Holm is over those)")

    # The concession family
    w("\nCONCESSION FAMILY (buyer's agent, events 2-6; exact two-sided sign-flip; Holm over three).")
    conc = []
    for name, rs, a, b in (("C1 O - U (frozen code)", frozen, "organized", "single_a"),
                           ("C2 O - O_pass (frozen code)", frozen, "organized", "organized_pass"),
                           ("C3 B - F (departures code, team)", runs, org("B", False), org("F", False))):
        conc.append((name, per_seed(rs, SEEDS, ("buyer_agent",), diff(rs, "cut", LATE, a, b))))
    conc = [(n, pairs) for n, pairs in conc if pairs]
    ps = [fp.sign_flip_p([v for _, v in pairs]) for _, pairs in conc]
    for (name, pairs), p, adj in zip(conc, ps, fp.holm(ps) if ps else []):
        w(f"  {name:<36} {describe(pairs)}  p {p:.4f}  Holm p {adj:.4f}")
        w(f"      per seed: {per_seed_line(pairs)}")
    if len(conc) < 3:
        w(f"  (only {len(conc)} of the three contrasts have data; Holm is over those)")

    # Secondary estimates
    w("\nSECONDARY (estimates only).")
    for tag, events in W.items():
        w(f"  [{tag}]")
        w("    The primary statistics in this window, institutions pooled and per institution:")
        for insts in (INSTS, ("mediator",), ("buyer_agent",)):
            for name, pairs in h_family(runs, events, insts):
                w(f"      {'+'.join(insts):<21} {name:<50} {describe(pairs)}")
        w("    Team benchmark for the channel, B - F under team objectives (homogeneous):")
        for stat in ("profit", "margin", "buyer", "joint"):
            w(f"      {stat:<10} {describe(per_seed(runs, SEEDS, INSTS, diff(runs, stat, events, org('B', False), org('F', False))))}")
        w(f"      mixed B:F gap, team (firm profit) {describe(per_seed(runs, SEEDS, INSTS, mixed(runs, org('B', False), org('F', False), events)))}")
        w("    Agency costs, conflict minus team (homogeneous):")
        for n in ("F", "B", "Floor"):
            for stat in ("profit", "margin", "buyer", "joint", "below_cost", "loss_win"):
                w(f"      {n + '_conf - ' + n + '_team':<22} {stat:<10} {describe(per_seed(runs, SEEDS, INSTS, diff(runs, stat, events, org(n, True), org(n, False))))}")
        w("    Surplus DiDs (as H2 and H4):")
        for stat in ("buyer", "joint", "profit"):
            w(f"      H2-type {stat:<7} {describe(per_seed(runs, SEEDS, INSTS, did(runs, stat, events, org('B', True), org('F', True), org('B', False), org('F', False))))}")
            w(f"      H4-type {stat:<7} {describe(per_seed(runs, SEEDS, INSTS, did(runs, stat, events, org('Floor', True), org('B', True), org('Floor', False), org('B', False))))}")
        w("    Mixed gaps by position (xa: X is supplier A; xb: X is supplier B):")
        for x, y in ((org("B", True), org("F", True)), (org("B", False), org("F", False)),
                     (org("Floor", True), org("B", True))):
            for pos in (("xa",), ("xb",)):
                w(f"      {x + ':' + y:<34} {pos[0]} {describe(per_seed(runs, SEEDS, INSTS, mixed(runs, x, y, events, pos)))}")
        w("    Payoffs of choosing B against a rival's organization, per objective:")
        for conf in (False, True):
            b, f = org("B", conf), org("F", conf)

            def pay(x_in_mixed, homog_label):
                def fn(i, s):
                    ra = runs.get(("persistent", f"{b}_vs_{f}_xa", i, s))
                    rb = runs.get(("persistent", f"{b}_vs_{f}_xb", i, s))
                    rh = homog(runs, homog_label, i, s)
                    if not (ra and rb and rh):
                        return None
                    return firm_profit_in_mixed(ra, rb, events, x_in_mixed) - pa.market(rh[1], "supplier_profit", events) / 2
                return fn
            w(f"      {'conflict' if conf else 'team':<9} pi(B|F) - pi(F|F) {describe(per_seed(runs, SEEDS, INSTS, pay('x', f)))}")

            def pay2(i, s):
                ra = runs.get(("persistent", f"{b}_vs_{f}_xa", i, s))
                rb = runs.get(("persistent", f"{b}_vs_{f}_xb", i, s))
                rh = homog(runs, b, i, s)
                if not (ra and rb and rh):
                    return None
                return pa.market(rh[1], "supplier_profit", events) / 2 - firm_profit_in_mixed(ra, rb, events, "y")
            w(f"      {'conflict' if conf else 'team':<9} pi(B|B) - pi(F|B) {describe(per_seed(runs, SEEDS, INSTS, pay2))}")
        w("    Tier 2 (exploratory), homogeneous:")
        for n in ("Cap", "Cost"):
            for stat in ("profit", "margin", "buyer", "joint"):
                for conf in (False, True):
                    w(f"      {n + ' - B ' + ('conf' if conf else 'team'):<14} {stat:<7} {describe(per_seed(runs, SEEDS, INSTS, diff(runs, stat, events, org(n, conf), org('B', conf))))}")
                w(f"      {'(' + n + ' - B) DiD':<14} {stat:<7} {describe(per_seed(runs, SEEDS, INSTS, did(runs, stat, events, org(n, True), org('B', True), org(n, False), org('B', False))))}")

    w("\nLEVELS by homogeneous cell, events 2-6 (mean over seeds of the institution-pooled value):")
    for n in ORG:
        for conf in (False, True):
            label = org(n, conf)
            vals = {}
            for stat in STATS:
                pairs = per_seed(runs, SEEDS, INSTS, lambda i, s, label=label, stat=stat:
                                 STATS[stat](homog(runs, label, i, s), LATE) if homog(runs, label, i, s) else None)
                vals[stat] = np.mean([v for _, v in pairs]) if pairs else float("nan")
            w(f"  {label:<20} " + "  ".join(f"{k} {v:.3f}" for k, v in vals.items()))
    # Messages
    homog_analyst = [org(n, c) for n in ("B", "Floor", "Cap", "Cost") for c in (False, True)]
    mixed_cells = [f"{org('B', c)}_vs_{org('F', c)}_{p}" for c in (False, True) for p in ("xa", "xb")] + \
                  [f"{org('Floor', True)}_vs_{org('B', True)}_{p}" for p in ("xa", "xb")]
    w("\nMESSAGE CODING (analyst-written messages, events 2-6; per-seed means over institutions, suppliers and "
      "events; the mean over seeds is shown with the usual statistics).")
    for label, only in [(c, None) for c in homog_analyst] + [(c, o) for c in mixed_cells
                                                           for o in sorted(ANALYST_ORGS) if c.startswith(o + "_vs_")
                                                           or c.split("_vs_")[1].rsplit("_", 1)[0] == o]:
        w(f"  {label}" + (f" (the {only} firm)" if only else ""))
        for key in ("words", "dollar", "lower_bound", "relayed", "adherence", "cost_quoted"):
            w(f"    {key:<12} {describe(cell_mean(runs, label, INSTS, LATE, message_codes, key, only))}")
    w("  Message length by event (mean words over seeds, institutions and suppliers):")
    for label in homog_analyst:
        by = [np.mean([v for _, v in cell_mean(runs, label, INSTS, range(k, k + 1), message_codes, "words")] or [np.nan])
              for k in LATE]
        w(f"    {label:<20} " + "  ".join(f"e{k} {v:.1f}" for k, v in zip(LATE, by)))

    w("\nFLOOR USE (floor organizations, events 2-6).")
    for label in (org("Floor", False), org("Floor", True)) + tuple(c for c in mixed_cells if c.startswith("floor")):
        w(f"  {label}")
        for key in ("set_pos", "set_zero", "none", "malformed", "x", "analyst_reasked", "returned", "raised",
                    "first_gap"):
            w(f"    {key:<15} {describe(cell_mean(runs, label, INSTS, LATE, floor_codes, key, None))}")
    xs = [floor_codes(r, k, sid)["x"] for key_, r in runs.items() if key_[1] in FLOOR_ORGS for k in LATE
          if k in r[2] for sid in ("A", "B") if floor_codes(r, k, sid) and not np.isnan(floor_codes(r, k, sid)["x"])]
    w(f"  minimum profit when set, homogeneous floor cells pooled: n {len(xs)}, mean "
      f"{np.mean(xs) if xs else float('nan'):.3f}, median {np.median(xs) if xs else float('nan'):.3f}")

    w("\nCAP (brief_cap cells, events 2-6).")
    for label in (org("Cap", False), org("Cap", True)):
        for key in ("reasked", "cut", "original_words", "final_words"):
            w(f"  {label:<20} {key:<15} {describe(cell_mean(runs, label, INSTS, LATE, cap_codes, key))}")

    w("\nDO THE ANALYSTS PUSH FOR MARGIN? (plan: (a) or (b); a 95% interval wholly above 0 counts)")
    a = seed_diff(cell_mean(runs, org("Floor", True), INSTS, LATE, floor_codes, "set_pos"),
                  cell_mean(runs, org("Floor", False), INSTS, LATE, floor_codes, "set_pos"))
    b = seed_diff(cell_mean(runs, org("B", True), INSTS, LATE, message_codes, "lower_bound"),
                  cell_mean(runs, org("B", False), INSTS, LATE, message_codes, "lower_bound"))
    for name, pairs in (("(a) positive floor share, Floor_conf - Floor_team", a),
                        ("(b) lower-bound language share, B_conf - B_team", b)):
        d = np.array([v for _, v in pairs])
        lo = d.mean() - T975_19 * d.std(ddof=1) / np.sqrt(len(d)) if len(d) == 20 else float("nan")
        w(f"  {name:<52} {describe(pairs)}  {'PUSH' if lo > 0 else 'no'}")

    # The verbatim sample (plan: three messages per analyst cell under each objective, fixed RNG seed 2026)
    rng = random.Random(2026)
    sample = ["# Verbatim sample of analyst messages (departures stage)", ""]
    for label in homog_analyst:
        pool = [(s, i, k, sid) for (env, lab, i, s), r in sorted(runs.items()) if lab == label
                for k in LATE if k in r[2] for sid in ("A", "B")]
        for s, i, k, sid in rng.sample(pool, min(3, len(pool))):
            r = runs[("persistent", label, i, s)]
            ev = r[2][k]
            sample += [f"## {label}, {i}, seed {s}, event {k}, supplier {sid}", "", "```",
                       ev["analyst_briefings"][sid], "```", "",
                       "Bids that followed (price, delivery, tier): "
                       + "; ".join(f"r{b['round']} ${b['price']:.2f} {b['delivery_weeks']}wk T{b['quality_tier']}"
                                   for b in ev["bids"][sid]), ""]
    OUT.with_name(OUT.stem + "_message_sample.md").write_text("\n".join(sample) + "\n", encoding="utf-8",
                                                               newline="\n")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
