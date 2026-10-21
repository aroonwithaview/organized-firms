# Reproducing the proposal's results

This file says where each number in Table 1, Figure 2, Section 3 and the appendix of the proposal comes from, how to recompute it from the recorded runs, and which command lines make the runs. The README has an overview and a list of the proposal's terms and the names that the code uses for them.

## 1. What is in the package

| Path | Content |
|---|---|
| `data/runs/` | The recorded runs behind the results: 8 folders, 1,490 runs, 83.1 MB (section 4 lists them). Each run is a folder with `run.json`, `events.jsonl` and `summary.csv`. The `logs/` subfolders of the original output folders are not included, because the analysis does not read them. |
| `analysis/` | The scripts that compute the numbers. `make_tables.py` runs all of them. |
| `analysis/results/` | The output of the scripts. The line numbers below refer to these files. |
| `examples/demo_run/` | One recorded run, readable without a model call (see the README). |
| `orgsim/`, `scripts/run_demo.py`, `tests/` | The environment, the runner and the tests. |

## 2. Recompute the tables

```bash
pip install -r requirements.txt        # the analysis needs numpy only
python analysis/make_tables.py
```

This runs 14 steps in under a minute and makes no model call. Each step reads the runs in `data/runs/` and writes to `analysis/results/`. The script compares each file that it writes with the file that was there before and prints whether they are identical, so a run also checks that the shipped results are what the code writes. Each script also runs alone, for example `python analysis/pooled40_numbers.py`. `pilot17_analysis.py`, `followup17_primary.py`, `departures18_analysis.py`, `replication18_analysis.py` and `confirm17_analysis.py` take the folders of runs as arguments (usage is in each docstring), so a new set of runs in `outputs/` can be analysed with the same code; the other scripts read fixed folders of `data/runs/`.

The first lines of the outputs give the number of runs read and the number of integrity problems found: a run that is not complete, a run found in two folders, or a run with a failed or truncated bidder request. The number of problems is 0 in every output, and the commit and code digest of the runs are printed beside it.

## 3. Where each number comes from

### 3.1 Samples and conventions

- The unit is the seed: one persistent market, which is one contract and one pair of cost types repeated at each of 6 events. Every value is first computed for a seed.
- Within a seed the values of the two positions (`xa`, `xb`) are pooled event by event, and the values of the two intermediaries (`mediator`, `buyer_agent`) are then averaged. The statistics across seeds are the mean, the standard error, and a 95% interval from the t distribution (19 degrees of freedom for 20 seeds, 39 for 40).
- Events 2 to 6 are the events with a record; event 1 has none and serves as a placebo.
- Sample P (pilot): persistent environment, seeds 3 to 22 (`pilot17`: 3, 7, 9, 12, 19; `followup17`: the other 15), both intermediaries, events 2 to 6.
- Sample M (mixed markets): persistent environment, seeds 23 to 62 (`departures18`: 23 to 42; `replication18` and `replication18_team`: 43 to 62), markets of a briefed firm (B) against a forwarding firm (F), both positions, both intermediaries, events 2 to 6. The conflict cell is `brief_revenue:forward_revenue` and the common-objective cell is `brief:forward`. A count "of 800" is 40 seeds x 2 intermediaries x 2 positions x 5 events, one firm's final bid in each.
- Sample H (homogeneous markets): persistent environment, seeds 23 to 42 (`departures18`), both firms of the same organization, both intermediaries, events 2 to 6.
- In the output files, `team` is the common objective and `conflict` is the revenue-seeking bidder.

### 3.2 Table 1

| Entry | Number | Script | Data | Sample and definition | Output file : line |
|---|---|---|---|---|---|
| Record | slope 0.87 against 0.05 | `pilot17_analysis.py` on `pilot17;followup17;random17` (step 2 of `make_tables.py`), section E | `pilot17`, `followup17` (`random17` is read by the same call and not used here) | Sample P, `single_a` and `single_norecord`, 400 points each (20 seeds x 2 intermediaries x 2 suppliers x 5 events). Pooled OLS slope of the round-1 markup on the supplier's final markup at its most recent earlier event with the same own cost type | `followup17_pooled_results.txt` : 96 (0.874) and 98 (0.047). Line 97 is `single_b`, a second run of the same firm type: 0.881 |
| Objective | 91 against 8, of 800 | `pooled40_numbers.py`, section A | `departures18`, `replication18`, `replication18_team` | Sample M, F firm's final bids priced below its own cost, conflict against common objective | `pooled40_results.txt` : 6 (91 of 800) and 4 (8 of 800) |
| Objective | +$6.74 and +$6.43 | `pooled40_numbers.py`, section C | same | Sample M, F firm's cost at its final terms and its final price, conflict minus common objective, mean over 40 seeds | `pooled40_results.txt` : 16 (+6.7441, interval 5.20 to 8.29) and 18 (+6.4253, interval 4.87 to 7.98) |
| Objective | joint surplus 71.9 against 76.1 | `departures18_analysis.py`, LEVELS | `departures18` (and `confirm17_concession`, which the script also reads for another family of tests) | Sample H, cells `forward_revenue` and `forward`, joint surplus per event | `departures18_results.txt` : 286 (71.912) and 285 (76.075) |
| Objective | efficient: 84.3 | `efficient_surplus.py` | `departures18` | Sample H, `outcome.efficient_surplus` of each event | `efficient_surplus.txt` : 2 and 3 (84.335) |
| Briefing | 6 against 91, of 800 | `pooled40_numbers.py`, section A | as for Objective | Sample M, final bids below own cost, B firm and F firm under conflict | `pooled40_results.txt` : 7 (6 of 800) and 6 (91 of 800) |
| Briefing | win share +0.17 (0.10 to 0.24) | `pooled40_numbers.py`, section E | same | Sample M, conflict cell, share of events won by B minus share won by F, per seed | `pooled40_results.txt` : 24 (+0.1725, interval 0.1008 to 0.2442) |
| Briefing | profit +1.50 (0.60 to 2.39); net of the gap at event 1, +1.04 (-0.16 to 2.23) | `pooled40_numbers.py`, section F | same | Sample M, conflict cell, B firm's profit per event minus F firm's, per seed. The net figure is the events 2 to 6 gap minus the event 1 gap, per seed | `pooled40_results.txt` : 28 (+1.4978, 0.6049 to 2.3908), 29 (+1.0358, -0.1564 to 2.2280); the gap at event 1 is on 27 (+0.4620) |
| Formal authority | no detectable effect on profit or markups | `departures18_analysis.py`, primary family | `departures18` | 20 seeds (23 to 42), events 2 to 6. H3: profit gap of the `floor_revenue` firm against the `brief_revenue` firm in mixed markets. H4: difference in differences of the final-bid markup, (`floor_revenue` minus `brief_revenue`) minus (`floor` minus `brief`). Exact sign-flip tests, Holm correction over four | `departures18_results.txt` : 18 (H3, mean -0.062, Holm p 1.0000) and 20 (H4, mean -0.491, Holm p 1.0000) |
| Two agents | -1.14 to +0.72, none significant | `followup17_primary.py`, primary family | `pilot17`, `followup17` | Sample P, profit per event. P2: market profit of `organized` minus the mean of `single_a` and `single_b`. P3: profit gap of an `organized` firm against a `single` firm in mixed markets, both positions. Exact sign-flip tests, Holm correction over six | `followup17_primary_results.txt` : 10 (P2, mediator, -0.267), 12 (P3, mediator, -1.143), 16 (P2, buyer's agent, +0.717), 18 (P3, buyer's agent, +0.183). The smallest unadjusted p is 0.0566 on line 12 (Holm p 0.2829) |

### 3.3 Figure 2

| Panel | Number | Script | Sample and definition | Output file : line |
|---|---|---|---|---|
| A | 8, 0, 91, 6 (of 800 each) | `pooled40_numbers.py`, section A | Sample M. Profit-seeking forwarded (F, common), profit-seeking briefed (B, common), revenue-seeking forwarded (F, conflict), revenue-seeking briefed (B, conflict) | `pooled40_results.txt` : 4, 5, 6, 7 |
| B | 58.4% and 8.0% | `pooled40_numbers.py`, section B | Sample M, conflict cell, the briefed bidder's round-1 price (800 bids). A bid counts when its price equals a dollar figure in its own briefing for that event, or, as the placebo, in the briefing written for the same firm, intermediary, position and event in the next seed of the same 20-seed batch (the last seed uses the first) | `pooled40_results.txt` : 12 (467 of 800 and 64 of 800); the two batches are on 10 and 11 |

### 3.4 Section 3

| Statement | Number | Where |
|---|---|---|
| The record | 0.87 against 0.05 | As in Table 1 |
| The bidder's objective | 91 and 8 of 800; the revenue objective lowers efficiency, mostly through inefficient terms | `pooled40_results.txt` : 6 and 4. In `adversarial_out.txt` (from `analysis/departures18_check/adversarial.py`, sample H, `forward_revenue` minus `forward`): 32 (joint surplus -4.163), 33 (loss from the wrong winner, +0.700), 34 (loss from inefficient terms, +3.463) |
| The briefing | 6 of 800 against 91 of 800; 58.4% against 8.0% | As in Table 1 and Figure 2 |
| The briefing | the sentence quoted from one briefing | `departures18_results_message_sample.md` : 192. The file holds three messages sampled at random (seed 2026 in `departures18_analysis.py`) from each analyst cell; the message is the one headed on line 165 (cell `brief_revenue`, buyer's agent, seed 28, event 3, supplier A). The same text is in `examples/demo_run/events.jsonl`, event 3, `analyst_briefings`, supplier A |
| Formal authority | no detectable effect | As in Table 1 |
| Formal authority | analysts usually set a minimum, a low one: median $2.00; one bid in 2,160 sent back | `departures18_results.txt` : 419 and 429 (share of events with a positive minimum: 0.748 under the common objective, 0.890 under conflict) and 458 (720 minimums set, both objectives pooled, median 2.000). `departures18_floor_bunching.txt` : 4 (2,160 bids under a set minimum, $0 included; 1 sent back) |
| Wins and profit | briefed firm wins more and earns more; it earns +0.46 more at event 1; the increase after event 1, +1.04, has an interval that includes zero | As in Table 1 (`pooled40_results.txt` : 24, 28, 27, 29) |
| Wins and profit | with a common objective, profit does not differ detectably between a two-agent firm and a single agent with the record | As the last row of Table 1 |

### 3.5 Appendix

The entries follow the appendix in order. Line numbers after an output file name refer to `analysis/results/`. Line numbers after a code file refer to the file in this package. Where an output prints a rounded value, the entry says what it prints. The statements of Appendix C are plans, not results. Two entries are marked "Not reproduced", because the package cannot recompute them: the cost figures of A.7 (they cover runs that are not in the package) and the price ratio of C.5 (it rests on list prices).

| Statement | Number | Where |
|---|---|---|
| A.2: eight combinations of contract and cost types | 8 | `market_combinations.txt` : 1. The script lists the combinations and their probabilities from the code of the draw (four at 3/16 and four at 1/16) and counts the combinations drawn for seeds 3 to 62 and 0 to 9999 |
| A.2: cost of a bid, `c0 + gd (8 - d)^2 + gq q^2`, with `(c0, gd, gq)` = (85, 0.5, 1.8) for a speed specialist and (85, 1.8, 0.4) for a quality specialist | 85, 0.5, 1.8; 85, 1.8, 0.4 | `orgsim/primitives.py` : `SupplierCosts.cost`, lines 210 to 214 (`delivery_max` 8 on line 208); the parameters in `get_fixed_suppliers`, line 369 (speed) and line 373 (quality) |
| A.2: value of a bid, `150 + 60 wd (8 - d)/8 + 50 wq q/5`, and score = value minus price | 150, 60, 50, 8, 5 | `orgsim/primitives.py` : `BuyerPreferences.value`, lines 90 to 96, and `score`, lines 98 to 100; the defaults are on lines 76 (150), 83 (60), 84 (50), 87 (8) and 88 (5). `get_contract_scenarios` sets the base value 150 again on lines 399 and 404 |
| A.2: weights of the two contracts | 0.80 and 0.20; 0.25 and 0.75 | `orgsim/primitives.py` : `get_contract_scenarios`, line 400 (speed priority) and line 405 (quality priority) |
| A.2: delivery 1 to 8 weeks, quality tiers 2 to 5 | 1 to 8; 2 to 5 | `orgsim/llm_agents.py` : `DELIVERY_WEEKS_RANGE` (line 197) and `QUALITY_TIERS` (line 198); `_is_bid` (from line 201) does not accept a bid outside them. The comment on `ContractTerms.quality_tier` is `orgsim/primitives.py` : 28 |
| A.2: each contract has probability 1/2, each supplier's cost type 1/2, and the second supplier's type is the opposite of the first's with probability 3/4 | 1/2, 1/2, 3/4 | `orgsim/sequence.py` : `pre_draw_event_sequence`, `rng.choice(scenarios)` on line 353, `rng.choice(type_choices)` on line 354 and `rng.random() < p_different` on line 356, with `P_DIFFERENT = 0.75` on line 291. The probabilities are stated to the agents in `DRAWN_PERSISTENT`, `orgsim/primitives_prompts.py` : 61 to 65. The frequencies of the eight combinations are in `market_combinations.txt` : 2 to 10 |
| A.2: in the persistent environment one draw is kept for all six events; the random environment redraws at each event and is used in control runs | 6 events | `orgsim/sequence.py` : lines 361 and 362 (`plan[:1] * n_events`). The `environment` field of `run.json` is `random` in `random17`, in `confirm17_memory` and in the random cells of `pilot17` |
| A.2: only the round-3 bid is scored; the highest score wins and is paid its price | round 3 | `orgsim/institutions.py` : `_compute_outcome`, scores of the final bids on lines 52 and 53, winner on line 68, supplier profit on line 95, and the description in the docstring, lines 39 to 47. `run_mediator` and `run_buyer_agent` take the last round's bid as the final bid (docstring of `primitives_rules`, lines 271 and 272) |
| A.2: if every score is negative the buyer awards nothing and all surpluses are zero | 0 | `orgsim/institutions.py` : line 57 (the test) and lines 58 to 65 (the outcome with no winner and zero surpluses). `buyer_walks_away=True` is passed by `run_mediator` (line 461) and `run_buyer_agent` (line 578). A score of exactly 0 is accepted (docstring, line 42) |
| A.2: a tie goes to supplier A | A | `orgsim/institutions.py` : docstring, lines 43 and 44 (the first supplier in the order of `final_bids`), and `max(scores, key=scores.get)` on line 68, which returns the first maximum. The order is A then B: `SUPPLIER_IDS` in `orgsim/sequence.py` : 20 |
| A.2: the intermediary replies after rounds 1 and 2 only | rounds 1 and 2 of 3 | `orgsim/institutions.py` : `run_mediator`, line 414 (`if round_num < num_rounds`) and `run_buyer_agent`, line 552. The `rounds` field of every `run.json` is 3 |
| A.2: what the mediator and the buyer's agent see | none | `orgsim/institutions.py` : `primitives_rules`, lines 266 to 300, which states each sentence of the rules and the code that makes it true |
| A.2: the efficient supplier and terms maximize value minus cost | none | `orgsim/primitives.py` : `compute_efficient_outcome` (line 429) and `compute_max_efficient_surplus` (line 457); see A.5 |

Table A1 (A.3). The first column is the code name; the other columns say where the code or the role text defines the row.

| Code name | The bidder reads from event 2 | The analyst | Where |
|---|---|---|---|
| `single` | its own record | none | `orgsim/organization.py` : comment line 15, `build_primitives_bidder_context` line 186; role text `SINGLE_ROLE`, `orgsim/primitives_prompts.py` : 73 |
| `single_norecord` | nothing about past events | none | `orgsim/organization.py` : lines 16 and 187; `NORECORD_ROLE`, `orgsim/primitives_prompts.py` : 76 |
| `organized` | the analyst's briefing | writes the briefing | `orgsim/organization.py` : lines 17 and 189; `ORGANIZED_ROLE` (line 77) and `ANALYST_ROLE` (line 87) in `orgsim/primitives_prompts.py` |
| `organized_pass` | the full record | none; the code passes it on | `orgsim/organization.py` : lines 18, 19 and 191, and `forwards_the_record` (line 63); `PASS_ROLE`, `orgsim/primitives_prompts.py` : 83 |
| `brief` | the analyst's briefing | writes the briefing | `orgsim/organization.py` : lines 33 and 193; `DEPARTURE_ROLE`, `orgsim/primitives_prompts.py` : 96 |
| `forward` | the full record | none; the code passes it on | `orgsim/organization.py` : lines 34 and 199, and `forwards_the_record` (line 63) |
| `floor` | the briefing and a notice of the minimum | as `brief`, and sets a minimum profit per contract or none; the code returns a bid priced below cost plus the minimum once, then raises its price to that sum | `orgsim/organization.py` : lines 35 and 196; `orgsim/departures.py` : docstring lines 3 to 10 and `floor_briefing` (line 189); `PrimitivesSupplierAgent`, `orgsim/llm_agents.py` : 737 (docstring, lines 747 to 752); `FLOOR_ROLE` and `ANALYST_FLOOR_ROLE`, `orgsim/primitives_prompts.py` : 111 and 123 |
| `brief_cap` | the briefing | as `brief`, with at most 50 words; asked again once, then cut | `orgsim/organization.py` : lines 36, 43 (`BRIEFING_WORD_LIMIT = 50`) and 194; `orgsim/departures.py` : docstring lines 12 to 14 and `capped_briefing` (line 94); `ANALYST_CAP_ROLE`, `orgsim/primitives_prompts.py` : 131 |
| `brief_cost` | the briefing | as `brief`, and shown the bidder's cost figures | `orgsim/organization.py` : lines 37 and 193; `ANALYST_COST_ROLE`, `orgsim/primitives_prompts.py` : 103 |

| Statement | Number | Where |
|---|---|---|
| A.3: the 50-word cap, with one re-ask | 50 words; one re-ask | `orgsim/organization.py` : line 43; `orgsim/departures.py` : lines 12 to 14 and `capped_briefing` (line 94, re-ask text `CAP_REASK` on line 71). The runs record the words before and after in `departures18_results.txt` : 461 to 468 |
| A.3: the types with a `_revenue` variant | single, organized, brief, forward, floor, brief_cap, brief_cost | `orgsim/organization.py` : `PRIMITIVES_REVENUE_ORGANIZATIONS` (line 27: `single_revenue`, `organized_revenue`) and `DEPARTURE_REVENUE_ORGANIZATIONS` (line 44: the five bases of line 41 with the suffix). `single_norecord` and `organized_pass` are in neither |
| A.3: the bidders of `brief`, `forward`, `brief_cap` and `brief_cost` read one role text from the first event; `organized` and `organized_pass` use other role texts | one role text | `orgsim/primitives_prompts.py` : `DEPARTURE_ROLE` (lines 96 to 101) assigned to the four types and their `_revenue` variants on lines 134 and 135; `ORGANIZED_ROLE` and `PASS_ROLE` in the dictionary on lines 132 and 133. `floor` has its own role text, `FLOOR_ROLE`, line 136 |
| A.3: the objective told to the members | profit over the six events; revenue for the bidder under conflict | `orgsim/primitives_prompts.py` : `TEAM_OBJECTIVE` (line 35), `REVENUE_OWNER` (line 37), `REVENUE_KNOWLEDGE` (line 38), `REVENUE_BIDDER_OBJECTIVE` (line 144), `ANALYST_PROFIT_OBJECTIVE` (line 146) and `REVENUE_ANALYST_OBJECTIVES` (line 147) |
| A.4: the pilot seeds | 3, 7, 9, 12, 19 | `analysis/followup17_primary.py` : `PILOT_SEEDS`, line 24; the `seeds` field of the `run.json` files of `pilot17`; `followup17_primary_results.txt` : 2 lists the other 15 seeds |
| A.4: the intermediaries of each set of Table A2 | both, or one | The `institutions` field of each `run.json`; section 4.2 gives it for each folder. `confirm17_memory` has the mediator only and `confirm17_concession` the buyer's agent only |
| A.4: commit of each set, runs complete, no uncommitted changes | `9f1cf25`; `653e656` | The `git_commit`, `git_dirty` and `status` fields of each `run.json` (section 4.2). The outputs print the commit, the dirty flag and the code digest: `pilot17_results.txt` : 1; `followup17_pooled_results.txt` : 1; `followup17_primary_results.txt` : 3; `departures18_results.txt` : 2 and 4; `replication18_results.txt` : 2 and 4; `confirm17_results.txt` : 3 (`confirm17_memory` and `confirm17_concession`) |
| A.4: the 40-seed results use the two mixed markets run on all 40 seeds | `brief:forward`, `brief_revenue:forward_revenue` | `analysis/pooled40_numbers.py` : `ROOTS`, line 28; `analysis/panel_reanalysis.py` : `CELLS`, line 23 |
| A.4: 100 one-shot second-price auctions, runs of earlier designs and test runs | 100 | Not in the package. No result of the package uses them |
| A.5: a final bid is the round-3 bid; markup is (price - cost)/cost at the bid's own terms | none | `orgsim/institutions.py` : `_true_costs_and_inflation`, lines 6 to 22 (`all_inflation_rates`); for the round-1 bid, `markup_r1` in `analysis/pilot17_analysis.py` : 105 to 110 |
| A.5: the efficient benchmark E is the largest value minus cost over the two suppliers; the grid is 1.0 to 8.0 weeks in steps of 0.1 and the four tiers | 1.0 to 8.0; 0.1; 4 tiers | `orgsim/primitives.py` : `compute_efficient_outcome`, `range(2, 6)` on line 441 and `range(10, 81)` on line 442 (delivery tenths); `compute_max_efficient_surplus`, line 457. Output: `efficient_surplus.txt` : 2 and 3 (84.335) |
| A.5: the loss from the wrong winner, `E - Bw`, and from wrong terms, `Bw - J` | +0.700 and +3.463 | `analysis/departures18_check/adversarial.py` : lines 155 and 156; `adversarial_out.txt` : 33 and 34 |
| A.5: an opening price matches its briefing if it equals, to the cent, a dollar figure of two or three digits in it | two or three digits | `analysis/panel_reanalysis.py` : `PRICE`, line 26 (`\$\s?(\d{2,3}(?:\.\d{1,2})?)`), figures rounded to the cent on line 129 and the price on line 131 (function `anchoring`, line 112) |
| A.5: the placebo uses the briefing of the next seed; seed 42 is followed by 23 and 62 by 43 | 42 to 23; 62 to 43 | `analysis/panel_reanalysis.py` : `other = seeds[(i + 1) % len(seeds)]`, line 117, applied to the lists of `BATCHES` in `analysis/pooled40_numbers.py` : 29 (23 to 42 and 43 to 62) |
| A.5: the slope in Table 1 uses 400 supplier-events | 400 | `analysis/pilot17_analysis.py` : section E, lines 230 to 263 (a supplier-event is one supplier at one of events 2 to 6 in one run: 20 seeds x 2 intermediaries x 2 suppliers x 5 events); `followup17_pooled_results.txt` : 96 and 98 print `n=400` |
| A.6: an exact two-sided sign-flip test over the 2^n sign patterns, with n = 20 and, for one test, n = 15 | 2^20; 2^15 | `analysis/followup17_primary.py` : `sign_flip_p`, lines 28 to 38 (the share of the `1 << n` patterns whose absolute mean is at least the observed one). n = 20 in every test of `followup17_primary_results.txt` : 8 to 18 and `departures18_results.txt` : 10 to 28; n = 15 in `followup17_primary_results.txt` : 22 |
| A.7: model, default sampling, no tools, adaptive reasoning, cap of 16,000 tokens per reply | `claude-haiku-5-5`; 16,000 | The `model`, `backend`, `base_url`, `default_sampling`, `no_tools` and `max_tokens` fields of every `run.json`. In `orgsim/llm.py` : lines 147 and 148 drop `extra_body` for a hosted endpoint (the thinking switch is sent in `extra_body`, line 83, so none is sent), lines 149 to 156 drop the sampling settings under `default_sampling`, and lines 157 and 158 set `max_tokens`. That Claude Haiku 5.5 reasons adaptively by default is stated in `scripts/run_demo.py` : 216 and 217; no recorded field sets it |
| A.7: cost of a six-event market: mean $0.052 over 1,824 sequences, about $0.059 under the mediator and $0.045 under the buyer's agent; 1,918 runs cost $94.95 | $0.052; 1,824; $0.059; $0.045; 1,918; $94.95 | Not reproduced. These figures cover runs that are not in the package: it holds 1,490 runs. Section 3.6 gives the cost of the 570 runs that it does hold in `pilot17`, `followup17` and `random17`, which is a different sample |
| B.1: single agents with a record open with a lower bid than those without | no number in the text | `followup17_pooled_results.txt` : 190 and 191 (mediator) and 196 and 197 (buyer's agent), section K: round-1 markup of a `single_norecord` firm (X) and of a `single` firm (U) in mixed markets, sample P, events 2 to 6. Without a record: 14.99% and 14.75%. With a record: 9.65% and 6.35%. U opens lower in 20 of 20 seeds under each intermediary: `opening_lower_with_record.txt` : 4, 6 and 8 (step 13 of `make_tables.py`) |
| B.2: $6.74 and $6.43 | +6.74, +6.43 | As in Table 1 |
| B.2: 76.1 to 71.9, against 84.3; 3.46 of 4.16 | 76.075, 71.912, 84.335, 3.463, 4.163 | `departures18_results.txt` : 285 and 286; `efficient_surplus.txt` : 2 and 3; `adversarial_out.txt` : 32 to 34 |
| B.3: B bids below cost less often than F in 32 of 40 seeds, more often in 2 | 32 and 2 (6 seeds equal) | `pooled40_results.txt` : 21 (sample M, conflict cell, per seed) |
| B.3: at event 1 the two bidders do not differ detectably | 8 and 11 of 160; per-seed difference -0.0187 (interval -0.0645 to +0.0270) | `pooled40_results.txt` : 32, 33, 34 |
| B.3: under a common objective a briefed bidder never bids below cost | 0 of 800 | `pooled40_results.txt` : 5 |
| B.3: winning bids of a revenue-seeking bidder cost $3.70 more and are priced $5.25 higher | +3.698 and +5.254 | `adversarial_out.txt` : 28 and 27 (sample H, winning bids only, `brief_revenue` minus `brief`, 20 seeds) |
| B.4: win share +0.17, profit +1.50, event 1 +0.46, net +1.04 | as in Table 1 | `pooled40_results.txt` : 24, 28, 27, 29 |
| B.4: four estimates between -1.14 and +0.72 | as in Table 1 | `followup17_primary_results.txt` : 10, 12, 16, 18 |
| C.2: market profit per event in the pilot and follow-up runs in the persistent environment | about 11 to 14 under the mediator; 3 to 4 under the buyer's agent | `followup17_pooled_results.txt`, section A: lines 5 to 9 (persistent environment, mediator, five arms of 20 seeds: 11.23 to 13.85) and lines 10 to 14 (persistent environment, buyer's agent: 3.08 to 4.05). The random-environment rows are lower under the mediator (lines 17 to 21: 7.71 to 10.64) |
| C.5: Sonnet 5.5 costs about 20 times Haiku's price per token | 20 | Not reproduced. The ratio comes from the providers' list prices, which are not in the package. `PRICE_IN` and `PRICE_OUT` in `analysis/pilot17_analysis.py` : 113 are the prices of the model of the runs only |
| C.2 to C.6: the bonus levels and the numbers of seeds and runs | none | These are plans. No script or output of the package computes them |

### 3.6 Other statements that rest on the code or the run files

- Six events of three rounds, one model for every agent (`claude-haiku-5-5`), the primitives prompts, and the persistent environment are recorded in the `events`, `rounds`, `model`, `prompts` and `environment` fields of every `run.json`.
- A firm's record holds its own bids, the intermediary's replies and its own results, and never a rival's bids: `build_primitives_record` in `orgsim/sequence.py`. The buyer awards the contract to the final bid with the highest score (value minus price) and awards none if every score is negative: `orgsim/institutions.py`.
- "A market costs a few cents to run" (Section 6): `followup17_pooled_results.txt` : 136 to 167, section H. The spend per six-event market, by arm, is $0.029 to $0.067, and $0.045 on average over the 570 runs of that file. The script prices tokens at $0.10 per million prompt tokens and $0.50 per million completion tokens (`PRICE_IN` and `PRICE_OUT` in `pilot17_analysis.py`); the prices are an input, not a result. The 570 runs are `pilot17`, `followup17` and `random17`. They are not the sample of A.7, which is 1,824 sequences and includes runs that are not in this package, so the mean here ($0.045) and the mean there ($0.052) are means over different runs.

### 3.7 All outputs

| File in `analysis/results/` | Script | Content |
|---|---|---|
| `pilot17_results.txt` | `pilot17_analysis.py` | Descriptive report of `pilot17` alone |
| `followup17_pooled_results.txt` | `pilot17_analysis.py` on three folders | The same report for `pilot17`, `followup17` and `random17` together |
| `followup17_primary_results.txt` | `followup17_primary.py` | Tests on sample P |
| `departures18_results.txt`, `departures18_results_message_sample.md` | `departures18_analysis.py` | Tests and estimates for `departures18`, and a sample of analysts' messages |
| `replication18_results.txt` | `replication18_analysis.py` | The profit gap of the briefed firm in conflict, from `replication18` and `replication18_team` |
| `panel_reanalysis_results.txt` | `panel_reanalysis.py` | Post hoc estimates of mixed-market quantities from `departures18`, `replication18` and `replication18_team` (event-1 margin, shares of final bids below cost, cost and price of the F firm, match of the opening price with the briefing, win share); `pooled40_numbers.py` uses its functions |
| `pooled40_results.txt` | `pooled40_numbers.py` | The same quantities pooled over seeds 23 to 62 |
| `departures18_extra_checks.txt` | `departures18_extra_checks.py` | Event-1 comparisons and leave-out checks for `departures18` |
| `departures18_floor_bunching.txt` | `departures18_floor_bunching.py` | How often bids sit at the minimum profit |
| `adversarial_out.txt` | `departures18_check/adversarial.py` | Decomposition of surplus, winning-bid prices and costs, and robustness checks for `departures18`, written without the other scripts' loader |
| `efficient_surplus.txt` | `efficient_surplus.py` | Efficient and joint surplus by cell |
| `market_combinations.txt` | `market_combinations.py` | The markets of the persistent environment |
| `opening_lower_with_record.txt` | `opening_lower_with_record.py` | In how many seeds a single agent with a record opens lower than one without, in mixed markets of sample P |
| `confirm17_results.txt` | `confirm17_analysis.py` | The planned tests of two earlier findings: profit without a record in the random environment under the mediator (`confirm17_memory`), and the price cut within an event under the buyer's agent (`confirm17_concession`) |

## 4. Command lines that make the runs

### 4.1 One run

Every recorded run is one call of `scripts/run_demo.py`: one intermediary, one organization or one mixed market, one seed. The flags below are the recorded ones (`argv` in `run.json`); the same flags are in all 1,490 runs apart from `--environment`, `--institutions`, `--organizations` or `--mixed`, `--seeds` and `--out`.

```bash
python scripts/run_demo.py --backend openai --hosted --default-sampling --no-tools \
    --base-url https://api.anthropic.com/v1/ --model claude-haiku-5-5 \
    --max-tokens 16000 --max-retries 30 --timeout 300 \
    --prompts primitives --events 6 --rounds 3 --environment persistent \
    --institutions buyer_agent --mixed brief_revenue:forward_revenue --seeds 23 \
    --out outputs/departures18/persistent_brief_revenue_vs_forward_revenue_xa_buyer_agent_s23
```

- The key goes in `OPENAI_API_KEY` (README, "Hosted Claude models").
- The runner's default prompt set is `advice`. The proposal's design uses the primitives-only prompts, so `--prompts primitives` is required. `--environment persistent` is also required: the default is `random`.
- `--no-tools` is required with `--prompts primitives` and a model backend.
- `--default-sampling` sends no sampling settings. The model samples, and the seed does not fix its output, so a new run is a new draw. The seed fixes the contract and the cost types of the market, which are the same for every organization and intermediary.
- The folder name is `<environment>_<label>_<intermediary>_s<seed>`. For `--organizations X` the label is `X`. For `--mixed A:B` the label is `A_vs_B_xa`, and the other position, `--mixed B:A`, is `A_vs_B_xb`. The labels `single_a` and `single_b` are two independent runs of the same command (`--organizations single`): firm types U and U'.
- A job runner launched the runs, one call per folder, and is not part of this repository. The `manifest.json` in each folder of `data/runs/` lists the arguments of every job except the common flags, with its start and end times.

### 4.2 The recorded runs

| Folder | Runs | Code (commit) | Environment | Cells (`--organizations` or `--mixed`) | Seeds | Intermediaries |
|---|---|---|---|---|---|---|
| `pilot17` | 140 | `9f1cf25` | persistent | `single` (twice: `single_a`, `single_b`), `single_norecord`, `organized`, `organized_pass`; mixed `organized:single`, `single:organized`, `single_norecord:single`, `single:single_norecord` | 3, 7, 9, 12, 19 | both |
| `pilot17` | | | persistent | `single_revenue`, `organized_revenue` | same | buyer's agent only |
| `pilot17` | | | random | `single`, `single_norecord`, `organized`, `organized_pass` | same | both |
| `followup17` | 270 | `9f1cf25` | persistent | the cells of the first row of `pilot17` | 4, 5, 6, 8, 10, 11, 13 to 18, 20, 21, 22 | both |
| `random17` | 160 | `9f1cf25` | random | `single` (`single_a`), `single_norecord`, `organized`, `organized_pass` for the seeds of `followup17`; `single` (`single_b`) for seeds 3 to 22 | as stated | both |
| `confirm17_concession` | 60 | `9f1cf25` | persistent | `single` (`single_a`), `organized`, `organized_pass` | 23 to 42 | buyer's agent only |
| `confirm17_memory` | 60 | `9f1cf25` | random | `single` (twice: `single_a`, `single_b`), `single_norecord` | 23 to 42 | mediator only |
| `departures18` | 640 | `653e656` | persistent | `brief`, `forward`, `floor`, `brief_cap`, `brief_cost` and each with the suffix `_revenue`; mixed `brief:forward`, `forward:brief`, `brief_revenue:forward_revenue`, `forward_revenue:brief_revenue`, `floor_revenue:brief_revenue`, `brief_revenue:floor_revenue` | 23 to 42 | both |
| `replication18` | 80 | `653e656` | persistent | mixed `brief_revenue:forward_revenue`, `forward_revenue:brief_revenue` | 43 to 62 | both |
| `replication18_team` | 80 | `653e656` | persistent | mixed `brief:forward`, `forward:brief` | 43 to 62 | both |

The commit of each folder is in the `git_commit` field of its `run.json`, and the digest of its code is in `source_sha256`. The recorded digests are `46258eff7f4a...` for `9f1cf25` and `b4d072575b9f...` for `653e656`. The history of this repository is a single commit that holds the code of `653e656`; `9f1cf25`, which made `pilot17`, `followup17`, `random17`, `confirm17_memory` and `confirm17_concession`, is not included. The code changed between the two commits, so new runs of those folders' cells with the current code are not reruns of the recorded ones. The analysis recomputes every number from the recorded runs with the current code.

### 4.3 A set of runs

The loop below makes `replication18` (80 runs). The other folders differ in the cells, the seeds and the output folder.

```bash
COMMON="--backend openai --hosted --default-sampling --no-tools --base-url https://api.anthropic.com/v1/ \
    --model claude-haiku-5-5 --max-tokens 16000 --max-retries 30 --timeout 300 \
    --prompts primitives --events 6 --rounds 3 --environment persistent"
for inst in mediator buyer_agent; do
  for seed in $(seq 43 62); do
    python scripts/run_demo.py $COMMON --institutions $inst --mixed brief_revenue:forward_revenue --seeds $seed \
        --out outputs/replication18/persistent_brief_revenue_vs_forward_revenue_xa_${inst}_s${seed}
    python scripts/run_demo.py $COMMON --institutions $inst --mixed forward_revenue:brief_revenue --seeds $seed \
        --out outputs/replication18/persistent_brief_revenue_vs_forward_revenue_xb_${inst}_s${seed}
  done
done
```

To analyse new runs with the same code, give the scripts the new folders: for example `python analysis/replication18_analysis.py analysis/results/replication18_results.txt outputs/replication18 outputs/replication18_team`.

## 5. Notes

- The docstrings of the analysis scripts refer to analysis plans that were written before the runs. The plans are not part of this package.
- The scripts keep the names that they had when the results were first computed, which is why the folders and files are named after stages (`pilot17`, `departures18`).
- `analysis/departures18_check/` holds a second loader and a second set of statistics, written separately from the other scripts. `adversarial.py` is run in step 10 of `make_tables.py`, and the numbers in section 3 that rest on it are marked above.
- `efficient_surplus.py` and `market_combinations.py` only read quantities that the runs or the code define: the efficient surplus recorded in each event, and the draw of the market.
