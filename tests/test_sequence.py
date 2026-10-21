import re
import subprocess
import sys
from pathlib import Path

import pytest

from orgsim.llm_agents import RULE_PRICE_TAG
from orgsim.mocks import MockBuyerAgentPlatform, MockPlatformMediator, MockProcurementAnalyst, MockStrategicSupplier
from orgsim.organization import (ANALYST_ORGANIZATIONS, FIRST_EVENT_NOTE, MEMO_HEADER, AnalystBriefing,
                                 organization_of_each)
from orgsim.primitives import ContractTerms, RunOutcome, SupplierBid, get_contract_scenarios
from orgsim.sequence import (SUPPLIER_IDS, build_mediator_cross_event_history, pre_draw_event_sequence,
                             run_sequence)

HISTORY_HEADER = "YOUR PAST PROCUREMENT HISTORY WITH THIS PLATFORM:"
ROOT = Path(__file__).resolve().parents[1]


class Recording(MockStrategicSupplier):
    def __init__(self, seed):
        super().__init__(seed=seed)
        self.contexts = []

    def generate_bid(self, supplier_costs, regime_context, round_num, history):
        self.contexts.append(regime_context)
        return super().generate_bid(supplier_costs, regime_context, round_num, history)


def _run(organization, analysts):
    agents = {sid: Recording(seed=i) for i, sid in enumerate(SUPPLIER_IDS)}
    plan = pre_draw_event_sequence(3, get_contract_scenarios(), seed=7)
    results = run_sequence("Mediator", 3, 2, agents, plan, mediator=MockPlatformMediator(),
                           analysts=analysts, organization=organization)
    return agents, results


def _analysts():
    return {sid: MockProcurementAnalyst() for sid in SUPPLIER_IDS}


def test_single_bidder_sees_raw_history_and_no_memo():
    agents, _ = _run("single", None)
    assert any(HISTORY_HEADER in c for c in agents["A"].contexts)
    assert not any(MEMO_HEADER in c for c in agents["A"].contexts)


@pytest.mark.parametrize("organization", ANALYST_ORGANIZATIONS)
def test_organized_bidder_sees_memo_and_no_raw_history(organization):
    agents, _ = _run(organization, _analysts())
    for sid in SUPPLIER_IDS:
        assert any(MEMO_HEADER in c for c in agents[sid].contexts)
        assert not any(HISTORY_HEADER in c for c in agents[sid].contexts)


def test_first_event_message_is_identical_for_every_organization():
    single, _ = _run("single", None)
    first = single["A"].contexts[0]
    assert first.startswith(FIRST_EVENT_NOTE)
    for organization in ANALYST_ORGANIZATIONS:
        organized, _ = _run(organization, _analysts())
        assert organized["A"].contexts[0] == first, organization


def test_the_two_organized_firms_are_routed_alike():
    # They differ in what the bidder is told (its system prompt), which run_sequence does not see.
    read = {}
    for organization in ANALYST_ORGANIZATIONS:
        agents, results = _run(organization, _analysts())
        read[organization] = ([agents[sid].contexts for sid in SUPPLIER_IDS], [r["analyst_briefings"] for r in results])
    assert read["organized"] == read["organized_guided"]


def test_every_supplier_bids_in_every_event():
    plan = pre_draw_event_sequence(4, get_contract_scenarios(), seed=7)
    for organization, analysts in (("single", None), *((o, _analysts()) for o in ANALYST_ORGANIZATIONS)):
        agents = {sid: Recording(seed=i) for i, sid in enumerate(SUPPLIER_IDS)}
        results = run_sequence("Mediator", 4, 2, agents, plan, mediator=MockPlatformMediator(),
                               analysts=analysts, organization=organization)
        assert len(results) == 4, organization
        for r in results:
            bids = r["outcome"].all_bids
            assert sorted(bids) == SUPPLIER_IDS, organization
            assert all([b.round_num for b in bids[sid]] == [1, 2] for sid in SUPPLIER_IDS), organization


def test_outcome_keeps_the_institution_and_the_result_names_the_contract():
    _, results = _run("single", None)
    assert all(r["outcome"].regime == "Mediator" for r in results)
    assert all(r["contract"] in {"Speed Priority", "Quality Priority"} for r in results)


def test_a_losing_suppliers_record_says_another_supplier_won():
    agents, results = _run("single", None)
    loser = next(sid for sid in SUPPLIER_IDS if sid != results[0]["winner"])
    record = agents[loser].contexts[2]                  # event 2, round 1: the record of event 1
    assert "Result: You LOST. Another supplier won." in record
    assert not any("The other supplier" in c for a in agents.values() for c in a.contexts)


class RecordingAnalyst(MockProcurementAnalyst):
    def __init__(self):
        self.histories = []

    def generate_briefing(self, supplier_id, history_text):
        self.histories.append(history_text)
        return super().generate_briefing(supplier_id, history_text)


@pytest.mark.parametrize("organization", ANALYST_ORGANIZATIONS)
def test_organized_analyst_receives_raw_history_and_bidder_gets_its_exact_memo(organization):
    analysts = {sid: RecordingAnalyst() for sid in SUPPLIER_IDS}
    agents = {sid: Recording(seed=i) for i, sid in enumerate(SUPPLIER_IDS)}
    plan = pre_draw_event_sequence(5, get_contract_scenarios(), seed=7)
    results = run_sequence("Mediator", 5, 2, agents, plan, mediator=MockPlatformMediator(),
                           analysts=analysts, organization=organization)
    assert any(HISTORY_HEADER in h for h in analysts["A"].histories)
    memo = results[3]["analyst_briefings"]["A"]          # event 4: 3 past events
    assert any(MEMO_HEADER + memo in c for c in agents["A"].contexts)
    assert not memo.split("A: ", 1)[-1].startswith("Only")  # mock parsed the runner's history layout


@pytest.mark.parametrize("organization", ANALYST_ORGANIZATIONS)
def test_the_analyst_is_first_called_before_event_2_and_briefs_each_supplier_after_that(organization):
    analysts = {sid: RecordingAnalyst() for sid in SUPPLIER_IDS}
    agents = {sid: Recording(seed=i) for i, sid in enumerate(SUPPLIER_IDS)}
    plan = pre_draw_event_sequence(4, get_contract_scenarios(), seed=7)
    results = run_sequence("Mediator", 4, 2, agents, plan, mediator=MockPlatformMediator(),
                           analysts=analysts, organization=organization)
    assert results[0]["analyst_briefings"] == {}                          # no record yet: no briefing
    assert [sorted(r["analyst_briefings"]) for r in results[1:]] == [SUPPLIER_IDS] * 3
    assert [len(analysts[sid].histories) for sid in SUPPLIER_IDS] == [3, 3]       # events 2, 3 and 4
    for sid in SUPPLIER_IDS:                       # the bidder reads the memo of the event, and never the record
        for k in range(2, 5):
            memo = MEMO_HEADER + results[k - 1]["analyst_briefings"][sid]
            assert agents[sid].contexts[2 * (k - 1)].startswith(memo)
            assert HISTORY_HEADER not in agents[sid].contexts[2 * (k - 1)]


def test_specialist_constants_match_primitives():
    from orgsim.primitives import get_fixed_suppliers
    from orgsim.sequence import QUALITY_SPECIALIST, SPEED_SPECIALIST
    s, q = get_fixed_suppliers()
    assert SPEED_SPECIALIST == {"c_base": s.c_base, "gamma_delivery": s.gamma_delivery, "gamma_quality": s.gamma_quality}
    assert QUALITY_SPECIALIST == {"c_base": q.c_base, "gamma_delivery": q.gamma_delivery, "gamma_quality": q.gamma_quality}


def test_organization_and_analysts_must_agree():
    with pytest.raises(ValueError, match="a single-agent firm has no analysts"):
        _run("single", _analysts())
    for organization in ANALYST_ORGANIZATIONS:
        with pytest.raises(ValueError, match="an organized firm needs one analyst per supplier"):
            _run(organization, None)
        with pytest.raises(ValueError, match="an organized firm needs one analyst per supplier"):
            _run(organization, {"A": MockProcurementAnalyst()})


@pytest.mark.parametrize("changes, message", [
    ({"regime_name": "Buyer_Agent"}, "unknown institution: 'Buyer_Agent'"),
    ({"n_events": 3}, "the event plan has 2 events, fewer than 3"),
    ({"analysts": {"A": MockProcurementAnalyst()}}, "an organized firm needs one analyst per supplier"),
], ids=["institution", "short plan", "one analyst"])
def test_bad_inputs_fail_before_anyone_bids(changes, message):
    agents = {sid: Recording(seed=i) for i, sid in enumerate(SUPPLIER_IDS)}
    kw = dict(regime_name="Mediator", n_events=2, n_rounds=2, supplier_agents=agents,
              event_plan=pre_draw_event_sequence(2, get_contract_scenarios(), seed=7),
              mediator=MockPlatformMediator(), analysts=_analysts(), organization="organized")
    kw.update(changes)
    with pytest.raises(ValueError, match=message):
        run_sequence(**kw)
    assert not any(a.contexts for a in agents.values())


@pytest.mark.parametrize("regime", ["Mediator", "BuyerAgent"])
def test_fewer_than_two_rounds_fail_before_anyone_bids(regime):
    agents = {sid: Recording(seed=i) for i, sid in enumerate(SUPPLIER_IDS)}
    plan = pre_draw_event_sequence(1, get_contract_scenarios(), seed=7)
    with pytest.raises(ValueError, match="at least 2 rounds"):
        run_sequence(regime, 1, 1, agents, plan, mediator=MockPlatformMediator(),
                     buyer_agent_platform=MockBuyerAgentPlatform(), organization="single")
    assert not any(a.contexts for a in agents.values())


class FixedPostureAnalyst(MockProcurementAnalyst):
    def __init__(self, posture):
        self.posture = posture

    def generate_briefing(self, supplier_id, history_text):
        return AnalystBriefing(self.posture, f"ANALYST BRIEFING - {supplier_id}: test. Posture: {self.posture}.")


@pytest.mark.parametrize("organization", ANALYST_ORGANIZATIONS)
def test_results_record_the_analysts_postures(organization):
    analysts = {"A": FixedPostureAnalyst(AnalystBriefing.TRUSTING),
                "B": FixedPostureAnalyst(AnalystBriefing.DEFENSIVE)}
    _, organized = _run(organization, analysts)
    _, single = _run("single", None)
    assert organized[0]["analyst_postures"] == {}                       # no record yet, so no briefing
    assert [r["analyst_postures"] for r in organized[1:]] == [{"A": "TRUSTING", "B": "DEFENSIVE"}] * 2
    assert all(r["analyst_postures"] == {} for r in single)             # the single firm has no analyst


@pytest.mark.parametrize("organization", ANALYST_ORGANIZATIONS)
def test_single_bidder_reads_what_the_organized_analyst_reads(organization):
    """The design rule: the firms differ only in who reads the record (and in the briefing step)."""
    plan = pre_draw_event_sequence(4, get_contract_scenarios(), seed=7)

    def bidders(organization, analysts):
        agents = {sid: Recording(seed=i) for i, sid in enumerate(SUPPLIER_IDS)}
        results = run_sequence("Mediator", 4, 2, agents, plan, mediator=MockPlatformMediator(),
                               analysts=analysts, organization=organization)
        return agents, results

    analysts = {sid: RecordingAnalyst() for sid in SUPPLIER_IDS}
    single, _ = bidders("single", None)
    organized, results = bidders(organization, analysts)
    for sid in SUPPLIER_IDS:
        for k in range(2, 5):                                   # events with a record
            record = analysts[sid].histories[k - 2]             # what the analyst read
            u = record
            o = MEMO_HEADER + results[k - 1]["analyst_briefings"][sid]
            ctx_u = single[sid].contexts[2 * (k - 1)]           # round 1 of event k
            ctx_o = organized[sid].contexts[2 * (k - 1)]
            assert ctx_u.startswith(u) and ctx_o.startswith(o)
            assert ctx_u[len(u):] == ctx_o[len(o):]             # the same rules follow


@pytest.mark.parametrize("organization", ANALYST_ORGANIZATIONS)
def test_no_supplier_agent_is_told_the_upcoming_contract(organization):
    speed, quality = get_contract_scenarios()
    plan = [(speed, "speed", "quality"), (quality, "quality", "speed"), (speed, "speed", "quality")]

    def run(organization, analysts):
        agents = {sid: Recording(seed=i) for i, sid in enumerate(SUPPLIER_IDS)}
        run_sequence("Mediator", 3, 2, agents, plan, mediator=MockPlatformMediator(),
                     analysts=analysts, organization=organization)
        return agents

    analysts = {sid: RecordingAnalyst() for sid in SUPPLIER_IDS}
    single, organized = run("single", None), run(organization, analysts)
    for agents in (single, organized):
        for sid in SUPPLIER_IDS:
            assert "Priority" not in agents[sid].contexts[0]            # event 1, round 1
            assert "Quality Priority" not in agents[sid].contexts[2]    # event 2, round 1: only event 1 is past
    for sid in SUPPLIER_IDS:
        record = analysts[sid].histories[0]                             # what the analyst read before event 2
        assert record.startswith(HISTORY_HEADER)                        # the record and nothing added to it
        assert "Speed Priority" in record and "Quality Priority" not in record
        assert single[sid].contexts[2].split("\n\nRULES:", 1)[0] == record


class RecordingMediator(MockPlatformMediator):
    def __init__(self):
        self.descriptions = []

    def generate_feedback(self, buyer_prefs, all_bids, target_supplier_id, round_num,
                          buyer_description="", previous_bids=None, previous_feedback=None):
        self.descriptions.append((target_supplier_id, buyer_description))
        return super().generate_feedback(buyer_prefs, all_bids, target_supplier_id, round_num,
                                         buyer_description, previous_bids, previous_feedback)


class RecordingBuyerAgent(MockBuyerAgentPlatform):
    def __init__(self):
        self.descriptions = []

    def generate_counteroffer_feedback(self, buyer_prefs, supplier_bid, supplier_id, round_num,
                                       history, buyer_description=""):
        self.descriptions.append(buyer_description)
        return super().generate_counteroffer_feedback(buyer_prefs, supplier_bid, supplier_id,
                                                      round_num, history, buyer_description)


def test_buyer_agent_institution_gets_the_ordinal_description_and_no_memory():
    platform = RecordingBuyerAgent()
    agents = {sid: Recording(seed=i) for i, sid in enumerate(SUPPLIER_IDS)}
    plan = pre_draw_event_sequence(3, get_contract_scenarios(), seed=7)
    results = run_sequence("BuyerAgent", 3, 2, agents, plan, buyer_agent_platform=platform,
                           organization="single")
    assert all(r["outcome"].regime == "BuyerAgent" for r in results)
    assert set(platform.descriptions) <= {sc.description_for_platform for sc, _, _ in plan}
    assert len(platform.descriptions) == 3 * 2 * 1     # events x suppliers x (rounds - 1)


@pytest.mark.parametrize("regime", ["Mediator", "BuyerAgent"])
def test_the_institution_needs_its_platform_before_anyone_bids(regime):
    agents = {sid: Recording(seed=i) for i, sid in enumerate(SUPPLIER_IDS)}
    plan = pre_draw_event_sequence(1, get_contract_scenarios(), seed=7)
    with pytest.raises(ValueError, match=f"the {regime} institution needs its platform agent"):
        run_sequence(regime, 1, 2, agents, plan, organization="single")
    assert not any(a.contexts for a in agents.values())


class OverpricedInEventOne(Recording):
    """Bids far above the buyer's value in event 1 (two rounds), and as the mock does afterwards."""

    def generate_bid(self, supplier_costs, regime_context, round_num, history):
        in_event_one = len(self.contexts) < 2
        bid = super().generate_bid(supplier_costs, regime_context, round_num, history)
        if in_event_one:
            bid.terms.price = 250.0
        return bid


def test_a_walk_away_is_recorded_and_the_next_event_runs(capsys):
    plan = pre_draw_event_sequence(3, get_contract_scenarios(), seed=7)
    mediator = RecordingMediator()
    agents = {sid: OverpricedInEventOne(seed=i) for i, sid in enumerate(SUPPLIER_IDS)}
    results = run_sequence("Mediator", 3, 2, agents, plan, mediator=mediator, organization="single")
    assert results[0]["winner"] is None and results[0]["awarded"] is False
    assert results[0]["buyer_surplus"] == results[0]["supplier_profit"] == results[0]["joint_surplus"] == 0.0
    assert results[1]["awarded"] is True
    assert "Winner: none (buyer walked away)" in capsys.readouterr().out
    note = "No contract was awarded; the buyer walked away."
    for sid in SUPPLIER_IDS:
        assert f"Result: {note}" in agents[sid].contexts[2]                  # event 2, round 1
    memory = [d for sid, d in mediator.descriptions if sid == "A"][1]        # the mediator's memory before event 2
    assert f"Outcome: {note}" in memory


def test_the_mediator_reads_every_suppliers_memory():
    plan = pre_draw_event_sequence(3, get_contract_scenarios(), seed=7)
    mediator = RecordingMediator()
    agents = {sid: Recording(seed=i) for i, sid in enumerate(SUPPLIER_IDS)}
    run_sequence("Mediator", 3, 2, agents, plan, mediator=mediator, organization="single")
    event_2 = [d for _, d in mediator.descriptions[2:4]]            # one call per supplier in event 2
    assert len(set(event_2)) == 1                                   # the same text whoever is answered
    text = event_2[0]
    assert text.index("SUPPLIER A:") < text.index("SUPPLIER B:")
    assert text.endswith(plan[1][0].description_for_platform_detailed)


class RulePricedInEventOne(Recording):
    """Its bids in event 1 (two rounds) carry the tag that marks a price set by the below-cost rule."""

    def generate_bid(self, supplier_costs, regime_context, round_num, history):
        in_event_one = len(self.contexts) < 2
        bid = super().generate_bid(supplier_costs, regime_context, round_num, history)
        if in_event_one:
            bid.reasoning += RULE_PRICE_TAG
        return bid


def test_the_record_marks_a_price_set_by_rule():
    plan = pre_draw_event_sequence(2, get_contract_scenarios(), seed=7)
    agents = {sid: RulePricedInEventOne(seed=i) for i, sid in enumerate(SUPPLIER_IDS)}
    run_sequence("Mediator", 2, 2, agents, plan, mediator=MockPlatformMediator(), organization="single")
    lines = [l for l in agents["A"].contexts[2].splitlines() if "You bid" in l]    # event 2, round 1: event 1
    assert len(lines) == 2 and all("(price set by rule: cost + 5%), " in l for l in lines)
    plain, _ = _run("single", None)
    assert "price set by rule" not in plain["A"].contexts[2]                       # an ordinary bid is not marked


def _memory_entry(prices):
    bids = [SupplierBid("A", ContractTerms(p, 3.0, 3), round_num=k) for k, p in enumerate(prices, 1)]
    outcome = RunOutcome(regime="Mediator", run_id=1, winner_id="A", winning_bid=bids[-1].terms,
                         winner_true_cost=100.0, buyer_surplus=0.0, supplier_profit=0.0, joint_surplus=0.0,
                         efficient_surplus=1.0, efficient_winner_id="A", all_bids={"A": bids},
                         all_true_costs={}, all_inflation_rates={})
    return {"scenario_name": "Speed Priority", "outcome": outcome, "feedback_history": [],
            "own_cost_type": "speed"}


def _memory_of_bids(prices):
    return build_mediator_cross_event_history("A", [_memory_entry(prices)])


@pytest.mark.parametrize("prices, signal", [
    ([100.0, 110.0], "Revealed: price increase ($100\u2192$110)"),
    ([110.0, 100.0], "Revealed: price concession ($110\u2192$100)"),
    ([100.0, 102.0], "Revealed: Held firm across rounds."),
])
def test_the_mediator_memory_reports_a_price_increase_instead_of_held_firm(prices, signal):
    memory = _memory_of_bids(prices)
    assert signal in memory
    assert ("Held firm" in memory) is signal.endswith("Held firm across rounds.")


def test_the_mediator_memory_keeps_the_last_events_and_says_how_many_it_left_out():
    entries = [_memory_entry([100.0 + k, 110.0 + k]) for k in range(3)]
    full = build_mediator_cross_event_history("A", entries)
    assert "  Event 1 (" in full and "omitted" not in full
    cut = build_mediator_cross_event_history("A", entries, max_events=2)
    assert "in 3 previous procurement event(s)" in cut                      # the count is of all the events
    assert "[1 earlier event(s) omitted for brevity]" in cut
    assert "  Event 1 (" not in cut and "  Event 2 (" in cut and "  Event 3 (" in cut   # numbered as in the full memory
    assert build_mediator_cross_event_history("A", entries, max_events=3) == full       # nothing to leave out
    assert build_mediator_cross_event_history("A", [], max_events=2) == ""


def test_max_history_events_limits_the_mediators_memory_as_well_as_the_suppliers_record():
    plan = pre_draw_event_sequence(4, get_contract_scenarios(), seed=7)

    def last_memory_and_record(limit):
        mediator = RecordingMediator()
        agents = {sid: Recording(seed=i) for i, sid in enumerate(SUPPLIER_IDS)}
        run_sequence("Mediator", 4, 2, agents, plan, mediator=mediator, organization="single",
                     max_history_events=limit)
        return mediator.descriptions[-1][1], agents["A"].contexts[-1]       # event 4: three events are past

    full, _ = last_memory_and_record(0)
    cut, cut_record = last_memory_and_record(1)
    assert full.count("  Event 1 (") == 2 and "omitted" not in full         # the memory of each of the two suppliers
    assert cut.count("[2 earlier event(s) omitted for brevity]") == 2
    assert cut.count("  Event 3 (") == 2 and "  Event 1 (" not in cut and "  Event 2 (" not in cut
    assert cut.count("in 3 previous procurement event(s)") == 2
    assert "[2 earlier event(s) omitted for brevity]" in cut_record         # the supplier's record is cut the same way
    assert len(cut) < len(full)


def test_on_event_gets_each_result_right_after_its_event_and_before_the_next_starts():
    plan = pre_draw_event_sequence(3, get_contract_scenarios(), seed=7)
    agents = {sid: Recording(seed=i) for i, sid in enumerate(SUPPLIER_IDS)}
    seen = []

    def on_event(result):
        seen.append((result, len(agents["A"].contexts)))     # A's bids so far: two per finished event

    results = run_sequence("Mediator", 3, 2, agents, plan, mediator=MockPlatformMediator(),
                           organization="single", on_event=on_event)
    assert [bids for _, bids in seen] == [2, 4, 6]
    assert [r["event"] for r, _ in seen] == [1, 2, 3]
    assert all(r is returned for (r, _), returned in zip(seen, results))   # the dicts in the returned list


class FailsInEventTwo(Recording):
    """Bids as the mock does in event 1 (two rounds), then raises on its first bid of event 2."""

    def generate_bid(self, supplier_costs, regime_context, round_num, history):
        if len(self.contexts) >= 2:
            raise RuntimeError("model unavailable")
        return super().generate_bid(supplier_costs, regime_context, round_num, history)


def test_events_finished_before_a_failure_were_passed_to_on_event():
    plan = pre_draw_event_sequence(3, get_contract_scenarios(), seed=7)
    agents = {sid: FailsInEventTwo(seed=i) for i, sid in enumerate(SUPPLIER_IDS)}
    kept = []
    with pytest.raises(RuntimeError, match="model unavailable"):
        run_sequence("Mediator", 3, 2, agents, plan, mediator=MockPlatformMediator(),
                     organization="single", on_event=kept.append)
    assert [r["event"] for r in kept] == [1]


def test_an_exception_from_on_event_stops_the_sequence():
    plan = pre_draw_event_sequence(3, get_contract_scenarios(), seed=7)
    agents = {sid: Recording(seed=i) for i, sid in enumerate(SUPPLIER_IDS)}

    def stop(result):
        raise KeyError("stop here")

    with pytest.raises(KeyError):
        run_sequence("Mediator", 3, 2, agents, plan, mediator=MockPlatformMediator(),
                     organization="single", on_event=stop)
    assert len(agents["A"].contexts) == 2                   # only event 1 ran


def test_the_sequence_runner_does_not_import_the_language_model_agents():
    code = "import sys, orgsim.sequence; sys.exit('orgsim.llm_agents' in sys.modules)"
    done = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, capture_output=True, timeout=120)
    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")[-500:]


# ---- the second-price auction as one event of run_sequence ----

def _second_price(n_events=1, n_rounds=3, organization="single", analysts=None, events_in_plan=1):
    agents = {sid: Recording(seed=i) for i, sid in enumerate(SUPPLIER_IDS)}
    plan = pre_draw_event_sequence(events_in_plan, get_contract_scenarios(), seed=7)
    results = run_sequence("SecondPrice", n_events, n_rounds, agents, plan, organization=organization,
                           analysts=analysts)
    return agents, results


@pytest.mark.parametrize("n_rounds", [3, 1, 0])
def test_the_second_price_auction_is_one_sealed_bid_for_each_supplier_whatever_the_number_of_rounds(n_rounds):
    agents, results = _second_price(n_rounds=n_rounds)
    (result,) = results
    outcome = result["outcome"]
    assert outcome.regime == "SecondPrice" and result["event"] == 1
    assert [len(outcome.all_bids[sid]) for sid in SUPPLIER_IDS] == [1, 1]
    assert [len(agents[sid].contexts) for sid in SUPPLIER_IDS] == [1, 1]
    assert outcome.mediator_feedback == {} and result["analyst_briefings"] == result["analyst_postures"] == {}
    assert result["awarded"] is True and outcome.winning_bid.price >= outcome.all_bids[result["winner"]][0].terms.price


def test_the_second_price_bidder_reads_the_rules_alone_with_no_record_and_no_first_event_note():
    agents, _ = _second_price()
    for sid in SUPPLIER_IDS:
        (context,) = agents[sid].contexts
        assert context.startswith("RULES (second-price auction):")
        for text in (FIRST_EVENT_NOTE, "PAST EVENTS", HISTORY_HEADER, MEMO_HEADER, "platform"):
            assert text not in context, text


@pytest.mark.parametrize("kwargs, message", [
    ({"n_events": 2, "events_in_plan": 2}, "the second-price auction is a one-time event, so it needs n_events == 1"),
    ({"organization": "organized", "analysts": {sid: MockProcurementAnalyst() for sid in SUPPLIER_IDS}},
     "it needs organization 'single'"),
    ({"organization": "organized_guided", "analysts": {sid: MockProcurementAnalyst() for sid in SUPPLIER_IDS}},
     "it needs organization 'single'"),
    ({"analysts": {sid: MockProcurementAnalyst() for sid in SUPPLIER_IDS}}, "a single-agent firm has no analysts"),
])
def test_the_second_price_auction_needs_one_event_and_a_single_firm(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _second_price(**kwargs)


def test_the_ai_institutions_still_need_two_rounds_and_a_platform_and_an_unknown_institution_is_refused():
    agents = {sid: Recording(seed=i) for i, sid in enumerate(SUPPLIER_IDS)}
    plan = pre_draw_event_sequence(1, get_contract_scenarios(), seed=7)
    with pytest.raises(ValueError, match="at least 2 rounds"):
        run_sequence("Mediator", 1, 1, agents, plan, mediator=MockPlatformMediator(), organization="single")
    with pytest.raises(ValueError, match="needs its platform agent"):
        run_sequence("BuyerAgent", 1, 2, agents, plan, organization="single")
    for name in ("FirstPrice", "Second_price", "second_price", ""):
        with pytest.raises(ValueError, match="unknown institution"):
            run_sequence(name, 1, 2, agents, plan, organization="single")


# ---- mixed markets: the two suppliers have different organizations ----

ORGANIZATIONS = ("single", "organized", "organized_guided")
MIXED_PAIRS = [(a, b) for a in ORGANIZATIONS for b in ORGANIZATIONS if a != b]
PAIR_IDS = [":".join(pair) for pair in MIXED_PAIRS]


class MarkedAnalyst(MockProcurementAnalyst):
    """Keeps the record it reads, and writes a briefing that names its supplier and counts its calls, to follow
    where the briefing goes."""

    def __init__(self):
        self.histories = []

    def generate_briefing(self, supplier_id, history_text):
        self.histories.append(history_text)
        return AnalystBriefing(AnalystBriefing.GUARDED, f"BRIEFING-MARK-{supplier_id}-{len(self.histories)}")


def _market(organization, n_events=4, institution="Mediator", mediator=None, buyer_agent=None):
    """Run a market with mock bidders and platforms and a marked analyst for each supplier whose organization has
    one; return the bidders, the analysts and the results."""
    organizations = organization_of_each(organization, SUPPLIER_IDS)
    agents = {sid: Recording(seed=i) for i, sid in enumerate(SUPPLIER_IDS)}
    analysts = {sid: MarkedAnalyst() for sid in SUPPLIER_IDS if organizations[sid] in ANALYST_ORGANIZATIONS}
    plan = pre_draw_event_sequence(n_events, get_contract_scenarios(), seed=7)
    results = run_sequence(institution, n_events, 2, agents, plan, mediator=mediator or MockPlatformMediator(),
                           buyer_agent_platform=buyer_agent or MockBuyerAgentPlatform(), analysts=analysts or None,
                           organization=organization)
    return agents, analysts, results


@pytest.mark.parametrize("pair", MIXED_PAIRS, ids=PAIR_IDS)
def test_each_supplier_in_a_mixed_market_reads_what_it_reads_in_the_market_of_its_own_organization(pair):
    mixed, mixed_analysts, mixed_results = _market(dict(zip(SUPPLIER_IDS, pair)))
    alike = {sid: _market(organization) for sid, organization in zip(SUPPLIER_IDS, pair)}   # both suppliers alike
    for sid in SUPPLIER_IDS:
        alike_agents, alike_analysts, alike_results = alike[sid]
        assert mixed[sid].contexts == alike_agents[sid].contexts                     # what its bidder reads
        for field in ("analyst_briefings", "analyst_postures"):                      # what its analyst writes
            assert [r[field].get(sid) for r in mixed_results] == [r[field].get(sid) for r in alike_results]
        assert (sid in mixed_analysts) == (sid in alike_analysts)                    # an analyst if organized
        if sid in mixed_analysts:
            assert mixed_analysts[sid].histories == alike_analysts[sid].histories    # and the record it reads
    assert set(mixed_analysts) == {sid for sid, organization in zip(SUPPLIER_IDS, pair)
                                   if organization in ANALYST_ORGANIZATIONS}
    assert [r["winner"] for r in mixed_results] == [r["winner"] for r in alike["A"][2]]     # the same bids and results


def test_in_a_single_versus_organized_market_the_organized_suppliers_analyst_briefs_that_supplier_only():
    agents, analysts, results = _market({"A": "single", "B": "organized"})
    assert set(analysts) == {"B"} and len(analysts["B"].histories) == 3                  # events 2, 3 and 4
    assert results[0]["analyst_briefings"] == {}
    assert [r["analyst_briefings"] for r in results[1:]] == [{"B": f"BRIEFING-MARK-B-{k}"} for k in (1, 2, 3)]
    assert [r["analyst_postures"] for r in results[1:]] == [{"B": "GUARDED"}] * 3
    for k in range(2, 5):
        a_context, b_context = agents["A"].contexts[2 * (k - 1)], agents["B"].contexts[2 * (k - 1)]
        assert a_context.startswith(HISTORY_HEADER) and MEMO_HEADER not in a_context      # A reads its own record
        assert b_context.startswith(MEMO_HEADER + f"BRIEFING-MARK-B-{k - 1}") and HISTORY_HEADER not in b_context


@pytest.mark.parametrize("pair", MIXED_PAIRS, ids=PAIR_IDS)
def test_no_supplier_in_a_mixed_market_ever_receives_the_other_suppliers_record_or_briefing(pair):
    organization = dict(zip(SUPPLIER_IDS, pair))
    agents, analysts, _ = _market(organization)
    records = {sid: set() for sid in SUPPLIER_IDS}          # each supplier's record, as its analyst or bidder reads it
    for sid in SUPPLIER_IDS:
        if sid in analysts:
            records[sid] |= set(analysts[sid].histories)
        else:
            records[sid] |= {c.split("\n\nRULES:", 1)[0] for c in agents[sid].contexts if c.startswith(HISTORY_HEADER)}
    for me, rival in (("A", "B"), ("B", "A")):
        assert records[rival]                                  # there is a record that could leak
        for context in agents[me].contexts:
            assert f"BRIEFING-MARK-{rival}-" not in context
            assert not any(record in context for record in records[rival])
    for sid, name in organization.items():
        if name == "single":
            assert any(c.startswith(HISTORY_HEADER) for c in agents[sid].contexts)
            assert not any(MEMO_HEADER in c for c in agents[sid].contexts)
        else:
            assert any(c.startswith(MEMO_HEADER + f"BRIEFING-MARK-{sid}-") for c in agents[sid].contexts)
            assert not any(HISTORY_HEADER in c for c in agents[sid].contexts)
            assert not any(record in c for record in records[sid] for c in agents[sid].contexts)


class AllCallsMediator(MockPlatformMediator):
    """Keeps every call (all its arguments) to the mediator."""

    def __init__(self):
        self.calls = []

    def generate_feedback(self, *args, **kw):
        self.calls.append(repr((args, sorted(kw.items()))))
        return super().generate_feedback(*args, **kw)


class AllCallsBuyerAgent(MockBuyerAgentPlatform):
    """Keeps every call (all its arguments) to the buyer agent."""

    def __init__(self):
        self.calls = []

    def generate_counteroffer_feedback(self, *args, **kw):
        self.calls.append(repr((args, sorted(kw.items()))))
        return super().generate_counteroffer_feedback(*args, **kw)


@pytest.mark.parametrize("institution, platform_class, argument", [
    ("Mediator", AllCallsMediator, "mediator"), ("BuyerAgent", AllCallsBuyerAgent, "buyer_agent")])
def test_the_platforms_are_sent_the_same_in_every_market_whatever_the_organizations(institution, platform_class,
                                                                                     argument):
    calls = {}
    for organization in [*ORGANIZATIONS, *({"A": a, "B": b} for a, b in MIXED_PAIRS)]:
        platform = platform_class()
        _market(organization, institution=institution, **{argument: platform})
        calls[repr(organization)] = platform.calls
    assert len(calls) == 3 + 6 and all(calls.values())
    assert len({tuple(each) for each in calls.values()}) == 1             # the same calls in the nine markets
    assert not re.search(r"organi[sz]|analyst|briefing|memo\b|BRIEFING-MARK", " ".join(calls["'single'"]), re.I)


@pytest.mark.parametrize("organization", ORGANIZATIONS)
def test_a_mapping_of_one_organization_for_both_suppliers_is_the_homogeneous_market(organization):
    alike, alike_analysts, alike_results = _market(organization)
    mapped, mapped_analysts, mapped_results = _market({"A": organization, "B": organization})
    assert [mapped[sid].contexts for sid in SUPPLIER_IDS] == [alike[sid].contexts for sid in SUPPLIER_IDS]
    assert [r["analyst_briefings"] for r in mapped_results] == [r["analyst_briefings"] for r in alike_results]
    assert ({sid: a.histories for sid, a in mapped_analysts.items()}
            == {sid: a.histories for sid, a in alike_analysts.items()})


@pytest.mark.parametrize("pair, right, too_many, too_few", [
    (("single", "organized"), {"B"}, {"A", "B"}, set()),
    (("organized_guided", "single"), {"A"}, {"A", "B"}, set()),
    (("organized", "organized_guided"), {"A", "B"}, None, {"A"}),
], ids=["single:organized", "organized_guided:single", "organized:organized_guided"])
def test_a_mixed_market_needs_the_analysts_of_its_organized_suppliers_and_only_those(pair, right, too_many,
                                                                                      too_few):
    def run(analyst_ids):
        agents = {sid: Recording(seed=i) for i, sid in enumerate(SUPPLIER_IDS)}
        plan = pre_draw_event_sequence(2, get_contract_scenarios(), seed=7)
        analysts = {sid: MockProcurementAnalyst() for sid in analyst_ids} or None
        return run_sequence("Mediator", 2, 2, agents, plan, mediator=MockPlatformMediator(), analysts=analysts,
                            organization=dict(zip(SUPPLIER_IDS, pair)))

    assert len(run(right)) == 2
    if too_many:
        with pytest.raises(ValueError, match="a single-agent firm has no analysts"):
            run(too_many)
    with pytest.raises(ValueError, match="an organized firm needs one analyst per supplier"):
        run(too_few)
    with pytest.raises(ValueError, match="an analyst is for a supplier of the market"):
        run(right | {"C"})


@pytest.mark.parametrize("organization", [{"A": "single"}, {"A": "single", "B": "organized", "C": "single"},
                                          ["single", "organized"]])
def test_a_market_must_name_the_organization_of_each_supplier_before_anyone_bids(organization):
    agents = {sid: Recording(seed=i) for i, sid in enumerate(SUPPLIER_IDS)}
    plan = pre_draw_event_sequence(2, get_contract_scenarios(), seed=7)
    with pytest.raises(ValueError):
        run_sequence("Mediator", 2, 2, agents, plan, mediator=MockPlatformMediator(), organization=organization)
    assert not any(a.contexts for a in agents.values())


def test_an_unknown_organization_of_one_supplier_fails_before_anyone_bids():
    agents = {sid: Recording(seed=i) for i, sid in enumerate(SUPPLIER_IDS)}
    plan = pre_draw_event_sequence(2, get_contract_scenarios(), seed=7)
    with pytest.raises(ValueError, match="unknown organization: 'hybrid'"):
        run_sequence("Mediator", 2, 2, agents, plan, mediator=MockPlatformMediator(),
                     organization={"A": "single", "B": "hybrid"})
    assert not any(a.contexts for a in agents.values())


def test_the_second_price_auction_refuses_a_mixed_market():
    agents = {sid: Recording(seed=i) for i, sid in enumerate(SUPPLIER_IDS)}
    plan = pre_draw_event_sequence(1, get_contract_scenarios(), seed=7)
    with pytest.raises(ValueError, match="it needs organization 'single'"):
        run_sequence("SecondPrice", 1, 3, agents, plan, organization={"A": "single", "B": "organized"},
                     analysts={"B": MockProcurementAnalyst()})
    assert not any(a.contexts for a in agents.values())
    run_sequence("SecondPrice", 1, 3, agents, plan, organization={"A": "single", "B": "single"})   # both single: fine
    assert [len(a.contexts) for a in agents.values()] == [1, 1]
