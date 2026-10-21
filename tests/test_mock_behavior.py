"""Behavior tests for the mock agents, the agent interfaces and bid parsing."""
import inspect

import pytest

from orgsim import interfaces, organization
from orgsim.interfaces import parse_bid_json
from orgsim.mocks import (
    MockBuyerAgentPlatform, MockPlatformMediator, MockProcurementAnalyst, MockStrategicSupplier,
)
from orgsim.organization import AnalystBriefing
from orgsim.primitives import ContractTerms, get_contract_scenarios, get_fixed_suppliers

HISTORY_HEADER = "YOUR PAST PROCUREMENT HISTORY WITH THIS PLATFORM:"
PRESSURE = "Your pricing is on the higher end relative to the field."
POSITIVE = "Your bid is well-positioned overall."


def _history(n_events, profit=20.0, first_feedback=PRESSURE):
    """History text in the layout the sequence runner writes (one block per event)."""
    lines = [HISTORY_HEADER,
             f"You have participated in {n_events} previous procurement event(s) through this same platform. "
             f"Here is what happened:\n"]
    for i in range(1, n_events + 1):
        fb = first_feedback if i == 1 else PRESSURE
        lines += [f"  Event {i} (Speed Priority, your costs: speed specialist):",
                  "    Round 1: You bid $150.00, 3.0wk, tier 3",
                  f'    Feedback: "{fb}"',
                  f"    Result: You WON. You were paid $150.00. Your profit was ${profit:.2f}.",
                  ""]
    return "\n".join(lines)


def test_mocks_keep_the_interface_signatures():
    pairs = [
        (interfaces.SupplierAgent.generate_bid, MockStrategicSupplier.generate_bid),
        (interfaces.PlatformMediator.generate_feedback, MockPlatformMediator.generate_feedback),
        (interfaces.BuyerAgentPlatform.generate_counteroffer_feedback,
         MockBuyerAgentPlatform.generate_counteroffer_feedback),
        (organization.ProcurementAnalyst.generate_briefing, MockProcurementAnalyst.generate_briefing),
    ]
    for abstract, mock in pairs:
        assert inspect.signature(mock) == inspect.signature(abstract)


def test_mock_analyst_reaches_every_posture():
    analyst = MockProcurementAnalyst()
    assert analyst.generate_briefing("A", _history(2)).posture == AnalystBriefing.GUARDED
    defensive = analyst.generate_briefing("A", _history(3, profit=5.0))
    assert defensive.posture == AnalystBriefing.DEFENSIVE and defensive.is_defensive
    trusting = analyst.generate_briefing("A", _history(3, profit=5.0, first_feedback=POSITIVE))
    assert trusting.posture == AnalystBriefing.TRUSTING and not trusting.is_defensive
    assert analyst.generate_briefing("A", _history(3, profit=20.0)).posture == AnalystBriefing.GUARDED


def test_mock_analyst_memo_does_not_quote_the_raw_history():
    memo = MockProcurementAnalyst().generate_briefing("A", _history(4, profit=5.0)).text
    assert HISTORY_HEADER not in memo
    assert "Event 1 (" not in memo and "Round 1:" not in memo


def test_mock_supplier_is_deterministic_and_ignores_regime_context():
    s, q = get_fixed_suppliers()

    def bids(context, sup):
        agent, history, out = MockStrategicSupplier(seed=5), [], []
        for r in (1, 2, 3):
            b = agent.generate_bid(sup, context, r, history)
            out.append(b.terms.to_dict())
            history.append({"round": r, "bid": b.terms, "feedback": "Improve delivery and quality."})
        return out

    for sup in (s, q):
        assert bids("RULES", sup) == bids("RULES", sup)
        assert bids("RULES", sup) == bids(HISTORY_HEADER + "\nEvent 1 ...\n\nRULES", sup)  # context is ignored


def test_mock_supplier_answers_feedback_keywords_in_later_rounds():
    def round_two(sup, feedback):
        agent = MockStrategicSupplier(seed=3)
        first = agent.generate_bid(sup, "R", 1, [])
        return agent.generate_bid(sup, "R", 2, [{"round": 1, "bid": first.terms, "feedback": feedback}]).terms

    for sup in get_fixed_suppliers():
        base = round_two(sup, "nothing relevant")
        assert round_two(sup, "a faster delivery?").delivery_weeks == pytest.approx(max(1.0, base.delivery_weeks - 0.5))
        assert round_two(sup, "a higher quality tier?").quality_tier == min(5, base.quality_tier + 1)
        assert base.price > 0


def test_mock_feedback_texts_are_informative():
    s, q = get_fixed_suppliers()
    sc = get_contract_scenarios()[0]
    bids = {sid: MockStrategicSupplier(seed=i).generate_bid(sup, "R", 1, []).terms
            for i, (sid, sup) in enumerate([(s.supplier_id, s), (q.supplier_id, q)])}
    assert MockPlatformMediator().generate_feedback(sc.buyer, bids, s.supplier_id, 1).strip()
    alone = MockPlatformMediator().generate_feedback(sc.buyer, {s.supplier_id: bids[s.supplier_id]},
                                                     s.supplier_id, 1)
    assert alone.startswith("Your bid is well-positioned") and "participant" not in alone   # no supplier count
    prev = ContractTerms(price=100.0, delivery_weeks=6.0, quality_tier=2)
    cur = ContractTerms(price=95.0, delivery_weeks=4.0, quality_tier=3)
    first = MockBuyerAgentPlatform().generate_counteroffer_feedback(sc.buyer, cur, s.supplier_id, 1, [])
    later = MockBuyerAgentPlatform().generate_counteroffer_feedback(sc.buyer, cur, s.supplier_id, 2, [{"bid": prev}])
    assert "significantly better offer" in first
    assert "delivery improvement" in later and "Good quality improvement" in later


def test_parse_bid_json_plain_embedded_and_failing_text():
    good = '{"price": 120.5, "delivery_weeks": 3, "quality_tier": 4, "reasoning": "x"}'
    plain = parse_bid_json(good, "A", 2)
    embedded = parse_bid_json("Here is my bid: " + good + " thanks", "A", 2)
    assert plain.terms == embedded.terms == ContractTerms(120.5, 3.0, 4)
    assert plain.supplier_id == "A" and plain.round_num == 2
    with pytest.raises(ValueError):
        parse_bid_json("no bid here", "A", 1)


def test_abstract_interfaces_cannot_be_instantiated():
    for cls in (interfaces.SupplierAgent, interfaces.PlatformMediator, interfaces.BuyerAgentPlatform):
        with pytest.raises(TypeError):
            cls()


def test_mock_supplier_bids_above_cost_and_in_range_for_both_profiles():
    for sup in get_fixed_suppliers():
        for seed in range(20):
            terms = MockStrategicSupplier(seed=seed).generate_bid(sup, "R", 1, []).terms
            assert terms.price > sup.cost(terms)
            assert 1.0 <= terms.delivery_weeks <= 8.0 and 2 <= terms.quality_tier <= 5


def _record(prices, profits):
    lines = [HISTORY_HEADER, f"You have participated in {len(prices)} previous procurement event(s) through "
                             f"this same platform. Here is what happened:\n"]
    for i, (price, profit) in enumerate(zip(prices, profits), 1):
        lines += [f"  Event {i} (Speed Priority, your costs: speed specialist):",
                  f"    Round 1: You bid ${price:.2f}, 3.0wk, tier 3",
                  f'    Feedback: "{PRESSURE}"',
                  f"    Result: You WON. You were paid ${price:.2f}. Your profit was ${profit:.2f}.", ""]
    return "\n".join(lines)


def test_mock_analyst_text_makes_no_claim_about_honest_competition_or_absent_rivals():
    # Profits fall from 30 to under 18 (margin warning); two of the last three wins are well above average.
    text = MockProcurementAnalyst().generate_briefing(
        "A", _record([100.0, 100.0, 100.0, 150.0, 150.0], [30.0, 30.0, 30.0, 5.0, 5.0])).text
    assert "MARGIN WARNING" in text and "COMPETITIVE SIGNAL" in text
    assert "Honest competition" not in text and "not at all" not in text
    assert "(recent events). Maintain margin discipline" in text
    assert "other suppliers may be bidding less aggressively. " in text


def test_mock_defensive_text_matches_the_corrected_guidance():
    text = MockProcurementAnalyst().generate_briefing("A", _history(3, profit=5.0)).text
    assert "that signals competing bids may have weakened" in text
    assert "competing bids are likely still strong" in text
    assert "Do not assume you are the only bidder" in text
    assert "may have thinned" not in text and "still active" not in text
