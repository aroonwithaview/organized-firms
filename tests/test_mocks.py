from orgsim.mocks import (
    MockBuyerAgentPlatform, MockPlatformMediator, MockProcurementAnalyst, MockStrategicSupplier,
)
from orgsim.organization import AnalystBriefing
from orgsim.primitives import get_contract_scenarios, get_fixed_suppliers


def test_mock_supplier_returns_bid_in_range():
    s, _ = get_fixed_suppliers()
    bid = MockStrategicSupplier(seed=1).generate_bid(s, "RULES: test", round_num=1, history=[])
    assert bid.supplier_id == s.supplier_id
    assert 1.0 <= bid.terms.delivery_weeks <= 8.0
    assert 2 <= bid.terms.quality_tier <= 5


def test_mock_platforms_return_text():
    s, q = get_fixed_suppliers()
    sc = get_contract_scenarios()[0]
    bids = {sid: MockStrategicSupplier(seed=i).generate_bid(sup, "R", 1, []).terms
            for i, (sid, sup) in enumerate([(s.supplier_id, s), (q.supplier_id, q)])}
    assert isinstance(MockPlatformMediator().generate_feedback(sc.buyer, bids, s.supplier_id, 1), str)
    assert isinstance(
        MockBuyerAgentPlatform().generate_counteroffer_feedback(sc.buyer, bids[s.supplier_id], s.supplier_id, 1, []),
        str,
    )


def test_mock_analyst_first_event_is_guarded():
    b = MockProcurementAnalyst().generate_briefing("A", "")
    assert b.posture == AnalystBriefing.GUARDED


def test_mock_buyer_agent_does_not_say_how_many_suppliers_there_are():
    s, _ = get_fixed_suppliers()
    sc = get_contract_scenarios()[0]
    text = MockBuyerAgentPlatform().generate_counteroffer_feedback(
        sc.buyer, MockStrategicSupplier(seed=1).generate_bid(s, "R", 1, []).terms, s.supplier_id, 1, [])
    assert "from another supplier" in text and "the other supplier" not in text
