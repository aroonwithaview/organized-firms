import json
import re

import pytest

from orgsim.primitives import (
    BuyerPreferences, ContractTerms, RunOutcome, SupplierCosts,
    compute_efficient_outcome, compute_max_efficient_surplus,
    get_contract_scenarios, get_fixed_suppliers,
)


def _buyers():
    scenarios = get_contract_scenarios()
    assert [sc.name for sc in scenarios] == ["Speed Priority", "Quality Priority"]
    speed, quality = (sc.buyer for sc in scenarios)
    return speed, quality


def test_speed_buyer_speed_specialist_matches_closed_form():
    # Notation: y = delivery_max - delivery_weeks (weeks of speed-up), t = quality tier.
    # Linear value, quadratic cost: optimal speed-up y* = a / (2 * gamma_d) with
    # a = w_d * beta_d / delivery_max = 0.8 * 60 / 8 = 6 and gamma_d = 0.5, so y* = 6,
    # i.e. 2.0 weeks. Quality is worth 2 per tier and costs 1.8 * t^2; the unconstrained
    # optimum in t is below 2, so the tier is the corner t = 2 of the 2-5 range.
    speed, _ = _buyers()
    s, _ = get_fixed_suppliers()
    surplus, terms = compute_efficient_outcome(speed, s)
    assert terms.delivery_weeks == pytest.approx(2.0)
    assert terms.quality_tier == 2
    # base + delivery value + quality value - (base cost + delivery cost + quality cost)
    assert surplus == pytest.approx(150 + 36 + 4 - (85 + 18 + 7.2))


def test_quality_buyer_quality_specialist_matches_closed_form():
    # Same notation: y = delivery_max - delivery_weeks, t = quality tier.
    # a = 0.25 * 60 / 8 = 1.875 and gamma_d = 1.8, so y* is about 0.52; the 0.1 grid gives
    # 0.5, i.e. 7.5 weeks. Quality is worth 7.5 per tier and costs 0.4 * t^2; the
    # unconstrained optimum in t is above 5, so the tier is the corner t = 5 of the 2-5 range.
    _, quality = _buyers()
    _, q = get_fixed_suppliers()
    surplus, terms = compute_efficient_outcome(quality, q)
    assert terms.delivery_weeks == pytest.approx(7.5)
    assert terms.quality_tier == 5
    # base + delivery value + quality value - (base cost + delivery cost + quality cost)
    assert surplus == pytest.approx(150 + 0.9375 + 37.5 - (85 + 0.45 + 10))


def test_efficient_winner_matches_specialization():
    # The speed buyer facing Q ties between quality tiers 2 and 3, so no test
    # asserts the tier for that pair.
    speed, quality = _buyers()
    s, q = get_fixed_suppliers()
    assert compute_max_efficient_surplus(speed, [s, q])[1] == s.supplier_id
    assert compute_max_efficient_surplus(quality, [s, q])[1] == q.supplier_id


def test_cost_and_score_definitions():
    s, _ = get_fixed_suppliers()
    t = ContractTerms(price=120.0, delivery_weeks=4.0, quality_tier=3)
    assert s.cost(t) == pytest.approx(85 + 0.5 * 16 + 1.8 * 9)
    buyer = BuyerPreferences()
    assert buyer.score(t) == pytest.approx(buyer.value(t) - 120.0)


def test_max_efficient_surplus_returns_best_surplus_and_terms():
    speed, quality = _buyers()
    s, q = get_fixed_suppliers()
    for buyer, winner in ((speed, s), (quality, q)):
        for order in ([s, q], [q, s]):
            surplus, wid, terms = compute_max_efficient_surplus(buyer, order)
            best_surplus, best_terms = compute_efficient_outcome(buyer, winner)
            assert wid == winner.supplier_id
            assert surplus == pytest.approx(best_surplus)
            assert terms == best_terms
    assert compute_max_efficient_surplus(speed, [s, q])[0] == pytest.approx(150 + 36 + 4 - (85 + 18 + 7.2))
    assert compute_max_efficient_surplus(quality, [s, q])[0] == pytest.approx(150 + 0.9375 + 37.5 - (85 + 0.45 + 10))


def test_exact_tie_goes_to_first_listed_supplier():
    # Same-type event: identical costs tie exactly; strict '>' keeps the first listed.
    # So correct_winner is not meaningful in same-type events (see README).
    speed, _ = _buyers()
    s, _ = get_fixed_suppliers()
    twin = SupplierCosts(supplier_id="twin", name="twin", c_base=s.c_base,
                         gamma_delivery=s.gamma_delivery, gamma_quality=s.gamma_quality)
    assert compute_max_efficient_surplus(speed, [s, twin])[1] == s.supplier_id
    assert compute_max_efficient_surplus(speed, [twin, s])[1] == "twin"


def _outcome(winner_id, efficient_winner_id, joint, efficient):
    t = ContractTerms(price=0.0, delivery_weeks=2.0, quality_tier=2)
    return RunOutcome(regime="r", run_id=0, winner_id=winner_id, winning_bid=t, winner_true_cost=0.0,
                      buyer_surplus=0.0, supplier_profit=0.0, joint_surplus=joint,
                      efficient_surplus=efficient, efficient_winner_id=efficient_winner_id,
                      all_bids={}, all_true_costs={}, all_inflation_rates={})


def test_efficiency_ratio_and_correct_winner():
    assert _outcome("S", "S", 80.0, 80.0).efficiency_ratio == pytest.approx(1.0)
    assert _outcome("S", "S", 40.0, 80.0).efficiency_ratio == pytest.approx(0.5)
    assert _outcome("S", "S", 1.0, 0.0).efficiency_ratio == 0.0   # non-positive denominator -> 0.0
    assert _outcome("S", "S", 1.0, 2.0).correct_winner is True
    assert _outcome("Q", "S", 1.0, 2.0).correct_winner is False


def test_an_outcome_without_a_winner_is_not_awarded():
    out = RunOutcome(regime="r", run_id=0, winner_id=None, winning_bid=None, winner_true_cost=0.0,
                     buyer_surplus=0.0, supplier_profit=0.0, joint_surplus=0.0,
                     efficient_surplus=80.0, efficient_winner_id="S",
                     all_bids={}, all_true_costs={}, all_inflation_rates={})
    assert out.awarded is False and _outcome("S", "S", 40.0, 80.0).awarded is True
    assert out.correct_winner is False
    assert out.efficiency_ratio == 0.0                       # joint surplus 0
    summary = out.summary()
    assert summary["winner"] is None and summary["awarded"] is False and summary["winner_inflation"] == 0


def test_summary_has_these_keys_and_rounds_money_to_cents_and_ratios_to_four_places():
    out = RunOutcome(regime="Mediator", run_id=3, winner_id="B", winning_bid=ContractTerms(120.0, 3.0, 4),
                     winner_true_cost=100.0, buyer_surplus=10.126, supplier_profit=19.881,
                     joint_surplus=30.007, efficient_surplus=36.0, efficient_winner_id="A",
                     all_bids={}, all_true_costs={"A": 90.0, "B": 100.0},
                     all_inflation_rates={"A": 0.5, "B": 0.199999})
    summary = out.summary()
    assert set(summary) == {"regime", "run_id", "winner", "awarded", "correct_winner", "buyer_surplus",
                            "supplier_profit", "joint_surplus", "efficiency_ratio", "winner_inflation"}
    assert summary["regime"] == "Mediator" and summary["run_id"] == 3
    assert summary["winner"] == "B" and summary["awarded"] is True and summary["correct_winner"] is False
    assert summary["buyer_surplus"] == 10.13 and summary["supplier_profit"] == 19.88   # to cents
    assert summary["joint_surplus"] == 30.01                                           # to cents too (30.007)
    assert summary["efficiency_ratio"] == 0.8335                                       # 30.007 / 36, to four places
    assert summary["winner_inflation"] == 0.2                  # the winner's markup, rounded: B's, not A's


def test_contract_terms_round_trip_through_a_dict_and_through_json():
    terms = ContractTerms(price=123.45, delivery_weeks=3.5, quality_tier=4)
    as_dict = terms.to_dict()
    assert as_dict == {"price": 123.45, "delivery_weeks": 3.5, "quality_tier": 4}
    assert isinstance(as_dict["quality_tier"], int)                  # 4.0 == 4, so the equality above cannot tell
    assert ContractTerms.from_dict(as_dict) == terms
    assert ContractTerms.from_dict(json.loads(json.dumps(as_dict))) == terms
    # from_dict converts the types: whole numbers in the dict become a float price and delivery time.
    whole = ContractTerms.from_dict({"price": 120, "delivery_weeks": 3, "quality_tier": 4})
    assert whole == ContractTerms(120.0, 3.0, 4)
    assert isinstance(whole.price, float) and isinstance(whole.delivery_weeks, float)
    assert isinstance(whole.quality_tier, int)
    # A tier given as 4.0 becomes the int 4 (4.0 == 4, so only the type shows it).
    from_float = ContractTerms.from_dict({"price": 120, "delivery_weeks": 3, "quality_tier": 4.0})
    assert from_float == ContractTerms(120.0, 3.0, 4) and isinstance(from_float.quality_tier, int)


def test_score_value_profit_against_hand_values():
    s, _ = get_fixed_suppliers()
    t = ContractTerms(price=120.0, delivery_weeks=4.0, quality_tier=3)
    buyer = BuyerPreferences()
    assert buyer.value(t) == pytest.approx(150 + 0.5 * 60 * (8 - 4) / 8 + 0.5 * 50 * 3 / 5)
    assert buyer.score(t) == pytest.approx(60.0)
    assert s.profit(t) == pytest.approx(120.0 - (85 + 0.5 * 16 + 1.8 * 9))


def test_joint_surplus_identity_price_cancels():
    _, quality = _buyers()
    _, q = get_fixed_suppliers()
    for price in (90.0, 120.0, 150.0):
        t = ContractTerms(price=price, delivery_weeks=7.5, quality_tier=5)
        assert quality.score(t) + q.profit(t) == pytest.approx(quality.value(t) - q.cost(t))
    t = ContractTerms(price=100.0, delivery_weeks=7.5, quality_tier=5)
    assert quality.value(t) - q.cost(t) == pytest.approx(compute_efficient_outcome(quality, q)[0])


def test_cost_description_says_a_loser_earns_nothing_and_names_no_outside_option():
    for supplier in get_fixed_suppliers():
        text = supplier.describe_for_agent()
        assert "outside option" not in text.lower()
        assert "If you don't win, you earn nothing from this event.\n\n" in text
        assert not hasattr(supplier, "outside_option")


def test_neither_buyer_description_mentions_a_price_range():
    for sc in get_contract_scenarios():
        for text in (sc.buyer.describe_for_platform(), sc.buyer.describe_for_platform_detailed()):
            assert "$100" not in text and "180" not in text and "price $" not in text
            assert re.search(r"Acceptable ranges: delivery 1.8 weeks, quality tier 2.5\.", text), text


# The cost description that the bidder of the second-price auction reads, worked out by hand. Speed specialist: base
# 85, delivery 0.5 * (8 - weeks)^2, quality 1.8 * tier^2; quality specialist: 1.8 * (8 - weeks)^2 and 0.4 * tier^2.
# For example 2wk/T2 costs 85 + 0.5 * 36 + 1.8 * 4 = 110.2 for the first and 85 + 1.8 * 36 + 0.4 * 4 = 151.4 for the
# second; the cheapest configuration is 8wk/T2 and the most expensive 1wk/T5.
SEALED_BID_TAIL = """Your total cost = base + delivery cost + quality cost.
If you win, your profit is the price you are paid minus your total cost at the terms you offered.
If you don't win, you earn nothing from this event."""
SEALED_BID_DESCRIPTIONS = {
    "S": """YOUR PRIVATE COST STRUCTURE (only you know this):
Base production cost: $85.0

DELIVERY COST (faster = more expensive, quadratic):
    1wk: +$24.5
    2wk: +$18.0
    3wk: +$12.5
    4wk: +$8.0
    5wk: +$4.5
    6wk: +$2.0
    7wk: +$0.5
    8wk: +$0.0

QUALITY COST (higher tier = more expensive, quadratic):
    T2: +$7.2
    T3: +$16.2
    T4: +$28.8
    T5: +$45.0

EXAMPLE TOTAL COSTS (base + delivery + quality):
    2wk/T2 → total cost $110.2
    3wk/T3 → total cost $113.7
    4wk/T4 → total cost $121.8
    5wk/T5 → total cost $134.5
    6wk/T2 → total cost $94.2
    2wk/T5 → total cost $148.0
    7wk/T3 → total cost $101.7

COST RANGE: Your cheapest possible configuration (8wk/T2) costs $92.2. Your most expensive (1wk/T5) costs $154.5.

""" + SEALED_BID_TAIL,
    "Q": """YOUR PRIVATE COST STRUCTURE (only you know this):
Base production cost: $85.0

DELIVERY COST (faster = more expensive, quadratic):
    1wk: +$88.2
    2wk: +$64.8
    3wk: +$45.0
    4wk: +$28.8
    5wk: +$16.2
    6wk: +$7.2
    7wk: +$1.8
    8wk: +$0.0

QUALITY COST (higher tier = more expensive, quadratic):
    T2: +$1.6
    T3: +$3.6
    T4: +$6.4
    T5: +$10.0

EXAMPLE TOTAL COSTS (base + delivery + quality):
    2wk/T2 → total cost $151.4
    3wk/T3 → total cost $133.6
    4wk/T4 → total cost $120.2
    5wk/T5 → total cost $111.2
    6wk/T2 → total cost $93.8
    2wk/T5 → total cost $159.8
    7wk/T3 → total cost $90.4

COST RANGE: Your cheapest possible configuration (8wk/T2) costs $86.6. Your most expensive (1wk/T5) costs $183.2.

""" + SEALED_BID_TAIL,
}


def test_the_sealed_bid_cost_description_is_exactly_this_for_each_cost_type():
    for supplier in get_fixed_suppliers():
        assert supplier.describe_for_sealed_bid() == SEALED_BID_DESCRIPTIONS[supplier.supplier_id]


def test_the_sealed_bid_description_has_the_tables_of_the_other_one_and_none_of_its_advice():
    for supplier in get_fixed_suppliers():
        sealed, ai = supplier.describe_for_sealed_bid(), supplier.describe_for_agent()
        assert sealed.split("COST RANGE:")[0] == ai.split("COST RANGE:")[0]            # the same tables
        assert "ANY bid below" not in sealed and "guaranteed below cost" not in sealed
        assert "Your profit = bid price" not in sealed        # untrue when the winner is paid the second score
        for advice in ("MUST", "STRATEGIC NOTE", "Play to YOUR", "generous", "bid above"):
            assert advice in ai and advice not in sealed, advice            # the other description has each of them
        assert "margin" not in sealed.lower()


def test_the_cost_description_of_the_ai_institutions_is_as_it_was():
    s, q = get_fixed_suppliers()
    for supplier, rest in ((s, "$92.2. Your most expensive (1wk/T5) costs $154.5. "),
                           (q, "$86.6. Your most expensive (1wk/T5) costs $183.2. ")):
        text = supplier.describe_for_agent()
        assert ("costs " + rest + "ANY bid below ") in text
        assert text.endswith(
            "Your total cost = base + delivery cost + quality cost.\n"
            "Your profit = bid price - total cost. You MUST bid above your total cost.\n"
            "If you don't win, you earn nothing from this event.\n\n"
            "STRATEGIC NOTE: Buyers generally prefer faster delivery and higher quality, but each buyer weighs "
            "these differently. Play to YOUR cost advantages \u2014 offer generous terms where YOU are cheap, and "
            "don't waste money on dimensions that are expensive for you.")
