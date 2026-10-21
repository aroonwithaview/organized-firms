import hashlib
import re

import pytest

from orgsim.institutions import run_buyer_agent, run_first_price, run_mediator, run_second_price
from orgsim.interfaces import RULE_PRICE_TAG, SupplierAgent
from orgsim.mocks import MockBuyerAgentPlatform, MockPlatformMediator, MockStrategicSupplier
from orgsim.primitives import (ContractTerms, SupplierBid, compute_max_efficient_surplus,
                               get_contract_scenarios, get_fixed_suppliers)

TOL = 0.05  # bids may sit between frontier grid points (0.1 weeks)


def _setup():
    s, q = get_fixed_suppliers()
    sc = get_contract_scenarios()[0]
    eff, winner, _ = compute_max_efficient_surplus(sc.buyer, [s, q])
    agents = {s.supplier_id: MockStrategicSupplier(seed=1), q.supplier_id: MockStrategicSupplier(seed=2)}
    return sc, [s, q], agents, eff, winner


def test_mediator_runs_and_respects_frontier():
    sc, sups, agents, eff, winner = _setup()
    out = run_mediator(1, sc.buyer, sups, agents, MockPlatformMediator(), eff, winner,
                       num_rounds=3, buyer_description=sc.description_for_platform)
    assert out.regime == "Mediator"
    assert out.joint_surplus <= eff + TOL
    assert set(out.mediator_feedback) == {s.supplier_id for s in sups}


def test_buyer_agent_runs_and_respects_frontier():
    sc, sups, agents, eff, winner = _setup()
    out = run_buyer_agent(1, sc.buyer, sups, agents, MockBuyerAgentPlatform(), eff, winner,
                          num_rounds=3, buyer_description=sc.description_for_platform)
    assert out.regime == "BuyerAgent"
    assert out.joint_surplus <= eff + TOL


def test_sealed_bid_institutions_run():
    sc, sups, agents, eff, winner = _setup()
    for fn, label in ((run_first_price, "FirstPrice"), (run_second_price, "SecondPrice")):
        out = fn(run_id=1, buyer=sc.buyer, suppliers=sups, agents=agents,
                 efficient_surplus=eff, efficient_winner_id=winner)
        assert out.regime == label
        assert out.joint_surplus <= eff + TOL


class Scripted(SupplierAgent):
    """Bids the given terms, one per round (the last repeats); records each call's history length."""

    def __init__(self, *terms):
        self.terms, self.history_lengths = terms, []

    def generate_bid(self, supplier_costs, regime_context, round_num, history):
        self.history_lengths.append(len(history))
        t = self.terms[min(round_num, len(self.terms)) - 1]
        return SupplierBid(supplier_costs.supplier_id,
                           ContractTerms(t.price, t.delivery_weeks, t.quality_tier), round_num=round_num)


def _speed_buyer_and_pair():
    s, q = get_fixed_suppliers()
    return get_contract_scenarios()[0].buyer, [s, q]


def test_first_price_pays_the_bid_second_price_pays_up_to_the_second_score():
    # Speed buyer: value = 150 + 6 * (8 - weeks) + 2 * tier.
    # S bids $140, 2 wk, T2: value 190, score 50, cost 85 + 0.5*36 + 1.8*4 = 110.2.
    # Q bids $130, 6 wk, T4: value 170, score 40.
    buyer, sups = _speed_buyer_and_pair()
    bids = {"S": ContractTerms(140.0, 2.0, 2), "Q": ContractTerms(130.0, 6.0, 4)}
    fp, sp = (fn(run_id=1, buyer=buyer, suppliers=sups,
                 agents={sid: Scripted(t) for sid, t in bids.items()},
                 efficient_surplus=79.8, efficient_winner_id="S")
              for fn in (run_first_price, run_second_price))
    assert fp.winner_id == sp.winner_id == "S"
    assert fp.winning_bid.price == pytest.approx(140.0)            # paid its bid
    assert fp.buyer_surplus == pytest.approx(50.0)
    assert sp.winning_bid.price == pytest.approx(190.0 - 40.0)     # value minus the second-best score
    assert sp.buyer_surplus == pytest.approx(40.0)                 # the buyer gets the second-best score
    assert (sp.winning_bid.delivery_weeks, sp.winning_bid.quality_tier) == (2.0, 2)
    assert sp.supplier_profit == pytest.approx(150.0 - 110.2)
    assert fp.joint_surplus == pytest.approx(sp.joint_surplus) == pytest.approx(190.0 - 110.2)


@pytest.mark.parametrize("run, platform", [(run_mediator, MockPlatformMediator),
                                            (run_buyer_agent, MockBuyerAgentPlatform)])
def test_only_final_bids_decide_the_award(run, platform):
    # S's first bid would win (score 90), but its final bid scores 5; Q's final bid scores 40.
    buyer, sups = _speed_buyer_and_pair()
    agents = {"S": Scripted(ContractTerms(100.0, 2.0, 2), ContractTerms(185.0, 2.0, 2)),
              "Q": Scripted(ContractTerms(160.0, 6.0, 4), ContractTerms(130.0, 6.0, 4))}
    out = run(1, buyer, sups, agents, platform(), 79.8, "S", num_rounds=2)
    assert out.winner_id == "Q"
    assert out.winning_bid == ContractTerms(130.0, 6.0, 4)
    assert [len(out.all_bids[sid]) for sid in ("S", "Q")] == [2, 2]


ABOVE_VALUE = ContractTerms(250.0, 3.0, 3)    # the speed buyer's value for 3.0 weeks and tier 3 is 186

AI_INSTITUTIONS = [(run_mediator, MockPlatformMediator), (run_buyer_agent, MockBuyerAgentPlatform)]


@pytest.mark.parametrize("run, platform", AI_INSTITUTIONS)
def test_buyer_walks_away_when_every_final_bid_is_priced_above_its_value(run, platform):
    buyer, sups = _speed_buyer_and_pair()
    agents = {"S": Scripted(ABOVE_VALUE), "Q": Scripted(ABOVE_VALUE)}
    out = run(1, buyer, sups, agents, platform(), 79.8, "S", num_rounds=2)
    assert out.winner_id is None and out.winning_bid is None and out.awarded is False
    assert out.buyer_surplus == out.supplier_profit == out.joint_surplus == out.winner_true_cost == 0.0
    assert [len(out.all_bids[sid]) for sid in ("S", "Q")] == [2, 2]          # the bids are still recorded
    assert set(out.all_true_costs) == set(out.all_inflation_rates) == {"S", "Q"}


@pytest.mark.parametrize("run, platform", AI_INSTITUTIONS)
def test_a_final_bid_with_score_exactly_zero_is_awarded(run, platform):
    buyer, sups = _speed_buyer_and_pair()
    at_value = ContractTerms(0.0, 3.0, 3)
    at_value.price = buyer.value(at_value)                                   # price equal to value: score 0
    agents = {"S": Scripted(at_value), "Q": Scripted(ABOVE_VALUE)}
    out = run(1, buyer, sups, agents, platform(), 79.8, "S", num_rounds=2)
    assert out.winner_id == "S" and out.awarded is True and out.buyer_surplus == 0.0


@pytest.mark.parametrize("run, platform", AI_INSTITUTIONS)
def test_with_one_bid_above_value_and_one_below_the_one_below_wins(run, platform):
    buyer, sups = _speed_buyer_and_pair()
    agents = {"S": Scripted(ABOVE_VALUE), "Q": Scripted(ContractTerms(130.0, 6.0, 4))}
    out = run(1, buyer, sups, agents, platform(), 79.8, "S", num_rounds=2)
    assert out.winner_id == "Q" and out.awarded is True and out.buyer_surplus == pytest.approx(40.0)


@pytest.mark.parametrize("run", [run_first_price, run_second_price])
def test_sealed_bid_institutions_award_even_when_both_scores_are_negative(run):
    buyer, sups = _speed_buyer_and_pair()
    agents = {"S": Scripted(ContractTerms(250.0, 3.0, 3)), "Q": Scripted(ContractTerms(260.0, 3.0, 3))}
    out = run(run_id=1, buyer=buyer, suppliers=sups, agents=agents,
              efficient_surplus=79.8, efficient_winner_id="S")
    assert out.winner_id == "S" and out.awarded is True and out.winning_bid is not None
    assert out.buyer_surplus < 0


class DescriptionRecordingMediator(MockPlatformMediator):
    def __init__(self):
        self.descriptions = []

    def generate_feedback(self, buyer_prefs, all_bids, target_supplier_id, round_num,
                          buyer_description="", **kw):
        self.descriptions.append((target_supplier_id, buyer_description))
        return super().generate_feedback(buyer_prefs, all_bids, target_supplier_id, round_num,
                                         buyer_description, **kw)


@pytest.mark.parametrize("memory, expected", [
    ({"S": "MEMORY S", "Q": "MEMORY Q"}, "MEMORY Q\n\nMEMORY S\n\nDESC"),      # by supplier id, for either target
    ({"S": "MEMORY S"}, "MEMORY S\n\nDESC"),                                    # a supplier without memory adds none
    (None, "DESC"),
])
def test_mediator_gets_the_memory_of_every_supplier(memory, expected):
    buyer, sups = _speed_buyer_and_pair()
    agents = {"S": Scripted(ContractTerms(140.0, 2.0, 2)), "Q": Scripted(ContractTerms(130.0, 6.0, 4))}
    mediator = DescriptionRecordingMediator()
    run_mediator(1, buyer, sups, agents, mediator, 79.8, "S", num_rounds=2, buyer_description="DESC",
                 mediator_cross_event_history=memory)
    assert sorted(mediator.descriptions) == [("Q", expected), ("S", expected)]


def test_mediator_gets_the_buyers_priorities_after_the_memories_when_no_description_is_given():
    buyer, sups = _speed_buyer_and_pair()
    agents = {"S": Scripted(ContractTerms(140.0, 2.0, 2)), "Q": Scripted(ContractTerms(130.0, 6.0, 4))}
    mediator = DescriptionRecordingMediator()
    run_mediator(1, buyer, sups, agents, mediator, 79.8, "S", num_rounds=2,
                 mediator_cross_event_history={"S": "MEMORY S", "Q": "MEMORY Q"})
    expected = "MEMORY Q\n\nMEMORY S\n\n" + buyer.describe_for_platform()
    assert "Priority ranking" in expected                                      # the buyer's priorities, not just memories
    assert sorted(mediator.descriptions) == [("Q", expected), ("S", expected)]


class RecordingMediator(MockPlatformMediator):
    def __init__(self):
        self.calls = []

    def generate_feedback(self, buyer_prefs, all_bids, target_supplier_id, round_num, **kw):
        self.calls.append((target_supplier_id, round_num))
        return super().generate_feedback(buyer_prefs, all_bids, target_supplier_id, round_num, **kw)


class RecordingBuyerAgent(MockBuyerAgentPlatform):
    def __init__(self):
        self.calls = []

    def generate_counteroffer_feedback(self, buyer_prefs, supplier_bid, supplier_id, round_num, history, **kw):
        self.calls.append((supplier_id, round_num))
        return super().generate_counteroffer_feedback(buyer_prefs, supplier_bid, supplier_id, round_num, history, **kw)


@pytest.mark.parametrize("run, platform_cls", [(run_mediator, RecordingMediator),
                                                (run_buyer_agent, RecordingBuyerAgent)])
def test_platform_answers_every_round_but_the_last(run, platform_cls):
    buyer, sups = _speed_buyer_and_pair()
    agents = {"S": Scripted(ContractTerms(140.0, 2.0, 2)), "Q": Scripted(ContractTerms(130.0, 6.0, 4))}
    platform = platform_cls()
    out = run(1, buyer, sups, agents, platform, 79.8, "S", num_rounds=3)
    assert sorted(platform.calls) == [("Q", 1), ("Q", 2), ("S", 1), ("S", 2)]
    assert all(len(out.mediator_feedback[sid]) == 2 for sid in ("S", "Q"))
    assert all(agents[sid].history_lengths == [0, 1, 2] for sid in ("S", "Q"))


def test_sealed_bid_institutions_need_two_suppliers():
    sc, sups, agents, eff, winner = _setup()
    for fn in (run_first_price, run_second_price):
        with pytest.raises(ValueError):
            fn(run_id=1, buyer=sc.buyer, suppliers=sups[:1], agents=agents,
               efficient_surplus=eff, efficient_winner_id=winner)


@pytest.mark.parametrize("run, platform", [(run_mediator, MockPlatformMediator),
                                            (run_buyer_agent, MockBuyerAgentPlatform)])
def test_ai_institutions_need_at_least_two_rounds(run, platform):
    sc, sups, agents, eff, winner = _setup()
    for rounds in (0, 1):
        with pytest.raises(ValueError, match="at least 2 rounds"):
            run(1, sc.buyer, sups, agents, platform(), eff, winner, num_rounds=rounds)
    out = run(1, sc.buyer, sups, agents, platform(), eff, winner, num_rounds=2)
    assert [len(out.all_bids[s.supplier_id]) for s in sups] == [2, 2]


class RecordingSupplier(MockStrategicSupplier):
    """A mock supplier that keeps the `regime_context` of every call."""

    def __init__(self, seed):
        super().__init__(seed=seed)
        self.contexts = []

    def generate_bid(self, supplier_costs, regime_context, round_num, history):
        self.contexts.append(regime_context)
        return super().generate_bid(supplier_costs, regime_context, round_num, history)


AI_ROUNDS = 3  # rounds of the two AI institutions; the sealed-bid institutions have one
RULES_ROUNDS = {"first_price": 1, "second_price": 1, "mediator": AI_ROUNDS, "buyer_agent": AI_ROUNDS}
EXPECTED_RULES_DIGESTS = {
    "first_price": "28aeae45eac9f73d",
    "second_price": "f1fdf673a4e95dea",       # the pricing sentence was corrected (see SECOND_PRICE_RULES)
    "mediator": "e3f01249638f7045",
    "buyer_agent": "8ba0fb8c1ab0fd3a",
}


def _run_recording(name):
    """Run one institution on the fixed pair with no cross-event history; return the two suppliers."""
    sc, sups, _, eff, winner = _setup()
    agents = {s.supplier_id: RecordingSupplier(seed=i) for i, s in enumerate(sups, start=1)}
    common = dict(run_id=1, buyer=sc.buyer, suppliers=sups, agents=agents,
                  efficient_surplus=eff, efficient_winner_id=winner)
    ai = dict(num_rounds=AI_ROUNDS, buyer_description=sc.description_for_platform)
    if name == "first_price":
        run_first_price(**common)
    elif name == "second_price":
        run_second_price(**common)
    elif name == "mediator":
        run_mediator(mediator=MockPlatformMediator(), **common, **ai)
    elif name == "buyer_agent":
        run_buyer_agent(buyer_agent=MockBuyerAgentPlatform(), **common, **ai)
    else:
        raise ValueError(name)
    return list(agents.values())


def test_rules_texts_are_unchanged():
    # Each institution sends one text to both suppliers in every round; the digests pin that text.
    got = {}
    for name, rounds in RULES_ROUNDS.items():
        recorders = _run_recording(name)
        assert [len(r.contexts) for r in recorders] == [rounds, rounds], name
        texts = {c for r in recorders for c in r.contexts}
        assert len(texts) == 1, name
        got[name] = hashlib.sha256(texts.pop().encode("utf-8")).hexdigest()[:16]
    assert got == EXPECTED_RULES_DIGESTS, "re-check what suppliers are told"


def test_buyer_agent_rules_say_the_platform_sees_the_bids_of_this_event():
    (text,) = {c for r in _run_recording("buyer_agent") for c in r.contexts}
    assert "can see all your previous bids in this event and will use that information in its feedback." in text


class FeedbackRecordingMediator(MockPlatformMediator):
    """Replies with a distinct text per supplier and round; keeps a copy of the earlier feedback it was given."""

    def __init__(self):
        self.calls = []

    def generate_feedback(self, buyer_prefs, all_bids, target_supplier_id, round_num,
                          buyer_description="", previous_bids=None, previous_feedback=None):
        earlier = {sid: list(texts) for sid, texts in previous_feedback.items()}
        self.calls.append((round_num, target_supplier_id, earlier))
        return f"reply to {target_supplier_id} in round {round_num}"


def test_every_mediator_call_in_a_round_sees_only_the_feedback_of_earlier_rounds():
    buyer, sups = _speed_buyer_and_pair()
    agents = {"S": Scripted(ContractTerms(140.0, 2.0, 2)), "Q": Scripted(ContractTerms(130.0, 6.0, 4))}
    mediator = FeedbackRecordingMediator()
    out = run_mediator(1, buyer, sups, agents, mediator, 79.8, "S", num_rounds=3)
    seen = {(round_num, sid): earlier for round_num, sid, earlier in mediator.calls}
    assert sorted(seen) == [(1, "Q"), (1, "S"), (2, "Q"), (2, "S")]
    assert seen[(1, "S")] == seen[(1, "Q")] == {"S": [], "Q": []}
    round_1 = {"S": ["reply to S in round 1"], "Q": ["reply to Q in round 1"]}
    assert seen[(2, "S")] == round_1                  # the first supplier is answered first in round 2
    assert seen[(2, "Q")] == round_1                  # the second one's call does not hold "reply to S in round 2"
    assert out.mediator_feedback == {"S": ["reply to S in round 1", "reply to S in round 2"],
                                     "Q": ["reply to Q in round 1", "reply to Q in round 2"]}  # every reply is kept


class Marking(Scripted):
    """Scripted; keeps a copy of each history it is given, and tags its bids in `tagged_rounds` as the
    below-cost rule does."""

    def __init__(self, *terms, tagged_rounds=()):
        super().__init__(*terms)
        self.tagged_rounds, self.histories = tagged_rounds, []

    def generate_bid(self, supplier_costs, regime_context, round_num, history):
        self.histories.append([dict(h) for h in history])
        bid = super().generate_bid(supplier_costs, regime_context, round_num, history)
        if round_num in self.tagged_rounds:
            bid.reasoning += RULE_PRICE_TAG
        return bid


@pytest.mark.parametrize("run, platform", AI_INSTITUTIONS)
def test_the_history_says_which_bids_had_their_price_set_by_the_rule(run, platform):
    buyer, sups = _speed_buyer_and_pair()
    agents = {"S": Marking(ContractTerms(140.0, 2.0, 2), tagged_rounds=(1,)),
              "Q": Marking(ContractTerms(130.0, 6.0, 4))}
    run(1, buyer, sups, agents, platform(), 79.8, "S", num_rounds=3)
    marks = {sid: [[h["price_set_by_rule"] for h in history] for history in agent.histories]
             for sid, agent in agents.items()}
    assert marks == {"S": [[], [True], [True, False]],            # the histories of rounds 1, 2 and 3
                     "Q": [[], [False], [False, False]]}


# The second-price rules text, as the bidder of that auction reads it. Every sentence is true of run_second_price: the
# highest score wins, and the winner is paid the highest price at which its score is still at least the other bid's
# score, which is at least the price it bid (the next tests). An earlier version said "you will NOT be paid the price
# you bid", which is untrue when the two scores are equal, and "the most you can be paid while still winning".
SECOND_PRICE_RULES = (
    "RULES (second-price auction): You submit one bid. Both suppliers bid at the same time. The buyer scores each bid "
    "based on all terms (price, delivery, quality). The highest-scoring bid wins.\n\n"
    "PRICING: If you win, you are paid the highest price at which your terms would still score at least as high as "
    "the other supplier's bid. This price is at least the price you bid.\n\n"
    "There are no further rounds.\n\n"
    "This is a one-time procurement event. Your bid does not affect any future interactions.")


def test_the_second_price_rules_text_is_exactly_this_and_gives_no_advice_and_names_no_platform():
    (text,) = {c for r in _run_recording("second_price") for c in r.contexts}
    assert text == SECOND_PRICE_RULES
    platform = r"\b(platforms?|facilitator|mediator|feedback|analyst|briefing|history|record|past)\b"
    advice = r"\b(should|must|never|always|avoid|strateg\w+|margin\w*|competit\w+|best|trust\w*)\b"
    assert not re.search(platform, text, re.I) and not re.search(advice, text, re.I)
    assert "NOT" not in text                                # the old text's claim that the bid price is never paid


@pytest.mark.parametrize("price", [60.0, 100.0, 139.0, 150.0])
def test_the_second_price_paid_is_the_same_whatever_the_winner_bid_and_never_less_than_its_bid(price):
    # Speed buyer: S offers 2 wk, T2 (value 190); Q bids $130, 6 wk, T4 (value 170, score 40). S wins at any price
    # up to 150, where its score is 40 too and, tied, the first supplier wins; it is paid 190 - 40 = 150 each time.
    buyer, sups = _speed_buyer_and_pair()
    agents = {"S": Scripted(ContractTerms(price, 2.0, 2)), "Q": Scripted(ContractTerms(130.0, 6.0, 4))}
    out = run_second_price(run_id=1, buyer=buyer, suppliers=sups, agents=agents,
                           efficient_surplus=79.8, efficient_winner_id="S")
    assert out.winner_id == "S"
    assert out.winning_bid.price == pytest.approx(150.0) and out.winning_bid.price >= price
    assert out.all_bids["S"][0].terms.price == price                 # the submitted price is kept beside the price paid
    assert out.supplier_profit == pytest.approx(150.0 - 110.2)       # cost: 85 + 0.5 * 36 + 1.8 * 4


def test_the_second_price_auction_pays_the_price_at_which_the_winner_scores_what_the_runner_up_scores():
    # Now Q offers 6 wk, T4 at $100 (score 70) and S offers 2 wk, T2 at $140 (score 50). Q wins and is paid
    # 170 - 50 = 120, which is above its bid and above its cost 85 + 1.8 * 4 + 0.4 * 16 = 98.6.
    buyer, sups = _speed_buyer_and_pair()
    agents = {"S": Scripted(ContractTerms(140.0, 2.0, 2)), "Q": Scripted(ContractTerms(100.0, 6.0, 4))}
    out = run_second_price(run_id=1, buyer=buyer, suppliers=sups, agents=agents,
                           efficient_surplus=79.8, efficient_winner_id="S")
    assert out.winner_id == "Q" and out.winning_bid.price == pytest.approx(120.0)
    assert out.supplier_profit == pytest.approx(120.0 - 98.6) and out.buyer_surplus == pytest.approx(50.0)


@pytest.mark.parametrize("bidder", ["S", "Q"])
def test_a_price_equal_to_the_cost_at_the_offered_terms_is_never_beaten_whatever_the_rival_bids(bidder):
    # The benchmark S: in run_second_price, the bidder's profit is the same at every price at which it wins (it is
    # paid the price at which its score equals the rival's), and a price equal to its cost wins exactly where winning
    # pays. A lower price only adds wins at a loss, a higher one only loses profitable wins. Checked on a grid of
    # terms, rival bids (below, at and above the rival's cost, with negative scores too) and prices, for the supplier
    # listed first (which wins ties) and the one listed second.
    buyer, sups = _speed_buyer_and_pair()
    own = next(s for s in sups if s.supplier_id == bidder)
    rival_id = "Q" if bidder == "S" else "S"
    rivals = [ContractTerms(price, weeks, tier) for price in (40.0, 100.0, 130.0, 170.0, 240.0)
              for weeks, tier in ((6.0, 4), (3.0, 2), (8.0, 2), (1.0, 5))]
    for weeks, tier in ((2.0, 2), (4.0, 3), (6.0, 4), (7.5, 5), (8.0, 2)):
        cost = own.cost(ContractTerms(0.0, weeks, tier))
        for rival in rivals:
            def profit(price):
                agents = {bidder: Scripted(ContractTerms(price, weeks, tier)), rival_id: Scripted(rival)}
                out = run_second_price(run_id=1, buyer=buyer, suppliers=sups, agents=agents,
                                       efficient_surplus=79.8, efficient_winner_id="S")
                return out.supplier_profit if out.winner_id == bidder else 0.0

            at_cost = profit(cost)
            for price in (0.5 * cost, 0.9 * cost, 0.99 * cost, 1.01 * cost, 1.1 * cost, 1.5 * cost, cost + 40, 400.0):
                assert profit(price) <= at_cost + 1e-9, (bidder, weeks, tier, rival, price)
