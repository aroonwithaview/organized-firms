"""The rules that the primitives set shows a bidder (orgsim.institutions.primitives_rules), sentence by sentence,
against what the code does: the rounds and the replies, the final bid, the score, the payment, the walk-away, and
what each platform sees and remembers. Each test names the sentence it checks."""
import re
from types import SimpleNamespace

import pytest

from orgsim import prompts as P
from orgsim.institutions import primitives_rules, run_buyer_agent, run_mediator
from orgsim.interfaces import BuyerAgentPlatform, PlatformMediator, SupplierAgent
from orgsim.llm import ChatClient
from orgsim.llm_agents import LocalBuyerAgentPlatform, LocalPlatformMediator
from orgsim.primitives import ContractTerms, SupplierBid, get_contract_scenarios
from orgsim.sequence import SUPPLIER_IDS, make_supplier, pre_draw_event_sequence, run_sequence

SCENARIOS = get_contract_scenarios()
SPEED_PRIORITY, QUALITY_PRIORITY = SCENARIOS
INSTITUTIONS = ("Mediator", "BuyerAgent")


class Scripted(SupplierAgent):
    """Bids as scripted by round: `terms[round_num - 1]`; keeps what it was told in each round."""

    def __init__(self, *terms):
        self.terms, self.told = terms, []

    def generate_bid(self, supplier_costs, regime_context, round_num, history):
        self.told.append((round_num, regime_context, [h["round"] for h in history]))
        return SupplierBid(supplier_costs.supplier_id, self.terms[min(round_num, len(self.terms)) - 1], "", round_num)


class RecordingMediator(PlatformMediator):
    def __init__(self):
        self.calls = []

    def generate_feedback(self, buyer_prefs, all_bids, target_supplier_id, round_num, buyer_description="",
                          previous_bids=None, previous_feedback=None):
        self.calls.append(SimpleNamespace(
            buyer=buyer_prefs, all_bids=dict(all_bids), target=target_supplier_id, round=round_num,
            description=buyer_description, previous_bids={k: list(v) for k, v in (previous_bids or {}).items()},
            previous_feedback={k: list(v) for k, v in (previous_feedback or {}).items()}))
        return f"mediator reply to {target_supplier_id} after round {round_num}"


class RecordingBuyerAgent(BuyerAgentPlatform):
    def __init__(self):
        self.calls = []

    def generate_counteroffer_feedback(self, buyer_prefs, supplier_bid, supplier_id, round_num, history,
                                       buyer_description=""):
        self.calls.append(SimpleNamespace(buyer=buyer_prefs, bid=supplier_bid, target=supplier_id, round=round_num,
                                          history=[dict(h) for h in history], description=buyer_description))
        return f"buyer agent reply to {supplier_id} after round {round_num}"


def play(institution, bids, rounds=3, events=2, scenario=0, seed=7, organization="single", **kwargs):
    """Run `events` events of the primitives set with scripted bidders; `bids` = {supplier: (terms by round, ...)}."""
    agents = {sid: Scripted(*bids[sid]) for sid in SUPPLIER_IDS}
    mediator, buyer_agent = RecordingMediator(), RecordingBuyerAgent()
    plan = pre_draw_event_sequence(max(events, 2), SCENARIOS, seed=seed)
    plan = [(SCENARIOS[scenario], a, b) for _, a, b in plan]
    results = run_sequence(institution, events, rounds, agents, plan, mediator=mediator,
                           buyer_agent_platform=buyer_agent, organization=organization, prompt_set="primitives",
                           **kwargs)
    return results, agents, mediator, buyer_agent, plan


def terms(price, weeks, tier):
    return ContractTerms(price=price, delivery_weeks=weeks, quality_tier=tier)


def value(buyer, weeks, tier):
    """The buyer's value written out: base + w_d * beta_d * (8 - weeks) / 8 + w_q * beta_q * tier / 5."""
    return (buyer.base_value + buyer.w_delivery * buyer.beta_delivery * (8 - weeks) / 8
            + buyer.w_quality * buyer.beta_quality * tier / 5)


# ---- "Each event has N rounds. ... after rounds 1 to N-1 you receive a reply from the platform. Your bid in
#      round N is your final bid." ----

def _rounds_named(text):
    """The rounds that the rules say get a reply, read from the sentence."""
    named = re.search(r"after (round 1|rounds 1 and 2|rounds 1 to ([0-9]+)) you receive a reply", text).group(1)
    return {"round 1": [1], "rounds 1 and 2": [1, 2]}.get(named, list(range(1, int(named.rsplit(" ", 1)[1]) + 1)))


@pytest.mark.parametrize("rounds", [2, 3, 4, 6])
@pytest.mark.parametrize("institution", INSTITUTIONS)
def test_an_event_has_the_stated_number_of_rounds_and_a_reply_follows_exactly_the_rounds_the_rules_name(
        institution, rounds):
    flat = (terms(130.0, 5.0, 3),)
    results, agents, mediator, buyer_agent, _ = play(institution, {"A": flat, "B": flat}, rounds=rounds, events=2)
    rules = primitives_rules(institution, rounds)
    assert f"Each event has {rounds} rounds." in rules and f"Your bid in round {rounds} is your final bid." in rules
    named = _rounds_named(rules)
    assert named == list(range(1, rounds))                                   # all rounds but the last
    for sid in SUPPLIER_IDS:
        told = agents[sid].told
        assert [r for r, _, _ in told] == list(range(1, rounds + 1)) * 2        # a bid in every round of both events
        assert [len(earlier) for _, _, earlier in told] == list(range(rounds)) * 2   # the earlier rounds it sees
    calls = mediator.calls if institution == "Mediator" else buyer_agent.calls
    for event_calls in (calls[:len(calls) // 2], calls[len(calls) // 2:]):
        for sid in SUPPLIER_IDS:
            assert sorted(c.round for c in event_calls if c.target == sid) == named   # a reply after exactly those
    outcome = results[0]["outcome"]
    assert {sid: len(replies) for sid, replies in outcome.mediator_feedback.items()} == {"A": rounds - 1,
                                                                                           "B": rounds - 1}
    assert {sid: len(bids) for sid, bids in outcome.all_bids.items()} == {"A": rounds, "B": rounds}


@pytest.mark.parametrize("institution", INSTITUTIONS)
def test_only_final_bids_decide_the_award_and_earlier_bids_and_the_replies_do_not(institution):
    """'Only final bids are used to decide the award; earlier bids and the replies are not.' (both institutions)"""
    final = {"A": terms(140.0, 4.0, 3), "B": terms(150.0, 3.0, 4)}
    def play_with_early(early_a, early_b):
        return play(institution, {"A": (early_a, early_a, final["A"]), "B": (early_b, early_b, final["B"])})[0][0]

    one = play_with_early(terms(100.0, 8.0, 2), terms(100.0, 1.0, 5))
    other = play_with_early(terms(500.0, 1.0, 5), terms(20.0, 8.0, 2))
    for field in ("winner", "awarded", "buyer_surplus", "supplier_profit", "joint_surplus", "efficiency",
                  "score_margin_pct"):
        assert one[field] == other[field], field
    assert one["outcome"].winning_bid == other["outcome"].winning_bid
    assert one["outcome"].all_true_costs == other["outcome"].all_true_costs     # costs at the final terms
    scores = {sid: SPEED_PRIORITY.buyer.score(final[sid]) for sid in final}
    assert one["winner"] == max(scores, key=scores.get)


# ---- "A final bid's score is the buyer's value for its delivery and quality minus its price; the higher-scoring final
#      bid wins, and the winner is paid its price." ----

@pytest.mark.parametrize("institution", INSTITUTIONS)
@pytest.mark.parametrize("scenario", [0, 1], ids=["Speed Priority", "Quality Priority"])
def test_the_higher_score_wins_and_is_paid_its_price_whatever_the_prices_and_the_terms(institution, scenario):
    buyer = SCENARIOS[scenario].buyer
    # A is cheaper in price and in terms; B offers more for the buyer. Neither wins by price alone.
    final = {"A": terms(110.0, 6.0, 2), "B": terms(125.0, 2.0, 5)}
    scores = {sid: value(buyer, t.delivery_weeks, t.quality_tier) - t.price for sid, t in final.items()}
    assert scores["A"] != scores["B"]
    expected = max(scores, key=scores.get)
    result, _, _, _, plan = play(institution, {sid: (final[sid],) for sid in final}, scenario=scenario)
    event = result[0]
    assert buyer.score(final["A"]) == pytest.approx(scores["A"])
    assert buyer.score(final["B"]) == pytest.approx(scores["B"])
    assert event["winner"] == expected and event["awarded"]
    won = final[expected]
    outcome = event["outcome"]
    assert outcome.winning_bid.price == won.price                              # paid its price, not the other's
    cost = make_supplier(expected, plan[0][1] if expected == "A" else plan[0][2]).cost(won)
    assert outcome.supplier_profit == pytest.approx(won.price - cost)          # its profit: price minus total cost
    assert outcome.buyer_surplus == pytest.approx(scores[expected])
    assert outcome.winner_true_cost == pytest.approx(cost)


@pytest.mark.parametrize("institution", INSTITUTIONS)
def test_a_tie_in_score_goes_to_the_first_supplier_which_the_rules_do_not_state(institution):
    buyer = SPEED_PRIORITY.buyer
    same = terms(120.0, 5.0, 3)
    equal_score = {"A": (same,), "B": (terms(120.0, 5.0, 3),)}
    assert play(institution, equal_score)[0][0]["winner"] == "A"
    assert not re.search(r"\b(tie|ties|tied|tiebreak)\b", primitives_rules(institution, 3).lower())
    assert buyer.score(same) == buyer.score(terms(120.0, 5.0, 3))


# ---- "BUYER'S OUTSIDE OPTION: ... if both final bids are priced above the buyer's value for their terms, no
#      contract is awarded." ----

@pytest.mark.parametrize("institution", INSTITUTIONS)
@pytest.mark.parametrize("scenario", [0, 1], ids=["Speed Priority", "Quality Priority"])
def test_no_contract_is_awarded_if_both_final_bids_are_priced_above_the_buyers_value(institution, scenario):
    buyer = SCENARIOS[scenario].buyer
    above = {"A": terms(value(buyer, 4.0, 3) + 0.01, 4.0, 3), "B": terms(value(buyer, 6.0, 5) + 40.0, 6.0, 5)}
    event = play(institution, {sid: (above[sid],) for sid in above}, scenario=scenario, events=2)[0]
    first = event[0]
    assert first["winner"] is None and first["awarded"] is False
    assert (first["buyer_surplus"], first["supplier_profit"], first["joint_surplus"]) == (0.0, 0.0, 0.0)
    assert first["outcome"].winning_bid is None
    assert all(buyer.score(t) < 0 for t in above.values())


@pytest.mark.parametrize("institution", INSTITUTIONS)
def test_one_bid_above_the_value_does_not_stop_an_award_to_the_other_and_a_bid_at_the_value_is_awarded(institution):
    buyer = SPEED_PRIORITY.buyer
    above = terms(value(buyer, 4.0, 3) + 5.0, 4.0, 3)
    below = terms(value(buyer, 6.0, 4) - 10.0, 6.0, 4)
    first = play(institution, {"A": (above,), "B": (below,)})[0][0]
    assert first["winner"] == "B" and first["awarded"]                           # one bid above is no walk-away
    at_value = terms(value(buyer, 4.0, 3), 4.0, 3)
    assert buyer.score(at_value) == 0.0
    second = play(institution, {"A": (above,), "B": (at_value,)})[0][0]
    assert second["winner"] == "B" and second["awarded"]                         # a score of 0 is accepted


@pytest.mark.parametrize("institution", INSTITUTIONS)
def test_the_record_of_a_walk_away_says_so_and_nobody_is_paid(institution):
    buyer = SPEED_PRIORITY.buyer
    above = {sid: terms(value(buyer, 5.0, 3) + 30.0, 5.0, 3) for sid in SUPPLIER_IDS}
    results, agents, _, _, _ = play(institution, {sid: (above[sid],) for sid in above}, events=2)
    assert results[0]["awarded"] is False
    second_event_context = agents["A"].told[3][1]            # event 2 (rounds=3): the first round's context of A
    assert "Result: No contract was awarded; the buyer walked away." in second_event_context


# ---- "The platform ... sees every supplier's bids in the event and the buyer's scoring, and it remembers past events
#      on this platform." (the mediator) ----

def test_the_mediator_sees_every_suppliers_bids_in_the_event_the_buyers_scoring_and_the_past_events():
    early = {"A": (terms(150.0, 5.0, 3), terms(145.0, 5.0, 3), terms(140.0, 5.0, 3)),
             "B": (terms(151.0, 4.0, 4), terms(146.0, 4.0, 4), terms(141.0, 4.0, 4))}
    results, _, mediator, _, plan = play("Mediator", early, events=2)
    assert len(mediator.calls) == 2 * 2 * 2                                     # 2 events x 2 rounds with feedback x 2
    for call in mediator.calls:
        assert set(call.all_bids) == {"A", "B"}                                 # every supplier's current bid
        assert set(call.previous_bids) == {"A", "B"}                            # and every supplier's bids so far
        assert len(call.previous_bids["A"]) == len(call.previous_bids["B"]) == call.round
        assert "BUYER SCORING FUNCTION" in call.description and "value = 150" in call.description
    first_event, second_event = mediator.calls[:4], mediator.calls[4:]
    assert not any("PLATFORM MEMORY" in c.description for c in first_event)     # nothing to remember yet
    for call in second_event:
        assert "PLATFORM MEMORY — SUPPLIER A" in call.description
        assert "PLATFORM MEMORY — SUPPLIER B" in call.description
        assert "Round 1: Bid $150.00, 5.0wk, tier 3" in call.description         # the earlier event's bids, of A
        assert "Round 1: Bid $151.00, 4.0wk, tier 4" in call.description         # and of B
        assert "Outcome:" in call.description
    assert results[1]["event"] == 2


def test_the_mediator_prompt_says_it_works_for_both_and_the_rules_say_it_is_instructed_to():
    assert "You work for BOTH the buyer and the suppliers." in P.MEDIATOR_PROMPT
    assert primitives_rules("Mediator", 3).count("The platform is instructed to work for both the buyer and the "
                                                 "suppliers.") == 1


# ---- "The platform is instructed to work on behalf of the buyer, to get the best deal for the buyer. It knows the
#      buyer's priorities and sees your bids in this event." (the buyer's agent) ----

def test_the_buyer_agent_knows_the_priorities_and_sees_only_the_suppliers_own_bids_in_the_event():
    early = {"A": (terms(150.0, 5.0, 3), terms(145.0, 5.0, 3), terms(140.0, 5.0, 3)),
             "B": (terms(151.0, 4.0, 4), terms(146.0, 4.0, 4), terms(141.0, 4.0, 4))}
    _, _, _, buyer_agent, plan = play("BuyerAgent", early, events=2)
    assert len(buyer_agent.calls) == 2 * 2 * 2
    for call in buyer_agent.calls:
        assert call.description == SPEED_PRIORITY.description_for_platform        # the buyer's priorities (ordinal)
        assert "Priority ranking: delivery speed >> quality" in call.description
        assert call.bid == early[call.target][call.round - 1]                     # this supplier's current bid
        assert [h["bid"] for h in call.history] == list(early[call.target][:call.round - 1])   # its earlier bids, only
        assert all(h["round"] < call.round for h in call.history)
        assert "PLATFORM MEMORY" not in call.description                          # no past event is passed on
    assert "acting as the BUYER'S AGENT" in P.BUYER_AGENT_PROMPT
    assert "Your goal is to get the best possible deal for the buyer" in P.BUYER_AGENT_PROMPT
    assert "It knows the buyer's priorities and sees your bids in this event." in primitives_rules("BuyerAgent", 3)


def test_the_buyer_agents_message_holds_the_suppliers_own_bids_and_none_of_the_other_suppliers():
    """'When it replies to you, it does not see the other supplier's bids.' (the buyer agent's rules only: the
    platform as a whole sees the other supplier's bids, when it replies to that supplier)."""
    class Replies:
        def __init__(self):
            self.calls = []
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kw):
            self.calls.append(kw)
            message = SimpleNamespace(content="a reply", tool_calls=None)
            return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])

    fake = Replies()
    agent = LocalBuyerAgentPlatform(ChatClient(fake), "m", P.BUYER_AGENT_PROMPT)
    bids = {"A": (terms(150.25, 5.0, 3), terms(145.75, 5.0, 3), terms(140.5, 5.0, 3)),
            "B": (terms(161.0, 4.0, 4), terms(166.0, 4.0, 4), terms(171.0, 4.0, 4))}
    agents = {sid: Scripted(*bids[sid]) for sid in SUPPLIER_IDS}
    suppliers = [make_supplier("A", "speed"), make_supplier("B", "quality")]
    run_buyer_agent(1, SPEED_PRIORITY.buyer, suppliers, agents, agent, 50.0, "A", num_rounds=3,
                    buyer_description=SPEED_PRIORITY.description_for_platform, rules="RULES")
    texts = [call["messages"][1]["content"] for call in fake.calls]
    assert len(texts) == 4
    for text, (sid, other) in zip(texts, (("A", "B"), ("B", "A"), ("A", "B"), ("B", "A"))):
        assert f"Supplier {sid}'s current bid" in text
        for t in bids[other]:
            assert f"${t.price:.2f}" not in text                                  # nothing of the other supplier's bids
    assert "Round 1: $150.25" in texts[2] and "Round 1: $161.00" in texts[3]       # each sees its own earlier bid
    for text, (sid, other) in zip(texts, (("A", "B"), ("B", "A"), ("A", "B"), ("B", "A"))):
        assert f"Supplier {other}" not in text                                    # and the other is not even named
        for t in bids[other]:                                                     # nor any term of one of its bids
            assert f"{t.delivery_weeks}wk, tier {t.quality_tier}" not in text
    sentence = "When it replies to you, it does not see the other supplier's bids."
    assert sentence in primitives_rules("BuyerAgent", 3) and "other supplier's bids" not in primitives_rules(
        "Mediator", 3)
    assert primitives_rules("BuyerAgent", 3).count("It does not see past events. " + sentence) == 1


class EventPriced(SupplierAgent):
    """Bids a price that says its event and round: 100 + 10 * event + round, plus `extra`. The event is counted from
    the round-1 calls."""

    def __init__(self, extra):
        self.extra, self.event = extra, 0

    def generate_bid(self, supplier_costs, regime_context, round_num, history):
        if round_num == 1:
            self.event += 1
        return SupplierBid(supplier_costs.supplier_id, terms(100.0 + 10 * self.event + round_num + self.extra, 5.0, 3),
                           "", round_num)


def test_the_buyer_agent_does_not_see_past_events():
    """'It does not see past events.' (the buyer's agent rules): in a run of three events, what the agent is sent in an
    event holds bids of that event only, and nothing of the record or the platform's memory."""
    class Replies:
        def __init__(self):
            self.calls = []
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kw):
            self.calls.append(kw)
            message = SimpleNamespace(content="a reply", tool_calls=None)
            return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])

    fake = Replies()
    agent = LocalBuyerAgentPlatform(ChatClient(fake), "m", P.BUYER_AGENT_PROMPT)
    agents = {"A": EventPriced(0.0), "B": EventPriced(0.5)}
    run_sequence("BuyerAgent", 3, 3, agents, pre_draw_event_sequence(3, SCENARIOS, seed=7), buyer_agent_platform=agent,
                 organization="single", prompt_set="primitives")
    assert len(fake.calls) == 3 * 2 * 2                                  # events x suppliers x rounds with feedback
    for index, call in enumerate(fake.calls):
        assert call["messages"][0]["content"] == P.BUYER_AGENT_PROMPT
        user = call["messages"][1]["content"]
        prices = {float(price) for price in re.findall(r"[$]([0-9]+[.][0-9]{2})", user)}
        assert {int((price - 100) // 10) for price in prices} == {index // 4 + 1}     # every bid is of this event
        assert not re.search(r"PLATFORM MEMORY|Result:|Event [0-9]|previous procurement|past event", user)
    # while the mediator is sent the memory of the earlier events, as its rules say
    assert "remembers past events" in primitives_rules("Mediator", 3)
    assert "past events" not in primitives_rules("BuyerAgent", 3).replace("It does not see past events.", "")


def test_what_the_mediators_message_holds_is_what_the_rules_say_it_sees():
    class Replies:
        def __init__(self):
            self.calls = []
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kw):
            self.calls.append(kw)
            message = SimpleNamespace(content="a reply", tool_calls=None)
            return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])

    fake = Replies()
    mediator = LocalPlatformMediator(ChatClient(fake), "m", P.MEDIATOR_PROMPT)
    bids = {"A": (terms(150.25, 5.0, 3), terms(145.75, 5.0, 3)), "B": (terms(161.0, 4.0, 4), terms(166.0, 4.0, 4))}
    agents = {sid: Scripted(*bids[sid]) for sid in SUPPLIER_IDS}
    suppliers = [make_supplier("A", "speed"), make_supplier("B", "quality")]
    run_mediator(1, SPEED_PRIORITY.buyer, suppliers, agents, mediator, 50.0, "A", num_rounds=2,
                 buyer_description=SPEED_PRIORITY.description_for_platform_detailed,
                 mediator_cross_event_history={"A": "PLATFORM MEMORY — SUPPLIER A: x",
                                               "B": "PLATFORM MEMORY — SUPPLIER B: y"},
                 rules="RULES")
    for call in fake.calls:
        text = call["messages"][1]["content"]
        assert "$150.25" in text and "$161.00" in text                              # every supplier's current bid
        assert "BUYER SCORING FUNCTION" in text                                     # the buyer's scoring
        assert "PLATFORM MEMORY — SUPPLIER A" in text and "PLATFORM MEMORY — SUPPLIER B" in text   # its memory


# ---- "The contract that is up is not named when the event starts." (random) and "The contract is not named when the
#      first event starts." (persistent); "One other supplier takes part in each event; your company is not told its
#      costs or its bids. The same other supplier takes part in every event." ----

def test_the_bidders_context_names_no_contract_and_no_other_supplier_bid_before_the_rules():
    flat = {"A": (terms(150.25, 5.0, 3),), "B": (terms(161.0, 4.0, 4),)}
    _, agents, _, _, plan = play("Mediator", flat, events=2, rounds=2)
    assert len(SUPPLIER_IDS) == 2                                                   # "one other supplier"
    names = [sc.name for sc in SCENARIOS]
    for sid, other in (("A", "B"), ("B", "A")):
        for round_num, context, _ in agents[sid].told:
            first_part = context.split("RULES:", 1)[0]
            if "YOUR COMPANY'S RECORD OF PAST EVENTS" in first_part:
                first_part = first_part.split("YOUR COMPANY'S RECORD OF PAST EVENTS", 1)[0]    # before the record
            assert not any(name in first_part for name in names), (sid, round_num)     # nothing names the contract
            assert f"${flat[other][0].price:.2f}" not in context                         # the other's bid is not in it
            assert not re.search(r"Supplier [AB]", context)                              # and the other is not named


def test_no_text_of_the_first_event_names_the_contract_in_either_environment():
    """'The contract that is up is not named when the event starts.' (random) / 'The contract is not named when the
    first event starts.' (persistent): what a bidder is sent in event 1 has no name of a contract, whatever the plan."""
    flat = {"A": (terms(150.25, 5.0, 3),), "B": (terms(161.0, 4.0, 4),)}
    names = [sc.name for sc in SCENARIOS]
    for environment in ("random", "persistent"):
        plan = pre_draw_event_sequence(2, SCENARIOS, seed=7, environment=environment)
        agents = {sid: Scripted(*flat[sid]) for sid in SUPPLIER_IDS}
        run_sequence("Mediator", 2, 2, agents, plan, mediator=RecordingMediator(), organization="single",
                     prompt_set="primitives")
        for sid in SUPPLIER_IDS:
            first_event = [context for round_num, context, _ in agents[sid].told[:2]]
            assert len(first_event) == 2 and not any(name in context for name in names for context in first_event)


def test_the_same_other_supplier_takes_part_in_every_event():
    """'One other supplier takes part in each event ... The same other supplier takes part in every event.': the
    runner uses the same two supplier agents, A and B, in every event of a sequence; each is asked for a bid in every
    round of every event, and each event's bids are those of A and B and of nobody else."""
    class Remembers(Scripted):
        def __init__(self, *terms):
            super().__init__(*terms)
            self.events_seen, self.served = [], []

        def generate_bid(self, supplier_costs, regime_context, round_num, history):
            self.events_seen.append(regime_context.split("\n\n", 1)[0])           # the event line opens the context
            self.served.append(supplier_costs.supplier_id)                         # and the supplier it bids as
            return super().generate_bid(supplier_costs, regime_context, round_num, history)

    for institution in INSTITUTIONS:
        for environment in ("random", "persistent"):
            agents = {sid: Remembers(terms(130.0 + i, 5.0, 3)) for i, sid in enumerate(SUPPLIER_IDS)}
            plan = pre_draw_event_sequence(4, SCENARIOS, seed=3, environment=environment)
            results = run_sequence(institution, 4, 2, agents, plan, mediator=RecordingMediator(),
                                   buyer_agent_platform=RecordingBuyerAgent(), organization="single",
                                   prompt_set="primitives")
            assert set(agents) == {"A", "B"}
            for sid, agent in agents.items():                                    # the very objects, in all 4 events
                assert agent.events_seen == [f"EVENT {event} of 4" for event in (1, 2, 3, 4) for _ in (1, 2)], sid
                assert agent.served == [sid] * 8, sid                          # and always as the same supplier
            for result in results:
                assert set(result["outcome"].all_bids) == {"A", "B"}
                assert all(len(bids) == 2 and all(bid.supplier_id == sid for bid in bids)
                           for sid, bids in result["outcome"].all_bids.items())
            assert [len(agent.told) for agent in agents.values()] == [8, 8]    # a bid in every round of all 4 events
