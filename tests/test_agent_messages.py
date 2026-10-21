"""What the agents read: the mediator's message inside a run, nothing of one supplier's bids in the other
supplier's messages, and the mark on a price the below-cost rule set."""
import json
from types import SimpleNamespace

import pytest

from orgsim import prompts as P
from orgsim.institutions import run_mediator
from orgsim.llm import ChatClient
from orgsim.llm_agents import (
    LocalBuyerAgentPlatform, LocalPlatformMediator, LocalProcurementAnalyst, LocalSupplierAgent,
)
from orgsim.mocks import MockPlatformMediator, MockStrategicSupplier
from orgsim.organization import ANALYST_ORGANIZATIONS, MEMO_HEADER
from orgsim.primitives import get_contract_scenarios, get_fixed_suppliers
from orgsim.sequence import SUPPLIER_IDS, pre_draw_event_sequence, run_sequence


class Replies:
    """Fake OpenAI client: reply k is replies[k] (the last repeats); records every request."""

    def __init__(self, *replies):
        self.replies, self.calls = replies, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        content = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
        msg = SimpleNamespace(content=content, tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")])

    def user_messages(self):
        return [c["messages"][1]["content"] for c in self.calls]


def test_mediator_message_in_a_run_has_both_histories_and_no_reply_from_the_same_round():
    # The mediator's real message inside run_mediator; the two halves are also tested apart.
    s, q = get_fixed_suppliers()
    sc = get_contract_scenarios()[0]
    fake = Replies("R1 to S", "R1 to Q", "R2 to S", "R2 to Q")
    agents = {"S": MockStrategicSupplier(seed=1), "Q": MockStrategicSupplier(seed=2)}
    out = run_mediator(1, sc.buyer, [s, q], agents, LocalPlatformMediator(ChatClient(fake), "m", "MEDIATOR"),
                       79.8, "S", num_rounds=3, buyer_description="DESC",
                       mediator_cross_event_history={"S": "MEMORY S", "Q": "MEMORY Q"})
    r1_s, r1_q, r2_s, r2_q = fake.user_messages()
    assert all(m.startswith("MEMORY Q\n\nMEMORY S\n\nDESC\n\n") for m in (r1_s, r1_q, r2_s, r2_q))
    assert "BID HISTORY" not in r1_s + r1_q                                     # no history that repeats the bids
    assert "Your feedback" not in r1_s + r1_q                                   # nothing answered yet
    assert "BID HISTORY FOR SUPPLIER Q (THIS SUPPLIER):\n" in r2_q and "BID HISTORY FOR SUPPLIER S:\n" in r2_q
    for msg in (r2_s, r2_q):
        assert "Your feedback (R1): R1 to S" in msg and "Your feedback (R1): R1 to Q" in msg
    assert "R2 to S" not in r2_q                                                # written earlier in this round
    assert out.mediator_feedback == {"S": ["R1 to S", "R2 to S"], "Q": ["R1 to Q", "R2 to Q"]}


# No code path puts one supplier's bid into a message that the other supplier's agents read.
BIDS = {"A": {"price": 171.37, "delivery_weeks": 4.3, "quality_tier": 3, "reasoning": "REASON-OF-A"},
        "B": {"price": 163.59, "delivery_weeks": 5.7, "quality_tier": 4, "reasoning": "REASON-OF-B"}}


@pytest.mark.parametrize("organization", ["single", "organized", "organized_guided"])
@pytest.mark.parametrize("regime", ["Mediator", "BuyerAgent"])
def test_no_supplier_reads_the_other_suppliers_bids(regime, organization):
    bidder = {sid: Replies(json.dumps(BIDS[sid])) for sid in SUPPLIER_IDS}
    analyst = {sid: Replies("The record is short. Posture: GUARDED.") for sid in SUPPLIER_IDS}
    platform = Replies("Consider your terms.")
    suppliers = {sid: LocalSupplierAgent(ChatClient(bidder[sid]), "m", P.executive_prompt(organization))
                 for sid in SUPPLIER_IDS}
    analysts = ({sid: LocalProcurementAnalyst(ChatClient(analyst[sid]), "m", P.ANALYST_PROMPT)
                 for sid in SUPPLIER_IDS} if organization in ANALYST_ORGANIZATIONS else None)
    plan = pre_draw_event_sequence(3, get_contract_scenarios(), seed=7)
    run_sequence(regime, 3, 3, suppliers, plan, analysts=analysts, organization=organization,
                 mediator=LocalPlatformMediator(ChatClient(platform), "m", P.MEDIATOR_PROMPT),
                 buyer_agent_platform=LocalBuyerAgentPlatform(ChatClient(platform), "m", P.BUYER_AGENT_PROMPT))
    platform_text = " ".join(platform.user_messages())
    assert "171.37" in platform_text and "163.59" in platform_text      # the platform does see both
    for sid, other in (("A", "B"), ("B", "A")):
        read = [m.get("content") or "" for fake in (bidder[sid], analyst[sid])
                for call in fake.calls for m in call["messages"]]
        assert read and not any(f"{BIDS[other]['price']:.2f}" in t or BIDS[other]["reasoning"] in t
                                for t in read)


HISTORY_HEADER = "YOUR PAST PROCUREMENT HISTORY WITH THIS PLATFORM:"


def test_the_guided_bidder_reads_the_note_and_the_guidance_and_never_the_record():
    # An LLM-agent run of the guided organized firm with fake clients: what each agent is sent.
    bidder = {sid: Replies(json.dumps(BIDS[sid])) for sid in SUPPLIER_IDS}
    analyst = {sid: Replies("The record is short. Posture: GUARDED.") for sid in SUPPLIER_IDS}
    suppliers = {sid: LocalSupplierAgent(ChatClient(bidder[sid]), "m", P.executive_prompt("organized_guided"),
                                         use_tools=False) for sid in SUPPLIER_IDS}
    analysts = {sid: LocalProcurementAnalyst(ChatClient(analyst[sid]), "m", P.ANALYST_PROMPT) for sid in SUPPLIER_IDS}
    plan = pre_draw_event_sequence(3, get_contract_scenarios(), seed=7)
    results = run_sequence("Mediator", 3, 2, suppliers, plan, analysts=analysts, organization="organized_guided",
                           mediator=MockPlatformMediator())
    for sid in SUPPLIER_IDS:
        systems = {call["messages"][0]["content"] for call in bidder[sid].calls}
        assert systems == {P.executive_prompt("organized_guided")}
        (system,) = systems
        assert P.GUIDED_NOTE in system and P.ANALYST_GUIDANCE in system and P.ORGANIZED_NOTE in system
        assert P.SINGLE_NOTE not in system
        users = bidder[sid].user_messages()
        assert len(users) == 3 * 2                                              # events x rounds
        assert not any(HISTORY_HEADER in m for m in users)                      # the bidder never receives the record
        for k in (2, 3):                                                        # from event 2 it reads the memo
            memo = MEMO_HEADER + results[k - 1]["analyst_briefings"][sid]
            assert all(memo in m for m in users[2 * (k - 1):2 * k])
        assert not any(MEMO_HEADER in m for m in users[:2])                     # in event 1 there is nothing to read
        # the analyst runs before events 2 and 3, with the analyst prompt, and is sent the record
        assert len(analyst[sid].calls) == 2
        assert {call["messages"][0]["content"] for call in analyst[sid].calls} == {P.ANALYST_PROMPT}
        assert all(HISTORY_HEADER in m for m in analyst[sid].user_messages())


# A price the below-cost rule set is marked where the bidder reads its earlier rounds, as in the record.
CHEAP = json.dumps({"price": 90.0, "delivery_weeks": 3.0, "quality_tier": 2, "reasoning": "cheap"})
OK = json.dumps({"price": 160.0, "delivery_weeks": 3.0, "quality_tier": 2, "reasoning": "ok"})


def test_a_rule_set_price_is_marked_in_the_next_rounds_message():
    fake = Replies(CHEAP, CHEAP, OK)       # round 1: below cost twice, so the rule sets the price
    agents = {"A": LocalSupplierAgent(ChatClient(fake), "m", "S", use_tools=False),
              "B": MockStrategicSupplier(seed=1)}
    plan = pre_draw_event_sequence(1, get_contract_scenarios(), seed=7)
    run_sequence("Mediator", 1, 2, agents, plan, mediator=MockPlatformMediator(), organization="single")
    round_2 = fake.user_messages()[2]
    line = next(l for l in round_2.splitlines() if l.strip().startswith("Round 1: You bid"))
    assert "(price set by rule: cost + 5%)" in line


def test_a_price_the_model_set_is_not_marked_in_the_next_rounds_message():
    fake = Replies(OK)
    agents = {"A": LocalSupplierAgent(ChatClient(fake), "m", "S", use_tools=False),
              "B": MockStrategicSupplier(seed=1)}
    plan = pre_draw_event_sequence(1, get_contract_scenarios(), seed=7)
    run_sequence("Mediator", 1, 2, agents, plan, mediator=MockPlatformMediator(), organization="single")
    round_2 = fake.user_messages()[1]
    line = next(l for l in round_2.splitlines() if l.strip().startswith("Round 1: You bid"))
    assert "price set by rule" not in line and line.strip().startswith("Round 1: You bid $160.00, ")
