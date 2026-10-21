import hashlib
import io
import json
import re
from types import SimpleNamespace

import pytest

from orgsim import prompts as P
from orgsim.llm import ChatClient, make_client
from orgsim.llm_agents import (
    COST_CALCULATOR_TOOL, COST_CALCULATOR_TOOL_SEALED, NO_TOOL_TAG, RULE_PRICE_TAG, SEALED_NO_TOOL_ROUND_TEXT,
    SEALED_TOOL_ROUND_TEXT, THINK_BLOCK, api_call_with_retry, execute_cost_tool,
    LocalBuyerAgentPlatform, LocalPlatformMediator, LocalProcurementAnalyst, LocalSupplierAgent,
)
from orgsim.organization import AnalystBriefing
from orgsim.primitives import ContractTerms, get_contract_scenarios, get_fixed_suppliers


class FakeBase:
    """Stands in for an OpenAI client; records every request."""

    def __init__(self, content):
        self.calls, self._content = [], content
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        msg = SimpleNamespace(content=self._content, tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")])


def test_supplier_agent_uses_given_model_and_prompt_and_parses_bid():
    s, _ = get_fixed_suppliers()
    base = FakeBase(json.dumps({"price": 150.0, "delivery_weeks": 3.0, "quality_tier": 2, "reasoning": "ok"}))
    agent = LocalSupplierAgent(ChatClient(base), model="test-model", system_prompt="SYS", use_tools=False)
    bid = agent.generate_bid(s, "RULES: test", round_num=1, history=[])
    assert (bid.terms.price, bid.terms.delivery_weeks, bid.terms.quality_tier) == (150.0, 3.0, 2)
    assert base.calls[0]["model"] == "test-model"
    assert base.calls[0]["messages"][0] == {"role": "system", "content": "SYS"}


def test_analyst_classifies_posture_from_text():
    base = FakeBase("The platform acknowledged our strength when we won. Posture: TRUSTING.")
    analyst = LocalProcurementAnalyst(ChatClient(base), model="m", system_prompt="ANALYST")
    b = analyst.generate_briefing("A", "Event 1 (Speed Priority): ...")
    assert b.posture == AnalystBriefing.TRUSTING
    assert base.calls[0]["messages"][0]["content"] == "ANALYST"


def test_analyst_is_called_only_with_a_record():
    analyst = LocalProcurementAnalyst(ChatClient(FakeBase("unused")), model="m", system_prompt="ANALYST")
    with pytest.raises(ValueError):
        analyst.generate_briefing("A", "")


def test_analyst_is_not_told_the_upcoming_contract():
    history = "Event 1 (Speed Priority): ..."
    base = FakeBase("The record is thin. Posture: GUARDED.")
    LocalProcurementAnalyst(ChatClient(base), model="m", system_prompt="ANALYST").generate_briefing("A", history)
    user_message = base.calls[0]["messages"][1]["content"]
    assert history in user_message
    assert "upcoming" not in user_message.lower()
    for sc in get_contract_scenarios():
        if sc.name not in history:
            assert sc.name not in user_message


def test_hosted_mode_drops_local_sampling_fields():
    base = FakeBase("{}")
    ChatClient(base, local=False).chat.completions.create(model="m", messages=[], extra_body={"top_k": 20})
    assert "extra_body" not in base.calls[0]


def test_local_mode_keeps_local_sampling_fields():
    base = FakeBase("{}")
    ChatClient(base, local=True).chat.completions.create(model="m", messages=[], extra_body={"top_k": 20})
    assert base.calls[0]["extra_body"] == {"top_k": 20}


# Failure behavior, bidder parity and per-agent model and prompt.
BID = json.dumps({"price": 150.0, "delivery_weeks": 3.0, "quality_tier": 2, "reasoning": "ok"})


class Scripted:
    """Fake OpenAI client: replies with `contents` in turn (the last repeats);
    `tool_calls[i]` attaches tool calls to reply i; an Exception content is raised."""

    def __init__(self, *contents, tool_calls=None):
        self.contents, self.tool_calls, self.calls = list(contents), tool_calls or {}, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        i = len(self.calls)
        self.calls.append(kw)
        content = self.contents[min(i, len(self.contents) - 1)]
        if isinstance(content, Exception):
            raise content
        msg = SimpleNamespace(content=content, tool_calls=self.tool_calls.get(i))
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")])


def _tool_call(delivery, tier):
    return SimpleNamespace(id="c1", function=SimpleNamespace(
        name="compute_my_cost", arguments=json.dumps({"delivery_weeks": delivery, "quality_tier": tier})))


# 1. Parity: the bidder's messages depend on the organization only through the system prompt.
def test_bidder_messages_differ_by_organization_only_in_the_system_prompt():
    s, _ = get_fixed_suppliers()
    ctx = "PAST EVENTS: none yet.\n\nRULES: test"
    history = [{"round": 1, "bid": ContractTerms(130.0, 3.0, 2), "feedback": "The field is close."}]
    sent = {}
    for org in P.ORGANIZATIONS:
        fake = Scripted("", BID, tool_calls={0: [_tool_call(3, 2)]})
        agent = LocalSupplierAgent(ChatClient(fake), model="m", system_prompt=P.executive_prompt(org))
        agent.generate_bid(s, ctx, round_num=2, history=history)
        sent[org] = [c["messages"] for c in fake.calls]
    single = sent["single"]
    assert set(sent) == set(P.ORGANIZATIONS) and len(P.ORGANIZATIONS) == 3
    for org in P.ORGANIZATIONS:
        assert [m[1:] for m in sent[org]] == [m[1:] for m in single], org
    assert len({m[0]["content"] for org in P.ORGANIZATIONS for m in sent[org]}) == 3      # the system prompts do differ
    user_text = " ".join(m["content"] for m in single[-1][1:] if m["role"] != "system").replace(ctx, "")
    assert not re.search(r"\b(analyst|briefing|memo)\b", user_text, re.I)


# 2. Every agent sends the model and system prompt it was given (the tool path included).
def test_every_agent_sends_its_model_and_system_prompt():
    s, q = get_fixed_suppliers()
    sc = get_contract_scenarios()[0]
    bids = {"S": ContractTerms(140.0, 3.0, 2), "Q": ContractTerms(145.0, 6.0, 4)}
    fakes = {k: Scripted("", BID, tool_calls={0: [_tool_call(3, 2)]}) if k == "supplier" else Scripted("text")
             for k in ("supplier", "mediator", "buyer_agent", "analyst")}
    LocalSupplierAgent(ChatClient(fakes["supplier"]), model="M1", system_prompt="P1").generate_bid(s, "R", 1, [])
    LocalPlatformMediator(ChatClient(fakes["mediator"]), model="M2", system_prompt="P2").generate_feedback(
        sc.buyer, bids, "S", 1)
    LocalBuyerAgentPlatform(ChatClient(fakes["buyer_agent"]), model="M3", system_prompt="P3") \
        .generate_counteroffer_feedback(sc.buyer, bids["S"], "S", 1, [])
    LocalProcurementAnalyst(ChatClient(fakes["analyst"]), model="M4", system_prompt="P4").generate_briefing(
        "S", "Event 1 (Speed Priority): ...")
    for (name, fake), (model, prompt) in zip(fakes.items(), [("M1", "P1"), ("M2", "P2"), ("M3", "P3"), ("M4", "P4")]):
        assert fake.calls, name
        for call in fake.calls:
            assert call["model"] == model, name
            assert call["messages"][0] == {"role": "system", "content": prompt}, name


# 3. A generation failure is never replaced by scripted content.
def test_analyst_failure_raises_instead_of_writing_a_briefing():
    for content in (RuntimeError("server unavailable"), None):
        analyst = LocalProcurementAnalyst(ChatClient(Scripted(content)), model="m", system_prompt="A")
        with pytest.raises(Exception):
            analyst.generate_briefing("A", "Event 1 (Speed Priority): ...")


@pytest.mark.parametrize("content", [
    "I would bid about 150.",                                       # never valid JSON
    json.dumps({"delivery_weeks": 3, "reasoning": "fast"}),         # fields missing
    json.dumps({"price": "N/A", "delivery_weeks": 2, "quality_tier": 4}),
])
@pytest.mark.parametrize("use_tools", [False, True])
def test_bidder_failure_raises_instead_of_bidding_defaults(content, use_tools):
    s, _ = get_fixed_suppliers()
    agent = LocalSupplierAgent(ChatClient(Scripted(content)), model="m", system_prompt="S", use_tools=use_tools)
    with pytest.raises(Exception):
        agent.generate_bid(s, "RULES: test", round_num=1, history=[])


# A reply without the bid fields is followed by a second call; an empty analyst reply raises.
NOT_A_BID = {
    "tool call written as text": json.dumps(
        {"name": "compute_my_cost", "arguments": {"delivery_weeks": 3, "quality_tier": 2}}),
    "price only": json.dumps({"price": 150.0}),
}


@pytest.mark.parametrize("reply", list(NOT_A_BID.values()), ids=list(NOT_A_BID))
@pytest.mark.parametrize("use_tools", [False, True])
def test_reply_without_the_bid_fields_is_followed_by_a_second_call(reply, use_tools):
    s, _ = get_fixed_suppliers()
    fake = Scripted(reply, BID)
    agent = LocalSupplierAgent(ChatClient(fake), model="m", system_prompt="S", use_tools=use_tools)
    bid = agent.generate_bid(s, "RULES: test", round_num=1, history=[])
    assert len(fake.calls) == 2
    assert (bid.terms.price, bid.terms.delivery_weeks, bid.terms.quality_tier) == (150.0, 3.0, 2)
    assert bid.reasoning == "[STATED]: ok"  # no tag: the model made this bid


@pytest.mark.parametrize("content", ["", " \n ", None])
def test_empty_analyst_reply_raises_value_error(content):
    analyst = LocalProcurementAnalyst(ChatClient(Scripted(content)), model="m", system_prompt="A")
    with pytest.raises(ValueError):
        analyst.generate_briefing("A", "Event 1 (Speed Priority): ...")


# Bids that code shaped carry a tag in their reasoning.
CHEAP = json.dumps({"price": 90.0, "delivery_weeks": 3.0, "quality_tier": 2, "reasoning": "cheap"})


@pytest.mark.parametrize("use_tools", [False, True])
def test_price_below_cost_on_both_attempts_is_set_by_rule_and_marked(use_tools):
    s, _ = get_fixed_suppliers()
    agent = LocalSupplierAgent(ChatClient(Scripted(CHEAP)), model="m", system_prompt="S", use_tools=use_tools)
    bid = agent.generate_bid(s, "RULES: test", round_num=1, history=[])
    cost = s.cost(ContractTerms(0, 3.0, 2))
    assert bid.terms.price == round(cost * 1.05, 2)
    assert (bid.terms.delivery_weeks, bid.terms.quality_tier) == (3.0, 2)
    assert bid.reasoning.startswith("[STATED]: cheap") and bid.reasoning.endswith(RULE_PRICE_TAG)
    assert NO_TOOL_TAG not in bid.reasoning


def test_price_corrected_on_the_second_attempt_is_not_marked():
    s, _ = get_fixed_suppliers()
    fake = Scripted(CHEAP, BID)
    agent = LocalSupplierAgent(ChatClient(fake), model="m", system_prompt="S", use_tools=False)
    bid = agent.generate_bid(s, "RULES: test", round_num=1, history=[])
    assert len(fake.calls) == 2 and "CORRECTION" in fake.calls[1]["messages"][1]["content"]
    assert bid.terms.price == 150.0
    assert bid.reasoning == "[STATED]: ok"


def _client_that_fails_when_tools_are_offered(error):
    """Fake client: a request that offers tools raises `error`; any other request gets BID."""
    calls = []

    def create(**kw):
        calls.append(kw)
        if "tools" in kw:
            raise RuntimeError(error)
        msg = SimpleNamespace(content=BID, tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")])

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), calls


@pytest.mark.parametrize("error", ["tool server error", "HTTP 400 bad request"])
def test_bid_made_without_the_cost_tool_after_a_tool_failure_is_marked(error, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda seconds: None)  # the 400 path pauses between tries
    s, _ = get_fixed_suppliers()
    fake, calls = _client_that_fails_when_tools_are_offered(error)
    agent = LocalSupplierAgent(ChatClient(fake), model="m", system_prompt="S", use_tools=True)
    bid = agent.generate_bid(s, "RULES: test", round_num=1, history=[])
    assert bid.terms.price == 150.0
    assert bid.reasoning == "[STATED]: ok" + NO_TOOL_TAG
    assert "tools" in calls[0] and "tools" not in calls[-1]


# make_client takes the key from the environment (no request is made).
def test_make_client_reads_the_key_from_the_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "key-from-env")
    assert make_client("http://localhost:1234/v1")._base.api_key == "key-from-env"


def test_make_client_prefers_a_passed_key_and_defaults_to_a_placeholder(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "key-from-env")
    assert make_client("http://localhost:1234/v1", api_key="passed")._base.api_key == "passed"
    monkeypatch.delenv("OPENAI_API_KEY")
    assert make_client("http://localhost:1234/v1")._base.api_key == "not-needed"


# A bid outside the contract space (finite positive price, delivery 1 to 8 weeks, whole quality
# tier 2 to 5) is unusable: the model is asked again, and after three attempts the run stops.
def _bid_json(**changes):
    fields = {"price": 150.0, "delivery_weeks": 3.0, "quality_tier": 2, "reasoning": "ok"}
    fields.update(changes)
    return json.dumps(fields)


OUTSIDE_THE_SPACE = {
    "NaN price": _bid_json(price=float("nan")),
    "infinite price": _bid_json(price=float("inf")),
    "zero price": _bid_json(price=0),
    "negative price": _bid_json(price=-5),
    "delivery 0.5": _bid_json(delivery_weeks=0.5),
    "delivery 9": _bid_json(delivery_weeks=9),
    "negative delivery": _bid_json(delivery_weeks=-4),
    "tier 1": _bid_json(quality_tier=1),
    "tier 6": _bid_json(quality_tier=6),
    "tier 9": _bid_json(quality_tier=9),
    "tier 3.5": _bid_json(quality_tier=3.5),
    'price "N/A"': _bid_json(price="N/A"),
    'tier "high"': _bid_json(quality_tier="high"),
    "delivery null": _bid_json(delivery_weeks=None),
}


def test_bid_helper_matches_the_valid_bid():
    assert _bid_json() == BID


@pytest.mark.parametrize("reply", list(OUTSIDE_THE_SPACE.values()), ids=list(OUTSIDE_THE_SPACE))
def test_bid_outside_the_contract_space_is_followed_by_a_second_call(reply):
    s, _ = get_fixed_suppliers()
    fake = Scripted(reply, BID)
    agent = LocalSupplierAgent(ChatClient(fake), model="m", system_prompt="S", use_tools=False)
    bid = agent.generate_bid(s, "RULES: test", round_num=1, history=[])
    assert len(fake.calls) == 2
    assert (bid.terms.price, bid.terms.delivery_weeks, bid.terms.quality_tier) == (150.0, 3.0, 2)


@pytest.mark.parametrize("reply", list(OUTSIDE_THE_SPACE.values()), ids=list(OUTSIDE_THE_SPACE))
def test_the_same_bid_outside_the_space_three_times_raises_runtime_error(reply):
    s, _ = get_fixed_suppliers()
    fake = Scripted(reply)
    agent = LocalSupplierAgent(ChatClient(fake), model="m", system_prompt="S", use_tools=False)
    with pytest.raises(RuntimeError, match="no usable bid from the model in 3 attempts") as info:
        agent.generate_bid(s, "RULES: test", round_num=1, history=[])
    assert len(fake.calls) == 3
    assert "last problem: reply has no usable bid" in str(info.value)
    assert info.value.__cause__ is None


def test_bid_outside_the_space_on_every_call_stops_the_tool_path_too():
    s, _ = get_fixed_suppliers()
    agent = LocalSupplierAgent(ChatClient(Scripted(_bid_json(quality_tier=9))), model="m",
                               system_prompt="S", use_tools=True)
    with pytest.raises(RuntimeError, match="no usable bid from the model in 3 attempts"):
        agent.generate_bid(s, "RULES: test", round_num=1, history=[])


@pytest.mark.parametrize("edge", [{"delivery_weeks": 1.0}, {"delivery_weeks": 8}, {"quality_tier": 5},
                                  {"quality_tier": 3.0}], ids=["1 week", "8 weeks", "tier 5", "tier 3.0"])
def test_bid_on_the_edge_of_the_space_is_accepted_at_once(edge):
    s, _ = get_fixed_suppliers()
    fake = Scripted(_bid_json(**edge), BID)
    agent = LocalSupplierAgent(ChatClient(fake), model="m", system_prompt="S", use_tools=False)
    bid = agent.generate_bid(s, "RULES: test", round_num=1, history=[])
    assert len(fake.calls) == 1
    assert bid.terms.delivery_weeks == float(edge.get("delivery_weeks", 3.0))
    assert bid.terms.quality_tier == int(edge.get("quality_tier", 2))


def test_tool_loop_reprompt_states_the_ranges():
    s, _ = get_fixed_suppliers()
    fake = Scripted(_bid_json(quality_tier=9), BID)
    agent = LocalSupplierAgent(ChatClient(fake), model="m", system_prompt="S", use_tools=True)
    bid = agent.generate_bid(s, "RULES: test", round_num=1, history=[])
    assert len(fake.calls) == 2 and bid.terms.quality_tier == 2
    reprompt = [m for m in fake.calls[1]["messages"] if m["role"] == "user"][-1]["content"]
    assert reprompt.startswith("Now submit your bid for this round as a JSON object with fields:")
    assert "Delivery must be between 1 and 8 weeks and the quality tier a whole number from 2 to 5." in reprompt
    assert reprompt.endswith("The compute_my_cost tool is not available for this reply.")  # tools are withdrawn


# The RuntimeError names the last problem and chains the last exception.
def test_runtime_error_names_a_server_error_and_chains_it():
    s, _ = get_fixed_suppliers()
    boom = RuntimeError("connection refused")
    agent = LocalSupplierAgent(ChatClient(Scripted(boom)), model="m", system_prompt="S", use_tools=False)
    with pytest.raises(RuntimeError) as info:
        agent.generate_bid(s, "RULES: test", round_num=1, history=[])
    assert "last problem: RuntimeError: connection refused" in str(info.value)
    assert info.value.__cause__ is boom


def test_runtime_error_names_a_prose_reply_as_having_no_usable_bid():
    s, _ = get_fixed_suppliers()
    agent = LocalSupplierAgent(ChatClient(Scripted("I would bid about 150.")), model="m",
                               system_prompt="S", use_tools=False)
    with pytest.raises(RuntimeError) as info:
        agent.generate_bid(s, "RULES: test", round_num=1, history=[])
    assert "last problem: reply has no usable bid" in str(info.value)
    assert info.value.__cause__ is None  # the parser absorbs the decode error


def test_runtime_error_names_only_the_last_problem():
    s, _ = get_fixed_suppliers()
    fake = Scripted(RuntimeError("first failure"), RuntimeError("second failure"), _bid_json(quality_tier=9))
    agent = LocalSupplierAgent(ChatClient(fake), model="m", system_prompt="S", use_tools=False)
    with pytest.raises(RuntimeError) as info:
        agent.generate_bid(s, "RULES: test", round_num=1, history=[])
    assert "last problem: reply has no usable bid" in str(info.value)
    assert "failure" not in str(info.value)
    assert info.value.__cause__ is None


# The answer outside the thinking block is the model's answer. A bid inside the thinking block is
# used only when there is nothing outside it.
THINK_DRAFT = "<think>draft: " + _bid_json(price=200.0, reasoning="draft") + "</think>"


@pytest.mark.parametrize("use_tools", [False, True])
def test_a_thinking_draft_is_not_used_when_the_answer_outside_it_is_unusable(use_tools):
    s, _ = get_fixed_suppliers()
    fake = Scripted(THINK_DRAFT + _bid_json(quality_tier=9, reasoning="final"), BID)
    agent = LocalSupplierAgent(ChatClient(fake), model="m", system_prompt="S", use_tools=use_tools)
    bid = agent.generate_bid(s, "RULES: test", round_num=1, history=[])
    assert len(fake.calls) == 2
    assert bid.terms.price == 150.0 and bid.reasoning == "[STATED]: ok"  # the second answer, not the draft


@pytest.mark.parametrize("use_tools", [False, True])
def test_the_answer_outside_the_thinking_block_is_used_over_a_draft_inside_it(use_tools):
    s, _ = get_fixed_suppliers()
    fake = Scripted(THINK_DRAFT + _bid_json(price=160.0, reasoning="final"))
    agent = LocalSupplierAgent(ChatClient(fake), model="m", system_prompt="S", use_tools=use_tools)
    bid = agent.generate_bid(s, "RULES: test", round_num=1, history=[])
    assert len(fake.calls) == 1
    assert bid.terms.price == 160.0 and bid.reasoning == "[STATED]: final"


@pytest.mark.parametrize("use_tools", [False, True])
def test_a_bid_inside_the_thinking_block_is_used_when_nothing_is_outside_it(use_tools):
    s, _ = get_fixed_suppliers()
    fake = Scripted(THINK_DRAFT)
    agent = LocalSupplierAgent(ChatClient(fake), model="m", system_prompt="S", use_tools=use_tools)
    bid = agent.generate_bid(s, "RULES: test", round_num=1, history=[])
    assert len(fake.calls) == 1
    assert bid.terms.price == 200.0 and bid.reasoning == "[STATED]: draft"


# Numeric strings are numbers; JSON booleans are not.
@pytest.mark.parametrize("fields, expected", [
    ({"quality_tier": "3.0"}, (150.0, 3.0, 3)),
    ({"quality_tier": "3"}, (150.0, 3.0, 3)),
    ({"price": "150"}, (150.0, 3.0, 2)),
    ({"delivery_weeks": "3.5"}, (150.0, 3.5, 2)),
], ids=['tier "3.0"', 'tier "3"', 'price "150"', 'delivery "3.5"'])
@pytest.mark.parametrize("use_tools", [False, True])
def test_a_numeric_string_is_accepted_at_once(fields, expected, use_tools):
    s, _ = get_fixed_suppliers()
    fake = Scripted(_bid_json(**fields), BID)
    agent = LocalSupplierAgent(ChatClient(fake), model="m", system_prompt="S", use_tools=use_tools)
    bid = agent.generate_bid(s, "RULES: test", round_num=1, history=[])
    assert len(fake.calls) == 1
    assert (bid.terms.price, bid.terms.delivery_weeks, bid.terms.quality_tier) == expected


BOOLEAN_FIELDS = {
    "price true": _bid_json(price=True),
    "delivery true": _bid_json(delivery_weeks=True),
    "tier true": _bid_json(quality_tier=True),
}


@pytest.mark.parametrize("reply", list(BOOLEAN_FIELDS.values()), ids=list(BOOLEAN_FIELDS))
def test_a_json_boolean_is_not_a_number_so_the_bid_is_asked_for_again(reply):
    s, _ = get_fixed_suppliers()
    fake = Scripted(reply, BID)
    agent = LocalSupplierAgent(ChatClient(fake), model="m", system_prompt="S", use_tools=False)
    bid = agent.generate_bid(s, "RULES: test", round_num=1, history=[])
    assert len(fake.calls) == 2
    assert (bid.terms.price, bid.terms.delivery_weeks, bid.terms.quality_tier) == (150.0, 3.0, 2)


@pytest.mark.parametrize("reply", list(BOOLEAN_FIELDS.values()), ids=list(BOOLEAN_FIELDS))
def test_a_json_boolean_three_times_raises_runtime_error(reply):
    s, _ = get_fixed_suppliers()
    fake = Scripted(reply)
    agent = LocalSupplierAgent(ChatClient(fake), model="m", system_prompt="S", use_tools=False)
    with pytest.raises(RuntimeError, match="no usable bid from the model in 3 attempts"):
        agent.generate_bid(s, "RULES: test", round_num=1, history=[])
    assert len(fake.calls) == 3


# One parser for both paths: a valid bid is accepted wherever the reply puts it.
@pytest.mark.parametrize("use_tools", [False, True])
def test_a_bid_with_a_nested_object_that_has_a_price_key_is_accepted_at_once(use_tools):
    s, _ = get_fixed_suppliers()
    nested = _bid_json(price=160.0, reasoning="nested", cost_check={"total_cost": 104.7, "price": 150})
    fake = Scripted(nested, BID)
    agent = LocalSupplierAgent(ChatClient(fake), model="m", system_prompt="S", use_tools=use_tools)
    bid = agent.generate_bid(s, "RULES: test", round_num=1, history=[])
    assert len(fake.calls) == 1
    assert bid.terms.price == 160.0 and bid.reasoning == "[STATED]: nested"


@pytest.mark.parametrize("use_tools", [False, True])
def test_a_bid_written_inside_prose_is_accepted_at_once(use_tools):
    s, _ = get_fixed_suppliers()
    fake = Scripted("Here is my bid:\n" + _bid_json(price=165.0, reasoning="prose") + "\nThank you.", BID)
    agent = LocalSupplierAgent(ChatClient(fake), model="m", system_prompt="S", use_tools=use_tools)
    bid = agent.generate_bid(s, "RULES: test", round_num=1, history=[])
    assert len(fake.calls) == 1
    assert bid.terms.price == 165.0 and bid.reasoning == "[STATED]: prose"


# NO TOOL marks a bid made after the tool path failed: an error, or no usable bid in five calls.
def test_bid_made_after_five_calls_without_a_usable_bid_is_marked_no_tool():
    s, _ = get_fixed_suppliers()
    fake = Scripted(*(["I would bid about 150."] * 5), BID)
    agent = LocalSupplierAgent(ChatClient(fake), model="m", system_prompt="S", use_tools=True)
    bid = agent.generate_bid(s, "RULES: test", round_num=1, history=[])
    assert len(fake.calls) == 6
    assert bid.terms.price == 150.0 and bid.reasoning == "[STATED]: ok" + NO_TOOL_TAG


# The analyst's briefing is what is outside the thinking block.
def test_analyst_reply_that_is_only_a_thinking_block_raises_value_error():
    analyst = LocalProcurementAnalyst(ChatClient(Scripted("<think>reasoning only</think>")), model="m",
                                      system_prompt="A")
    with pytest.raises(ValueError):
        analyst.generate_briefing("A", "Event 1 (Speed Priority): ...")


def test_analyst_briefing_has_no_thinking_and_its_posture_comes_from_the_briefing_only():
    reply = ("<think>This platform looks DEFENSIVE at first,\nso I weigh it carefully.</think>\n"
             "Posture: TRUSTING. The platform acknowledged our strength.")
    analyst = LocalProcurementAnalyst(ChatClient(Scripted(reply)), model="m", system_prompt="A")
    b = analyst.generate_briefing("A", "Event 1 (Speed Priority): ...")
    assert b.posture == AnalystBriefing.TRUSTING
    assert "think" not in b.text.lower() and "weigh it carefully" not in b.text
    assert b.text.endswith("Posture: TRUSTING. The platform acknowledged our strength.")


# A thinking block is removed from every reply, including one cut off before its closing tag.
@pytest.mark.parametrize("text, expected", [
    ("a<think>x</think>b", "ab"),
    ("a<think>unclosed", "a"),
    ("<think>a</think>b<think>c</think>d", "bd"),
    ("<think>a</think>b<think>c", "b"),
    ("<think>multi\nline</think>\nPosture: TRUSTING.", "\nPosture: TRUSTING."),
    ("no thinking here", "no thinking here"),
], ids=["closed", "unclosed", "two closed", "closed then unclosed", "multi-line", "no block"])
def test_think_block_removes_closed_and_unclosed_blocks(text, expected):
    assert THINK_BLOCK.sub("", text) == expected


def test_analyst_reply_that_is_only_an_unclosed_thinking_block_raises_value_error():
    analyst = LocalProcurementAnalyst(ChatClient(Scripted("<think>reasoning that never ends")),
                                      model="m", system_prompt="A")
    with pytest.raises(ValueError):
        analyst.generate_briefing("A", "Event 1 (Speed Priority): ...")


def test_analyst_briefing_followed_by_an_unclosed_thinking_block_keeps_only_the_briefing():
    reply = ("Posture: TRUSTING. The platform acknowledged our strength."
             "<think>On second thought it looks DEFENSIVE")
    analyst = LocalProcurementAnalyst(ChatClient(Scripted(reply)), model="m", system_prompt="A")
    b = analyst.generate_briefing("A", "Event 1 (Speed Priority): ...")
    assert b.posture == AnalystBriefing.TRUSTING
    assert b.text.endswith("Posture: TRUSTING. The platform acknowledged our strength.")
    assert "think" not in b.text.lower() and "second thought" not in b.text


# The posture is the one the briefing states (see LocalProcurementAnalyst._classify_posture), not any label it names.
def _posture_of(text):
    analyst = LocalProcurementAnalyst(ChatClient(Scripted(text)), model="m", system_prompt="A")
    return analyst.generate_briefing("A", "Event 1 (Speed Priority): ...").posture


@pytest.mark.parametrize("text, expected", [
    # Four briefings that the first-label rule (DEFENSIVE, then TRUSTING, named anywhere) got wrong.
    ("Two events is too few to call this platform DEFENSIVE; I recommend GUARDED.", "GUARDED"),
    ("The evidence does not yet justify a TRUSTING posture. Recommendation: GUARDED.", "GUARDED"),
    ("Recommended posture: TRUSTING. There is no sign that a DEFENSIVE stance is needed.", "TRUSTING"),
    ("Posture: GUARDED (not DEFENSIVE, not TRUSTING).", "GUARDED"),
    # 1. The last "posture ... LABEL" on a line, in any case and formatting.
    ("**Posture:** trusting", "TRUSTING"),
    ("The platform may look DEFENSIVE at first, but my posture is guarded.", "GUARDED"),
    ("POSTURE RECOMMENDATION: GUARDED\nThe platform is not yet TRUSTING, and not DEFENSIVE.", "GUARDED"),
    ("Posture: DEFENSIVE. After the third event, the posture is now TRUSTING.", "TRUSTING"),    # the last statement
    ("Posture: TRUSTING. A more cautious posture might be needed if the platform changes.", "TRUSTING"),
    ("Posture: unclear.\nDEFENSIVE is too strong a word; I recommend TRUSTING.", "TRUSTING"),   # same line only
    ("The three postures are DEFENSIVE, TRUSTING and GUARDED. I recommend GUARDED.", "GUARDED"),   # "postures" is no statement
    ("My posture toward the platform is distrusting, hence GUARDED.", "GUARDED"),   # a label is a whole word
    # 2. Otherwise the first "recommend" with a label after it in the same sentence.
    ("Given the DEFENSIVE tone of the platform, I still recommend a TRUSTING posture.", "TRUSTING"),
    ("TRUSTING would be premature; the analyst recommends GUARDED.", "GUARDED"),
    ("We recommended GUARDED and nothing has changed since; TRUSTING needs more events.", "GUARDED"),
    ("Recommend guarded for now; TRUSTING needs more events.", "GUARDED"),
    # 3. Otherwise any mention, as before: DEFENSIVE, then TRUSTING, else GUARDED (negations included).
    ("The platform looks DEFENSIVE to me.", "DEFENSIVE"),
    ("Perhaps TRUSTING, perhaps GUARDED.", "TRUSTING"),
    ("Not clearly DEFENSIVE, not clearly TRUSTING either.", "DEFENSIVE"),
    ("Recommendation: TRUSTING is premature; the platform seems DEFENSIVE.", "DEFENSIVE"),   # no "recommend" here
    ("The record is too thin to say.", "GUARDED"),
    ("I recommend caution. GUARDED beats TRUSTING here.", "TRUSTING"),    # the label is in the next sentence: rule 3
    # Rule 1 needs the label near "posture": a label before it, and another one later in the text, do not decide.
    ("I recommend a DEFENSIVE posture: the platform pressured us in every round, "
     "and a TRUSTING approach would hand it our margins.", "DEFENSIVE"),
    ("I recommend a TRUSTING posture; its feedback varied with our results, "
     "so GUARDED caution would waste good guidance.", "TRUSTING"),
    ("Adopt a DEFENSIVE posture. Three events of uniform pressure rule out TRUSTING.", "DEFENSIVE"),
    ("Posture: GUARDED.\nA TRUSTING posture needs 3+ events of accurate feedback, "
     "and a DEFENSIVE posture needs a clear pattern.", "GUARDED"),
    ("I recommend GUARDED, but my posture on this platform: DEFENSIVE.", "DEFENSIVE"),     # three words between
    ("I recommend GUARDED, but my posture on this whole platform: DEFENSIVE.", "GUARDED"),  # four words: too far
    # Rule 3 takes whole words: a word that contains a label names no posture.
    ("The record is thin. Bid defensively on price but stay open to the feedback.", "GUARDED"),
    ("We remain distrusting of its claims after two events.", "GUARDED"),
    ("The platform looks defensive to me.", "DEFENSIVE"),                       # a whole word, in any case
])
def test_the_posture_is_the_one_the_briefing_states(text, expected):
    assert _posture_of(text) == expected


def _platform_reply(kind, reply):
    sc = get_contract_scenarios()[0]
    bids = {"S": ContractTerms(140.0, 3.0, 2), "Q": ContractTerms(145.0, 6.0, 4)}
    client = ChatClient(Scripted(reply))
    if kind == "mediator":
        return LocalPlatformMediator(client, model="m", system_prompt="P").generate_feedback(
            sc.buyer, bids, "S", 1)
    return LocalBuyerAgentPlatform(client, model="m", system_prompt="P").generate_counteroffer_feedback(
        sc.buyer, bids["S"], "S", 1, [])


@pytest.mark.parametrize("kind", ["mediator", "buyer agent"])
@pytest.mark.parametrize("reply", ["Sharpen your price.<think>and then I considered the other supplier",
                                   "<think>closed</think>Sharpen your price.<think>unclosed"],
                         ids=["answer then unclosed", "closed, answer, unclosed"])
def test_unclosed_thinking_block_in_a_platform_reply_leaves_no_thinking_text(kind, reply):
    assert _platform_reply(kind, reply) == "Sharpen your price."


# A platform reply with nothing outside the thinking block stops the run, as an empty analyst
# briefing does: the bidders are never sent an empty message.
EMPTY_PLATFORM_REPLIES = {
    "empty": "",
    "whitespace": " \n ",
    "none": None,
    "only a thinking block": "<think>reasoning only</think>",
    "only an unclosed thinking block": "<think>reasoning that never ends",
    "thinking and whitespace": "<think>x</think> \n",
}
EMPTY_PLATFORM_ERRORS = {
    "mediator": "the mediator model returned empty feedback for supplier S",
    "buyer agent": "the buyer-agent model returned an empty reply for supplier S",
}


@pytest.mark.parametrize("kind", list(EMPTY_PLATFORM_ERRORS))
@pytest.mark.parametrize("reply", list(EMPTY_PLATFORM_REPLIES.values()), ids=list(EMPTY_PLATFORM_REPLIES))
def test_empty_platform_reply_raises_value_error(kind, reply):
    with pytest.raises(ValueError) as info:
        _platform_reply(kind, reply)
    assert str(info.value) == EMPTY_PLATFORM_ERRORS[kind]


@pytest.mark.parametrize("kind", ["mediator", "buyer agent"])
def test_a_normal_platform_reply_is_returned_without_surrounding_whitespace(kind):
    assert _platform_reply(kind, "\n Sharpen your price. \n") == "Sharpen your price."


@pytest.mark.parametrize("kind", ["mediator", "buyer agent"])
def test_empty_platform_reply_is_written_to_the_debug_log_before_the_error(kind, monkeypatch):
    log = io.StringIO()
    monkeypatch.setattr("orgsim.llm_agents.DEBUG_LOG_FILE", log)
    with pytest.raises(ValueError):
        _platform_reply(kind, "<think>only thinking</think>")
    assert "EMPTY" in log.getvalue() and "only thinking" in log.getvalue()


# The debug log covers every model call: the no-tool bid path and the analyst as well as the tool loop
# and the platforms.
def _debug_log(monkeypatch):
    log = io.StringIO()
    monkeypatch.setattr("orgsim.llm_agents.DEBUG_LOG_FILE", log)
    return log


def _entries(text):
    """(header, body) for each entry of a debug log: a rule, the header and a time stamp, a rule, the body."""
    parts = text.split("=" * 80)
    return [(parts[i].split("\n")[1].strip(), parts[i + 1].strip("\n")) for i in range(1, len(parts) - 1, 2)]


def test_the_no_tool_path_logs_each_request_and_each_unusable_reply(monkeypatch):
    log = _debug_log(monkeypatch)
    s, _ = get_fixed_suppliers()
    replies = ["I cannot decide.", "Still undecided.", "<think>no answer</think>A third refusal."]
    agent = LocalSupplierAgent(ChatClient(Scripted(*replies)), model="m", system_prompt="S", use_tools=False)
    with pytest.raises(RuntimeError, match="no usable bid"):
        agent.generate_bid(s, "RULES: test", round_num=1, history=[])
    entries = _entries(log.getvalue())
    requests = [body for header, body in entries if header.startswith("[S] No-tool request")]
    assert len(requests) == 3 and all("ROUND 1: Submit your bid as JSON" in r for r in requests)   # the user message
    assert [body for header, body in entries if header.startswith("[S] No-tool raw reply")] == replies  # in full
    assert [header for header, _ in entries if header.startswith("[S] No usable bid")] == [
        f"[S] No usable bid in the reply (attempt {k}/3)" for k in (1, 2, 3)]


def test_the_no_tool_path_logs_each_exception(monkeypatch):
    log = _debug_log(monkeypatch)
    s, _ = get_fixed_suppliers()
    agent = LocalSupplierAgent(ChatClient(Scripted(RuntimeError("server unavailable"))), model="m",
                               system_prompt="S", use_tools=False)
    with pytest.raises(RuntimeError, match="no usable bid"):
        agent.generate_bid(s, "RULES: test", round_num=1, history=[])
    entries = _entries(log.getvalue())
    assert len([h for h, _ in entries if h.startswith("[S] No-tool request")]) == 3
    assert [b for h, b in entries if h.startswith("[S] No-tool call raised")] == ["RuntimeError: server unavailable"] * 3
    assert not [h for h, _ in entries if "raw reply" in h]             # no reply came back


def test_the_below_cost_correction_request_and_both_replies_are_logged(monkeypatch):
    log = _debug_log(monkeypatch)
    s, _ = get_fixed_suppliers()
    agent = LocalSupplierAgent(ChatClient(Scripted(CHEAP, BID)), model="m", system_prompt="S", use_tools=False)
    agent.generate_bid(s, "RULES: test", round_num=1, history=[])
    entries = _entries(log.getvalue())
    requests = [b for h, b in entries if h.startswith("[S] No-tool request")]
    assert len(requests) == 2 and "CORRECTION: Your previous bid" in requests[1] and "CORRECTION" not in requests[0]
    assert [b for h, b in entries if h.startswith("[S] No-tool raw reply")] == [CHEAP, BID]


def test_the_analyst_request_with_the_history_and_the_raw_reply_are_logged(monkeypatch):
    log = _debug_log(monkeypatch)
    history = "Event 1 (Speed Priority): you won at $140."
    reply = "<think>weighing it</think>The record is thin. Posture: GUARDED."
    LocalProcurementAnalyst(ChatClient(Scripted(reply)), model="m", system_prompt="A").generate_briefing("A", history)
    logged = dict(_entries(log.getvalue()))
    assert history in logged["[Analyst→A] Request (history sent)"]
    assert logged["[Analyst→A] Raw reply"] == reply                    # in full, thinking block included


def test_each_transient_error_is_logged_before_it_is_retried(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    log = _debug_log(monkeypatch)
    outcomes = iter([RuntimeError("HTTP 503 busy"), RuntimeError("HTTP 429 slow down"), "done"])

    def call():
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    assert api_call_with_retry(call) == "done"
    assert _entries(log.getvalue()) == [
        ("API transient error (attempt 1/3), retrying in 30s", "RuntimeError: HTTP 503 busy"),
        ("API transient error (attempt 2/3), retrying in 60s", "RuntimeError: HTTP 429 slow down"),
    ]


def _raises(error):
    def call():
        raise error
    return call


def test_only_an_error_that_is_retried_is_logged_by_the_retry_wrapper(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    log = _debug_log(monkeypatch)
    with pytest.raises(RuntimeError, match="invalid request"):
        api_call_with_retry(_raises(RuntimeError("invalid request")))      # not transient: raised at once
    assert log.getvalue() == ""
    with pytest.raises(RuntimeError, match="HTTP 503"):
        api_call_with_retry(_raises(RuntimeError("HTTP 503 busy")))
    assert len(_entries(log.getvalue())) == 2                              # attempts 1 and 2 retried, the third raised


@pytest.mark.parametrize("error", ["tool server error", "HTTP 400 bad request"])
def test_the_exceptions_that_send_a_bidder_to_the_no_tool_fallback_are_logged(error, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    log = _debug_log(monkeypatch)
    s, _ = get_fixed_suppliers()
    fake, _ = _client_that_fails_when_tools_are_offered(error)
    agent = LocalSupplierAgent(ChatClient(fake), model="m", system_prompt="S", use_tools=True)
    bid = agent.generate_bid(s, "RULES: test", round_num=1, history=[])
    assert bid.reasoning.endswith(NO_TOOL_TAG)
    entries = _entries(log.getvalue())
    raised = [(h, b) for h, b in entries if h.startswith("[S] Tool path raised")]
    assert all(b == f"RuntimeError: {error}" for _, b in raised)
    assert [h for h, _ in entries].count("[S] Falling back to no-tools") == 1
    if "400" in error:       # the first failure, then the three retries with tools, each retried by the wrapper
        assert [h for h, _ in raised] == ["[S] Tool path raised"] + [
            f"[S] Tool path raised again (retry {k}/3)" for k in (1, 2, 3)]
        assert len([h for h, _ in entries if h.startswith("API transient error")]) == 2 * 4
    else:
        assert [h for h, _ in raised] == ["[S] Tool path raised"]
        assert not [h for h, _ in entries if h.startswith("API transient error")]


@pytest.mark.parametrize("reply", ["", None, "<think>reasoning only</think>"], ids=["empty", "none", "thinking only"])
def test_the_analysts_raw_reply_is_logged_before_the_empty_check(reply, monkeypatch):
    log = _debug_log(monkeypatch)
    analyst = LocalProcurementAnalyst(ChatClient(Scripted(reply)), model="m", system_prompt="A")
    with pytest.raises(ValueError, match="empty briefing"):
        analyst.generate_briefing("A", "Event 1 (Speed Priority): ...")
    logged = dict(_entries(log.getvalue()))
    assert logged["[Analyst→A] Raw reply"] == (reply or "(empty reply)")


@pytest.mark.parametrize("use_tools", [False, True])
def test_a_bid_inside_an_unclosed_thinking_block_is_not_used(use_tools):
    s, _ = get_fixed_suppliers()
    fake = Scripted("<think>draft: " + _bid_json(price=200.0, reasoning="draft") + " and I keep thinking", BID)
    agent = LocalSupplierAgent(ChatClient(fake), model="m", system_prompt="S", use_tools=use_tools)
    bid = agent.generate_bid(s, "RULES: test", round_num=1, history=[])
    assert len(fake.calls) == 2
    assert bid.terms.price == 150.0 and bid.reasoning == "[STATED]: ok"


def test_assistant_message_that_carries_a_tool_call_keeps_no_thinking_in_the_history():
    s, _ = get_fixed_suppliers()
    fake = Scripted("<think>planning the call, no end yet", BID, tool_calls={0: [_tool_call(3, 2)]})
    agent = LocalSupplierAgent(ChatClient(fake), model="m", system_prompt="S", use_tools=True)
    agent.generate_bid(s, "RULES: test", round_num=1, history=[])
    assistant = [m for m in fake.calls[1]["messages"] if m["role"] == "assistant"]
    assert len(assistant) == 1 and assistant[0]["content"] == "" and assistant[0]["tool_calls"]


def _mediator_message(target, round_num, **kw):
    """The user message the mediator sends when it answers `target`."""
    base = FakeBase("Sharpen your price.")
    sc = get_contract_scenarios()[0]
    bids = {"A": ContractTerms(140.0, 3.0, 2), "B": ContractTerms(145.0, 6.0, 4)}
    LocalPlatformMediator(ChatClient(base), model="m", system_prompt="MEDIATOR").generate_feedback(
        sc.buyer, bids, target, round_num, **kw)
    return base.calls[0]["messages"][1]["content"]


# In round 2 every call sees only the feedback of round 1 (run_mediator gives every call in a round the
# same copy), so no supplier has a round-2 reply yet, whichever supplier is answered.
HISTORY_KW = dict(
    buyer_description="MEMORY OF A\n\nMEMORY OF B\n\nBUYER DESCRIPTION",
    previous_bids={"A": [ContractTerms(150.0, 4.0, 2), ContractTerms(140.0, 3.0, 2)],
                   "B": [ContractTerms(160.0, 7.0, 4), ContractTerms(145.0, 6.0, 4)]},
    previous_feedback={"A": ["fb A1"], "B": ["fb B1"]},
)


def test_mediator_reads_both_suppliers_histories_and_memories():
    msg = _mediator_message("B", 2, **HISTORY_KW)
    assert msg.startswith(HISTORY_KW["buyer_description"] + "\n\n")      # both memories reach the mediator
    a = msg.index("BID HISTORY FOR SUPPLIER A:\n")
    b = msg.index("BID HISTORY FOR SUPPLIER B (THIS SUPPLIER):\n")
    current = msg.index("CURRENT BIDS FROM ALL SUPPLIERS:")
    assert a < b < current                                               # by supplier id, then the current bids
    block_a, block_b = msg[a:b], msg[b:current]
    assert "Round 1: $150.00, 4.0wk, tier 2" in block_a and "Your feedback (R1): fb A1" in block_a
    assert "Round 2: $140.00, 3.0wk, tier 2" in block_a                  # this round's bid, not yet answered
    assert "Round 2: $145.00, 6.0wk, tier 4" in block_b and "Your feedback (R1): fb B1" in block_b
    assert "(R2)" not in msg                                             # no round-2 reply in any call of round 2
    assert msg.count("(THIS SUPPLIER)") == 2                             # its history and its current bid


def test_the_mediator_marks_whichever_supplier_it_is_answering():
    msg = _mediator_message("A", 2, **HISTORY_KW)
    assert "BID HISTORY FOR SUPPLIER A (THIS SUPPLIER):\n" in msg and "BID HISTORY FOR SUPPLIER B:\n" in msg
    assert msg.count("(THIS SUPPLIER)") == 2


def test_the_mediator_has_no_history_block_in_round_1():
    # run_mediator's call in round 1: previous_bids already holds the current bids, no reply exists yet.
    bids = {"A": [ContractTerms(140.0, 3.0, 2)], "B": [ContractTerms(145.0, 6.0, 4)]}
    msg = _mediator_message("A", 1, previous_bids=bids, previous_feedback={"A": [], "B": []})
    assert "BID HISTORY" not in msg and "Your feedback (R" not in msg
    assert "CURRENT BIDS FROM ALL SUPPLIERS:" in msg and "Supplier A (THIS SUPPLIER): $140.00" in msg
    assert "BID HISTORY" not in _mediator_message("A", 1)                   # nor when no history is passed at all


def test_buyer_agent_message_holds_only_this_suppliers_bids():
    base = FakeBase("We need more.")
    sc = get_contract_scenarios()[0]
    history = [{"round": 1, "bid": ContractTerms(150.0, 4.0, 2), "feedback": "Do better."}]
    LocalBuyerAgentPlatform(ChatClient(base), model="m", system_prompt="BUYER").generate_counteroffer_feedback(
        sc.buyer, ContractTerms(140.0, 3.0, 2), "A", 2, history, buyer_description="DESC")
    assert base.calls[0]["messages"][1]["content"] == (
        "DESC\n\n"
        "Supplier A's current bid: $140.00, 3.0wk, tier 2\n\n"
        "NEGOTIATION HISTORY (use this to identify flexibility):\n"
        "  Round 1: $150.00, 4.0wk, tier 2\n\n"
        "Generate your adversarial negotiation message for round 2. 2-3 sentences.")


def test_cost_tool_labels_cost_plus_five_percent_and_does_not_promise_a_loss():
    s, _ = get_fixed_suppliers()
    text = execute_cost_tool(s, {"delivery_weeks": 3, "quality_tier": 2})
    total = s.cost(ContractTerms(0, 3.0, 2))
    assert f"  Cost + 5%:     ${total * 1.05:.2f}\n" in text            # the value column stays aligned
    assert f"  Cost + 15%:    ${total * 1.15:.2f}\n" in text and f"  Cost + 25%:    ${total * 1.25:.2f}\n" in text
    assert len({line.index("$") for line in text.splitlines()[1:-1]}) == 1  # one column of values, labels aside
    assert text.endswith(f"WARNING: A bid below ${total:.2f} would lose money if it won.")
    assert "Min bid" not in text and "guarantees" not in text and "margin" not in text


def test_tool_limit_nudge_asks_for_the_bid_of_this_round():
    s, _ = get_fixed_suppliers()
    fake = Scripted("", "", BID, tool_calls={0: [_tool_call(3, 2)], 1: [_tool_call(4, 2)]})
    agent = LocalSupplierAgent(ChatClient(fake), model="m", system_prompt="S", use_tools=True)
    agent.generate_bid(s, "RULES: test", round_num=1, history=[])
    nudge = [m for m in fake.calls[2]["messages"] if m["role"] == "user"][-1]["content"]
    assert nudge.startswith("You have used all your cost calculations. Now submit your bid for this round "
                            "as a JSON object")
    assert "final" not in nudge.lower() and "tools" not in fake.calls[2]


def test_a_below_cost_correction_is_a_no_tool_request_with_the_no_tool_round_text():
    s, _ = get_fixed_suppliers()
    fake = Scripted(CHEAP, BID)
    agent = LocalSupplierAgent(ChatClient(fake), model="m", system_prompt="S", use_tools=True)
    agent.generate_bid(s, "RULES: test", round_num=1, history=[])
    first, correction = fake.calls
    assert "tools" in first and "tools" not in correction
    text = correction["messages"][1]["content"]
    assert "compute_my_cost" not in text and "ROUND 1: Submit your bid as JSON." in text
    assert "CORRECTION" in text and "If it won, this bid would lose money. " in text
    assert "guarantees" not in text


def test_the_fallback_after_a_tool_failure_carries_the_no_tool_round_text():
    s, _ = get_fixed_suppliers()
    fake, calls = _client_that_fails_when_tools_are_offered("tool server error")
    agent = LocalSupplierAgent(ChatClient(fake), model="m", system_prompt="S", use_tools=True)
    history = [{"round": 1, "bid": ContractTerms(150.0, 3.0, 2), "feedback": "Sharpen your price."}]
    agent.generate_bid(s, "RULES: test", round_num=2, history=history)
    assert "compute_my_cost" in calls[0]["messages"][1]["content"]             # the tool request names the tool
    fallback = calls[-1]
    assert "tools" not in fallback
    text = fallback["messages"][1]["content"]
    assert "compute_my_cost" not in text and "ROUND 2: Submit your bid as JSON." in text
    assert "Your costs from previous rounds" not in text


def test_the_tool_instruction_points_to_earlier_costs_only_when_there_are_earlier_rounds():
    s, _ = get_fixed_suppliers()
    history = [{"round": 1, "bid": ContractTerms(150.0, 3.0, 2), "feedback": "Sharpen your price."}]
    for round_num, past, points_back in ((1, [], False), (2, history, True)):
        fake = Scripted(BID)
        agent = LocalSupplierAgent(ChatClient(fake), model="m", system_prompt="S", use_tools=True)
        agent.generate_bid(s, "RULES: test", round_num=round_num, history=past)
        text = fake.calls[0]["messages"][1]["content"]
        assert "You have a compute_my_cost tool." in text
        assert ("PREVIOUS ROUNDS:" in text) is points_back
        assert ("Your costs from previous rounds are shown above" in text) is points_back


def test_analyst_message_frames_the_record_without_calling_it_complete():
    history = "Event 1 (Speed Priority): ..."
    base = FakeBase("The record is thin. Posture: GUARDED.")
    LocalProcurementAnalyst(ChatClient(base), model="m", system_prompt="ANALYST").generate_briefing("A", history)
    assert base.calls[0]["messages"][1]["content"] == (
        "You are the procurement analyst for Supplier A.\n\n"
        "Here is the interaction history with this platform:\n\n"
        + history + "\n\nProduce your strategic briefing.")


def test_cost_tool_description_does_not_call_the_cost_plus_five_percent_a_minimum_bid():
    description = COST_CALCULATOR_TOOL["function"]["description"]
    assert description.endswith("Returns: the total cost, a cost breakdown, and the cost plus 5%, 15% and 25%.")
    assert "minimum" not in description and "margin" not in description


# What the bidders of the AI institutions read must not change when another kind of bidder is added. Each digest
# covers every request of one scenario: the messages in the order sent (the tool exchange included) and the tools
# offered. The digests were taken from the code before the sealed-bid bidder was added.
class Snapshotting(Scripted):
    """Scripted, but keeps a copy of each request as it was sent (the tool loop appends to its message list)."""

    def __init__(self, *contents, **kw):
        super().__init__(*contents, **kw)
        self.snapshots = []

    def _create(self, **kw):
        self.snapshots.append({"messages": json.loads(json.dumps(kw["messages"])), "tools": kw.get("tools"),
                               "tool_choice": kw.get("tool_choice")})
        return super()._create(**kw)


EARLIER_ROUNDS = [
    {"round": 1, "bid": ContractTerms(150.0, 3.0, 2), "feedback": "Sharpen your price.", "price_set_by_rule": True},
    {"round": 2, "bid": ContractTerms(140.0, 4.0, 3), "feedback": "The field is close."},
]
TWO_TOOL_CALLS = {0: [_tool_call(3, 2)], 1: [_tool_call(4, 2)]}
# name: (which fixed supplier, the cost tool offered, the rounds before, the model's replies, its tool calls)
AI_BIDDER_SCENARIOS = {
    "speed specialist, no tool, round 1": (0, False, [], [BID], None),
    "quality specialist, no tool, round 1": (1, False, [], [BID], None),
    "no tool, round 3 with two earlier rounds": (0, False, EARLIER_ROUNDS, [BID], None),
    "tool, round 1, two cost calls": (0, True, [], ["", "", BID], TWO_TOOL_CALLS),
    "tool, round 3 with two earlier rounds": (1, True, EARLIER_ROUNDS, [BID], None),
    "tool, a prose reply is asked again": (0, True, [], ["I would bid about 150.", BID], None),
    "no tool, a bid below cost is sent back": (0, False, [], [CHEAP, BID], None),
    "tool, a bid below cost is sent back without the tool": (0, True, [], [CHEAP, BID], None),
}
EXPECTED_AI_BIDDER_DIGESTS = {
    "speed specialist, no tool, round 1": "fa3df99bc2988d31",
    "quality specialist, no tool, round 1": "a7b2f5ff2a204c0c",
    "no tool, round 3 with two earlier rounds": "c820891d88a24991",
    "tool, round 1, two cost calls": "a571c0c81d117e86",
    "tool, round 3 with two earlier rounds": "a431cac22e13f485",
    "tool, a prose reply is asked again": "44733f29a7b866ea",
    "no tool, a bid below cost is sent back": "d70fbd2c874a815e",
    "tool, a bid below cost is sent back without the tool": "4edb9fa656b9c49c",
}


def _ai_bidder_digest(scenario):
    index, use_tools, history, replies, tool_calls = scenario
    fake = Snapshotting(*replies, tool_calls=tool_calls)
    agent = LocalSupplierAgent(ChatClient(fake), model="m", system_prompt="SYSTEM", use_tools=use_tools)
    agent.generate_bid(get_fixed_suppliers()[index], "RULES: test", round_num=len(history) + 1, history=history)
    text = json.dumps(fake.snapshots, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def test_the_requests_of_the_ai_institutions_bidders_are_unchanged():
    got = {name: _ai_bidder_digest(scenario) for name, scenario in AI_BIDDER_SCENARIOS.items()}
    assert got == EXPECTED_AI_BIDDER_DIGESTS, "what the bidders of the AI institutions read has changed"


# ---- the bidder of the second-price auction (sealed_bid=True) ----

SEALED_NO_TOOL_TEXT = "Submit your bid as JSON. Keep reasoning brief (1-2 sentences)."
SEALED_TOOL_TEXT = ("You have a compute_my_cost tool that returns your total cost for given delivery and quality "
                    "terms. You may call it at most twice. After your tool calls, output only the JSON bid object.")


def _sealed_agent(fake, use_tools):
    return LocalSupplierAgent(ChatClient(fake), model="m", system_prompt="SYSTEM", use_tools=use_tools,
                              sealed_bid=True)


def test_the_sealed_bid_bidder_without_the_tool_sends_exactly_these_messages():
    s, _ = get_fixed_suppliers()
    fake = Snapshotting(BID)
    bid = _sealed_agent(fake, False).generate_bid(s, "RULES: test", round_num=1, history=[])
    assert (bid.terms.price, bid.reasoning) == (150.0, "[STATED]: ok")
    assert fake.snapshots == [{
        "messages": [{"role": "system", "content": "SYSTEM"},
                     {"role": "user", "content": s.describe_for_sealed_bid() + "\n\nREGIME: RULES: test\n\n"
                                                 + SEALED_NO_TOOL_TEXT}],
        "tools": None, "tool_choice": None}]
    assert (SEALED_NO_TOOL_ROUND_TEXT, SEALED_TOOL_ROUND_TEXT) == (SEALED_NO_TOOL_TEXT, SEALED_TOOL_TEXT)


def test_the_sealed_bid_bidder_with_the_tool_sends_the_neutral_tool_and_the_neutral_texts():
    s, _ = get_fixed_suppliers()
    fake = Snapshotting("", "", BID, tool_calls={0: [_tool_call(3, 2)], 1: [_tool_call(4, 2)]})
    _sealed_agent(fake, True).generate_bid(s, "RULES: test", round_num=1, history=[])
    first, second, third = fake.snapshots
    assert first["messages"][1]["content"] == (s.describe_for_sealed_bid() + "\n\nREGIME: RULES: test\n\n"
                                              + SEALED_TOOL_TEXT)
    assert first["tools"] == second["tools"] == [COST_CALCULATOR_TOOL_SEALED] and third["tools"] is None
    # The tool result is the breakdown and the total, nothing more: 3wk/T2 costs 85 + 0.5 * 5^2 + 1.8 * 2^2 = 104.7
    # and 4wk/T2 costs 85 + 0.5 * 4^2 + 1.8 * 2^2 = 100.2 (the speed specialist's delivery cost is 0.5 * (8 - weeks)^2).
    assert second["messages"][-1]["content"] == (
        "Cost breakdown for 3.0wk / T2:\n  Base cost:     $85.00\n  + Delivery:    $12.50\n"
        "  + Quality:     $7.20\n  = TOTAL COST:  $104.70")
    assert third["messages"][-2]["content"] == (
        "Cost breakdown for 4.0wk / T2:\n  Base cost:     $85.00\n  + Delivery:    $8.00\n"
        "  + Quality:     $7.20\n  = TOTAL COST:  $100.20")
    assert third["messages"][-1]["content"] == (
        "You have used all your cost calculations. Now submit your bid for this round as a JSON object with these "
        "exact fields: price, delivery_weeks, quality_tier, reasoning. Output ONLY the JSON object, nothing else.")


def test_the_sealed_bid_tool_has_the_name_and_parameters_of_the_other_one_and_a_description_without_advice():
    sealed, ai = COST_CALCULATOR_TOOL_SEALED["function"], COST_CALCULATOR_TOOL["function"]
    assert (sealed["name"], sealed["parameters"]) == (ai["name"], ai["parameters"])
    assert sealed["description"] == ("Compute your exact total cost for a given set of contract terms. "
                                     "Returns the total cost and a cost breakdown.")
    assert "BEFORE" not in sealed["description"] and "5%" not in sealed["description"]
    assert COST_CALCULATOR_TOOL_SEALED["type"] == COST_CALCULATOR_TOOL["type"] == "function"
    assert ai["description"].endswith("and the cost plus 5%, 15% and 25%.")           # the other one is as it was


def test_the_sealed_bid_cost_result_is_the_breakdown_the_other_one_starts_with():
    s, _ = get_fixed_suppliers()
    args = {"delivery_weeks": 3, "quality_tier": 2}
    sealed, ai = execute_cost_tool(s, args, sealed_bid=True), execute_cost_tool(s, args)
    assert ai.startswith(sealed + "\n  Cost + 5%:") and sealed.endswith("= TOTAL COST:  $104.70")
    assert "Cost +" not in sealed and "WARNING" not in sealed and "lose money" not in sealed
    assert ai.endswith("WARNING: A bid below $104.70 would lose money if it won.")


@pytest.mark.parametrize("use_tools", [False, True])
def test_a_sealed_bid_below_cost_is_taken_as_submitted(use_tools):
    s, _ = get_fixed_suppliers()
    fake = Snapshotting(CHEAP)
    bid = _sealed_agent(fake, use_tools).generate_bid(s, "RULES: test", round_num=1, history=[])
    assert len(fake.snapshots) == 1                                    # no correction request
    assert (bid.terms.price, bid.terms.delivery_weeks, bid.terms.quality_tier) == (90.0, 3.0, 2)   # cost 104.70
    assert bid.reasoning == "[STATED]: cheap"                         # not marked as set by the cost-plus-5% rule
    assert RULE_PRICE_TAG not in bid.reasoning and NO_TOOL_TAG not in bid.reasoning
    assert "CORRECTION" not in fake.snapshots[0]["messages"][1]["content"]


def test_the_sealed_bid_bidder_refuses_earlier_rounds():
    s, _ = get_fixed_suppliers()
    fake = Snapshotting(BID)
    with pytest.raises(ValueError, match="no earlier rounds"):
        _sealed_agent(fake, False).generate_bid(s, "RULES: test", round_num=2, history=EARLIER_ROUNDS)
    assert fake.snapshots == []                                       # and nothing was sent


def test_the_sealed_bid_fallback_after_a_tool_failure_carries_the_sealed_no_tool_text(monkeypatch):
    s, _ = get_fixed_suppliers()
    fake, calls = _client_that_fails_when_tools_are_offered("tool server error")
    bid = _sealed_agent(fake, True).generate_bid(s, "RULES: test", round_num=1, history=[])
    assert "tools" in calls[0] and "tools" not in calls[-1]
    text = calls[-1]["messages"][1]["content"]
    assert text == s.describe_for_sealed_bid() + "\n\nREGIME: RULES: test\n\n" + SEALED_NO_TOOL_TEXT
    assert "compute_my_cost" not in text and bid.reasoning.endswith(NO_TOOL_TAG)    # still marked: made without it


def test_the_sealed_bid_bidder_asked_again_after_a_reply_without_a_bid_reads_the_neutral_reprompt():
    s, _ = get_fixed_suppliers()
    fake = Snapshotting("I would bid about 150.", BID)
    _sealed_agent(fake, True).generate_bid(s, "RULES: test", round_num=1, history=[])
    first, second = fake.snapshots
    assert first["tools"] == [COST_CALCULATOR_TOOL_SEALED] and second["tools"] is None
    assert second["messages"][-2] == {"role": "assistant", "content": "I would bid about 150."}
    assert second["messages"][-1] == {"role": "user", "content": (
        "Now submit your bid for this round as a JSON object with fields: price, delivery_weeks, quality_tier, "
        "reasoning. Delivery must be between 1 and 8 weeks and the quality tier a whole number from 2 to 5. "
        "The compute_my_cost tool is not available for this reply.")}


def test_the_ai_institutions_bidder_is_not_sealed_by_default():
    agent = LocalSupplierAgent(ChatClient(Scripted(BID)), model="m", system_prompt="S")
    assert agent.sealed_bid is False
