"""Single, organized and guided organized suppliers, alike or mixed in one market, under the AI mediator and the
AI buyer agent, and the second-price auction run once for the structural benchmark. The primitives prompt set
(--prompts primitives) gives the agents of the AI institutions texts that state the objective, the information and
the rules and no advice, for six organizations, in a random or a persistent environment."""
import argparse
import csv
import functools
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))  # find orgsim without installing it

import numpy

from orgsim import llm_agents
from orgsim import primitives_prompts as PP
from orgsim import prompts as P
from orgsim.interfaces import FLOOR_PRICE_TAG, NO_TOOL_TAG, RULE_PRICE_TAG
from orgsim.llm import sum_usage
from orgsim.mocks import (
    MockBuyerAgentPlatform, MockPlatformMediator, MockProcurementAnalyst, MockStrategicSupplier,
)
from orgsim.organization import (
    ALL_PRIMITIVES_ANALYST_ORGANIZATIONS, ALL_PRIMITIVES_ORGANIZATIONS, ANALYST_ORGANIZATIONS, DEPARTURE_ORGANIZATIONS,
    PRIMITIVES_ORGANIZATIONS, departure_base, organization_of_each,
)
from orgsim.primitives import get_contract_scenarios
from orgsim.sequence import SUPPLIER_IDS, pre_draw_event_sequence, run_sequence

INSTITUTIONS = {"mediator": "Mediator", "buyer_agent": "BuyerAgent", "second_price": "SecondPrice"}
# The institutions that run in rounds with a platform agent, over events with a record; the default.
AI_INSTITUTIONS = ("mediator", "buyer_agent")
# The default of --organizations. This very list is the default, so that a command line that gives
# --organizations (a list of its own, even one equal to this) can be told from one that does not.
DEFAULT_ORGANIZATIONS = list(P.ORGANIZATIONS)
# The organizations of each prompt set. single and organized are in both, with the texts of their set. A command line
# may name any of ALL_ORGANIZATIONS; _check_options refuses the ones that are not in the set of --prompts. With
# --prompts primitives and no --organizations, the markets are those of the six organizations of
# PRIMITIVES_ORGANIZATIONS (the departures organizations are run only when named).
ORGANIZATIONS_OF = {"advice": tuple(P.ORGANIZATIONS), "primitives": tuple(ALL_PRIMITIVES_ORGANIZATIONS)}
ALL_ORGANIZATIONS = list(P.ORGANIZATIONS) + [name for name in ALL_PRIMITIVES_ORGANIZATIONS
                                              if name not in P.ORGANIZATIONS]
DEFAULT_BASE_URL = "http://localhost:1234/v1"
# The mock suppliers seed numpy with seed * 10 + 1 at most, and numpy takes seeds up to 2**32 - 1.
MAX_SEED = 429496729
# The largest --timeout in seconds (a day). socket.settimeout(1e9) raises OverflowError on Windows.
MAX_TIMEOUT = 86400
# A secret of this many characters or more is taken for a real credential (a key is far longer). A
# shorter one is probably a dummy value ("local") or a harmless query value (?v=2).
CREDENTIAL_LENGTH = 8
OUTPUT_FILES = ("events.jsonl", "summary.csv", "run.json")
# What an earlier run left that a new run deletes. run.json is replaced instead, before anything else.
# run.json.tmp is the temporary file of an update to run.json that a kill cut off.
EARLIER_OUTPUTS = ("events.jsonl", "summary.csv", "run.json.tmp")
# organization is the organization both suppliers have, or "<a>:<b>" in a mixed market; organization_a and
# organization_b are the organization of each supplier. prompt_set and environment are the settings of --prompts and
# --environment.
FIELDS = ["institution", "organization", "organization_a", "organization_b", "prompt_set", "environment", "seed",
          "event", "contract", "type_a", "type_b", "same_type", "winner", "awarded", "correct_winner",
          "efficient_winner", "efficiency", "buyer_surplus", "supplier_profit", "joint_surplus"]
# The settings of a run before --prompts and --environment existed.
ADVICE_SETTING = {"prompt_set": "advice", "environment": "random"}
# Bids that code shaped rather than the model, counted per event so they can be audited: bids whose
# price the below-cost rule set, and bids made without the cost tool after the tool path failed.
# With --no-tools no bid uses the tool, so none is counted.
TAG_FIELDS = {"n_rule_priced_bids": RULE_PRICE_TAG, "n_tool_fallback_bids": NO_TOOL_TAG}
# An audit of the platforms: the replies in the event that quote a price the other supplier bid in it.
QUOTES_FIELD = "n_rival_price_quotes"
COLUMNS = FIELDS + list(TAG_FIELDS) + [QUOTES_FIELD]
# The bids whose price the floor raised (a floor organization of the primitives set), counted per event. The column is
# in the summary.csv of a run that names a departures organization (zero where no supplier has a floor), and in no
# other, so that the summary of a run of the earlier organizations or of the advice set is as it was.
FLOOR_FIELD = "n_floor_raised_bids"


def _mock_note(organizations, mixed=False):
    """What a mock run prints and records about itself. Mock agents ignore what they are told, so the rows of
    different organizations (or, when `mixed`, of different mixed markets) are identical by construction: that is
    said when the run has more than one. A run of one (the second-price auction always has one) has no such rows
    to compare, and its note says only what a mock run is for."""
    if len(organizations) > 1:
        names = ", ".join(organizations[:-1]) + " and " + organizations[-1]
        return (f"Mock agents ignore what they are told, so the {names} rows are identical by construction; "
                f"use --backend openai with a model to compare the {'markets' if mixed else 'organizations'}.")
    return ("Mock agents ignore what they are told, so a mock run only checks that the pipeline runs and shows "
            "nothing about how a model bids; use --backend openai with a model for that.")


MOCK_NOTE = _mock_note(P.ORGANIZATIONS)  # the note of a run with the default organizations, all of them
EPILOG = f"""\
Output goes to the folder given by --out. A new run first replaces run.json and deletes the
events.jsonl, summary.csv and run.json.tmp that an earlier run left in that folder.
  events.jsonl  one JSON record per event, written as the event ends, so a run that stops keeps
                its finished events. A record holds the event's result, the outcome's fields and,
                for each supplier, every round's bid with its stated reasoning and the platform's
                reply to that bid (null for the last round, which gets no reply).
  summary.csv   one row per event; written only when the run completes.
  run.json      the run's settings, software versions and source, its start and end time, and its
                status (running, complete or failed, with the error). A model backend adds its usage,
                written after each event and when the run ends. A run that is killed outright stays
                running: its usage then leaves out the requests of the event in progress, and also
                those of every finished event whose update was cut off by the kill (which can leave
                a run.json.tmp) or failed with a warning. It leaves out the API key and the user
                name, password and query string of --base-url, within the limits below.

Credentials are removed from every string in run.json, an error message included: the API key
(--api-key or $OPENAI_API_KEY) and the user name, password and query values of --base-url. The
limits:
  - a credential of {CREDENTIAL_LENGTH} or more characters is removed wherever it appears;
  - a shorter one is removed only where it stands as a whole word (not next to a letter, digit, _
    or -), so one joined to other characters in a server's error text can remain;
  - encoded forms, such as base64 of user:password, are not matched;
  - a query value shorter than {CREDENTIAL_LENGTH} characters is not treated as a credential,
    although the recorded base_url and argv never include the query.

n_rival_price_quotes in summary.csv audits the platforms: the number of their replies in the
event, to either supplier, that contain a price the other supplier bid in this event, written as a
number with two decimals (123.45) or, for a whole number of dollars, as $ and the integer ($124).
The mediator is told never to reveal one supplier's bid details to the other, and the buyer agent
never sees the other supplier's bids, so a count above 0 is worth reading in events.jsonl. It is
an audit signal, not a verdict, because a coincidence can match. Prices from past events are not
counted.

--mixed A_ORG:B_ORG ... runs markets in which supplier A and supplier B have different organizations,
for example single:organized (A is a single firm, B an organized firm), in place of the markets of
--organizations, whose two suppliers share one. Each supplier is treated by its own organization: its
bidder reads the prompt of that organization and its own record or its own analyst's briefing, and only
an organized supplier has an analyst. Neither supplier nor platform is told the rival's organization.
Ties go to A, so run both orders of a pair. In events.jsonl and summary.csv, organization_a and
organization_b hold the organization of each supplier, and organization holds the one that both have, or
<a>:<b> in a mixed market. run.json records mixed (the pairs, or null), and organizations is null in a run
of mixed markets.

--prompts primitives gives the agents of the AI institutions (mediator and buyer_agent) the texts of
orgsim/primitives_prompts.py in place of the advice prompts: their objective, what they know and do not know, who
decides, the payoffs and the institution's rules, and no advice. It needs --no-tools with a model, --events of at
least 2, and neither second_price nor --max-history-events. Its organizations, in --organizations (all of them if
none is given) or --mixed, are single, single_norecord (the firm keeps no record, so its bidder is never shown a
past event), organized, organized_pass (the analyst passes the record to the bidder unchanged; no model is called
for it), single_revenue and organized_revenue (the bidder is judged on the firm's revenue, the total price of the
contracts it wins; in organized_revenue the analyst on the firm's profit). A bid stands as submitted (there is no
below-cost correction and no price set by rule, so n_rule_priced_bids is 0) and no posture is read from a briefing
(analyst_postures holds null for each briefing; for organized_pass, analyst_briefings holds the record that the
bidder was sent). --environment persistent repeats event 1 of each seed's random plan, the contract and both cost
types, at every event; it needs --prompts primitives. run.json records prompts and environment, and every row of
summary.csv and every line of events.jsonl holds them as prompt_set and environment. The texts state the
probabilities of the draw (1 in 2 for the contract and for each cost type, 3 in 4 for the other supplier's type
being the opposite); the command has no option for them, and the set refuses a plan drawn with others.

The departures organizations of --prompts primitives, run only when they are named in --organizations or --mixed (they
are refused with --prompts advice): in each, the firm's analyst holds the record and sends the bidder one message
before each event from the second. brief: the analyst writes a briefing. forward: no model is called, and the message is
the full record, the very text that single reads, forwarded unchanged. floor: as brief, and the analyst may end its
briefing with a minimum profit per contract for the event ("MINIMUM PROFIT PER CONTRACT: $X" or "MINIMUM PROFIT PER
CONTRACT: none"; asked again once if the line is missing or malformed, and then no floor is set), which the code
enforces: a bid priced below its total cost plus the minimum is returned to the bidder once, and if the bid it sends
back is also below, its price is raised to the total cost plus the minimum (the bid's reasoning is tagged, and its
"floor" in events.jsonl says what happened; "departures" holds the analyst's briefings, the minimum and the counts;
summary.csv counts the raised bids in n_floor_raised_bids, a column that a run naming a departures organization has).
In floor the last line that is not only marks is read with some tolerance (any case, markdown marks, an optional dollar
sign), and the bidder reads the briefing without any line that begins with the words MINIMUM PROFIT PER CONTRACT (in any
case, markdown marks aside), then a notice: the minimum, none, or, if the analyst gave no valid line in two answers,
that no minimum is set (floor_status in "departures" tells the cases apart: set, none, malformed).
brief_cap: as brief, with the briefing at most 50 words: an analyst whose briefing is longer is asked once more, and a
second briefing that is still too long is cut to its first 50 words (events.jsonl records both briefings, whether it
was cut and the final word count under "departures"). brief_cost: as brief, and the analyst is also shown the bidder's
cost figures. Each also with the suffix _revenue (the bidder is judged on revenue, and is told that the analyst's
objective is profit). Their agents are counted in run.json under bidder_<organization> and analyst_<organization>.

second_price is the second-score sealed-bid auction, one bid for each of the two suppliers, run for
one event (--events 1, --organizations single; several auctions come from several --seeds). Its
bidder reads a prompt that states primitives only and no advice. In its events.jsonl record each
supplier's one bid in "bids" holds the price and terms as submitted, outcome.winning_bid holds the
price paid (the highest price at which the winner's score is still at least the other bid's score),
outcome.all_true_costs each supplier's cost at the terms it bid and outcome.all_inflation_rates
(submitted price - cost) / cost, whose distance from 0 is how far the bid is from the benchmark
(a price equal to the cost at the terms bid is optimal whatever the rival bids). The surpluses in
summary.csv are at the price paid. A bid below cost is taken as submitted, so n_rule_priced_bids
is 0.

analyst_postures in events.jsonl is read from the text of each briefing by a keyword rule, which
does not understand a negation. Check it against analyst_briefings before relying on it; no agent
acts on it.

run.json names the code in two ways. git_commit is the commit of the git repository whose top level
is the folder above scripts/ ("unknown" for an exported copy, or a folder inside another
repository), and git_dirty says whether tracked files differ from that commit (null when unknown).
source_sha256 identifies the code without the git history: the SHA-256 (hex digest) over
orgsim/*.py and scripts/run_demo.py in sorted order of their relative paths (written with /, sorted
as plain strings), where each file adds its relative path, a NUL byte, its bytes as stored and a
NUL byte. The repository's .gitattributes gives the .py files LF line endings in every
checkout, so the value is the same on every platform; a copy with other line endings gives another
value.

--backend openai means any OpenAI-compatible chat endpoint, local or hosted. The seed fixes the
event plan and the mock agents but not the model's sampling, so two runs with a model can differ.
Give the API key through the OPENAI_API_KEY environment variable rather than --api-key: a key on
the command line can be recorded by the shell and shown in process listings. Console output and
the --debug-log file are not scrubbed of keys, unlike run.json: a server error that repeats a key
is printed and logged as it is.

Against Anthropic's OpenAI-compatible endpoint, use --backend openai --hosted --default-sampling
--no-tools --base-url https://api.anthropic.com/v1/ --model <id> --max-tokens N --max-retries 30
--timeout 300, with the key in OPENAI_API_KEY. On Claude 4.7 and later models, the Claude 5 models
among them, a temperature below 1, a top_p other than 0.99, or both together return HTTP 400, and
every agent request sets both; --default-sampling sends no sampling settings, so the provider's
defaults apply. --max-tokens caps each reply, hidden reasoning included: Claude 5 models think by
default (adaptively on Claude Haiku 5.5: the model decides whether and how much), and the reasoning
is billed as output and counts toward the cap but is not returned. run.json records under "usage",
in total and for each agent type: requests (every call made through the client, including calls
that raised before or after reaching the server), failed_requests (the calls that raised; a failed
call can still be billed, for example one interrupted after the server finished), prompt_tokens and
completion_tokens (as the endpoint reported them), largest_completion_tokens (the largest
completion of any single request), truncated (replies cut at the cap), no_usage (replies whose
usage lacks a whole-number token count; any count that is there is still added) and finish_reasons
(replies by finish reason: "none" if a reply has none, "other" if it is not 1 to 40 lowercase
letters and underscores). In total only, each HTTP attempt counts in http_attempts; one that
receives a response also counts in http_responses by status code (strings such as "200" and "529"),
so attempts minus responses are the attempts that timed out or could not connect. Such an attempt
can still be billed if the server finished it. A call that the SDK retries counts once in requests.
Whether those completion tokens include the hidden reasoning is not documented: compare a cheap
smoke run's usage once with the provider's usage page before pricing a run from them. A cut reply
can end empty or partial, so check that truncated is 0; a no_usage above 0 means the token counts
are low.
"""


def _agent_labels(args=None):
    """The label of each agent type by its system prompt, which the client counts usage under. Under --prompts
    primitives there is one label for each bidder prompt (bidder_<organization>), the analyst's two prompts
    (analyst, and analyst_revenue for organized_revenue) and the platforms'; the system prompts depend on the run's
    number of events and its environment, so `args` gives them. These are the six organizations of
    PRIMITIVES_ORGANIZATIONS. The agents of a departures organization are counted under labels of their own that do not
    depend on the prompt, because several have the same prompt (see _usage_client)."""
    if getattr(args, "prompts", "advice") == "primitives":
        labels = {PP.bidder_prompt(organization, args.events, args.environment): f"bidder_{organization}"
                  for organization in PRIMITIVES_ORGANIZATIONS}
        labels.update({PP.analyst_prompt(organization, args.events, args.environment): label
                       for organization, label in (("organized", "analyst"), ("organized_revenue", "analyst_revenue"))})
        labels.update({P.MEDIATOR_PROMPT: "mediator", P.BUYER_AGENT_PROMPT: "buyer_agent"})
        return labels
    labels = {P.executive_prompt(organization): f"bidder_{organization}" for organization in P.ORGANIZATIONS}
    labels.update({P.sealed_bid_prompt(): "bidder_sealed", P.ANALYST_PROMPT: "analyst",
                   P.MEDIATOR_PROMPT: "mediator", P.BUYER_AGENT_PROMPT: "buyer_agent"})
    return labels


def _usage_client(client, organization, kind):
    """The chat client that an agent of this organization holds. The agents of the six organizations of
    PRIMITIVES_ORGANIZATIONS share the cell's client, which counts usage under the label of the system prompt
    (_agent_labels). The agents of a departures organization hold a view of it that counts under
    <kind>_<organization> (bidder_brief, analyst_brief_cost, ...), because brief and forward, for one, have the same
    bidder prompt and brief and the six organizations' organized have the same analyst prompt."""
    if organization in DEPARTURE_ORGANIZATIONS:
        return client.labeled(f"{kind}_{organization}")
    return client


def _make_client(args):
    """The chat client of one cell, which counts usage under the label of each system prompt."""
    from orgsim.llm import make_client
    return make_client(args.base_url, api_key=args.api_key, local=not args.hosted,
                       default_sampling=args.default_sampling, max_tokens=args.max_tokens,
                       max_retries=args.max_retries, timeout=args.timeout, agent_labels=_agent_labels(args))


def build_sealed_bid_agents(args, seed):
    """The two bidders of the second-price auction. A model bidder reads the primitives-only prompt
    (P.sealed_bid_prompt()) and, with sealed_bid=True, the versions of the shared texts that state the
    primitives and the rules and give no advice; the auction has no platform agent and no analyst."""
    if args.backend == "mock":
        return {sid: MockStrategicSupplier(seed=seed * 10 + i) for i, sid in enumerate(SUPPLIER_IDS)}
    from orgsim.llm_agents import LocalSupplierAgent
    client = _make_client(args)
    return {sid: LocalSupplierAgent(client, model=args.model, system_prompt=P.sealed_bid_prompt(),
                                    use_tools=not args.no_tools, sealed_bid=True)
            for sid in SUPPLIER_IDS}


def build_agents(organization, args, seed):
    """The agents of one cell. `organization` is one name, which both suppliers have, or a mapping from each
    supplier id to its name (a mixed market). A supplier's bidder reads the prompt of its own organization, and only
    a supplier whose organization is in ANALYST_ORGANIZATIONS has an analyst. Under --prompts primitives
    (args.prompts) the bidders and the analysts read the texts of orgsim/primitives_prompts.py for the run's number of
    events and its environment, the bidders are PrimitivesSupplierAgent with the run's number of rounds, and the
    suppliers with an analyst (PRIMITIVES_ANALYST_ORGANIZATIONS) are those of organized and organized_revenue."""
    primitives = getattr(args, "prompts", "advice") == "primitives"
    organizations = organization_of_each(organization, SUPPLIER_IDS)
    analyst_organizations = ALL_PRIMITIVES_ANALYST_ORGANIZATIONS if primitives else ANALYST_ORGANIZATIONS
    with_analyst = [sid for sid in SUPPLIER_IDS if organizations[sid] in analyst_organizations]
    if args.backend == "mock":
        suppliers = {sid: MockStrategicSupplier(seed=seed * 10 + i) for i, sid in enumerate(SUPPLIER_IDS)}
        analysts = {sid: MockProcurementAnalyst(floor_line=departure_base(organizations[sid]) == "floor")
                    for sid in with_analyst} or None
        return suppliers, MockPlatformMediator(), MockBuyerAgentPlatform(), analysts
    from orgsim.llm_agents import (
        LocalBuyerAgentPlatform, LocalPlatformMediator, LocalProcurementAnalyst, LocalSupplierAgent,
        PrimitivesSupplierAgent,
    )
    client = _make_client(args)
    if primitives:
        suppliers = {sid: PrimitivesSupplierAgent(
                         _usage_client(client, organizations[sid], "bidder"), model=args.model,
                         system_prompt=PP.bidder_prompt(organizations[sid], args.events, args.environment),
                         n_rounds=args.rounds)
                     for sid in SUPPLIER_IDS}
    else:
        suppliers = {sid: LocalSupplierAgent(client, model=args.model,
                                             system_prompt=P.executive_prompt(organizations[sid]),
                                             use_tools=not args.no_tools)
                     for sid in SUPPLIER_IDS}
    mediator = LocalPlatformMediator(client, model=args.model, system_prompt=P.MEDIATOR_PROMPT)
    buyer_agent = LocalBuyerAgentPlatform(client, model=args.model, system_prompt=P.BUYER_AGENT_PROMPT)
    if primitives:
        analysts = {sid: LocalProcurementAnalyst(
                        _usage_client(client, organizations[sid], "analyst"), model=args.model,
                        system_prompt=PP.analyst_prompt(organizations[sid], args.events, args.environment))
                    for sid in with_analyst} or None
    else:
        analysts = {sid: LocalProcurementAnalyst(client, model=args.model, system_prompt=P.ANALYST_PROMPT)
                    for sid in with_analyst} or None
    return suppliers, mediator, buyer_agent, analysts


def _price_patterns(price):
    """Regular expressions for the ways a reply can quote a price: the number with two decimals
    ("123.45") and, for a whole number of dollars, also "$" and the integer ("$124"). A longer
    number that holds the same digits ("1123.45", "123.456", "$1245", "$124.50") is another price."""
    patterns = [rf"(?<!\d){re.escape(f'{price:.2f}')}(?!\d)"]
    if price == int(price):
        patterns.append(rf"\${int(price)}(?!\d|\.\d)")
    return patterns


def _rival_price_quotes(outcome):
    """The number of platform replies in the event, to either supplier, that contain any price the
    other supplier bid in this event (in any round), in the forms of _price_patterns. A reply counts
    once however many prices it holds. Prices from past events are not counted."""
    count = 0
    for sid, replies in outcome.mediator_feedback.items():
        rival_prices = {bid.terms.price for other, bids in outcome.all_bids.items() if other != sid
                        for bid in bids}
        patterns = [pattern for price in rival_prices for pattern in _price_patterns(price)]
        count += sum(any(re.search(pattern, reply) for pattern in patterns) for reply in replies)
    return count


def _market_label(market):
    """The organization column of a cell: the organization both suppliers have, or "<a>:<b>" when A has a and B has
    b. `market` is a name or a mapping from supplier id to name, as run_sequence takes it."""
    organizations = organization_of_each(market, SUPPLIER_IDS)
    first, second = organizations["A"], organizations["B"]
    return first if first == second else f"{first}:{second}"


def _market_fields(market):
    """The three organization fields of a row or a record: organization, and the organization of A and of B."""
    organizations = organization_of_each(market, SUPPLIER_IDS)
    return {"organization": _market_label(market), "organization_a": organizations["A"],
            "organization_b": organizations["B"]}


def _row(institution, organization, seed, r, setting=ADVICE_SETTING, columns=COLUMNS):
    """One summary.csv row. The columns come from the event's result, except the efficient winner
    and the three counts (four with FLOOR_FIELD in `columns`), which come from the outcome. `organization` is a name
    or a mapping from supplier id to name (a mixed market). `setting` holds the run's prompt_set and environment."""
    outcome = r["outcome"]
    row = {"institution": institution, **_market_fields(organization), **setting, "seed": seed, **r,
           "efficient_winner": outcome.efficient_winner_id}
    for field, tag in TAG_FIELDS.items():
        row[field] = sum(tag in b.reasoning for bids in outcome.all_bids.values() for b in bids)
    row[QUOTES_FIELD] = _rival_price_quotes(outcome)
    if FLOOR_FIELD in columns:
        row[FLOOR_FIELD] = sum(FLOOR_PRICE_TAG in b.reasoning for bids in outcome.all_bids.values() for b in bids)
    return {column: row[column] for column in columns}


def _outcome_fields(outcome):
    """The outcome's fields, unrounded. When nothing was awarded, winner_true_cost is 0.0, a
    placeholder and not a cost."""
    return {
        "regime": outcome.regime,
        "run_id": outcome.run_id,
        "winner_id": outcome.winner_id,
        "awarded": outcome.awarded,
        "winning_bid": outcome.winning_bid.to_dict() if outcome.winning_bid is not None else None,
        "winner_true_cost": outcome.winner_true_cost,
        "buyer_surplus": outcome.buyer_surplus,
        "supplier_profit": outcome.supplier_profit,
        "joint_surplus": outcome.joint_surplus,
        "efficient_surplus": outcome.efficient_surplus,
        "efficient_winner_id": outcome.efficient_winner_id,
        "correct_winner": outcome.correct_winner,
        "efficiency_ratio": outcome.efficiency_ratio,
        "all_true_costs": outcome.all_true_costs,
        "all_inflation_rates": outcome.all_inflation_rates,
    }


def _record(institution, organization, seed, r, setting=ADVICE_SETTING):
    """One events.jsonl record: the event's result with the outcome object replaced by its fields,
    and for each supplier every round's bid with its stated reasoning and the platform's reply.
    `setting` holds the run's prompt_set and environment.

    A reply is stored with the bid it answers: bids[sid][k]["feedback"] answers bid k. Both
    institutions give no reply after the last round, so that bid's "feedback" is null. A bid of a supplier whose firm's
    analyst set a minimum profit (a floor organization) also holds "floor", what the enforcement did to it: the
    minimum, whether the bid was returned (with the first bid, the follow-up and the bid sent back) and whether its
    price was raised.
    """
    outcome = r["outcome"]
    bids = {}
    for sid, supplier_bids in outcome.all_bids.items():
        replies = outcome.mediator_feedback.get(sid, [])
        bids[sid] = [{"round": b.round_num, **b.terms.to_dict(), "reasoning": b.reasoning,
                      "feedback": replies[k] if k < len(replies) else None,
                      **({"floor": b.floor_audit} if b.floor_audit else {})}
                     for k, b in enumerate(supplier_bids)]
    return {"institution": institution, **_market_fields(organization), **setting, "seed": seed,
            **{key: _outcome_fields(value) if key == "outcome" else value for key, value in r.items()},
            "bids": bids}


def _dumps(record):
    """A record as one JSON line. A value JSON cannot hold (NaN, an object) raises; none is
    turned into text."""
    return json.dumps(record, allow_nan=False, ensure_ascii=False)


def _save_event(jl, rows, institution, organization, seed, r, setting=ADVICE_SETTING, columns=COLUMNS):
    """Write one finished event to events.jsonl, flushed at once, and keep its summary row."""
    line = _dumps(_record(institution, organization, seed, r, setting))
    rows.append(_row(institution, organization, seed, r, setting, columns))
    jl.write(line + "\n")
    jl.flush()


def _finish_event(jl, rows, institution, organization, seed, checkpoint, r, setting=ADVICE_SETTING, columns=COLUMNS):
    """What run_sequence calls when an event ends: save the event, then update run.json."""
    _save_event(jl, rows, institution, organization, seed, r, setting, columns)
    checkpoint()


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git(root, *args):
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=10)


def _git_state(root=REPO):
    """(commit, dirty) for the git repository whose top level is `root`: the commit HEAD names, and
    whether tracked files differ from it (None if git status fails). ("unknown", None) when root
    is not the top level of a repository (an exported copy, or a folder inside some other
    repository, whose commit says nothing about this code), when HEAD names no commit, or when git
    is not available."""
    try:
        top = _git(root, "rev-parse", "--show-toplevel")
        if top.returncode != 0 or not os.path.samefile(top.stdout.strip(), root):
            return "unknown", None
        head = _git(root, "rev-parse", "HEAD")
        commit = head.stdout.strip()
        if head.returncode != 0 or not commit:
            return "unknown", None
    except (OSError, subprocess.SubprocessError):
        return "unknown", None
    try:  # no optional locks: a run must not take the index lock while another process commits
        status = _git(root, "--no-optional-locks", "status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.SubprocessError):
        return commit, None
    return commit, (bool(status.stdout.strip()) if status.returncode == 0 else None)


def _source_sha256(root=REPO):
    """SHA-256 of the code that makes the run, which identifies it without the git history. The
    recipe is in the epilog: orgsim/*.py and scripts/run_demo.py in sorted order of their relative
    paths, each adding its relative path (with /), a NUL byte, its bytes and a NUL byte."""
    paths = sorted([p.relative_to(root).as_posix() for p in (root / "orgsim").glob("*.py")]
                   + ["scripts/run_demo.py"])
    digest = hashlib.sha256()
    for rel in paths:
        digest.update(rel.encode("utf-8") + b"\0" + (root / rel).read_bytes() + b"\0")
    return digest.hexdigest()


def _openai_version():
    try:
        import openai
    except ImportError:
        return "not installed"
    return openai.__version__


def _clean_base_url(url):
    """The URL as recorded in run.json: scheme, host and path. The user-info (user:password@), the
    query string and the fragment are dropped, because they can carry credentials. A value that is
    not an http(s) URL is not recorded."""
    if not re.match(r"https?://", url, re.IGNORECASE):
        return "<not an http(s) URL>"
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError:
        return "<unparseable URL>"
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc.rpartition("@")[2], parts.path, "", ""))


def _url_secrets(url):
    """What in a URL must not be recorded, as written and decoded: the user name and the password
    (some services take a token as the user name, with a dummy password) and each query value of
    CREDENTIAL_LENGTH characters or more (a short one such as ?v=2 is not a credential and would
    redact unrelated numbers). A server can repeat these in an error message."""
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError:
        return []
    userinfo, at, _ = parts.netloc.rpartition("@")
    found = [value for value in (pair.partition("=")[2] for pair in parts.query.split("&"))
             if len(value) >= CREDENTIAL_LENGTH]
    if at:
        user, _, password = userinfo.partition(":")
        found += [user, password]
    return [s for secret in found for s in (secret, urllib.parse.unquote(secret)) if s]


def _secrets(args):
    """The strings that must not reach run.json, longest first so that one inside another leaves
    nothing behind: the API key as given and as the environment holds it, and the credentials in
    --base-url."""
    found = [args.api_key, os.environ.get("OPENAI_API_KEY"), *_url_secrets(args.base_url)]
    return sorted({s for s in found if s}, key=len, reverse=True)


def _redacted(argv):
    """argv as recorded: the value of --api-key replaced by <redacted> and the value of --base-url
    cleaned, in both spellings (parse_args takes no abbreviation of an option, so these are the
    only ones)."""
    handlers = {"--api-key": lambda value: "<redacted>", "--base-url": _clean_base_url}
    out, handler = [], None
    for token in argv:
        flag, equals, value = token.partition("=")
        if handler is not None:  # the value of the option before it
            out.append(handler(token))
            handler = None
        elif flag in handlers and equals:
            out.append(f"{flag}={handlers[flag](value)}")
        elif flag in handlers:
            out.append(token)
            handler = handlers[flag]
        else:
            out.append(token)
    return out


def _scrub(text, secrets):
    """The text without the secrets. A secret of CREDENTIAL_LENGTH characters or more is removed
    wherever it appears, even joined to other characters ("api_key%3D<key>", "Bearer%20<key>",
    "x-<key>"), because a server can repeat a key in any form. A shorter one is probably a dummy
    value, and only whole tokens are replaced, so that a dummy key such as "local" does not damage
    "localhost" in the recorded settings."""
    for secret in secrets:
        if len(secret) >= CREDENTIAL_LENGTH:
            text = text.replace(secret, "<redacted>")
        else:
            text = re.sub(rf"(?<![\w-]){re.escape(secret)}(?![\w-])", "<redacted>", text)
    return text


def _scrubbed(value, secrets):
    """A copy of a JSON value with the secrets removed from every string in it."""
    if isinstance(value, str):
        return _scrub(value, secrets)
    if isinstance(value, dict):
        return {key: _scrubbed(item, secrets) for key, item in value.items()}
    if isinstance(value, list):
        return [_scrubbed(item, secrets) for item in value]
    return value


def _write_json(path, data, secrets=()):
    """Write data to path as JSON; every string in it passes through the scrubber first."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(_scrubbed(data, secrets), f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def _write_json_atomic(path, data, secrets=()):
    """_write_json, but the file is never half written: the JSON goes to a temporary file in the same
    folder, which then replaces `path` in one step, so a process that is killed leaves the old file or
    the new one. If anything fails, the temporary file is removed and `path` is as it was."""
    temporary = path.with_name(path.name + ".tmp")
    try:
        _write_json(temporary, data, secrets)
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _write_json_safely(path, data, secrets=()):
    """Write the end of the record. _write_json_atomic keeps the old file whole if the write is cut off, but
    os.replace fails with PermissionError whenever another program has a handle open on the file (a sync
    client, an indexer, an editor). On an OSError the data is written directly, which such a handle usually
    allows, so a lock does not lose the record."""
    try:
        _write_json_atomic(path, data, secrets)
    except OSError:
        _write_json(path, data, secrets)


def _runs_ai_institution(args):
    """Whether the command runs the mediator or the buyer agent (the institutions that have rounds)."""
    return any(name in AI_INSTITUTIONS for name in args.institutions)


def _run_info(args, raw_argv):
    """The run.json fields known at the start. Options that apply only to the openai backend are
    null for a mock run, and so is "usage", which an openai run fills in after each event and at its end.
    "rounds" is null when only the second-price auction runs: it has one sealed bid and no rounds. "organizations"
    is null in a run of mixed markets, which "mixed" lists; "mixed" is null otherwise. "prompts" and "environment"
    are the settings of --prompts and --environment."""
    mock = args.backend == "mock"
    commit, dirty = _git_state()
    info = {"status": "running", "error": None}
    if mock:
        info["note"] = _mock_note([_market_label(market) for market in _markets(args)], mixed=bool(args.mixed))
    info.update({
        "started_at": _now(),
        "finished_at": None,
        "usage": None,
        "argv": _redacted(raw_argv),
        "backend": args.backend,
        "model": args.model,
        "base_url": None if mock else _clean_base_url(args.base_url),
        "hosted": None if mock else args.hosted,
        "default_sampling": None if mock else args.default_sampling,
        "max_tokens": None if mock else args.max_tokens,
        "max_retries": None if mock else args.max_retries,
        "timeout": None if mock else args.timeout,
        "no_tools": None if mock else args.no_tools,
        "institutions": args.institutions,
        "organizations": None if args.mixed else args.organizations,
        "mixed": args.mixed,
        "prompts": args.prompts,
        "environment": args.environment,
        "events": args.events,
        "rounds": args.rounds if _runs_ai_institution(args) else None,
        "max_history_events": args.max_history_events,
        "seeds": args.seeds,
        "debug_log": args.debug_log,
        "git_commit": commit,
        "git_dirty": dirty,
        "source_sha256": _source_sha256(),
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "openai": _openai_version(),
    })
    return info


def _at_least(n):
    """An argparse type: an integer of at least n."""
    def parse(text):
        value = int(text)
        if value < n:
            raise argparse.ArgumentTypeError(f"must be at least {n}")
        return value
    parse.__name__ = "integer"  # argparse names the type in its message for a non-integer
    return parse


def _mixed_market(text):
    """An argparse type: "<organization of A>:<organization of B>", two different organizations. The text is kept as
    given, because run.json records it so."""
    first, colon, second = text.partition(":")
    if not colon or first not in ALL_ORGANIZATIONS or second not in ALL_ORGANIZATIONS:
        raise argparse.ArgumentTypeError(
            f"must be <organization of A>:<organization of B>, each one of {', '.join(P.ORGANIZATIONS)} "
            "(for example single:organized), or with --prompts primitives one of "
            f"{', '.join(ALL_PRIMITIVES_ORGANIZATIONS)}")
    if first == second:
        raise argparse.ArgumentTypeError(
            "must give two different organizations (--organizations runs markets whose suppliers share one)")
    return text


_mixed_market.__name__ = "market"


def _seed(text):
    value = int(text)
    if not 0 <= value <= MAX_SEED:
        raise argparse.ArgumentTypeError(f"must be between 0 and {MAX_SEED}")
    return value


_seed.__name__ = "integer"


def _seconds(text):
    """An argparse type: a number of seconds from 1 to MAX_TIMEOUT. A whole number stays an integer (300, not 300.0)."""
    value = float(text)
    if not 1 <= value <= MAX_TIMEOUT:  # false for nan too
        raise argparse.ArgumentTypeError(f"must be a number from 1 to {MAX_TIMEOUT}")
    return int(value) if value == int(value) else value


_seconds.__name__ = "number"


class _Formatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """Show each option's default and keep the line breaks of the epilog."""


def build_parser():
    # No abbreviations: run.json hides the value of --api-key by the option's full name.
    ap = argparse.ArgumentParser(description=__doc__, epilog=EPILOG, formatter_class=_Formatter,
                                 allow_abbrev=False)
    ap.add_argument("--backend", choices=["mock", "openai"], default="mock",
                    help="mock: scripted agents that need no model; openai: language-model agents "
                         "on an OpenAI-compatible endpoint")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL,
                    help="chat endpoint of the model server (openai backend only); run.json leaves "
                         "out any user name, password and query string in it")
    ap.add_argument("--api-key", default=None,
                    help="key for the endpoint (openai backend only); defaults to $OPENAI_API_KEY, "
                         "and local servers need none")
    ap.add_argument("--model", default=None,
                    help="model name (required with the openai backend, not allowed with mock)")
    ap.add_argument("--hosted", action="store_true",
                    help="drop the local-server sampling fields (top_k, thinking switch) that hosted "
                         "endpoints may reject (openai backend only)")
    ap.add_argument("--default-sampling", action="store_true",
                    help="send no sampling settings, so the provider's defaults apply; needed for Claude 5 "
                         "models (openai backend only)")
    ap.add_argument("--max-tokens", type=_at_least(1), default=None, metavar="N",
                    help="cap each reply, hidden reasoning included (openai backend only)")
    ap.add_argument("--max-retries", type=_at_least(0), default=None, metavar="N",
                    help="retries for connection errors, rate limits and server errors (the OpenAI SDK's own "
                         "retry with backoff); the SDK default is 2 (openai backend only)")
    ap.add_argument("--timeout", type=_seconds, default=None, metavar="SECONDS",
                    help="seconds before a request is abandoned and retried (the OpenAI SDK's default is 600); "
                         f"from 1 to {MAX_TIMEOUT}; connecting keeps the SDK's limit of 5 seconds "
                         "(openai backend only)")
    ap.add_argument("--no-tools", action="store_true",
                    help="bidders do not get the cost tool (openai backend only)")
    ap.add_argument("--institutions", nargs="+", choices=list(INSTITUTIONS), default=list(AI_INSTITUTIONS),
                    help="institutions to run, each at most once: mediator (the AI mediator), "
                         "buyer_agent (the AI buyer agent) or second_price (a one-time second-score "
                         "sealed-bid auction, the structural benchmark; needs --events 1 and "
                         "--organizations single, and has no rounds)")
    ap.add_argument("--organizations", nargs="+", choices=ALL_ORGANIZATIONS, default=DEFAULT_ORGANIZATIONS,
                    help="how the supplier is organized, each at most once: single (the bidder reads "
                         "its own record), organized (an analyst reads it and briefs the bidder) or "
                         "organized_guided (as organized, and the bidder also reads the analysis guidance); "
                         "both suppliers of a market have the same organization. With --prompts primitives the "
                         "organizations are single, single_norecord (the firm keeps no record), organized, "
                         "organized_pass (the analyst passes the record to the bidder unchanged, with no model "
                         "call), single_revenue and organized_revenue (the bidder is judged on revenue), all of "
                         "them if none is given; organized_guided is not among them, and the other four are not "
                         "organizations of --prompts advice. The departures organizations of --prompts primitives "
                         "(brief, forward, floor, brief_cap, brief_cost, each also with the suffix _revenue) are run "
                         "only when named")
    ap.add_argument("--prompts", choices=list(PP.PROMPT_SETS), default="advice",
                    help="the texts the agents of the AI institutions read: advice (the earlier prompts, which "
                         "carry strategic advice; every earlier run was made with them) or primitives (texts "
                         "that state the objective, the information and the rules and give no advice; it needs "
                         "--no-tools with a model, --events of at least 2, and neither --institutions "
                         "second_price nor --max-history-events)")
    ap.add_argument("--environment", choices=list(PP.ENVIRONMENTS), default="random",
                    help="how a seed's events are drawn: random (each event's contract and cost types are drawn "
                         "independently) or persistent (event 1 of the random plan of the seed is repeated at every "
                         "event); persistent needs --prompts primitives, because the advice prompts say that a "
                         "supplier's costs are drawn anew for each event")
    ap.add_argument("--mixed", nargs="+", type=_mixed_market, default=None, metavar="A_ORG:B_ORG",
                    help="markets in which supplier A and supplier B are organized differently, each given as "
                         "<organization of A>:<organization of B> (for example single:organized), each at most "
                         "once; they take the place of --organizations, which cannot be given with it, and "
                         "second_price cannot be run in them. Neither supplier nor platform is told the rival's "
                         "organization. Ties go to A, so run both orders of a pair. With --prompts primitives "
                         "the pairs are of the organizations of that set")
    ap.add_argument("--events", type=_at_least(1), default=6,
                    help="events per sequence; the two suppliers keep a record of the earlier events")
    ap.add_argument("--rounds", type=_at_least(1), default=3,
                    help="bidding rounds per event of the mediator and the buyer agent, at least 2 because "
                         "the platform answers between rounds; the second-price auction has one sealed bid "
                         "and does not use it")
    ap.add_argument("--max-history-events", type=_at_least(0), default=0, metavar="N",
                    help="keep only the last N past events in each supplier's record and in the mediator's "
                         "memory of each supplier, which keeps prompts within a model's context in long "
                         "runs; 0 keeps every past event")
    ap.add_argument("--seeds", type=_seed, nargs="+", default=[1],
                    help=f"seeds from 0 to {MAX_SEED}, each at most once; a seed fixes the event plan "
                         "and the mock agents")
    ap.add_argument("--out", default="outputs/demo",
                    help="folder for events.jsonl, summary.csv and run.json")
    ap.add_argument("--debug-log", metavar="PATH", default=None,
                    help="log the model calls to this file (the bidders' requests and replies, the "
                         "platforms' replies, the analysts' histories and replies, and the error of "
                         "each failed attempt), for finding out why a run failed")
    return ap


def _check_prompt_set(ap, args):
    """Reject what the prompt set of --prompts cannot take: organizations that belong to the other set, the options
    that its texts would make untrue, and a persistent environment under the advice prompts."""
    allowed = ORGANIZATIONS_OF[args.prompts]
    named = [("--organizations", name) for name in ([] if args.organizations is DEFAULT_ORGANIZATIONS
                                                    else args.organizations)]
    named += [("--mixed", name) for pair in args.mixed or [] for name in pair.split(":")]
    for flag, name in named:
        if name not in allowed:
            primitives = args.prompts == "primitives"
            ap.error(f"{flag} names {name}, which is not an organization of --prompts {args.prompts} "
                     f"(those are {', '.join(PRIMITIVES_ORGANIZATIONS if primitives else allowed)})"
                     + (f"; its departures organizations are {', '.join(DEPARTURE_ORGANIZATIONS)}"
                        if primitives else ""))
    if args.prompts == "advice":
        if args.environment == "persistent":
            ap.error("--environment persistent needs --prompts primitives: the advice prompts say that a supplier's "
                     "cost profile is drawn anew for each event, which would be untrue if event 1 were repeated at "
                     "every event")
        return
    if args.backend == "openai" and not args.no_tools:
        ap.error("--prompts primitives needs --no-tools: its bidders have no cost tool")
    if "second_price" in args.institutions:
        ap.error("--prompts primitives does not apply to --institutions second_price: its bidder reads a prompt of "
                 "its own")
    if args.events < 2:
        ap.error("--prompts primitives needs --events of at least 2: its texts describe a sequence of events in "
                 "which a company has a record of past events")
    if args.max_history_events:
        ap.error("--prompts primitives takes no --max-history-events: its texts say that the analyst forwards the "
                 "full record and that the platform remembers past events")


def _check_options(ap, args):
    """Reject the combinations of options that argparse cannot see."""
    if args.backend == "openai" and not args.model:
        ap.error("--model is required with --backend openai")
    if args.backend == "mock":
        for flag, given in (("--model", args.model is not None), ("--api-key", args.api_key is not None),
                            ("--hosted", args.hosted), ("--default-sampling", args.default_sampling),
                            ("--max-tokens", args.max_tokens is not None),
                            ("--max-retries", args.max_retries is not None),
                            ("--timeout", args.timeout is not None), ("--no-tools", args.no_tools),
                            ("--base-url", args.base_url != DEFAULT_BASE_URL)):
            if given:
                ap.error(f"{flag} applies only to --backend openai")
    for flag, values in (("--seeds", args.seeds), ("--institutions", args.institutions),
                         ("--organizations", args.organizations), ("--mixed", args.mixed or [])):
        if len(set(values)) != len(values):
            ap.error(f"{flag} lists a value more than once")
    if args.mixed:
        if args.organizations is not DEFAULT_ORGANIZATIONS:  # given on the command line, whatever it lists
            ap.error("--mixed takes the place of --organizations: give one of the two, not both")
        if "second_price" in args.institutions:
            ap.error("--mixed does not apply to --institutions second_price: its auction is single only")
    _check_prompt_set(ap, args)
    if _runs_ai_institution(args) and args.rounds < 2:
        ap.error("argument --rounds: must be at least 2 for the mediator and the buyer agent, "
                 "because the platform answers between rounds")
    if "second_price" in args.institutions and not args.mixed:
        if args.events != 1:
            ap.error("--institutions second_price needs --events 1: its rules tell the bidder that this is a "
                     "one-time event, which is true only when no record of earlier events exists "
                     "(for several auctions give several --seeds)")
        if args.organizations != ["single"]:
            ap.error("--institutions second_price needs --organizations single: with one event there is no "
                     "record for an analyst to read, so the organizations would not differ")


def _clients_of(*agents):
    """The distinct chat clients behind some agents that count usage (the model agents hold one; an agent that holds a
    labeled view of a client counts in that client, its `root`)."""
    found = {}
    for agent in agents:
        client = getattr(agent, "client", None)
        client = getattr(client, "root", client)
        if hasattr(client, "usage_summary"):
            found.setdefault(id(client), client)
    return found


def _markets(args):
    """The market of each cell, in the order given: with --mixed, one for each pair, a mapping from supplier id to
    the organization of that supplier; otherwise one for each organization, the name that both suppliers have."""
    if args.mixed:
        return [dict(zip(SUPPLIER_IDS, pair.split(":"))) for pair in args.mixed]
    return list(args.organizations)


def _columns(args):
    """The columns of summary.csv: COLUMNS, and FLOOR_FIELD when a supplier of the run has a departures organization."""
    named = [name for market in _markets(args) for name in organization_of_each(market, SUPPLIER_IDS).values()]
    return COLUMNS + [FLOOR_FIELD] if any(name in DEPARTURE_ORGANIZATIONS for name in named) else COLUMNS


def _run_cells(args, jl, rows, clients, checkpoint):
    """Run every seed x institution x market cell, a market being an organization or, with --mixed, a pair
    of organizations. Each finished event goes to jl at once, and then checkpoint() is called. The clients that
    count usage are collected in `clients` before the cell runs, so that a cell that fails is counted too."""
    setting = {"prompt_set": args.prompts, "environment": args.environment}
    columns = _columns(args)
    for seed in args.seeds:
        plan = pre_draw_event_sequence(args.events, get_contract_scenarios(), seed=seed, environment=args.environment,
                                       prompt_set=args.prompts)
        for inst in args.institutions:
            for org in _markets(args):
                print(f"=== seed {seed} | {inst} | {_market_label(org)} ===")
                if inst == "second_price":
                    suppliers, mediator, buyer_agent, analysts = build_sealed_bid_agents(args, seed), None, None, None
                else:
                    suppliers, mediator, buyer_agent, analysts = build_agents(org, args, seed)
                clients.update(_clients_of(*suppliers.values(), mediator, buyer_agent, *(analysts or {}).values()))
                run_sequence(
                    INSTITUTIONS[inst], args.events, args.rounds, suppliers, plan,
                    mediator=mediator, buyer_agent_platform=buyer_agent, analysts=analysts,
                    seq_id=1, organization=org, max_history_events=args.max_history_events,
                    on_event=functools.partial(_finish_event, jl, rows, inst, org, seed, checkpoint,
                                               setting=setting, columns=columns),
                    prompt_set=args.prompts,
                )


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        # The runner prints symbols that a piped Windows console (cp1252) cannot encode.
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    ap = build_parser()
    args = ap.parse_args(raw_argv)
    _check_options(ap, args)
    if args.prompts == "primitives" and args.organizations is DEFAULT_ORGANIZATIONS:
        args.organizations = list(PRIMITIVES_ORGANIZATIONS)    # no --organizations: every organization of the set

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    secrets = _secrets(args)
    run = _run_info(args, raw_argv)
    _write_json(out / "run.json", run, secrets)  # replaces the record of an earlier run, first
    if args.backend == "mock":
        print(_mock_note([_market_label(market) for market in _markets(args)], mixed=bool(args.mixed)))

    rows, clients, debug_file = [], {}, None

    def checkpoint():
        """Rewrite run.json, still "running", with the usage so far, so that a run that is killed
        outright (no exception to catch, no `finally`) keeps the usage as of its last update. A mock
        run has no usage. A write that fails with an OSError (another program holds the file) is a
        warning: it must not fail the run, and the write when the run ends tries again. An update
        that failed leaves the usage in run.json behind until the next one succeeds."""
        if args.backend == "mock":
            return
        run["usage"] = sum_usage(client.usage_summary() for client in clients.values())
        try:
            _write_json_atomic(out / "run.json", run, secrets)
        except OSError as exc:
            print(_scrub(f"warning: could not update {out / 'run.json'} after this event ({exc}); the run continues",
                         secrets))

    try:
        for name in EARLIER_OUTPUTS:  # a file that cannot be deleted (held open) fails this run
            try:
                (out / name).unlink(missing_ok=True)
            except OSError as exc:
                raise type(exc)(f"cannot delete {out / name}, left by an earlier run: {exc}") from exc
        if args.debug_log:
            debug_file = open(args.debug_log, "w", encoding="utf-8")
            llm_agents.DEBUG_LOG_FILE = debug_file
        with open(out / "events.jsonl", "w", encoding="utf-8", newline="\n") as jl:
            _run_cells(args, jl, rows, clients, checkpoint)
        with open(out / "summary.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_columns(args))
            writer.writeheader()
            writer.writerows(rows)
        run["status"] = "complete"
    except BaseException as exc:  # an interrupted run is marked failed too
        run["status"] = "failed"
        run["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if debug_file is not None:
            llm_agents.DEBUG_LOG_FILE = None
            debug_file.close()
        if args.backend != "mock":  # the requests of every cell's client, whether the run completed or failed
            run["usage"] = sum_usage(client.usage_summary() for client in clients.values())
        run["finished_at"] = _now()
        _write_json_safely(out / "run.json", run, secrets)
    print(f"Wrote {len(rows)} events to {out}: {', '.join(OUTPUT_FILES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
