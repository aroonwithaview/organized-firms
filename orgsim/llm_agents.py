"""LLM-backed agents (OpenAI-compatible chat completions).

Copied from the earlier simulation. Changes: the model and the system prompt are constructor
arguments instead of module globals; the analyst no longer receives an early-event instruction to
default to GUARDED, no longer has its DEFENSIVE posture overridden, and no longer writes a canned
briefing before the first event; its posture is read from a statement in the briefing ("Posture:
GUARDED", "I recommend GUARDED") before any other mention of a label (see _classify_posture); a
failed or unusable model reply now raises instead of being replaced by a scripted briefing or a
default bid; a platform reply that is empty once its thinking block is removed raises instead of
reaching the bidders as an empty message; and the mediator is shown the bid history of every
supplier in the event, not only of the one it is answering. Bids whose price the below-cost rule
set, or that were made without the cost tool after the tool path failed (an error, or no usable
bid in five calls), are marked in their reasoning (the tags are defined in orgsim.interfaces). A
price the rule set is also marked in the bidder's list of its earlier rounds of the event.

Several messages were also corrected to be true in this environment. Bids are asked for "this
round", not as a "final" bid. The instruction to use the cost tool is sent only in the
conversation that offers it (a fallback or a correction gets the no-tool instruction) and points
to earlier costs only when earlier rounds exist; the message that withdraws the tool says so. The
cost plus 5% is not called a minimum bid, and the cost plus 15% and 25% are not called margins.
The cost tool's warning and the below-cost correction say that a bid below cost would lose money
if it won, not that it guarantees a loss. The analyst's history is not called complete.

The bidder of the primitives prompt set is a PrimitivesSupplierAgent. It sends the context that the runner and the
institution built, the earlier rounds of the event and a round line, and takes every bid as submitted. The analyst of
that set is sent its message with write_briefing, and its reply is the briefing as written. The advice set's
generate_briefing and generate_bid are unchanged (the shared parts of their code moved into _ask and
_previous_rounds_block).

The bidder of the second-price auction (a LocalSupplierAgent with sealed_bid=True) reads versions of
its own of the texts above that carry advice or are untrue of that auction: the cost description
(SupplierCosts.describe_for_sealed_bid), the round instruction, and the cost tool's description and
result. Its bid is taken as submitted (no below-cost correction, no cost-plus-5% rule), and it
accepts no earlier rounds. The AI institutions' bidders read the texts above, unchanged
(tests/test_llm_agents.py and tests/test_demo.py pin digests of their requests).

The sampling settings follow the model card of the open-weight model used in earlier runs
(Qwen3.5-27B); adjust them for other models.
"""
import json
import math
import re
import time
from datetime import datetime

from orgsim.primitives import BuyerPreferences, ContractTerms, SupplierBid, SupplierCosts
from orgsim.departures import FLOOR_RAISED_NOTE, below_floor, dollars, floor_reask_text, follow_up_turns
from orgsim.interfaces import (
    FLOOR_PRICE_TAG, NO_TOOL_TAG, RULE_PRICE_TAG, BuyerAgentPlatform, PlatformMediator, SupplierAgent,
)
from orgsim.organization import AnalystBriefing, ProcurementAnalyst
from orgsim.primitives_prompts import REPLY_LABEL, round_line


# A reasoning model's thinking block, closed or cut off before its closing tag (a truncated reply).
THINK_BLOCK = re.compile(r'<think>.*?(?:</think>|\Z)', re.DOTALL)

DEBUG_LOG_FILE = None  # set to a file opened for writing (UTF-8) to log every model call, including each failed attempt

def debug_log(header: str, content: str = "", data: object = None):
    """Write a debug entry to the log file (no-op if debugging is off)."""
    if DEBUG_LOG_FILE is None:
        return
    DEBUG_LOG_FILE.write(f"\n{'='*80}\n")
    DEBUG_LOG_FILE.write(f"  {header}\n")
    DEBUG_LOG_FILE.write(f"  {datetime.now().isoformat()}\n")
    DEBUG_LOG_FILE.write(f"{'='*80}\n")
    if content:
        DEBUG_LOG_FILE.write(content + "\n")
    if data is not None:
        try:
            DEBUG_LOG_FILE.write(json.dumps(data, indent=2, default=str) + "\n")
        except TypeError:
            DEBUG_LOG_FILE.write(repr(data) + "\n")
    DEBUG_LOG_FILE.flush()


API_MAX_RETRIES = 3
API_RETRY_DELAY = 30  # seconds to wait before retrying a transient error

def api_call_with_retry(func, *args, **kwargs):
    """Wrap an API call with retry logic for transient errors (HTTP 400, 429 or 503, or an
    UNAVAILABLE or RESOURCE_EXHAUSTED error). Each error that is retried goes to the debug log."""
    for attempt in range(API_MAX_RETRIES):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_str = str(e)
            # 400 is retried because the local server used in earlier runs returned intermittent 400s.
            if ("400" in error_str or "503" in error_str or "429" in error_str
                    or "UNAVAILABLE" in error_str
                    or "RESOURCE_EXHAUSTED" in error_str) and attempt < API_MAX_RETRIES - 1:
                wait = API_RETRY_DELAY * (attempt + 1)
                debug_log(f"API transient error (attempt {attempt+1}/{API_MAX_RETRIES}), retrying in {wait}s",
                          f"{type(e).__name__}: {e}")
                print(f"    ⏳ API transient error, retrying in {wait}s "
                      f"(attempt {attempt+2}/{API_MAX_RETRIES})...", flush=True)
                time.sleep(wait)
            else:
                raise


# Cost calculator tool definition (OpenAI function-calling format)
COST_CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "compute_my_cost",
        "description": (
            "Compute your exact total cost for a given set of contract terms. "
            "Call this BEFORE setting your bid price to know precisely what "
            "these terms will cost you. Returns: the total cost, a cost breakdown, "
            "and the cost plus 5%, 15% and 25%."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "delivery_weeks": {
                    "type": "number",
                    "description": "Delivery time in weeks (1.0 to 8.0)",
                },
                "quality_tier": {
                    "type": "integer",
                    "description": "Quality tier (2, 3, 4, or 5)",
                },
            },
            "required": ["delivery_weeks", "quality_tier"],
        },
    },
}


# The second-price bidder's cost tool. The description of COST_CALCULATOR_TOOL tells the bidder to call it
# before setting its price and says it returns the cost plus 5%, 15% and 25%, which suggests markups; this one
# says what the tool returns and nothing else. The tool's name and parameters are the same.
COST_CALCULATOR_TOOL_SEALED = {
    "type": "function",
    "function": {
        **COST_CALCULATOR_TOOL["function"],
        "description": (
            "Compute your exact total cost for a given set of contract terms. "
            "Returns the total cost and a cost breakdown."
        ),
    },
}


def execute_cost_tool(supplier_costs: SupplierCosts, args: dict, sealed_bid: bool = False) -> str:
    """Execute the compute_my_cost tool and return a formatted result string. The second-price bidder
    (sealed_bid) gets the breakdown and the total cost only: the cost plus 5%, 15% and 25% suggest
    markups, and the warning that a bid below cost would lose money if it won is untrue when the
    winner is paid the second score's price."""
    # Default missing args — local models sometimes omit quality_tier
    delivery = float(args.get("delivery_weeks", 6.0))
    quality = int(args.get("quality_tier", 3))
    terms = ContractTerms(
        price=0,
        delivery_weeks=delivery,
        quality_tier=quality,
    )
    total = supplier_costs.cost(terms)

    # Break down costs
    c_del = supplier_costs.gamma_delivery * (supplier_costs.delivery_max - terms.delivery_weeks) ** 2
    c_qual = supplier_costs.gamma_quality * terms.quality_tier ** 2

    breakdown = (
        f"Cost breakdown for {terms.delivery_weeks}wk / T{terms.quality_tier}:\n"
        f"  Base cost:     ${supplier_costs.c_base:.2f}\n"
        f"  + Delivery:    ${c_del:.2f}\n"
        f"  + Quality:     ${c_qual:.2f}\n"
        f"  = TOTAL COST:  ${total:.2f}"
    )
    if sealed_bid:
        return breakdown
    return (
        breakdown + "\n"
        f"  Cost + 5%:     ${total * 1.05:.2f}\n"
        f"  Cost + 15%:    ${total * 1.15:.2f}\n"
        f"  Cost + 25%:    ${total * 1.25:.2f}\n"
        f"WARNING: A bid below ${total:.2f} would lose money if it won."
    )


# What the second-price bidder is told about the round, in place of the AI institutions' texts. Those say
# "pick your 1-2 best configurations" and "Focus on your cost calculation and margin protection", which are
# advice, and number the round. The limit of two cost calls is the agent's (MAX_TOOL_ROUNDS).
SEALED_TOOL_ROUND_TEXT = (
    "You have a compute_my_cost tool that returns your total cost for given delivery and quality terms. "
    "You may call it at most twice. After your tool calls, output only the JSON bid object."
)
SEALED_NO_TOOL_ROUND_TEXT = "Submit your bid as JSON. Keep reasoning brief (1-2 sentences)."


BID_KEYS = ("price", "delivery_weeks", "quality_tier")

# The contract space the frontier is computed over (primitives.compute_efficient_outcome).
DELIVERY_WEEKS_RANGE = (1.0, 8.0)
QUALITY_TIERS = (2, 3, 4, 5)


def _is_bid(data) -> bool:
    """Whether parsed JSON is a usable bid: the three bid fields, a finite positive price, and
    terms inside the contract space (delivery 1-8 weeks, integer quality tier 2-5). A tool call
    written as text, or a bid outside the space, is not."""
    if not (isinstance(data, dict) and all(key in data for key in BID_KEYS)):
        return False
    # A JSON boolean is not a number (float(True) would pass as 1.0).
    if any(isinstance(data[key], bool) for key in BID_KEYS):
        return False
    try:
        price = float(data["price"])
        weeks = float(data["delivery_weeks"])
        tier = float(data["quality_tier"])
    except (TypeError, ValueError, OverflowError):
        return False
    return (math.isfinite(price) and price > 0
            and DELIVERY_WEEKS_RANGE[0] <= weeks <= DELIVERY_WEEKS_RANGE[1]
            and tier.is_integer() and int(tier) in QUALITY_TIERS)


def _previous_rounds_block(supplier_costs: SupplierCosts, history: list[dict],
                           reply_label: str = "Feedback received", primitives: bool = False) -> str:
    """The part of a bidder's message that lists its earlier rounds of the event, each with the cost of its terms
    and the platform's reply (called `reply_label`: "Feedback received" for the advice set, the primitives set's
    "Platform's reply"), ending in a blank line; "" in the first round. A bid whose price the below-cost rule set is
    marked so in the advice set; in the primitives set (`primitives`) no rule sets a price, and a bid whose price the
    floor raised is marked so instead (FLOOR_RAISED_NOTE)."""
    if not history:
        return ""
    text = "PREVIOUS ROUNDS:\n"
    for h in history:
        bid = h.get("bid")
        if bid:
            # Include the cost so the model doesn't re-call the tool
            cost_at_terms = supplier_costs.cost(
                ContractTerms(price=0, delivery_weeks=bid.delivery_weeks,
                              quality_tier=bid.quality_tier)
            )
            if primitives:
                rule = FLOOR_RAISED_NOTE if h.get("price_raised_by_floor") else ""
            else:
                rule = " (price set by rule: cost + 5%)" if h.get("price_set_by_rule") else ""
            text += (
                f"  Round {h.get('round', '?')}: You bid ${bid.price:.2f}{rule}, "
                f"{bid.delivery_weeks}wk delivery, tier {bid.quality_tier} "
                f"(your cost was ${cost_at_terms:.2f})\n"
            )
        fb = h.get("feedback")
        if fb:
            text += f"  {reply_label}: {fb}\n"
    return text + "\n"


class LocalSupplierAgent(SupplierAgent):
    """Supplier agent using any OpenAI-compatible chat endpoint.

    Supports tool-calling: the supplier can call compute_my_cost() to get
    exact cost breakdowns before submitting a bid. This removes mental
    arithmetic from the LLM and lets it focus on strategic decisions.
    """

    MAX_RETRIES = 2
    MAX_TOOL_ROUNDS = 2  # max tool calls before forcing a final answer
    MAX_LOOP_ITERS = 5   # total loop iterations (tool calls + JSON retries)
    # At most MAX_TOOL_ROUNDS cost-tool calls per bid; MAX_LOOP_ITERS also allows JSON re-prompts.

    def __init__(self, client, model: str, system_prompt: str, use_tools: bool = True, sealed_bid: bool = False):
        """sealed_bid makes this the bidder of the second-price auction. It reads the sealed-bid versions
        of the texts that the AI institutions' bidders share (the cost description, the round
        instruction and the cost tool), which state the primitives and the rules and give no advice. It
        has no earlier rounds, and its bid is taken as submitted: the below-cost correction and the
        cost-plus-5% rule are not applied, because the correction says that a bid below cost would lose
        money if it won, which is untrue when the winner is paid the second score's price, and because
        the rule would put an artificial markup into the bid whose markup measures the distance from the
        benchmark."""
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.use_tools = use_tools
        self.sealed_bid = sealed_bid

    def _call_llm_with_tools(
        self, messages: list[dict], model: str, supplier_costs: SupplierCosts,
    ) -> dict:
        """Call the LLM with tool support, handling tool calls in a loop.
        Returns the final bid_data dict."""

        sid = supplier_costs.supplier_id
        tool = COST_CALCULATOR_TOOL_SEALED if self.sealed_bid else COST_CALCULATOR_TOOL
        tools = [tool] if self.use_tools else None
        tool_calls_made = 0

        debug_log(f"[{sid}] _call_llm_with_tools START",
                  f"use_tools={self.use_tools}, MAX_TOOL_ROUNDS={self.MAX_TOOL_ROUNDS}, "
                  f"MAX_LOOP_ITERS={self.MAX_LOOP_ITERS}")
        # Log the initial messages sent to the LLM
        debug_log(f"[{sid}] Initial messages ({len(messages)} msgs)",
                  data=[{"role": m.get("role"), "content": m.get("content", "")[:500]}
                        for m in messages])

        for loop_iter in range(self.MAX_LOOP_ITERS):
            # Disable thinking once tools are exhausted — LM Studio's parser
            # scans <think> blocks for tool-call patterns, creating false
            # positives that loop the model (lmstudio-bug-tracker #1592).
            # With tools done we just need clean JSON, not reasoning.
            want_thinking = tool_calls_made < self.MAX_TOOL_ROUNDS

            debug_log(f"[{sid}] Loop iter {loop_iter+1}/{self.MAX_LOOP_ITERS}",
                      f"tool_calls_made={tool_calls_made}, want_thinking={want_thinking}, "
                      f"tools_available={tools is not None and tool_calls_made < self.MAX_TOOL_ROUNDS}")

            def _do_call():
                # Qwen3.5-27B official sampling parameters (model card):
                #   Thinking mode:     temp=1.0, top_p=0.95, top_k=20,
                #                      presence_penalty=1.5, repetition_penalty=1.0
                #   Non-thinking mode: temp=0.7, top_p=0.8,  top_k=20,
                #                      presence_penalty=1.5, repetition_penalty=1.0
                # presence_penalty=1.5 is critical — it penalises tokens that
                # have already appeared, which discourages the model from
                # looping on tool calls or repeating reasoning chains.
                if want_thinking:
                    temp, top_p = 1.0, 0.95
                else:
                    temp, top_p = 0.7, 0.8
                kwargs = {
                    "model": model,
                    "messages": messages,
                    "temperature": temp,
                    "top_p": top_p,
                    "presence_penalty": 1.5,
                    "extra_body": {
                        "top_k": 20,
                        "chat_template_kwargs": {
                            "enable_thinking": want_thinking,
                        },
                    },
                }
                # Only pass tools if we haven't exhausted tool rounds
                if tools and tool_calls_made < self.MAX_TOOL_ROUNDS:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = "auto"
                # Don't set response_format on the final call — some local
                # models return empty responses with schema enforcement.
                # Instead we rely on the JSON nudge message added below.
                return self.client.chat.completions.create(**kwargs)

            response = api_call_with_retry(_do_call)
            msg = response.choices[0].message

            # Log raw response
            raw_resp_content = msg.content or ""
            has_tool_calls = bool(msg.tool_calls)
            debug_log(f"[{sid}] LLM response (iter {loop_iter+1})",
                      f"has_tool_calls={has_tool_calls}, "
                      f"content_len={len(raw_resp_content)}\n"
                      f"--- RAW CONTENT (first 2000 chars) ---\n"
                      f"{raw_resp_content[:2000]}\n"
                      f"--- END RAW CONTENT ---")

            # Check if the LLM wants to call a tool
            if msg.tool_calls:
                # Strip <think>...</think> from content BEFORE storing in history.
                content_for_history = THINK_BLOCK.sub('', msg.content or '').strip()
                # Process each tool call
                messages.append({
                    "role": "assistant",
                    "content": content_for_history,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                })

                for tc in msg.tool_calls:
                    tool_calls_made += 1
                    if tc.function.name == "compute_my_cost":
                        args = json.loads(tc.function.arguments)
                        result = execute_cost_tool(supplier_costs, args, sealed_bid=self.sealed_bid)
                        debug_log(f"[{sid}] Tool call #{tool_calls_made}",
                                  f"args={args}\nresult:\n{result}")
                        print(f"      🔧 {sid} called compute_my_cost("
                              f"{args.get('delivery_weeks')}wk/T{args.get('quality_tier')})", flush=True)
                    else:
                        result = f"Unknown tool: {tc.function.name}"
                        debug_log(f"[{sid}] Unknown tool call: {tc.function.name}")

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
                # If we've hit the tool limit, add a nudge to produce final JSON
                if tool_calls_made >= self.MAX_TOOL_ROUNDS:
                    nudge = ("You have used all your cost calculations. Now submit "
                             "your bid for this round as a JSON object with these exact fields: "
                             "price, delivery_weeks, quality_tier, reasoning. "
                             "Output ONLY the JSON object, nothing else.")
                    messages.append({"role": "user", "content": nudge})
                    debug_log(f"[{sid}] Tool limit reached — nudge injected",
                              f"tool_calls_made={tool_calls_made}")
                continue  # Loop back for next LLM call

            # No tool calls — this is the final response
            raw_content = msg.content or ""
            content = THINK_BLOCK.sub('', raw_content).strip()

            debug_log(f"[{sid}] No tool calls — attempting JSON parse",
                      f"stripped_content_len={len(content)}\n"
                      f"--- STRIPPED CONTENT ---\n{content[:1000]}\n--- END ---")

            # Try to parse as JSON (direct parse first, then regex extraction)
            bid = self._try_extract_json(content, raw_content)
            if bid is not None:
                debug_log(f"[{sid}] ✅ Bid extracted successfully",
                          f"total_tool_calls={tool_calls_made}, loop_iters={loop_iter+1}",
                          data=bid)
                return bid

            debug_log(f"[{sid}] ❌ JSON extraction failed — re-prompting",
                      f"content was: {content[:500]}")
            # Model returned text instead of JSON — ask again with forced format
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": "Now submit your bid for this round as a JSON object with fields: "
                           "price, delivery_weeks, quality_tier, reasoning."
                           " Delivery must be between 1 and 8 weeks and the quality tier "
                           "a whole number from 2 to 5."
                           " The compute_my_cost tool is not available for this reply.",
            })
            # Force JSON on next iteration
            tools = None
            continue

        # Last resort: should not reach here normally
        debug_log(f"[{sid}] ❌ LOOP EXHAUSTED — raising ValueError",
                  f"tool_calls_made={tool_calls_made}")
        raise ValueError(f"no valid bid after {self.MAX_LOOP_ITERS} model calls")

    @staticmethod
    def _try_extract_json(content: str, raw_content: str = "") -> dict | None:
        """Try multiple strategies to extract a bid JSON from model output.

        1. Direct json.loads on stripped content
        2. Regex extraction of JSON with "price" key from stripped content
        3. Search inside <think> block for JSON, only when the reply has nothing outside
           the thinking block (reasoning distills sometimes embed the answer inside the
           thinking trace). Only a closed block, one that ends with </think>, is searched;
           a reply cut off inside an unclosed block yields no bid.

        A parsed value counts only if it is a usable bid (see _is_bid); anything else, such
        as a tool call written as text or a bid outside the contract space, goes on to the
        next strategy.

        Returns parsed dict or None if all strategies fail.
        """
        if content:
            # Strategy 1: direct parse
            try:
                parsed = json.loads(content)
                if _is_bid(parsed):
                    return parsed
            except json.JSONDecodeError:
                pass
            # Strategy 2: regex extraction (handles "Here is my bid: {...}")
            json_match = re.search(r'\{[^{}]*"price"[^{}]*\}', content, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    if _is_bid(parsed):
                        return parsed
                except json.JSONDecodeError:
                    pass

        # Strategy 3: search inside thinking block, only when there is no answer outside it
        if raw_content and not content:
            think_match = re.search(r'<think>(.*?)</think>', raw_content, re.DOTALL)
            if think_match:
                bid_in_think = re.search(
                    r'\{[^{}]*"price"\s*:[^{}]*"delivery_weeks"\s*:[^{}]*\}',
                    think_match.group(1), re.DOTALL,
                )
                if bid_in_think:
                    try:
                        parsed = json.loads(bid_in_think.group())
                        if _is_bid(parsed):
                            return parsed
                    except json.JSONDecodeError:
                        pass
        return None

    def _call_llm_no_tools(self, prompt_text: str, model: str, sid: str = "", earlier_turns: tuple[dict, ...] = (),
                           with_reply: bool = False) -> dict | tuple[dict, str]:
        """Call the model without tools: the main path when use_tools is False, the
        fallback after the tool path failed, and the path for below-cost corrections.
        Each attempt's request, raw reply and exception goes to the debug log, labelled with `sid`.
        `earlier_turns` are chat messages that follow the user message in the request (the primitives bidder's
        follow-up turn to a bid, with its own first reply before it). With `with_reply` the result is the bid and the
        reply it was read from, without its thinking block.

        Two known failure modes for reasoning/distill models avoided here:

        1. a JSON-schema response_format causes empty responses on some local
           models (same issue noted in _call_llm_with_tools — schema enforcement
           suppresses output entirely). We rely on JSON instruction in the prompt
           instead and extract with regex.

        2. enable_thinking=False causes empty outputs on reasoning distills that
           are trained to always think first. With thinking enabled, the model
           generates <think>...</think> followed by the actual JSON answer. After
           stripping thinking the JSON is present. Without thinking the model
           produces nothing.

        Extra safety net: if content is empty after stripping (model put everything
        inside the thinking block), we search for a valid JSON bid within the
        thinking trace itself before giving up. Only a closed thinking block, one that
        ends with </think>, is searched; a reply cut off inside an unclosed block has no
        bid to find.
        """
        def _do_call():
            # Qwen3.5-27B thinking mode official params (model card):
            #   temp=1.0, top_p=0.95, top_k=20, presence_penalty=1.5
            # presence_penalty=1.5 discourages repetitive reasoning loops.
            return self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt_text},
                    *earlier_turns,
                ],
                temperature=1.0,
                top_p=0.95,
                presence_penalty=1.5,
                # No response_format: schema enforcement returns empty on some models.
                # No enable_thinking=False: reasoning distills output nothing when
                # forced to skip thinking. We strip the thinking block below instead.
                extra_body={
                    "top_k": 20,
                    "chat_template_kwargs": {"enable_thinking": True},
                },
            )

        label = f"[{sid}] " if sid else ""
        for retry in range(3):
            last_problem, last_error = "reply has no usable bid", None  # unless this attempt raises
            debug_log(f"{label}No-tool request (attempt {retry+1}/3)", prompt_text)
            try:
                response = api_call_with_retry(_do_call)
                raw_content = response.choices[0].message.content or ""
                debug_log(f"{label}No-tool raw reply (attempt {retry+1}/3)", raw_content or "(empty reply)")
                content = THINK_BLOCK.sub('', raw_content).strip()
                # The same parser as the tool loop: whole reply first, then a flat object,
                # then the thinking block only when nothing is outside it.
                parsed = self._try_extract_json(content, raw_content)
                if parsed is not None:
                    return (parsed, content) if with_reply else parsed
                debug_log(f"{label}No usable bid in the reply (attempt {retry+1}/3)")
                print(f"    [warning] reply has no usable bid (attempt {retry+1}/3)")
            except (json.JSONDecodeError, Exception) as e:
                debug_log(f"{label}No-tool call raised (attempt {retry+1}/3)", f"{type(e).__name__}: {e}")
                print(f"    ⚠ JSON parse failed (attempt {retry+1}/3): {e}")
                last_problem, last_error = f"{type(e).__name__}: {e}", e
                continue

        # A bid must come from the model: after three failed attempts the run stops.
        raise RuntimeError(f"no usable bid from the model in 3 attempts; last problem: {last_problem}") from last_error

    def generate_bid(
        self,
        supplier_costs: SupplierCosts,
        regime_context: str,
        round_num: int,
        history: list[dict],
    ) -> SupplierBid:
        sid = supplier_costs.supplier_id
        debug_log(f"[{sid}] generate_bid ROUND {round_num}",
                  f"regime={regime_context[:80]}...\n"
                  f"history_len={len(history)}, use_tools={self.use_tools}\n"
                  f"cost_structure: c_base={supplier_costs.c_base}, "
                  f"gamma_del={supplier_costs.gamma_delivery}, "
                  f"gamma_qual={supplier_costs.gamma_quality}")

        if self.sealed_bid and history:
            raise ValueError("the bidder of a sealed-bid auction has no earlier rounds")
        description = (supplier_costs.describe_for_sealed_bid() if self.sealed_bid
                       else supplier_costs.describe_for_agent())
        user_msg = f"{description}\n\n"
        user_msg += f"REGIME: {regime_context}\n\n"
        user_msg += _previous_rounds_block(supplier_costs, history)

        # The round instruction, with the cost tool or without it. A request sent without tools
        # (the main path when use_tools is off, the fallback after a tool failure, and a below-cost
        # correction) carries the no-tool text, which never mentions the tool.
        if self.sealed_bid:
            tool_round_text, no_tool_round_text = SEALED_TOOL_ROUND_TEXT, SEALED_NO_TOOL_ROUND_TEXT
        else:
            tool_round_text = (
                f"ROUND {round_num}: You have a compute_my_cost tool. "
                f"IMPORTANT: You may call it AT MOST TWICE — pick your 1-2 "
                f"best configurations, check costs, then IMMEDIATELY submit "
                f"your bid as JSON. Do NOT explore more than 2 options. "
            )
            if history:  # the message has a PREVIOUS ROUNDS block that shows those costs
                tool_round_text += (
                    f"Your costs from previous rounds are shown above — do NOT "
                    f"re-check configurations you already know the cost for. "
                )
            tool_round_text += f"After your tool calls, output ONLY the JSON bid object."
            no_tool_round_text = (
                f"ROUND {round_num}: Submit your bid as JSON. "
                f"Keep reasoning brief (1-2 sentences). Focus on your cost "
                f"calculation and margin protection."
            )
        tool_msg = user_msg + tool_round_text
        no_tool_msg = user_msg + no_tool_round_text
        user_msg = tool_msg if self.use_tools else no_tool_msg

        # Build messages
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_msg},
        ]

        base_prompt = no_tool_msg  # every request sent without tools: the fallback and the corrections
        debug_log(f"[{sid}] Full user_msg for round {round_num}", user_msg)

        # Set when code, not the model, shaped the bid; they become tags on its reasoning.
        no_tool_fallback = False
        price_set_by_rule = False

        # ── Attempt bid with below-cost guardrail ──
        for attempt in range(self.MAX_RETRIES):
            if attempt > 0:
                # Correction: use simple no-tools call
                debug_log(f"[{sid}] Below-cost correction attempt {attempt+1}")
                bid_data = self._call_llm_no_tools(correction_prompt, self.model, sid=sid)
            elif self.use_tools:
                try:
                    bid_data = self._call_llm_with_tools(
                        list(messages), self.model, supplier_costs,
                    )
                except Exception as e:
                    debug_log(f"[{sid}] Tool path raised", f"{type(e).__name__}: {e}")
                    error_str = str(e)
                    if "400" in error_str:
                        # LM Studio transient 400 — retry with tools after a pause
                        retried = False
                        for retry in range(3):
                            wait = 5 * (retry + 1)
                            print(f"    ⏳ {supplier_costs.supplier_id} got 400 error, "
                                  f"retrying in {wait}s (attempt {retry+2}/4)...",
                                  flush=True)
                            time.sleep(wait)
                            try:
                                bid_data = self._call_llm_with_tools(
                                    list(messages), self.model, supplier_costs,
                                )
                                retried = True
                                break
                            except Exception as retry_error:
                                debug_log(f"[{sid}] Tool path raised again (retry {retry+1}/3)",
                                          f"{type(retry_error).__name__}: {retry_error}")
                                continue
                        if not retried:
                            print(f"    ⚠ {supplier_costs.supplier_id} 400 persisted "
                                  f"after retries — falling back to no-tools", flush=True)
                            debug_log(f"[{sid}] Falling back to no-tools")
                            no_tool_fallback = True
                            bid_data = self._call_llm_no_tools(base_prompt, self.model, sid=sid)
                    else:
                        print(f"    ⚠ Tool call failed for {supplier_costs.supplier_id}: {e} "
                              f"— falling back to no-tools", flush=True)
                        debug_log(f"[{sid}] Falling back to no-tools")
                        no_tool_fallback = True
                        bid_data = self._call_llm_no_tools(base_prompt, self.model, sid=sid)
            else:
                bid_data = self._call_llm_no_tools(base_prompt, self.model, sid=sid)

            debug_log(f"[{sid}] Raw bid_data from LLM", data=bid_data)
            # _is_bid has checked these values, so they convert.
            terms = ContractTerms(
                price=float(bid_data["price"]),
                delivery_weeks=float(bid_data["delivery_weeks"]),
                quality_tier=int(float(bid_data["quality_tier"])),
            )

            true_cost = supplier_costs.cost(terms)
            debug_log(f"[{sid}] Bid vs cost check",
                      f"price=${terms.price:.2f}, cost=${true_cost:.2f}, "
                      f"margin={((terms.price/true_cost)-1)*100:.1f}%, "
                      f"terms={terms.delivery_weeks}wk/T{terms.quality_tier}")

            if self.sealed_bid or terms.price >= true_cost:  # a sealed bid is taken as submitted
                break

            print(f"    ⚠ {supplier_costs.supplier_id} bid ${terms.price:.2f} "
                  f"below cost ${true_cost:.2f} — re-prompting", flush=True)
            correction_prompt = (
                base_prompt + f"\n\n"
                f"CORRECTION: Your previous bid of ${terms.price:.2f} is BELOW "
                f"your true cost of ${true_cost:.2f} for the terms you chose "
                f"({terms.delivery_weeks}wk, tier {terms.quality_tier}). If it won, this bid would lose money. "
                f"You MUST bid above ${true_cost:.2f}. Resubmit with a higher "
                f"price, or choose different terms that lower your cost."
            )
        else:
            if terms.price < true_cost:
                print(f"    ⚠ {supplier_costs.supplier_id} still below cost after "
                      f"retry — forcing price to cost + 5%", flush=True)
                terms = ContractTerms(
                    price=round(true_cost * 1.05, 2),
                    delivery_weeks=terms.delivery_weeks,
                    quality_tier=terms.quality_tier,
                )
                price_set_by_rule = True

        reasoning = f"[STATED]: {bid_data.get('reasoning', '')}"
        if price_set_by_rule:
            reasoning += RULE_PRICE_TAG
        if no_tool_fallback:
            reasoning += NO_TOOL_TAG
        return SupplierBid(
            supplier_id=supplier_costs.supplier_id,
            terms=terms,
            reasoning=reasoning,
            round_num=round_num,
        )


class PrimitivesSupplierAgent(LocalSupplierAgent):
    """The bidder of the primitives prompt set (orgsim/primitives_prompts.py), on any OpenAI-compatible chat endpoint.

    It has no cost tool, and what it sends is the context that the runner built and the institution completed
    (the event line, the costs, what the bidder knows of past events and the rules; see sequence.run_sequence),
    then the earlier rounds of the event with the platform's reply, then the round line. Nothing is added: no
    REGIME: label, no cost description of its own, no advice. A bid is taken as submitted: the below-cost correction
    and the rule that sets the price to cost plus 5% are not applied, because the primitives set has no such rule
    and none of its texts mentions one. `n_rounds` is the number of rounds of an event, which the round line names.

    The one exception is the bidder of a floor organization. Before an event the runner tells it the minimum profit
    per contract that its firm's analyst has set, or None (set_minimum_profit). While one is set, in every round, a
    bid whose price is below its total cost at its delivery and quality plus the minimum (strictly) is returned once,
    with a follow-up turn (orgsim.departures.floor_reask_text) after the bidder's own reply; if the bid sent back is
    also below the minimum, its price is set to its total cost, at the terms of the bid sent back, plus the minimum,
    and the bid's reasoning is tagged FLOOR_PRICE_TAG. The bid's floor_audit says what happened.
    """

    def __init__(self, client, model: str, system_prompt: str, n_rounds: int):
        super().__init__(client, model, system_prompt, use_tools=False)
        self.n_rounds = n_rounds
        self.minimum_profit = None

    def set_minimum_profit(self, minimum: float | None) -> None:
        """The minimum profit per contract of the coming event, in dollars, or None if the analyst set none."""
        if minimum is not None and not minimum >= 0:
            raise ValueError(f"the minimum profit is a non-negative number of dollars, not {minimum!r}")
        self.minimum_profit = minimum

    @staticmethod
    def _terms_of(bid_data: dict) -> ContractTerms:
        # _is_bid has checked these values, so they convert.
        return ContractTerms(
            price=float(bid_data["price"]),
            delivery_weeks=float(bid_data["delivery_weeks"]),
            quality_tier=int(float(bid_data["quality_tier"])),
        )

    def generate_bid(
        self,
        supplier_costs: SupplierCosts,
        regime_context: str,
        round_num: int,
        history: list[dict],
    ) -> SupplierBid:
        sid = supplier_costs.supplier_id
        if not 1 <= round_num <= self.n_rounds:
            raise ValueError(f"round {round_num} is not a round of an event of {self.n_rounds} rounds")
        user_msg = f"{regime_context}\n\n"
        user_msg += _previous_rounds_block(supplier_costs, history, REPLY_LABEL, primitives=True)
        user_msg += round_line(round_num, self.n_rounds)
        debug_log(f"[{sid}] generate_bid ROUND {round_num}", f"history_len={len(history)}")
        debug_log(f"[{sid}] Full user_msg for round {round_num}", user_msg)
        if self.minimum_profit is None:
            bid_data = self._call_llm_no_tools(user_msg, self.model, sid=sid)
            debug_log(f"[{sid}] Raw bid_data from LLM", data=bid_data)
            terms, audit, raised = self._terms_of(bid_data), None, False
        else:
            bid_data, terms, audit, raised = self._bid_above_the_minimum(user_msg, supplier_costs, sid)
        return SupplierBid(
            supplier_id=sid,
            terms=terms,
            reasoning=f"[STATED]: {bid_data.get('reasoning', '')}" + (FLOOR_PRICE_TAG if raised else ""),
            round_num=round_num,
            floor_audit=audit,
        )

    def _bid_above_the_minimum(self, user_msg: str, supplier_costs: SupplierCosts, sid: str):
        """The bid of a round in which the firm's analyst has set a minimum profit: the bidder's bid if its price is not
        below its total cost plus the minimum; otherwise the bid it sends back when it is returned once, with its price
        raised to that sum if it is still below. Returns the bid's data, its terms, the audit and whether the price was
        raised. The audit holds the minimum, whether the bid was returned (the first bid and the bid sent back, each
        with its terms, its total cost and its reasoning as received, and the follow-up turn) and whether the price
        was raised."""
        minimum = self.minimum_profit
        bid_data, reply = self._call_llm_no_tools(user_msg, self.model, sid=sid, with_reply=True)
        debug_log(f"[{sid}] Raw bid_data from LLM", data=bid_data)
        terms = self._terms_of(bid_data)
        cost = supplier_costs.cost(terms)
        audit = {"minimum_profit": minimum, "reasked": False, "raised": False}
        if not below_floor(terms.price, cost, minimum):
            return bid_data, terms, audit, False
        follow_up = floor_reask_text(terms.price, cost, minimum)
        print(f"    ⚠ {sid} bid ${dollars(terms.price)} is below its cost ${dollars(cost)} plus the minimum profit "
              f"${minimum:.2f} — asking again", flush=True)
        audit.update(reasked=True, first={"price": terms.price, "delivery_weeks": terms.delivery_weeks,
                                           "quality_tier": terms.quality_tier, "cost": cost,
                                           "reasoning": bid_data.get("reasoning", "")}, follow_up=follow_up)
        # Its own reply, as it wrote it, then the follow-up (a reply that holds only its thinking is its bid as JSON).
        turns = follow_up_turns(reply if reply.strip() else json.dumps(bid_data), follow_up)
        debug_log(f"[{sid}] Bid below the minimum profit, asking again", follow_up)
        bid_data, _ = self._call_llm_no_tools(user_msg, self.model, sid=sid, earlier_turns=turns, with_reply=True)
        debug_log(f"[{sid}] Raw bid_data from LLM (the bid sent back)", data=bid_data)
        terms = self._terms_of(bid_data)
        cost = supplier_costs.cost(terms)
        audit["resent"] = {"price": terms.price, "delivery_weeks": terms.delivery_weeks,
                           "quality_tier": terms.quality_tier, "cost": cost, "reasoning": bid_data.get("reasoning", "")}
        if not below_floor(terms.price, cost, minimum):
            return bid_data, terms, audit, False
        print(f"    ⚠ {sid} bid ${dollars(terms.price)} is still below its cost ${dollars(cost)} plus the minimum "
              f"profit ${minimum:.2f} — raising the price to ${dollars(cost + minimum)}", flush=True)
        terms = ContractTerms(price=cost + minimum, delivery_weeks=terms.delivery_weeks,
                              quality_tier=terms.quality_tier)
        audit["raised"] = True
        return bid_data, terms, audit, True


class LocalPlatformMediator(PlatformMediator):
    """Platform mediator using any OpenAI-compatible chat endpoint."""

    def __init__(self, client, model: str, system_prompt: str):
        self.client = client
        self.model = model
        self.system_prompt = system_prompt

    def generate_feedback(
        self,
        buyer_prefs: BuyerPreferences,
        all_bids: dict[str, ContractTerms],
        target_supplier_id: str,
        round_num: int,
        buyer_description: str = "",
        previous_bids: dict[str, list[ContractTerms]] | None = None,
        previous_feedback: dict[str, list[str]] | None = None,
    ) -> str:
        user_msg = f"{buyer_description or buyer_prefs.describe_for_platform()}\n\n"

        # Include memory context: every supplier's bids so far in this event, with the feedback
        # sent to that supplier (the prompt says the mediator knows each supplier's history).
        # In round 1 no reply exists yet and the history would only repeat the current bids.
        if previous_bids and previous_feedback and any(previous_feedback.values()):
            for sid in sorted(previous_bids):
                prev_bids = previous_bids[sid]
                prev_fb = previous_feedback.get(sid, [])
                if prev_bids:
                    marker = " (THIS SUPPLIER)" if sid == target_supplier_id else ""
                    user_msg += f"BID HISTORY FOR SUPPLIER {sid}{marker}:\n"
                    for i, prev_bid in enumerate(prev_bids):
                        round_label = i + 1
                        user_msg += (
                            f"  Round {round_label}: ${prev_bid.price:.2f}, "
                            f"{prev_bid.delivery_weeks}wk, tier {prev_bid.quality_tier}\n"
                        )
                        if i < len(prev_fb):
                            user_msg += f"  Your feedback (R{round_label}): {prev_fb[i]}\n"
                    user_msg += "\n"

        user_msg += "CURRENT BIDS FROM ALL SUPPLIERS:\n"
        for sid, bid in all_bids.items():
            label = "(THIS SUPPLIER)" if sid == target_supplier_id else ""
            user_msg += (
                f"  Supplier {sid} {label}: ${bid.price:.2f}, "
                f"{bid.delivery_weeks}wk, tier {bid.quality_tier}\n"
            )

        if round_num == 1:
            user_msg += (
                f"\nGenerate directional feedback for Supplier {target_supplier_id} "
                f"(round {round_num}). Focus on identifying their apparent cost "
                f"advantages and suggesting how to sharpen their competitive position. "
                f"2-4 sentences."
            )
        else:
            user_msg += (
                f"\nGenerate directional feedback for Supplier {target_supplier_id} "
                f"(round {round_num}). First acknowledge what changed since your "
                f"last feedback. If they followed your advice, build on it. If they "
                f"did NOT adjust a dimension you suggested, treat that as a signal "
                f"it may be too costly for them — pivot your advice accordingly. "
                f"2-4 sentences."
            )

        def _do_call():
            # Qwen3.5-27B non-thinking mode: temp=0.3 (low for mediator),
            # presence_penalty=1.5 per model card to prevent repetition.
            return self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.3,
                top_p=0.8,
                presence_penalty=1.5,
                extra_body={
                    "top_k": 20,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
        response = api_call_with_retry(_do_call)
        reply = response.choices[0].message.content
        content = THINK_BLOCK.sub('', reply or '').strip()
        if not content:
            # An unusable reply stops the run; the bidders are never sent an empty message.
            debug_log(f"[Mediator→{target_supplier_id}] EMPTY feedback (round {round_num})",
                      f"raw reply: {reply!r}")
            raise ValueError(
                f"the mediator model returned empty feedback for supplier {target_supplier_id}")
        debug_log(f"[Mediator→{target_supplier_id}] Feedback (round {round_num})",
                  f"bids: {', '.join(f'{sid}=${b.price:.2f}/{b.delivery_weeks}wk/T{b.quality_tier}' for sid, b in all_bids.items())}\n"
                  f"feedback:\n{content}")
        return content


class LocalBuyerAgentPlatform(BuyerAgentPlatform):
    """Buyer-agent platform using any OpenAI-compatible chat endpoint."""

    def __init__(self, client, model: str, system_prompt: str):
        self.client = client
        self.model = model
        self.system_prompt = system_prompt

    def generate_counteroffer_feedback(
        self,
        buyer_prefs: BuyerPreferences,
        supplier_bid: ContractTerms,
        supplier_id: str,
        round_num: int,
        history: list[dict],
        buyer_description: str = "",
    ) -> str:
        user_msg = f"{buyer_description or buyer_prefs.describe_for_platform()}\n\n"
        user_msg += (
            f"Supplier {supplier_id}'s current bid: ${supplier_bid.price:.2f}, "
            f"{supplier_bid.delivery_weeks}wk, tier {supplier_bid.quality_tier}\n\n"
        )

        if history:
            user_msg += "NEGOTIATION HISTORY (use this to identify flexibility):\n"
            for h in history:
                bid = h.get("bid")
                if bid:
                    user_msg += (
                        f"  Round {h.get('round', '?')}: ${bid.price:.2f}, "
                        f"{bid.delivery_weeks}wk, tier {bid.quality_tier}\n"
                    )
            user_msg += "\n"

        user_msg += f"Generate your adversarial negotiation message for round {round_num}. 2-3 sentences."

        def _do_call():
            # Qwen3.5-27B non-thinking mode: temp=0.4 (low for buyer agent),
            # presence_penalty=1.5 per model card to prevent repetition.
            return self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.4,
                top_p=0.8,
                presence_penalty=1.5,
                extra_body={
                    "top_k": 20,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
        response = api_call_with_retry(_do_call)
        reply = response.choices[0].message.content
        content = THINK_BLOCK.sub('', reply or '').strip()
        if not content:
            # An unusable reply stops the run; the bidders are never sent an empty message.
            debug_log(f"[BuyerAgent→{supplier_id}] EMPTY reply (round {round_num})",
                      f"raw reply: {reply!r}")
            raise ValueError(
                f"the buyer-agent model returned an empty reply for supplier {supplier_id}")
        debug_log(f"[BuyerAgent→{supplier_id}] Counteroffer (round {round_num})",
                  f"their bid: ${supplier_bid.price:.2f}/{supplier_bid.delivery_weeks}wk/T{supplier_bid.quality_tier}\n"
                  f"message:\n{content}")
        return content


# How LocalProcurementAnalyst._classify_posture reads the posture from a briefing.
_POSTURES = {"DEFENSIVE": AnalystBriefing.DEFENSIVE, "TRUSTING": AnalystBriefing.TRUSTING,
             "GUARDED": AnalystBriefing.GUARDED}
_LABEL = r"\b(" + "|".join(_POSTURES) + r")\b"
# "Posture: TRUSTING", "Recommended posture: GUARDED", "the posture is DEFENSIVE": the word "posture",
# at most three words, and a label, all on one line.
_STATED_POSTURE = re.compile(r"\bposture\b(?:[^\w\n]+\w+){0,3}?[^\w\n]+" + _LABEL, re.IGNORECASE)
# "I recommend GUARDED", "recommend a TRUSTING posture": "recommend" and the next label in its sentence.
_RECOMMENDED_POSTURE = re.compile(r"\brecommend(?:s|ed)?\b.*?" + _LABEL, re.IGNORECASE | re.DOTALL)
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")


class LocalProcurementAnalyst(ProcurementAnalyst):
    """Procurement analyst using any OpenAI-compatible chat endpoint."""

    def __init__(self, client, model: str, system_prompt: str):
        self.client = client
        self.model = model
        self.system_prompt = system_prompt

    @staticmethod
    def _classify_posture(briefing_text: str) -> str:
        """Read the posture from the free-form briefing text.

        The analyst is asked to recommend one of DEFENSIVE / TRUSTING / GUARDED but words the
        answer in many ways, and may name the other labels while it explains its choice. The text
        is read in this order, ignoring case:

        1. an explicit statement: the last "posture" that has a label after it on the same line
           with at most three words between them ("Posture: TRUSTING", "Recommended posture:
           GUARDED", "the posture is DEFENSIVE"); a label further away is not the statement;
        2. else the first "recommend", "recommends" or "recommended" with a label after it in
           the same sentence ("I recommend GUARDED", "recommend a TRUSTING posture");
        3. else a label named as a whole word anywhere in the text, so that "defensively" and
           "distrusting" name none: DEFENSIVE if it is named, then TRUSTING, otherwise GUARDED.

        This is a keyword rule, not a reading of the briefing. It does not understand a negation
        ("not DEFENSIVE"), so check the posture against the stored briefing before relying on it.
        No code acts on the posture; run_sequence records it in its results (analyst_postures).
        """
        stated = _STATED_POSTURE.findall(briefing_text)
        if stated:
            return _POSTURES[stated[-1].upper()]
        for sentence in _SENTENCE_BREAK.split(briefing_text):
            recommended = _RECOMMENDED_POSTURE.search(sentence)
            if recommended:
                return _POSTURES[recommended.group(1).upper()]
        for label in ("DEFENSIVE", "TRUSTING"):  # DEFENSIVE first
            if re.search(rf"\b{label}\b", briefing_text, re.IGNORECASE):
                return _POSTURES[label]
        return AnalystBriefing.GUARDED  # safe default

    def generate_briefing(
        self,
        supplier_id: str,
        history_text: str,
    ) -> AnalystBriefing:

        if not history_text:
            raise ValueError("the analyst is called only once the firm has a record of past events")

        user_prompt = (
            f"You are the procurement analyst for Supplier {supplier_id}.\n\n"
            f"Here is the interaction history with this platform:\n\n"
            f"{history_text}\n\n"
            f"Produce your strategic briefing."
        )

        briefing_text = self._ask(supplier_id, user_prompt)

        full_text = f"ANALYST BRIEFING — {supplier_id}: {briefing_text}"
        posture = self._classify_posture(briefing_text)

        return AnalystBriefing(posture=posture, text=full_text)

    def write_briefing(self, supplier_id: str, message: str, earlier_turns: tuple[dict, ...] = ()) -> AnalystBriefing:
        """The primitives prompt set: `message` is what the analyst reads (primitives_prompts.analyst_message). The
        reply is the briefing as written: no title is put before it and no posture is read from it (the posture
        is None). `earlier_turns` follow the message in the conversation (the analyst's first reply and the
        follow-up that asks again), so that the reply answers them."""
        if not message:
            raise ValueError("the analyst is sent a message")
        return AnalystBriefing(posture=None, text=self._ask(supplier_id, message, earlier_turns))

    def _ask(self, supplier_id: str, user_prompt: str, earlier_turns: tuple[dict, ...] = ()) -> str:
        """Send the analyst's message, and the turns that follow it if there are any, and return its reply without the
        thinking block and the white space around it."""

        def _do_call():
            # Qwen3.5-27B non-thinking mode: presence_penalty=1.5 per model
            # card to prevent repetitive reasoning loops.
            return self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                    *earlier_turns,
                ],
                temperature=0.3,
                top_p=0.8,
                presence_penalty=1.5,
                extra_body={
                    "top_k": 20,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )

        # A briefing must come from the model: a failed call raises after the retries in
        # api_call_with_retry, and an empty reply raises here (after the raw reply is logged).
        debug_log(f"[Analyst→{supplier_id}] Request (history sent)", user_prompt)
        response = api_call_with_retry(_do_call)
        reply = response.choices[0].message.content
        debug_log(f"[Analyst→{supplier_id}] Raw reply", reply or "(empty reply)")
        content = THINK_BLOCK.sub('', reply or "")
        if not content.strip():
            raise ValueError(f"the analyst model returned an empty briefing for {supplier_id}")
        return content.strip()
