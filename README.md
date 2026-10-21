# Organized firms in procurement

## Overview

This repository is the prototype that accompanies a Stage 1 proposal on how the internal organization of firms built from AI agents shapes their exchange with other firms. It holds the simulation code (`orgsim/`), the runner (`scripts/run_demo.py`), the tests, the recorded runs behind the proposal's results (`data/runs/`) and the scripts that compute the proposal's tables from those runs (`analysis/`).

**What the environment is.** Two supplier firms compete for a buyer's contract in repeated procurement events of several rounds, run by an AI intermediary: an AI mediator that coaches both suppliers, or an AI agent that negotiates for the buyer. The researcher sets the primitives: the buyer's values, the suppliers' costs and the rules of the market. Language-model agents produce the behavior, and the code scores every outcome against the efficient outcome. A supplier firm is one agent, with or without a record of past events, or two agents: an analyst that holds the record and a bidder that sets the bids. The analyst either briefs the bidder or forwards the record unchanged, and in some firms it may set a binding minimum profit per contract. The bidder pursues profit or revenue. Every message, bid and outcome is recorded. This environment has not been run against a deployed system (see `docs/ADAPTERS.md`).

**Install.** Python 3.11 or later (tested with 3.14.3). On Windows, clone into a short path such as `C:\src`: some run folders have long names, and Windows limits a path to 260 characters unless long paths are enabled (`git config --global core.longpaths true`).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Quick start with the mock backend (no model calls).**

```bash
python -m pytest
python scripts/run_demo.py --backend mock --events 2 --rounds 2
python scripts/run_demo.py --backend mock --prompts primitives --environment persistent \
    --mixed brief_revenue:forward_revenue --institutions buyer_agent --events 6 --rounds 3 --seeds 28
```

The mock backend uses deterministic stand-in agents. It checks that the pipeline runs and cannot show effects of organization, which need a model (see "Mock backend" under "Design"). Output goes to `outputs/demo`, which git ignores.

**A recorded run.** `examples/demo_run/` holds a recorded run, not one made by the mock backend. It is the buyer's-agent market of seed 28 in which both suppliers are briefed firms whose bidder seeks revenue (`--organizations brief_revenue`), run on Claude Haiku 5.5 with the code of this repository (commit 653e656). Its three files (`run.json`, `events.jsonl`, `summary.csv`) are as the runner wrote them, and `--out` in `run.json` is the path used when the run was recorded. Reading it needs no model call. In `events.jsonl`, `analyst_briefings` holds what each analyst sent its bidder before an event, `bids` holds each round's bid with its stated reasoning and the platform's reply, and `outcome` holds the winner, the surpluses and each supplier's markup over its true cost. The tests check that this code made the run (`source_sha256`) and that the run is complete.

**Reproducing the proposal's design and tables.** `REPRODUCE.md` gives, for each number in Table 1, Figure 2 and Section 3 of the proposal, the script, the data, the sample and the output line, and it gives the command lines that make the runs. The proposal's runs use the primitives-only prompts, the persistent environment, 6 events of 3 rounds and one model for every agent. The command-line default prompt set is `advice`, so `--prompts primitives` must be given. The recorded runs (about 81 MB) are in `data/runs/`. `python analysis/make_tables.py` recomputes every table from them in about half a minute, makes no model call, and writes the results to `analysis/results/`.

**Terms.** The proposal's terms and the names used in the code, the run folders and the analysis outputs:

| Proposal | Code |
|---|---|
| AI mediator; buyer's AI agent | `--institutions mediator`; `--institutions buyer_agent` |
| one agent with a record; without a record | `single` (U); `single_norecord` (U0) |
| noise: a second run of the same firm type | `single_b` (U'), beside `single_a` (U) |
| briefed firm: the analyst briefs the bidder | `brief` (B); `organized` (O) in the first runs |
| forwarding firm: the analyst forwards the record | `forward` (F); `organized_pass` (O_pass) in the first runs |
| minimum profit (formal authority) | `floor`: the analyst may set a binding minimum profit per contract |
| common objective: bidder and analyst both pursue profit | the plain name (`brief`, `forward`, `floor`); `team` in the analysis outputs |
| revenue-seeking bidder (conflicting objectives) | the `_revenue` variants (`brief_revenue`, `forward_revenue`, `floor_revenue`); `conflict` in the analysis outputs |
| a briefed firm against a forwarding firm | `--mixed brief:forward` and `--mixed forward:brief`; run folders end in `xa` (the first-named firm is supplier A) or `xb` |
| primitives-only prompts | `--prompts primitives` |
| events 2 to 6 (a record exists) | `LATE` in the analysis scripts; event 1 is the placebo |

The rest of this file is the reference documentation of the code.

## Reference

Prototype for a study of how the internal organization of firms shapes the exchange between them.

The primitives are set by the researcher: buyer values, supplier costs, contract terms and institutions. The efficient frontier is computed by exhaustive search over a grid of delivery times (0.1-week steps) and whole quality tiers, and behavior is generated by language-model agents. A supplier is a single agent, an organized firm, or an organized firm whose bidder also reads the analysis guidance. In an organized firm, an analyst keeps the firm's history and a separate agent bids. The two suppliers of a market have the same organization or, in a mixed market, different ones. The agents read either the earlier prompts, which carry strategic advice (`--prompts advice`, the default), or a set of texts that state primitives only and give no advice (`--prompts primitives`, see "The primitives prompt set"), in a random or a persistent environment (`--environment`).

## Contents

- `orgsim/primitives.py`: contract terms, buyer value, supplier cost, the efficient frontier, outcome metrics.
- `orgsim/institutions.py`: sealed first-price RFQ, second-price, and two AI institutions:
  - an AI mediator that coaches suppliers and passes only final bids to the buyer;
  - an AI buyer agent that negotiates for the buyer and presses suppliers on every term without revealing which terms the buyer values most.
- `orgsim/interfaces.py`: the agent interfaces (supplier, mediator, buyer agent) and the markers for bids shaped by code.
- `orgsim/organization.py`: the analyst's briefing type and what the bidder sees in each organization.
- `orgsim/prompts.py`: the advice set's system prompts, built from shared blocks. The bidder of the single firm and the bidder of the guided organized firm read the analysis guidance; the bidder of the organized firm does not, and its analyst does.
- `orgsim/primitives_prompts.py`: the primitives set's system prompts and the texts that go with each event (the event line, the round line, the analyst's message).
- `orgsim/llm.py`, `orgsim/llm_agents.py`: agents that call any OpenAI-compatible chat endpoint.
- `orgsim/sequence.py`: repeated events with persistent supplier memory.
- `orgsim/mocks.py`: deterministic stand-in agents, used by the tests and by the demo's mock backend.
- `scripts/run_demo.py`: single, organized and guided organized suppliers, alike or mixed in one market, under the mediator and the buyer agent, and the second-price auction run once for the structural benchmark; the primitives set, with its six organizations, in either environment.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest
python scripts/run_demo.py --backend mock --events 2 --rounds 2
```

To use an open-weight model served locally (tested with LM Studio; other OpenAI-compatible servers are untested):

```bash
python scripts/run_demo.py --backend openai --base-url http://localhost:1234/v1 --model <model-id> --events 6 --rounds 3 --seeds 1 2
```

For a hosted OpenAI-compatible endpoint, add `--hosted --base-url <endpoint>` and set `OPENAI_API_KEY` (or pass `--api-key`). For Claude models, see "Hosted Claude models" below.

Tested with Python 3.14.3; requires Python 3.11 or later. The runner prints progress with non-ASCII symbols; on Windows, when output is piped or redirected (including `pytest -s`), set `PYTHONUTF8=1` (`scripts/run_demo.py` switches its own output to UTF-8). `requirements-lock.txt` records the full tested environment: `pip install -r requirements-lock.txt` reproduces it exactly.

### Hosted Claude models

To run against Claude through Anthropic's OpenAI-compatible endpoint, put the key in `OPENAI_API_KEY` and use:

```bash
python scripts/run_demo.py --backend openai --hosted --default-sampling --no-tools \
    --base-url https://api.anthropic.com/v1/ --model <model-id> --max-tokens 16000 --max-retries 30 --timeout 300 \
    --events 2 --rounds 2
```

The second-price auction (see Design) takes the same options and one event for each seed; it has no rounds:

```bash
python scripts/run_demo.py --backend openai --hosted --default-sampling --no-tools \
    --base-url https://api.anthropic.com/v1/ --model <model-id> --max-tokens 16000 --max-retries 30 --timeout 300 \
    --institutions second_price --organizations single --events 1 --seeds 1 2 3 4 5 6 7 8
```

Markets whose two suppliers are organized differently (see Design) take `--mixed` in place of `--organizations`; run both orders of a pair:

```bash
python scripts/run_demo.py --backend openai --hosted --default-sampling --no-tools \
    --base-url https://api.anthropic.com/v1/ --model <model-id> --max-tokens 16000 --max-retries 30 --timeout 300 \
    --mixed single:organized organized:single --events 6 --rounds 3 --seeds 1 2 3 4
```

- `--default-sampling` sends no sampling settings, so the provider's defaults apply. Anthropic's OpenAI SDK
  compatibility page (read 2026-10-15) says that on Claude 4.7 and later models, which include the Claude 5 models,
  a temperature below 1, a `top_p` other than 0.99, or a request that sets both returns HTTP 400. Every agent
  request sets both.
- `--hosted` drops the local-server fields in `extra_body`. `--no-tools` leaves out the cost tool, because the
  bidder's prompt already holds the cost table.
- `--max-tokens N` caps each reply. On Claude 5 models thinking is on by default; on Claude Haiku 5.5 it is adaptive,
  so the model decides whether and how much to reason. Any reasoning is billed as output and counts toward the cap,
  but the endpoint does not return it. A reply cut at the cap can end empty or partial.
- `--max-retries N` sets the OpenAI SDK's own number of retries, with backoff, for connection errors, rate limits and
  server errors. The SDK's default is 2. With no `retry-after` header the SDK waits 0.5, 1, 2, 4, then 8 seconds
  before each retry (with jitter), so 30 retries keep one call going through about three minutes of overload errors
  (HTTP 529 or 5xx), where 8 retries last about 35 seconds; once they are used up, a 529 on a platform or analyst call
  ends the run, because the agents' own retry covers only HTTP 400, 429 and 503.
- `--timeout SECONDS` (1 to 86400) is how long the SDK waits on a request before it abandons it and retries it, as
  for a connection error (the SDK's default is 600), so set it above the time the longest reply takes. The number
  covers reading, writing and the connection pool; connecting keeps the SDK's limit of 5 seconds.
- `run.json` records `usage`, for the whole run and for each agent type that sent a request (`bidder_single`,
  `bidder_organized`, `bidder_organized_guided`, `bidder_sealed` (the second-price auction's bidders), `analyst`,
  `mediator`, `buyer_agent`): `requests` (every call made
  through the client, including calls that raised before or after reaching the server), `failed_requests` (the calls
  that raised; a failed call can still be billed, for example one interrupted after the server finished),
  `prompt_tokens` and `completion_tokens` as the endpoint reported them, `largest_completion_tokens` (the largest
  completion of any single request), `truncated` (replies cut at the cap, finish reason "length"; check that it is 0),
  `no_usage` (replies whose usage lacks a whole-number token count; any count that is there is still added, and a
  value above 0 means the token counts are low) and `finish_reasons` (replies by finish reason: "none" if a reply has
  none, "other" if it is not 1 to 40 lowercase letters and underscores). Two more counts are for the whole run only.
  Each HTTP attempt counts in `http_attempts`; one that receives a response also counts in `http_responses` by status
  code (`"200"`, `"529"`), so attempts minus responses are the attempts that got no response (they timed out, could
  not connect, or lost the connection). Such an attempt can still be billed if the server finished it. A call that the
  SDK retries counts once in `requests`. Anthropic does not document whether the endpoint's completion tokens include
  the hidden reasoning, so before pricing a run from these counts, compare a cheap smoke run's `usage` once with the
  provider's usage page (for example 2 events, 2 rounds and one seed, with the cap the full run will use). Prompts
  grow with past events, so a short run's tokens per request understate a longer run's.
- Prompt caching is not available on this endpoint.

## Design

- **Single firm (`single`).** One agent holds the costs, the history and the analysis guidance, and bids.
- **Organized firm (`organized`).** An analyst holds the history and the analysis guidance and sends a briefing before each event after the first. The bidder holds the costs and the briefing, and does not read the guidance.
- **Guided organized firm (`organized_guided`).** Organized in every respect: the analyst holds the history and sends the briefing, and the bidder holds the costs and the briefing and never sees the history. The bidder also reads the analysis guidance, the same text that the single firm's agent and the analysts read, under a note that says the guidance is written for the analyst and that the bidder is the CFO.

The single firm and the guided organized firm differ only in who holds the history and in the briefing step (and in the notes that describe them); this is the comparison the design intends. The organized firm and the guided organized firm differ only in whether the bidder reads the analysis guidance. At the first event no history exists and no briefing is sent, so a gap between organizations there comes from the bidder's instructions alone, apart from the model's sampling: the single bidder reads the guidance and the organized bidder does not.

- **Mixed markets (`--mixed`).** In a mixed market supplier A and supplier B have different organizations, such as `single:organized` (A is a single firm, B an organized firm), so an organized firm can face a single rival. Each supplier is treated by its own organization: its bidder reads the prompt of that organization, and its own record or its own analyst's briefing, and only an organized supplier has an analyst. Neither supplier nor platform is told the rival's organization, and nothing that a supplier or a platform is sent depends on it. A homogeneous market is the special case of a mixed one in which both suppliers have the same organization, and it sends what it sent before mixed markets were added. The seed fixes the contracts and the cost types, which do not depend on the organizations, so a pair and its reverse face the same plan. Ties go to A, so run both orders of a pair (`single:organized` and `organized:single`). `--mixed` takes the place of `--organizations`, and the second-price auction is not run in a mixed market (its auction is single only).
- **Second-price auction and the structural benchmark S (`--institutions second_price`).** Each of the two suppliers submits one sealed bid, the buyer scores it as value minus price, and the higher score wins. The winner keeps its delivery and quality and is paid the highest price at which its score is still at least the other bid's score, which is at least the price it bid. Under this rule, bidding a price equal to one's true cost at the terms one offers is optimal whatever the rival bids: the price paid does not depend on the winner's own price, and a bid at cost wins whenever winning pays a profit and loses whenever winning would be at a loss (a higher price risks losing a profitable win, a lower one risks winning at a loss). So the markup of a bid over its true cost (`outcome.all_inflation_rates` in `events.jsonl`) measures how far the agent is from S, for which it is 0. The bidder reads a prompt that states primitives only (`SEALED_BID_EXECUTIVE` in `orgsim/prompts.py`: its objective, profit, and what it does not know), because the AI institutions' bidder prompts cannot serve as the benchmark's bidder: they say the platform does not disclose how many suppliers compete (the sealed-bid rules say two), they are mostly about a platform's feedback, they call a contract won at a thin margin a failure in the board's eyes, which is advice and not the profit objective S assumes, and they require the price to be above cost, which these rules do not. Every other text this bidder reads (the cost description, the rules, the round instruction and the cost tool) states primitives or rules and gives no advice, and a bid below cost is taken as submitted. The auction is one-shot by design: its rules tell the bidder that this is a one-time event, which is true only when no record of earlier events exists. So the command needs `--events 1` and `--organizations single` (with one event there is no record for an analyst, and the organizations would not differ), and `--rounds` does not apply to it; for several auctions, give several `--seeds`. The bidder is not told how the buyer values delivery and quality, so S benchmarks its price relative to its cost and not its choice of terms.
- Before the first event every bidder in the AI institutions sees the same line (the second-price bidder has no record and sees none). Neither the bidders nor the analyst are told which contract is up; suppliers learn a past contract's name from their record.
- The bidder of an organized firm (`organized` or `organized_guided`) reads the briefing under a short title saying it comes from a member of the team working in the firm's interest, which states the organized firms' aligned objectives. The earlier simulation's title also called the memo confidential; that sentence is dropped because the single firm's bidder is told nothing comparable.
- Every supplier bids in every event. (A decline/probe rule from earlier work was dropped: it acted only through the analyst, so it confounded the comparison of single and organized firms.)
- **Walk-away.** Under the AI mediator and the AI buyer agent, the buyer awards no contract when every final bid is priced above its value for the terms, as their rules say; that event's surpluses are 0 and `awarded` is False in `summary.csv`. The sealed-bid institutions announce no walk-away and always award.
- **What a supplier learns after an event.** In the AI institutions: its own bids, the platform's feedback, and the result: it won (it was paid its bid, and its profit is stated), another supplier won, or no contract was awarded. Its record never shows rival bids, scores or the number of rivals, though a platform's reply can quote a rival's price (`n_rival_price_quotes` counts such replies). The second-price auction is one event and leaves no record.
- **What the platforms see.** The mediator sees every supplier's bids and its own feedback in the current event, the buyer's scoring function, and its memory of every supplier. The buyer agent sees only the current supplier's bids in the current event and an ordinal description of the buyer's priorities. Both are told not to say how many suppliers take part. The AI institutions need at least two rounds.
- **Environment rule.** In the AI institutions, a bid below cost is sent back once. If it is still below cost, it is raised to cost plus 5%. The second-price auction takes every bid as submitted. So does the primitives prompt set (`--prompts primitives`), which has no such rule, except that the bid of a `floor` organization is returned once and then raised when it is priced below its total cost plus the minimum profit that its analyst set (see "The primitives prompt set").
- **What the model must produce.** A model reply that cannot be used stops the run, and nothing scripted takes its place: a bidder gets up to five calls on the cost-tool path (the path is run up to three more times after an HTTP 400 error) and then three attempts without the tool (three attempts if the tool is off) to give a bid with a positive price, delivery of 1 to 8 weeks and a whole-number quality tier from 2 to 5 (the space the frontier is computed over), and an analyst call that still fails after transient errors are retried, or that returns an empty briefing, stops the run. An empty reply from either platform also stops the run. Two kinds of bid are shaped by code and marked in their reasoning: a price set by the rule above, and a bid made without the cost tool after the tool path failed (an error, or no usable bid in five calls). `summary.csv` counts both per event (`n_rule_priced_bids`, `n_tool_fallback_bids`). Reasoning that a server returns inside `<think>...</think>` is removed before a reply is read, including a block left unclosed. When a bidder's reply has nothing outside a closed block, the bid is looked for inside the block. A reply with a closing tag but no opening tag is not handled (the server used in earlier runs did not produce one).
- **Mock backend.** It checks that the pipeline runs. The mock supplier ignores what it is told about past events, so the rows of different organizations are identical by construction; organizational effects need a model backend. A mock run of one organization, such as the second-price auction, has no such rows to compare.
- **Same-type events.** When both suppliers draw the same cost type, they tie exactly on the frontier and the first-listed supplier counts as the efficient winner. So `correct_winner` is not meaningful in those events; filter on `same_type` (in `summary.csv`) or use `efficiency`.

## The primitives prompt set

`--prompts primitives` gives the agents of the mediator and the buyer agent texts that state primitives only: the agent's objective, what it knows and does not know, who decides, the payoffs, and the institution's rules. No sentence gives advice, and each is meant to be true of the code in both environments. The platforms' own prompts (`MEDIATOR_PROMPT` and `BUYER_AGENT_PROMPT`) are the advice set's, unchanged. `--prompts advice` is the default and sends what it sent before this set existed.

- **Where the texts are.** `orgsim/primitives_prompts.py` holds the system prompts: a shared block (the supplier's objective, the payoffs, the costs, the two contracts, and how the events are drawn, which differs between the environments), a role block for each organization, and the output format. The cost description is `SupplierCosts.describe_for_primitives`, and the rules are `primitives_rules` in `orgsim/institutions.py`, whose docstring lists the code that makes each statement true. What a bidder knows of past events is `build_primitives_bidder_context` in `orgsim/organization.py`. The record of past events that a bidder or an analyst reads is `build_primitives_record` in `orgsim/sequence.py`: it says "your company" in the third person, so that every sentence is true of the analyst as well as of the bidder, and it calls the platform's answer to a bid a reply, as the rules do; the advice set's record (`build_supplier_history`) is unchanged.
- **Organizations** (`--organizations`, or inside `--mixed`; the advice set's `organized_guided` is not among them, and these names are refused with `--prompts advice`):
  - `single` (U): the bidder holds the firm's record, which appears in its message from the second event on.
  - `single_norecord` (U0): the firm keeps no record, and the bidder is never shown a past event. The mediator's memory of the supplier does not depend on this.
  - `organized` (O): the analyst holds the record and writes the bidder a briefing before each event after the first. The bidder never sees the record.
  - `organized_pass` (O_pass): the record passes through the analyst unchanged. No analyst model is called; from the second event on the code puts the record, the very text that `single` reads under a title, into the bidder's message.
  - `single_revenue` (U_i): as `single`, and the bidder's objective is the firm's revenue, the total price of the contracts it wins.
  - `organized_revenue` (O_i): as `organized`; the bidder's objective is revenue and the analyst's is the firm's profit, and each is told the other's.
  With no `--organizations`, a run of the primitives set covers all six.
- **Departures organizations** (named in `--organizations` or inside `--mixed`, with `--prompts primitives` only; they are refused with `--prompts advice`, and a run that names no organization does not include them). In each, the firm has an analyst who holds the record and sends the bidder one message before each event from the second, under a heading that begins `MESSAGE FROM YOUR COMPANY'S PROCUREMENT ANALYST`. Each has a conflict variant with the suffix `_revenue` (`brief_revenue`, `forward_revenue`, `floor_revenue`, `brief_cap_revenue`, `brief_cost_revenue`): the bidder is judged on revenue and is told that the analyst's objective is the firm's profit (also in `forward_revenue`, where no model plays the analyst), and an analyst is told both objectives.
  - `brief` (B): the analyst writes a briefing, and the heading says so.
  - `forward` (F): no analyst model is called; the message is the full record, the very text that `single` reads, forwarded unchanged, and the heading says so.
  - `floor`: as `brief`, and the analyst may end its briefing with one line, `MINIMUM PROFIT PER CONTRACT: $X` or `MINIMUM PROFIT PER CONTRACT: none`, setting a minimum profit per contract for the event. The last line that is not only marks (a line such as `---`, `***`, `___` or a code fence is skipped) is read with some tolerance: the words and `none` may be in any case, white space and markdown marks around the line or between its parts do not matter (`*`, `_` and backticks at either end of the line, and `#`, `>`, `-`, `•`, `+`, `–`, `◦` and a list number such as `1.` or `1)` at its start), the words may have more than one white-space character between them, the dollar sign is optional, and the value may end with a period, also after closing marks; words after the value, `$1,000`, a negative value and a range are malformed. If that line is not a floor line, the analyst is asked once more (the follow-up begins `Write your briefing again.`), and if it is still not, no floor is set. The bidder reads the answer without any line that begins, once white space, marks and a list number at its start are removed, with the words `MINIMUM PROFIT PER CONTRACT` (in any case, with white space between them), whatever follows and wherever the line is; nothing else is removed, and a line with other words before the words stays. Then come a blank line and a notice: `MINIMUM PROFIT PER CONTRACT FOR THIS EVENT, SET BY YOUR COMPANY'S ANALYST: $X` (to two decimals) or `... : none`, or, when the analyst gave no valid line in two answers, `NO MINIMUM PROFIT PER CONTRACT IS SET FOR THIS EVENT`. The heading of the message says that its last line is a notice of the minimum (`MESSAGE FROM YOUR COMPANY'S PROCUREMENT ANALYST (written by the analyst, except the last line, a notice of the minimum):`). In every round of the event, a bid whose price is less than its total cost at its delivery and quality plus the minimum (strictly) is returned to the bidder once, with a follow-up turn that gives the bid's price and its total cost with the digits they have beyond the cent (up to nine decimals) and the minimum to the cent; the platform does not see the returned bid. If the bid it sends back is also below, its price is raised to its total cost, at the terms of that bid, plus the minimum, and the bid's reasoning is tagged. The bidder's role text names this rule from the first event. The record and the list of earlier rounds mark a raised price with `(price raised to its total cost plus the minimum profit your company's analyst set)`, and no note of a price set by rule is ever printed in the primitives set. `events.jsonl` records, for each supplier of `floor` from the second event on, under `departures` (a mapping from the supplier to `organization`, `raw_briefings` (the analyst's one or two answers as written), `reasked`, `follow_up`, `floor_line` (the last line of the last answer that is not only marks), `floor_status` (`set`, `none`, or `malformed` when the analyst gave no valid line in two answers), `minimum_profit` (a number, or `null`), `bids_returned` and `bids_raised`), and each bid's enforcement under `floor` in the bid (`minimum_profit`, `reasked`, `raised` and, when the bid was returned, `first`, `follow_up` and `resent`; `first` and `resent` hold `price`, `delivery_weeks`, `quality_tier`, `cost` and `reasoning`, the reasoning as the bidder wrote it); `summary.csv` counts the raised bids in `n_floor_raised_bids`, a column that every run naming a departures organization has.
  - `brief_cap`: as `brief`, with the briefing at most 50 words. An analyst whose briefing is longer is asked once more, in the same conversation, with a follow-up that says how long the briefing was; a second briefing that is still too long is cut to its first 50 words. The follow-up says so: if the second briefing still has more than 50 words, the bidder reads only its first 50 words. The heading and the analyst's role text state the limit. `events.jsonl` records, under `departures` and for each supplier of `brief_cap`, the original briefing, whether the analyst was asked again, the second briefing, whether it was cut and the final word count.
  - `brief_cost`: as `brief`, and the analyst is also shown the bidder's cost figures for the coming event, in the third person; from the second event on, the title of the bidder's cost structure says that its analyst is shown it.
  The bidders of `brief`, `forward`, `brief_cap` and `brief_cost` read one role text, so their first events are identical and differ only from the second event on. Their agents are counted in `run.json` under `bidder_<organization>` and `analyst_<organization>` (the bidder prompts of `brief` and `forward` are the same, so these labels do not come from the prompt).
- **Environment (`--environment`).** `random` is the plan of the earlier runs: each event's contract and cost types are drawn independently. `persistent` repeats event 1 of the random plan of the same seed, the contract and both cost types, at every event. The shared block says which one the run is in. `persistent` needs `--prompts primitives`, because the advice prompts say that a supplier's cost profile is drawn anew for each event. The set's texts state the probabilities of the draw (1 in 2 for the contract and for each supplier's cost type, 3 in 4 for the other supplier's type being the opposite); the command has no option to change them, and `pre_draw_event_sequence` and `run_sequence` refuse a plan drawn with another `p_different` under this set.
- **What else changes in a run.** A bid stands as submitted: there is no below-cost correction and no price set by rule (`n_rule_priced_bids` is 0). The one exception is the bidder of a `floor` organization, whose bid priced below its total cost plus the minimum profit that its analyst set is returned once and then raised (see `floor`). No posture is read from a briefing (`analyst_postures` is `null` for each briefing; for `organized_pass`, `analyst_briefings` holds the record that the bidder was sent). A model run needs `--no-tools`. The set takes at least 2 events, and neither `--max-history-events` (the texts say that the record is full and that the platform remembers past events) nor the second-price auction (its bidder reads a prompt of its own).
- **Usage labels** in `run.json`: `bidder_single`, `bidder_single_norecord`, `bidder_organized`, `bidder_organized_pass`, `bidder_single_revenue` and `bidder_organized_revenue` (one for each bidder prompt), `analyst` and `analyst_revenue` (the analyst of `organized_revenue`), `mediator` and `buyer_agent`. These labels name the primitives set's prompts when `--prompts primitives` is given. The agents of the departures organizations are counted under `bidder_<organization>` and `analyst_<organization>` (`analyst_brief`, `analyst_brief_cost`, ...), and nothing is counted under `other`.
- **Output fields.** `summary.csv` and `events.jsonl` hold `prompt_set` and `environment`, and `run.json` holds `prompts` and `environment`.

```bash
python scripts/run_demo.py --backend openai --hosted --default-sampling --no-tools \
    --base-url https://api.anthropic.com/v1/ --model <model-id> --max-tokens 16000 --max-retries 30 --timeout 300 \
    --prompts primitives --environment persistent --institutions buyer_agent \
    --mixed single_norecord:single single:single_norecord --events 6 --rounds 3 --seeds 3 7 9 12 19
```

## Outputs

`scripts/run_demo.py` writes three files to `--out` (default `outputs/demo`, relative to the current directory).
It first replaces `run.json` and deletes the `events.jsonl`, `summary.csv` and `run.json.tmp` that an earlier run left
there.

- `run.json`: how the run was made (the command with the `--api-key` value redacted and any credentials removed
  from `--base-url`, backend, model, endpoint, flags, institutions, organizations or mixed markets, events, rounds, seeds,
  the git commit and whether tracked files
  had uncommitted changes, `source_sha256` (a digest of `orgsim/*.py` and `scripts/run_demo.py` that identifies the code
  even after the history is rewritten; the recipe is in `scripts/run_demo.py`), Python and package versions,
  start and end times) and its status: `running`, `complete`, or `failed` with the error. With a model backend it
  also holds `usage`, written after each event and when the run ends, whether it completed or failed: `requests`,
  `failed_requests`, `prompt_tokens`, `completion_tokens`, `largest_completion_tokens`, `truncated`, `no_usage` and
  `finish_reasons`, in total and for each agent type that sent a request, and `http_attempts` and `http_responses`
  for the whole run (`null` for the mock backend; "Hosted Claude models" explains the fields). A run that is killed
  outright stays `running`; its `usage` is as of the last update that succeeded (an update cut off by the kill can
  leave a `run.json.tmp` beside it), so it leaves out the requests of the event in progress. A failed run writes no
  `summary.csv`; if an earlier run's `events.jsonl`, `summary.csv` or `run.json.tmp` cannot be deleted (for example,
  because another program has it open), the run stops at once, marked failed, and the files not yet deleted stay.
  Credentials of 8 or more characters are removed from `run.json` wherever they appear; shorter ones only where
  they stand as whole words. A query value in `--base-url` counts as a credential only if it has 8 or more
  characters, and encoded forms (such as base64) are not matched. Console output and the `--debug-log` file are not
  scrubbed.
- `events.jsonl`: one line per event, written as soon as the event ends, so a run that stops keeps every finished
  event. Each line holds the event's contract and cost types, the outcome (winner or none, surpluses, the
  efficient surplus and efficient winner, each supplier's true cost and markup), every round's bid with its stated
  reasoning and the platform's reply to it (null after the last round), and for the organized firms (`organized` and `organized_guided`; in the primitives set `organized`, `organized_revenue` and `organized_pass`) the analysts' briefings and postures. With a model analyst, a posture is read from the briefing's text (an
  explicit statement such as "Posture: GUARDED" or "I recommend GUARDED" if there is one, otherwise a keyword
  rule), so check it against the briefing before relying on it. The mock analyst writes the posture it chose into its briefing. In the primitives set no posture is read, so each posture is `null`; the briefing of `organized_pass` is the record that its bidder was sent.
  For `second_price`, `bids` holds each supplier's one bid as submitted (price and terms, no reply), `outcome.winning_bid`
  the price paid (the second-score price), `outcome.all_true_costs` each supplier's cost at the terms it bid, and
  `outcome.all_inflation_rates` the markup of its submitted price over that cost; the surpluses, here and in
  `summary.csv`, are at the price paid. `run.json` records `rounds` as `null` when only `second_price` runs.
  Every line of `events.jsonl` and every row of `summary.csv` holds `organization_a` and `organization_b`, the
  organization of each supplier, beside `organization` (the organization both suppliers have, or `<a>:<b>` in a mixed
  market). `run.json` records `mixed` (the pairs, or `null`), and `organizations` is `null` in a run of mixed markets.
  Every line and every row also holds `prompt_set` (`advice` or `primitives`) and `environment` (`random` or
  `persistent`), which `run.json` records as `prompts` and `environment`.
- `summary.csv`: one row per event, written when the run completes. Columns:
  `institution`, `organization` (the organization both suppliers have, or `<a>:<b>` in a mixed market),
  `organization_a`, `organization_b` (the organization of each supplier), `prompt_set`, `environment`, `seed`,
  `event`, `contract`, `type_a`, `type_b`, `same_type`;
  `winner` (empty if the buyer walked away), `awarded`, `correct_winner` (whether the winner is `efficient_winner`,
  the supplier whose best terms give the highest joint surplus, with ties going to A; False when nothing was
  awarded), `efficient_winner`;
  `efficiency` (joint surplus over the maximum on the frontier's grid, so it can slightly exceed 1 for an off-grid
  bid), `buyer_surplus`, `supplier_profit`, `joint_surplus`;
  `n_rule_priced_bids` and `n_tool_fallback_bids` (bids shaped by code, as above), and `n_rival_price_quotes`
  (an audit signal: platform replies in the event that contain a price the other supplier bid in that event;
  a coincidence can also match).

The seed fixes the event plan (contracts and cost types, shared by every institution and organization) and the
mock agents. It does not fix a model's sampling. `--debug-log PATH` logs the bidders' requests and replies, the platforms' replies, the analysts' histories and replies, and the errors of failed bidder attempts and retried calls. It is a debugging aid, not a full record: on the cost-tool path it cuts each message of a bid's first request to 500 characters (the bidder's system prompt is logged only there) and each reply to 2,000. `--max-history-events N` keeps only the last N past events in each
supplier's record and in the mediator's memory (0, the default, keeps all), which keeps prompts within a model's
context in long runs. With the mock
backend the rows of the organizations are identical by construction; when a run has more than one organization (or, with `--mixed`, more than one market) the script says so when it starts, and otherwise it says only that a mock run checks that the pipeline runs.

## Known differences beyond routing (to be tested next)

- On a local server that honours the thinking switch (LM Studio was used), the analyst call uses a lower
  temperature and no reasoning mode, and the bidder calls use the model's reasoning mode, except the calls made
  after the second tool call, which use no reasoning mode and a temperature of 0.7.
- With `--hosted`, the thinking switch and `top_k` are not sent, so whether a call reasons is left to the
  endpoint's default; the temperatures are still sent. With `--default-sampling`, no sampling settings are sent
  either, so every call uses the model's defaults. On Claude 5 models thinking is then on by default (adaptive on
  Claude Haiku 5.5: the model decides whether and how much to reason; the run sends no effort setting). Any
  reasoning is billed as output, counts toward `--max-tokens`, and is not returned.
- The AI institutions' prompts still carry strategic advice from earlier work. The block that holds the advice, `EXECUTIVE_BASE`, has the same text in every organization; the organizations differ in who reads the analysis guidance. Their statements of fact were corrected to match this environment (the provenance notes in `orgsim/prompts.py` list them). Only the second-price bidder's prompt states primitives only in this set; the primitives prompt set (`--prompts primitives`) gives the AI institutions such prompts.

## Connecting a deployed system

See `docs/ADAPTERS.md`.

## License

BSD-3-Clause.
