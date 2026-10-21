import hashlib
import re

import pytest

from orgsim import prompts as P


def test_every_bidder_shares_base_and_output_format():
    for org in P.ORGANIZATIONS:
        prompt = P.executive_prompt(org)
        assert prompt.startswith(P.EXECUTIVE_BASE) and prompt.endswith(P.BID_OUTPUT), org


def test_analysis_guidance_reaches_the_analyst_and_the_single_and_guided_bidders_but_not_the_organized_bidder():
    assert P.executive_prompt("single").count(P.ANALYST_GUIDANCE) == 1
    assert P.executive_prompt("organized_guided").count(P.ANALYST_GUIDANCE) == 1      # the same text, unchanged
    assert P.ANALYST_GUIDANCE not in P.executive_prompt("organized")
    assert P.ANALYST_PROMPT.count(P.ANALYST_GUIDANCE) == 1


def test_routing_table_is_exactly_the_design():
    assert P.BIDDER_PARTS["single"] == (P.EXECUTIVE_BASE, P.SINGLE_NOTE, P.ANALYST_GUIDANCE, P.BID_OUTPUT)
    assert P.BIDDER_PARTS["organized"] == (P.EXECUTIVE_BASE, P.ORGANIZED_NOTE, P.BID_OUTPUT)
    assert P.BIDDER_PARTS["organized_guided"] == (P.EXECUTIVE_BASE, P.ORGANIZED_NOTE, P.GUIDED_NOTE,
                                                  P.ANALYST_GUIDANCE, P.BID_OUTPUT)
    assert P.ORGANIZATIONS == ("single", "organized", "organized_guided")     # the order of the command's default


def test_the_organizations_differ_only_where_the_design_says():
    single, organized, guided = (P.BIDDER_PARTS[org] for org in ("single", "organized", "organized_guided"))
    # organized_guided is the organized prompt with the guidance, and the note that introduces it, before the output
    assert guided == organized[:2] + (P.GUIDED_NOTE, P.ANALYST_GUIDANCE) + organized[2:]
    # and the single prompt with the notes about the record exchanged: the same base, guidance and output format
    assert (single[0], single[2:]) == (guided[0], guided[3:])
    assert single[1] == P.SINGLE_NOTE and guided[1:3] == (P.ORGANIZED_NOTE, P.GUIDED_NOTE)


def test_analyst_prompt_is_role_guidance_output():
    assert P.ANALYST_PROMPT == "\n\n".join([P.ANALYST_ROLE, P.ANALYST_GUIDANCE, P.ANALYST_OUTPUT])


def test_posture_specific_bidding_rules_are_not_in_the_base():
    # Paragraphs removed from the earlier bidder prompt; see the module docstring of orgsim/prompts.py.
    assert "Pay close attention to your analyst" not in P.EXECUTIVE_BASE
    assert "The difference matters enormously for how you bid" not in P.EXECUTIVE_BASE


def test_prompt_is_its_parts_in_order_with_one_joiner():
    joiners = set()
    for org, parts in P.BIDDER_PARTS.items():
        prompt = P.executive_prompt(org)
        gaps, pos = [], 0
        for part in parts:
            assert prompt.count(part) == 1, (org, part[:40])
            i = prompt.index(part, pos)          # ValueError if a part is out of order
            gaps.append(prompt[pos:i])
            pos = i + len(part)
        gaps.append(prompt[pos:])
        assert gaps[0] == "" and gaps[-1] == "", (org, "text outside the known parts")
        joiners |= set(gaps[1:-1])
    assert len(joiners) == 1, joiners             # the same joiner everywhere, nothing else between parts


ROLE_WORDS = re.compile(r"\b(analyst|briefing|memo)\b", re.I)


def test_text_that_both_bidders_read_names_no_analyst():
    for name in ("EXECUTIVE_BASE", "BID_OUTPUT", "ANALYST_GUIDANCE"):
        assert not ROLE_WORDS.search(getattr(P, name)), name


POLICY_WORDS = re.compile(r"trust|guarded|defens|skeptic|honest|adversar|margin|floor|concession"
                          r"|\b(follow|defer|heed|rely|weigh)\b", re.I)


def test_routing_notes_state_routing_only():
    for name in ("SINGLE_NOTE", "ORGANIZED_NOTE"):
        note = getattr(P, name)
        assert note.startswith("YOUR INFORMATION:"), name
        assert "\n" not in note, name
        hit = POLICY_WORDS.search(note)
        assert not hit, (name, hit and hit.group(0))


def test_the_guided_note_is_one_line_that_says_who_does_what_and_gives_no_advice():
    note = P.GUIDED_NOTE
    assert "\n" not in note and not POLICY_WORDS.search(note)
    sentences = note.split(". ")
    assert len(sentences) == 4
    assert sentences[0] == ("The guidance that follows is written for an analyst who briefs a CFO; in your firm your "
                            "procurement analyst works from it, and you are the CFO")
    assert sentences[1] == 'Where it addresses its reader as "you", it means your analyst'
    assert sentences[2] == ("You do not see the record, so the checks it describes are made by your analyst, from the "
                            "record")
    assert sentences[3] == ("Its instruction not to recommend prices or terms applies to your analyst: you decide the "
                            "bid's price and terms yourself.")
    assert note == (                                                                  # and nothing else in it
        'The guidance that follows is written for an analyst who briefs a CFO; in your firm your procurement analyst '
        'works from it, and you are the CFO. Where it addresses its reader as "you", it means your analyst. You do not '
        'see the record, so the checks it describes are made by your analyst, from the record. Its instruction not to '
        "recommend prices or terms applies to your analyst: you decide the bid's price and terms yourself.")
    # It parallels what the single firm's note says of the same guidance, and the two say who "you" and "the CFO" are.
    assert 'Where it says "the CFO", that is you.' in P.SINGLE_NOTE
    assert 'Where it addresses its reader as "you", it means your analyst.' in note
    # It says nothing of the guidance's other "you" (the supplier firm, the CFO of a sample briefing), and the sentence
    # that the provenance paragraph reads as about the firm is still in the guidance, unchanged.
    assert ("Do not assume you are the only bidder, but do not compress prices just because the platform demands it."
            in P.ANALYST_GUIDANCE)


def _long_lines(text):
    return [l.strip() for l in text.splitlines() if len(l.strip()) >= 20]


def test_text_does_not_cross_the_firm_boundary():
    organized = P.executive_prompt("organized")
    for line in _long_lines(P.ANALYST_GUIDANCE):
        assert line not in organized, line[:60]
    for line in _long_lines(P.EXECUTIVE_BASE):
        assert line not in P.ANALYST_PROMPT, line[:60]


def test_posture_labels_in_the_base_stay_in_one_paragraph():
    # Known asymmetry: the base uses the labels, the guidance defines them, and the organized
    # bidder meets the definitions only through the briefing.
    paras = [p for p in P.EXECUTIVE_BASE.split("\n\n") if re.search(r"\b(TRUSTING|GUARDED|DEFENSIVE)\b", p)]
    assert len(paras) == 1 and paras[0].startswith("INFORMATION STRATEGY")


EXPECTED_DIGESTS = {
    "EXECUTIVE_BASE": "bacc19c7e4dac296",
    "BID_OUTPUT": "e019bc0406d59195",
    "ANALYST_ROLE": "1d103a6ff7933ceb",
    "ANALYST_GUIDANCE": "fc089b80f70cfe46",
    "ANALYST_OUTPUT": "42533e38766fc5d7",
    "MEDIATOR_PROMPT": "134fec88661cd42d",
    "BUYER_AGENT_PROMPT": "35fec5954a5c3914",
    "SINGLE_NOTE": "aef92ff6a1362edd",
    "ORGANIZED_NOTE": "eac22283b55aba5d",
    "GUIDED_NOTE": "aff3be0b4ea4f2d4",
    "SEALED_BID_EXECUTIVE": "d6ac54f4fe997915",           # new text of its own: the second-price bidder's prompt
    "ANALYST_PROMPT": "4c9e3f96ad72ebe1",            # the analyst's prompt is the same for both organized firms
}
# The whole prompt of each bidder. The first two have the digests that they had before the third organization was
# added: the single and the organized prompt are unchanged by it.
EXPECTED_BIDDER_DIGESTS = {
    "single": "bc2f9f12d2278516",
    "organized": "5a21ffe52d242d5f",
    "organized_guided": "ad76771bd17a39a1",
}


def test_text_blocks_are_unchanged():
    for name, digest in EXPECTED_DIGESTS.items():
        got = hashlib.sha256(getattr(P, name).encode("utf-8")).hexdigest()[:16]
        assert got == digest, f"{name} changed; if intended, re-check bidder parity, then update the digest"


def test_bidder_prompts_are_unchanged():
    assert set(EXPECTED_BIDDER_DIGESTS) == set(P.ORGANIZATIONS)
    for org, digest in EXPECTED_BIDDER_DIGESTS.items():
        got = hashlib.sha256(P.executive_prompt(org).encode("utf-8")).hexdigest()[:16]
        assert got == digest, f"the {org} bidder's prompt changed; if intended, re-check bidder parity, then update"


def test_platform_prompts_keep_the_number_of_participants_hidden():
    assert "- NEVER reveal how many suppliers are participating." in P.MEDIATOR_PROMPT.splitlines()
    assert "The competition has improved and the gap is narrowing." in P.MEDIATOR_PROMPT
    assert "The competing offer has improved" not in P.MEDIATOR_PROMPT
    paragraphs = P.BUYER_AGENT_PROMPT.split("\n\n")
    i = next(k for k, p in enumerate(paragraphs) if p.startswith("STRATEGIC EXAGGERATION:"))
    assert paragraphs[i + 1] == (
        "WHAT YOU SEE: You see only this supplier's bids in this event, not the competing offers, and you "
        "are not told how many suppliers are participating. Never state a number of competitors.")
    assert P.BUYER_AGENT_PROMPT.count("WHAT YOU SEE:") == 1
    # its example of what to say names no plural number of rivals
    assert "your quality is insufficient compared to the competition." in P.BUYER_AGENT_PROMPT
    assert "compared to competing offers" not in P.BUYER_AGENT_PROMPT


def test_buyer_agent_prompt_does_not_claim_to_see_the_competing_offers():
    text = P.BUYER_AGENT_PROMPT
    assert "You may overstate how strong the competition is." in text
    assert 'say "we have a proposal that is better across the board" whatever the competing offers are.' in text
    for gone in ("how strong the competing offer is", "even if the gap is small", "the other offer is actually weaker"):
        assert gone not in text
    assert ("MANUFACTURED URGENCY: Imply the supplier is at risk of losing, whatever its actual position. "
            "Always maintain the impression of intense competition.") in text.split("\n\n")
    for gone in ("even when they're actually well ahead", "the actual gap between offers"):
        assert gone not in text                              # the agent sees neither the lead nor the other offers


def test_the_honest_platform_sentence_does_not_promise_a_win():
    assert "When they say you are well-positioned, you are." in P.EXECUTIVE_BASE
    assert "well-positioned, you win" not in P.EXECUTIVE_BASE


def test_module_docstring_describes_the_provenance_with_the_corrections():
    doc = " ".join(P.__doc__.split())
    assert ("Every block except SINGLE_NOTE, ORGANIZED_NOTE, GUIDED_NOTE and SEALED_BID_EXECUTIVE comes from the "
            "earlier simulation") in doc
    assert "SEALED_BID_EXECUTIVE is new text, for the bidder of the second-price auction" in doc
    assert "It is not one of the organizations: it shares only BID_OUTPUT with the bidder prompts above" in doc
    assert "the AI institutions' versions are unchanged." in doc
    assert "GUIDED_NOTE is new text, for the third organization." in doc
    assert ('that where the guidance addresses its reader as "you" it means the analyst (as SINGLE_NOTE says that '
            '"the CFO" is the bidder)') in doc
    assert ('The DEFENSIVE sentence "Do not assume you are the only bidder ..." is about the supplier firm, so it is '
            'true as the bidder, the CFO of that firm, reads it; ANALYST_GUIDANCE stays verbatim.') in doc
    assert ("the single and the organized firm differ there only in the bidder's instructions: the single bidder reads "
            "ANALYST_GUIDANCE and the organized bidder does not.") in doc
    assert ("In organized_guided the analyst holds the record and briefs the bidder from the second event, as in the "
            "organized firm, and the bidder also reads ANALYST_GUIDANCE, unchanged.") in doc
    assert ("tests/test_prompts.py pins a digest of each block and of each bidder prompt, the second-price "
            "bidder's included.") in doc
    assert "are original, apart from the factual corrections listed below." in doc
    assert ("MEDIATOR_PROMPT and BUYER_AGENT_PROMPT (platform side) are the earlier simulation's apart from "
            "the changes listed in the factual-corrections paragraph.") in doc
    assert "copied verbatim" not in doc and "(platform side) are unchanged" not in doc
    assert "In MEDIATOR_PROMPT:" in doc and "In BUYER_AGENT_PROMPT:" in doc   # both prompts' edits are listed
    assert "ANALYST_ROLE no longer calls the history complete." in doc
    assert "were removed for every firm (before the factual corrections listed below)." in doc
    assert 'and "you win" after an honest platform says you are well-positioned (now "you are")' in doc
    assert "the MANUFACTURED URGENCY paragraph's talk of the supplier being well ahead" in doc
    assert ("Text outside this module was corrected the same way: the cost description (no outside option), "
            "the buyer descriptions (no price range), the buyer agent's rules (\"in this event\"), the cost "
            "tool's labels, and the records (rule-set prices are marked in the record and within the event; "
            "the mediator's memory reports price increases).") in doc
    assert ("The corrections to the bidder's and the analyst's messages are listed in the module docstring "
            "of orgsim/llm_agents.py.") in doc
    assert "none of which the agent sees" in doc                          # the BUYER_AGENT_PROMPT entry
    assert "complete" not in P.ANALYST_ROLE                                 # the edit that sentence lists


def test_every_listed_organization_has_a_prompt():
    assert set(P.ORGANIZATIONS) == set(P.BIDDER_PARTS)
    for org in P.ORGANIZATIONS:
        assert P.executive_prompt(org)


@pytest.mark.parametrize("bad", ["hybrid", "", "Single", None])
def test_unknown_organization_is_rejected(bad):
    with pytest.raises(ValueError):
        P.executive_prompt(bad)


# ---- the bidder of the second-price auction ----

SEALED_TEXT = (
    "You decide the bid of a supplier company in a procurement auction for one contract. Your company's objective "
    "is profit: if you win, your profit is the price you are paid minus your true cost of fulfilling the contract "
    "at the terms you offered; if you lose, your profit is zero. Choose the bid that gives your company the "
    "highest expected profit.\n\n"
    "Your private costs are given below. You do not know the other supplier's costs.")
PLATFORM_WORDS = re.compile(r"\b(platforms?|facilitator|mediator|buyer agent|feedback|analyst|briefing|memo|CFO|board"
                            r"|history|record|past|events)\b", re.I)
ADVICE_WORDS = re.compile(r"\b(should|must|never|always|avoid|ensure|careful\w*|recommend\w*|strateg\w+|margin\w*"
                          r"|competit\w+|rival\w*|trust\w*|honest|generous|aggressive|conservative|floor)\b", re.I)


def test_the_sealed_bid_executive_is_exactly_the_text_that_was_asked_for():
    assert P.SEALED_BID_EXECUTIVE == SEALED_TEXT
    assert P.SEALED_BID_EXECUTIVE.count("\n") == 2                       # two paragraphs, one line each


def test_the_sealed_bid_prompt_is_that_text_and_the_output_format():
    assert P.sealed_bid_prompt() == P.SEALED_BID_EXECUTIVE + "\n\n" + P.BID_OUTPUT
    assert P.sealed_bid_prompt().endswith(P.BID_OUTPUT) and P.sealed_bid_prompt().count(P.BID_OUTPUT) == 1


def test_the_sealed_bid_prompt_is_not_an_organization():
    assert "sealed" not in P.BIDDER_PARTS and "sealed" not in P.ORGANIZATIONS
    assert P.ORGANIZATIONS == ("single", "organized", "organized_guided")            # the default of the command
    with pytest.raises(ValueError):
        P.executive_prompt("sealed")
    assert P.sealed_bid_prompt() not in {P.executive_prompt(org) for org in P.ORGANIZATIONS}


def test_the_sealed_bid_prompt_shares_only_the_output_format_with_the_bidders_of_the_ai_institutions():
    sealed = P.sealed_bid_prompt()
    for name in ("EXECUTIVE_BASE", "SINGLE_NOTE", "ORGANIZED_NOTE", "GUIDED_NOTE", "ANALYST_GUIDANCE"):
        assert getattr(P, name) not in sealed, name
        for line in _long_lines(getattr(P, name)):
            assert line not in sealed, (name, line[:50])
    for org in P.ORGANIZATIONS:
        assert P.SEALED_BID_EXECUTIVE not in P.executive_prompt(org), org
        assert not set(_long_lines(P.SEALED_BID_EXECUTIVE)) & set(_long_lines(P.executive_prompt(org))), org
    shared = set(_long_lines(sealed)) & set(_long_lines(P.executive_prompt("single")))
    assert shared == set(_long_lines(P.BID_OUTPUT))                      # BID_OUTPUT's own lines and nothing else


def test_the_sealed_bid_executive_names_no_platform_and_gives_no_advice():
    assert not PLATFORM_WORDS.search(P.SEALED_BID_EXECUTIVE), PLATFORM_WORDS.search(P.SEALED_BID_EXECUTIVE)
    assert not ADVICE_WORDS.search(P.SEALED_BID_EXECUTIVE), ADVICE_WORDS.search(P.SEALED_BID_EXECUTIVE)
    # and the words do catch what the AI institutions' bidder is told
    assert PLATFORM_WORDS.search(P.EXECUTIVE_BASE) and ADVICE_WORDS.search(P.EXECUTIVE_BASE)


EXPECTED_SEALED_PROMPT_DIGEST = "b90eb1cd20a056e5"


def test_the_sealed_bid_prompt_is_unchanged():
    got = hashlib.sha256(P.sealed_bid_prompt().encode("utf-8")).hexdigest()[:16]
    assert got == EXPECTED_SEALED_PROMPT_DIGEST, "the second-price bidder's prompt changed; if intended, re-audit it"
