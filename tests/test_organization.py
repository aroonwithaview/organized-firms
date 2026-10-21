import hashlib

import pytest

from orgsim import prompts as P
from orgsim.organization import (ANALYST_ORGANIZATIONS, FIRST_EVENT_NOTE, MEMO_HEADER, build_bidder_context,
                                 organization_of_each)


def test_single_bidder_sees_raw_history():
    assert build_bidder_context("single", "EVENT 1: won at 130", None) == "EVENT 1: won at 130"


def test_single_firm_has_no_briefing():
    with pytest.raises(ValueError):
        build_bidder_context("single", "history", "memo")


def test_the_organizations_with_an_analyst_are_all_but_the_single_firm():
    assert ANALYST_ORGANIZATIONS == ("organized", "organized_guided")
    assert set(ANALYST_ORGANIZATIONS) == set(P.ORGANIZATIONS) - {"single"}


@pytest.mark.parametrize("organization", ANALYST_ORGANIZATIONS)
def test_organized_bidder_sees_memo_not_history(organization):
    ctx = build_bidder_context(organization, "EVENT 1: won at 130", "Feedback was accurate. Posture: TRUSTING.")
    assert ctx == MEMO_HEADER + "Feedback was accurate. Posture: TRUSTING."


def test_the_organizations_with_an_analyst_are_routed_alike():
    contexts = {organization: build_bidder_context(organization, "EVENT 1: won at 130", "A memo.")
                for organization in ANALYST_ORGANIZATIONS}
    assert len(set(contexts.values())) == 1 and "EVENT 1" not in contexts["organized_guided"]


def test_first_event_note_is_identical_for_every_organization():
    for organization in P.ORGANIZATIONS:
        assert build_bidder_context(organization, "", None) == FIRST_EVENT_NOTE, organization


@pytest.mark.parametrize("organization", ANALYST_ORGANIZATIONS)
def test_organized_firm_with_history_needs_a_briefing(organization):
    with pytest.raises(ValueError, match="needs its analyst's briefing"):
        build_bidder_context(organization, "EVENT 1: won at 130", None)


def test_unknown_organization_is_rejected():
    with pytest.raises(ValueError):
        build_bidder_context("hybrid", "history", None)


@pytest.mark.parametrize("organization", ANALYST_ORGANIZATIONS)
def test_organized_firm_gets_no_briefing_before_its_first_event(organization):
    with pytest.raises(ValueError, match="no briefing before its first event"):
        build_bidder_context(organization, "", "memo")


def test_every_listed_organization_is_routed():
    for org in P.ORGANIZATIONS:
        assert build_bidder_context(org, "", None) == FIRST_EVENT_NOTE


def test_bidder_facing_lines_are_unchanged():
    # Pinned like the prompt blocks in tests/test_prompts.py: an edit here changes what bidders read.
    digests = {name: hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
               for name, text in (("MEMO_HEADER", MEMO_HEADER), ("FIRST_EVENT_NOTE", FIRST_EVENT_NOTE))}
    assert digests == {"MEMO_HEADER": "468f576e9aa98f66", "FIRST_EVENT_NOTE": "b98b73e2113b0a7b"}, "re-check bidder parity"


# ---- the organization of each supplier ----

def test_organization_of_each_gives_every_supplier_the_one_name_or_its_own():
    assert organization_of_each("single", ["A", "B"]) == {"A": "single", "B": "single"}
    given = {"B": "organized", "A": "single"}
    got = organization_of_each(given, ["A", "B"])
    assert got == {"A": "single", "B": "organized"} and list(got) == ["A", "B"]
    got["A"] = "organized_guided"
    assert given == {"B": "organized", "A": "single"}                       # a new dict: the caller's is not touched


@pytest.mark.parametrize("organization", [{"A": "single"}, {"A": "single", "B": "organized", "C": "single"},
                                          {"A": "single", "C": "organized"}, None, 3, ["single", "organized"]])
def test_organization_of_each_refuses_what_does_not_name_each_supplier(organization):
    with pytest.raises(ValueError):
        organization_of_each(organization, ["A", "B"])


def test_the_names_are_checked_where_a_bidder_context_is_built_not_in_organization_of_each():
    assert organization_of_each({"A": "hybrid", "B": "single"}, ["A", "B"])["A"] == "hybrid"
    with pytest.raises(ValueError, match="unknown organization: 'hybrid'"):
        build_bidder_context("hybrid", "history", None)
