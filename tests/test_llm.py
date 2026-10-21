import json
import socket
from types import SimpleNamespace

import httpx
import pytest

from orgsim import llm
from orgsim.llm import ChatClient, make_client, sum_usage

LABELS = {"MEDIATOR PROMPT": "mediator", "BIDDER PROMPT": "bidder"}
KEY = "fake-key-for-tests-1"
URL = "http://example.test/v1"
# A request that sets every sampling field and every other field an agent can send.
REQUEST = dict(model="m", messages=[{"role": "system", "content": "S"}, {"role": "user", "content": "U"}],
               temperature=0.7, top_p=0.8, presence_penalty=1.5, frequency_penalty=0.2, top_k=20,
               tools=[{"type": "function"}], tool_choice="auto", max_tokens=50)


def counts(requests=0, failed=0, prompt=0, completion=0, largest=0, truncated=0, no_usage=0, reasons=None):
    """The counts of one label, as usage_summary() lists them."""
    return {"requests": requests, "failed_requests": failed, "prompt_tokens": prompt, "completion_tokens": completion,
            "largest_completion_tokens": largest, "truncated": truncated, "no_usage": no_usage,
            "finish_reasons": dict(reasons or {})}


def summary_of(attempts=0, http=None, by_agent=None, **kw):
    """The whole usage_summary(): the total counts, the HTTP attempts and responses, and the counts by label."""
    return {**counts(**kw), "http_attempts": attempts, "http_responses": dict(http or {}), "by_agent": by_agent or {}}


class Endpoint:
    """Stands in for an OpenAI client. It records every request and answers each in turn (the last
    answer repeats). An answer is a (prompt tokens, completion tokens) pair or a triple that adds the
    finish reason (otherwise "stop"); "none" is a response whose usage is None, "absent" a response
    with no usage attribute at all, "no finish reason" a choice without finish_reason, "no choices" a
    response without a choices attribute and "empty choices" one whose choices are an empty list; an
    object with usage fields is used as the usage; an exception (any BaseException, a KeyboardInterrupt
    included) is raised."""

    def __init__(self, *answers):
        self.calls, self.answers = [], list(answers) or [(1, 1)]
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        answer = self.answers[min(len(self.calls), len(self.answers)) - 1]
        if isinstance(answer, BaseException):
            raise answer
        message = SimpleNamespace(content="ok", tool_calls=None)
        choice = SimpleNamespace(message=message, finish_reason="stop")
        response = SimpleNamespace(choices=[choice], usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))
        if isinstance(answer, tuple):
            response.usage = SimpleNamespace(prompt_tokens=answer[0], completion_tokens=answer[1])
            choice.finish_reason = answer[2] if len(answer) > 2 else "stop"
        elif answer == "none":
            response.usage = None
        elif answer == "absent":
            del response.usage
        elif answer == "no finish reason":
            del choice.finish_reason
        elif answer == "no choices":
            del response.choices
        elif answer == "empty choices":
            response.choices = []
        else:
            response.usage = answer
        return response


def ask(client, system="S", **extra):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": "question"}]
    return client.chat.completions.create(model="m", messages=messages, **extra)


@pytest.fixture
def fake_openai(monkeypatch):
    """Replace openai.OpenAI (no client is built and no network is used). `.kwargs` lists the keyword
    arguments of each construction; the clients answer through `.endpoint`."""
    built = SimpleNamespace(kwargs=[], endpoint=Endpoint())

    class FakeOpenAI:
        def __init__(self, **kwargs):
            built.kwargs.append(kwargs)
            self.chat = built.endpoint.chat

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)
    return built


@pytest.fixture
def no_network(monkeypatch):
    """Any attempt to open a connection raises an error: the tests that build a real SDK client use a mock transport."""
    def refuse(*args, **kwargs):
        raise OSError("a test tried to use the network")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


OVERLOADED = {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}


class LostBody(httpx.SyncByteStream):
    """The body of a response whose headers came: reading it times out."""

    def __iter__(self):
        raise httpx.ReadTimeout("body read timed out")


def completion_body(content="ok", prompt=11, completion=7, finish="stop"):
    """A valid chat completion, as JSON."""
    return {"id": "chatcmpl-1", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}}


def client_over(monkeypatch, handler, **options):
    """A client from make_client (the real OpenAI SDK) whose HTTP goes to `handler` through
    httpx.MockTransport: no network. The SDK's pauses between retries are skipped."""
    import openai

    class MockedHttpx(openai.DefaultHttpxClient):
        def __init__(self, **kwargs):
            super().__init__(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr("openai.DefaultHttpxClient", MockedHttpx)
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    return make_client(URL, api_key=KEY, agent_labels=LABELS, **options)


# ---- default_sampling ----

def test_default_sampling_removes_the_five_fields_and_keeps_the_rest():
    base = Endpoint()
    ChatClient(base, default_sampling=True).chat.completions.create(**REQUEST)
    assert base.calls == [{k: REQUEST[k] for k in ("model", "messages", "tools", "tool_choice", "max_tokens")}]


def test_default_sampling_removes_top_k_from_extra_body_and_keeps_the_rest_of_it():
    extra = {"top_k": 20, "chat_template_kwargs": {"enable_thinking": True}}
    base = Endpoint()
    ChatClient(base, default_sampling=True).chat.completions.create(model="m", messages=[], extra_body=extra)
    assert base.calls[0]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}
    assert extra["top_k"] == 20                                           # the caller's own dict is not changed
    base = Endpoint()
    ChatClient(base, default_sampling=True).chat.completions.create(model="m", messages=[], extra_body={"top_k": 20})
    assert "extra_body" not in base.calls[0]                              # nothing is left of it: no empty dict is sent


def test_without_default_sampling_every_field_is_sent_as_it_is():
    base = Endpoint()
    ChatClient(base).chat.completions.create(**REQUEST, extra_body={"top_k": 20})
    assert base.calls == [{**REQUEST, "extra_body": {"top_k": 20}}]


# ---- max_tokens ----

def test_max_tokens_is_sent_on_every_request_and_replaces_the_agents_value():
    base = Endpoint()
    client = ChatClient(base, max_tokens=100)
    client.chat.completions.create(model="m", messages=[])
    client.chat.completions.create(model="m", messages=[], max_tokens=5000)
    assert [call["max_tokens"] for call in base.calls] == [100, 100]


def test_no_max_tokens_is_added_when_none_is_set():
    base = Endpoint()
    client = ChatClient(base)
    client.chat.completions.create(model="m", messages=[])
    client.chat.completions.create(model="m", messages=[], max_tokens=7)
    assert "max_tokens" not in base.calls[0] and base.calls[1]["max_tokens"] == 7


# ---- usage ----

def test_usage_adds_up_across_calls_and_by_agent_label():
    client = ChatClient(Endpoint((100, 10), (200, 20), (300, 30), (400, 40)), agent_labels=LABELS)
    for system in ("MEDIATOR PROMPT", "BIDDER PROMPT", "MEDIATOR PROMPT", "A PROMPT NOBODY NAMED"):
        ask(client, system)
    assert client.usage_summary() == summary_of(
        requests=4, prompt=1000, completion=100, largest=40, reasons={"stop": 4},
        by_agent={
            "bidder": counts(requests=1, prompt=200, completion=20, largest=20, reasons={"stop": 1}),
            "mediator": counts(requests=2, prompt=400, completion=40, largest=30, reasons={"stop": 2}),
            "other": counts(requests=1, prompt=400, completion=40, largest=40, reasons={"stop": 1}),
        })


def test_the_largest_completion_and_the_replies_cut_at_the_cap_are_recorded():
    client = ChatClient(Endpoint((100, 10, "stop"), (100, 250, "length"), (100, 90, "length"), (100, 5, "stop")),
                        agent_labels=LABELS)
    for system in ("MEDIATOR PROMPT", "BIDDER PROMPT", "BIDDER PROMPT", "MEDIATOR PROMPT"):
        ask(client, system)
    summary = client.usage_summary()
    assert (summary["largest_completion_tokens"], summary["truncated"]) == (250, 2)     # the largest of any request
    assert summary["by_agent"]["bidder"] == counts(requests=2, prompt=200, completion=340, largest=250, truncated=2,
                                                   reasons={"length": 2})
    assert summary["by_agent"]["mediator"] == counts(requests=2, prompt=200, completion=15, largest=10,
                                                     reasons={"stop": 2})


def test_the_usage_field_is_not_called_max_completion_tokens():
    # In the OpenAI API max_completion_tokens names the output cap, and run.json also has a top-level max_tokens.
    client = ChatClient(Endpoint())
    ask(client)
    summary = client.usage_summary()
    for part in (summary, summary["by_agent"]["other"]):
        assert "largest_completion_tokens" in part and "max_completion_tokens" not in part


@pytest.mark.parametrize("finish_reason, key", [("stop", "stop"), ("tool_calls", "tool_calls"),
                                                ("content_filter", "content_filter"), (None, "none")])
def test_only_a_finish_reason_of_length_counts_as_truncated(finish_reason, key):
    client = ChatClient(Endpoint((1, 1, finish_reason)))
    ask(client)
    summary = client.usage_summary()
    assert summary["truncated"] == 0 and summary["finish_reasons"] == {key: 1}


@pytest.mark.parametrize("reason", ["stop", "length", "tool_calls", "content_filter", "end_turn", "_", "x", "a" * 40])
def test_a_finish_reason_of_one_to_forty_lowercase_letters_and_underscores_is_kept(reason):
    client = ChatClient(Endpoint((1, 1, reason)))
    ask(client)
    assert client.usage_summary()["finish_reasons"] == {reason: 1}


@pytest.mark.parametrize("reason", [
    f"stop {KEY}", "\ud800", "stop\ud800", "Stop", "stop ", " stop", "stop\n", "", "a" * 41, "stop-now", "tool calls",
    "stop1", "sto\u0440", "\u00e9", 7, True, ["stop"], {"stop": 1}],
    ids=["a key", "a lone surrogate", "a word and a lone surrogate", "capital", "trailing space", "leading space",
         "trailing newline", "empty", "41 letters", "hyphen", "space inside", "digit", "Cyrillic letter",
         "accented letter", "number", "bool", "list", "dict"])
def test_any_other_finish_reason_counts_as_other_and_never_reaches_the_summary(reason):
    client = ChatClient(Endpoint((1, 1, reason)), agent_labels=LABELS)
    ask(client, "BIDDER PROMPT")
    summary = client.usage_summary()
    assert summary["finish_reasons"] == {"other": 1} and summary["by_agent"]["bidder"]["finish_reasons"] == {"other": 1}
    assert summary["truncated"] == 0
    assert KEY not in json.dumps(summary)
    json.dumps(summary, ensure_ascii=False).encode("utf-8")                       # it can be written as UTF-8 JSON


def test_finish_reasons_are_tallied_with_none_for_a_missing_one_and_truncated_is_the_length_count():
    client = ChatClient(Endpoint((1, 1, "stop"), (1, 1, "length"), (1, 1, "length"), "no finish reason", "no choices",
                                 (1, 1, None), (1, 1, "tool_calls")), agent_labels=LABELS)
    for system in ("BIDDER PROMPT", "BIDDER PROMPT", "MEDIATOR PROMPT", "BIDDER PROMPT", "MEDIATOR PROMPT",
                   "BIDDER PROMPT", "MEDIATOR PROMPT"):
        ask(client, system)
    summary = client.usage_summary()
    assert summary["finish_reasons"] == {"length": 2, "none": 3, "stop": 1, "tool_calls": 1}
    assert summary["by_agent"]["bidder"]["finish_reasons"] == {"length": 1, "none": 2, "stop": 1}
    assert summary["by_agent"]["mediator"]["finish_reasons"] == {"length": 1, "none": 1, "tool_calls": 1}
    for part in (summary, *summary["by_agent"].values()):
        assert part["truncated"] == part["finish_reasons"].get("length", 0)
    assert sum(summary["finish_reasons"].values()) == summary["requests"] - summary["failed_requests"]


@pytest.mark.parametrize("odd, no_usage, tokens, reasons", [
    ("none", 1, (7, 3), {"stop": 2}),                  # usage is None: nothing to add
    ("absent", 1, (7, 3), {"stop": 2}),                # no usage attribute
    ("no finish reason", 0, (8, 4), {"none": 1, "stop": 1}),
    ("no choices", 0, (8, 4), {"none": 1, "stop": 1}),
    ("empty choices", 0, (8, 4), {"none": 1, "stop": 1}),
], ids=["usage is None", "no usage attribute", "no finish reason", "no choices", "empty choices"])
def test_accounting_tolerates_a_response_with_a_missing_part_and_makes_the_gap_visible(odd, no_usage, tokens, reasons):
    client = ChatClient(Endpoint(odd, (7, 3)))
    ask(client)
    ask(client)
    summary = client.usage_summary()
    assert (summary["requests"], summary["failed_requests"], summary["truncated"]) == (2, 0, 0)
    assert (summary["prompt_tokens"], summary["completion_tokens"]) == tokens
    assert summary["no_usage"] == no_usage and summary["finish_reasons"] == reasons


def test_no_usage_counts_a_reply_with_any_token_count_missing_and_the_usable_counts_are_still_added():
    partial = [SimpleNamespace(prompt_tokens=5),                                            # no completion count
               SimpleNamespace(prompt_tokens=None, completion_tokens=4),                    # no prompt count
               SimpleNamespace(prompt_tokens="many", completion_tokens=True),               # neither is a whole number
               SimpleNamespace(prompt_tokens=2, completion_tokens=1)]                       # complete
    client = ChatClient(Endpoint(*partial), agent_labels=LABELS)
    for _ in partial:
        ask(client, "BIDDER PROMPT")
    summary = client.usage_summary()
    assert (summary["requests"], summary["prompt_tokens"], summary["completion_tokens"]) == (4, 7, 5)
    assert summary["largest_completion_tokens"] == 4
    assert summary["no_usage"] == 3 and summary["by_agent"]["bidder"]["no_usage"] == 3


def test_failed_requests_are_counted_separately_and_the_error_is_raised_again():
    client = ChatClient(Endpoint((10, 1), RuntimeError("boom"), (20, 2)), agent_labels=LABELS)
    ask(client, "BIDDER PROMPT")
    with pytest.raises(RuntimeError, match="boom"):
        ask(client, "BIDDER PROMPT")
    ask(client, "BIDDER PROMPT")
    summary = client.usage_summary()
    assert {k: summary[k] for k in counts()} == counts(requests=3, failed=1, prompt=30, completion=3, largest=2,
                                                      reasons={"stop": 2})
    assert summary["by_agent"]["bidder"]["failed_requests"] == 1


def test_a_keyboard_interrupt_from_the_endpoint_propagates_and_counts_as_a_failed_call():
    client = ChatClient(Endpoint((10, 1), KeyboardInterrupt(), (20, 2)), agent_labels=LABELS)
    ask(client, "BIDDER PROMPT")
    with pytest.raises(KeyboardInterrupt):
        ask(client, "BIDDER PROMPT")
    summary = client.usage_summary()
    assert (summary["requests"], summary["failed_requests"], summary["prompt_tokens"]) == (2, 1, 10)
    assert summary["by_agent"]["bidder"]["failed_requests"] == 1
    assert summary["finish_reasons"] == {"stop": 1}                      # the interrupted call has no reply to tally


def test_a_request_without_a_system_message_counts_under_other():
    client = ChatClient(Endpoint(), agent_labels=LABELS)
    client.chat.completions.create(model="m", messages=[{"role": "user", "content": "MEDIATOR PROMPT"}])
    client.chat.completions.create(model="m", messages=[])
    assert list(client.usage_summary()["by_agent"]) == ["other"]
    assert client.usage_summary()["by_agent"]["other"]["requests"] == 2


def test_a_new_client_has_no_usage_and_the_summary_is_a_copy():
    client = ChatClient(Endpoint())
    assert client.usage_summary() == summary_of()
    ask(client)
    summary = client.usage_summary()
    summary["requests"] = 99
    summary["by_agent"]["other"]["requests"] = 99
    summary["by_agent"]["other"]["finish_reasons"]["stop"] = 99
    summary["finish_reasons"]["stop"] = 99
    again = client.usage_summary()
    assert again["requests"] == 1 and again["by_agent"]["other"]["requests"] == 1
    assert again["finish_reasons"] == {"stop": 1} and again["by_agent"]["other"]["finish_reasons"] == {"stop": 1}


def test_sum_usage_adds_summaries_by_agent_and_keeps_the_largest_completion():
    one = ChatClient(Endpoint((1, 10, "length"), RuntimeError("boom")), agent_labels=LABELS)
    two = ChatClient(Endpoint((2, 20), (4, 7, "length"), "none"), agent_labels=LABELS)
    ask(one, "MEDIATOR PROMPT")
    with pytest.raises(RuntimeError):
        ask(one, "BIDDER PROMPT")
    ask(two, "MEDIATOR PROMPT")
    ask(two, "unnamed")
    ask(two, "unnamed")
    for _ in range(3):
        one.count_http_attempt(SimpleNamespace())
    two.count_http_attempt(SimpleNamespace())
    one.count_http_response(SimpleNamespace(status_code=200))
    one.count_http_response(SimpleNamespace(status_code=529))
    two.count_http_response(SimpleNamespace(status_code=200))
    assert sum_usage([one.usage_summary(), two.usage_summary()]) == summary_of(
        requests=5, failed=1, prompt=7, completion=37, largest=20, truncated=2, no_usage=1,
        reasons={"length": 2, "stop": 2},
        attempts=4, http={"200": 2, "529": 1},
        by_agent={
            "bidder": counts(requests=1, failed=1),
            "mediator": counts(requests=2, prompt=3, completion=30, largest=20, truncated=1,
                               reasons={"length": 1, "stop": 1}),
            "other": counts(requests=2, prompt=4, completion=7, largest=7, truncated=1, no_usage=1,
                            reasons={"length": 1, "stop": 1}),
        })
    assert sum_usage([]) == summary_of()


# ---- HTTP attempts and responses, counted through httpx hooks ----

def test_the_request_hook_counts_every_attempt_and_never_raises():
    client = ChatClient(Endpoint())
    for request in (SimpleNamespace(method="POST"), object(), None, "anything"):
        client.count_http_attempt(request)
    assert client.usage_summary()["http_attempts"] == 4


def test_the_response_hook_tallies_every_status_and_never_raises():
    client = ChatClient(Endpoint())
    for status in (200, 529, 200, 429):
        client.count_http_response(SimpleNamespace(status_code=status))
    client.count_http_response(object())                 # no status code: tallied as unknown, no exception
    client.count_http_response(None)
    client.count_http_response(SimpleNamespace(status_code="not a number"))
    client.count_http_response(SimpleNamespace(status_code=True))
    assert client.usage_summary()["http_responses"] == {"200": 2, "429": 1, "529": 1, "unknown": 4}


def test_http_attempts_and_responses_are_reported_for_the_whole_client_and_not_by_label():
    client = ChatClient(Endpoint(), agent_labels=LABELS)
    ask(client, "BIDDER PROMPT")
    client.count_http_attempt(SimpleNamespace())
    client.count_http_response(SimpleNamespace(status_code=200))
    summary = client.usage_summary()
    assert (summary["http_attempts"], summary["http_responses"]) == (1, {"200": 1})
    assert all("http_attempts" not in part and "http_responses" not in part for part in summary["by_agent"].values())


def test_http_responses_count_the_retries_the_sdk_makes_inside_one_call(monkeypatch, no_network):
    statuses = iter([529, 529, 200])

    def handler(request):
        status = next(statuses)
        if status == 200:
            return httpx.Response(200, json=completion_body())
        return httpx.Response(status, json={"type": "error", "error": {"type": "overloaded_error", "message": "busy"}})

    client = client_over(monkeypatch, handler, max_retries=3)
    assert ask(client, "BIDDER PROMPT").choices[0].message.content == "ok"
    summary = client.usage_summary()
    assert summary["http_responses"] == {"200": 1, "529": 2}                       # three attempts of one call
    assert summary["http_attempts"] == 3
    assert {k: summary[k] for k in ("requests", "failed_requests", "prompt_tokens", "completion_tokens")} == {
        "requests": 1, "failed_requests": 0, "prompt_tokens": 11, "completion_tokens": 7}


def test_attempts_that_time_out_or_cannot_connect_are_attempts_but_not_responses(monkeypatch, no_network):
    attempts = []

    def handler(request):
        attempts.append(request)
        number = len(attempts)
        if number == 1:
            raise httpx.ReadTimeout("timed out", request=request)             # the server may have finished it
        if number == 2:
            return httpx.Response(529, json=OVERLOADED)
        if number == 3:
            return httpx.Response(200, headers={"content-type": "application/json"}, stream=LostBody())
        if number == 4:
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(200, json=completion_body())

    client = client_over(monkeypatch, handler, max_retries=30)
    assert ask(client, "BIDDER PROMPT").choices[0].message.content == "ok"
    summary = client.usage_summary()
    assert len(attempts) == summary["http_attempts"] == 5
    assert summary["http_responses"] == {"200": 2, "529": 1}       # 2 attempts got no response: 5 - 3
    assert (summary["requests"], summary["failed_requests"]) == (1, 0)
    assert summary["prompt_tokens"] == 11                          # only the last reply is counted


def test_a_call_whose_retries_are_used_up_counts_once_as_failed_and_every_attempt_is_tallied(monkeypatch, no_network):
    import openai

    def handler(request):
        return httpx.Response(529, json={"type": "error", "error": {"type": "overloaded_error", "message": "busy"}})

    client = client_over(monkeypatch, handler, max_retries=2)
    with pytest.raises(openai.APIStatusError):
        ask(client, "BIDDER PROMPT")
    summary = client.usage_summary()
    assert (summary["requests"], summary["failed_requests"]) == (1, 1)
    assert summary["http_responses"] == {"529": 3}                                 # the first try and two retries
    assert summary["http_attempts"] == 3


def test_a_call_that_times_out_every_time_is_one_failed_request_with_attempts_and_no_responses(monkeypatch,
                                                                                               no_network):
    import openai

    def handler(request):
        raise httpx.ReadTimeout("timed out", request=request)

    client = client_over(monkeypatch, handler, max_retries=2, timeout=30)
    with pytest.raises(openai.APITimeoutError):
        ask(client, "BIDDER PROMPT")
    summary = client.usage_summary()
    assert (summary["requests"], summary["failed_requests"]) == (1, 1)
    assert (summary["http_attempts"], summary["http_responses"]) == (3, {})


@pytest.mark.parametrize("options, read", [({"timeout": 300}, 300), ({"timeout": 1.5}, 1.5),
                                          ({"timeout": 86400}, 86400), ({}, 600)],
                         ids=["300 seconds", "1.5 seconds", "86400 seconds", "not set: the SDK's default of 600"])
def test_the_timeout_reaches_reading_writing_and_the_pool_and_connecting_keeps_the_sdks_5_seconds(options, read,
                                                                                                 monkeypatch,
                                                                                                 no_network):
    seen = []

    def handler(request):
        seen.append(request.extensions["timeout"])
        return httpx.Response(200, json=completion_body())

    ask(client_over(monkeypatch, handler, **options), "BIDDER PROMPT")
    assert seen == [{"connect": 5, "read": read, "write": read if options else 600, "pool": read if options else 600}]


def test_the_connect_limit_is_the_openai_sdks_own():
    import openai
    assert llm.CONNECT_TIMEOUT == openai.DEFAULT_TIMEOUT.connect == 5.0         # the README and the epilog say 5 s


def test_make_client_installs_both_hooks_on_the_http_client_it_gives_the_sdk(fake_openai):
    client = make_client(URL, api_key=KEY)
    hooks = fake_openai.kwargs[0]["http_client"].event_hooks
    assert hooks["request"] == [client.count_http_attempt] and hooks["response"] == [client.count_http_response]


# ---- make_client ----

def test_make_client_passes_the_options_through(monkeypatch, fake_openai):
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    client = make_client("http://localhost:1234/v1", default_sampling=True, max_tokens=100, agent_labels=LABELS)
    ask(client, "BIDDER PROMPT", temperature=0.7, top_p=0.8)
    first = fake_openai.endpoint.calls[0]
    assert first.keys() == {"model", "messages", "max_tokens"} and first["max_tokens"] == 100
    assert client.usage_summary()["by_agent"]["bidder"]["requests"] == 1
    plain = make_client("http://localhost:1234/v1")
    ask(plain, temperature=0.7)
    second = fake_openai.endpoint.calls[1]
    assert second["temperature"] == 0.7 and "max_tokens" not in second


def test_max_retries_and_timeout_reach_the_openai_constructor_only_when_they_are_set(fake_openai):
    make_client(URL, api_key=KEY, max_retries=8, timeout=300)
    make_client(URL, api_key=KEY)
    make_client(URL, api_key=KEY, max_retries=0, timeout=1.5)
    first, second, third = fake_openai.kwargs
    assert (first["max_retries"], first["timeout"]) == (8, httpx.Timeout(300, connect=5.0))
    assert "max_retries" not in second and "timeout" not in second                 # the SDK's own defaults apply
    assert (third["max_retries"], third["timeout"]) == (0, httpx.Timeout(1.5, connect=5.0))   # 0 is a setting too


def test_max_retries_and_timeout_are_the_names_the_openai_sdk_takes(no_network):
    import openai
    set_both = make_client(URL, api_key=KEY, max_retries=8, timeout=300)
    assert set_both._base.max_retries == 8                                          # built, never called
    assert set_both._base.timeout == httpx.Timeout(300, connect=5.0)
    unset = make_client(URL, api_key=KEY)
    assert unset._base.max_retries == openai.DEFAULT_MAX_RETRIES == 2
    assert unset._base.timeout == openai.DEFAULT_TIMEOUT
