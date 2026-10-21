"""One path for every model: OpenAI-compatible chat completions."""
import re
from types import SimpleNamespace

# The sampling settings the agents code (a local model's card sets them). default_sampling drops them.
SAMPLING_FIELDS = ("temperature", "top_p", "presence_penalty", "frequency_penalty", "top_k")
# What usage_summary() counts, in its order. Each is a sum over requests, except the largest completion
# (a maximum) and the finish reasons (a tally, {reason: requests}).
USAGE_FIELDS = ("requests", "failed_requests", "prompt_tokens", "completion_tokens", "largest_completion_tokens",
                "truncated", "no_usage", "finish_reasons")
_LARGEST = ("largest_completion_tokens",)
_TALLIES = ("finish_reasons",)
# A finish reason is recorded only if it looks like one ("stop", "length", "tool_calls"). Other text from a server
# could hold a credential or a lone surrogate (which UTF-8 cannot encode), and would reach run.json as a key.
_FINISH_REASON = re.compile(r"[a-z_]{1,40}")
# Seconds allowed for connecting when a timeout is set: the OpenAI SDK's own limit (openai.DEFAULT_TIMEOUT.connect).
CONNECT_TIMEOUT = 5.0


def _new_counts():
    """The counts of one label before any request."""
    return {name: {} if name in _TALLIES else 0 for name in USAGE_FIELDS}


def _tokens(usage, name):
    """A token count from a response's usage object: None if it is missing or not a whole number."""
    value = getattr(usage, name, None)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _finish_reason(response):
    """The finish reason of a response's first choice: "none" if it has none, "other" if it is not a string of
    1 to 40 lowercase letters and underscores."""
    try:
        reason = response.choices[0].finish_reason
    except (AttributeError, IndexError, TypeError):
        return "none"
    if reason is None:
        return "none"
    return reason if isinstance(reason, str) and _FINISH_REASON.fullmatch(reason) else "other"


def _add_tally(total, tally):
    """Add the counts of a {key: count} tally to another, in place."""
    for key, count in tally.items():
        total[key] = total.get(key, 0) + count


def _combine(counts_list):
    """Several dicts of USAGE_FIELDS added up (a copy, even of one dict); the largest completion is the
    largest of the largest, and the tallies are added key by key and listed in sorted order."""
    total = _new_counts()
    for counts in counts_list:
        for name in USAGE_FIELDS:
            if name in _TALLIES:
                _add_tally(total[name], counts[name])
            elif name in _LARGEST:
                total[name] = max(total[name], counts[name])
            else:
                total[name] += counts[name]
    for name in _TALLIES:
        total[name] = dict(sorted(total[name].items()))
    return total


def sum_usage(summaries):
    """The sum of several usage_summary() dicts (a demo run has one client for each cell)."""
    summaries = list(summaries)
    labels = sorted({label for summary in summaries for label in summary["by_agent"]})
    by_agent = {label: _combine(summary["by_agent"][label] for summary in summaries if label in summary["by_agent"])
                for label in labels}
    http_responses = {}
    for summary in summaries:
        _add_tally(http_responses, summary.get("http_responses") or {})
    return {**_combine(summaries), "http_attempts": sum(summary.get("http_attempts") or 0 for summary in summaries),
            "http_responses": dict(sorted(http_responses.items())), "by_agent": by_agent}


class ChatClient:
    """Wraps an OpenAI-compatible client.

    The agents send temperature, top_p and presence_penalty as standard fields, and top_k and a
    thinking switch in `extra_body`. Hosted endpoints may reject `extra_body`, so `local=False`
    drops it; the endpoint must accept the standard fields. The earlier runs used LM Studio.

    default_sampling: remove temperature, top_p, presence_penalty, frequency_penalty and top_k (also
        from `extra_body`) from every request, so that the provider's defaults apply. On Claude 5
        models a temperature below 1, a top_p other than 0.99, or both together return HTTP 400, and
        the agents set both.
    max_tokens: send this cap on every request, replacing any value the agent passed.
    agent_labels: {system prompt text: label}. Usage is counted per label; a request whose system
        prompt is not in the mapping counts under "other". Agents whose system prompts are the same cannot be
        told apart this way: labeled(label) gives such an agent a view of this client whose requests count
        under `label`, whatever their system prompt.

    usage_summary() returns the counts of the calls made through this client, in total and for each
    label:
      requests, failed_requests: every call made through the client, including calls that raised
        before or after reaching the server, and the calls that raised. A failed call can still be
        billed (for example, one interrupted after the server had finished).
      prompt_tokens, completion_tokens: as the responses report them.
      largest_completion_tokens: the largest completion of a single request.
      truncated: replies whose finish reason is "length" (cut at the cap).
      no_usage: replies whose usage lacks a whole-number prompt or completion count (the counts that
        are there are still added).
      finish_reasons: {finish reason: replies}; "none" for a reply without one, "other" for one that
        is not 1 to 40 lowercase letters and underscores (it is not recorded); truncated is
        finish_reasons["length"].
    Only for the client as a whole: http_attempts (an integer) and http_responses ({status code: HTTP
    responses}, keys are strings), counted by count_http_attempt and count_http_response, the hooks
    that make_client gives the SDK's HTTP client. Each HTTP attempt counts in http_attempts; one
    that receives a response also counts in http_responses by status code, so attempts minus
    responses are the attempts that timed out or could not connect. Such an attempt can still be
    billed if the server finished it. A call that the SDK retries counts once in requests.
    """

    def __init__(self, base_client, local: bool = True, *, default_sampling: bool = False,
                 max_tokens: int | None = None, agent_labels: dict[str, str] | None = None):
        self._base = base_client
        self._local = local
        self._default_sampling = default_sampling
        self._max_tokens = max_tokens
        self._agent_labels = dict(agent_labels or {})
        self._usage = {}
        self._http_attempts = 0
        self._http = {}
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _label(self, kwargs):
        """The label of the agent type that sends a request: that of its system prompt, else "other"."""
        for message in kwargs.get("messages") or []:
            if isinstance(message, dict) and message.get("role") == "system":
                content = message.get("content")
                return self._agent_labels.get(content, "other") if isinstance(content, str) else "other"
        return "other"

    def labeled(self, label: str) -> "LabeledClient":
        """A view of this client for one agent: its requests go through this client and are counted in it, under
        `label` in place of the label of their system prompt."""
        return LabeledClient(self, label)

    def _create(self, **kwargs):
        return self._create_as(None, kwargs)

    def _create_as(self, label, kwargs):
        """Send a request, counting it under `label`, or under the label of its system prompt when `label` is None."""
        if not self._local:
            kwargs.pop("extra_body", None)
        if self._default_sampling:
            for name in SAMPLING_FIELDS:
                kwargs.pop(name, None)
            extra = {k: v for k, v in (kwargs.get("extra_body") or {}).items() if k not in SAMPLING_FIELDS}
            if extra:
                kwargs["extra_body"] = extra
            else:
                kwargs.pop("extra_body", None)
        if self._max_tokens is not None:
            kwargs["max_tokens"] = self._max_tokens
        counts = self._usage.setdefault(label or self._label(kwargs), _new_counts())
        counts["requests"] += 1
        try:
            response = self._base.chat.completions.create(**kwargs)
        except BaseException:
            counts["failed_requests"] += 1
            raise
        usage = getattr(response, "usage", None)
        prompt, completion = _tokens(usage, "prompt_tokens"), _tokens(usage, "completion_tokens")
        if prompt is None or completion is None:
            counts["no_usage"] += 1
        counts["prompt_tokens"] += prompt or 0
        if completion is not None:
            counts["completion_tokens"] += completion
            counts["largest_completion_tokens"] = max(counts["largest_completion_tokens"], completion)
        reason = _finish_reason(response)
        _add_tally(counts["finish_reasons"], {reason: 1})
        if reason == "length":
            counts["truncated"] += 1
        return response

    def count_http_attempt(self, request):
        """Count an HTTP attempt (the request hook of the SDK's HTTP client, called before each attempt,
        retried ones included). It never raises."""
        self._http_attempts += 1

    def count_http_response(self, response):
        """Tally the status code of an HTTP response (the hook of the SDK's HTTP client, called for
        every response, retried attempts included). It never raises: a response without an integer
        status code is tallied as "unknown"."""
        try:
            status = getattr(response, "status_code", None)
            key = str(int(status)) if isinstance(status, int) and not isinstance(status, bool) else "unknown"
            _add_tally(self._http, {key: 1})
        except Exception:  # a hook that raised would turn a counting problem into a failed request
            pass

    def usage_summary(self):
        """The counts of the calls made so far, in total and by label (a copy; see the class)."""
        by_agent = {label: _combine([counts]) for label, counts in sorted(self._usage.items())}
        return {**_combine(by_agent.values()), "http_attempts": self._http_attempts,
                "http_responses": dict(sorted(self._http.items())), "by_agent": by_agent}


class LabeledClient:
    """What an agent holds in place of the ChatClient when its requests are counted under a label of their own
    (ChatClient.labeled). It has the `chat.completions.create` of a client and nothing else; the counts are those
    of `root`, the ChatClient, which is where usage_summary() is."""

    def __init__(self, root: ChatClient, label: str):
        self.root, self.label = root, label
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        return self.root._create_as(self.label, kwargs)


def make_client(base_url: str, api_key: str | None = None, local: bool = True, *, default_sampling: bool = False,
                max_tokens: int | None = None, max_retries: int | None = None, timeout: float | None = None,
                agent_labels: dict[str, str] | None = None) -> ChatClient:
    """Client for an OpenAI-compatible endpoint. The key defaults to $OPENAI_API_KEY; local
    servers do not need one. default_sampling, max_tokens and agent_labels are options of ChatClient.
    max_retries, if set, is the OpenAI SDK's own number of retries (with backoff) for connection
    errors, rate limits and server errors; timeout, if set, is the SDK's timeout in seconds for
    reading, writing and the connection pool, after which it abandons a request and retries it
    (connecting keeps the SDK's limit of CONNECT_TIMEOUT seconds). Otherwise the SDK's defaults apply.
    The SDK's HTTP client reports every attempt and every response to the ChatClient, which counts
    them (http_attempts, and http_responses by status code)."""
    import os
    import httpx
    import openai
    key = api_key or os.environ.get("OPENAI_API_KEY") or "not-needed"
    sdk_options = {}
    if max_retries is not None:
        sdk_options["max_retries"] = max_retries
    if timeout is not None:
        sdk_options["timeout"] = httpx.Timeout(timeout, connect=CONNECT_TIMEOUT)
    client = ChatClient(None, local=local, default_sampling=default_sampling, max_tokens=max_tokens,
                        agent_labels=agent_labels)
    http_client = openai.DefaultHttpxClient(event_hooks={"request": [client.count_http_attempt],
                                                         "response": [client.count_http_response]})
    client._base = openai.OpenAI(base_url=base_url, api_key=key, http_client=http_client, **sdk_options)
    return client
