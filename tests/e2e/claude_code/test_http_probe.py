"""Harness coverage for the count_tokens / tool_search HTTP probes.

No `e2e` marker, so these run without a proxy. They pin the wire shape the probes
put on the wire (a regression here means the migrated pydantic bodies drifted from
what the old httpx probes sent), the Anthropic-native routing the shared `Gateway`
methods use, the rate-limiter token every probe must take, and the mapping from a
typed `Result` to the diagnostic string the compat matrix records. The file sits
directly under `claude_code/` (not a feature dir) so the compat-result hook skips
it instead of trying to record it as a matrix cell.

Everything is dependency-injected: a fake `Gateway`, a fake `Transport`, and a fake
rate limiter, so nothing reaches the process env, the network, or `monkeypatch`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from e2e_http import (
    AuthHeaders,
    NetworkError,
    RateLimitedError,
    Result,
    Success,
    UnauthorizedError,
    UnknownApiError,
    ValidationError,
)
from models import (
    AnthropicContentBlock,
    AnthropicMessagesBody,
    AnthropicMessagesResponse,
    ChatChoice,
    CountTokensBody,
    CountTokensResponse,
)
from e2e_gateway import Gateway
from pydantic import BaseModel

from claude_code.cli_driver import RATE_LIMIT_SHAPED_RE
from claude_code.http_probe import (
    assert_count_tokens_shape,
    assert_tool_search_shape,
    probe_count_tokens,
    probe_tool_search,
)
from claude_code.rate_limiter import infer_provider


def _dump(model: BaseModel) -> dict[str, object]:
    return model.model_dump(by_alias=True, exclude_none=True)


@dataclass
class _FakeLimiter:
    acquired: list[str] = field(default_factory=list)

    def acquire(self, provider: str) -> None:
        self.acquired.append(provider)


@dataclass
class _GatewayCall:
    key: str
    body: BaseModel


@dataclass
class _FakeGateway:
    """Captures the (key, body) each probe hands to the shared Gateway and
    returns a canned Result, so a probe test never needs a live proxy."""

    ct_result: Result[CountTokensResponse]
    msg_result: Result[AnthropicMessagesResponse]
    count_tokens_calls: list[_GatewayCall] = field(default_factory=list)
    messages_calls: list[_GatewayCall] = field(default_factory=list)

    def count_tokens(self, key: str, body: CountTokensBody) -> Result[CountTokensResponse]:
        self.count_tokens_calls.append(_GatewayCall(key=key, body=body))
        return self.ct_result

    def messages(self, key: str, body: AnthropicMessagesBody) -> Result[AnthropicMessagesResponse]:
        self.messages_calls.append(_GatewayCall(key=key, body=body))
        return self.msg_result


def _ok_count_tokens(n: int = 7) -> Result[CountTokensResponse]:
    return Success(data=CountTokensResponse(input_tokens=n))


def _ok_messages() -> Result[AnthropicMessagesResponse]:
    return Success(data=AnthropicMessagesResponse(content=[AnthropicContentBlock(type="text")]))


class TestProbeWireShape:
    """The probe bodies must serialize to exactly what the old httpx probes sent."""

    def test_count_tokens_body_and_key(self):
        gw = _FakeGateway(ct_result=_ok_count_tokens(), msg_result=_ok_messages())
        result = probe_count_tokens(
            gateway=gw,
            api_key="sk-master",
            model="claude-haiku-4-5",
            message="hello world",
            rate_limiter=_FakeLimiter(),
        )
        assert isinstance(result, Success)
        assert len(gw.count_tokens_calls) == 1
        call = gw.count_tokens_calls[0]
        assert call.key == "sk-master"
        assert _dump(call.body) == {
            "model": "claude-haiku-4-5",
            "messages": [{"role": "user", "content": "hello world"}],
        }

    def test_tool_search_body_and_key(self):
        gw = _FakeGateway(ct_result=_ok_count_tokens(), msg_result=_ok_messages())
        result = probe_tool_search(
            gateway=gw,
            api_key="sk-master",
            model="claude-opus-4-7",
            rate_limiter=_FakeLimiter(),
        )
        assert isinstance(result, Success)
        assert len(gw.messages_calls) == 1
        call = gw.messages_calls[0]
        assert call.key == "sk-master"
        body = _dump(call.body)
        assert body["model"] == "claude-opus-4-7"
        assert body["max_tokens"] == 64
        assert "stream" not in body
        assert body["tools"] == [
            {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"},
            {
                "name": "add_numbers",
                "description": "Add two integers",
                "input_schema": {
                    "type": "object",
                    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                    "required": ["a", "b"],
                },
            },
        ]

    def test_each_probe_takes_one_rate_limiter_token(self):
        gw = _FakeGateway(ct_result=_ok_count_tokens(), msg_result=_ok_messages())
        limiter = _FakeLimiter()
        probe_count_tokens(gateway=gw, api_key="k", model="claude-haiku-4-5-bedrock", rate_limiter=limiter)
        probe_tool_search(gateway=gw, api_key="k", model="claude-sonnet-4-5-vertex", rate_limiter=limiter)
        assert limiter.acquired == [
            infer_provider("claude-haiku-4-5-bedrock"),
            infer_provider("claude-sonnet-4-5-vertex"),
        ]


@dataclass
class _TransportCall:
    path: str
    headers: BaseModel
    json: BaseModel
    response_type: type[BaseModel]


@dataclass
class _FakeTransport:
    """Enough of the Transport surface for the two Anthropic-native Gateway
    methods: bearer() for the auth header, post() to capture the request."""

    canned: Result[BaseModel]
    posts: list[_TransportCall] = field(default_factory=list)

    def bearer(self, key: str) -> AuthHeaders:
        return AuthHeaders(authorization=f"Bearer {key}")

    def post(self, path, *, headers, json, response_type):
        self.posts.append(_TransportCall(path=path, headers=headers, json=json, response_type=response_type))
        return self.canned


class TestGatewayAnthropicRoutes:
    """Gateway.count_tokens / messages hit the native routes with the
    anthropic-version header and the right typed response."""

    def test_count_tokens_route(self):
        transport = _FakeTransport(canned=_ok_count_tokens())
        gateway = Gateway(transport=transport)  # pyright: ignore[reportArgumentType]  # fake transport double
        gateway.count_tokens("sk-x", CountTokensBody(model="m", messages=[]))
        assert len(transport.posts) == 1
        post = transport.posts[0]
        assert post.path == "/v1/messages/count_tokens"
        assert post.response_type is CountTokensResponse
        assert _dump(post.headers) == {"authorization": "Bearer sk-x", "anthropic-version": "2023-06-01"}

    def test_messages_route(self):
        transport = _FakeTransport(canned=_ok_messages())
        gateway = Gateway(transport=transport)  # pyright: ignore[reportArgumentType]  # fake transport double
        gateway.messages("sk-x", AnthropicMessagesBody(model="m", messages=[], max_tokens=8))
        assert len(transport.posts) == 1
        post = transport.posts[0]
        assert post.path == "/v1/messages"
        assert post.response_type is AnthropicMessagesResponse
        assert _dump(post.headers) == {"authorization": "Bearer sk-x", "anthropic-version": "2023-06-01"}


class TestAssertCountTokensShape:
    def test_positive_passes(self):
        assert assert_count_tokens_shape(_ok_count_tokens(3)) is None

    def test_zero_fails(self):
        err = assert_count_tokens_shape(Success(data=CountTokensResponse(input_tokens=0)))
        assert err is not None and "positive" in err

    def test_unknown_api_error_reports_status(self):
        err = assert_count_tokens_shape(UnknownApiError(status_code=400, body="bad model"))
        assert err is not None and "status 400" in err and "bad model" in err

    def test_validation_error_reports_body(self):
        err = assert_count_tokens_shape(ValidationError(message="input_tokens missing"))
        assert err is not None and "input_tokens missing" in err

    def test_unauthorized_and_network(self):
        assert "401" in (assert_count_tokens_shape(UnauthorizedError()) or "")
        assert "transport error" in (assert_count_tokens_shape(NetworkError(message="conn reset")) or "")

    def test_rate_limited_is_classifiable_by_compat_summary(self):
        # The compat conftest classifies a fail as rate-limited iff the error
        # text matches RATE_LIMIT_SHAPED_RE; the '429' must survive into it.
        err = assert_count_tokens_shape(RateLimitedError(body="Too Many Requests"))
        assert err is not None
        assert RATE_LIMIT_SHAPED_RE.search(err) is not None


class TestAssertToolSearchShape:
    def test_content_shape_passes(self):
        ok = Success(data=AnthropicMessagesResponse(content=[AnthropicContentBlock(type="text")]))
        assert assert_tool_search_shape(ok) is None

    def test_choices_shape_passes(self):
        ok = Success(data=AnthropicMessagesResponse(choices=[ChatChoice()]))
        assert assert_tool_search_shape(ok) is None

    def test_neither_content_nor_choices_fails_and_reports_keys(self):
        # extra="allow" keeps the unmodeled keys so the diagnostic can list what
        # the response actually contained (triage parity with the old probe).
        resp = AnthropicMessagesResponse.model_validate({"model": "m", "stop_reason": "end_turn"})
        err = assert_tool_search_shape(Success(data=resp))
        assert err is not None and "content" in err and "choices" in err
        assert "stop_reason" in err

    def test_upstream_400_reports_status(self):
        err = assert_tool_search_shape(UnknownApiError(status_code=400, body="tool type rejected"))
        assert err is not None and "status 400" in err

    def test_rate_limited_is_classifiable_by_compat_summary(self):
        err = assert_tool_search_shape(RateLimitedError(body="slow down"))
        assert err is not None
        assert RATE_LIMIT_SHAPED_RE.search(err) is not None
