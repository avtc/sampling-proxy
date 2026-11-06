"""Tests for per-upstream authentication and the aggregated /v1/models endpoint."""
import json
import httpx
import pytest
import sampling_proxy as sp


# ---------------------------------------------------------------- resolve_env_value
def test_resolve_env_value(monkeypatch):
    monkeypatch.setenv("TEST_ZAI_KEY", "sk-zai-123")
    assert sp.resolve_env_value("${TEST_ZAI_KEY}") == "sk-zai-123"
    assert sp.resolve_env_value("prefix-${TEST_ZAI_KEY}-suffix") == "prefix-sk-zai-123-suffix"
    monkeypatch.delenv("TEST_MISSING_KEY", raising=False)
    assert sp.resolve_env_value("${TEST_MISSING_KEY}") == ""
    assert sp.resolve_env_value("${TEST_MISSING_KEY:-fallback}") == "fallback"
    assert sp.resolve_env_value("plain-value") == "plain-value"


# ---------------------------------------------------------------- normalize_upstream_auth
def test_normalize_default_is_passthrough():
    assert sp.normalize_upstream_auth(None)["type"] == "passthrough"
    assert sp.normalize_upstream_auth({})["type"] == "passthrough"


def test_normalize_api_key_bearer_default(monkeypatch):
    monkeypatch.setenv("TEST_KEY", "abc")
    auth = sp.normalize_upstream_auth({"type": "api_key", "api_key": "${TEST_KEY}"})
    assert auth == {"type": "api_key", "api_key": "abc", "header": "authorization", "scheme": "Bearer"}


def test_normalize_api_key_token_alias_and_custom_header():
    auth = sp.normalize_upstream_auth({"type": "api_key", "token": "xyz", "header": "x-api-key"})
    assert auth["header"] == "x-api-key"
    assert auth["scheme"] == ""  # no scheme for non-authorization headers
    assert auth["api_key"] == "xyz"


def test_normalize_api_key_empty_falls_back_to_passthrough(monkeypatch):
    monkeypatch.delenv("TEST_NOPE_KEY", raising=False)
    assert sp.normalize_upstream_auth({"type": "api_key", "api_key": "${TEST_NOPE_KEY}"})["type"] == "passthrough"


def test_normalize_unknown_type_falls_back():
    assert sp.normalize_upstream_auth({"type": "weird"})["type"] == "passthrough"


# ---------------------------------------------------------------- apply_upstream_auth
def test_apply_passthrough_untouched():
    headers = {"authorization": "Bearer client", "x-api-key": "client-key"}
    sp.apply_upstream_auth(headers, {"auth": {"type": "passthrough"}})
    assert headers == {"authorization": "Bearer client", "x-api-key": "client-key"}


def test_apply_api_key_strips_and_injects_bearer():
    headers = {
        "authorization": "Bearer client-cred",
        "x-api-key": "client-key",
        "content-type": "application/json",
        "anthropic-version": "2023-06-01",
    }
    sp.apply_upstream_auth(
        headers,
        {"auth": {"type": "api_key", "api_key": "zai-secret", "header": "authorization", "scheme": "Bearer"}},
    )
    assert headers["authorization"] == "Bearer zai-secret"
    assert "x-api-key" not in headers
    assert headers["content-type"] == "application/json"
    assert headers["anthropic-version"] == "2023-06-01"


def test_apply_api_key_xapikey_header():
    headers = {"authorization": "Bearer client"}
    sp.apply_upstream_auth(
        headers,
        {"auth": {"type": "api_key", "api_key": "local-token", "header": "x-api-key", "scheme": ""}},
    )
    assert headers["x-api-key"] == "local-token"
    assert "authorization" not in headers


# ---------------------------------------------------------------- models payload helpers
def test_build_models_response_payload():
    models = [{"id": "a", "display_name": "a"}, {"id": "b", "display_name": "b"}]
    payload = sp.build_models_response_payload(models)
    assert payload["data"] == models
    assert payload["has_more"] is False
    assert payload["first_id"] == "a"
    assert payload["last_id"] == "b"
    empty = sp.build_models_response_payload([])
    assert empty["data"] == []
    assert empty["first_id"] is None
    assert empty["last_id"] is None


def test_extract_model_entries():
    openai_style = {"data": [{"id": "m1"}, {"id": "m2"}, "m3", {}, None]}
    entries = sp._extract_model_entries(openai_style)
    assert [e["id"] for e in entries] == ["m1", "m2", "m3"]

    anthropic_style = {"data": [{"type": "model", "id": "glm-5", "display_name": "GLM 5"}]}
    assert sp._extract_model_entries(anthropic_style)[0]["display_name"] == "GLM 5"

    llama_style = {"models": [{"id": "qwen"}]}
    assert sp._extract_model_entries(llama_style)[0]["id"] == "qwen"

    assert sp._extract_model_entries(None) == []
    assert sp._extract_model_entries("oops") == []


def test_models_path_for_upstream():
    # Anthropic-format upstreams always expose /v1/models (base_url may have a path prefix)
    assert sp._models_path_for_upstream({"supports_anthropic": True, "supports_openai": False, "base_path": "/api/anthropic"}) == "/v1/models"
    assert sp._models_path_for_upstream({"supports_anthropic": True, "supports_openai": True, "base_path": ""}) == "/v1/models"
    # OpenAI-format: append /models when base_url already has a path (e.g. .../v1), else /v1/models
    assert sp._models_path_for_upstream({"supports_anthropic": False, "supports_openai": True, "base_path": "/v1"}) == "/models"
    assert sp._models_path_for_upstream({"supports_anthropic": False, "supports_openai": True, "base_path": ""}) == "/v1/models"
    assert sp._models_path_for_upstream({"supports_anthropic": False, "supports_openai": False}) is None


def test_binding_model_entries_skips_globs():
    sp.MODEL_UPSTREAM_BINDING.clear()
    sp.MODEL_UPSTREAM_BINDING.update({"glm-5": "zai", "glm-*": "zai", "*": "local"})
    entries = sp._binding_model_entries()
    assert [e["id"] for e in entries] == ["glm-5"]


def test_resolve_anthropic_default_upstream_prefers_wildcard():
    sp.UPSTREAM_CONFIGS.clear()
    sp.UPSTREAM_CONFIGS.update({"zai": ZAI_UPSTREAM, "anthropic": ANTHROPIC_UPSTREAM})
    # Wildcard points to an Anthropic-capable upstream -> preferred over config order
    sp.MODEL_UPSTREAM_BINDING.clear()
    sp.MODEL_UPSTREAM_BINDING.update({"*": "anthropic"})
    assert sp.resolve_anthropic_default_upstream() == "anthropic"
    # Wildcard points to a non-Anthropic upstream -> ignored for Anthropic endpoints
    local_openai_only = {**LOCAL_UPSTREAM, "supports_anthropic": False}
    sp.UPSTREAM_CONFIGS["local"] = local_openai_only
    sp.MODEL_UPSTREAM_BINDING.update({"*": "local"})
    assert sp.resolve_anthropic_default_upstream() == "zai"  # first Anthropic-capable in config order
    # No wildcard at all
    sp.MODEL_UPSTREAM_BINDING.clear()
    assert sp.resolve_anthropic_default_upstream() == "zai"


# ---------------------------------------------------------------- test fixtures
class FakeResponse:
    def __init__(self, payload=None, status_code=200, headers=None):
        self._payload = payload
        self.status_code = status_code
        self.headers = headers or {"content-type": "application/json"}
        self.content = json.dumps(payload).encode() if payload is not None else b"{}"
        self.text = self.content.decode()

    def json(self):
        return self._payload

    async def aclose(self):
        pass


class FakeUpstreamClient:
    def __init__(self, payload=None, status_code=200, error_body=None, error_headers=None):
        self.payload = payload or {"data": []}
        self.status_code = status_code
        self.error_body = error_body
        self.error_headers = error_headers
        self.calls = []

    async def aclose(self):
        pass

    async def get(self, path, headers=None, timeout=None, params=None):
        self.calls.append({"path": path, "headers": dict(headers or {}), "params": params})
        if self.status_code != 200:
            return FakeResponse(status_code=self.status_code)
        return FakeResponse(self.payload)

    async def request(self, method, url, headers=None, params=None, content=None):
        self.calls.append({"method": method, "url": str(url), "headers": dict(headers or {}), "content": content})
        if self.status_code != 200:
            return FakeResponse(self.error_body, status_code=self.status_code, headers=self.error_headers)
        # Minimal valid Anthropic message response (passthrough mode)
        return FakeResponse({
            "id": "msg_1", "type": "message", "role": "assistant",
            "content": [{"type": "text", "text": "hi"}], "model": "glm-5",
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        })


ZAI_UPSTREAM = {
    "name": "zai", "base_url": "https://z.example/api/anthropic", "base_path": "/api/anthropic",
    "supports_anthropic": True, "supports_openai": False,
    "auth": {"type": "api_key", "api_key": "zai-secret", "header": "authorization", "scheme": "Bearer"},
}
LOCAL_UPSTREAM = {
    "name": "local", "base_url": "http://l.example/v1", "base_path": "/v1",
    "supports_anthropic": False, "supports_openai": True,
    "auth": {"type": "passthrough"},
}
ANTHROPIC_UPSTREAM = {
    "name": "anthropic", "base_url": "https://api.anthropic.com", "base_path": "",
    "supports_anthropic": True, "supports_openai": False,
    "auth": {"type": "passthrough"},
}


def test_binding_helpers_handle_alias_objects():
    # Plain string binding
    assert sp.binding_upstream_name("zai-anthropic") == "zai-anthropic"
    assert sp.binding_model_rewrite("zai-anthropic", "glm-5.3") == "glm-5.3"
    # Alias object binding: {"upstream": ..., "model": ...}
    alias = {"upstream": "zai-anthropic", "model": "glm-5.3"}
    assert sp.binding_upstream_name(alias) == "zai-anthropic"
    assert sp.binding_model_rewrite(alias, "anthropic/glm-5.3") == "glm-5.3"
    # Alias object without model rewrite keeps request model
    assert sp.binding_model_rewrite({"upstream": "zai"}, "whatever") == "whatever"


def test_resolve_upstream_for_model_with_alias_object(monkeypatch):
    sp.MODEL_UPSTREAM_BINDING.clear()
    sp.MODEL_UPSTREAM_BINDING.update({
        "anthropic/glm-5.3": {"upstream": "zai", "model": "glm-5.3"},
        "*": "anthropic",
    })
    assert sp.resolve_upstream_for_model("anthropic/glm-5.3") == "zai"
    assert sp.resolve_upstream_for_model("something-else") == "anthropic"


def setup_proxy(monkeypatch, upstreams, clients, bindings, models_cfg=None):
    """Reset module globals for an isolated test scenario."""
    sp.UPSTREAM_CONFIGS.clear()
    sp.UPSTREAM_CONFIGS.update(upstreams)
    sp.UPSTREAM_CLIENTS.clear()
    sp.UPSTREAM_CLIENTS.update(clients)
    sp.MODEL_UPSTREAM_BINDING.clear()
    sp.MODEL_UPSTREAM_BINDING.update(bindings)
    sp.MODELS_ENDPOINT_CONFIG.update(
        models_cfg
        or {"enabled": True, "cache_ttl_seconds": 0, "include_binding_models": True, "extra_models": []}
    )
    sp._MODELS_CACHE["models"] = []
    sp._MODELS_CACHE["fetched_at"] = 0.0
    sp.OVERRIDE_MODEL_NAME = None
    sp.VALIDATION_CONFIG = {"enabled": False}
    sp.PARALLEL_LIMITS = {}
    sp.MODEL_SEMAPHORES = {}
    sp.GLOBAL_LIMIT = None
    sp.GLOBAL_SEMAPHORE = None
    sp.throttle_manager = None
    sp.SAMPLING_PROXY_BASE_PATH = ""
    sp.FIRST_AVAILABLE_MODEL = "any"
    sp.ENABLE_OVERRIDE_LOGS = False
    sp.ENABLE_DEBUG_LOGS = False
    sp.DEFAULT_SAMPLING_PARAMS = {}
    sp.OVERRIDE_SAMPLING_PARAMS = {}
    sp.MODEL_SAMPLING_PARAMS = {}


# ---------------------------------------------------------------- aggregate_models
@pytest.mark.asyncio
async def test_aggregate_models_merges_and_dedupes(monkeypatch):
    zai_client = FakeUpstreamClient({"data": [{"id": "glm-5"}, {"id": "glm-4.7"}]})
    local_client = FakeUpstreamClient({"data": [{"id": "glm-5"}, {"id": "qwen-3"}]})
    setup_proxy(
        monkeypatch,
        upstreams={"zai": ZAI_UPSTREAM, "local": LOCAL_UPSTREAM},
        clients={"zai": zai_client, "local": local_client},
        bindings={"glm-5": "zai", "glm-*": "zai"},
        models_cfg={"enabled": True, "cache_ttl_seconds": 0, "include_binding_models": True, "extra_models": ["custom-model"]},
    )

    payload = await sp.aggregate_models({"authorization": "Bearer client-token", "x-api-key": "client-key"})

    ids = [m["id"] for m in payload["data"]]
    # Binding literals first, then discovered models (deduped), then extra models
    assert ids == ["glm-5", "glm-4.7", "qwen-3", "custom-model"]

    # ZAI upstream polled at /v1/models with its configured key, client creds stripped
    zai_call = zai_client.calls[0]
    assert zai_call["path"] == "/v1/models"
    assert zai_call["headers"]["authorization"] == "Bearer zai-secret"
    assert "x-api-key" not in zai_call["headers"]

    # Passthrough local upstream inherits the client's credentials
    local_call = local_client.calls[0]
    assert local_call["path"] == "/models"
    assert local_call["headers"]["authorization"] == "Bearer client-token"


@pytest.mark.asyncio
async def test_get_auth_headers_forwards_anthropic_beta_for_oauth(monkeypatch):
    """OAuth subscription tokens require anthropic-beta/anthropic-version headers on /v1/models polls."""
    incoming = {
        "authorization": "Bearer oauth-token",
        "anthropic-beta": "oauth-2025-04-20",
        "anthropic-version": "2023-06-01",
        "user-agent": "claude-cli/2.0 (external, cli)",
    }
    headers = sp.get_auth_headers(ANTHROPIC_UPSTREAM, incoming)
    assert headers["authorization"] == "Bearer oauth-token"
    assert headers["anthropic-beta"] == "oauth-2025-04-20"
    assert headers["anthropic-version"] == "2023-06-01"
    # api_key upstream: creds replaced, anthropic headers still forwarded
    headers2 = sp.get_auth_headers(ZAI_UPSTREAM, incoming)
    assert headers2["authorization"] == "Bearer zai-secret"
    assert headers2["anthropic-beta"] == "oauth-2025-04-20"


@pytest.mark.asyncio
async def test_aggregate_models_auth_failure_serves_bindings(monkeypatch):
    failing = FakeUpstreamClient(status_code=401)
    setup_proxy(
        monkeypatch,
        upstreams={"zai": ZAI_UPSTREAM},
        clients={"zai": failing},
        bindings={"glm-5.3": "zai"},
    )
    payload = await sp.aggregate_models()
    assert [m["id"] for m in payload["data"]] == ["glm-5.3"]


@pytest.mark.asyncio
async def test_aggregate_models_cache(monkeypatch):
    zai_client = FakeUpstreamClient({"data": [{"id": "glm-5"}]})
    setup_proxy(
        monkeypatch,
        upstreams={"zai": ZAI_UPSTREAM},
        clients={"zai": zai_client},
        bindings={},
        models_cfg={"enabled": True, "cache_ttl_seconds": 60, "include_binding_models": True, "extra_models": []},
    )
    payload1 = await sp.aggregate_models()
    assert len(zai_client.calls) == 1
    payload2 = await sp.aggregate_models()  # served from cache
    assert len(zai_client.calls) == 1
    assert payload1 == payload2


# ---------------------------------------------------------------- HTTP endpoint tests
def test_models_endpoint_aggregates(monkeypatch):
    from fastapi.testclient import TestClient

    zai_client = FakeUpstreamClient({"data": [{"id": "glm-4.7", "display_name": "GLM 4.7"}]})
    setup_proxy(
        monkeypatch,
        upstreams={"zai": ZAI_UPSTREAM},
        clients={"zai": zai_client},
        bindings={"glm-5.3": "zai", "glm-5.3-flash": "zai"},
    )
    # Skip startup model poll in lifespan
    sp.OVERRIDE_MODEL_NAME = "glm-5.3"

    with TestClient(sp.app) as client:
        # Lifespan recreated real httpx clients — swap fakes back in
        sp.UPSTREAM_CLIENTS.clear()
        sp.UPSTREAM_CLIENTS.update({"zai": zai_client})
        r = client.get("/v1/models", headers={"authorization": "Bearer anthropic-client-key"})
        assert r.status_code == 200
        data = r.json()["data"]
        ids = [m["id"] for m in data]
        assert ids == ["glm-5.3", "glm-5.3-flash", "glm-4.7"]
        assert data[0]["type"] == "model"
        assert data[2]["display_name"] == "GLM 4.7"
        assert r.json()["has_more"] is False


def test_messages_routes_by_binding_and_applies_auth(monkeypatch):
    from fastapi.testclient import TestClient

    zai_client = FakeUpstreamClient()
    anthropic_client = FakeUpstreamClient()
    setup_proxy(
        monkeypatch,
        upstreams={"zai": ZAI_UPSTREAM, "anthropic": ANTHROPIC_UPSTREAM},
        clients={"zai": zai_client, "anthropic": anthropic_client},
        bindings={"glm-5.3": "zai", "claude-*": "anthropic"},
    )
    sp.OVERRIDE_MODEL_NAME = "glm-5.3"  # skip startup poll

    body = {"model": "glm-5.3", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]}

    with TestClient(sp.app) as client:
        # Lifespan recreated real httpx clients — swap fakes back in
        sp.UPSTREAM_CLIENTS.clear()
        sp.UPSTREAM_CLIENTS.update({"zai": zai_client, "anthropic": anthropic_client})

        # glm-5.3 -> zai upstream: configured API key injected, client creds stripped
        r = client.post("/v1/messages", json=body, headers={"authorization": "Bearer client-cred", "x-api-key": "client-cred"})
        assert r.status_code == 200
        assert r.json()["content"][0]["text"] == "hi"
        sent = zai_client.calls[0]
        assert sent["headers"]["authorization"] == "Bearer zai-secret"
        assert "x-api-key" not in sent["headers"]

        # claude-sonnet-4-5 -> anthropic upstream: passthrough auth
        r2 = client.post(
            "/v1/messages",
            json={**body, "model": "claude-sonnet-4-5"},
            headers={"x-api-key": "sk-ant-real-key"},
        )
        assert r2.status_code == 200
        sent2 = anthropic_client.calls[0]
        assert sent2["headers"]["x-api-key"] == "sk-ant-real-key"


def test_unknown_model_routes_via_wildcard_binding(monkeypatch):
    """A '*' binding as last entry acts as the default upstream for unmatched models."""
    from fastapi.testclient import TestClient

    zai_client = FakeUpstreamClient()
    anthropic_client = FakeUpstreamClient()
    setup_proxy(
        monkeypatch,
        upstreams={"zai": ZAI_UPSTREAM, "anthropic": ANTHROPIC_UPSTREAM},
        clients={"zai": zai_client, "anthropic": anthropic_client},
        bindings={"glm-5.3": "zai", "*": "anthropic"},
    )

    with TestClient(sp.app) as client:
        sp.UPSTREAM_CLIENTS.clear()
        sp.UPSTREAM_CLIENTS.update({"zai": zai_client, "anthropic": anthropic_client})
        # Exact binding wins
        r = client.post("/v1/messages", json={"model": "glm-5.3", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 200
        assert len(zai_client.calls) == 1 and not anthropic_client.calls
        # Unknown model falls through to the wildcard -> anthropic, passthrough auth
        r2 = client.post(
            "/v1/messages",
            json={"model": "claude-sonnet-4-5-20250929", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]},
            headers={"authorization": "Bearer oauth-subscription-token"},
        )
        assert r2.status_code == 200
        assert len(anthropic_client.calls) == 1
        assert anthropic_client.calls[0]["headers"]["authorization"] == "Bearer oauth-subscription-token"


def test_unknown_model_without_wildcard_returns_400(monkeypatch):
    from fastapi.testclient import TestClient

    zai_client = FakeUpstreamClient()
    setup_proxy(
        monkeypatch,
        upstreams={"zai": ZAI_UPSTREAM},
        clients={"zai": zai_client},
        bindings={"glm-5.3": "zai"},
    )

    with TestClient(sp.app) as client:
        sp.UPSTREAM_CLIENTS.clear()
        sp.UPSTREAM_CLIENTS.update({"zai": zai_client})
        r = client.post("/v1/messages", json={"model": "claude-sonnet-4-5", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 400
        assert "No upstream binding" in r.json()["error"]["message"]


def test_requests_without_model_fall_back_to_wildcard(monkeypatch):
    """Model-less requests (count_tokens/event_logging edge cases) honor the '*' binding,
    not just the first Anthropic-capable upstream in config order."""
    from fastapi.testclient import TestClient

    zai_client = FakeUpstreamClient()
    anthropic_client = FakeUpstreamClient()
    setup_proxy(
        monkeypatch,
        upstreams={"zai": ZAI_UPSTREAM, "anthropic": ANTHROPIC_UPSTREAM},  # zai first in config order
        clients={"zai": zai_client, "anthropic": anthropic_client},
        bindings={"*": "anthropic"},
    )

    with TestClient(sp.app) as client:
        sp.UPSTREAM_CLIENTS.clear()
        sp.UPSTREAM_CLIENTS.update({"zai": zai_client, "anthropic": anthropic_client})
        # Generation request without a model field -> wildcard, not config-order-first zai
        r = client.post("/v1/messages", json={"max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 200
        assert len(anthropic_client.calls) == 1 and not zai_client.calls
        # Anthropic aux endpoint without a model field
        r2 = client.post("/api/event_logging/batch", json={"events": []})
        assert r2.status_code == 200
        assert len(anthropic_client.calls) == 2 and not zai_client.calls


def test_api_hello_probe_answered_locally(monkeypatch):
    """Claude Code's keyless connectivity probe (HEAD/GET /api/hello) must answer 200
    like api.anthropic.com does, without touching any upstream."""
    from fastapi.testclient import TestClient

    failing_client = FakeUpstreamClient(status_code=500)
    setup_proxy(
        monkeypatch,
        upstreams={"anthropic": ANTHROPIC_UPSTREAM},
        clients={"anthropic": failing_client},
        bindings={"*": "anthropic"},
    )

    with TestClient(sp.app) as client:
        sp.UPSTREAM_CLIENTS.clear()
        sp.UPSTREAM_CLIENTS.update({"anthropic": failing_client})
        r = client.head("/api/hello")
        assert r.status_code == 200
        r2 = client.get("/api/hello")
        assert r2.status_code == 200
        assert r2.json() == {"message": "hello"}
        # Probe never reached any upstream
        assert not failing_client.calls


def test_wildcard_ignored_when_format_incompatible(monkeypatch):
    """A '*' binding to an Anthropic-only upstream must not 400 OpenAI-format requests;
    they fall back to an OpenAI-capable upstream instead."""
    from fastapi.testclient import TestClient

    anthropic_client = FakeUpstreamClient()
    local_client = FakeUpstreamClient()
    setup_proxy(
        monkeypatch,
        upstreams={"anthropic": ANTHROPIC_UPSTREAM, "local": LOCAL_UPSTREAM},
        clients={"anthropic": anthropic_client, "local": local_client},
        bindings={"*": "anthropic"},  # wildcard is Anthropic-only
    )

    with TestClient(sp.app) as client:
        sp.UPSTREAM_CLIENTS.clear()
        sp.UPSTREAM_CLIENTS.update({"anthropic": anthropic_client, "local": local_client})
        # OpenAI-format generation request without a model: must route to local, not 400
        r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 200
        assert len(local_client.calls) == 1 and not anthropic_client.calls


def test_alias_binding_routes_and_rewrites_model(monkeypatch):
    """Alias IDs containing 'claude'/'anthropic' pass Claude Code's discovery filter;
    the proxy routes them to the bound upstream and rewrites the model for the upstream."""
    from fastapi.testclient import TestClient

    zai_client = FakeUpstreamClient()
    setup_proxy(
        monkeypatch,
        upstreams={"zai": ZAI_UPSTREAM},
        clients={"zai": zai_client},
        bindings={
            "anthropic/glm-5.3": {"upstream": "zai", "model": "glm-5.3"},
            "*": "zai",
        },
    )

    with TestClient(sp.app) as client:
        sp.UPSTREAM_CLIENTS.clear()
        sp.UPSTREAM_CLIENTS.update({"zai": zai_client})
        r = client.post(
            "/v1/messages",
            json={"model": "anthropic/glm-5.3", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]},
            headers={"authorization": "Bearer client-cred"},
        )
        assert r.status_code == 200
        sent = json.loads(zai_client.calls[0]["content"])
        assert sent["model"] == "glm-5.3"  # rewritten for the upstream
        assert zai_client.calls[0]["headers"]["authorization"] == "Bearer zai-secret"
        # count_tokens passthrough rewrites too
        r2 = client.post("/v1/messages/count_tokens", json={"model": "anthropic/glm-5.3", "messages": []})
        assert r2.status_code == 200
        sent2 = json.loads(zai_client.calls[1]["content"])
        assert sent2["model"] == "glm-5.3"


def test_models_endpoint_binding_entries_have_description(monkeypatch):
    from fastapi.testclient import TestClient

    zai_client = FakeUpstreamClient({"data": [{"id": "glm-5", "description": "GLM 5"}]})
    setup_proxy(
        monkeypatch,
        upstreams={"zai": ZAI_UPSTREAM},
        clients={"zai": zai_client},
        bindings={"anthropic/glm-5.3": {"upstream": "zai", "model": "glm-5.3"}, "glm-*": "zai"},
    )

    with TestClient(sp.app) as client:
        sp.UPSTREAM_CLIENTS.clear()
        sp.UPSTREAM_CLIENTS.update({"zai": zai_client})
        r = client.get("/v1/models")
        assert r.status_code == 200
        data = r.json()["data"]
        # Alias binding literal is listed with its alias id and a description
        assert data[0]["id"] == "anthropic/glm-5.3"
        assert "zai" in data[0]["description"]
        # Descriptions from upstream payloads pass through
        glm5 = next(m for m in data if m["id"] == "glm-5")
        assert glm5["description"] == "GLM 5"


def test_upstream_error_body_and_ratelimit_headers_logged(monkeypatch, caplog):
    """Non-2xx generation responses log the upstream error body and ratelimit headers,
    so 429s (plan limit vs capacity throttle) can be diagnosed from the proxy log."""
    import logging
    from fastapi.testclient import TestClient

    anthropic_client = FakeUpstreamClient(
        status_code=429,
        error_body={"type": "error", "error": {"type": "rate_limit_error", "message": "Usage limit reached. Limit resets at 12:00PM"}},
        error_headers={
            "content-type": "application/json",
            "retry-after": "1800",
            "anthropic-ratelimit-unified-status": "limit_exceeded",
        },
    )
    setup_proxy(
        monkeypatch,
        upstreams={"anthropic": ANTHROPIC_UPSTREAM},
        clients={"anthropic": anthropic_client},
        bindings={"*": "anthropic"},
    )

    with caplog.at_level(logging.INFO, logger="sampling_proxy"):
        with TestClient(sp.app) as client:
            sp.UPSTREAM_CLIENTS.clear()
            sp.UPSTREAM_CLIENTS.update({"anthropic": anthropic_client})
            r = client.post("/v1/messages", json={"model": "claude-sonnet-4-5", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]})
            assert r.status_code == 429

    logged = " ".join(rec.getMessage() for rec in caplog.records)
    assert "429" in logged
    assert "Usage limit reached" in logged
    assert "anthropic-ratelimit-unified-status" in logged
    assert "limit_exceeded" in logged


def test_request_start_logged_with_model(monkeypatch, caplog):
    """'Request started' includes model/stream/path so error lines are self-explanatory."""
    import logging
    from fastapi.testclient import TestClient

    anthropic_client = FakeUpstreamClient()
    setup_proxy(
        monkeypatch,
        upstreams={"anthropic": ANTHROPIC_UPSTREAM},
        clients={"anthropic": anthropic_client},
        bindings={"*": "anthropic"},
    )

    with caplog.at_level(logging.INFO, logger="sampling_proxy"):
        with TestClient(sp.app) as client:
            sp.UPSTREAM_CLIENTS.clear()
            sp.UPSTREAM_CLIENTS.update({"anthropic": anthropic_client})
            r = client.post("/v1/messages", json={"model": "claude-sonnet-4-5", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]})
            assert r.status_code == 200

    logged = " ".join(rec.getMessage() for rec in caplog.records)
    assert "model='claude-sonnet-4-5'" in logged
    assert "path='/v1/messages'" in logged


def test_debug_logs_gated_and_redacted(monkeypatch, caplog):
    """Debug header logs are gated by ENABLE_DEBUG_LOGS and redact credentials;
    final (post-auth) headers are logged so the actually-sent auth is traceable."""
    import logging
    from fastapi.testclient import TestClient

    zai_client = FakeUpstreamClient()
    setup_proxy(
        monkeypatch,
        upstreams={"zai": ZAI_UPSTREAM},
        clients={"zai": zai_client},
        bindings={"glm-*": "zai"},
    )
    secret = "super-secret-client-token-1234567890"

    # Gated off: no header/debug logs even though the request flows through
    with caplog.at_level(logging.DEBUG, logger="sampling_proxy"):
        with TestClient(sp.app) as client:
            sp.UPSTREAM_CLIENTS.clear()
            sp.UPSTREAM_CLIENTS.update({"zai": zai_client})
            r = client.post("/v1/messages", json={"model": "glm-5", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]}, headers={"authorization": f"Bearer {secret}"})
            assert r.status_code == 200
    assert not any("Outgoing Request Headers" in rec.getMessage() for rec in caplog.records)
    assert not any("Final outgoing headers" in rec.getMessage() for rec in caplog.records)

    # Gated on: debug lines appear, but no credential is ever logged in full
    zai_client.calls.clear()
    caplog.clear()
    sp.ENABLE_DEBUG_LOGS = True
    with caplog.at_level(logging.DEBUG, logger="sampling_proxy"):
        with TestClient(sp.app) as client:
            sp.UPSTREAM_CLIENTS.clear()
            sp.UPSTREAM_CLIENTS.update({"zai": zai_client})
            r = client.post("/v1/messages", json={"model": "glm-5", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]}, headers={"authorization": f"Bearer {secret}"})
            assert r.status_code == 200

    all_logs = " ".join(rec.getMessage() for rec in caplog.records)
    assert secret not in all_logs                      # client credential redacted/stripped
    assert "zai-secret" not in all_logs                # configured upstream key redacted
    assert "Outgoing Request Headers" in all_logs      # initial headers (redacted)
    assert "Final outgoing headers" in all_logs        # post-auth headers (redacted)
    assert "upstream 'zai'" in all_logs                # resolution traceable


@pytest.mark.asyncio
async def test_poll_models_flag_skips_upstream(monkeypatch):
    """poll_models=false skips an upstream entirely in /v1/models aggregation
    (no request, no warning noise — e.g. Anthropic passthrough or a stopped local server)."""
    zai_client = FakeUpstreamClient({"data": [{"id": "glm-4.7"}]})
    local_client = FakeUpstreamClient({"data": [{"id": "qwen"}]})
    anthropic_client = FakeUpstreamClient({"data": [{"id": "claude-x"}]})
    setup_proxy(
        monkeypatch,
        upstreams={
            "zai": {**ZAI_UPSTREAM, "poll_models": True},
            "local": {**LOCAL_UPSTREAM, "poll_models": False},
            "anthropic": {**ANTHROPIC_UPSTREAM, "poll_models": False},
        },
        clients={"zai": zai_client, "local": local_client, "anthropic": anthropic_client},
        bindings={},
    )

    payload = await sp.aggregate_models()
    ids = [m["id"] for m in payload["data"]]
    assert ids == ["glm-4.7"]          # only the enabled upstream contributed
    assert len(zai_client.calls) == 1
    assert not local_client.calls      # skipped: no request at all
    assert not anthropic_client.calls  # skipped: no request at all


@pytest.mark.asyncio
async def test_fetch_upstream_models_respects_flag(monkeypatch):
    class ExplodingClient:
        async def get(self, *a, **k):
            raise AssertionError("must not be called when poll_models=false")
    sp.UPSTREAM_CLIENTS.clear()
    sp.UPSTREAM_CLIENTS["x"] = ExplodingClient()
    assert await sp.fetch_upstream_models("x", {"poll_models": False}) == []


def test_exception_formatting_includes_type_for_empty_messages():
    err = httpx.ConnectError("")  # empty message, like a stopped local server
    assert "ConnectError" in sp._format_exception(err)
    err2 = RuntimeError("boom")
    assert "boom" in sp._format_exception(err2)


def test_startup_poll_skipped_when_poll_models_disabled(monkeypatch, caplog):
    """The legacy FIRST_AVAILABLE_MODEL startup poll also respects poll_models=false:
    no request, no warning, no startup delay from an unreachable upstream."""
    import logging
    from fastapi.testclient import TestClient

    local_off = {**LOCAL_UPSTREAM, "poll_models": False, "base_url": "http://127.0.0.1:59999"}  # valid URL, nothing listening
    setup_proxy(
        monkeypatch,
        upstreams={"local": local_off},
        clients={"local": FakeUpstreamClient()},
        bindings={"*": "local"},
    )

    with caplog.at_level(logging.INFO, logger="sampling_proxy"):
        with TestClient(sp.app) as client:
            assert client.get("/").status_code == 200

    logged = " ".join(rec.getMessage() for rec in caplog.records)
    assert "Polling" not in logged
    assert sp.FIRST_AVAILABLE_MODEL == "any"


def test_sanitize_accept_encoding_drops_br_zstd():
    """Only encodings httpx can transparently decompress are offered upstream,
    so stripped content-encoding headers never precede compressed bodies."""
    h = {"accept-encoding": "gzip, deflate, br, zstd"}
    sp._sanitize_accept_encoding(h)
    assert h["accept-encoding"] == "gzip, deflate"

    h2 = {"accept-encoding": "br"}
    sp._sanitize_accept_encoding(h2)
    assert h2["accept-encoding"] == "gzip, deflate"

    h3 = {"accept-encoding": "gzip;q=0.9, deflate, identity"}
    sp._sanitize_accept_encoding(h3)
    assert h3["accept-encoding"] == "gzip;q=0.9, deflate, identity"

    h4 = {}  # absent: untouched
    sp._sanitize_accept_encoding(h4)
    assert h4 == {}


def test_forwarded_request_never_offers_br_zstd(monkeypatch):
    from fastapi.testclient import TestClient

    anthropic_client = FakeUpstreamClient()
    setup_proxy(
        monkeypatch,
        upstreams={"anthropic": ANTHROPIC_UPSTREAM},
        clients={"anthropic": anthropic_client},
        bindings={"*": "anthropic"},
    )
    with TestClient(sp.app) as client:
        sp.UPSTREAM_CLIENTS.clear()
        sp.UPSTREAM_CLIENTS.update({"anthropic": anthropic_client})
        r = client.post(
            "/v1/messages",
            json={"model": "claude-sonnet-4-5", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]},
            headers={"accept-encoding": "gzip, deflate, br, zstd"},
        )
        assert r.status_code == 200
        sent_ae = anthropic_client.calls[0]["headers"].get("accept-encoding")
        assert "br" not in sent_ae and "zstd" not in sent_ae
        assert "gzip" in sent_ae


def test_log_upstream_error_handles_binary_body(monkeypatch, caplog):
    """Compressed/binary error bodies are noted, not dumped as mojibake."""
    import logging

    class BinaryBodyResponse:
        status_code = 429
        headers = {"content-type": "application/json", "content-encoding": "br"}
        content = b"\x8b\x02\x80\xbc\x11\x00\xb5\x9d"  # not valid UTF-8

        @property
        def text(self):
            return "\ufffd\ufffd\u25c8\ufffd"  # what httpx .text would produce

        async def aclose(self):
            pass

    with caplog.at_level(logging.INFO, logger="sampling_proxy"):
        sp.log_upstream_error(1, "anthropic", BinaryBodyResponse())
    logged = " ".join(rec.getMessage() for rec in caplog.records)
    assert "binary" in logged
    assert "content-encoding" in logged
    assert "\ufffd" not in logged
