"""Tests for the Eden AI provider plugin, run from a Hermes Agent checkout with the plugin installed
under ``$HERMES_HOME/plugins/model-providers/edenai`` (see the README)."""

import io
import json
import sys

import pytest

from providers import get_provider_profile, profile_for_host

_CATALOG = {
    "object": "list",
    "data": [
        {"id": "mistral-small", "context_length": 32_768},
        {"id": "mistral-small-latest", "context_length": 131_072},
        {"id": "claude-sonnet-5.5", "context_length": 1_000_000},
    ],
}


def _profile():
    p = get_provider_profile("edenai")
    assert p is not None
    return p


@pytest.fixture
def server(monkeypatch):
    """A fake Eden AI host: records each request, serves ``_CATALOG`` until ``down`` is set."""

    class Server:
        requests: list = []
        down = False

    def fake_open(req, timeout):
        Server.requests.append(req)
        if Server.down:
            raise OSError("down")
        return io.BytesIO(json.dumps(_CATALOG).encode())

    Server.requests = []
    monkeypatch.setattr("hermes_cli.urllib_security.open_credentialed_url", fake_open)
    monkeypatch.setattr(sys.modules[type(_profile()).__module__], "_catalog_cache", {})
    return Server


class TestCatalog:
    def test_lists_model_names_from_the_models_view(self, server):
        assert _profile().fetch_models(api_key="k") == [
            "mistral-small", "mistral-small-latest", "claude-sonnet-5.5",
        ]
        (req,) = server.requests
        assert req.full_url == "https://api.edenai.run/v3/models?view=models"
        assert req.get_header("Authorization") == "Bearer k"
        assert req.get_header("X-title") == "Hermes Agent"
        assert not req.get_header("User-agent").startswith("Python-urllib")

    def test_a_custom_endpoint_is_probed_by_model_name(self, server):
        from hermes_cli.models import probe_api_models

        result = probe_api_models("k", "https://api.eu.edenai.run/v3")
        assert result["models"] == ["mistral-small", "mistral-small-latest", "claude-sonnet-5.5"]
        assert result["probed_url"] == "https://api.eu.edenai.run/v3/models?view=models"

    def test_context_window_is_matched_by_exact_name(self, server):
        p = _profile()
        assert p.live_context_length("mistral-small-latest") == 131_072
        assert p.live_context_length("mistral-small") == 32_768
        assert p.live_context_length("mistral") is None
        assert len(server.requests) == 1  # one fetch serves every lookup

    def test_an_outage_costs_one_request_until_it_expires(self, server):
        server.down = True
        p = _profile()
        assert p.fetch_models() is None
        assert p.live_context_length("claude-sonnet-5.5") is None
        assert len(server.requests) == 1

    def test_a_refresh_refetches(self, server):
        p = _profile()
        p.fetch_models()
        p.invalidate_catalog_cache()
        p.fetch_models()
        assert len(server.requests) == 2

    def test_each_account_has_its_own_catalog(self, server):
        p = _profile()
        p.fetch_models(api_key="a")
        p.fetch_models(api_key="b")
        assert len(server.requests) == 2


class TestRequests:
    def test_custom_endpoints_at_eden_hosts_use_this_profile(self):
        assert profile_for_host("https://api.edenai.run/v3").name == "edenai"
        assert profile_for_host("https://api.eu.edenai.run/v3").name == "edenai"

    def test_session_id_is_stable_for_a_conversation(self):
        p = _profile()
        first = p.build_extra_body(session_id="20260101_120000_abc123")
        assert first["session_id"]
        assert p.build_extra_body(session_id="20260101_120000_abc123") == first
        assert p.build_extra_body(session_id=None) == {}

    @pytest.mark.parametrize(("config", "effort"), [
        (None, None),  # the model's default
        ({"enabled": True, "effort": "high"}, "high"),
        ({"enabled": True, "effort": "max"}, "max"),
        ({"enabled": False}, "none"),
    ])
    def test_reasoning_effort_is_top_level(self, config, effort):
        extra_body, top = _profile().build_api_kwargs_extras(reasoning_config=config)
        assert extra_body == {}
        assert top.get("reasoning_effort") == effort
