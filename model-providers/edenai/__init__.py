"""Eden AI provider profile: one OpenAI-compatible API for many labs' models."""

import json
import logging
import time
import urllib.request
from typing import Any

from agent.opencode_affinity import resolve_affinity_key
from hermes_cli.version_info import hermes_user_agent
from providers import register_provider
from providers.base import ProviderProfile

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.edenai.run/v3"
_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
_CATALOG_TTL_SECONDS = 600.0
# A failed fetch is not retried for this long, so an outage costs one timeout a minute, not one a turn.
_FAILURE_TTL_SECONDS = 60.0
# (catalog URL, API key) -> (fetched at, catalog items, or None after a failed fetch).
_catalog_cache: dict[tuple[str, str], tuple[float, list[dict] | None]] = {}


def _catalog_url(base_url: str | None) -> str:
    """Eden AI routes a model name (``claude-sonnet-4.5``) to one of the providers serving it, so
    the catalog is its models view, one entry per name, rather than one ``provider/model`` id per
    route. A custom base URL (``api.eu.edenai.run``) keeps the view."""
    return (base_url or _BASE_URL).rstrip("/") + "/models?view=models"


class EdenAIProfile(ProviderProfile):
    """Eden AI: model names and their context windows from its catalog, a session id that
    survives compression, and top-level ``reasoning_effort``."""

    def _catalog(self, *, api_key: str | None, base_url: str | None, timeout: float) -> list[dict] | None:
        """The account's catalog items (its own hosted models included), cached per URL and key."""
        from hermes_cli.urllib_security import open_credentialed_url

        url = _catalog_url(base_url)
        key = (url, api_key or "")
        cached = _catalog_cache.get(key)
        if cached:
            ttl = _CATALOG_TTL_SECONDS if cached[1] is not None else _FAILURE_TTL_SECONDS
            if time.monotonic() - cached[0] < ttl:
                return cached[1]
        req = urllib.request.Request(url, headers={
            "Accept": "application/json", "User-Agent": hermes_user_agent(), **self.default_headers,
        })
        if api_key:
            req.add_header("Authorization", f"Bearer {api_key}")
        items: list[dict] | None = None
        try:
            with open_credentialed_url(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode())
            data = payload.get("data") if isinstance(payload, dict) else None
            items = [item for item in data if isinstance(item, dict)] if isinstance(data, list) else None
        except (OSError, ValueError) as exc:
            logger.debug("edenai catalog: %s", exc)
        _catalog_cache[key] = (time.monotonic(), items)
        return items

    def fetch_models(
        self, *, api_key: str | None = None, base_url: str | None = None, timeout: float = 8.0,
    ) -> list[str] | None:
        """Model names; the LLM catalog lists chat models only."""
        items = self._catalog(api_key=api_key, base_url=base_url, timeout=timeout)
        return [item["id"] for item in items if isinstance(item.get("id"), str)] if items is not None else None

    def probe_models(
        self, *, base_url: str, api_key: str | None = None, timeout: float = 5.0,
    ) -> tuple[list[str] | None, str, str] | None:
        """Setup and model validation on a custom endpoint check against model names too."""
        ids = self.fetch_models(api_key=api_key, base_url=base_url, timeout=timeout)
        return (ids, _catalog_url(base_url), base_url) if ids is not None else None

    def invalidate_catalog_cache(self) -> None:
        _catalog_cache.clear()

    def live_context_length(self, model: str, *, api_key: str = "", base_url: str = "") -> int | None:
        """The window listed for exactly ``model``: ``mistral-small`` and ``mistral-small-latest``
        are different models, so a substring match would take the wrong one."""
        for item in self._catalog(api_key=api_key, base_url=base_url, timeout=8.0) or []:
            if item.get("id") == model:
                return item.get("context_length")
        return None

    def build_extra_body(self, *, session_id: str | None = None, **context: Any) -> dict[str, Any]:
        """``session_id`` keeps a conversation's calls on one provider (its prompt cache) and groups
        them as one session. The conversation's root scope comes first: compression does not rotate
        it, and auxiliary calls, which pass no ``session_id``, share it."""
        key = resolve_affinity_key(session_id)
        return {"session_id": key} if key else {}

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, **context: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Top-level ``reasoning_effort``: ``none`` turns reasoning off, an unset effort is left to
        the model's default. Eden AI takes every level below and retries one a model does not take
        with the nearest one it does, so the core ``supports_reasoning`` allowlist is not consulted."""
        config = reasoning_config if isinstance(reasoning_config, dict) else {}
        if config.get("enabled") is False:
            return {}, {"reasoning_effort": "none"}
        effort = str(config.get("effort") or "").strip().lower()
        return ({}, {"reasoning_effort": effort}) if effort in _EFFORTS else ({}, {})


edenai = EdenAIProfile(
    name="edenai", aliases=("eden", "eden-ai"), display_name="Eden AI",
    description="Eden AI: one API for the models of many labs and providers",
    signup_url="https://app.edenai.run/", env_vars=("EDENAI_API_KEY", "EDENAI_BASE_URL"),
    base_url=_BASE_URL, auth_type="api_key",
    # A custom endpoint at any Eden AI host (api.eu.edenai.run, ...) uses this profile too.
    hostnames=("edenai.run",),
    default_headers={"HTTP-Referer": "https://hermes-agent.nousresearch.com", "X-Title": "Hermes Agent"},
    supports_prompt_cache_key=True,
    default_aux_model="gemini-2.5-flash-lite",
    # Lead the picker; the live catalog's other names follow.
    fallback_models=("claude-sonnet-5.5", "gpt-5.5", "gemini-2.5-pro", "deepseek-v4.1-flash"),
)

register_provider(edenai)
