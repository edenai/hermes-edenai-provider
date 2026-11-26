# hermes-edenai-provider

A [Hermes Agent](https://github.com/NousResearch/hermes-agent) model-provider plugin for [Eden AI](https://www.edenai.co), an OpenAI-compatible API for the models of many labs and providers.

Eden AI routes a model name such as `claude-sonnet-4.5` to one of the providers serving it. The plugin lists those names, so Hermes' picker shows one entry per model rather than one per provider route.

## Install

```bash
hermes plugins install edenai/hermes-edenai-provider/model-providers/edenai --enable
hermes chat --provider edenai -m claude-sonnet-5.5
```

Create an API key at [app.edenai.run](https://app.edenai.run/). The installer asks for it; to keep it, put it in `~/.hermes/.env` as `EDENAI_API_KEY`. Then `hermes model` and pick **Eden AI**.

## What it does

| | |
|---|---|
| Provider id | `edenai` (aliases `eden`, `eden-ai`) |
| Base URL | `https://api.edenai.run/v3` (override with `EDENAI_BASE_URL`, e.g. `https://api.eu.edenai.run/v3`) |
| Auth | `EDENAI_API_KEY`, sent as `Authorization: Bearer` |
| Model list | live from `GET <base URL>/models?view=models`, cached 10 minutes per key; if that fails, the fallback list in `__init__.py` |
| Context windows | from the same list, by exact model name |
| Reasoning | Hermes' reasoning effort, sent as `reasoning_effort` |
| Sessions | a `session_id` per conversation, kept across context compression, so a conversation stays on one provider and its prompt cache |

A custom endpoint at any `edenai.run` host (set up through Hermes' "custom" provider) uses this profile too.

The plugin registers one provider profile and nothing else: no tools, hooks, middleware or extra dependencies.

**Network and data.** Prompts, tool schemas and the API key go to Eden AI (`api.edenai.run`, or the host in `EDENAI_BASE_URL`), and from there to the provider serving the chosen model. Requests carry `X-Title: Hermes Agent` and `HTTP-Referer: https://hermes-agent.nousresearch.com`, so Eden AI can show them as Hermes calls. The plugin stores nothing on disk.

## Check it

From a Hermes Agent checkout:

```bash
hermes plugins validate model-providers/edenai
```

The tests run inside a Hermes Agent checkout, with the plugin installed under `$HERMES_HOME/plugins/model-providers/edenai`:

```bash
cp tests/test_edenai_profile.py <hermes-agent>/tests/providers/
cd <hermes-agent> && scripts/run_tests.sh tests/providers/test_edenai_profile.py
```

## License

MIT.
