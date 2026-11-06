# Sampling Proxy

A middleware server for OpenAI-compatible backends with passthrough (OpenAI/Anthropic), Anthropic-to-OpenAI conversion, sampling params override, and mid-generation response validation.

## Features

- **Passthrough Modes**: OpenAI, Anthropic, and Anthropic-to-OpenAI conversion
- **Multiple Upstream Servers**: Route different models to different backends with glob pattern matching
- **Per-Upstream Auth**: Passthrough client credentials (API key or OAuth/subscription token) or inject a configured `${ENV_VAR}` API key per upstream
- **Gateway Model Discovery**: Aggregated `/v1/models` across all upstreams for clients like Claude Code (`CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1`)
- **Parameter Override**: Apply custom sampling parameters per model
- **Parallel Request Limits**: Limit concurrent requests per model with automatic queueing
- **Request Throttling:** Configurable cooldown delays between requests per model
- **Streaming Support**: Both streaming and non-streaming responses
- **Garbage Detection**: Validate responses and auto-retry when garbage output is detected
- **Mid-Stream Validation**: Detect garbage during generation (not just at the end)
- **Flexible Validator API**: Supports Anthropic and OpenAI API formats

## Quick Start

```bash
# Clone and setup
git clone https://github.com/avtc/sampling-proxy.git
cd sampling-proxy
python -m venv sampling-proxy

# Activate venv and install
source sampling-proxy/bin/activate  # Linux/macOS
sampling-proxy\Scripts\activate     # Windows
pip install -r requirements.txt

# Configure and run
cp config_sample.json config.json
# The sample's auth placeholders read env vars at startup, e.g. (PowerShell):
#   $env:ZAI_API_KEY = "..."; $env:LOCAL_API_KEY = "..."
# (an unset var degrades that upstream to passthrough auth with a warning)
python sampling_proxy.py
```

### One-line Scripts (auto-activate venv)

```bash
./sampling_proxy.sh        # Linux/macOS
.\sampling_proxy.ps1       # Windows
```

Both scripts auto-activate the `sampling-proxy` venv and run the proxy.

## Sample Validator Setup (llama.cpp)

Run a small model as a validator for garbage detection:

```bash
llama-server --hf-repo unsloth/Qwen3.5-4B-GGUF --hf-file Qwen3.5-4B-UD-Q6_K_XL.gguf --host 127.0.0.1 --port 1235 -ngl 99 --parallel 2 --jinja -fa on -c 40000 --chat-template-kwargs "{\"enable_thinking\": false}" --temp 1 --min-p 0 --top-p 0.95 --top-k 20 --repeat-penalty 1 --presence-penalty 1.5 --cache-ram 0
```

## Command Line Options

```bash
python sampling_proxy.py --help
```

| Option | Description |
|--------|-------------|
| `--config, -c` | Path to config JSON file (default: `config.json`) |
| `--host` | Proxy server host (overrides config) |
| `--port` | Proxy server port (overrides config) |
| `--base-path` | Proxy base path (overrides config) |
| `--target-base-url` | Backend URL (repeatable; `name=url` targets a specific upstream, a bare URL overrides the first) |
| `--model-bindings` | Model-to-upstream bindings as JSON (overrides config; values may be upstream names or alias objects) |
| `--debug-logs, -d` | Enable debug logging (overrides config) |
| `--override-logs, -o` | Enable override logs — show when sampling params are overridden (overrides config) |
| `--override-sampling-params` | Sampling params as JSON string, e.g. `'{"temperature": 0.7}'` (overrides config) |
| `--override-only-anthropic` | Apply overrides only to Anthropic-format requests |
| `--override-model-name` | Rewrite the model name in every request |
| `--parallel-limits` | Parallel limits as JSON string, e.g. `'{"global": 10, "glm-5": 2}'` (overrides config) |

## Garbage Detection

Enable validation to detect and retry garbage responses.

**Detected issue:**
- **Repetition loops**: Same phrase repeated 3+ times

### Key Options

| Option | Description | Default |
|--------|-------------|---------|
| `validation.enabled` | Enable response validation | `false` |
| `validation.validator_url` | Validator endpoint URL | `http://127.0.0.1:1235` |
| `validation.validator_model` | Model name for validation | `Qwen3.5-4B-UD-Q6_K_XL.gguf` |
| `validation.max_retries` | Max retry attempts | `1` |
| `validation.mid_stream_validation_enabled` | Validate during streaming | `true` |
| `validation.mid_stream_validation_interval_words` | Check every N words | `300` |

### Mid-Stream Validation

When enabled, validates responses periodically during streaming:

- Catches repetition loops at ~300 words (configurable)
- Interrupts garbage immediately and retries
- Reduces latency by not waiting for full garbage responses

```json
{
  "validation": {
    "enabled": true,
    "mid_stream_validation_enabled": true,
    "mid_stream_validation_interval_words": 300
  }
}
```

### Logs

Failed validation responses saved to `~/.sampling-proxy/logs/`

## Parallel Request Limits

Limit concurrent requests per model to prevent backend overload:

```json
{
  "parallel_limits": {
    "GLM-5": 2,
    "GLM-4.7": 3
  }
}
```

When limit is reached, additional requests queue automatically. Logs show queue status:

```
[INFO] Queueing for GLM-5, 2 requests waiting (limit: 2)
[INFO] Slot acquired GLM-5, used: 2/2
[INFO] Slot released GLM-5, used: 1/2
```

## Request Throttling

Add cooldown delays between requests to prevent backend overload:

```json
{
  "throttle": {
    "enabled": true,
    "global": {
      "start_pause_seconds": null,
      "end_pause_seconds": null
    },
    "per_model": {
      "GLM-5-turbo": {
        "start_pause_seconds": 1.0,
        "end_pause_seconds": 5.0
      }
    }
  }
}
```

**Timers:**
- **start_pause_seconds:** Cooldown after sending a request (delays the next request)
- **end_pause_seconds:** Cooldown after response completes (delays the next request)
- `null` disables the timer (default)

Per-model settings override global. Both timers default to `null` (disabled).

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `/chat/completions` | OpenAI chat completions |
| `/messages` | Anthropic messages (converted) |
| `/v1/models` | Aggregated model list across all upstreams (gateway discovery) |
| `/` | Health check (shows upstreams and bindings) |

## Multiple Upstream Servers

Route different models to different backend servers. Each upstream has its own URL, timeout, and format support.

### Configuration

```json
{
  "listen": {
    "host": "0.0.0.0",
    "port": 8001,
    "base_path": ""
  },
  "upstreams": [
    {
      "name": "local-openai",
      "base_url": "http://127.0.0.1:8000/v1",
      "connect_timeout_seconds": 5.0,
      "timeout_seconds": 1200.0,
      "supports_openai": true,
      "supports_anthropic": false
    },
    {
      "name": "zai-anthropic",
      "base_url": "https://api.z.ai/api/anthropic",
      "connect_timeout_seconds": 5.0,
      "timeout_seconds": 1200.0,
      "supports_openai": false,
      "supports_anthropic": true
    }
  ],
  "model_upstream_binding": {
    "glm-*": "zai-anthropic",
    "*": "local-openai"
  }
}
```

Upstreams also accept an `auth` block (passthrough vs injected `${ENV_VAR}` key) and a `poll_models` flag — see [Per-Upstream Authentication](#per-upstream-authentication) and [Gateway Model Discovery (`/v1/models`)](#gateway-model-discovery-v1models); `config_sample.json` shows the complete shape.

### Model Binding Rules

- **Exact match** takes priority: `"glm-5": "zai-anthropic"`
- **Alias objects** route and rewrite: `"anthropic/glm-5": {"upstream": "zai-anthropic", "model": "glm-5"}` — requests for `anthropic/glm-5` are forwarded with `"model": "glm-5"`. Needed for Claude Code model discovery, which only lists IDs containing `claude` or `anthropic`
- **Glob patterns** via `fnmatch`: `"glm-*": "zai-anthropic"` matches `glm-4.7`, `glm-5-Turbo`, etc.
- **Wildcard** `"*"` catches everything not matched above — use it as the **last** entry to set a default upstream (recommended, e.g. `"*": "anthropic"` so unknown/new model names don't fail with a `400`)
- Patterns are evaluated in config order — first match wins
- If no pattern matches, the proxy returns a `400` error
- Requests that carry **no model** (e.g. `count_tokens`, `event_logging`) also fall back to the `"*"` binding, then to the first Anthropic-capable upstream

### Per-Upstream Authentication

Each upstream controls how requests are authenticated:

```json
{
  "upstreams": [
    {
      "name": "anthropic",
      "base_url": "https://api.anthropic.com",
      "supports_anthropic": true,
      "auth": { "type": "passthrough" }
    },
    {
      "name": "zai-anthropic",
      "base_url": "https://api.z.ai/api/anthropic",
      "supports_anthropic": true,
      "auth": {
        "type": "api_key",
        "api_key": "${ZAI_API_KEY}",
        "header": "authorization",
        "scheme": "Bearer"
      }
    }
  ]
}
```

- **`passthrough`** (default) — client credentials (`authorization`, `x-api-key`) are forwarded untouched. Use this for the official Anthropic API so both API keys and subscription (OAuth) login keep working.
- **`api_key`** — client credentials are stripped and the configured key is injected instead:
  - `api_key` (alias: `token`) — literal value, or `${ENV_VAR}` / `${ENV_VAR:-default}` placeholder resolved from the **proxy's environment at startup** (restart the proxy after changing env vars)
  - `header` — `authorization` (default) or `x-api-key`, or any custom header
  - `scheme` — `Bearer` by default for `authorization`; empty for other headers

If an `api_key` placeholder resolves to an empty string (unset env var), the proxy logs a warning at startup and falls back to `passthrough` for that upstream.

### Gateway Model Discovery (`/v1/models`)

With multiple upstreams, `GET /v1/models` (also `/models`) returns an **aggregated** model list instead of forwarding to a single upstream:

1. Literal (non-glob) model names from `model_upstream_binding` — always routable through the proxy
2. Models discovered by polling each upstream's models endpoint, **unless the upstream sets `"poll_models": false`** (per-upstream auth applied; client credentials are reused for `passthrough` upstreams)
3. `extra_models` from config

```json
"models_endpoint": {
  "enabled": true,
  "cache_ttl_seconds": 300,
  "include_binding_models": true,
  "extra_models": []
}
```

The response uses Anthropic's List Models format. If all upstream polls fail, the binding literals (and a stale cache) are served so the model picker is never empty.

Set `"poll_models": false` on upstreams where polling is pointless or noisy:

- an `api.anthropic.com` passthrough upstream — the warm-up poll has no client credentials to forward, so it always 401s; and Claude Code's built-in picker rows already list Anthropic models anyway
- an often-stopped local server — avoids connect-error warnings on every cache refresh (set it back to `true` when you want its live models listed)

The flag also disables the legacy startup poll (`Polling .../v1/models to get available models...`), which fetches a fallback model name for the Anthropic→OpenAI conversion path. If all your OpenAI-capable upstreams support Anthropic natively (passthrough), the conversion fallback is never used and the startup poll is pure overhead; the conversion fallback also honors `override.model_name` if you ever need to pin it.

### Claude Code Setup

Point Claude Code at the proxy and enable gateway model discovery:

```
ANTHROPIC_BASE_URL=http://127.0.0.1:8001
CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1
```

Notes:

- Claude Code always sends `model` in `/v1/messages` requests, so routing via `model_upstream_binding` always applies. Add a `"*": "anthropic"` wildcard as the last binding so new/unknown model IDs route to a default upstream instead of erroring.
- **Discovery only keeps Anthropic-looking IDs**: Claude Code filters `/v1/models` entries client-side, keeping only IDs that contain `claude` or `anthropic` (e.g. `glm-5.3` or `qwen-3` are dropped from the picker). To surface third-party models, bind them under an alias ID containing `anthropic` and rewrite the model name for the upstream:

```json
"model_upstream_binding": {
  "anthropic/glm-5.3": { "upstream": "zai-anthropic", "model": "glm-5.3" }
}
```

  A request for `anthropic/glm-5.3` routes to `zai-anthropic` and is forwarded with `"model": "glm-5.3"` (also on `count_tokens`). Plain string bindings still work for direct API calls with the real model name.
- **Discovery has a ~3 second timeout** and treats redirects or a slow response as silent failure. The proxy warms the model cache at startup, serves a stale cache instantly while refreshing in the background, and caps upstream polls at 2.5s — binding literals are always listed even when every upstream is unreachable.
- The keyless `HEAD /api/hello` connectivity probe is answered locally with `200 {"message": "hello"}`, matching `api.anthropic.com`.
- **Subscription (Pro/Max) login works**: keep the `anthropic` upstream on `passthrough` auth and do **not** set `ANTHROPIC_API_KEY`/`ANTHROPIC_AUTH_TOKEN`. Claude Code attaches its OAuth bearer token and `anthropic-beta` (OAuth capability) headers, which the proxy forwards untouched to `api.anthropic.com` — required, since stripping `anthropic-beta` fails those requests with `401`. Upstreams configured with `api_key` auth strip client credentials, so your OAuth token never leaks to third-party upstreams. A `429` from the Anthropic upstream with `rate_limit_error` / message `"Error"` is generic subscription-traffic throttling — retryable, not a plan limit.
- **Add a non-Anthropic model to the picker while keeping the subscription** (discovery is skipped without a gateway credential, so gateway `/v1/models` never populates the picker on subscription login):

```bash
export ANTHROPIC_CUSTOM_MODEL_OPTION="anthropic/glm-5.3"          # must match an alias binding
export ANTHROPIC_CUSTOM_MODEL_OPTION_NAME="GLM 5.3 (zai)"
export ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION="Routed via sampling-proxy"
# optional: export ANTHROPIC_CUSTOM_MODEL_OPTION_SUPPORTED_CAPABILITIES=thinking
```

  This adds one row at the bottom of `/model` next to the built-in aliases, with no credential env vars — the claude.ai login stays active, and the alias binding routes + rewrites the model for the upstream. For several custom rows, use the `modelPicker` setting instead (user `settings.json`, Claude Code v2.1.242+): rows are appended after the built-in lineup, accept arbitrary model IDs (no claude/anthropic filter — plain bound IDs like `glm-5.3` work directly), and support `label`/`description`/`behavesAs` per row:

```json
"modelPicker": {
  "options": [
    { "model": "glm-5.3", "label": "GLM 5.3 (zai)", "description": "via sampling-proxy → zai-anthropic" }
  ]
}
```

- API-key mode also works: set `ANTHROPIC_API_KEY` (or `ANTHROPIC_AUTH_TOKEN`) to your Anthropic key — it passes through to the `anthropic` upstream only and is stripped for upstreams with configured keys.

### Legacy Single-Upstream Config

The old `server.target_base_url` format still works — it auto-wraps as a single upstream:

```json
{
  "server": {
    "target_base_url": "http://127.0.0.1:8000",
    "supports_openai": true,
    "supports_anthropic": false
  }
}
```

## License

MIT License
