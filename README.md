# Sampling Proxy

A middleware server for OpenAI-compatible backends with passthrough (OpenAI/Anthropic), Anthropic-to-OpenAI conversion, sampling params override, and mid-generation response validation.

## Features

- **Passthrough Modes**: OpenAI, Anthropic, and Anthropic-to-OpenAI conversion
- **Multiple Upstream Servers**: Route different models to different backends with glob pattern matching
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
| `--config, -c` | Path to config JSON file |
| `--host` | Proxy server host |
| `--port` | Proxy server port |
| `--target-base-url` | Backend URL (repeatable, use `name=url` to target specific upstream) |
| `--model-bindings` | Model-to-upstream bindings as JSON string |
| `--debug-logs, -d` | Enable debug logging |
| `--enforce-params, -e` | Enforce parameters as JSON |

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
| `/models` | List available models |
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

### Model Binding Rules

- **Exact match** takes priority: `"glm-5": "zai-anthropic"`
- **Glob patterns** via `fnmatch`: `"glm-*": "zai-anthropic"` matches `glm-4.7`, `glm-5-Turbo`, etc.
- **Wildcard** `"*"` catches everything not matched above
- Patterns are evaluated in config order — first match wins
- If no pattern matches, the proxy returns a `400` error

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
