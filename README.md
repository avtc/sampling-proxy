# Sampling Proxy

A middleware server for OpenAI-compatible backends with passthrough (OpenAI/Anthropic), Anthropic-to-OpenAI conversion, sampling params override, and mid-generation response validation.

## Features

- **Passthrough Modes**: OpenAI, Anthropic, and Anthropic-to-OpenAI conversion
- **Parameter Override**: Apply custom sampling parameters per model
- **Parallel Request Limits**: Limit concurrent requests per model with automatic queueing
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
| `--target-base-url` | Backend URL |
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

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `/chat/completions` | OpenAI chat completions |
| `/messages` | Anthropic messages (converted) |
| `/models` | List available models |
| `/` | Health check |

## License

MIT License
