# Sampling Proxy

A middleware server that intercepts and modifies sampling parameters for generation requests to OpenAI-compatible backends. It allows overriding specific parameters per model name when they are not set in the request, or enforcing parameter overrides when they are set in the request. The server supports both OpenAI-compatible and Anthropic request formats, enabling the use of Claude Code with OpenAI-compatible backends.

## Features

- **Parameter Override**: Automatically applies custom sampling parameters to generation requests
- **Model-Specific Settings**: Configure different parameters for different models
- **Format Conversion**: Converts between Anthropic and OpenAI request/response formats
- **Streaming Support**: Handles both streaming and non-streaming responses
- **Enforced Parameters**: Option to enforce specific parameters that override all others
- **Debug Logging**: Comprehensive logging for troubleshooting

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Setup with Virtual Environment

1. **Clone or download the project**:
   ```bash
   git clone https://github.com/avtc/sampling-proxy.git
   cd sampling-proxy
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv sampling-proxy
   ```

3. **Activate the virtual environment**:

   **On Windows:**
   ```cmd
   sampling-proxy\Scripts\activate
   ```

   **On macOS/Linux:**
   ```bash
   source sampling-proxy/bin/activate
   ```

4. **Make the shell script executable**:
   ```bash
   chmod +x ./sampling_proxy.sh
   ```

5. **Create configuration file**:
   ```bash
   cp config_sample.json config.json
   ```
   Then edit `config.json` to match your specific configuration needs.

6. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Updating from Git

To update your existing installation to the latest version from the git repository:

1. **Navigate to the project directory**:
   ```bash
   cd sampling-proxy
   ```

2. **Activate the virtual environment**:

   **On Windows:**
   ```cmd
   sampling-proxy\Scripts\activate
   ```

   **On macOS/Linux:**
   ```bash
   source sampling-proxy/bin/activate
   ```

3. **Pull the latest changes**:
   ```bash
   git pull origin main
   ```

4. **Update dependencies** (if requirements.txt has changed):
   ```bash
   pip install -r requirements.txt --upgrade
   ```

5. **Restart the proxy server** if it's currently running.

## Usage

### Basic Usage

Run the proxy server with default settings:

```bash
python sampling_proxy.py
```

This will start the proxy server on `http://0.0.0.0:8001` and forward requests to an OpenAI-compatible backend at `http://127.0.0.1:8000/v1`.

### Command Line Options

```bash
python sampling_proxy.py --help
```

Available options:
- `--config, -c`: Path to configuration JSON file (default: config.json)
- `--host`: Host address for the proxy server (overrides config)
- `--port`: Port for the proxy server (overrides config)
- `--base-path`: Base path for the proxy server (overrides config)
- `--target-base-url`: OpenAI compatible backend base url (overrides config)
- `--debug-logs, -d`: Enable detailed debug logging (overrides config)
- `--override-logs, -o`: Show when sampling parameters are overridden (overrides config)
- `--enforce-params, -e`: Enforce specific parameters as JSON string (overrides config)

### Examples

1. **Run with custom target base url and debug logging**:
   ```bash
   python sampling_proxy.py --target-base-url http://127.0.0.1:8000/v1 --debug-logs
   ```

2. **Run with a custom configuration file**:
   ```bash
   python sampling_proxy.py --config my-config.json
   ```

3. **Run with enforced parameters**:
   ```bash
   python sampling_proxy.py --enforce-params '{"temperature": 0.7, "top_p": 0.9}'
   ```

4. **Run with override logs to see parameter changes**:
   ```bash
   python sampling_proxy.py --override-logs
   ```

## Configuration

The proxy uses an external `config.json` file for configuration. A sample configuration file `config_sample.json` is provided - copy it to `config.json` and modify as needed. You can specify a custom config file path with the `--config` command-line argument.

### Sampling Parameter Priority

The proxy applies sampling parameters in the following priority order (from highest to lowest):

1. **Enforced sampling parameters** (always override everything)
2. **Parameters specified in the request**
3. **Model-specific sampling parameters**
4. **Default sampling parameters** (fallback values)

## API Endpoints

The proxy handles the following endpoints:

### Generation Endpoints (with parameter override)
- `/generate` - SGLang generation endpoint
- `/completions` - OpenAI completions
- `/chat/completions` - OpenAI chat completions
- `/messages` - Anthropic messages (converted to OpenAI format)

### Other Endpoints (proxied without modification)
- `/models` - List available models
- All other endpoints are passed through to the backend

### Health Check
- `/` - Returns proxy configuration and status

## Example Usage with Clients

### OpenAI Client
```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8001",  # Point to the proxy
    api_key="not-required"
)

response = client.chat.completions.create(
    model="your-model",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Anthropic Client
```python
from anthropic import Anthropic

client = Anthropic(
    base_url="http://localhost:8001",  # Point to the proxy
    api_key="not-required"
)

response = client.messages.create(
    model="your-model",
    max_tokens=100,
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Troubleshooting

### Enable Debug Logging
```bash
python sampling_proxy.py --debug-logs --override-logs
```

### Common Issues

1. **Connection Refused**: Ensure your backend server is running and accessible
2. **404 Errors**: Check if the backend supports the requested endpoints
3. **Parameter Not Applied**: Use `--override-logs` to see when parameters are being overridden

### Logs
The proxy provides detailed logging including:
- Incoming requests
- Parameter overrides
- Backend communication
- Error details

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Quick Start Scripts

For convenience, use the provided scripts to start the proxy with the correct virtual environment:

### Linux/macOS
```bash
./sampling_proxy.sh
```

### Windows
```powershell
.\sampling_proxy.ps1
```

Both scripts will automatically activate the `sampling_proxy` virtual environment and start the proxy server.

## Garbage Detection Mode

The proxy can validate AI responses using a local or remote model with OpenAI-compatible or Anthropic-compatible API, and automatically retry when garbage output is detected.

### Features

- **Repetition detection**: Catches loops where the same phrase is repeated 3+ times
- **Truncation detection**: Identifies responses that cut off mid-sentence
- **Malformed tool calls**: Detects invalid JSON in tool use blocks
- **Auto-retry**: Automatically retries with exponential backoff (1s, 2s delays)
- **Fail-open**: If validator is unavailable, responses pass through unmodified
- **Both backend modes**: Works with `openai_convert` and `anthropic_passthrough` modes
- **Flexible validator API**: Supports both Anthropic and OpenAI API formats

### Validation Support by Mode

| Mode | Non-streaming | Streaming |
|------|---------------|-----------|
| `anthropic_passthrough` | ✅ Full support | ✅ Full support |
| `openai_convert` | ✅ Full support | ✅ Full support |

**Note**: Streaming validation works by buffering the entire response, validating, then streaming to the client. This adds latency but ensures garbage detection for all response types.

### Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Start a validator endpoint with a small model (e.g., Qwen 3.5 0.8B, or any model that can classify text)
   - Can use LM Studio, Ollama, vLLM, or any OpenAI/Anthropic-compatible server

3. Copy the sample config:
   ```bash
   cp config_zai_sample.json config.json
   ```

4. Edit `config.json` to configure validation:
   - `validation.validator_url`: Validator endpoint URL (default: http://127.0.0.1:1234)
   - `validation.validator_model`: Model name at the validator endpoint
   - `validation.validator_api_format`: API format - `"anthropic"` or `"openai"`
   - `validation.validator_timeout_seconds`: Request timeout (default: 30.0)
   - `validation.max_retries`: Max retry attempts (default: 3)

5. Start the proxy:
   ```bash
   python sampling_proxy.py
   ```

6. Configure Claude Code to use the proxy as your API endpoint:
   - Base URL: `http://localhost:8001`

### Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `server.supports_openai` | Backend supports OpenAI format requests | `true` |
| `server.supports_anthropic` | Backend supports Anthropic format requests | `false` |
| `server.connect_timeout_seconds` | Connection timeout for backend | `5.0` |
| `server.timeout_seconds` | Read timeout for backend (per chunk) | `1200.0` |
| `logging.enable_debug_logs` | Enable debug logs | `false` |
| `logging.enable_override_logs` | Enable sampling param override logs | `false` |
| `logging.enable_validation_logs` | Enable validation process logs | `false` |
| `validation.enabled` | Enable response validation | `false` |
| `validation.validator_url` | Validator endpoint URL | `http://127.0.0.1:1234` |
| `validation.validator_model` | Model name for validation | `qwen-3.5-0.8b` |
| `validation.supports_openai` | Validator supports OpenAI format | `true` |
| `validation.supports_anthropic` | Validator supports Anthropic format | `false` |
| `validation.connect_timeout_seconds` | Connection timeout for validator | `5.0` |
| `validation.timeout_seconds` | Read timeout for validator | `300.0` |
| `validation.max_retries` | Max retry attempts | `3` |
| `validation.retry_base_delay_seconds` | Initial retry delay | `1.0` |
| `validation.retry_multiplier` | Backoff multiplier | `2.0` |

### Logs and Failed Responses

- Validation failures are logged to `~/.cache/garbage-proxy/logs/`
- Failed responses are saved to `~/.cache/garbage-proxy/failed/`