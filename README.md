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

5. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

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

The proxy uses an external `config.json` file for configuration. You can specify a custom config file path with the `--config` command-line argument.

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