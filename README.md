# Sampling Proxy

A middleware server that intercepts and modifies sampling parameters for generation requests to OpenAI-compatible backends. It supports both OpenAI-compatible and Anthropic request formats, allowing you to override default sampling parameters for different models.

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
   git clone <repository-url>
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

4. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

Run the proxy server with default settings:

```bash
python sampling-proxy.py
```

This will start the proxy server on `http://0.0.0.0:8001` and forward requests to an OpenAI-compatible backend at `http://127.0.0.1:8000`.

### Command Line Options

```bash
python sampling-proxy.py --help
```

Available options:
- `--host`: Host address for the proxy server (default: 0.0.0.0)
- `--port`: Port for the proxy server (default: 8001)
- `--target-host`: Host address for the backend (default: 127.0.0.1)
- `--target-port`: Port for the backend (default: 8000)
- `--debug-logs, -d`: Enable detailed debug logging
- `--override-logs, -o`: Show when sampling parameters are overridden
- `--enforce-params, -e`: Enforce specific parameters as JSON string

### Examples

1. **Run with custom ports and debug logging**:
   ```bash
   python sampling-proxy.py --port 8080 --target-port 8081 --debug-logs
   ```

2. **Run with enforced parameters**:
   ```bash
   python sampling-proxy.py --enforce-params '{"temperature": 0.7, "top_p": 0.9}'
   ```

3. **Run with override logs to see parameter changes**:
   ```bash
   python sampling-proxy.py --override-logs
   ```

### Environment Variables

You can also configure the proxy using environment variables:

- `TARGET_HOST`: Backend host address (default: 127.0.0.1)
- `TARGET_PORT`: Backend port (default: 8000)
- `SAMPLING_PROXY_HOST`: Proxy host address (default: 0.0.0.0)
- `SAMPLING_PROXY_PORT`: Proxy port (default: 8001)

Example:
```bash
export TARGET_HOST=192.168.1.100
export TARGET_PORT=8080
export SAMPLING_PROXY_PORT=9090
python sampling-proxy.py
```

## Configuration

### Sampling Parameters

The proxy applies sampling parameters in the following priority order:

1. **Enforced Parameters** (highest priority - always override)
2. **Request Parameters** (parameters sent in the request)
3. **Model-Specific Parameters** (configured per model)
4. **Default Parameters** (fallback values)

### Customizing Parameters

Edit the `sampling-proxy.py` file to modify:

1. **Default Parameters**:
   ```python
   DEFAULT_SAMPLING_PARAMS = {
       "top_p": 0.95,
       "min_p": 0.05,
       "top_k": 40,
       "repetition_penalty": 1.05,
       "temperature": 0.6,
   }
   ```

2. **Model-Specific Parameters**:
   ```python
   MODEL_SAMPLING_PARAMS = {
       "your-model-name": {
           "temperature": 0.15,
           "top_p": 0.95,
           "repetition_penalty": 1.0,
           "top_k": 40,
           "min_p": 0.01,
       },
   }
   ```

3. **Enforced Parameters** (always override):
   ```python
   ENFORCED_SAMPLING_PARAMS = {
       "temperature": 0.7,
       "top_p": 0.9,
   }
   ```

## API Endpoints

The proxy handles the following endpoints:

### Generation Endpoints (with parameter override)
- `/generate` - SGLang generation endpoint
- `/v1/completions` - OpenAI completions
- `/v1/chat/completions` - OpenAI chat completions
- `/v1/messages` - Anthropic messages (converted to OpenAI format)

### Other Endpoints (proxied without modification)
- `/v1/models` - List available models
- All other endpoints are passed through to the backend

### Health Check
- `/` - Returns proxy configuration and status

## Example Usage with Clients

### OpenAI Client
```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8001/v1",  # Point to the proxy
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
python sampling-proxy.py --debug-logs --override-logs
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
./sampling-proxy.sh
```

### Windows
```powershell
.\sampling-proxy.ps1
```

Both scripts will automatically activate the `sampling-proxy` virtual environment and start the proxy server.