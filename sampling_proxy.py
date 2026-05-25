import os
import json
import logging
import httpx
from typing import Optional
from fastapi import FastAPI, Request, Response, status
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import uvicorn
import asyncio # Import asyncio for potential sleep
import argparse # Import argparse for command-line arguments
import threading

# Configure logger
logger = logging.getLogger("sampling_proxy")
logger.setLevel(logging.DEBUG)
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:     %(message)s",
)

# Global request counter for log correlation
_request_counter = 0
_request_counter_lock = threading.Lock()

def get_request_id():
    """Get next request ID for log correlation."""
    global _request_counter
    with _request_counter_lock:
        _request_counter += 1
        return _request_counter

def log_info(request_id: int, message: str):
    """Log info with request ID prefix."""
    logger.info("[R:%s] %s", request_id, message)

# Import validator module for garbage detection
from validator import (
    validate_response,
    validate_response_partial,
    save_failed_response,
    save_mid_stream_failure,
    create_error_message,
    calculate_retry_delay,
    ValidationResult,
    StreamingValidator,
    StreamingValidationBuffer,
    count_words_in_text,
    extract_text_from_sse_chunks,
    build_anthropic_error_stream,
    build_openai_error_stream,
    build_anthropic_error_json,
    build_openai_error_json
)

# Import throttle manager for request throttling
from throttle_manager import ThrottleManager


def load_config(config_path="config.json"):
    """
    Load configuration from JSON file.
    Returns a dictionary with configuration values.
    If config file doesn't exist or is invalid, returns default values.
    """
    default_config = {
        "server": {
            "target_base_url": "http://127.0.0.1:8000/v1",
            "sampling_proxy_base_path": "",
            "sampling_proxy_host": "0.0.0.0",
            "sampling_proxy_port": 8001,
            "connect_timeout_seconds": 5.0,
            "timeout_seconds": 1200.0,
            "supports_openai": True,
            "supports_anthropic": False
        },
        "logging": {
            "enable_debug_logs": False,
            "enable_override_logs": False
        },
        "default_sampling_params": {},
        "override": {
            "only_anthropic": False,
            "model_name": None,
            "sampling_params": {}
        },
        "model_sampling_params": {},
        "validation": {
            "enabled": False,
            "validator_url": "http://127.0.0.1:1234",
            "validator_model": "qwen-3.5-0.8b",
            "supports_openai": True,
            "supports_anthropic": False,
            "connect_timeout_seconds": 5.0,
            "timeout_seconds": 300.0,
            "max_retries": 3,
            "retry_base_delay_seconds": 1.0,
            "retry_multiplier": 2.0,
            "mid_stream_validation_enabled": False,
            "mid_stream_validation_interval_words": 300
        },
        "parallel_limits": {},
        "throttle": {
            "enabled": False,
            "global": {
                "start_pause_seconds": None,
                "end_pause_seconds": None
            },
            "per_model": {}
        }
    }
    
    if not os.path.exists(config_path):
        logger.warning(f"Config file '{config_path}' not found. Using default values.")
        return default_config
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Merge with defaults to ensure all required keys exist
        merged_config = default_config.copy()
        for key, value in config.items():
            if key in merged_config:
                if isinstance(merged_config[key], dict) and isinstance(value, dict):
                    merged_config[key].update(value)
                else:
                    merged_config[key] = value
            else:
                merged_config[key] = value
        
        # Filter out null values from sampling params (convert to empty dicts)
        if merged_config.get("default_sampling_params"):
            merged_config["default_sampling_params"] = {
                k: v for k, v in merged_config["default_sampling_params"].items()
                if v is not None
            }
        
        # Filter out null values from override.sampling_params
        if merged_config.get("override"):
            override_config = merged_config["override"]
            if "sampling_params" in override_config:
                override_config["sampling_params"] = {
                    k: v for k, v in override_config["sampling_params"].items()
                    if v is not None
                }
        
        if merged_config.get("model_sampling_params"):
            filtered_model_params = {}
            for model, params in merged_config["model_sampling_params"].items():
                filtered_params = {
                    k: v for k, v in params.items()
                    if v is not None
                }
                if filtered_params:  # Only include models with non-null params
                    filtered_model_params[model] = filtered_params
            merged_config["model_sampling_params"] = filtered_model_params
        
        logger.info(f"Configuration loaded from '{config_path}'")
        return merged_config
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config file '{config_path}': {e}. Using default values.")
        return default_config
    except Exception as e:
        logger.error(f"Error loading config file '{config_path}': {e}. Using default values.")
        return default_config

def extract_base_path(url):
    """
    Extract the base path from a URL.
    For example, "http://127.0.0.1:8000/abc/v4" returns "/abc/v4"
    """
    from urllib.parse import urlparse
    parsed = urlparse(url)
    return parsed.path

def transform_path(original_path, from_base_path, to_base_path):
    """
    Transform a path from one base path to another.
    For example, with from_base_path="/v1" and to_base_path="/abc/v4":
    "/v1/completions" -> "/abc/v4/completions"
    "/v1/chat/completions" -> "/abc/v4/chat/completions"
    
    If original_path doesn't start with from_base_path, it's returned unchanged.
    """
    # Ensure base paths start with /
    if not from_base_path.startswith('/'):
        from_base_path = '/' + from_base_path
    if not to_base_path.startswith('/'):
        to_base_path = '/' + to_base_path
    
    # Remove trailing slashes for consistent comparison
    from_base_path = from_base_path.rstrip('/')
    to_base_path = to_base_path.rstrip('/')
    
    # Check if the path starts with the from_base_path
    if original_path.startswith(from_base_path):
        # Replace the base path
        return original_path.replace(from_base_path, to_base_path, 1)
    else:
        # Path doesn't start with the expected base path, return as is
        return original_path

# --- Configuration ---
# These will be initialized in the main block after loading config
TARGET_BASE_URL = None
TARGET_BASE_PATH = None
SAMPLING_PROXY_HOST = None
SAMPLING_PROXY_PORT = None
SAMPLING_PROXY_BASE_PATH = None
ENABLE_DEBUG_LOGS = False
ENABLE_OVERRIDE_LOGS = False
ENABLE_VALIDATION_LOGS = False
DEFAULT_SAMPLING_PARAMS = {}
OVERRIDE_CONFIG = {}
OVERRIDE_ONLY_ANTHROPIC = False
OVERRIDE_MODEL_NAME = None
OVERRIDE_SAMPLING_PARAMS = {}
MODEL_SAMPLING_PARAMS = {}

# Server capability configuration
# Determines what formats the backend server supports for passthrough
SERVER_SUPPORTS_OPENAI = True
SERVER_SUPPORTS_ANTHROPIC = False
VALIDATION_CONFIG = {"enabled": False}
THROTTLE_CONFIG = {"enabled": False}

# Per-model parallel request limits (model_name -> asyncio.Semaphore)
PARALLEL_LIMITS = {}
MODEL_SEMAPHORES = {}

# Global parallel request limit (across all models)
GLOBAL_LIMIT = None
GLOBAL_SEMAPHORE = None

# Throttle manager for request pacing
throttle_manager = None

# List of API path suffixes that are considered "generation" endpoints.
# Note: We check if the path ENDS WITH these suffixes to handle various prefixes
GENERATION_ENDPOINT_SUFFIXES = [
    "generate",            # Common SGLang generation endpoint
    "completions",         # OpenAI-compatible completions endpoint
    "chat/completions",    # OpenAI-compatible chat completions endpoint
    "v1/messages",         # Anthropic-compatible messages endpoint
]

# List of Anthropic-specific endpoints that should be handled locally
ANTHROPIC_ENDPOINTS = [
    "api/event_logging/batch",  # Anthropic event logging endpoint
    "v1/messages/count_tokens", # Anthropic token counting endpoint
]

# Global variable to store the first available model name from /models to be used for anthropic requests
FIRST_AVAILABLE_MODEL = "any" # sglang allows any model name, vllm require exact match

# Initialize an httpx AsyncClient for making requests to the upstream server.
# This client is designed for efficient connection pooling.
# A higher timeout is set to accommodate potentially long LLM generation times.
# Note: This will be re-initialized after config loading in the main block
client = None

def get_model_semaphore(model_name: str):
    """Get the semaphore for a model if a parallel limit is configured. Returns None if no limit."""
    return MODEL_SEMAPHORES.get(model_name.lower())

def extract_model_for_throttle(request_data: dict) -> str:
    """Extract model name from request data for throttle lookup."""
    model = request_data.get("model")
    if model:
        return model
    return "global"

def get_global_semaphore():
    """Get the global semaphore if a global limit is configured. Returns None if no limit."""
    return GLOBAL_SEMAPHORE

# --- FastAPI Application Lifespan Setup ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events for the FastAPI application.
    Ensures the httpx client is properly closed when the application shuts down.
    """
    global FIRST_AVAILABLE_MODEL, client
    logger.info("FastAPI application startup.")

    # Initialize client with the correct TARGET_BASE_URL and timeout from config
    connect_timeout = CONFIG["server"].get("connect_timeout_seconds", 5.0)
    read_timeout = CONFIG["server"].get("timeout_seconds", 1200.0)
    timeout = httpx.Timeout(connect=connect_timeout, read=read_timeout, write=read_timeout, pool=connect_timeout)
    client = httpx.AsyncClient(base_url=TARGET_BASE_URL, timeout=timeout)
    
    # Validate server capabilities - at least one format must be supported
    if not SERVER_SUPPORTS_OPENAI and not SERVER_SUPPORTS_ANTHROPIC:
        raise ValueError(
            "Invalid configuration: server must support at least one format. "
            "Set 'supports_openai: true' and/or 'supports_anthropic: true' in config."
        )

    # Poll /models to get the first available model (only if server supports OpenAI and no override model set)
    # Skip polling if: 1) server doesn't support OpenAI (/models is OpenAI-only), or 2) override model already configured
    if SERVER_SUPPORTS_OPENAI and not OVERRIDE_MODEL_NAME:
        # If target base path is empty, use /v1/models for standard OpenAI/Anthropic servers
        # Otherwise, the base path already includes the prefix
        if not TARGET_BASE_PATH:
            models_path = "/v1/models"
        else:
            models_path = "/models"

        try:
            logger.info(f"Polling {TARGET_BASE_URL}{models_path} to get available models...")
            response = await client.get(models_path)
            if response.status_code == 200:
                models_data = response.json()
                if "data" in models_data and len(models_data["data"]) > 0:
                    FIRST_AVAILABLE_MODEL = models_data["data"][0]["id"]
                    logger.info(f"Successfully retrieved first available model: {FIRST_AVAILABLE_MODEL}")
                else:
                    logger.warning("No models found in /models response")
            else:
                logger.warning(f"Failed to get models from {models_path}. Status: {response.status_code}")
        except Exception as e:
            logger.warning(f"Error polling {models_path}: {e}")
    elif OVERRIDE_MODEL_NAME:
        # Use the override model name as the first available model
        FIRST_AVAILABLE_MODEL = OVERRIDE_MODEL_NAME
        logger.info(f"Using override model name: {FIRST_AVAILABLE_MODEL}")
    else:
        logger.info("Skipping model polling (server doesn't support OpenAI format)")
    
    yield # Application starts here
    logger.info("FastAPI application shutdown.")
    if client:
        await client.aclose()
        logger.info("HTTPX client closed.")

# --- FastAPI Application Setup ---
app = FastAPI(
    title="Sampling Proxy",
    description="A middleware server to override sampling parameters for generation requests, supports OpenAI-compatible and Anthropic request formats.",
    version="1.0.0",
    lifespan=lifespan # Register the lifespan context manager
)

@app.get("/")
async def read_root():
    """
    Root endpoint for a basic health check and to display middleware configuration.
    """
    return {
        "message": "Sampling Proxy is running.",
        "target_backend": TARGET_BASE_URL,
        "sampling_proxy_port": SAMPLING_PROXY_PORT,
        "default_sampling_params": DEFAULT_SAMPLING_PARAMS,
        "override": OVERRIDE_CONFIG,
        "model_sampling_params_configured": list(MODEL_SAMPLING_PARAMS.keys()),
        "generation_endpoints_monitored": GENERATION_ENDPOINT_SUFFIXES,
        "anthropic_endpoints_handled_locally": ANTHROPIC_ENDPOINTS,
        "debug_logs_enabled": ENABLE_DEBUG_LOGS,
        "parallel_limits": {**({"global": GLOBAL_LIMIT} if GLOBAL_LIMIT is not None else {}), **PARALLEL_LIMITS},
    }

def parse_sse_to_response(sse_text: str) -> Optional[dict]:
    """Parse SSE stream text to extract final response dict."""
    content_blocks = {}
    current_index = None
    message_data = None

    for line in sse_text.split('\n'):
        line = line.strip()
        if not line:
            continue

        if line.startswith('data: '):
            data_str = line[6:]
            if data_str == '[DONE]':
                continue

            try:
                data = json.loads(data_str)
                event_type = data.get('type')

                if event_type == 'message_start':
                    message_data = data.get('message', {})
                elif event_type == 'content_block_start':
                    index = data.get('index', 0)
                    content_blocks[index] = data.get('content_block', {}).copy()
                    current_index = index
                elif event_type == 'content_block_delta':
                    index = data.get('index', 0)
                    delta = data.get('delta', {})

                    if index not in content_blocks:
                        content_blocks[index] = {}

                    if delta.get('type') == 'text_delta':
                        existing_text = content_blocks[index].get('text', '')
                        content_blocks[index]['text'] = existing_text + delta.get('text', '')
                    elif delta.get('type') == 'input_json_delta':
                        existing_json = content_blocks[index].get('_partial_json', '')
                        content_blocks[index]['_partial_json'] = existing_json + delta.get('partial_json', '')
                elif event_type == 'message_stop':
                    # Build final response
                    if message_data:
                        content = []
                        for idx in sorted(content_blocks.keys()):
                            block = content_blocks[idx]
                            block_type = block.get('type', 'text')
                            if block_type == 'tool_use':
                                partial_json = block.pop('_partial_json', '')
                                if partial_json:
                                    try:
                                        block['input'] = json.loads(partial_json)
                                    except json.JSONDecodeError:
                                        block['input'] = {}
                                content.append(block)
                            else:
                                content.append(block)

                        message_data['content'] = content
                        return message_data

            except json.JSONDecodeError:
                continue

    return None


def parse_openai_sse_to_response(sse_text: str) -> Optional[dict]:
    """Parse OpenAI SSE stream text to reconstruct the full response dict."""
    content_parts = []
    tool_calls = {}  # index -> {id, name, arguments}
    finish_reason = None
    response_id = None
    model = None
    usage = None

    for line in sse_text.split('\n'):
        line = line.strip()
        if not line:
            continue

        if line.startswith('data: '):
            data_str = line[6:]
            if data_str == '[DONE]':
                continue

            try:
                data = json.loads(data_str)

                # Extract metadata from first chunk
                if response_id is None:
                    response_id = data.get('id')
                    model = data.get('model')

                # Extract usage if present
                if 'usage' in data:
                    usage = data['usage']

                choices = data.get('choices', [])
                if choices:
                    choice = choices[0]
                    delta = choice.get('delta', {})
                    finish_reason = choice.get('finish_reason') or finish_reason

                    # Handle text content
                    if 'content' in delta and delta['content']:
                        content_parts.append(delta['content'])

                    # Handle tool calls
                    if 'tool_calls' in delta:
                        for tc in delta['tool_calls']:
                            idx = tc.get('index', 0)
                            if idx not in tool_calls:
                                tool_calls[idx] = {'id': '', 'name': '', 'arguments': ''}

                            if 'id' in tc:
                                tool_calls[idx]['id'] = tc['id']
                            if 'function' in tc:
                                if 'name' in tc['function']:
                                    tool_calls[idx]['name'] = tc['function']['name']
                                if 'arguments' in tc['function']:
                                    tool_calls[idx]['arguments'] += tc['function']['arguments']

            except json.JSONDecodeError:
                continue

    # Build final OpenAI response
    if response_id is None:
        return None

    # Build message content
    message = {'role': 'assistant'}
    if content_parts:
        message['content'] = ''.join(content_parts)
    else:
        message['content'] = ''

    if tool_calls:
        message['tool_calls'] = []
        for idx in sorted(tool_calls.keys()):
            tc = tool_calls[idx]
            message['tool_calls'].append({
                'id': tc['id'] or f'call_{idx}',
                'type': 'function',
                'function': {
                    'name': tc['name'],
                    'arguments': tc['arguments']
                }
            })

    response = {
        'id': response_id,
        'object': 'chat.completion',
        'created': 0,
        'model': model or '',
        'choices': [{
            'index': 0,
            'message': message,
            'finish_reason': finish_reason or 'stop'
        }]
    }

    if usage:
        response['usage'] = usage

    return response


def convert_openai_sse_to_anthropic_chunks(sse_text: str) -> list:
    """Convert OpenAI SSE chunks to Anthropic SSE chunks for streaming."""
    anthropic_chunks = []
    content_block_index = 0
    has_tool_calls = False

    # First, collect all chunks to determine structure
    openai_chunks = []
    for line in sse_text.split('\n'):
        line = line.strip()
        if line.startswith('data: ') and line[6:] != '[DONE]':
            try:
                openai_chunks.append(json.loads(line[6:]))
            except json.JSONDecodeError:
                pass

    # Generate message_start
    if openai_chunks:
        first_chunk = openai_chunks[0]
        anthropic_chunks.append({
            'type': 'message_start',
            'message': {
                'id': first_chunk.get('id', 'msg_unknown'),
                'type': 'message',
                'role': 'assistant',
                'content': [],
                'model': first_chunk.get('model', ''),
                'stop_reason': None,
                'usage': {'input_tokens': 0, 'output_tokens': 0}
            }
        })

    # Process each chunk
    for data in openai_chunks:
        choices = data.get('choices', [])
        if not choices:
            continue

        choice = choices[0]
        delta = choice.get('delta', {})

        # Handle text content
        if 'content' in delta and delta['content']:
            if not has_tool_calls:
                # Only emit text deltas if we haven't started tool calls
                anthropic_chunks.append({
                    'type': 'content_block_delta',
                    'index': content_block_index,
                    'delta': {
                        'type': 'text_delta',
                        'text': delta['content']
                    }
                })

        # Handle tool calls
        if 'tool_calls' in delta:
            has_tool_calls = True
            for tc in delta['tool_calls']:
                idx = tc.get('index', 0)
                if 'function' in tc:
                    func = tc['function']
                    if 'name' in func:
                        # Start new tool call block
                        content_block_index = idx
                        anthropic_chunks.append({
                            'type': 'content_block_start',
                            'index': idx,
                            'content_block': {
                                'type': 'tool_use',
                                'id': tc.get('id', f'toolu_{idx}'),
                                'name': func['name'],
                                'input': {}
                            }
                        })
                    elif 'arguments' in func:
                        # Arguments delta
                        anthropic_chunks.append({
                            'type': 'content_block_delta',
                            'index': idx,
                            'delta': {
                                'type': 'input_json_delta',
                                'partial_json': func['arguments']
                            }
                        })

        # Handle finish_reason
        if 'finish_reason' in choice and choice['finish_reason']:
            finish_reason = choice['finish_reason']
            stop_reason_map = {
                'stop': 'end_turn',
                'length': 'max_tokens',
                'tool_calls': 'tool_use',
                'content_filter': 'stop_sequence',
                'function_call': 'tool_use'
            }
            stop_reason = stop_reason_map.get(finish_reason, 'end_turn')

            # Add usage if present
            usage_data = None
            if 'usage' in data:
                usage_data = {
                    'input_tokens': data['usage'].get('prompt_tokens', 0),
                    'output_tokens': data['usage'].get('completion_tokens', 0)
                }

            anthropic_chunks.append({
                'type': 'message_delta',
                'delta': {'stop_reason': stop_reason},
                'usage': usage_data or {'output_tokens': 0}
            })
            anthropic_chunks.append({'type': 'message_stop'})

    # Ensure we have message_stop if not added
    if anthropic_chunks and anthropic_chunks[-1].get('type') != 'message_stop':
        anthropic_chunks.append({'type': 'message_stop'})

    return anthropic_chunks

class _StatusStreamingResponse(StreamingResponse):
    """StreamingResponse that allows the async generator to override the status code before any bytes are sent."""
    def __init__(self, content, status_holder: dict, **kwargs):
        super().__init__(content, **kwargs)
        self._status_holder = status_holder

    async def __call__(self, scope, receive, send):
        # Override status code if the generator set it
        if "status_code" in self._status_holder:
            self.status_code = self._status_holder["status_code"]
        await super().__call__(scope, receive, send)


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def proxy_target_requests(path: str, request: Request):
    """
    Catch-all route to proxy all incoming requests to the upstream server.
    For POST requests to configured generation endpoints, it applies
    the sampling parameter override logic.
    Supports streaming responses from the upstream server back to the client.
    """
    # Access ENABLE_DEBUG_LOGS from the global scope
    global ENABLE_DEBUG_LOGS

    # Get request ID at the START for proper log correlation
    request_id = get_request_id()

    if ENABLE_DEBUG_LOGS:
        logger.info(f"\n--- Incoming Request: {request.method} {path} ---")
    # Normalize path by removing leading/trailing slashes for consistent matching
    original_path = path
    path = path.strip('/')
    if ENABLE_DEBUG_LOGS:
        logger.debug(f"Normalized path for matching: '{path}' (Original: '{original_path}')")

    # Handle Anthropic-specific endpoints
    if path in ANTHROPIC_ENDPOINTS:
        if SERVER_SUPPORTS_ANTHROPIC:
            # In Anthropic passthrough mode, proxy these endpoints upstream as-is
            if ENABLE_DEBUG_LOGS:
                logger.debug(f"Passthrough Anthropic endpoint '{path}' to upstream")

            target_path = transform_path("/" + original_path, SAMPLING_PROXY_BASE_PATH, TARGET_BASE_PATH)
            passthrough_headers = dict(request.headers)
            passthrough_headers.pop("host", None)
            passthrough_headers.pop("content-length", None)
            passthrough_body = await request.body()

            # Strip TARGET_BASE_PATH to avoid doubling it (httpx client has base_url set)
            if TARGET_BASE_PATH and target_path.startswith(TARGET_BASE_PATH):
                relative_path = target_path[len(TARGET_BASE_PATH):]
                if relative_path and not relative_path.startswith('/'):
                    relative_path = '/' + relative_path
            else:
                relative_path = target_path

            # Preserve query string from original request
            passthrough_url = httpx.URL(path=relative_path, query=request.url.query.encode("utf-8"))

            try:
                upstream_response = await client.request(
                    method=request.method,
                    url=passthrough_url,
                    headers=passthrough_headers,
                    content=passthrough_body,
                )

                return Response(
                    content=upstream_response.content,
                    status_code=upstream_response.status_code,
                    media_type=upstream_response.headers.get("content-type", "application/json"),
                )
            except Exception as e:
                if ENABLE_DEBUG_LOGS:
                    logger.error(f"Failed to proxy '{path}' upstream: {e}")
                return Response(
                    content=json.dumps({"error": {"type": "api_error", "message": f"Failed to proxy request upstream: {str(e)}"}}),
                    status_code=502,
                    media_type="application/json"
                )
        else:
            # In conversion mode, handle locally
            if ENABLE_DEBUG_LOGS:
                logger.debug(f"Handling Anthropic endpoint '{path}' locally")

            if path == "api/event_logging/batch":
                # Handle event logging endpoint - return success response
                if ENABLE_DEBUG_LOGS:
                    logger.debug("Processing event logging request")

                try:
                    # Read the request body to acknowledge receipt
                    body = await request.body()
                    if ENABLE_DEBUG_LOGS:
                        logger.debug(f"Event logging body received: {len(body)} bytes")

                    # Return a success response that mimics what Anthropic expects
                    response_data = {
                        "status": "success",
                        "message": "Events logged successfully"
                    }

                    return Response(
                        content=json.dumps(response_data),
                        status_code=200,
                        media_type="application/json"
                    )
                except Exception as e:
                    if ENABLE_DEBUG_LOGS:
                        logger.error(f"Error processing event logging: {e}")
                    return Response(
                        content=json.dumps({"error": "Failed to process events"}),
                        status_code=500,
                        media_type="application/json"
                    )

            elif path == "v1/messages/count_tokens":
                # Handle token counting endpoint
                if ENABLE_DEBUG_LOGS:
                    logger.debug("Processing token counting request")

                try:
                    # Read and parse the request body
                    body = await request.body()
                    if ENABLE_DEBUG_LOGS:
                        logger.debug(f"Token counting body received: {len(body)} bytes")

                    request_data = json.loads(body.decode('utf-8'))
                    messages = request_data.get("messages", [])
                    model = request_data.get("model", "claude-3-sonnet-20241022")

                    if ENABLE_DEBUG_LOGS:
                        logger.debug(f"Token counting request - model: {model}, messages: {messages}")

                    # Simple token estimation (rough approximation)
                    # In a real implementation, you might want to use a proper tokenizer
                    total_tokens = 0
                    for message in messages:
                        content = message.get("content", "")
                        if isinstance(content, list):
                            # Handle complex content format
                            for content_item in content:
                                if isinstance(content_item, dict) and content_item.get("type") == "text":
                                    text = content_item.get("text", "")
                                    # Rough estimation: ~4 characters per token for English text
                                    total_tokens += len(text) // 4 + 1
                                elif isinstance(content_item, str):
                                    total_tokens += len(content_item) // 4 + 1
                        elif isinstance(content, str):
                            total_tokens += len(content) // 4 + 1
                        else:
                            total_tokens += len(str(content)) // 4 + 1

                    # Return response in Anthropic format
                    response_data = {
                        "input_tokens": total_tokens
                    }

                    if ENABLE_DEBUG_LOGS:
                        logger.debug(f"Token counting result: {total_tokens} tokens")

                    return Response(
                        content=json.dumps(response_data),
                        status_code=200,
                        media_type="application/json"
                    )
                except json.JSONDecodeError as e:
                    if ENABLE_DEBUG_LOGS:
                        logger.error(f"Invalid JSON in token counting request: {e}")
                    return Response(
                        content=json.dumps({"error": {"type": "invalid_request_error", "message": "Invalid JSON"}}),
                        status_code=400,
                        media_type="application/json"
                    )
                except Exception as e:
                    if ENABLE_DEBUG_LOGS:
                        logger.error(f"Error processing token counting: {e}")
                    return Response(
                        content=json.dumps({"error": {"type": "api_error", "message": "Failed to count tokens"}}),
                        status_code=500,
                        media_type="application/json"
                    )

            # For any other Anthropic endpoints, return a generic success
            return Response(
                content=json.dumps({"status": "ok"}),
                status_code=200,
                media_type="application/json"
            )

    # Prepare headers for the outgoing request to upstream server.
    # We copy the incoming headers and remove 'host' and 'content-length'
    # as httpx will manage these for the new request.
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None) # httpx will recalculate if body changes
    if ENABLE_DEBUG_LOGS:
        logger.debug(f"Outgoing Request Headers (initial): {headers}")

    request_content = None # This will hold the request body to be sent to target
    is_generation_request = False
    is_anthropic_request = False # Initialize Anthropic request flag
    incoming_json_body = {} # Initialize in case it's not a POST/JSON request

    # Determine if the current request path is a recognized generation endpoint
    # Use suffix matching to handle paths with or without v1 prefix
    is_generation_request = any(path.endswith(suffix) for suffix in GENERATION_ENDPOINT_SUFFIXES)
    is_anthropic_request = path.endswith("v1/messages") # Check if this is an Anthropic request
    if ENABLE_DEBUG_LOGS:
        logger.debug(f"is_generation_request after check: {is_generation_request}")
        logger.debug(f"is_anthropic_request: {is_anthropic_request}")

    # Construct the target URL based on server capabilities
    # Determine passthrough mode based on request format and server capabilities
    should_passthrough_anthropic = is_anthropic_request and SERVER_SUPPORTS_ANTHROPIC
    should_passthrough_openai = not is_anthropic_request and SERVER_SUPPORTS_OPENAI

    if is_anthropic_request:
        if should_passthrough_anthropic:
            # Keep Anthropic path as-is, no conversion
            target_path = transform_path("/" + original_path, SAMPLING_PROXY_BASE_PATH, TARGET_BASE_PATH)
            if ENABLE_DEBUG_LOGS:
                logger.debug(f"Anthropic passthrough mode - keeping path: {target_path}")
        else:
            # Convert /v1/messages to /chat/completions for OpenAI-compatible upstream server
            # First apply the path transformation, then change to chat completions
            transformed_path = transform_path("/" + original_path, SAMPLING_PROXY_BASE_PATH, TARGET_BASE_PATH)
            target_path = transformed_path.replace("/v1/messages", "/chat/completions", 1)
            if ENABLE_DEBUG_LOGS:
                logger.debug(f"Converting Anthropic request from {original_path} to {target_path}")
    else:
        if not SERVER_SUPPORTS_OPENAI:
            # OpenAI request but server doesn't support OpenAI - we can't convert OpenAI to Anthropic
            return Response(
                content=json.dumps({
                    "error": {
                        "type": "invalid_request_error",
                        "message": "Server does not support OpenAI format requests. Only Anthropic format is supported."
                    }
                }),
                status_code=400,
                media_type="application/json"
            )
        # Apply base path transformation
        target_path = transform_path("/" + original_path, SAMPLING_PROXY_BASE_PATH, TARGET_BASE_PATH)
    
    if ENABLE_DEBUG_LOGS:
        logger.debug(f"Path transformation: /{original_path} -> {target_path}")
        logger.debug(f"Base paths - Proxy: {SAMPLING_PROXY_BASE_PATH}, Target: {TARGET_BASE_PATH}")
    
    # Since httpx.AsyncClient is created with base_url=TARGET_BASE_URL,
    # we need to provide only the path portion relative to the target base path
    # Strip the TARGET_BASE_PATH from the beginning of target_path if it exists
    if TARGET_BASE_PATH and target_path.startswith(TARGET_BASE_PATH):
        relative_path = target_path[len(TARGET_BASE_PATH):]
        # Ensure the relative path starts with / if it's not empty
        if relative_path and not relative_path.startswith('/'):
            relative_path = '/' + relative_path
    else:
        relative_path = target_path
    
    if ENABLE_DEBUG_LOGS:
        logger.debug(f"Relative path for httpx: {relative_path}")
    
    # Ensure the query string is encoded to bytes as required by httpx.URL
    target_url = httpx.URL(path=relative_path, query=request.url.query.encode("utf-8"))
    if ENABLE_DEBUG_LOGS:
        logger.debug(f"Target upstream URL: {target_url}")

    # --- Sampling Parameter Override Logic ---
    if is_generation_request and request.method == "POST":
        if ENABLE_DEBUG_LOGS:
            logger.debug("This is a POST generation request. Applying override logic.")
        try:
            # Attempt to parse the incoming request body as JSON.
            # Generation requests typically send JSON payloads.
            raw_body = await request.body()
            if ENABLE_DEBUG_LOGS:
                logger.debug(f"Raw incoming request body: {raw_body.decode('utf-8')}")
            incoming_json_body = json.loads(raw_body) # This will be available for response processing
            if ENABLE_DEBUG_LOGS:
                logger.debug(f"Parsed incoming JSON body: {incoming_json_body}")

            # Handle Anthropic request based on server capabilities
            if is_anthropic_request:
                if should_passthrough_anthropic:
                    # Passthrough mode: keep request as-is, but apply sampling params
                    if ENABLE_DEBUG_LOGS:
                        logger.debug("Anthropic passthrough mode - keeping request format")
                    # Don't modify incoming_json_body - it stays as Anthropic format
                else:
                    # Convert Anthropic to OpenAI format
                    if ENABLE_DEBUG_LOGS:
                        logger.debug("Converting Anthropic request to OpenAI format.")

                    try:

                        # Extract Anthropic format data
                        anthropic_messages = incoming_json_body.get("messages", [])
                        anthropic_model = incoming_json_body.get("model")
                        anthropic_system = incoming_json_body.get("system")
                        anthropic_max_tokens = incoming_json_body.get("max_tokens")
                        anthropic_temperature = incoming_json_body.get("temperature")
                        anthropic_top_p = incoming_json_body.get("top_p")
                        anthropic_stream = incoming_json_body.get("stream", False)
                        anthropic_tools = incoming_json_body.get("tools")
                        anthropic_tool_choice = incoming_json_body.get("tool_choice")
                    
                        # Convert Anthropic messages to OpenAI format
                        openai_messages = []
                        if anthropic_system:
                            system_text = anthropic_system if isinstance(anthropic_system, str) else "\n".join(
                                block.get("text", "") for block in anthropic_system if isinstance(block, dict) and block.get("type") == "text"
                            )
                            if system_text:
                                openai_messages.append({"role": "system", "content": system_text})
                                if ENABLE_DEBUG_LOGS:
                                    logger.debug(f"Converted Anthropic top-level system prompt to OpenAI system message ({len(system_text)} chars)")
                        for msg_idx, msg in enumerate(anthropic_messages):
                            try:
                                # Map Anthropic roles to OpenAI roles
                                anthropic_role = msg.get("role", "user")
                                if anthropic_role == "user":
                                    openai_role = "user"
                                elif anthropic_role == "assistant":
                                    openai_role = "assistant"
                                elif anthropic_role == "system":
                                    openai_role = "system"
                                else:
                                    # Default to user for unknown roles
                                    openai_role = "user"
                                    if ENABLE_DEBUG_LOGS:
                                        logger.debug(f"Unknown Anthropic role '{anthropic_role}' mapped to 'user'")
                            
                                openai_msg = {
                                    "role": openai_role,
                                    "content": ""  # Initialize with empty string instead of None
                                }
                            
                                # Handle complex Anthropic content format
                                content = msg.get("content", [])
                                if isinstance(content, list):
                                    content_parts = []
                                    tool_calls = []
                                
                                    for content_item in content:
                                        if isinstance(content_item, dict):
                                            content_type = content_item.get("type")
                                        
                                            if content_type == "text":
                                                text_content = content_item.get("text", "")
                                                if text_content:
                                                    content_parts.append(text_content)
                                        
                                            elif content_type == "tool_use":
                                                # Convert Anthropic tool_use to OpenAI tool_call format
                                                tool_call = {
                                                    "id": content_item.get("id", f"call_{len(tool_calls)}"),
                                                    "type": "function",
                                                    "function": {
                                                        "name": content_item.get("name", ""),
                                                        "arguments": json.dumps(content_item.get("input", {}))
                                                    }
                                                }
                                                tool_calls.append(tool_call)
                                                if ENABLE_DEBUG_LOGS:
                                                    logger.debug(f"Converted Anthropic tool_use to OpenAI tool_call: {tool_call}")
                                        
                                            elif content_type == "tool_result":
                                                # Convert Anthropic tool_result to OpenAI tool call format
                                                tool_result_id = content_item.get("tool_use_id")
                                                result_content = content_item.get("content", "")
                                                is_error = content_item.get("is_error", False)
                                            
                                                # Create a tool_call message with the result
                                                if tool_result_id:
                                                    tool_call_msg = {
                                                        "role": "tool",
                                                        "tool_call_id": tool_result_id,
                                                        "content": str(result_content) if result_content else "No content"
                                                    }
                                                    if is_error:
                                                        tool_call_msg["content"] = f"Error: {result_content}"
                                                
                                                    openai_messages.append(tool_call_msg)
                                                    if ENABLE_DEBUG_LOGS:
                                                        logger.debug(f"Converted Anthropic tool_result to OpenAI tool message: {tool_call_msg}")
                                    
                                        elif isinstance(content_item, str):
                                            content_parts.append(content_item)
                                
                                    # Set content and tool_calls for the main message
                                    if content_parts:
                                        openai_msg["content"] = "".join(content_parts)
                                    else:
                                        # If no content parts but there are tool calls, set content to null
                                        # Otherwise set to empty string
                                        openai_msg["content"] = None if tool_calls else ""
                                
                                    if tool_calls:
                                        openai_msg["tool_calls"] = tool_calls
                            
                                elif isinstance(content, str):
                                    openai_msg["content"] = content if content else ""
                                elif content is None:
                                    openai_msg["content"] = ""
                                else:
                                    openai_msg["content"] = str(content)
                            
                                # Validate the message before adding
                                if openai_msg.get("role") != "tool" or "tool_call_id" in openai_msg:
                                    # Ensure content is never None for non-tool messages
                                    if openai_msg.get("content") is None and not openai_msg.get("tool_calls"):
                                        openai_msg["content"] = ""
                                
                                    # Only add if the message has valid content or tool calls
                                    if openai_msg.get("content") or openai_msg.get("tool_calls"):
                                        openai_messages.append(openai_msg)
                                        if ENABLE_DEBUG_LOGS:
                                            logger.debug(f"Converted message {msg_idx}: {openai_msg}")
                                    else:
                                        if ENABLE_DEBUG_LOGS:
                                            logger.debug(f"Skipping empty message {msg_idx}")
                                else:
                                    if ENABLE_DEBUG_LOGS:
                                        logger.debug(f"Skipping invalid tool message {msg_idx}")
                        
                            except Exception as e:
                                if ENABLE_DEBUG_LOGS:
                                    logger.error(f"Failed to convert message {msg_idx}: {e}")
                                # Continue with next message instead of failing completely
                                continue
                    
                        # Override model for Anthropic requests
                        if OVERRIDE_MODEL_NAME:
                            overridden_model = OVERRIDE_MODEL_NAME
                            if ENABLE_OVERRIDE_LOGS:
                                logger.info(f"OVERRIDE: Anthropic model '{anthropic_model}' OVERRIDDEN to '{OVERRIDE_MODEL_NAME}'")
                        else:
                            overridden_model = FIRST_AVAILABLE_MODEL if FIRST_AVAILABLE_MODEL else anthropic_model
                            if ENABLE_DEBUG_LOGS and FIRST_AVAILABLE_MODEL:
                                logger.debug(f"Using first available model '{FIRST_AVAILABLE_MODEL}' for Anthropic request")
                    
                        # Convert to OpenAI chat completions format
                        openai_request = {
                            "model": overridden_model,
                            "messages": openai_messages,
                            "max_tokens": anthropic_max_tokens,
                            "stream": anthropic_stream
                        }
                    
                        # Add optional parameters if present
                        if anthropic_temperature is not None:
                            openai_request["temperature"] = anthropic_temperature
                        if anthropic_top_p is not None:
                            openai_request["top_p"] = anthropic_top_p
                    
                        # Convert stop_sequences to stop
                        anthropic_stop_sequences = incoming_json_body.get("stop_sequences")
                        if anthropic_stop_sequences:
                            openai_request["stop"] = anthropic_stop_sequences

                        # Convert tools if present
                        if anthropic_tools:
                            openai_tools = []
                            for tool in anthropic_tools:
                                openai_tool = {
                                    "type": "function",
                                    "function": {
                                        "name": tool.get("name"),
                                        "description": tool.get("description", ""),
                                        "parameters": tool.get("input_schema", {})
                                    }
                                }
                                openai_tools.append(openai_tool)
                        
                            openai_request["tools"] = openai_tools
                            if ENABLE_DEBUG_LOGS:
                                logger.debug(f"Converted {len(anthropic_tools)} Anthropic tools to OpenAI format")
                    
                        # Convert tool_choice if present
                        if anthropic_tool_choice:
                            if anthropic_tool_choice == "auto":
                                openai_request["tool_choice"] = "auto"
                            elif anthropic_tool_choice == "any":
                                openai_request["tool_choice"] = "required"
                            elif anthropic_tool_choice == "none":
                                openai_request["tool_choice"] = "none"
                            elif isinstance(anthropic_tool_choice, dict):
                                tool_name = anthropic_tool_choice.get("name")
                                if tool_name:
                                    openai_request["tool_choice"] = {"type": "function", "function": {"name": tool_name}}
                            if ENABLE_DEBUG_LOGS:
                                logger.debug(f"Converted tool_choice: {anthropic_tool_choice} -> {openai_request.get('tool_choice')}")
                
                        # Validate the converted messages before proceeding
                        if not openai_messages:
                            if ENABLE_DEBUG_LOGS:
                                logger.error("No valid messages after conversion. Creating fallback message.")
                            # Create a simple fallback message
                            openai_messages = [{
                                "role": "user",
                                "content": "Please provide a response."
                            }]
                    
                        # Replace the incoming body with converted OpenAI format
                        incoming_json_body = openai_request
                        if ENABLE_DEBUG_LOGS:
                            logger.debug(f"Converted to OpenAI format: {incoming_json_body}")
                            logger.debug(f"Final message count: {len(openai_messages)}")
                
                    except Exception as e:
                        logger.error(f"Failed to convert Anthropic request to OpenAI format: {e}")
                        if ENABLE_DEBUG_LOGS:
                            logger.error(f"Original Anthropic request: {incoming_json_body}")
                    
                        # Create a minimal valid OpenAI request as fallback
                        fallback_model = OVERRIDE_MODEL_NAME if OVERRIDE_MODEL_NAME else (FIRST_AVAILABLE_MODEL if FIRST_AVAILABLE_MODEL else "gpt-3.5-turbo")
                        incoming_json_body = {
                            "model": fallback_model,
                            "messages": [{"role": "user", "content": "Conversion failed. Please respond."}],
                            "max_tokens": anthropic_max_tokens if 'anthropic_max_tokens' in locals() else 1000,
                            "stream": anthropic_stream if 'anthropic_stream' in locals() else False
                        }
                    
                        if ENABLE_DEBUG_LOGS:
                            logger.debug(f"Using fallback OpenAI request: {incoming_json_body}")

            # Apply model name override for Anthropic requests in passthrough mode
            if is_anthropic_request and should_passthrough_anthropic:
                if OVERRIDE_MODEL_NAME:
                    original_model_name = incoming_json_body.get("model")
                    incoming_json_body["model"] = OVERRIDE_MODEL_NAME
                    if ENABLE_OVERRIDE_LOGS:
                        logger.info(f"OVERRIDE: Anthropic passthrough model '{original_model_name}' OVERRIDDEN to '{OVERRIDE_MODEL_NAME}'")

            # Get the model name from the request
            model_name = incoming_json_body.get("model")
            if ENABLE_DEBUG_LOGS:
                logger.debug(f"Model name from request: {model_name}")
            
            # Apply model name override for non-Anthropic requests when applicable
            if not is_anthropic_request:
                # If only_anthropic is false, apply model name override to all non-Anthropic requests
                if not OVERRIDE_ONLY_ANTHROPIC and OVERRIDE_MODEL_NAME:
                    original_model_name = model_name
                    model_name = OVERRIDE_MODEL_NAME
                    incoming_json_body["model"] = model_name
                    if ENABLE_OVERRIDE_LOGS:
                        logger.info(f"OVERRIDE: Non-Anthropic model '{original_model_name}' OVERRIDDEN to '{OVERRIDE_MODEL_NAME}'")

            # Normalize message roles: convert 'developer' to 'system' for local servers
            # that don't recognize the newer OpenAI 'developer' role (e.g. vLLM, SGLang)
            messages = incoming_json_body.get("messages")
            if isinstance(messages, list):
                for msg in messages:
                    if isinstance(msg, dict) and msg.get("role") == "developer":
                        msg["role"] = "system"
                        if ENABLE_DEBUG_LOGS:
                            logger.debug("Converted message role 'developer' -> 'system'")

            # Determine where sampling parameters are expected in the request body
            # For /generate, they are typically in a 'sampling_params' sub-dictionary
            # For OpenAI-compatible endpoints, they are typically top-level keys
            if path == "generate": # Use normalized path here
                current_params_container = incoming_json_body.get("sampling_params", {})
                is_nested_params = True
                if ENABLE_DEBUG_LOGS:
                    logger.debug(f"Path is 'generate', using nested 'sampling_params'. Current container: {current_params_container}")
            else: # completions, chat/completions, v1/messages (normalized paths)
                current_params_container = incoming_json_body
                is_nested_params = False
                if ENABLE_DEBUG_LOGS:
                    logger.debug(f"Path is OpenAI-compatible, using top-level params. Current container: {current_params_container}")

            model_specific_params = MODEL_SAMPLING_PARAMS.get(model_name, {})
            if ENABLE_DEBUG_LOGS:
                logger.debug(f"Model-specific params for '{model_name}': {model_specific_params}")

            # First, apply override parameters - these override incoming parameters based on only_anthropic flag
            if OVERRIDE_SAMPLING_PARAMS:
                # Check if we should apply overrides (only_anthropic=false OR this is an Anthropic request)
                should_apply_overrides = not OVERRIDE_ONLY_ANTHROPIC or is_anthropic_request
                
                if should_apply_overrides:
                    if ENABLE_OVERRIDE_LOGS:
                        logger.info(f"OVERRIDE: Applying sampling parameter overrides: {OVERRIDE_SAMPLING_PARAMS}")
                        if OVERRIDE_ONLY_ANTHROPIC:
                            logger.info(f"OVERRIDE: Overrides only applied to Anthropic requests (is_anthropic_request={is_anthropic_request})")
                    for param, override_value in OVERRIDE_SAMPLING_PARAMS.items():
                        original_value = current_params_container.get(param, "not_set")
                        current_params_container[param] = override_value
                        if ENABLE_OVERRIDE_LOGS:
                            logger.info(f"OVERRIDE: '{param}' from '{original_value}' to '{override_value}'")

            # Then, apply model-specific default parameters (same logic as default_sampling_params, but per-model)
            if model_specific_params:
                for param, model_default_value in model_specific_params.items():
                    if param not in current_params_container:
                        # Skip if this parameter is being overridden (already handled above)
                        if OVERRIDE_SAMPLING_PARAMS and param in OVERRIDE_SAMPLING_PARAMS:
                            if ENABLE_DEBUG_LOGS:
                                logger.debug(f"Parameter '{param}' is overridden, skipping model-specific default application.")
                            continue
                        current_params_container[param] = model_default_value
                        if ENABLE_OVERRIDE_LOGS:
                            logger.debug(f"[model:{model_name}] Overriding '{param}' to '{model_default_value}' (was not in request).")
                    else:
                        if ENABLE_OVERRIDE_LOGS:
                            logger.debug(f"[model:{model_name}] Parameter '{param}' already present in request: {current_params_container[param]}. Not overriding.")

            # Then, apply global default parameters for any missing parameters not covered by model-specific or overrides
            for param, default_value in DEFAULT_SAMPLING_PARAMS.items():
                if param not in current_params_container:
                    # Skip if this parameter is being overridden (already handled above)
                    if OVERRIDE_SAMPLING_PARAMS and param in OVERRIDE_SAMPLING_PARAMS:
                        if ENABLE_DEBUG_LOGS:
                            logger.debug(f"Parameter '{param}' is overridden, skipping default application.")
                        continue
                    current_params_container[param] = default_value
                    if ENABLE_OVERRIDE_LOGS:
                        logger.debug(f"Overriding '{param}' to '{default_value}' (was not in request).")
                else:
                    if ENABLE_OVERRIDE_LOGS:
                        logger.debug(f"Parameter '{param}' already present in request: {current_params_container[param]}. Not overriding.")

            # Re-integrate the modified parameters back into the main body if they were nested
            if is_nested_params:
                incoming_json_body["sampling_params"] = current_params_container
            # If not nested, they are already updated in incoming_json_body

            request_content = json.dumps(incoming_json_body) # Serialize the modified JSON back to a string
            headers["content-type"] = "application/json" # Ensure content-type header is correct

            if ENABLE_DEBUG_LOGS:
                logger.debug(f"Final modified request body: {request_content}")
                logger.info(f"[{request.method} {original_path}] Overridden sampling params for model '{model_name}': {current_params_container}")

        except json.JSONDecodeError as e:
            logger.error(f"[{request.method} {original_path}] JSONDecodeError: {e}. Proxying raw body.")
            request_content = await request.body()
        except Exception as e:
            logger.error(f"[{request.method} {original_path}] Error processing generation request body: {e}. Proxying raw body.")
            request_content = await request.body()
    else:
        if ENABLE_DEBUG_LOGS:
            logger.debug(f"Not a POST generation request (is_generation_request={is_generation_request}, method={request.method}). Proxying raw body without modification.")
        request_content = await request.body()

    # --- Parallel Limit Semaphores ---
    # Acquire both global and model-specific semaphores if configured.
    # Semaphores are released after the response is fully sent (streaming or not).
    global_semaphore = None
    model_semaphore = None
    if is_generation_request and request.method == "POST" and model_name:
        # Acquire global semaphore first (if configured)
        global_semaphore = get_global_semaphore()
        if global_semaphore is not None:
            if global_semaphore._value == 0:
                waiting = len(global_semaphore._waiters) if global_semaphore._waiters else 0
                log_info(request_id, f"Queueing for global limit, {waiting} requests waiting (limit: {GLOBAL_LIMIT})")
            await global_semaphore.acquire()
            used = GLOBAL_LIMIT - global_semaphore._value
            log_info(request_id, f"Global slot acquired, used: {used}/{GLOBAL_LIMIT}")
        
        # Then acquire model-specific semaphore (if configured)
        model_semaphore = get_model_semaphore(model_name)
        if model_semaphore is not None:
            limit = PARALLEL_LIMITS.get(model_name.lower())
            if model_semaphore._value == 0:
                waiting = len(model_semaphore._waiters) if model_semaphore._waiters else 0
                log_info(request_id, f"Queueing for {model_name}, {waiting} requests waiting (limit: {limit})")
            await model_semaphore.acquire()
            used = limit - model_semaphore._value
            log_info(request_id, f"Slot acquired {model_name}, used: {used}/{limit}")

    # Helper to release semaphores after streaming completes
    semaphores_released = False
    is_streaming_response = False

    async def release_semaphores():
        nonlocal semaphores_released
        if semaphores_released:
            return
        semaphores_released = True
        # Release in reverse order of acquisition (model first, then global)
        if model_semaphore is not None:
            model_semaphore.release()
            limit = PARALLEL_LIMITS.get(model_name.lower())
            used = limit - model_semaphore._value
            log_info(request_id, f"Slot released {model_name}, used: {used}/{limit}")
        if global_semaphore is not None:
            global_semaphore.release()
            used = GLOBAL_LIMIT - global_semaphore._value
            log_info(request_id, f"Global slot released, used: {used}/{GLOBAL_LIMIT}")

    def wrap_stream_with_semaphore_release(generator):
        """Wrap an async generator to release semaphores when streaming completes."""
        nonlocal is_streaming_response
        is_streaming_response = True
        async def wrapped():
            try:
                async for chunk in generator:
                    yield chunk
            finally:
                await release_semaphores()
        return wrapped()

    # --- Forward Request and Handle Response ---
    try:
        if is_generation_request and request.method == "POST":
            # Check if this is actually a streaming request
            is_streaming_request = incoming_json_body.get("stream", False)

            # Log request start with attempt info
            max_retries = VALIDATION_CONFIG.get("max_retries", 3)
            max_attempts = 1 + max_retries
            log_info(request_id, f"Request started")

            if ENABLE_DEBUG_LOGS:
                logger.debug(f"Sending {'streaming' if is_streaming_request else 'non-streaming'} request.")

            if is_streaming_request:
                # For streaming requests, use streaming
                target_request_obj = client.build_request(
                    method=request.method,
                    url=target_url,
                    headers=headers,
                    params=request.query_params,
                    content=request_content,
                )
                # Send the request and get the raw response object, enabling streaming
                if throttle_manager:
                    await throttle_manager.wait_before_send(model_name, request_id)
                target_response = await client.send(target_request_obj, stream=True)
            else:
                # For non-streaming requests, fetch the full response
                if throttle_manager:
                    await throttle_manager.wait_before_send(model_name, request_id)
                target_response = await client.request(
                    method=request.method,
                    url=target_url,
                    headers=headers,
                    params=request.query_params,
                    content=request_content,
                )

            if is_streaming_request:
                # Handle streaming response
                # Prepare response headers for streaming
                response_headers = dict(target_response.headers)
                if ENABLE_DEBUG_LOGS:
                    logger.debug(f"Upstream Response Headers (raw): {response_headers}")

                # Remove headers that interfere with streaming
                # Use case-insensitive removal to catch all variants
                # Note: content-encoding must be removed because httpx automatically decompresses
                # responses when using aiter_bytes(), so we must not pass this header to clients
                headers_to_remove = ["content-length", "transfer-encoding", "connection", "content-encoding"]
                for header in headers_to_remove:
                    response_headers.pop(header, None)
                    response_headers.pop(header.upper(), None)
                    response_headers.pop(header.lower(), None)

                # Explicitly set Content-Type for SSE if it's a streaming chat/completion request
                # Use original_path for this check, as it's the actual path in the request
                if original_path.strip('/') in ["chat/completions", "completions", "v1/messages"]:
                    response_headers["content-type"] = "text/event-stream"
                    response_headers["cache-control"] = "no-cache"
                    response_headers["connection"] = "keep-alive"
                    if ENABLE_DEBUG_LOGS:
                        logger.debug("Setting response Content-Type to 'text/event-stream' for streaming request.")
                else:
                    if ENABLE_DEBUG_LOGS:
                        logger.debug(f"Not an OpenAI-compatible streaming path, keeping original Content-Type: {response_headers.get('content-type', 'N/A')}")

                # For streaming with validation in passthrough mode, buffer and validate first
                # Note: openai_convert streaming validation not yet supported (chunks are converted on-the-fly)
                if (is_anthropic_request and
                    should_passthrough_anthropic and
                    VALIDATION_CONFIG.get("enabled", False)):

                    # Capture outer scope variable to avoid UnboundLocalError
                    initial_response = target_response
                    stream_status_holder = {}

                    async def buffered_stream_with_validation():
                        nonlocal initial_response
                        max_retries = VALIDATION_CONFIG.get("max_retries", 3)
                        max_attempts = 1 + max_retries
                        attempt = 0
                        current_response = initial_response

                        while attempt < max_attempts:
                            attempt += 1

                            # Use unified buffer with mid-stream validation
                            buffer = StreamingValidationBuffer(VALIDATION_CONFIG, ENABLE_DEBUG_LOGS)

                            # Stream with immediate garbage detection interrupt
                            chunk_iterator = current_response.aiter_bytes()
                            garbage_event = buffer.get_garbage_event()
                            stream_done = False
                            garbage_interrupted = False

                            while not stream_done:
                                # Create task for reading next chunk
                                read_task = asyncio.create_task(chunk_iterator.__anext__())
                                # Create task for waiting on garbage event
                                garbage_wait_task = asyncio.create_task(garbage_event.wait())

                                done, pending = await asyncio.wait(
                                    [read_task, garbage_wait_task],
                                    return_when=asyncio.FIRST_COMPLETED
                                )

                                # Cancel pending tasks
                                for task in pending:
                                    task.cancel()
                                    try:
                                        await task
                                    except asyncio.CancelledError:
                                        pass

                                if garbage_wait_task in done:
                                    # Garbage detected - close connection immediately to stop upstream
                                    garbage_interrupted = True
                                    await current_response.aclose()
                                    break

                                if read_task in done:
                                    try:
                                        chunk = read_task.result()
                                        if not await buffer.add_chunk(chunk):
                                            # Garbage detected in add_chunk - close immediately
                                            garbage_interrupted = True
                                            await current_response.aclose()
                                            break
                                    except StopAsyncIteration:
                                        stream_done = True  # Stream ended normally
                                    except Exception:
                                        break  # Other error

                            # Only wait for validation if not already interrupted
                            if not garbage_interrupted:
                                await buffer.wait_for_pending_validation()
                                await current_response.aclose()

                            # Handle mid-stream garbage detection
                            if buffer.is_garbage_detected():
                                detection_info = buffer.get_detection_info()
                                issue_type = buffer.get_issue_type() or "unknown"
                                confidence = buffer.get_detection_confidence()
                                text_content = buffer.get_text_content()
                                word_count = buffer.get_word_count()

                                # Save the failed partial response (every attempt)
                                saved_path = save_mid_stream_failure(
                                    text_content, word_count, issue_type, attempt, buffer.get_chunks()
                                )
                                log_info(request_id, f"Validation failed (confidence: {confidence:.2f}, issue: {issue_type}) see: {saved_path}")

                                if attempt >= max_attempts:
                                    # Return error message as stream with proper error status code
                                    stream_status_holder["status_code"] = 529
                                    error_response = create_error_message(issue_type, saved_path)
                                    for event in build_anthropic_error_stream(error_response):
                                        yield event
                                    # Max attempts reached
                                    return

                                # Retry with backoff
                                delay = await calculate_retry_delay(attempt, VALIDATION_CONFIG)
                                if delay > 0:
                                    await asyncio.sleep(delay)

                                # Use streaming for retry (same as initial request)
                                if throttle_manager:
                                    await throttle_manager.wait_before_send(model_name, request_id)
                                retry_request_obj = client.build_request(
                                    method=request.method,
                                    url=target_url,
                                    headers=headers,
                                    params=request.query_params,
                                    content=request_content,
                                )
                                retry_response = await client.send(retry_request_obj, stream=True)
                                if retry_response.status_code == 200:
                                    log_info(request_id, f"Retry attempt {attempt}/{max_attempts}")
                                    current_response = retry_response
                                    continue
                                else:
                                    log_info(request_id, f"WARNING: Retry attempt {attempt}/{max_attempts} failed with status {retry_response.status_code}")
                                    await retry_response.aclose()
                                    break

                            # Validation - use same text extraction as mid-stream for consistency
                            chunks = buffer.get_chunks()
                            try:
                                # Extract text using same method as mid-stream validation
                                text_content = extract_text_from_sse_chunks(chunks)
                                word_count = count_words_in_text(text_content)

                                if text_content.strip():
                                    # Validation using same validator as mid-stream
                                    log_info(request_id, f"Validation started ({word_count} words)")
                                    validation_result = await validate_response_partial(text_content, VALIDATION_CONFIG)

                                    if validation_result.error:
                                        log_info(request_id, f"WARNING: Validator error: {validation_result.error}")
                                        for chunk in chunks:
                                            yield chunk
                                        return

                                    if validation_result.is_valid:
                                        log_info(request_id, f"Validation passed (is_valid={validation_result.is_valid}, issue_type={validation_result.issue_type}, confidence={validation_result.confidence:.2f})")
                                        for chunk in chunks:
                                            yield chunk
                                        return

                                    log_info(request_id, f"Validation failed (is_valid={validation_result.is_valid}, issue_type={validation_result.issue_type}, confidence={validation_result.confidence:.2f})")
                                    # Save failed response using same format as mid-stream
                                    saved_path = save_mid_stream_failure(text_content, word_count, validation_result.issue_type, attempt, chunks)
                                    log_info(request_id, f"Validation failed (confidence: {validation_result.confidence:.2f}, issue: {validation_result.issue_type}) see: {saved_path}")

                                    if attempt >= max_attempts:
                                        stream_status_holder["status_code"] = 529
                                        error_response = create_error_message(validation_result.issue_type, saved_path)
                                        for event in build_anthropic_error_stream(error_response):
                                            yield event
                                        # Max attempts reached, no retry
                                        return

                                    # Retry with backoff
                                    delay = await calculate_retry_delay(attempt, VALIDATION_CONFIG)
                                    if delay > 0:
                                        await asyncio.sleep(delay)

                                    # Use streaming for retry (same as initial request)
                                    if throttle_manager:
                                        await throttle_manager.wait_before_send(model_name, request_id)
                                    retry_request_obj = client.build_request(
                                        method="POST",
                                        url=target_url,
                                        headers=headers,
                                        content=request_content,
                                    )
                                    retry_response = await client.send(retry_request_obj, stream=True)
                                    if retry_response.status_code == 200:
                                        log_info(request_id, f"Retry attempt {attempt}/{max_attempts}")
                                        current_response = retry_response
                                        continue
                                    else:
                                        log_info(request_id, f"WARNING: Retry attempt {attempt}/{max_attempts} failed with status {retry_response.status_code}")
                                        await retry_response.aclose()
                                        for chunk in chunks:
                                            yield chunk
                                        return
                                else:
                                    # Empty content (tool-only response) - pass through without validation
                                    log_info(request_id, "Validation skipped (no text content)")
                                    for chunk in chunks:
                                        yield chunk
                                    return

                            except Exception as e:
                                log_info(request_id, f"ERROR during streaming validation: {e}")
                                for chunk in chunks:
                                    yield chunk
                                return

                    return _StatusStreamingResponse(
                        wrap_stream_with_semaphore_release(buffered_stream_with_validation()),
                        status_holder=stream_status_holder,
                        status_code=target_response.status_code,
                        headers=response_headers,
                        media_type=response_headers.get("content-type"),
                    )

                # For streaming with validation in openai_convert mode, buffer OpenAI chunks,
                # validate (validator handles both formats), then stream converted chunks
                if (is_anthropic_request and
                    not should_passthrough_anthropic and
                    VALIDATION_CONFIG.get("enabled", False)):

                    # Capture outer scope variable to avoid UnboundLocalError
                    initial_openai_response = target_response
                    openai_convert_status_holder = {}

                    async def buffered_openai_stream_with_validation():
                        nonlocal initial_openai_response
                        max_retries = VALIDATION_CONFIG.get("max_retries", 3)
                        max_attempts = 1 + max_retries
                        attempt = 0
                        current_response = initial_openai_response

                        while attempt < max_attempts:
                            attempt += 1

                            # Use unified buffer with mid-stream validation
                            buffer = StreamingValidationBuffer(VALIDATION_CONFIG, ENABLE_DEBUG_LOGS)

                            # Stream with immediate garbage detection interrupt
                            chunk_iterator = current_response.aiter_bytes()
                            garbage_event = buffer.get_garbage_event()
                            stream_done = False
                            garbage_interrupted = False

                            while not stream_done:
                                # Create task for reading next chunk
                                read_task = asyncio.create_task(chunk_iterator.__anext__())
                                # Create task for waiting on garbage event
                                garbage_wait_task = asyncio.create_task(garbage_event.wait())

                                done, pending = await asyncio.wait(
                                    [read_task, garbage_wait_task],
                                    return_when=asyncio.FIRST_COMPLETED
                                )

                                # Cancel pending tasks
                                for task in pending:
                                    task.cancel()
                                    try:
                                        await task
                                    except asyncio.CancelledError:
                                        pass

                                if garbage_wait_task in done:
                                    # Garbage detected - close connection immediately to stop upstream
                                    garbage_interrupted = True
                                    await current_response.aclose()
                                    break

                                if read_task in done:
                                    try:
                                        chunk = read_task.result()
                                        if not await buffer.add_chunk(chunk):
                                            # Garbage detected in add_chunk - close immediately
                                            garbage_interrupted = True
                                            await current_response.aclose()
                                            break
                                    except StopAsyncIteration:
                                        stream_done = True  # Stream ended normally
                                    except Exception:
                                        break  # Other error

                            # Only wait for validation if not already interrupted
                            if not garbage_interrupted:
                                await buffer.wait_for_pending_validation()
                                await current_response.aclose()

                            # Handle mid-stream garbage detection
                            if buffer.is_garbage_detected():
                                detection_info = buffer.get_detection_info()
                                issue_type = buffer.get_issue_type() or "unknown"
                                text_content = buffer.get_text_content()
                                word_count = buffer.get_word_count()

                                # Save the failed partial response (every attempt)
                                saved_path = save_mid_stream_failure(
                                    text_content, word_count, issue_type, attempt, buffer.get_chunks()
                                )
                                log_info(request_id, f"Validation failed (confidence: {buffer.get_detection_confidence():.2f}, issue: {issue_type}) see: {saved_path}")

                                if attempt >= max_attempts:
                                    openai_convert_status_holder["status_code"] = 529
                                    error_response = create_error_message(issue_type, saved_path)
                                    for event in build_anthropic_error_stream(error_response):
                                        yield event
                                    # Max attempts reached
                                    return

                                # Retry with backoff
                                delay = await calculate_retry_delay(attempt, VALIDATION_CONFIG)
                                if delay > 0:
                                    await asyncio.sleep(delay)

                                # Use streaming for retry (same as initial request)
                                if throttle_manager:
                                    await throttle_manager.wait_before_send(model_name, request_id)
                                retry_request_obj = client.build_request(
                                    method=request.method,
                                    url=target_url,
                                    headers=headers,
                                    params=request.query_params,
                                    content=request_content,
                                )
                                retry_response = await client.send(retry_request_obj, stream=True)
                                if retry_response.status_code == 200:
                                    log_info(request_id, f"Retry attempt {attempt}/{max_attempts}")
                                    current_response = retry_response
                                    continue
                                else:
                                    log_info(request_id, f"WARNING: Retry attempt {attempt}/{max_attempts} failed with status {retry_response.status_code}")
                                    await retry_response.aclose()
                                    response_text = buffer.get_content().decode('utf-8')
                                    anthropic_chunks = convert_openai_sse_to_anthropic_chunks(response_text)
                                    for ac in anthropic_chunks:
                                        yield f"event: {ac.get('type', 'message')}\ndata: {json.dumps(ac)}\n\n".encode()
                                    return

                            # Validation - use same text extraction as mid-stream for consistency
                            chunks = buffer.get_chunks()
                            try:
                                response_text = buffer.get_content().decode('utf-8')

                                # Extract text using same method as mid-stream validation
                                text_content = extract_text_from_sse_chunks(chunks)
                                word_count = count_words_in_text(text_content)

                                if text_content.strip():
                                    # Validation using same validator as mid-stream
                                    log_info(request_id, f"Validation started ({word_count} words)")
                                    validation_result = await validate_response_partial(text_content, VALIDATION_CONFIG)

                                    if validation_result.error:
                                        log_info(request_id, f"WARNING: Validator error: {validation_result.error}")
                                        anthropic_chunks = convert_openai_sse_to_anthropic_chunks(response_text)
                                        for ac in anthropic_chunks:
                                            yield f"event: {ac.get('type', 'message')}\ndata: {json.dumps(ac)}\n\n".encode()
                                        return

                                    if validation_result.is_valid:
                                        log_info(request_id, f"Validation passed (is_valid={validation_result.is_valid}, issue_type={validation_result.issue_type}, confidence={validation_result.confidence:.2f})")
                                        anthropic_chunks = convert_openai_sse_to_anthropic_chunks(response_text)
                                        for ac in anthropic_chunks:
                                            yield f"event: {ac.get('type', 'message')}\ndata: {json.dumps(ac)}\n\n".encode()
                                        return

                                    log_info(request_id, f"Validation failed (is_valid={validation_result.is_valid}, issue_type={validation_result.issue_type}, confidence={validation_result.confidence:.2f})")
                                    # Save failed response using same format as mid-stream
                                    saved_path = save_mid_stream_failure(text_content, word_count, validation_result.issue_type, attempt, chunks)
                                    log_info(request_id, f"Validation failed (confidence: {validation_result.confidence:.2f}, issue: {validation_result.issue_type}) see: {saved_path}")

                                    if attempt >= max_attempts:
                                        openai_convert_status_holder["status_code"] = 529
                                        error_response = create_error_message(validation_result.issue_type, saved_path)
                                        for event in build_anthropic_error_stream(error_response):
                                            yield event
                                        # Max attempts reached, no retry
                                        return

                                    # Retry with backoff
                                    delay = await calculate_retry_delay(attempt, VALIDATION_CONFIG)
                                    if delay > 0:
                                        await asyncio.sleep(delay)

                                    # Use streaming for retry (same as initial request)
                                    if throttle_manager:
                                        await throttle_manager.wait_before_send(model_name, request_id)
                                    retry_request_obj = client.build_request(
                                        method="POST",
                                        url=target_url,
                                        headers=headers,
                                        content=request_content,
                                    )
                                    retry_response = await client.send(retry_request_obj, stream=True)
                                    if retry_response.status_code == 200:
                                        log_info(request_id, f"Retry attempt {attempt}/{max_attempts}")
                                        current_response = retry_response
                                        continue
                                    else:
                                        log_info(request_id, f"WARNING: Retry attempt {attempt}/{max_attempts} failed with status {retry_response.status_code}")
                                        await retry_response.aclose()
                                        anthropic_chunks = convert_openai_sse_to_anthropic_chunks(response_text)
                                        for ac in anthropic_chunks:
                                            yield f"event: {ac.get('type', 'message')}\ndata: {json.dumps(ac)}\n\n".encode()
                                        return
                                else:
                                    # Empty content (tool-only response) - convert and pass through without validation
                                    log_info(request_id, "Validation skipped (no text content)")
                                    anthropic_chunks = convert_openai_sse_to_anthropic_chunks(response_text)
                                    for ac in anthropic_chunks:
                                        yield f"event: {ac.get('type', 'message')}\ndata: {json.dumps(ac)}\n\n".encode()
                                    return

                            except Exception as e:
                                log_info(request_id, f"ERROR during OpenAI streaming validation: {e}")
                                for chunk in chunks:
                                    yield chunk
                                return

                    return _StatusStreamingResponse(
                        wrap_stream_with_semaphore_release(buffered_openai_stream_with_validation()),
                        status_holder=openai_convert_status_holder,
                        status_code=target_response.status_code,
                        headers=response_headers,
                        media_type=response_headers.get("content-type"),
                    )

                # For streaming with validation in OpenAI passthrough mode
                if (not is_anthropic_request and
                    should_passthrough_openai and
                    VALIDATION_CONFIG.get("enabled", False)):

                    openai_passthrough_status_holder = {}

                    async def buffered_openai_passthrough_stream_with_validation():
                        max_retries = VALIDATION_CONFIG.get("max_retries", 3)
                        max_attempts = 1 + max_retries

                        attempt = 0
                        current_response = target_response

                        while attempt < max_attempts:
                            attempt += 1

                            # Use unified buffer with mid-stream validation
                            buffer = StreamingValidationBuffer(VALIDATION_CONFIG, ENABLE_DEBUG_LOGS)

                            # Stream with immediate garbage detection interrupt
                            chunk_iterator = current_response.aiter_bytes()
                            garbage_event = buffer.get_garbage_event()
                            stream_done = False
                            garbage_interrupted = False

                            while not stream_done:
                                # Create task for reading next chunk
                                read_task = asyncio.create_task(chunk_iterator.__anext__())
                                # Create task for waiting on garbage event
                                garbage_wait_task = asyncio.create_task(garbage_event.wait())

                                done, pending = await asyncio.wait(
                                    [read_task, garbage_wait_task],
                                    return_when=asyncio.FIRST_COMPLETED
                                )

                                # Cancel pending tasks
                                for task in pending:
                                    task.cancel()
                                    try:
                                        await task
                                    except asyncio.CancelledError:
                                        pass

                                if garbage_wait_task in done:
                                    # Garbage detected - close connection immediately to stop upstream
                                    garbage_interrupted = True
                                    await current_response.aclose()
                                    break

                                if read_task in done:
                                    try:
                                        chunk = read_task.result()
                                        if not await buffer.add_chunk(chunk):
                                            # Garbage detected in add_chunk - close immediately
                                            garbage_interrupted = True
                                            await current_response.aclose()
                                            break
                                    except StopAsyncIteration:
                                        stream_done = True  # Stream ended normally
                                    except Exception:
                                        break  # Other error

                            # Only wait for validation if not already interrupted
                            if not garbage_interrupted:
                                await buffer.wait_for_pending_validation()
                                await current_response.aclose()

                            # Handle mid-stream garbage detection
                            if buffer.is_garbage_detected():
                                detection_info = buffer.get_detection_info()
                                issue_type = buffer.get_issue_type() or "unknown"
                                text_content = buffer.get_text_content()
                                word_count = buffer.get_word_count()

                                # Save the failed partial response (every attempt)
                                saved_path = save_mid_stream_failure(
                                    text_content, word_count, issue_type, attempt, buffer.get_chunks()
                                )
                                log_info(request_id, f"Validation failed (confidence: {buffer.get_detection_confidence():.2f}, issue: {issue_type}) see: {saved_path}")

                                if attempt >= max_attempts:
                                    openai_passthrough_status_holder["status_code"] = 529
                                    error_response = create_error_message(issue_type, saved_path)
                                    for event in build_openai_error_stream(error_response):
                                        yield event
                                    # Max attempts reached
                                    return

                                # Retry with backoff
                                delay = await calculate_retry_delay(attempt, VALIDATION_CONFIG)
                                if delay > 0:
                                    await asyncio.sleep(delay)

                                # Use streaming for retry (same as initial request)
                                if throttle_manager:
                                    await throttle_manager.wait_before_send(model_name, request_id)
                                retry_request_obj = client.build_request(
                                    method=request.method,
                                    url=target_url,
                                    headers=headers,
                                    params=request.query_params,
                                    content=request_content,
                                )
                                retry_response = await client.send(retry_request_obj, stream=True)
                                if retry_response.status_code == 200:
                                    log_info(request_id, f"Retry attempt {attempt}/{max_attempts}")
                                    current_response = retry_response
                                    continue
                                else:
                                    log_info(request_id, f"WARNING: Retry attempt {attempt}/{max_attempts} failed with status {retry_response.status_code}")
                                    await retry_response.aclose()
                                    for chunk in buffer.get_chunks():
                                        yield chunk
                                    return

                            # Validation - use same text extraction as mid-stream for consistency
                            chunks = buffer.get_chunks()
                            try:
                                # Extract text using same method as mid-stream validation
                                text_content = extract_text_from_sse_chunks(chunks)
                                word_count = count_words_in_text(text_content)

                                if text_content.strip():
                                    # Validation using same validator as mid-stream
                                    log_info(request_id, f"Validation started ({word_count} words)")
                                    validation_result = await validate_response_partial(text_content, VALIDATION_CONFIG)

                                    if validation_result.error:
                                        log_info(request_id, f"WARNING: Validator error: {validation_result.error}")
                                        for chunk in chunks:
                                            yield chunk
                                        return

                                    if validation_result.is_valid:
                                        log_info(request_id, f"Validation passed (is_valid={validation_result.is_valid}, issue_type={validation_result.issue_type}, confidence={validation_result.confidence:.2f})")
                                        for chunk in chunks:
                                            yield chunk
                                        return

                                    log_info(request_id, f"Validation failed (is_valid={validation_result.is_valid}, issue_type={validation_result.issue_type}, confidence={validation_result.confidence:.2f})")
                                    # Save failed response using same format as mid-stream
                                    saved_path = save_mid_stream_failure(text_content, word_count, validation_result.issue_type, attempt, chunks)
                                    log_info(request_id, f"Validation failed (confidence: {validation_result.confidence:.2f}, issue: {validation_result.issue_type}) see: {saved_path}")

                                    if attempt >= max_attempts:
                                        openai_passthrough_status_holder["status_code"] = 529
                                        error_response = create_error_message(validation_result.issue_type, saved_path)
                                        for event in build_openai_error_stream(error_response):
                                            yield event
                                        # Max attempts reached, no retry
                                        return

                                    # Retry with backoff
                                    delay = await calculate_retry_delay(attempt, VALIDATION_CONFIG)
                                    if delay > 0:
                                        await asyncio.sleep(delay)

                                    # Use streaming for retry (same as initial request)
                                    if throttle_manager:
                                        await throttle_manager.wait_before_send(model_name, request_id)
                                    retry_request_obj = client.build_request(
                                        method="POST",
                                        url=target_url,
                                        headers=headers,
                                        content=request_content,
                                    )
                                    retry_response = await client.send(retry_request_obj, stream=True)
                                    if retry_response.status_code == 200:
                                        log_info(request_id, f"Retry attempt {attempt}/{max_attempts}")
                                        current_response = retry_response
                                        continue
                                    else:
                                        log_info(request_id, f"WARNING: Retry attempt {attempt}/{max_attempts} failed with status {retry_response.status_code}")
                                        await retry_response.aclose()
                                        for chunk in chunks:
                                            yield chunk
                                        return
                                else:
                                    # Empty content (tool-only response) - pass through without validation
                                    log_info(request_id, "Validation skipped (no text content)")
                                    for chunk in chunks:
                                        yield chunk
                                    return

                            except Exception as e:
                                log_info(request_id, f"ERROR during OpenAI passthrough streaming validation: {e}")
                                for chunk in chunks:
                                    yield chunk
                                return

                    return _StatusStreamingResponse(
                        wrap_stream_with_semaphore_release(buffered_openai_passthrough_stream_with_validation()),
                        status_holder=openai_passthrough_status_holder,
                        status_code=target_response.status_code,
                        headers=response_headers,
                        media_type=response_headers.get("content-type"),
                    )

                # Define a local async generator to yield chunks and close the httpx response
                async def stream_and_close_response():
                    chunk_count = 0
                    try:
                        async for chunk in target_response.aiter_bytes():
                            chunk_count += 1
                            
                            # Convert OpenAI streaming response to Anthropic format only in convert mode
                            if is_anthropic_request and not should_passthrough_anthropic and chunk:
                                try:
                                    chunk_str = chunk.decode('utf-8')
                                    if chunk_str.startswith('data: ') and not chunk_str.startswith('data: [DONE]'):
                                        try:
                                            openai_data = json.loads(chunk_str[6:])  # Remove 'data: ' prefix
                                            
                                            choice = openai_data.get("choices", [{}])[0]
                                            delta = choice.get("delta", {})
                                            
                                            # Handle different types of deltas
                                            if "content" in delta and delta["content"]:
                                                # Text content delta
                                                anthropic_data = {
                                                    "type": "content_block_delta",
                                                    "index": 0,
                                                    "delta": {
                                                        "type": "text_delta",
                                                        "text": delta["content"]
                                                    }
                                                }
                                            elif "tool_calls" in delta:
                                                # Tool call delta
                                                tool_calls = delta["tool_calls"]
                                                if tool_calls:  # Handle array of tool calls
                                                    tool_call = tool_calls[0]  # Take first tool call for simplicity
                                                    if "function" in tool_call:
                                                        function = tool_call["function"]
                                                        if "name" in function:
                                                            # Start of new tool call
                                                            anthropic_data = {
                                                                "type": "content_block_start",
                                                                "index": 0,
                                                                "content_block": {
                                                                    "type": "tool_use",
                                                                    "id": tool_call.get("id", f"toolu_{choice.get('index', 0)}"),
                                                                    "name": function["name"],
                                                                    "input": {}
                                                                }
                                                            }
                                                        elif "arguments" in function:
                                                            # Arguments for existing tool call
                                                            anthropic_data = {
                                                                "type": "content_block_delta",
                                                                "index": 0,
                                                                "delta": {
                                                                    "type": "input_json_delta",
                                                                    "partial_json": function["arguments"]
                                                                }
                                                            }
                                                    elif "id" in tool_call:
                                                        # Tool call ID update
                                                        anthropic_data = {
                                                            "type": "content_block_start",
                                                            "index": 0,
                                                            "content_block": {
                                                                "type": "tool_use",
                                                                "id": tool_call["id"],
                                                                "name": "",
                                                                "input": {}
                                                            }
                                                        }
                                                else:
                                                    # Skip if no tool calls
                                                    raise json.JSONDecodeError("No tool calls", chunk_str, 0)
                                            elif "finish_reason" in choice:
                                                # End of message
                                                finish_reason = choice["finish_reason"]
                                                stop_reason_map = {
                                                    "stop": "end_turn",
                                                    "length": "max_tokens",
                                                    "tool_calls": "tool_use",
                                                    "content_filter": "stop_sequence",
                                                    "function_call": "tool_use"
                                                }
                                                stop_reason = stop_reason_map.get(finish_reason, "end_turn")
                                                
                                                anthropic_data = {
                                                    "type": "message_stop",
                                                    "stop_reason": stop_reason
                                                }
                                            else:
                                                # Skip other deltas
                                                raise json.JSONDecodeError("Unhandled delta type", chunk_str, 0)
                                            
                                            # Add usage info if available
                                            if "usage" in openai_data:
                                                anthropic_data["usage"] = {
                                                    "input_tokens": openai_data["usage"].get("prompt_tokens", 0),
                                                    "output_tokens": openai_data["usage"].get("completion_tokens", 0)
                                                }
                                            
                                            converted_chunk = f"data: {json.dumps(anthropic_data)}\n\n"
                                            chunk = converted_chunk.encode('utf-8')
                                            
                                            #if ENABLE_DEBUG_LOGS:
                                            #    print(f"DEBUG: Converted streaming chunk to Anthropic format")
                                        except json.JSONDecodeError:
                                            # If we can't parse the JSON, just pass through the original chunk
                                            pass
                                except UnicodeDecodeError:
                                    # If we can't decode as UTF-8, pass through the original chunk
                                    pass
                            
                            # if ENABLE_DEBUG_LOGS: # Commented out for less verbose output during streaming
                            #     print(f"DEBUG: Yielding chunk {chunk_count}, size: {len(chunk)} bytes.")
                            yield chunk
                            # await asyncio.sleep(0) # Yield control to event loop, may help with some race conditions
                    except Exception as e:
                        logger.error(f"Exception during streaming chunks: {e}")
                        raise # Re-raise to propagate the error
                    finally:
                        # Ensure the httpx response is closed after iteration
                        if ENABLE_DEBUG_LOGS:
                            logger.debug(f"Upstream response connection closed by generator after {chunk_count} chunks.")
                        await target_response.aclose()

                return StreamingResponse(
                    wrap_stream_with_semaphore_release(stream_and_close_response()), # Use the local async generator
                    status_code=target_response.status_code,
                    headers=response_headers,
                    media_type=response_headers.get("content-type"),
                )
            else:
                # Handle non-streaming response
                if ENABLE_DEBUG_LOGS:
                    logger.debug(f"Upstream Response Headers (full): {target_response.headers}")
                    logger.debug(f"Upstream Response Status: {target_response.status_code}")
                    logger.debug(f"Upstream Response Content: {target_response.text}")
                
                # Handle Anthropic response conversion for non-streaming requests
                response_content = target_response.content
                
                # Log 404 errors specifically for debugging
                if target_response.status_code == 404:
                    if is_anthropic_request:
                        logger.warning(f"Anthropic request to {target_path} returned 404. Upstream server may not support OpenAI chat completions endpoint.")
                    else:
                        logger.warning(f"Request to {target_path} returned 404. Endpoint may not exist on upstream server.")

                if is_anthropic_request and not should_passthrough_anthropic and target_response.status_code == 200:
                    try:
                        openai_response = json.loads(response_content.decode('utf-8'))
                        
                        choice = openai_response.get("choices", [{}])[0]
                        message = choice.get("message", {})
                        
                        # Build Anthropic content array
                        anthropic_content = []
                        
                        # Handle text content
                        text_content = message.get("content", "")
                        if text_content:
                            anthropic_content.append({
                                "type": "text",
                                "text": text_content
                            })
                        
                        # Handle tool calls
                        tool_calls = message.get("tool_calls", [])
                        for tool_call in tool_calls:
                            function = tool_call.get("function", {})
                            try:
                                arguments = json.loads(function.get("arguments", "{}"))
                            except json.JSONDecodeError:
                                arguments = {}
                            
                            anthropic_tool_use = {
                                "type": "tool_use",
                                "id": tool_call.get("id", f"toolu_{len(anthropic_content)}"),
                                "name": function.get("name", ""),
                                "input": arguments
                            }
                            anthropic_content.append(anthropic_tool_use)
                            if ENABLE_DEBUG_LOGS:
                                logger.debug(f"Converted OpenAI tool_call to Anthropic tool_use: {anthropic_tool_use}")
                        
                        # Convert OpenAI finish_reason to Anthropic stop_reason
                        finish_reason = choice.get("finish_reason", "stop")
                        stop_reason_map = {
                            "stop": "end_turn",
                            "length": "max_tokens",
                            "tool_calls": "tool_use",
                            "content_filter": "stop_sequence",
                            "function_call": "tool_use"
                        }
                        stop_reason = stop_reason_map.get(finish_reason, "end_turn")
                        
                        # Convert OpenAI response to Anthropic format
                        anthropic_response = {
                            "id": openai_response.get("id", f"msg_{openai_response.get('created', 0)}"),
                            "type": "message",
                            "role": "assistant",
                            "content": anthropic_content,
                            "model": openai_response.get("model", ""),
                            "stop_reason": stop_reason,
                            "stop_sequence": None,
                            "usage": {
                                "input_tokens": openai_response.get("usage", {}).get("prompt_tokens", 0),
                                "output_tokens": openai_response.get("usage", {}).get("completion_tokens", 0)
                            }
                        }
                        
                        response_content = json.dumps(anthropic_response).encode('utf-8')
                        if ENABLE_DEBUG_LOGS:
                            logger.debug("Converted non-streaming response to Anthropic format")
                            
                    except (json.JSONDecodeError, UnicodeDecodeError, KeyError, IndexError) as e:
                        if ENABLE_DEBUG_LOGS:
                            logger.debug(f"Could not convert response to Anthropic format: {e}. Using original response.")
                        # Keep original response if conversion fails

                elif is_anthropic_request and should_passthrough_anthropic:
                    if ENABLE_DEBUG_LOGS:
                        logger.debug("Anthropic passthrough mode - keeping response format")
                elif not is_anthropic_request and should_passthrough_openai:
                    if ENABLE_DEBUG_LOGS:
                        logger.debug("OpenAI passthrough mode - keeping response format")

                # Validation logic for all modes (anthropic passthrough, openai convert, openai passthrough)
                validation_failed = False
                if (VALIDATION_CONFIG.get("enabled", False) and
                    target_response.status_code == 200):

                    max_retries = VALIDATION_CONFIG.get("max_retries", 3)
                    # Total attempts = 1 (initial) + max_retries
                    max_attempts = 1 + max_retries
                    attempt = 0

                    while attempt < max_attempts:
                        attempt += 1

                        # Parse response for validation
                        try:
                            response_dict = json.loads(response_content.decode('utf-8'))
                        except json.JSONDecodeError:
                            # Can't validate non-JSON, pass through
                            break

                        # Validate response
                        validation_result = await validate_response(response_dict, VALIDATION_CONFIG)

                        if validation_result.error:
                            # Validator failed, log and pass through
                            log_info(request_id, f"WARNING: Validator error: {validation_result.error}")
                            break

                        if validation_result.is_valid:
                            # Valid response, proceed
                            if ENABLE_DEBUG_LOGS:
                                logger.debug(f"Response validated successfully (attempt {attempt})")
                            break

                        # Invalid response - save and retry
                        saved_path = save_failed_response(response_dict, validation_result, attempt)
                        log_info(request_id, f"Validation failed (confidence: {validation_result.confidence:.2f}, issue: {validation_result.issue_type}) see: {saved_path}")

                        if attempt >= max_attempts:
                            # Max attempts reached, return error with proper status code
                            validation_failed = True
                            if is_anthropic_request:
                                error_response = build_anthropic_error_json(validation_result.issue_type, saved_path)
                            else:
                                error_response = build_openai_error_json(validation_result.issue_type, saved_path)
                            response_content = json.dumps(error_response).encode('utf-8')
                            break

                        # Retry with backoff
                        delay = await calculate_retry_delay(attempt, VALIDATION_CONFIG)
                        if delay > 0:
                            await asyncio.sleep(delay)

                        # Make retry request
                        log_info(request_id, f"Retry attempt {attempt}/{max_attempts}")
                        if throttle_manager:
                            await throttle_manager.wait_before_send(model_name, request_id)
                        retry_response = await client.request(
                            method="POST",
                            url=target_url,
                            headers=headers,
                            content=request_content,
                        )

                        if retry_response.status_code == 200:
                            response_content = retry_response.content
                            await retry_response.aclose()
                        else:
                            # Retry request failed, use last response
                            log_info(request_id, f"WARNING: Retry attempt {attempt}/{max_attempts} failed with status {retry_response.status_code}")
                            await retry_response.aclose()
                            break

                # Ensure the httpx response is closed after its content is read
                await target_response.aclose()
                
                # Create clean headers for the response, removing Content-Length to prevent mismatches
                # Also remove Content-Encoding since httpx automatically decompresses responses
                clean_headers = dict(target_response.headers)
                clean_headers.pop("content-length", None)
                clean_headers.pop("Content-Length", None)
                clean_headers.pop("content-encoding", None)
                clean_headers.pop("Content-Encoding", None)
                
                return Response(
                    content=response_content,
                    status_code=529 if validation_failed else target_response.status_code,
                    headers=clean_headers,
                    media_type="application/json" if validation_failed else target_response.headers.get("content-type"),
                )
        else:
            if ENABLE_DEBUG_LOGS:
                logger.debug("Sending non-generation request to upstream server.")
            # For all other requests (e.g., GET /models), fetch the full response
            target_response = await client.request(
                method=request.method,
                url=target_url, # Use original_path for the actual request
                headers=headers,
                params=request.query_params,
                content=request_content,
            )
            if ENABLE_DEBUG_LOGS:
                logger.debug(f"Upstream Response Headers (full): {target_response.headers}")
                logger.debug(f"Upstream Response Status: {target_response.status_code}")
            
            # Log 404 errors specifically for debugging
            if target_response.status_code == 404:
                logger.warning(f"Non-generation request to {target_path} returned 404. Endpoint may not exist on upstream server.")
            
            # Ensure the httpx response is closed after its content is read
            await target_response.aclose()

            # Clean headers - remove Content-Encoding since httpx auto-decompresses
            clean_headers = dict(target_response.headers)
            clean_headers.pop("content-encoding", None)
            clean_headers.pop("Content-Encoding", None)
            clean_headers.pop("content-length", None)
            clean_headers.pop("Content-Length", None)

            return Response(
                content=target_response.content,
                status_code=target_response.status_code,
                headers=clean_headers,
                media_type=target_response.headers.get("content-type"),
            )

    except httpx.ConnectError as e:
        logger.error(f"[{request.method} {original_path}] Connection error to upstream server: {e}")
        return Response(f"Could not connect to upstream server at {TARGET_BASE_URL}: {e}",
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
    except httpx.RequestError as e:
        logger.error(f"[{request.method} {original_path}] Request error to upstream server: {e}")
        return Response(f"An error occurred while requesting upstream server: {e}",
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
    except Exception as e:
        logger.error(f"[{request.method} {original_path}] An unexpected error occurred: {e}")
        return Response(f"An unexpected error occurred: {e}",
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
    finally:
        # --- Parallel Limit Semaphore Release ---
        # For streaming responses, semaphores are released by wrap_stream_with_semaphore_release()
        # when the stream completes. Do NOT release here — the finally runs when the handler
        # returns the StreamingResponse, NOT when streaming finishes.
        if not is_streaming_response and not semaphores_released:
            await release_semaphores()

        # --- Throttle After Send (generation requests only) ---
        if throttle_manager and is_generation_request and request.method == "POST":
            await throttle_manager.wait_after_send(model_name, request_id)

# --- Main execution block for direct script execution ---
if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description="Sampling Proxy"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config.json",
        help="Path to configuration JSON file (default: config.json)",
    )
    parser.add_argument(
        "--host",
        type=str,
        help="Host address for the Sampling Proxy server (overrides config)",
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Port for the Sampling Proxy server (overrides config)",
    )
    parser.add_argument(
        "--base-path",
        type=int,
        help="Base path for the Sampling Proxy server (overrides config)",
    )    
    parser.add_argument(
        "--target-base-url",
        type=str,
        help="Base URL for the upstream server (overrides config)",
    )
    parser.add_argument(
        "--debug-logs",
        "-d",
        action="store_true", # This makes it a boolean flag
        help="Enable detailed debug logging (overrides config)",
        default=None,  # Explicitly set default to None to detect when it's not provided
    )
    parser.add_argument(
        "--override-logs",
        "-o",
        action="store_true", # This makes it a boolean flag
        help="Enable override logs to show when sampling parameters are being overridden (overrides config)",
        default=None,  # Explicitly set default to None to detect when it's not provided
    )
    parser.add_argument(
        "--override-sampling-params",
        type=str,
        help="Override specific sampling parameters as JSON string. Example: '{\"temperature\": 0.7, \"top_p\": 0.9}' (overrides config)",
    )
    parser.add_argument(
        "--override-only-anthropic",
        action="store_true",
        help="Apply overrides only to Anthropic requests (overrides config)",
        default=None,
    )
    parser.add_argument(
        "--override-model-name",
        type=str,
        help="Override model name in requests (overrides config)",
    )
    parser.add_argument(
        "--parallel-limits",
        type=str,
        help="Override parallel limits as JSON string. Example: '{\"global\": 10, \"model-name\": 2}' (overrides config)",
    )

    args = parser.parse_args()

    # Load configuration with specified config file path
    CONFIG = load_config(args.config)
    
    # Override global constants with command-line arguments (take precedence over config)
    SAMPLING_PROXY_HOST = args.host if args.host is not None else CONFIG["server"]["sampling_proxy_host"]
    SAMPLING_PROXY_PORT = args.port if args.port is not None else CONFIG["server"]["sampling_proxy_port"]
    SAMPLING_PROXY_BASE_PATH = args.base_path if args.base_path is not None else CONFIG["server"].get("sampling_proxy_base_path", "")
    TARGET_BASE_URL = args.target_base_url if args.target_base_url is not None else CONFIG["server"]["target_base_url"]
    TARGET_BASE_PATH = extract_base_path(TARGET_BASE_URL)
    ENABLE_DEBUG_LOGS = args.debug_logs if args.debug_logs is not None else CONFIG["logging"]["enable_debug_logs"]
    ENABLE_OVERRIDE_LOGS = args.override_logs if args.override_logs is not None else CONFIG["logging"]["enable_override_logs"]
    ENABLE_VALIDATION_LOGS = CONFIG["logging"].get("enable_validation_logs", False)

    # Load sampling parameters from config
    DEFAULT_SAMPLING_PARAMS = CONFIG["default_sampling_params"]
    OVERRIDE_CONFIG = CONFIG["override"]
    OVERRIDE_ONLY_ANTHROPIC = OVERRIDE_CONFIG.get("only_anthropic", False)
    OVERRIDE_MODEL_NAME = OVERRIDE_CONFIG.get("model_name")
    OVERRIDE_SAMPLING_PARAMS = OVERRIDE_CONFIG.get("sampling_params", {})
    MODEL_SAMPLING_PARAMS = CONFIG["model_sampling_params"]

    # Load server capabilities from server config
    server_config = CONFIG.get("server", {})
    SERVER_SUPPORTS_OPENAI = server_config.get("supports_openai", True)
    SERVER_SUPPORTS_ANTHROPIC = server_config.get("supports_anthropic", False)
    VALIDATION_CONFIG = CONFIG.get("validation", {"enabled": False})
    VALIDATION_CONFIG["enable_validation_logs"] = ENABLE_VALIDATION_LOGS
    logger.info(f"Validation config loaded: enabled={VALIDATION_CONFIG.get('enabled')}, mid_stream_enabled={VALIDATION_CONFIG.get('mid_stream_validation_enabled')}")

    # Load throttle configuration
    THROTTLE_CONFIG = CONFIG.get("throttle", {"enabled": False})
    logger.info(f"Throttle config loaded: enabled={THROTTLE_CONFIG.get('enabled')}")

    # Parse override parameters from command line if provided (takes precedence over config)
    if args.override_sampling_params:
        try:
            parsed_params = json.loads(args.override_sampling_params)
            if isinstance(parsed_params, dict):
                OVERRIDE_SAMPLING_PARAMS = parsed_params
                OVERRIDE_CONFIG["sampling_params"] = OVERRIDE_SAMPLING_PARAMS
                logger.info(f"Override sampling parameters from command line: {OVERRIDE_SAMPLING_PARAMS}")
            else:
                logger.warning(f"--override-sampling-params must be a JSON object. Ignoring invalid input: {args.override_sampling_params}")
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in --override-sampling-params: {e}. Ignoring.")
    
    # Handle override-only-anthropic flag from command line
    if args.override_only_anthropic is not None:
        OVERRIDE_ONLY_ANTHROPIC = args.override_only_anthropic
        OVERRIDE_CONFIG["only_anthropic"] = OVERRIDE_ONLY_ANTHROPIC
        logger.info(f"Override only_anthropic from command line: {OVERRIDE_ONLY_ANTHROPIC}")
    
    # Handle override-model-name from command line
    if args.override_model_name is not None:
        OVERRIDE_MODEL_NAME = args.override_model_name
        OVERRIDE_CONFIG["model_name"] = OVERRIDE_MODEL_NAME
        logger.info(f"Override model_name from command line: {OVERRIDE_MODEL_NAME}")

    # Load parallel request limits and initialize semaphores
    # Command-line --parallel-limits overrides config if provided
    parallel_limits_raw = CONFIG.get("parallel_limits", {})
    if args.parallel_limits:
        try:
            parsed_limits = json.loads(args.parallel_limits)
            if isinstance(parsed_limits, dict):
                parallel_limits_raw = parsed_limits
                logger.info(f"Parallel limits from command line: {parsed_limits}")
            else:
                logger.warning(f"--parallel-limits must be a JSON object. Ignoring invalid input: {args.parallel_limits}")
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in --parallel-limits: {e}. Ignoring.")
    
    # Extract and remove the special "global" key before iterating model limits
    GLOBAL_LIMIT = parallel_limits_raw.pop("global", None)
    GLOBAL_SEMAPHORE = None
    if GLOBAL_LIMIT is not None:
        if isinstance(GLOBAL_LIMIT, int) and GLOBAL_LIMIT > 0:
            GLOBAL_SEMAPHORE = asyncio.Semaphore(GLOBAL_LIMIT)
            logger.info(f"Global parallel limit: {GLOBAL_LIMIT} concurrent request(s) across all models")
        else:
            logger.warning(f"Invalid global parallel limit: {GLOBAL_LIMIT}. Must be a positive integer. Skipping.")
    else:
        logger.info("No global parallel limit configured.")
    
    PARALLEL_LIMITS = {k.lower(): v for k, v in parallel_limits_raw.items()}
    MODEL_SEMAPHORES = {}
    for model_name, limit in PARALLEL_LIMITS.items():
        if isinstance(limit, int) and limit > 0:
            MODEL_SEMAPHORES[model_name] = asyncio.Semaphore(limit)
            logger.info(f"Parallel limit: model '{model_name}' limited to {limit} concurrent request(s)")
        else:
            logger.warning(f"Invalid parallel limit for model '{model_name}': {limit}. Must be a positive integer. Skipping.")
    if not PARALLEL_LIMITS:
        logger.info("No per-model parallel request limits configured.")

    # Initialize throttle manager
    throttle_manager = None
    if THROTTLE_CONFIG.get("enabled"):
        try:
            throttle_manager = ThrottleManager(THROTTLE_CONFIG, ENABLE_DEBUG_LOGS, 0)
            logger.info(f"Throttle manager initialized: enabled={throttle_manager.enabled}")
        except ValueError as e:
            logger.error(f"Invalid throttle configuration: {e}")
            raise

    logger.info(f"Starting Sampling Proxy server on http://{SAMPLING_PROXY_HOST}:{SAMPLING_PROXY_PORT}")
    logger.info(f"Proxying requests to upstream server at {TARGET_BASE_URL}")
    logger.info(f"Server capabilities: OpenAI={SERVER_SUPPORTS_OPENAI}, Anthropic={SERVER_SUPPORTS_ANTHROPIC}")
    logger.info(f"Debug logs are {'ENABLED' if ENABLE_DEBUG_LOGS else 'DISABLED'}.")
    logger.info(f"Override logs are {'ENABLED' if ENABLE_OVERRIDE_LOGS else 'DISABLED'}.")
    logger.info(f"Validation logs are {'ENABLED' if ENABLE_VALIDATION_LOGS else 'DISABLED'}.")
    uvicorn.run(app, host=SAMPLING_PROXY_HOST, port=SAMPLING_PROXY_PORT)
