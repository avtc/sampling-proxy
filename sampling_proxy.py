import os
import json
import httpx
from fastapi import FastAPI, Request, Response, status
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import uvicorn
import asyncio # Import asyncio for potential sleep
import argparse # Import argparse for command-line arguments

def load_config(config_path="config.json"):
    """
    Load configuration from JSON file.
    Returns a dictionary with configuration values.
    If config file doesn't exist or is invalid, returns default values.
    """
    default_config = {
        "server": {
            "target_base_url": "http://127.0.0.1:8000/v1",
            "sampling_proxy_base_path": "/v1",
            "sampling_proxy_host": "0.0.0.0",
            "sampling_proxy_port": 8001,
            "timeout_seconds": 1200.0
        },
        "logging": {
            "enable_debug_logs": False,
            "enable_override_logs": False
        },
        "default_sampling_params": {},
        "enforced_sampling_params": {},
        "model_sampling_params": {}
    }
    
    if not os.path.exists(config_path):
        print(f"WARNING: Config file '{config_path}' not found. Using default values.")
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
        
        if merged_config.get("enforced_sampling_params"):
            merged_config["enforced_sampling_params"] = {
                k: v for k, v in merged_config["enforced_sampling_params"].items()
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
        
        print(f"Configuration loaded from '{config_path}'")
        return merged_config
        
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in config file '{config_path}': {e}. Using default values.")
        return default_config
    except Exception as e:
        print(f"ERROR: Error loading config file '{config_path}': {e}. Using default values.")
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
DEFAULT_SAMPLING_PARAMS = {}
ENFORCED_SAMPLING_PARAMS = {}
MODEL_SAMPLING_PARAMS = {}

# List of API paths that are considered "generation" endpoints.
# Note: Paths here should NOT have leading/trailing slashes for direct comparison
GENERATION_ENDPOINTS = [
    "generate",            # Common SGLang generation endpoint
    "completions",      # OpenAI-compatible completions endpoint
    "chat/completions", # OpenAI-compatible chat completions endpoint
    "v1/messages",          # Anthropic-compatible messages endpoint
]

# List of Anthropic-specific endpoints that should be handled locally
ANTHROPIC_ENDPOINTS = [
    "api/event_logging/batch",  # Anthropic event logging endpoint
    "v1/messages/count_tokens", # Anthropic token counting endpoint
]

# Global variable to store the first available model name from /models
FIRST_AVAILABLE_MODEL = "any" # sglang allows any model name, vllm require exact match

# Initialize an httpx AsyncClient for making requests to the OpenAI Compatible backend.
# This client is designed for efficient connection pooling.
# A higher timeout is set to accommodate potentially long LLM generation times.
# Note: This will be re-initialized after config loading in the main block
client = None

# --- FastAPI Application Lifespan Setup ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events for the FastAPI application.
    Ensures the httpx client is properly closed when the application shuts down.
    """
    global FIRST_AVAILABLE_MODEL, client
    print("FastAPI application startup.")
    
    # Initialize client with the correct TARGET_BASE_URL and timeout from config
    timeout_seconds = CONFIG["server"].get("timeout_seconds", 1200.0)
    client = httpx.AsyncClient(base_url=TARGET_BASE_URL, timeout=timeout_seconds)
    
    # Poll /models to get the first available model
    try:
        print(f"Polling {TARGET_BASE_URL}/models to get available models...")
        response = await client.get("/models")
        if response.status_code == 200:
            models_data = response.json()
            if "data" in models_data and len(models_data["data"]) > 0:
                FIRST_AVAILABLE_MODEL = models_data["data"][0]["id"]
                print(f"Successfully retrieved first available model: {FIRST_AVAILABLE_MODEL}")
            else:
                print("WARNING: No models found in /models response")
        else:
            print(f"WARNING: Failed to get models from /models. Status: {response.status_code}")
    except Exception as e:
        print(f"WARNING: Error polling /models: {e}")
    
    yield # Application starts here
    print("FastAPI application shutdown.")
    if client:
        await client.aclose()
        print("HTTPX client closed.")

# --- FastAPI Application Setup ---
app = FastAPI(
    title="Sampling Proxy",
    description="A middleware server to override sampling parameters for generation requests, supports OpenAI Compatible target server and OpenAI Compatible and Anthropic requests.",
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
        "enforced_sampling_params": ENFORCED_SAMPLING_PARAMS,
        "model_sampling_params_configured": list(MODEL_SAMPLING_PARAMS.keys()),
        "generation_endpoints_monitored": GENERATION_ENDPOINTS,
        "anthropic_endpoints_handled_locally": ANTHROPIC_ENDPOINTS,
        "debug_logs_enabled": ENABLE_DEBUG_LOGS,
    }

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def proxy_target_requests(path: str, request: Request):
    """
    Catch-all route to proxy all incoming requests to the OpenAI Compatible backend.
    For POST requests to configured generation endpoints, it applies
    the sampling parameter override logic.
    Supports streaming responses from the OpenAI Compatible backend back to the client.
    """
    # Access ENABLE_DEBUG_LOGS from the global scope
    global ENABLE_DEBUG_LOGS

    if ENABLE_DEBUG_LOGS:
        print(f"\n--- Incoming Request: {request.method} {path} ---")
    # Normalize path by removing leading/trailing slashes for consistent matching
    original_path = path
    path = path.strip('/')
    if ENABLE_DEBUG_LOGS:
        print(f"DEBUG: Normalized path for matching: '{path}' (Original: '{original_path}')")

    # Handle Anthropic-specific endpoints locally
    if path in ANTHROPIC_ENDPOINTS:
        if ENABLE_DEBUG_LOGS:
            print(f"DEBUG: Handling Anthropic endpoint '{path}' locally")
        
        if path == "api/event_logging/batch":
            # Handle event logging endpoint - return success response
            if ENABLE_DEBUG_LOGS:
                print(f"DEBUG: Processing event logging request")
            
            try:
                # Read the request body to acknowledge receipt
                body = await request.body()
                if ENABLE_DEBUG_LOGS:
                    print(f"DEBUG: Event logging body received: {len(body)} bytes")
                
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
                    print(f"ERROR: Error processing event logging: {e}")
                return Response(
                    content=json.dumps({"error": "Failed to process events"}),
                    status_code=500,
                    media_type="application/json"
                )
        
        elif path == "v1/messages/count_tokens":
            # Handle token counting endpoint
            if ENABLE_DEBUG_LOGS:
                print(f"DEBUG: Processing token counting request")
            
            try:
                # Read and parse the request body
                body = await request.body()
                if ENABLE_DEBUG_LOGS:
                    print(f"DEBUG: Token counting body received: {len(body)} bytes")
                
                request_data = json.loads(body.decode('utf-8'))
                messages = request_data.get("messages", [])
                model = request_data.get("model", "claude-3-sonnet-20241022")
                
                if ENABLE_DEBUG_LOGS:
                    print(f"DEBUG: Token counting request - model: {model}, messages: {messages}")
                
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
                    print(f"DEBUG: Token counting result: {total_tokens} tokens")
                
                return Response(
                    content=json.dumps(response_data),
                    status_code=200,
                    media_type="application/json"
                )
            except json.JSONDecodeError as e:
                if ENABLE_DEBUG_LOGS:
                    print(f"ERROR: Invalid JSON in token counting request: {e}")
                return Response(
                    content=json.dumps({"error": {"type": "invalid_request_error", "message": "Invalid JSON"}}),
                    status_code=400,
                    media_type="application/json"
                )
            except Exception as e:
                if ENABLE_DEBUG_LOGS:
                    print(f"ERROR: Error processing token counting: {e}")
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

    # Prepare headers for the outgoing request to OpenAI Compatible backend.
    # We copy the incoming headers and remove 'host' and 'content-length'
    # as httpx will manage these for the new request.
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None) # httpx will recalculate if body changes
    if ENABLE_DEBUG_LOGS:
        print(f"DEBUG: Outgoing Request Headers (initial): {headers}")

    request_content = None # This will hold the request body to be sent to target
    is_generation_request = False
    is_anthropic_request = False # Initialize Anthropic request flag
    incoming_json_body = {} # Initialize in case it's not a POST/JSON request

    # Determine if the current request path is a recognized generation endpoint
    is_generation_request = path in GENERATION_ENDPOINTS
    is_anthropic_request = path == "v1/messages" # Check if this is an Anthropic request
    if ENABLE_DEBUG_LOGS:
        print(f"DEBUG: is_generation_request after check: {is_generation_request}")
        print(f"DEBUG: is_anthropic_request: {is_anthropic_request}")

    # Construct the target URL for the OpenAI Compatible backend
    # Redirect Anthropic requests to OpenAI chat completions endpoint
    if is_anthropic_request:
        # Convert /v1/messages to /chat/completions for OpenAI Compatible backend
        # First apply the path transformation, then change to chat completions
        transformed_path = transform_path("/" + original_path, SAMPLING_PROXY_BASE_PATH, TARGET_BASE_PATH)
        target_path = transformed_path.replace("/v1/messages", "/chat/completions", 1)
        if ENABLE_DEBUG_LOGS:
            print(f"DEBUG: Redirecting Anthropic request from {original_path} to {target_path}")
    else:
        # Apply base path transformation
        target_path = transform_path("/" + original_path, SAMPLING_PROXY_BASE_PATH, TARGET_BASE_PATH)
    
    if ENABLE_DEBUG_LOGS:
        print(f"DEBUG: Path transformation: /{original_path} -> {target_path}")
        print(f"DEBUG: Base paths - Proxy: {SAMPLING_PROXY_BASE_PATH}, Target: {TARGET_BASE_PATH}")
    
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
        print(f"DEBUG: Relative path for httpx: {relative_path}")
    
    # Ensure the query string is encoded to bytes as required by httpx.URL
    target_url = httpx.URL(path=relative_path, query=request.url.query.encode("utf-8"))
    if ENABLE_DEBUG_LOGS:
        print(f"DEBUG: Target OpenAI Compatible URL: {target_url}")

    # --- Sampling Parameter Override Logic ---
    if is_generation_request and request.method == "POST":
        if ENABLE_DEBUG_LOGS:
            print("DEBUG: This is a POST generation request. Applying override logic.")
        try:
            # Attempt to parse the incoming request body as JSON.
            # Generation requests typically send JSON payloads.
            raw_body = await request.body()
            if ENABLE_DEBUG_LOGS:
                print(f"DEBUG: Raw incoming request body: {raw_body.decode('utf-8')}")
            incoming_json_body = json.loads(raw_body) # This will be available for response processing
            if ENABLE_DEBUG_LOGS:
                print(f"DEBUG: Parsed incoming JSON body: {incoming_json_body}")

            # Handle Anthropic to OpenAI format conversion
            if is_anthropic_request:
                if ENABLE_DEBUG_LOGS:
                    print("DEBUG: Converting Anthropic request to OpenAI format.")
                
                try:
                
                    # Extract Anthropic format data
                    anthropic_messages = incoming_json_body.get("messages", [])
                    anthropic_model = incoming_json_body.get("model")
                    anthropic_max_tokens = incoming_json_body.get("max_tokens")
                    anthropic_temperature = incoming_json_body.get("temperature")
                    anthropic_top_p = incoming_json_body.get("top_p")
                    anthropic_stream = incoming_json_body.get("stream", False)
                    anthropic_tools = incoming_json_body.get("tools")
                    anthropic_tool_choice = incoming_json_body.get("tool_choice")
                    
                    # Convert Anthropic messages to OpenAI format
                    openai_messages = []
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
                                    print(f"DEBUG: Unknown Anthropic role '{anthropic_role}' mapped to 'user'")
                            
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
                                                print(f"DEBUG: Converted Anthropic tool_use to OpenAI tool_call: {tool_call}")
                                        
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
                                                    print(f"DEBUG: Converted Anthropic tool_result to OpenAI tool message: {tool_call_msg}")
                                    
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
                                        print(f"DEBUG: Converted message {msg_idx}: {openai_msg}")
                                else:
                                    if ENABLE_DEBUG_LOGS:
                                        print(f"DEBUG: Skipping empty message {msg_idx}")
                            else:
                                if ENABLE_DEBUG_LOGS:
                                    print(f"DEBUG: Skipping invalid tool message {msg_idx}")
                        
                        except Exception as e:
                            if ENABLE_DEBUG_LOGS:
                                print(f"ERROR: Failed to convert message {msg_idx}: {e}")
                            # Continue with next message instead of failing completely
                            continue
                    
                    # Override model with first available model for Anthropic requests
                    overridden_model = FIRST_AVAILABLE_MODEL if FIRST_AVAILABLE_MODEL else anthropic_model
                    if ENABLE_DEBUG_LOGS and FIRST_AVAILABLE_MODEL:
                        print(f"DEBUG: Overriding Anthropic model '{anthropic_model}' with first available model '{FIRST_AVAILABLE_MODEL}'")
                    
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
                            print(f"DEBUG: Converted {len(anthropic_tools)} Anthropic tools to OpenAI format")
                    
                    # Convert tool_choice if present
                    if anthropic_tool_choice:
                        if anthropic_tool_choice == "auto":
                            openai_request["tool_choice"] = "auto"
                        elif anthropic_tool_choice == "none":
                            openai_request["tool_choice"] = "none"
                        elif isinstance(anthropic_tool_choice, dict):
                            # Handle specific tool choice
                            tool_name = anthropic_tool_choice.get("name")
                            if tool_name:
                                openai_request["tool_choice"] = {"type": "function", "function": {"name": tool_name}}
                        if ENABLE_DEBUG_LOGS:
                            print(f"DEBUG: Converted tool_choice: {anthropic_tool_choice} -> {openai_request.get('tool_choice')}")
                
                    # Validate the converted messages before proceeding
                    if not openai_messages:
                        if ENABLE_DEBUG_LOGS:
                            print("ERROR: No valid messages after conversion. Creating fallback message.")
                        # Create a simple fallback message
                        openai_messages = [{
                            "role": "user",
                            "content": "Please provide a response."
                        }]
                    
                    # Replace the incoming body with converted OpenAI format
                    incoming_json_body = openai_request
                    if ENABLE_DEBUG_LOGS:
                        print(f"DEBUG: Converted to OpenAI format: {incoming_json_body}")
                        print(f"DEBUG: Final message count: {len(openai_messages)}")
                
                except Exception as e:
                    print(f"ERROR: Failed to convert Anthropic request to OpenAI format: {e}")
                    if ENABLE_DEBUG_LOGS:
                        print(f"ERROR: Original Anthropic request: {incoming_json_body}")
                    
                    # Create a minimal valid OpenAI request as fallback
                    incoming_json_body = {
                        "model": FIRST_AVAILABLE_MODEL if FIRST_AVAILABLE_MODEL else "gpt-3.5-turbo",
                        "messages": [{"role": "user", "content": "Conversion failed. Please respond."}],
                        "max_tokens": anthropic_max_tokens if 'anthropic_max_tokens' in locals() else 1000,
                        "stream": anthropic_stream if 'anthropic_stream' in locals() else False
                    }
                    
                    if ENABLE_DEBUG_LOGS:
                        print(f"DEBUG: Using fallback OpenAI request: {incoming_json_body}")

            model_name = incoming_json_body.get("model") # Get the model name from the request
            if ENABLE_DEBUG_LOGS:
                print(f"DEBUG: Model name from request: {model_name}")

            # Determine where sampling parameters are expected in the request body
            # For /generate, they are typically in a 'sampling_params' sub-dictionary
            # For OpenAI-compatible endpoints, they are typically top-level keys
            if path == "generate": # Use normalized path here
                current_params_container = incoming_json_body.get("sampling_params", {})
                is_nested_params = True
                if ENABLE_DEBUG_LOGS:
                    print(f"DEBUG: Path is 'generate', using nested 'sampling_params'. Current container: {current_params_container}")
            else: # completions, chat/completions, v1/messages (normalized paths)
                current_params_container = incoming_json_body
                is_nested_params = False
                if ENABLE_DEBUG_LOGS:
                    print(f"DEBUG: Path is OpenAI-compatible, using top-level params. Current container: {current_params_container}")

            model_specific_params = MODEL_SAMPLING_PARAMS.get(model_name, {})
            if ENABLE_DEBUG_LOGS:
                print(f"DEBUG: Model-specific params for '{model_name}': {model_specific_params}")

            # First, apply enforced parameters - these ALWAYS override incoming parameters
            if ENFORCED_SAMPLING_PARAMS:
                if ENABLE_DEBUG_LOGS:
                    print(f"DEBUG: Applying enforced parameters: {ENFORCED_SAMPLING_PARAMS}")
                for param, enforced_value in ENFORCED_SAMPLING_PARAMS.items():
                    original_value = current_params_container.get(param, "not_set")
                    current_params_container[param] = enforced_value
                    if ENABLE_OVERRIDE_LOGS:
                        print(f"DEBUG: ENFORCED '{param}' from '{original_value}' to '{enforced_value}'")

            # Then, apply default parameters for any missing parameters not enforced
            for param, default_value in DEFAULT_SAMPLING_PARAMS.items():
                if param not in current_params_container:
                    # Skip if this parameter is being enforced (already handled above)
                    if ENFORCED_SAMPLING_PARAMS and param in ENFORCED_SAMPLING_PARAMS:
                        if ENABLE_DEBUG_LOGS:
                            print(f"DEBUG: Parameter '{param}' is enforced, skipping default application.")
                        continue
                    
                    # If the parameter is missing from the incoming request,
                    # try to get it from model-specific settings, otherwise use the global default.
                    value_to_apply = model_specific_params.get(param, default_value)
                    current_params_container[param] = value_to_apply
                    if ENABLE_OVERRIDE_LOGS:
                        print(f"DEBUG: Overriding '{param}' to '{value_to_apply}' (was not in request).")
                else:
                    if ENABLE_OVERRIDE_LOGS:
                        print(f"DEBUG: Parameter '{param}' already present in request: {current_params_container[param]}. Not overriding.")

            # Re-integrate the modified parameters back into the main body if they were nested
            if is_nested_params:
                incoming_json_body["sampling_params"] = current_params_container
            # If not nested, they are already updated in incoming_json_body

            request_content = json.dumps(incoming_json_body) # Serialize the modified JSON back to a string
            headers["content-type"] = "application/json" # Ensure content-type header is correct

            if ENABLE_DEBUG_LOGS:
                print(f"DEBUG: Final modified request body: {request_content}")
                print(f"[{request.method} {original_path}] Overridden sampling params for model '{model_name}': {current_params_container}")

        except json.JSONDecodeError as e:
            print(f"ERROR: [{request.method} {original_path}] JSONDecodeError: {e}. Proxying raw body.")
            request_content = await request.body()
        except Exception as e:
            print(f"ERROR: [{request.method} {original_path}] Error processing generation request body: {e}. Proxying raw body.")
            request_content = await request.body()
    else:
        if ENABLE_DEBUG_LOGS:
            print(f"DEBUG: Not a POST generation request (is_generation_request={is_generation_request}, method={request.method}). Proxying raw body without modification.")
        request_content = await request.body()

    # --- Forward Request and Handle Response ---
    try:
        if is_generation_request and request.method == "POST":
            # Check if this is actually a streaming request
            is_streaming_request = incoming_json_body.get("stream", False)
            
            if ENABLE_DEBUG_LOGS:
                print(f"DEBUG: Sending {'streaming' if is_streaming_request else 'non-streaming'} request.")
            
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
                target_response = await client.send(target_request_obj, stream=True)
            else:
                # For non-streaming requests, fetch the full response
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
                    print(f"DEBUG: OpenAI Compatible Response Headers (raw): {response_headers}")

                # Remove headers that interfere with streaming
                # Use case-insensitive removal to catch all variants
                headers_to_remove = ["content-length", "transfer-encoding", "connection"]
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
                        print(f"DEBUG: Setting response Content-Type to 'text/event-stream' for streaming request.")
                else:
                    if ENABLE_DEBUG_LOGS:
                        print(f"DEBUG: Not an OpenAI-compatible streaming path, keeping original Content-Type: {response_headers.get('content-type', 'N/A')}")

                # Define a local async generator to yield chunks and close the httpx response
                async def stream_and_close_response():
                    chunk_count = 0
                    try:
                        async for chunk in target_response.aiter_bytes():
                            chunk_count += 1
                            
                            # Convert OpenAI streaming response to Anthropic format if needed
                            if is_anthropic_request and chunk:
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
                        print(f"ERROR: Exception during streaming chunks: {e}")
                        raise # Re-raise to propagate the error
                    finally:
                        # Ensure the httpx response is closed after iteration
                        if ENABLE_DEBUG_LOGS:
                            print(f"DEBUG: OpenAI Compatible response connection closed by generator after {chunk_count} chunks.")
                        await target_response.aclose()

                return StreamingResponse(
                    stream_and_close_response(), # Use the local async generator
                    status_code=target_response.status_code,
                    headers=response_headers,
                    media_type=response_headers.get("content-type"),
                )
            else:
                # Handle non-streaming response
                if ENABLE_DEBUG_LOGS:
                    print(f"DEBUG: OpenAI Compatible Response Headers (full): {target_response.headers}")
                    print(f"DEBUG: OpenAI Compatible Response Status: {target_response.status_code}")
                    print(f"DEBUG: OpenAI Compatible Response Content: {target_response.text}")
                
                # Handle Anthropic response conversion for non-streaming requests
                response_content = target_response.content
                
                # Log 404 errors specifically for debugging
                if target_response.status_code == 404:
                    if is_anthropic_request:
                        print(f"WARNING: Anthropic request to {target_path} returned 404. OpenAI Compatible backend may not support OpenAI chat completions endpoint.")
                    else:
                        print(f"WARNING: Request to {target_path} returned 404. Endpoint may not exist on OpenAI Compatible backend.")
                
                if is_anthropic_request and target_response.status_code == 200:
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
                                print(f"DEBUG: Converted OpenAI tool_call to Anthropic tool_use: {anthropic_tool_use}")
                        
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
                            print(f"DEBUG: Converted non-streaming response to Anthropic format")
                            
                    except (json.JSONDecodeError, UnicodeDecodeError, KeyError, IndexError) as e:
                        if ENABLE_DEBUG_LOGS:
                            print(f"DEBUG: Could not convert response to Anthropic format: {e}. Using original response.")
                        # Keep original response if conversion fails
                
                # Ensure the httpx response is closed after its content is read
                await target_response.aclose()
                
                # Create clean headers for the response, removing Content-Length to prevent mismatches
                clean_headers = dict(target_response.headers)
                clean_headers.pop("content-length", None)
                clean_headers.pop("Content-Length", None)
                
                return Response(
                    content=response_content,
                    status_code=target_response.status_code,
                    headers=clean_headers,
                    media_type=target_response.headers.get("content-type"),
                )
        else:
            if ENABLE_DEBUG_LOGS:
                print("DEBUG: Sending non-generation request to OpenAI Compatible.")
            # For all other requests (e.g., GET /models), fetch the full response
            target_response = await client.request(
                method=request.method,
                url=target_url, # Use original_path for the actual request
                headers=headers,
                params=request.query_params,
                content=request_content,
            )
            if ENABLE_DEBUG_LOGS:
                print(f"DEBUG: OpenAI Compatible Response Headers (full): {target_response.headers}")
                print(f"DEBUG: OpenAI Compatible Response Status: {target_response.status_code}")
            
            # Log 404 errors specifically for debugging
            if target_response.status_code == 404:
                print(f"WARNING: Non-generation request to {target_path} returned 404. Endpoint may not exist on OpenAI Compatible backend.")
            
            # Ensure the httpx response is closed after its content is read
            await target_response.aclose()
            return Response(
                content=target_response.content,
                status_code=target_response.status_code,
                headers=target_response.headers,
                media_type=target_response.headers.get("content-type"),
            )

    except httpx.ConnectError as e:
        print(f"ERROR: [{request.method} {original_path}] Connection error to OpenAI Compatible backend: {e}")
        return Response(f"Could not connect to OpenAI Compatible backend at {TARGET_BASE_URL}: {e}",
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
    except httpx.RequestError as e:
        print(f"ERROR: [{request.method} {original_path}] Request error to OpenAI Compatible backend: {e}")
        return Response(f"An error occurred while requesting OpenAI Compatible backend: {e}",
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
    except Exception as e:
        print(f"ERROR: [{request.method} {original_path}] An unexpected error occurred: {e}")
        return Response(f"An unexpected error occurred: {e}",
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

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
        help="Base URL for the OpenAI compatible backend (overrides config)",
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
        "--enforce-params",
        "-e",
        type=str,
        help="Enforce specific sampling parameters as JSON string. Example: '{\"temperature\": 0.7, \"top_p\": 0.9}' (overrides config)",
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
    
    # Load sampling parameters from config
    DEFAULT_SAMPLING_PARAMS = CONFIG["default_sampling_params"]
    ENFORCED_SAMPLING_PARAMS = CONFIG["enforced_sampling_params"]
    MODEL_SAMPLING_PARAMS = CONFIG["model_sampling_params"]
    
    # Parse enforced parameters from command line if provided (takes precedence over config)
    if args.enforce_params:
        try:
            parsed_params = json.loads(args.enforce_params)
            if isinstance(parsed_params, dict):
                ENFORCED_SAMPLING_PARAMS = parsed_params
                print(f"Enforced sampling parameters from command line: {ENFORCED_SAMPLING_PARAMS}")
            else:
                print(f"WARNING: --enforce-params must be a JSON object. Ignoring invalid input: {args.enforce_params}")
        except json.JSONDecodeError as e:
            print(f"WARNING: Invalid JSON in --enforce-params: {e}. Ignoring.")

    print(f"Starting Sampling Proxy server on http://{SAMPLING_PROXY_HOST}:{SAMPLING_PROXY_PORT}")
    print(f"Proxying requests to OpenAI Compatible backend at {TARGET_BASE_URL}")
    print(f"Debug logs are {'ENABLED' if ENABLE_DEBUG_LOGS else 'DISABLED'}.")
    print(f"Override logs are {'ENABLED' if ENABLE_OVERRIDE_LOGS else 'DISABLED'}.")
    uvicorn.run(app, host=SAMPLING_PROXY_HOST, port=SAMPLING_PROXY_PORT)
