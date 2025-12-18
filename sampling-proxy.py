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
            "target_host": "127.0.0.1",
            "target_port": "8000",
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

# --- Configuration ---
# These will be initialized in the main block after loading config
TARGET_HOST = None
TARGET_PORT = None
TARGET_BASE_URL = None
SAMPLING_PROXY_HOST = None
SAMPLING_PROXY_PORT = None
ENABLE_DEBUG_LOGS = False
ENABLE_OVERRIDE_LOGS = False
DEFAULT_SAMPLING_PARAMS = {}
ENFORCED_SAMPLING_PARAMS = {}
MODEL_SAMPLING_PARAMS = {}

# List of API paths that are considered "generation" endpoints.
# Note: Paths here should NOT have leading/trailing slashes for direct comparison
GENERATION_ENDPOINTS = [
    "generate",            # Common SGLang generation endpoint
    "v1/completions",      # OpenAI-compatible completions endpoint
    "v1/chat/completions", # OpenAI-compatible chat completions endpoint
    "v1/messages",          # Anthropic-compatible messages endpoint
]

# List of Anthropic-specific endpoints that should be handled locally
ANTHROPIC_ENDPOINTS = [
    "api/event_logging/batch",  # Anthropic event logging endpoint
]

# Global variable to store the first available model name from /v1/models
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
    
    # Poll /v1/models to get the first available model
    try:
        print(f"Polling {TARGET_BASE_URL}/v1/models to get available models...")
        response = await client.get("/v1/models")
        if response.status_code == 200:
            models_data = response.json()
            if "data" in models_data and len(models_data["data"]) > 0:
                FIRST_AVAILABLE_MODEL = models_data["data"][0]["id"]
                print(f"Successfully retrieved first available model: {FIRST_AVAILABLE_MODEL}")
            else:
                print("WARNING: No models found in /v1/models response")
        else:
            print(f"WARNING: Failed to get models from /v1/models. Status: {response.status_code}")
    except Exception as e:
        print(f"WARNING: Error polling /v1/models: {e}")
    
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
        # Convert /v1/messages to /v1/chat/completions for OpenAI Compatible backend
        target_path = "v1/chat/completions"
        if ENABLE_DEBUG_LOGS:
            print(f"DEBUG: Redirecting Anthropic request from {original_path} to {target_path}")
    else:
        target_path = original_path
    
    # Ensure the query string is encoded to bytes as required by httpx.URL
    target_url = httpx.URL(path=target_path, query=request.url.query.encode("utf-8"))
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
                
                # Extract Anthropic format data
                anthropic_messages = incoming_json_body.get("messages", [])
                anthropic_model = incoming_json_body.get("model")
                anthropic_max_tokens = incoming_json_body.get("max_tokens")
                anthropic_temperature = incoming_json_body.get("temperature")
                anthropic_top_p = incoming_json_body.get("top_p")
                anthropic_stream = incoming_json_body.get("stream", False)
                
                # Convert Anthropic messages to OpenAI format
                openai_messages = []
                for msg in anthropic_messages:
                    openai_msg = {
                        "role": msg.get("role"),
                        "content": ""
                    }
                    
                    # Handle complex Anthropic content format
                    content = msg.get("content", [])
                    if isinstance(content, list):
                        # Extract text from complex content objects
                        text_parts = []
                        for content_item in content:
                            if isinstance(content_item, dict) and content_item.get("type") == "text":
                                text_parts.append(content_item.get("text", ""))
                            elif isinstance(content_item, str):
                                text_parts.append(content_item)
                        openai_msg["content"] = "".join(text_parts)
                    elif isinstance(content, str):
                        openai_msg["content"] = content
                    else:
                        openai_msg["content"] = str(content)
                    
                    openai_messages.append(openai_msg)
                
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
                
                # Replace the incoming body with converted OpenAI format
                incoming_json_body = openai_request
                if ENABLE_DEBUG_LOGS:
                    print(f"DEBUG: Converted to OpenAI format: {incoming_json_body}")

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
            else: # v1/completions, v1/chat/completions, v1/messages (normalized paths)
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
                    if ENABLE_DEBUG_LOGS:
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

                response_headers.pop("content-length", None) # Remove Content-Length for streaming
                response_headers.pop("transfer-encoding", None) # Remove Transfer-Encoding for streaming

                # Explicitly set Content-Type for SSE if it's a streaming chat/completion request
                # Use original_path for this check, as it's the actual path in the request
                if original_path.strip('/') in ["v1/chat/completions", "v1/completions", "v1/messages"]:
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
                                            
                                            # Convert OpenAI format to Anthropic format
                                            anthropic_data = {
                                                "type": "content_block_delta",
                                                "index": 0,
                                                "delta": {
                                                    "type": "text_delta",
                                                    "text": openai_data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                                }
                                            }
                                            
                                            # Add usage info if available
                                            if "usage" in openai_data:
                                                anthropic_data["usage"] = {
                                                    "input_tokens": openai_data["usage"].get("prompt_tokens", 0),
                                                    "output_tokens": openai_data["usage"].get("completion_tokens", 0)
                                                }
                                            
                                            converted_chunk = f"data: {json.dumps(anthropic_data)}\n\n"
                                            chunk = converted_chunk.encode('utf-8')
                                            
                                            if ENABLE_DEBUG_LOGS:
                                                print(f"DEBUG: Converted streaming chunk to Anthropic format")
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
                        
                        # Convert OpenAI response to Anthropic format
                        anthropic_response = {
                            "id": openai_response.get("id", f"msg_{openai_response.get('created', 0)}"),
                            "type": "message",
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "text",
                                    "text": openai_response.get("choices", [{}])[0].get("message", {}).get("content", "")
                                }
                            ],
                            "model": openai_response.get("model", ""),
                            "stop_reason": openai_response.get("choices", [{}])[0].get("finish_reason", "end_turn"),
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
                return Response(
                    content=response_content,
                    status_code=target_response.status_code,
                    headers=target_response.headers,
                    media_type=target_response.headers.get("content-type"),
                )
        else:
            if ENABLE_DEBUG_LOGS:
                print("DEBUG: Sending non-generation request to OpenAI Compatible.")
            # For all other requests (e.g., GET /v1/models), fetch the full response
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
        "--target-host",
        type=str,
        help="Host address for the OpenAI compatible backend (overrides config)",
    )
    parser.add_argument(
        "--target-port",
        type=str, # Keep as string as it's used in f-string for URL
        help="Port for the OpenAI compatible backend (overrides config)",
    )
    parser.add_argument(
        "--debug-logs",
        "-d",
        action="store_true", # This makes it a boolean flag
        help="Enable detailed debug logging (overrides config)",
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
    TARGET_HOST = args.target_host if args.target_host is not None else CONFIG["server"]["target_host"]
    TARGET_PORT = args.target_port if args.target_port is not None else CONFIG["server"]["target_port"]
    TARGET_BASE_URL = f"http://{TARGET_HOST}:{TARGET_PORT}" # Reconstruct with new values
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
