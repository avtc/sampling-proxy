import os
import json
import httpx
from fastapi import FastAPI, Request, Response, status
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import uvicorn
import asyncio # Import asyncio for potential sleep
import argparse # Import argparse for command-line arguments

# --- Configuration ---
# SGLang backend server address and port
SGLANG_HOST = os.getenv("SGLANG_HOST", "192.168.1.14")
SGLANG_PORT = os.getenv("SGLANG_PORT", "8001")
SGLANG_BASE_URL = f"http://{SGLANG_HOST}:{SGLANG_PORT}"

# Middleware server address and port
MIDDLEWARE_HOST = os.getenv("MIDDLEWARE_HOST", "127.0.0.1")
MIDDLEWARE_PORT = int(os.getenv("MIDDLEWARE_PORT", "8001"))

ENABLE_DEBUG_LOGS = False

# Default sampling parameters to apply if not specified in the request
# These values will be used if no model-specific override is found and
# the parameter is missing from the incoming generation request.
DEFAULT_SAMPLING_PARAMS = {
    "top_p": 0.95,
    "min_p": 0.05,
    "top_k": 40,
    "repetition_penalty": 1.05,
    "temperature": 0.6,
}
# Enforced sampling parameters - these will ALWAYS override incoming parameters
# Set to empty dict {} to disable enforcement (default behavior)
# Any parameters specified here will override both incoming request parameters
# and model-specific defaults for ALL requests
#
# Priority order (highest to lowest):
# 1. ENFORCED_SAMPLING_PARAMS (always applied, overrides everything)
# 2. Parameters present in incoming request (kept as-is if not enforced)
# 3. MODEL_SAMPLING_PARAMS (applied if parameter missing from request and not enforced)
# 4. DEFAULT_SAMPLING_PARAMS (applied if parameter missing from request and not in model-specific)
#
# Usage examples:
# - To enforce temperature to 0.7 regardless of what client sends: {"temperature": 0.7}
# - To enforce multiple parameters: {"temperature": 0.7, "top_p": 0.9, "top_k": 50}
# - To disable enforcement completely: {}
ENFORCED_SAMPLING_PARAMS = {
    # Example: Uncomment and modify to enforce specific parameters
    # "temperature": 0.7,
    # "top_p": 0.9,
    # "top_k": 50,
}

# Model-specific sampling parameters
# You can customize this dictionary to set specific parameters for different models.
# If a parameter is not present here for a given model, the DEFAULT_SAMPLING_PARAMS
# will be used as a fallback.
MODEL_SAMPLING_PARAMS = {
    "/home/ubuntu/models/Autoround/Devstral-Small-2507-mistralai-w8g128": {
        "top_p": 0.95,
        "temperature": 0.15,
        "repetition_penalty": 1.0,
        "top_k": 40,
        "min_p": 0.01,
    },
    # Add more models and their specific parameters here, e.g.:
    # "your-custom-model": {
    #     "temperature": 0.9,
    #     "top_p": 0.85,
    # },
}

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

# Initialize an httpx AsyncClient for making requests to the SGLang backend.
# This client is designed for efficient connection pooling.
# A higher timeout is set to accommodate potentially long LLM generation times.
client = httpx.AsyncClient(base_url=SGLANG_BASE_URL, timeout=1200.0)

# Global variable to store the first available model name from /v1/models
FIRST_AVAILABLE_MODEL = "any" # sglang allows any model name, vllm require exact match

# --- FastAPI Application Lifespan Setup ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events for the FastAPI application.
    Ensures the httpx client is properly closed when the application shuts down.
    """
    global FIRST_AVAILABLE_MODEL
    print("FastAPI application startup.")
    
    # Poll /v1/models to get the first available model
    try:
        print(f"Polling {SGLANG_BASE_URL}/v1/models to get available models...")
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
    await client.aclose()
    print("HTTPX client closed.")

# --- FastAPI Application Setup ---
app = FastAPI(
    title="SGLang Sampling Parameter Override Middleware",
    description="A middleware server to override SGLang sampling parameters for generation requests.",
    version="1.0.0",
    lifespan=lifespan # Register the lifespan context manager
)

@app.get("/")
async def read_root():
    """
    Root endpoint for a basic health check and to display middleware configuration.
    """
    return {
        "message": "SGLang Sampling Parameter Override Middleware is running.",
        "sglang_backend": SGLANG_BASE_URL,
        "middleware_port": MIDDLEWARE_PORT,
        "default_sampling_params": DEFAULT_SAMPLING_PARAMS,
        "enforced_sampling_params": ENFORCED_SAMPLING_PARAMS,
        "model_sampling_params_configured": list(MODEL_SAMPLING_PARAMS.keys()),
        "generation_endpoints_monitored": GENERATION_ENDPOINTS,
        "anthropic_endpoints_handled_locally": ANTHROPIC_ENDPOINTS,
        "debug_logs_enabled": ENABLE_DEBUG_LOGS,
    }

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def proxy_sglang_requests(path: str, request: Request):
    """
    Catch-all route to proxy all incoming requests to the SGLang backend.
    For POST requests to configured generation endpoints, it applies
    the sampling parameter override logic.
    Supports streaming responses from the SGLang backend back to the client.
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

    # Prepare headers for the outgoing request to SGLang.
    # We copy the incoming headers and remove 'host' and 'content-length'
    # as httpx will manage these for the new request.
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None) # httpx will recalculate if body changes
    if ENABLE_DEBUG_LOGS:
        print(f"DEBUG: Outgoing Request Headers (initial): {headers}")

    request_content = None # This will hold the request body to be sent to sglang
    is_generation_request = False
    is_anthropic_request = False # Initialize Anthropic request flag
    incoming_json_body = {} # Initialize in case it's not a POST/JSON request

    # Determine if the current request path is a recognized generation endpoint
    is_generation_request = path in GENERATION_ENDPOINTS
    is_anthropic_request = path == "v1/messages" # Check if this is an Anthropic request
    if ENABLE_DEBUG_LOGS:
        print(f"DEBUG: is_generation_request after check: {is_generation_request}")
        print(f"DEBUG: is_anthropic_request: {is_anthropic_request}")

    # Construct the target URL for the SGLang backend
    # Redirect Anthropic requests to OpenAI chat completions endpoint
    if is_anthropic_request:
        # Convert /v1/messages to /v1/chat/completions for SGLang backend
        target_path = "v1/chat/completions"
        if ENABLE_DEBUG_LOGS:
            print(f"DEBUG: Redirecting Anthropic request from {original_path} to {target_path}")
    else:
        target_path = original_path
    
    # Ensure the query string is encoded to bytes as required by httpx.URL
    target_url = httpx.URL(path=target_path, query=request.url.query.encode("utf-8"))
    if ENABLE_DEBUG_LOGS:
        print(f"DEBUG: Target SGLang URL: {target_url}")

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
                    #if ENABLE_DEBUG_LOGS:
                    print(f"DEBUG: Overriding '{param}' to '{value_to_apply}' (was not in request).")
                else:
                    #if ENABLE_DEBUG_LOGS:
                    print(f"DEBUG: Parameter '{param}' already present in request: {current_params_container[param]}. Not overriding.")

            # Re-integrate the modified parameters back into the main body if they were nested
            if is_nested_params:
                incoming_json_body["sampling_params"] = current_params_container
            # If not nested, they are already updated in incoming_json_body

            request_content = json.dumps(incoming_json_body) # Serialize the modified JSON back to a string
            headers["content-type"] = "application/json" # Ensure content-type header is correct

            if ENABLE_DEBUG_LOGS:
                print(f"DEBUG: Final modified request body (to SGLang): {request_content}")
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
                print(f"DEBUG: Sending {'streaming' if is_streaming_request else 'non-streaming'} request to SGLang.")
            
            if is_streaming_request:
                # For streaming requests, use streaming
                sglang_request_obj = client.build_request(
                    method=request.method,
                    url=target_url,
                    headers=headers,
                    params=request.query_params,
                    content=request_content,
                )
                # Send the request and get the raw response object, enabling streaming
                sglang_response = await client.send(sglang_request_obj, stream=True)
            else:
                # For non-streaming requests, fetch the full response
                sglang_response = await client.request(
                    method=request.method,
                    url=target_url,
                    headers=headers,
                    params=request.query_params,
                    content=request_content,
                )

            if is_streaming_request:
                # Handle streaming response
                # Prepare response headers for streaming
                response_headers = dict(sglang_response.headers)
                if ENABLE_DEBUG_LOGS:
                    print(f"DEBUG: SGLang Response Headers (raw): {response_headers}")

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
                        async for chunk in sglang_response.aiter_bytes():
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
                            print(f"DEBUG: SGLang response connection closed by generator after {chunk_count} chunks.")
                        await sglang_response.aclose()

                return StreamingResponse(
                    stream_and_close_response(), # Use the local async generator
                    status_code=sglang_response.status_code,
                    headers=response_headers,
                    media_type=response_headers.get("content-type"),
                )
            else:
                # Handle non-streaming response
                if ENABLE_DEBUG_LOGS:
                    print(f"DEBUG: SGLang Response Headers (full): {sglang_response.headers}")
                    print(f"DEBUG: SGLang Response Status: {sglang_response.status_code}")
                    print(f"DEBUG: SGLang Response Content: {sglang_response.text}")
                
                # Handle Anthropic response conversion for non-streaming requests
                response_content = sglang_response.content
                
                # Log 404 errors specifically for debugging
                if sglang_response.status_code == 404:
                    if is_anthropic_request:
                        print(f"WARNING: Anthropic request to {target_path} returned 404. SGLang backend may not support OpenAI chat completions endpoint.")
                    else:
                        print(f"WARNING: Request to {target_path} returned 404. Endpoint may not exist on SGLang backend.")
                
                if is_anthropic_request and sglang_response.status_code == 200:
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
                await sglang_response.aclose()
                return Response(
                    content=response_content,
                    status_code=sglang_response.status_code,
                    headers=sglang_response.headers,
                    media_type=sglang_response.headers.get("content-type"),
                )
        else:
            if ENABLE_DEBUG_LOGS:
                print("DEBUG: Sending non-generation request to SGLang.")
            # For all other requests (e.g., GET /v1/models), fetch the full response
            sglang_response = await client.request(
                method=request.method,
                url=target_url, # Use original_path for the actual request
                headers=headers,
                params=request.query_params,
                content=request_content,
            )
            if ENABLE_DEBUG_LOGS:
                print(f"DEBUG: SGLang Response Headers (full): {sglang_response.headers}")
                print(f"DEBUG: SGLang Response Status: {sglang_response.status_code}")
            
            # Log 404 errors specifically for debugging
            if sglang_response.status_code == 404:
                print(f"WARNING: Non-generation request to {target_path} returned 404. Endpoint may not exist on SGLang backend.")
            
            # Ensure the httpx response is closed after its content is read
            await sglang_response.aclose()
            return Response(
                content=sglang_response.content,
                status_code=sglang_response.status_code,
                headers=sglang_response.headers,
                media_type=sglang_response.headers.get("content-type"),
            )

    except httpx.ConnectError as e:
        print(f"ERROR: [{request.method} {original_path}] Connection error to SGLANG backend: {e}")
        return Response(f"Could not connect to SGLang backend at {SGLANG_BASE_URL}: {e}",
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
    except httpx.RequestError as e:
        print(f"ERROR: [{request.method} {original_path}] Request error to SGLANG backend: {e}")
        return Response(f"An error occurred while requesting SGLang backend: {e}",
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
    except Exception as e:
        print(f"ERROR: [{request.method} {original_path}] An unexpected error occurred: {e}")
        return Response(f"An unexpected error occurred: {e}",
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

# --- Main execution block for direct script execution ---
if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description="SGLang Sampling Parameter Override Middleware Server."
    )
    parser.add_argument(
        "--host",
        type=str,
        default=MIDDLEWARE_HOST,
        help="Host address for the middleware server (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=MIDDLEWARE_PORT,
        help="Port for the middleware server (default: 8001)",
    )
    parser.add_argument(
        "--sglang-host",
        type=str,
        default=SGLANG_HOST,
        help="Host address for the SGLang backend (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--sglang-port",
        type=str, # Keep as string as it's used in f-string for URL
        default=SGLANG_PORT,
        help="Port for the SGLang backend (default: 8000)",
    )
    parser.add_argument(
        "--debug-logs",
        "-d",
        action="store_true", # This makes it a boolean flag
        help="Enable detailed debug logging.",
    )
    parser.add_argument(
        "--enforce-params",
        "-e",
        type=str,
        help="Enforce specific sampling parameters as JSON string. Example: '{\"temperature\": 0.7, \"top_p\": 0.9}'",
    )

    args = parser.parse_args()

    # Override global constants with command-line arguments
    MIDDLEWARE_HOST = args.host
    MIDDLEWARE_PORT = args.port
    SGLANG_HOST = args.sglang_host
    SGLANG_PORT = args.sglang_port
    SGLANG_BASE_URL = f"http://{SGLANG_HOST}:{SGLANG_PORT}" # Reconstruct with new values
    ENABLE_DEBUG_LOGS = args.debug_logs # Set debug logs based on argument
    
    # Parse enforced parameters from command line if provided
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

    print(f"Starting SGLang middleware server on http://{MIDDLEWARE_HOST}:{MIDDLEWARE_PORT}")
    print(f"Proxying requests to SGLang backend at {SGLANG_BASE_URL}")
    print(f"Debug logs are {'ENABLED' if ENABLE_DEBUG_LOGS else 'DISABLED'}.")
    uvicorn.run(app, host=MIDDLEWARE_HOST, port=MIDDLEWARE_PORT)
