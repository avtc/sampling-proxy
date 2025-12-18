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
SGLANG_HOST = os.getenv("SGLANG_HOST", "127.0.0.1")
SGLANG_PORT = os.getenv("SGLANG_PORT", "8000")
SGLANG_BASE_URL = f"http://{SGLANG_HOST}:{SGLANG_PORT}"

# Middleware server address and port
MIDDLEWARE_HOST = os.getenv("MIDDLEWARE_HOST", "0.0.0.0")
MIDDLEWARE_PORT = int(os.getenv("MIDDLEWARE_PORT", "8001"))

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
]

# Initialize an httpx AsyncClient for making requests to the SGLang backend.
# This client is designed for efficient connection pooling.
# A higher timeout is set to accommodate potentially long LLM generation times.
client = httpx.AsyncClient(base_url=SGLANG_BASE_URL, timeout=1200.0)

# --- FastAPI Application Lifespan Setup ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events for the FastAPI application.
    Ensures the httpx client is properly closed when the application shuts down.
    """
    print("FastAPI application startup.")
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
        "model_sampling_params_configured": list(MODEL_SAMPLING_PARAMS.keys()),
        "generation_endpoints_monitored": GENERATION_ENDPOINTS,
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

    # Construct the target URL for the SGLang backend
    # Ensure the query string is encoded to bytes as required by httpx.URL
    target_url = httpx.URL(path=original_path, query=request.url.query.encode("utf-8")) # Use original_path for target URL
    if ENABLE_DEBUG_LOGS:
        print(f"DEBUG: Target SGLang URL: {target_url}")

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
    incoming_json_body = {} # Initialize in case it's not a POST/JSON request

    # Determine if the current request path is a recognized generation endpoint
    is_generation_request = path in GENERATION_ENDPOINTS
    if ENABLE_DEBUG_LOGS:
        print(f"DEBUG: is_generation_request after check: {is_generation_request}")

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
            else: # v1/completions, v1/chat/completions (normalized paths)
                current_params_container = incoming_json_body
                is_nested_params = False
                if ENABLE_DEBUG_LOGS:
                    print(f"DEBUG: Path is OpenAI-compatible, using top-level params. Current container: {current_params_container}")

            model_specific_params = MODEL_SAMPLING_PARAMS.get(model_name, {})
            if ENABLE_DEBUG_LOGS:
                print(f"DEBUG: Model-specific params for '{model_name}': {model_specific_params}")

            # Iterate through all parameters defined in DEFAULT_SAMPLING_PARAMS.
            # Apply overrides if the parameter is not already present in the incoming request.
            for param, default_value in DEFAULT_SAMPLING_PARAMS.items():
                if param not in current_params_container:
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
            if ENABLE_DEBUG_LOGS:
                print("DEBUG: Sending streaming request to SGLang.")
            # For generation requests (POST to GENERATION_ENDPOINTS), use streaming
            # Build the request object
            sglang_request_obj = client.build_request(
                method=request.method,
                url=target_url, # Use original_path for the actual request
                headers=headers,
                params=request.query_params,
                content=request_content,
            )
            # Send the request and get the raw response object, enabling streaming
            sglang_response = await client.send(sglang_request_obj, stream=True)

            # Prepare response headers for streaming
            response_headers = dict(sglang_response.headers)
            if ENABLE_DEBUG_LOGS:
                print(f"DEBUG: SGLang Response Headers (raw): {response_headers}")

            response_headers.pop("content-length", None) # Remove Content-Length for streaming
            response_headers.pop("transfer-encoding", None) # Remove Transfer-Encoding for streaming

            # Explicitly set Content-Type for SSE if it's a streaming chat/completion request
            # Check if the original request intended streaming (from incoming_json_body)
            if incoming_json_body.get("stream") is True:
                # Use original_path for this check, as it's the actual path in the request
                if original_path.strip('/') in ["v1/chat/completions", "v1/completions"]:
                    response_headers["content-type"] = "text/event-stream"
                    response_headers["cache-control"] = "no-cache"
                    response_headers["connection"] = "keep-alive"
                    if ENABLE_DEBUG_LOGS:
                        print(f"DEBUG: Setting response Content-Type to 'text/event-stream' for streaming request.")
                else:
                    if ENABLE_DEBUG_LOGS:
                        print(f"DEBUG: Not an OpenAI-compatible streaming path, keeping original Content-Type: {response_headers.get('content-type', 'N/A')}")
            else:
                if ENABLE_DEBUG_LOGS:
                    print(f"DEBUG: Original request did not ask for streaming. Keeping original Content-Type: {response_headers.get('content-type', 'N/A')}")


            # Define a local async generator to yield chunks and close the httpx response
            async def stream_and_close_response():
                chunk_count = 0
                try:
                    async for chunk in sglang_response.aiter_bytes():
                        chunk_count += 1
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
            if ENABLE_DEBUG_LOGS:
                print("DEBUG: Sending non-streaming request to SGLang.")
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

    args = parser.parse_args()

    # Override global constants with command-line arguments
    MIDDLEWARE_HOST = args.host
    MIDDLEWARE_PORT = args.port
    SGLANG_HOST = args.sglang_host
    SGLANG_PORT = args.sglang_port
    SGLANG_BASE_URL = f"http://{SGLANG_HOST}:{SGLANG_PORT}" # Reconstruct with new values
    ENABLE_DEBUG_LOGS = args.debug_logs # Set debug logs based on argument

    print(f"Starting SGLang middleware server on http://{MIDDLEWARE_HOST}:{MIDDLEWARE_PORT}")
    print(f"Proxying requests to SGLang backend at {SGLANG_BASE_URL}")
    print(f"Debug logs are {'ENABLED' if ENABLE_DEBUG_LOGS else 'DISABLED'}.")
    uvicorn.run(app, host=MIDDLEWARE_HOST, port=MIDDLEWARE_PORT)
