import os
import asyncio
import httpx
import json
import logging
import time
import psutil
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

import uvicorn
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# Configurable Ollama host (via env variable or defaults to localhost)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

logging.basicConfig()
logger = logging.getLogger(__name__)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

app = FastAPI()

# Request metrics
OLLAMA_REQUEST_COUNT = Counter("ollama_requests_total", "Total requests", ["model"])
OLLAMA_ERROR_COUNT = Counter("ollama_requests_failed_total", "Total failed requests", ["model", "error_type"])

# Response time metrics
OLLAMA_TOTAL_DURATION = Histogram("ollama_response_seconds", "Total time spent for the response", ["model"])
OLLAMA_LOAD_DURATION = Histogram("ollama_load_duration_seconds", "Time spent loading the model", ["model"])
OLLAMA_PROMPT_EVAL_DURATION = Histogram("ollama_prompt_eval_duration_seconds", "Time spent evaluating prompt", ["model"])
OLLAMA_EVAL_DURATION = Histogram("ollama_eval_duration_seconds", "Time spent generating the response", ["model"])

# Token metrics
OLLAMA_PROMPT_EVAL_COUNT = Counter("ollama_tokens_processed_total", "Number of tokens in the prompt", ["model"])
OLLAMA_EVAL_COUNT = Counter("ollama_tokens_generated_total", "Number of tokens in the response", ["model"])
OLLAMA_TOKENS_PER_SECOND = Histogram(
    "ollama_tokens_per_second",
    "Tokens generated per second",
    ["model"],
    buckets=[5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
)

# System resource metrics (Gauge for current values)
OLLAMA_SYSTEM_CPU_USAGE = Gauge("ollama_system_cpu_percent", "Current CPU usage percentage")
OLLAMA_SYSTEM_MEMORY_USAGE = Gauge("ollama_system_memory_percent", "Current memory usage percentage")
OLLAMA_SYSTEM_MEMORY_AVAILABLE = Gauge("ollama_system_memory_available_bytes", "Available memory in bytes")

# Model inventory metrics
OLLAMA_MODELS_COUNT = Gauge("ollama_models_count", "Total number of models available")
OLLAMA_MODELS_SIZE = Gauge("ollama_models_size_bytes", "Total size of all models in bytes")

# Request queue metrics
OLLAMA_ACTIVE_REQUESTS = Gauge("ollama_active_requests", "Number of currently active requests")
OLLAMA_REQUEST_QUEUE_LENGTH = Gauge("ollama_request_queue_length", "Number of requests waiting in queue")

# Request latency percentiles
OLLAMA_REQUEST_LATENCY = Histogram(
    "ollama_request_latency_seconds",
    "Request latency in seconds",
    buckets=[0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600, 900],
)


def extract_and_record_metrics(response_data, model):
    """Extract and record metrics from Ollama response data."""
    if not isinstance(response_data, dict):
        return

    # https://github.com/ollama/ollama/blob/main/docs/api.md#response
    total_duration = response_data.get("total_duration", 0)  # total time spent in nanoseconds generating the response
    load_duration = response_data.get("load_duration", 0)  # time spent in nanoseconds loading the model
    prompt_eval_duration = response_data.get("prompt_eval_duration", 0)  # time spent in nanoseconds evaluating the prompt
    prompt_eval_count = response_data.get("prompt_eval_count", 0)  # number of tokens in the prompt
    eval_duration = response_data.get("eval_duration", 0)  # time spent in nanoseconds generating the response
    eval_count = response_data.get("eval_count", 0)  # number of tokens in the response

    if total_duration > 0:
        total_duration_seconds = total_duration / 1_000_000_000
        OLLAMA_TOTAL_DURATION.labels(model=model).observe(total_duration_seconds)
        logger.debug(f"Model: {model}, Total Duration: {total_duration_seconds:.2f} seconds")
    if load_duration > 0:
        load_duration_seconds = load_duration / 1_000_000_000
        OLLAMA_LOAD_DURATION.labels(model=model).observe(load_duration_seconds)
        logger.debug(f"Model: {model}, Load Duration: {load_duration_seconds:.2f} seconds")
    if prompt_eval_duration > 0:
        prompt_eval_time_seconds = prompt_eval_duration / 1_000_000_000
        OLLAMA_PROMPT_EVAL_DURATION.labels(model=model).observe(prompt_eval_time_seconds)
        logger.debug(f"Model: {model}, Prompt Eval Duration: {prompt_eval_time_seconds:.2f} seconds")
    if prompt_eval_count > 0:
        OLLAMA_PROMPT_EVAL_COUNT.labels(model=model).inc(prompt_eval_count)
        logger.debug(f"Model: {model}, Prompt Eval Count: {prompt_eval_count}")
    if eval_duration > 0:
        eval_duration_seconds = eval_duration / 1_000_000_000
        OLLAMA_EVAL_DURATION.labels(model=model).observe(eval_duration_seconds)
        logger.debug(f"Model: {model}, Eval Duration: {eval_duration_seconds:.2f} seconds")
    if eval_count > 0:
        OLLAMA_EVAL_COUNT.labels(model=model).inc(eval_count)
        logger.debug(f"Model: {model}, Eval Count: {eval_count}")
    if eval_duration > 0 and eval_count > 0:
        tps = eval_count / eval_duration * 1_000_000_000
        OLLAMA_TOKENS_PER_SECOND.labels(model=model).observe(tps)
        logger.debug(f"Model: {model}, Tokens per Second: {tps:.2f}")


async def update_system_metrics():
    """Update system resource metrics."""
    try:
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        OLLAMA_SYSTEM_CPU_USAGE.set(cpu_percent)
        logger.debug(f"System CPU Usage: {cpu_percent:.2f}%")

        # Memory usage
        memory = psutil.virtual_memory()
        OLLAMA_SYSTEM_MEMORY_USAGE.set(memory.percent)
        OLLAMA_SYSTEM_MEMORY_AVAILABLE.set(memory.available)
        logger.debug(f"System Memory: {memory.percent:.2f}% used, {memory.available / (1024**3):.2f} GB available")
    except Exception as e:
        logger.warning(f"Failed to update system metrics: {e}")


async def update_model_inventory():
    """Update model inventory metrics by fetching from Ollama API."""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            response = await client.get(f"{OLLAMA_HOST}/api/tags")
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])
                model_count = len(models)
                total_size = sum(m.get("size", 0) for m in models)

                OLLAMA_MODELS_COUNT.set(model_count)
                OLLAMA_MODELS_SIZE.set(total_size)
                logger.info(f"Model inventory updated: {model_count} models, {total_size / (1024**3):.2f} GB total size")
            else:
                logger.warning(f"Failed to fetch model inventory. Status code: {response.status_code}")
    except Exception as e:
        logger.warning(f"Failed to update model inventory: {e}")


# Start background tasks for periodic metric updates
class BackgroundTasks:
    def __init__(self):
        self.system_metrics_task = None
        self.model_inventory_task = None
        self._running = False

    async def start(self):
        self._running = True
        self.system_metrics_task = asyncio.create_task(self._system_metrics_loop())
        self.model_inventory_task = asyncio.create_task(self._model_inventory_loop())

    async def stop(self):
        self._running = False
        if self.system_metrics_task:
            self.system_metrics_task.cancel()
        if self.model_inventory_task:
            self.model_inventory_task.cancel()

    async def _system_metrics_loop(self):
        """Periodically update system metrics."""
        while self._running:
            try:
                await update_system_metrics()
            except asyncio.CancelledError:
                break
            await asyncio.sleep(10)  # Update every 10 seconds

    async def _model_inventory_loop(self):
        """Periodically update model inventory."""
        while self._running:
            try:
                await update_model_inventory()
            except asyncio.CancelledError:
                break
            await asyncio.sleep(60)  # Update every 60 seconds


background_tasks = BackgroundTasks()


@app.get("/metrics")
def metrics():
    """Expose Prometheus metrics."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/api/health")
def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "healthy", "ollama_host": OLLAMA_HOST}


@app.get("/api/models")
async def list_models():
    """List all available models from Ollama."""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            response = await client.get(f"{OLLAMA_HOST}/api/tags")
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Failed to fetch models: {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/chat")
@app.post("/api/generate")
async def chat_with_metrics(request: Request):
    """Handle chat and generate requests with streaming support and metrics extraction."""
    body = await request.json()
    model = body.get("model", "unknown")
    # logger.debug(f"Chat request body: {json.dumps(body, indent=4)}")
    is_streaming = body.get("stream", False)

    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)
    headers.pop("content-type", None)

    # Track active requests
    OLLAMA_ACTIVE_REQUESTS.inc()
    OLLAMA_REQUEST_COUNT.labels(model=model).inc()

    start_time = time.time()

    try:
        if is_streaming:
            async def generate_stream():
                endpoint = request.url.path  # /api/chat or /api/generate
                async with httpx.AsyncClient(timeout=httpx.Timeout(900.0, read=900.0)) as client:
                    async with client.stream("POST", f"{OLLAMA_HOST}{endpoint}", headers=headers, json=body, params=request.query_params) as response:

                        final_chunk_data = None

                        async for chunk in response.aiter_bytes():
                            # Forward the chunk immediately to the client
                            yield chunk

                            # Try to parse the chunk to look for metrics
                            if chunk:
                                try:
                                    chunk_text = chunk.decode('utf-8')
                                    lines = chunk_text.strip().split('\n')

                                    for line in lines:
                                        if line.strip():
                                            try:
                                                chunk_json = json.loads(line)
                                                # Check if this is the final chunk (contains "done": true)
                                                if chunk_json.get("done", False):
                                                    final_chunk_data = chunk_json
                                            except json.JSONDecodeError:
                                                continue

                                except UnicodeDecodeError:
                                    pass

                        # Extract metrics from the final chunk if available
                        if final_chunk_data:
                            extract_and_record_metrics(final_chunk_data, model)

            return StreamingResponse(generate_stream(), media_type="application/json")
        else:
            endpoint = request.url.path  # /api/chat or /api/generate
            async with httpx.AsyncClient(timeout=httpx.Timeout(900.0, read=900.0)) as client:
                response = await client.post(f"{OLLAMA_HOST}{endpoint}", headers=headers, json=body, params=request.query_params)

                if response.status_code == 200:
                    try:
                        response_data = response.json()
                        extract_and_record_metrics(response_data, model)
                    except (json.JSONDecodeError, TypeError):
                        pass

                return Response(content=response.content, status_code=response.status_code, headers=dict(response.headers))
    except Exception as e:
        # Record error metrics
        error_type = type(e).__name__.lower()
        OLLAMA_ERROR_COUNT.labels(model=model, error_type=error_type).inc()
        logger.error(f"Request failed for model {model}: {e}")
        raise
    finally:
        # Track request latency and active requests
        latency = time.time() - start_time
        OLLAMA_REQUEST_LATENCY.observe(latency)
        OLLAMA_ACTIVE_REQUESTS.dec()


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
async def simple_proxy(request: Request, path: str):
    """Simple pass-through proxy for all other endpoints."""
    logger.debug(f"Proxying {request.method} request to /{path}")
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)

    start_time = time.time()

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(900.0, read=900.0)) as client:
            response = await client.request(method=request.method, url=f"{OLLAMA_HOST}/{path}", headers=headers, content=await request.body(), params=request.query_params)

        logger.debug(f"Proxy response: {response.status_code} for {request.method} /{path}")
        return Response(content=response.content, status_code=response.status_code, headers=dict(response.headers))
    except Exception as e:
        logger.error(f"Proxy request failed for /{path}: {e}")
        raise
    finally:
        latency = time.time() - start_time
        OLLAMA_REQUEST_LATENCY.observe(latency)


async def verify_ollama_connection():
    """Verify connection to Ollama server at startup."""
    logger.debug(f"Verifying connection to Ollama server at {OLLAMA_HOST}")

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            response = await client.get(f"{OLLAMA_HOST}/api/version")
            if response.status_code == 200:
                version_data = response.json()
                logger.info(f"Connected to Ollama version: {version_data.get('version', 'unknown')}")
            else:
                logger.error(f"Failed to connect to Ollama server. Status code: {response.status_code}")
    except Exception as e:
        logger.error(f"Failed to connect to Ollama server at {OLLAMA_HOST}: {e}")
        logger.error("Please ensure Ollama is running and accessible at the configured host")


@app.on_event("startup")
async def startup_event():
    """Start background tasks on startup."""
    await verify_ollama_connection()
    await background_tasks.start()


@app.on_event("shutdown")
async def shutdown_event():
    """Stop background tasks on shutdown."""
    await background_tasks.stop()


async def main():
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
