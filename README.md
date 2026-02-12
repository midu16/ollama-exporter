# Ollama Prometheus Exporter

This is a **Prometheus Exporter** for **Ollama**, designed to monitor request statistics, response times, token usage, model performance, system resources, and model inventory. It runs as a FastAPI service and is **Docker-ready**.

## Features
- **Tracks requests per model** (`ollama_requests_total`)
- **Measures response time** (`ollama_response_seconds`)
- **Records model load times** (`ollama_load_duration_seconds`)
- **Tracks evaluation durations** (`ollama_prompt_eval_duration_seconds` and `ollama_eval_duration_seconds`)
- **Monitors token usage** (`ollama_tokens_processed_total` and `ollama_tokens_generated_total`)
- **Measures token generation rate** (`ollama_tokens_per_second`)
- **System resource monitoring**: CPU usage, memory usage, available memory
- **Model inventory**: Total models count and total size
- **Request concurrency**: Active requests and queue length
- **Error tracking**: Failed requests with error type classification
- **Request latency percentiles**: 50th, 90th, 95th, 99th percentiles
- **Transparent proxy** for all non-chat Ollama API endpoints
- **Health check endpoint** for monitoring
- **Model listing endpoint** to view available models

## Installation

### Running Locally

#### 1. Install Dependencies
```sh
pip install fastapi uvicorn prometheus_client httpx psutil
```
or 

```bash
pip install -r requirements.txt
```

#### 2. Run the Exporter
```sh
python ollama_exporter.py
```
By default, it connects to `http://localhost:11434` for Ollama.

### Running with Docker

#### 1. Build the Docker Image
```sh
docker build -t ollama-exporter .
```

#### 2. Run the Container
```sh
docker run -d --name ollama-exporter -p 8000:8000 \
  -e OLLAMA_HOST="http://192.168.1.100:11434" ollama-exporter
```

## Prometheus Integration

### Add to `prometheus.yml`
```yaml
scrape_configs:
  - job_name: 'ollama-metrics'
    static_configs:
      - targets: ['192.168.1.100:8000']
```
Restart Prometheus to apply changes:
```sh
docker restart <prometheus-container-name>
```

## Metrics

| Metric Name | Description | Type |
|-------------|-------------|------|
| `ollama_requests_total` | Total requests per model | Counter |
| `ollama_requests_failed_total` | Total failed requests with error type | Counter |
| `ollama_response_seconds` | Total time spent for the response | Histogram |
| `ollama_load_duration_seconds` | Time spent loading the model | Histogram |
| `ollama_prompt_eval_duration_seconds` | Time spent evaluating prompt | Histogram |
| `ollama_eval_duration_seconds` | Time spent generating the response | Histogram |
| `ollama_tokens_processed_total` | Number of tokens in the prompt | Counter |
| `ollama_tokens_generated_total` | Number of tokens in the response | Counter |
| `ollama_tokens_per_second` | Tokens generated per second | Histogram |
| `ollama_system_cpu_percent` | Current CPU usage percentage | Gauge |
| `ollama_system_memory_percent` | Current memory usage percentage | Gauge |
| `ollama_system_memory_available_bytes` | Available memory in bytes | Gauge |
| `ollama_models_count` | Total number of models available | Gauge |
| `ollama_models_size_bytes` | Total size of all models in bytes | Gauge |
| `ollama_active_requests` | Number of currently active requests | Gauge |
| `ollama_request_queue_length` | Number of requests waiting in queue | Gauge |
| `ollama_request_latency_seconds` | Request latency in seconds | Histogram |

## Grafana Integration
1. Open **Grafana**.
2. Go to **Dashboards → Import**.
3. Click **Upload JSON file** and select `dashboard.json` from the project directory.
4. Select your **Prometheus data source**.
5. Click **Import** to add the dashboard.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/metrics` | GET | Exposes Prometheus metrics |
| `/api/health` | GET | Health check endpoint |
| `/api/models` | GET | List all available models from Ollama |
| `/api/chat` | POST | Proxies requests to Ollama and logs metrics |
| `/api/generate` | POST | Proxies requests to Ollama and logs metrics |
| `/{path}` | Any | Proxies all other requests to Ollama API |

All other endpoints are proxied to the Ollama API.

## Usage Scenario

Suppose you want to monitor your local Ollama instance with Prometheus using this exporter:

1. **Start Ollama** locally (default: `http://localhost:11434`).
2. **Run the exporter** on your machine:
   ```sh
   OLLAMA_HOST=http://localhost:11434 python ollama_exporter.py
   # or with Docker:
   # docker run -d -p 8000:8000 -e OLLAMA_HOST="http://localhost:11434" ollama-exporter
   ```
3. **Configure your application (eg Open WebUI)** to use the exporter as the API endpoint:
   - Set `OLLAMA_HOST=http://localhost:8000` (the exporter will proxy and collect metrics).
4. **Prometheus** scrapes metrics from the exporter:
   - Add `localhost:8000/metrics` to your Prometheus scrape config.

This setup allows you to transparently monitor all Ollama API usage and performance via Prometheus and Grafana dashboards.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |

## License
There is no spoon.
