"""
Gunicorn configuration for the backend container / production deploys.

Values chosen for this workload:
  - The classifier + retrieval index are loaded lazily per worker
    (first request warms them), so workers are cheap to start but each
    holds its own copy of the artifacts in memory. sync workers with a
    modest count match the CPU-bound, low-concurrency profile of a
    demo/take-home deployment; scale `WEB_CONCURRENCY` for real traffic.
  - timeout 120s: live-LLM calls with retries can take tens of seconds;
    never let the worker be killed mid-LLM-call.
  - graceful_timeout 30s + graceful shutdown: in-flight requests finish
    (including any live LLM call) before the worker exits on SIGTERM.
  - access log with X-Request-ID: correlates a gunicorn access line with
    the application's structured JSON logs for the same request.
"""
import multiprocessing
import os

bind = os.environ.get("GUNICORN_BIND", "0.0.0.0:8000")
workers = int(os.environ.get("WEB_CONCURRENCY", multiprocessing.cpu_count() * 2 + 1))
worker_class = "sync"
timeout = int(os.environ.get("GUNICORN_TIMEOUT", "120"))
graceful_timeout = 30
keepalive = 5
max_requests = 1000          # recycle workers periodically (defensive vs slow leaks)
max_requests_jitter = 100

accesslog = "-"
errorlog = "-"
access_log_format = '%({x-request-id}i)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s %(M)sms'
loglevel = os.environ.get("LOG_LEVEL", "info")

# Forward SIGTERM as graceful shutdown (default behavior, made explicit
# here so container orchestrators' stop timeouts line up).
preload_app = False  # artifacts load lazily per worker; do not preload (each worker gets its own copy)
