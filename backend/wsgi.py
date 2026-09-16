"""
Production WSGI entry point.

Run with gunicorn (never `python -m app.api.app` in production):
    gunicorn -c gunicorn.conf.py wsgi:application

`application` is the Flask app object itself -- lazy-loading the model
artifacts on first request (see app.api.app.get_agent) keeps worker boot
fast and lets the orchestrator's healthcheck confirm liveness before the
(heavy) first inference warms the worker.
"""
from app.api.app import app as application

if __name__ == "__main__":
    application.run(host="0.0.0.0", port=8000, debug=False)
