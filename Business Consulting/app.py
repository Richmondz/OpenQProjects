# C:\Users\fishe\OneDrive\Documents\Domain Project\app.py
import os
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask, render_template, jsonify
from ai_routes import ai_bp 
from werkzeug.middleware.dispatcher import DispatcherMiddleware # For Prometheus
from prometheus_client import make_wsgi_app, Counter, Histogram # For Prometheus
import time # For Prometheus request timing (though timing is done in ai_routes.py)

# --- Logging Configuration ---
if not os.path.exists('logs'):
    os.mkdir('logs') # Directory will be created, but file logging is disabled below for now

log_formatter = logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
)

# File Handler (optional, logs to a file -  COMMENTED OUT TO PREVENT WinError 32 with reloader)
# file_handler = RotatingFileHandler(
#     'logs/app.log', maxBytes=10240, backupCount=10
# )
# file_handler.setFormatter(log_formatter)
# file_handler.setLevel(logging.INFO)

# Stream Handler (logs to console)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)
stream_handler.setLevel(logging.DEBUG) # Log DEBUG level and above to console

# Get the root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG) # Set root logger to DEBUG to capture all levels
root_logger.handlers.clear() # Clear any existing handlers (e.g. Flask's default)
root_logger.addHandler(stream_handler)
# root_logger.addHandler(file_handler) # Keep this commented out to avoid the PermissionError during dev

# --- Flask App Initialization ---
app = Flask(__name__)
logging.info("Flask app initialized.") # This will use the root_logger settings

# Register the AI blueprint
app.register_blueprint(ai_bp)
logging.info("AI Blueprint registered.")

# --- Prometheus Metrics Definitions ---
# These are defined here so they are created once when the app module is loaded.
API_ASK_REQUESTS = Counter(
    'api_ask_requests_total',
    'Total number of requests to the /api/ask endpoint',
    ['method', 'endpoint', 'service_context'] 
)
API_ASK_LATENCY = Histogram(
    'api_ask_request_latency_seconds',
    'Latency of requests to the /api/ask endpoint',
    ['endpoint', 'service_context']
)
API_ASK_ERRORS = Counter(
    'api_ask_errors_total',
    'Total number of errors in the /api/ask endpoint',
    ['endpoint', 'service_context', 'error_type']
)
RAG_CACHE_HITS = Counter( 
    'rag_cache_hits_total',
    'Total RAG cache hits for /api/ask',
    ['endpoint', 'service_context']
)
RAG_RETRIEVED_CHUNKS = Histogram( 
    'rag_retrieved_chunks_count',
    'Number of chunks retrieved by RAG for /api/ask',
    ['endpoint', 'service_context'],
    buckets=(0, 1, 2, 3, 4, 5, float("inf")) 
)


# --- Routes ---
@app.route("/")
def index():
    logging.info("Index route '/' accessed.")
    return render_template("index.html", project_name="Business Consultation")

@app.route("/api/hello", methods=["GET"])
def hello():
    logging.info("API route '/api/hello' accessed.")
    return jsonify({"msg": "Hello, world!"})

# --- WSGI App with Prometheus Metrics ---
# Add the Prometheus WSGI middleware to your app.
# This will create a /metrics endpoint.
app_dispatch = DispatcherMiddleware(app, {
    '/metrics': make_wsgi_app()
})
logging.info("Prometheus /metrics endpoint configured.")


if __name__ == "__main__":
    logging.info("Starting AI Business Consultant application...")
    from werkzeug.serving import run_simple
    # Running with use_reloader=True is convenient for development.
    # The file logging PermissionError should be resolved by not adding the file_handler.
    run_simple(hostname="0.0.0.0", port=5000, application=app_dispatch, use_reloader=True, use_debugger=True)
