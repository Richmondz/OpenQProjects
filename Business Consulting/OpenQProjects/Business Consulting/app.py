# C:\Users\fishe\OneDrive\Documents\Project\OpenQProjects\Business Consulting\app.py
import os
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask, render_template, jsonify
from flask_login import LoginManager
from flask_bcrypt import Bcrypt

# --- Import Blueprints, Models, etc. ---
from ai_routes import ai_bp
from auth_routes import auth_bp
from models import User, Session, init_db as initialize_database, bcrypt as bcrypt_from_models

# --- Logging Configuration ---
if not os.path.exists('logs'):
    os.mkdir('logs')

log_formatter = logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)
stream_handler.setLevel(logging.DEBUG)

root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
root_logger.handlers.clear()
root_logger.addHandler(stream_handler)


# --- Flask App Initialization ---
app = Flask(__name__)
logging.info("Flask app initialized.")

# --- Configuration for Security ---
# IMPORTANT: In a real app, this MUST come from an environment variable.
app.config['SECRET_KEY'] = 'a-very-secret-and-long-random-string-for-sessions'


# --- Initialize Extensions ---
bcrypt_from_models.init_app(app) # Initialize bcrypt with the app context

login_manager = LoginManager(app)
login_manager.login_view = 'auth_bp.login' # Redirect to this page if a user tries to access a protected page
login_manager.login_message_category = 'info'

@login_manager.user_loader
def load_user(user_id):
    session = Session()
    user = session.query(User).get(int(user_id))
    session.close()
    return user


# --- Register Blueprints ---
app.register_blueprint(ai_bp)
app.register_blueprint(auth_bp)
logging.info("AI and Auth Blueprints registered.")


# --- Prometheus Metrics Definitions ---
from prometheus_client import Counter, Histogram
# (Prometheus metrics code is unchanged)
API_ASK_REQUESTS = Counter('api_ask_requests_total', 'Total requests to /api/ask', ['method', 'endpoint', 'service_context'])
API_ASK_LATENCY = Histogram('api_ask_request_latency_seconds', 'Latency of /api/ask', ['endpoint', 'service_context'])
API_ASK_ERRORS = Counter('api_ask_errors_total', 'Errors on /api/ask', ['endpoint', 'service_context', 'error_type'])
RAG_CACHE_HITS = Counter('rag_cache_hits_total', 'RAG cache hits for /api/ask', ['endpoint', 'service_context'])
RAG_RETRIEVED_CHUNKS = Histogram('rag_retrieved_chunks_count', 'Chunks retrieved by RAG', ['endpoint', 'service_context'], buckets=(0, 1, 2, 3, 4, 5, float("inf")))


# --- Main Routes ---
@app.route("/")
def index():
    return render_template("index.html", project_name="Business Consultation")

@app.route("/api/hello", methods=["GET"])
def hello():
    return jsonify({"msg": "Hello, world!"})


# --- WSGI App with Prometheus Metrics ---
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from prometheus_client import make_wsgi_app

app_dispatch = DispatcherMiddleware(app, {
    '/metrics': make_wsgi_app()
})


# --- Main Execution Block ---
if __name__ == "__main__":
    with app.app_context():
        initialize_database() # Creates both 'users' and 'ai_cache' tables if they don't exist
        
    logging.info("Starting AI Business Consultant application...")
    from werkzeug.serving import run_simple
    run_simple(hostname="0.0.0.0", port=5000, application=app_dispatch, use_reloader=True, use_debugger=True)