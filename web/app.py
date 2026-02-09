"""Flask app factory for ShadowBridge web dashboard and task API."""

import logging
import secrets
import time
from flask import Flask, request as flask_request
from flask_socketio import SocketIO
from flask_cors import CORS

log = logging.getLogger(__name__)

socketio = SocketIO(
    cors_allowed_origins=["http://localhost:*", "http://127.0.0.1:*"],
    async_mode="threading",
)


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__,
                template_folder='templates',
                static_folder='static')
    app.config['SECRET_KEY'] = secrets.token_hex(32)
    CORS(app, origins=["http://localhost:*", "http://127.0.0.1:*"])

    # Register blueprints
    from web.routes.api import api_bp
    from web.routes.websocket import ws_bp
    from web.routes.predictive import predictive_bp
    from web.routes.ouroboros_api import ouroboros_bp
    from web.routes.knowledge import knowledge_bp
    app.register_blueprint(api_bp)
    app.register_blueprint(ws_bp)
    app.register_blueprint(predictive_bp)
    app.register_blueprint(ouroboros_bp)
    app.register_blueprint(knowledge_bp)

    # Register auth middleware
    from web.middleware.auth import require_auth
    app.before_request(require_auth)

    # Register rate limiting middleware
    from web.middleware.rate_limit import check_rate_limit
    app.before_request(check_rate_limit)

    # Audit logging for API requests
    @app.after_request
    def audit_log(response):
        if flask_request.path.startswith("/api/"):
            log.info(
                f"API {flask_request.method} {flask_request.path} "
                f"from={flask_request.remote_addr} status={response.status_code}"
            )
        return response

    socketio.init_app(app)
    app.socketio_enabled = True

    # Register SocketIO event handlers after app is created
    from web.routes.websocket import register_socketio_handlers
    register_socketio_handlers(socketio)

    log.info("Flask app created with SocketIO, auth, rate limiting, and task API routes")
    return app
