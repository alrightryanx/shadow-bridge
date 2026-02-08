"""Flask app factory for ShadowBridge web dashboard and task API."""

import logging
from flask import Flask
from flask_socketio import SocketIO
from flask_cors import CORS

log = logging.getLogger(__name__)

socketio = SocketIO(cors_allowed_origins="*", async_mode="threading")


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__,
                template_folder='templates',
                static_folder='static')
    app.config['SECRET_KEY'] = 'shadow-ai-web-key'
    CORS(app)

    # Register blueprints
    from web.routes.api import api_bp
    from web.routes.websocket import ws_bp
    app.register_blueprint(api_bp)
    app.register_blueprint(ws_bp)

    socketio.init_app(app)
    app.socketio_enabled = True

    # Register SocketIO event handlers after app is created
    from web.routes.websocket import register_socketio_handlers
    register_socketio_handlers(socketio)

    log.info("Flask app created with SocketIO and task API routes")
    return app
