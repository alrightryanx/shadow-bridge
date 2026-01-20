"""
Shadow Web Dashboard - Flask Application Factory
"""

from flask import Flask, render_template, jsonify
import os
import sys

# SocketIO is optional - may not work in PyInstaller frozen builds
socketio = None
try:
    from flask_socketio import SocketIO

    socketio = SocketIO()
except ImportError:
    pass


def _debug_log(msg, base_path=None):
    """Write debug info to a log file next to the EXE."""
    try:
        log_path = os.path.join(
            base_path or os.path.dirname(sys.executable), "web_debug.log"
        )
        with open(log_path, "a") as f:
            f.write(f"{msg}\n")
    except (IOError, OSError):
        pass  # Logging is best-effort


def create_app():
    """Create and configure the Flask application."""
    global socketio

    # Determine base path - handle both normal and PyInstaller frozen environments
    if getattr(sys, "frozen", False):
        # Running as PyInstaller bundle - use executable's directory
        base_path = os.path.dirname(sys.executable)
        meipass_path = getattr(sys, "_MEIPASS", base_path)

        _debug_log(f"=== Web Dashboard Init ===", base_path)
        _debug_log(f"sys.executable: {sys.executable}", base_path)
        _debug_log(f"base_path: {base_path}", base_path)
        _debug_log(f"_MEIPASS: {meipass_path}", base_path)

        # PRIORITY: External web folder next to executable (allows updates without rebuild)
        external_web = os.path.join(base_path, "web")
        external_templates = os.path.join(external_web, "templates", "analytics.html")

        # Also check tools/web for legacy layout
        tools_web = os.path.join(base_path, "tools", "web")
        tools_templates = os.path.join(tools_web, "templates", "analytics.html")

        # Bundled _MEIPASS fallback
        bundled_web = os.path.join(meipass_path, "web")

        _debug_log(f"external_templates: {external_templates}", base_path)
        _debug_log(
            f"external_templates exists: {os.path.exists(external_templates)}",
            base_path,
        )
        _debug_log(f"tools_templates: {tools_templates}", base_path)
        _debug_log(
            f"tools_templates exists: {os.path.exists(tools_templates)}", base_path
        )
        _debug_log(f"bundled_web: {bundled_web}", base_path)
        _debug_log(f"bundled_web exists: {os.path.exists(bundled_web)}", base_path)

        # Choose path - prefer external over bundled
        if os.path.exists(external_templates):
            web_path = external_web
            _debug_log(f"SELECTED: EXTERNAL - {web_path}", base_path)
        elif os.path.exists(tools_templates):
            web_path = tools_web
            _debug_log(f"SELECTED: TOOLS - {web_path}", base_path)
        elif os.path.exists(bundled_web):
            web_path = bundled_web
            _debug_log(f"SELECTED: BUNDLED - {web_path}", base_path)
        else:
            web_path = external_web  # Will fail gracefully
            _debug_log(f"SELECTED: FALLBACK (no valid path) - {web_path}", base_path)
    else:
        # Running as normal Python script
        web_path = os.path.dirname(os.path.abspath(__file__))

    template_folder = os.path.join(web_path, "templates")
    static_folder = os.path.join(web_path, "static")

    app = Flask(
        __name__,
        template_folder=template_folder,
        static_folder=static_folder,
        static_url_path="/static",
    )
    app.config["SECRET_KEY"] = os.urandom(24)

    # Initialize SocketIO if available - skip in frozen builds to avoid async_mode issues
    if socketio is not None:
        try:
            socketio.init_app(
                app,
                async_mode="threading",
                cors_allowed_origins="*",  # Allow all origins for remote dashboard access
            )
        except ValueError:
            # async_mode not supported in frozen environment, disable SocketIO
            print("Warning: SocketIO disabled (async_mode not available)")
    else:
        print("Warning: SocketIO not available")

    # Register blueprints
    from .routes.api import api_bp
    app.register_blueprint(api_bp, url_prefix="/api")
    
    # ML-dependent routes are optional (may not be available in frozen builds)
    try:
        from .routes.video import video_bp
        app.register_blueprint(video_bp, url_prefix="/video")
    except ImportError as e:
        print(f"Warning: Video routes not available: {e}")
    
    try:
        from .routes.audio import audio_bp
        app.register_blueprint(audio_bp, url_prefix="/audio")
    except ImportError as e:
        print(f"Warning: Audio routes not available: {e}")
    
    try:
        from .routes.ouroboros_telemetry import ouroboros_telemetry_bp
        app.register_blueprint(ouroboros_telemetry_bp)  # Ouroboros has its own /api/telemetry prefix
    except ImportError as e:
        print(f"Warning: Ouroboros telemetry routes not available: {e}")

    # Auth routes for multi-user collaboration
    try:
        from .routes.auth import auth_bp
        app.register_blueprint(auth_bp, url_prefix="/auth")
    except ImportError as e:
        print(f"Warning: Auth routes not available: {e}")

    # Collaboration routes for shared sessions
    try:
        from .routes.collaboration import collab_bp
        app.register_blueprint(collab_bp, url_prefix="/collab")
    except ImportError as e:
        print(f"Warning: Collaboration routes not available: {e}")

    # SECURITY: Add security headers to all responses
    @app.after_request
    def add_security_headers(response):
        # Prevent clickjacking
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        # Prevent MIME sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"
        # Enable XSS filter
        response.headers["X-XSS-Protection"] = "1; mode=block"
        # Referrer policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        # Content Security Policy - allow self and inline for dashboard functionality
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.socket.io https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "img-src 'self' data: blob:; "
            "connect-src 'self' ws: wss:; "
            "font-src 'self' https://fonts.gstatic.com; "
            "frame-ancestors 'self'"
        )
        return response

    # Main routes
    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/projects")
    def projects():
        return render_template("projects.html")

    @app.route("/projects/<project_id>")
    def project_detail(project_id):
        return render_template("project_detail.html", project_id=project_id)

    @app.route("/notes")
    def notes():
        return render_template("notes.html")

    @app.route("/images")
    def images():
        return render_template("images.html")

    @app.route("/audio")
    def audio():
        return render_template("audio.html")

    @app.route("/video")
    def video():
        return render_template("video.html")

    @app.route("/automations")
    def automations():
        return render_template("automations.html")

    @app.route("/agents")
    def agents():
        return render_template("agents.html")

    @app.route("/analytics")
    def analytics():
        return render_template("analytics.html")

    @app.route("/teams")
    def teams():
        return render_template("teams.html")

    @app.route("/audits")
    def audits():
        return render_template("audits.html")

    @app.route("/reasoning/<audit_id>")
    def reasoning(audit_id):
        return render_template("reasoning.html", audit_id=audit_id)

    @app.route("/settings")
    def settings():
        return render_template("settings.html")

    return app
