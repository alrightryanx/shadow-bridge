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
            base_path or os.path.dirname(os.path.abspath(__file__)), "web_debug.log"
        )
        with open(log_path, "a") as f:
            f.write(f"{msg}\n")
    except (IOError, OSError):
        pass  # Logging is best-effort


def create_app():
    """Create and configure the Flask application."""
    global socketio

    # Use script's parent directory as base path (web is in web/ subdirectory)
    base_path = os.path.dirname(os.path.abspath(__file__))
    # If we're running directly from web/ directory, go up one level
    if os.path.basename(base_path) == "web":
        base_path = os.path.dirname(base_path)

    _debug_log(f"=== Web Dashboard Init ===", base_path)
    _debug_log(f"sys.executable: {sys.executable}", base_path)
    _debug_log(f"base_path: {base_path}", base_path)
    _debug_log(f"Working directory: {os.getcwd()}", base_path)

    # Start SQLite agent sync thread
    try:
        from .services.sqlite_sync import start_sync_thread
        start_sync_thread()
        _debug_log("SQLite sync thread started", base_path)
    except Exception as e:
        _debug_log(f"Failed to start SQLite sync thread: {e}", base_path)

    if getattr(sys, "frozen", False):
        # Running from frozen bundle (MSI or EXE)
        meipass_path = getattr(sys, "_MEIPASS", base_path)

        # Priority 1: Relative to EXE (user overrides)
        external_web = os.path.join(base_path, "web")
        external_templates = os.path.join(external_web, "templates")

        # Priority 2: Next to EXE (if flattened)
        tools_web = base_path
        tools_templates = os.path.join(tools_web, "templates")

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
            allowed_origins = os.environ.get("CORS_ORIGINS", "http://localhost:6767,http://127.0.0.1:6767").split(",")
            socketio.init_app(
                app,
                async_mode="threading",
                cors_allowed_origins=allowed_origins,
            )
        except ValueError:
            # async_mode not supported in frozen environment, disable SocketIO
            print("Warning: SocketIO disabled (async_mode not available)")
    else:
        print("Warning: SocketIO not available")

    # Register blueprints
    import traceback

    try:
        from .routes.api import api_bp
        _debug_log("Registering API blueprint...", base_path)
        app.register_blueprint(api_bp, url_prefix="/api")
        _debug_log("API blueprint registered.", base_path)
    except Exception as e:
        print(f"Error registering API routes: {e}")
        _debug_log(f"API REG ERROR: {e}", base_path)
        traceback.print_exc()

    # ML-dependent routes are optional (may not be available in frozen builds)
    try:
        from .routes.video import video_bp
        _debug_log("Registering Video blueprint...", base_path)
        app.register_blueprint(video_bp, url_prefix="/video")
    except ImportError as e:
        _debug_log(f"Video routes skip (ImportError): {e}", base_path)
        print(f"Warning: Video routes not available: {e}")
    except Exception as e:
        _debug_log(f"Video routes skip (Error): {e}", base_path)
        print(f"Error registering Video routes: {e}")
        traceback.print_exc()

    try:
        from .routes.audio import audio_bp
        _debug_log("Registering Audio blueprint...", base_path)
        app.register_blueprint(audio_bp, url_prefix="/audio")
    except ImportError as e:
        _debug_log(f"Audio routes skip (ImportError): {e}", base_path)
        print(f"Warning: Audio routes not available: {e}")
    except Exception as e:
        _debug_log(f"Audio routes skip (Error): {e}", base_path)
        print(f"Error registering Audio routes: {e}")
        traceback.print_exc()

    try:
        from .routes.ouroboros_telemetry import ouroboros_telemetry_bp
        _debug_log("Registering Ouroboros blueprint...", base_path)
        app.register_blueprint(
            ouroboros_telemetry_bp
        )  # Ouroboros has its own /api/telemetry prefix
    except ImportError as e:
        _debug_log(f"Ouroboros routes skip (ImportError): {e}", base_path)
        print(f"Warning: Ouroboros telemetry routes not available: {e}")
    except Exception as e:
        _debug_log(f"Ouroboros routes skip (Error): {e}", base_path)
        print(f"Error registering Ouroboros routes: {e}")
        traceback.print_exc()

    # Auth routes for multi-user collaboration
    try:
        from .routes.auth import auth_bp
        _debug_log("Registering Auth blueprint...", base_path)
        if "auth_web" not in app.blueprints:
            app.register_blueprint(auth_bp, url_prefix="/auth")
            _debug_log("Auth blueprint registered.", base_path)
        else:
            _debug_log("Auth blueprint already registered, skipping.", base_path)
    except ImportError as e:
        _debug_log(f"Auth routes skip (ImportError): {e}", base_path)
        print(f"Warning: Auth routes not available: {e}")
    except Exception as e:
        if "already registered" not in str(e):
            _debug_log(f"Auth routes error: {e}", base_path)
            print("!!! CRITICAL ERROR REGISTERING AUTH BLUEPRINT !!!")
            print(f"Error: {e}")
            traceback.print_exc()
        else:
            _debug_log("Auth already registered (swallowed exception).", base_path)

    # Collaboration routes for shared sessions
    try:
        from .routes.collaboration import collab_bp

        app.register_blueprint(collab_bp, url_prefix="/collab")
    except ImportError as e:
        print(f"Warning: Collaboration routes not available: {e}")
    except Exception as e:
        print(f"Error registering Collaboration routes: {e}")
        traceback.print_exc()

    # Image generation routes
    try:
        from .routes.image import image_bp

        app.register_blueprint(image_bp, url_prefix="/api")
    except ImportError as e:
        print(f"Warning: Image generation routes not available: {e}")
    except Exception as e:
        print(f"Error registering Image routes: {e}")
        traceback.print_exc()

    # Music Studio routes for AI music generation and voice cloning
    try:
        from .routes.music import music_bp

        app.register_blueprint(music_bp, url_prefix="/music")
    except ImportError as e:
        print(f"Warning: Music routes not available: {e}")
    except Exception as e:
        print(f"Error registering Music routes: {e}")
        traceback.print_exc()

    # Ralph Swarm Integration - for launching and monitoring Ralph AI agents
    try:
        from .routes.ralph import ralph_bp

        app.register_blueprint(ralph_bp, url_prefix="/api")
    except ImportError as e:
        print(f"Warning: Ralph Swarm routes not available: {e}")
    except Exception as e:
        print(f"Error registering Ralph Swarm routes: {e}")
        traceback.print_exc()

    # Health Dashboard (Ouroboros V2)
    try:
        from .routes.health_routes import health_bp

        app.register_blueprint(health_bp)
    except ImportError as e:
        print(f"Warning: Health dashboard routes not available: {e}")
    except Exception as e:
        print(f"Error registering Health dashboard routes: {e}")
        traceback.print_exc()

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
            "script-src 'self' 'unsafe-inline' https://cdn.socket.io https://cdn.jsdelivr.net https://cdnjs.cloudflare.com; "
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

    @app.route("/audits")
    def audits():
        return render_template("audits.html")

    @app.route("/reasoning/<audit_id>")
    def reasoning(audit_id):
        return render_template("reasoning.html", audit_id=audit_id)

    @app.route("/settings")
    def settings():
        return render_template("settings.html")

    @app.context_processor
    def inject_version():
        return dict(version="1.0.48")

    # Auto-start autonomous loop if configured
    # Set AUTONOMOUS_AUTOSTART=1 in environment or backend/.env to enable
    try:
        from .services.autonomous_loop import get_autonomous_loop, get_config_value
        autostart = (get_config_value("AUTONOMOUS_AUTOSTART", "0") or "").lower() in ("1", "true", "yes", "on")
        if autostart:
            autostart_count = int(get_config_value("AUTONOMOUS_AGENT_COUNT", "5") or 5)
            autostart_focus = get_config_value("AUTONOMOUS_FOCUS", "backend-polish") or "backend-polish"
            autostart_provider = get_config_value("AUTONOMOUS_PROVIDER", "gemini") or "gemini"
            autostart_model = get_config_value("AUTONOMOUS_MODEL", "gemini-3-flash-preview") or "gemini-3-flash-preview"

            loop = get_autonomous_loop()
            if not loop.running:
                import threading
                def _delayed_start():
                    import time
                    time.sleep(3)  # Wait for server to be fully ready
                    try:
                        loop.start(
                            agent_count=autostart_count,
                            focus=autostart_focus,
                            provider=autostart_provider,
                            model=autostart_model,
                        )
                        _debug_log(f"Autonomous loop auto-started: {autostart_count} agents, focus={autostart_focus}", base_path)
                    except Exception as e:
                        _debug_log(f"Failed to auto-start autonomous loop: {e}", base_path)

                threading.Thread(target=_delayed_start, daemon=True).start()
                _debug_log("Autonomous loop auto-start scheduled", base_path)
        else:
            _debug_log("Autonomous loop auto-start disabled (set AUTONOMOUS_AUTOSTART=1 to enable)", base_path)
    except Exception as e:
        _debug_log(f"Failed to check autonomous auto-start: {e}", base_path)

    return app
