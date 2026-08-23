import os
from flask import Flask, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO

socketio = SocketIO(cors_allowed_origins="*", async_mode="threading")

FRONTEND_DIST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend_dist")


def create_app():
    # static_folder=None disables Flask's own implicit static-file route.
    # That route was silently shadowing the custom catch-all below whenever
    # static_url_path was set to "" - confirmed by testing directly: a
    # request to an unknown path returned Flask's raw 404 page instead of
    # ever reaching serve_frontend() below, which would have correctly
    # fallen back to index.html for client-side (SPA) routing.
    app = Flask(__name__, static_folder=None)
    CORS(app)
    app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024  # 1MB - this API only ever takes small JSON bodies

    from app.routes import bp
    app.register_blueprint(bp)

    socketio.init_app(app)
    from app import sockets  # noqa: F401 - registers WebSocket event handlers

    # Serve the built React app for everything that isn't an API route -
    # standard single-page-app pattern: a real file (JS/CSS/images) is
    # served directly; anything else (client-side routes) falls back to
    # index.html so React Router (or similar) can handle it.
    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def serve_frontend(path):
        if path and os.path.exists(os.path.join(FRONTEND_DIST, path)):
            return send_from_directory(FRONTEND_DIST, path)
        index_path = os.path.join(FRONTEND_DIST, "index.html")
        if os.path.exists(index_path):
            return send_from_directory(FRONTEND_DIST, "index.html")
        return (
            "Frontend not built yet. Run `npm run build` in frontend/ and copy "
            "the dist/ output to backend/frontend_dist/, or run the frontend "
            "dev server separately on its own port.",
            200,
        )

    return app
