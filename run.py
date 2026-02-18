#!/usr/bin/env python3
"""
AgentNate Launcher

Starts the FastAPI backend server.
Open http://localhost:8000 in your browser, or use with PyWebView for desktop app.
"""

import os
import sys
import subprocess
import time
import webbrowser
import argparse

# Ensure we're using the right Python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# Portable Playwright browser path (keeps browsers inside the app directory)
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", os.path.join(BASE_DIR, "python", ".playwright-browsers"))


def run_server(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
    """Run the FastAPI server."""
    import uvicorn
    os.environ["AGENTNATE_BASE_URL"] = f"http://{host}:{port}"

    print(f"""
    ============================================================
                         AgentNate v2.0
               Multi-Provider LLM Orchestrator
    ============================================================
      Backend:  http://{host}:{port}
      API Docs: http://{host}:{port}/docs
    ============================================================
    """)

    uvicorn.run(
        "backend.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


def run_desktop(host: str = "127.0.0.1", port: int = 8000):
    """Run as desktop app with PyWebView."""
    try:
        import webview
    except ImportError:
        print("PyWebView not installed. Install with: pip install pywebview")
        print("Falling back to browser mode...")
        run_browser(host, port)
        return

    # Start server in subprocess
    server_proc = subprocess.Popen(
        [sys.executable, "-c", f"""
import sys
sys.path.insert(0, r'{BASE_DIR}')
from run import run_server
run_server('{host}', {port})
"""],
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
    )

    # Wait for server to start
    print("Starting backend server...")
    time.sleep(3)

    # Create window
    print("Opening AgentNate window...")
    window = webview.create_window(
        "AgentNate",
        f"http://{host}:{port}",
        width=1400,
        height=900,
        min_size=(800, 600),
    )

    webview.start()

    # Cleanup
    server_proc.terminate()
    print("AgentNate closed.")


def run_browser(host: str = "127.0.0.1", port: int = 8000):
    """Run server and open in browser."""
    import threading

    # Open browser after short delay
    def open_browser():
        time.sleep(2)
        webbrowser.open(f"http://{host}:{port}")

    threading.Thread(target=open_browser, daemon=True).start()

    run_server(host, port)


def main():
    parser = argparse.ArgumentParser(description="AgentNate Launcher")
    parser.add_argument(
        "--mode",
        choices=["server", "desktop", "browser"],
        default="browser",
        help="Run mode: server (API only), desktop (PyWebView), browser (open in browser)"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev mode)")

    args = parser.parse_args()

    if args.mode == "server":
        run_server(args.host, args.port, args.reload)
    elif args.mode == "desktop":
        run_desktop(args.host, args.port)
    else:
        run_browser(args.host, args.port)


if __name__ == "__main__":
    main()
