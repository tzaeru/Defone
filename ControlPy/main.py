"""
ControlPy entry point. Starts the TCP server.

Usage:
    python main.py [port]            # run once
    python main.py --reload [port]   # auto-restart on file changes in ControlPy/
"""

import os
import sys
import subprocess
import time

from server import run_server, HOST, PORT


def _run_with_reload(port: int):
    """Spawn the server as a subprocess and restart it whenever .py files change."""
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler

    watch_dir = os.path.dirname(os.path.abspath(__file__))
    server_proc = None

    def start_server():
        nonlocal server_proc
        env = os.environ.copy()
        env["_CONTROLPY_CHILD"] = "1"
        server_proc = subprocess.Popen(
            [sys.executable, __file__, str(port)],
            env=env,
        )
        print(f"[reload] Server started (pid {server_proc.pid})")

    def restart_server():
        nonlocal server_proc
        if server_proc and server_proc.poll() is None:
            print("[reload] Stopping server...")
            server_proc.terminate()
            server_proc.wait(timeout=5)
        start_server()

    class ReloadHandler(FileSystemEventHandler):
        def __init__(self):
            self._last_reload = 0

        def on_modified(self, event):
            if event.is_directory or not event.src_path.endswith(".py"):
                return
            # Debounce: ignore events within 1s of last reload
            now = time.time()
            if now - self._last_reload < 1.0:
                return
            self._last_reload = now
            rel = os.path.relpath(event.src_path, watch_dir)
            print(f"[reload] {rel} changed, restarting...")
            restart_server()

    observer = Observer()
    observer.schedule(ReloadHandler(), watch_dir, recursive=True)
    observer.start()

    start_server()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[reload] Shutting down...")
        observer.stop()
        if server_proc and server_proc.poll() is None:
            server_proc.terminate()
            server_proc.wait(timeout=5)
    observer.join()


def main():
    args = [a for a in sys.argv[1:] if a != "--reload"]
    reload_mode = "--reload" in sys.argv[1:]

    port = PORT
    if args:
        try:
            port = int(args[0])
        except ValueError:
            print("Usage: python main.py [--reload] [port]", file=sys.stderr)
            sys.exit(1)

    if reload_mode and not os.environ.get("_CONTROLPY_CHILD"):
        _run_with_reload(port)
    else:
        run_server(host=HOST, port=port)


if __name__ == "__main__":
    main()
