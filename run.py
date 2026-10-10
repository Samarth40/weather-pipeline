"""
Single-file runner for the MLOps Weather Forecast AI Pipeline.
Starts:
  1. MLflow Tracking Server (port 5001)
  2. Flask ML Model & Inference API (port 5000)
  3. Streamlit Weather Dashboard (port 8502)

Usage:
    python run.py
"""

import os
import sys
import time
import signal
import socket
import subprocess
import webbrowser
from pathlib import Path

# Safe terminal encoding
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_DIR = Path(__file__).resolve().parent
MLFLOW_PORT = 5001
FLASK_PORT = 5000
STREAMLIT_PORT = 8502


def log(msg: str):
    print(msg, flush=True)


def find_python_executable() -> str:
    """Finds the best Python interpreter with dependencies (checking conda envs and current python)."""
    candidates = [
        # Check py311 conda env
        r"E:\anaconda\envs\py311\python.exe",
        r"E:\anaconda\envs\rag_cost_control\python.exe",
        # Check active conda prefix if any
        os.path.join(os.environ.get("CONDA_PREFIX", ""), "python.exe") if os.environ.get("CONDA_PREFIX") else "",
        # Current running python
        sys.executable,
    ]

    for cand in candidates:
        if cand and os.path.exists(cand):
            try:
                res = subprocess.run(
                    [cand, "-c", "import flask, streamlit, mlflow; print('OK')"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if res.returncode == 0 and "OK" in res.stdout:
                    return cand
            except Exception:
                continue

    return sys.executable


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def wait_for_port(port: int, host: str = "127.0.0.1", timeout: int = 30) -> bool:
    start_time = time.time()
    while time.time() - start_time < timeout:
        if is_port_in_use(port, host):
            return True
        time.sleep(0.5)
    return False


def verify_local_data():
    """Ensure local weather datasets and model directories exist."""
    local_data_dir = PROJECT_DIR / "local_data"
    weather_dir = local_data_dir / "weather-data"
    if not weather_dir.exists() or not list(weather_dir.glob("*.csv")):
        log("[*] Local data not found. Initializing via setup_local_data.py...")
        try:
            subprocess.run([sys.executable, str(PROJECT_DIR / "setup_local_data.py")], check=True)
            log("[OK] Local weather data initialized.")
        except Exception as e:
            log(f"[!] Note: setup_local_data completed with: {e}")
    else:
        log("[OK] Local weather datasets and models verified in ./local_data")


def main():
    log("=" * 60)
    log("  [>] Starting MLOps Weather Forecast AI Pipeline")
    log("=" * 60)

    python_exe = find_python_executable()
    log(f"[*] Python Interpreter: {python_exe}")

    # Prepare environment variables
    env = os.environ.copy()
    env["MLFLOW_TRACKING_URI"] = f"http://127.0.0.1:{MLFLOW_PORT}"
    env["MLFLOW_ALLOW_FILE_STORE"] = "true"
    env["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    # Ensure local storage adapter is active (no azure connection string)
    if "AZURE_STORAGE_CONNECTION_STRING" not in env or not env["AZURE_STORAGE_CONNECTION_STRING"]:
        env["AZURE_STORAGE_CONNECTION_STRING"] = "local"

    verify_local_data()

    processes = []

    def shutdown(sig=None, frame=None):
        log("\n[*] Stopping all MLOps services...")
        for p in processes:
            try:
                p.terminate()
            except Exception:
                pass
        time.sleep(1)
        for p in processes:
            try:
                p.kill()
            except Exception:
                pass
        log("[*] All services stopped. Goodbye!")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # 1. Start MLflow Tracking Server (port 5001)
    log(f"[*] Launching MLflow Tracking Server on port {MLFLOW_PORT}...")
    mlflow_cmd = [
        python_exe,
        "-m",
        "mlflow",
        "server",
        "--host",
        "127.0.0.1",
        "--port",
        str(MLFLOW_PORT),
        "--workers",
        "1",
    ]
    mlflow_proc = subprocess.Popen(
        mlflow_cmd,
        cwd=str(PROJECT_DIR),
        env=env,
    )
    processes.append(mlflow_proc)

    # Wait for MLflow
    log(f"[*] Waiting for MLflow Server on port {MLFLOW_PORT}...")
    if not wait_for_port(MLFLOW_PORT, timeout=25):
        log("[!] Warning: MLflow port check timed out. Proceeding anyway...")
    else:
        log(f"[OK] MLflow Server is live at http://127.0.0.1:{MLFLOW_PORT}")

    # 2. Start Flask Model API (port 5000)
    log(f"[*] Launching ML Model Flask API on port {FLASK_PORT}...")
    flask_cmd = [
        python_exe,
        "app.py",
    ]
    flask_proc = subprocess.Popen(
        flask_cmd,
        cwd=str(PROJECT_DIR / "ml_model"),
        env=env,
    )
    processes.append(flask_proc)

    # Wait for Flask API
    log(f"[*] Waiting for ML Model API on port {FLASK_PORT}...")
    if not wait_for_port(FLASK_PORT, timeout=25):
        log("[!] Warning: ML Model API port check timed out. Proceeding anyway...")
    else:
        log(f"[OK] ML Model API is live at http://127.0.0.1:{FLASK_PORT}")

    # 3. Start Streamlit Weather Dashboard
    dashboard_port = STREAMLIT_PORT
    if is_port_in_use(dashboard_port):
        dashboard_port = 8503 if is_port_in_use(8501) else 8501

    log(f"[*] Launching Streamlit Weather Dashboard on port {dashboard_port}...")
    streamlit_cmd = [
        python_exe,
        "-m",
        "streamlit",
        "run",
        str(PROJECT_DIR / "weather_app" / "streamlit_app.py"),
        "--server.port",
        str(dashboard_port),
        "--server.headless",
        "false",
        "--browser.gatherUsageStats",
        "false",
    ]
    streamlit_proc = subprocess.Popen(
        streamlit_cmd,
        cwd=str(PROJECT_DIR),
        env=env,
    )
    processes.append(streamlit_proc)

    # Wait for Streamlit
    if wait_for_port(dashboard_port, timeout=25):
        log(f"[OK] Weather Dashboard is live at http://localhost:{dashboard_port}")

    log("\n" + "=" * 60)
    log("  [SUCCESS] MLOps Weather AI Pipeline is running!")
    log(f"  [+] Weather Dashboard:  http://localhost:{dashboard_port}")
    log(f"  [+] MLflow Tracking UI: http://127.0.0.1:{MLFLOW_PORT}")
    log(f"  [+] ML Model API:       http://127.0.0.1:{FLASK_PORT}")
    log("  Press Ctrl+C in this terminal to stop all services.")
    log("=" * 60 + "\n")

    # Automatically open browser to dashboard
    try:
        webbrowser.open(f"http://localhost:{dashboard_port}")
    except Exception:
        pass

    try:
        while True:
            for p in processes:
                if p.poll() is not None:
                    log(f"[!] A service terminated unexpectedly (PID {p.pid}, Exit {p.returncode})")
                    shutdown()
            time.sleep(1)
    except KeyboardInterrupt:
        shutdown()


if __name__ == "__main__":
    main()
