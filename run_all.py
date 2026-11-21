"""
🌾 CropAdvisor — Run All Servers Concurrently
===============================================
Launches both the FastAPI Web App (port 8000) and the Streamlit Portal (port 8501).
Handles graceful shutdown on Ctrl+C.
"""
import sys
import os
import subprocess
import time
import webbrowser

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    python_exe = sys.executable

    print("\n" + "=" * 65)
    print("🌾 Crop Recommendation Decision Support System — VS Code Runner")
    print("=" * 65)
    print("🚀 1. FastAPI WebApp & API   : http://localhost:8000")
    print("🌾 2. Streamlit Portal        : http://localhost:8501")
    print("📑 3. Interactive API Docs    : http://localhost:8000/docs")
    print("=" * 65)
    print("💡 Press Ctrl+C anytime to stop all servers gracefully.\n")

    # Command for FastAPI
    fastapi_cmd = [
        python_exe, "-m", "uvicorn",
        "webapp.backend.main:app",
        "--reload",
        "--port", "8000"
    ]

    # Command for Streamlit
    streamlit_cmd = [
        python_exe, "-m", "streamlit",
        "run", os.path.join(root_dir, "app.py"),
        "--server.port", "8501",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false"
    ]

    procs = []
    try:
        # Start FastAPI
        print("▶️  Starting FastAPI server on http://localhost:8000...")
        p1 = subprocess.Popen(fastapi_cmd, cwd=root_dir)
        procs.append(p1)

        # Start Streamlit
        print("▶️  Starting Streamlit server on http://localhost:8501...")
        p2 = subprocess.Popen(streamlit_cmd, cwd=root_dir)
        procs.append(p2)

        time.sleep(2)
        print("\n✅ Both servers are online!")
        print("   - Open http://localhost:8000 in your browser for the Web App")
        print("   - Open http://localhost:8501 in your browser for Streamlit\n")

        # Monitor processes
        while True:
            for p in procs:
                if p.poll() is not None:
                    print(f"\n⚠️ Process {p.args} exited with code {p.returncode}")
                    return
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n🛑 Shutting down servers gracefully...")
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    p.kill()
        print("👋 All servers stopped.")

if __name__ == "__main__":
    main()
