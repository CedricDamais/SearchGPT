#!/usr/bin/env python3
"""
Launch script for SearchGPT FastAPI backend.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Launch the FastAPI backend."""
    project_root = Path(__file__).parent
    
    print("Starting SearchGPT API Backend...")
    print("API will be available at: http://localhost:8000")
    print("API docs will be available at: http://localhost:8000/docs")
    
    # Launch FastAPI with uvicorn
    try:
        subprocess.run([
            "uv", "run", "uvicorn", 
            "src.api.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload",
        ], cwd=project_root)
    except KeyboardInterrupt:
        print("\nShutting down SearchGPT API...")
    except Exception as e:
        print(f"Error running API: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()