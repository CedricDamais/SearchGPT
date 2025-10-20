#!/usr/bin/env python3
"""
Launch script for SearchGPT Streamlit frontend.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the Streamlit app."""
    # Get the project root directory
    project_root = Path(__file__).parent
    app_path = project_root / "src" / "frontend" / "app.py"
    
    if not app_path.exists():
        print(f"Error: Streamlit app not found at {app_path}")
        sys.exit(1)
    
    print("🚀 Starting SearchGPT Frontend...")
    print(f"📍 App location: {app_path}")
    print("🌐 Frontend will be available at: http://localhost:8501")
    print("🔧 Make sure the API is running at: http://localhost:8000")
    print("-" * 60)
    
    # Launch Streamlit
    try:
        subprocess.run([
            "uv", "run", "streamlit", "run", 
            str(app_path),
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--theme.base", "light",
            "--theme.primaryColor", "#1f77b4",
        ], cwd=project_root)
    except KeyboardInterrupt:
        print("\n👋 Shutting down SearchGPT Frontend...")
    except Exception as e:
        print(f"❌ Error running Streamlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()