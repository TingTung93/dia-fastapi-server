#!/usr/bin/env python3
"""
Quick test to verify the server starts correctly
"""

import subprocess
import sys
import time
import requests

def test_startup():
    """Test that the server starts without requiring HF_TOKEN"""
    print("🧪 Testing server startup...")
    
    # Start server process
    process = subprocess.Popen(
        [sys.executable, "start_server.py", "--check-only"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    # Read output
    output, _ = process.communicate()
    print(output)
    
    # Check return code
    if process.returncode == 0:
        print("✅ Environment check passed!")
        return True
    else:
        print("❌ Environment check failed!")
        return False

if __name__ == "__main__":
    success = test_startup()
    sys.exit(0 if success else 1)