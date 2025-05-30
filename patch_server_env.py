#!/usr/bin/env python3
"""Patch server.py to read configuration from environment variables"""

import os
import shutil
from datetime import datetime

def create_patch():
    """Create the patch for environment variable support"""
    
    patch_code = '''
# Add this after line 216 where SERVER_CONFIG is created:

# Update SERVER_CONFIG from environment variables
if os.getenv("DIA_DEBUG", "").lower() in ("1", "true", "yes"):
    SERVER_CONFIG.debug_mode = True
if os.getenv("DIA_SAVE_OUTPUTS", "").lower() in ("1", "true", "yes"):
    SERVER_CONFIG.save_outputs = True
if os.getenv("DIA_SHOW_PROMPTS", "").lower() in ("1", "true", "yes"):
    SERVER_CONFIG.show_prompts = True
if os.getenv("DIA_RETENTION_HOURS"):
    try:
        SERVER_CONFIG.output_retention_hours = int(os.getenv("DIA_RETENTION_HOURS"))
    except ValueError:
        pass

# Log configuration source
if any([os.getenv(f"DIA_{k}") for k in ["DEBUG", "SAVE_OUTPUTS", "SHOW_PROMPTS", "RETENTION_HOURS"]]):
    console.print("[cyan]Configuration updated from environment variables[/cyan]")
'''

    print("📝 Patch to add to server.py after line 217 (after SERVER_CONFIG = ServerConfig()):")
    print("="*60)
    print(patch_code)
    print("="*60)
    
    print("\n💡 This patch allows the server to read configuration from environment variables:")
    print("   • DIA_DEBUG=1")
    print("   • DIA_SAVE_OUTPUTS=1")
    print("   • DIA_SHOW_PROMPTS=1")
    print("   • DIA_RETENTION_HOURS=24")
    
    print("\n🔧 To apply this patch:")
    print("1. Edit src/server.py")
    print("2. Find line 217: SERVER_CONFIG = ServerConfig()")
    print("3. Add the patch code right after that line")
    print("4. Save the file")
    
    # Create backup
    server_file = "src/server.py"
    if os.path.exists(server_file):
        backup_file = f"src/server.py.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(server_file, backup_file)
        print(f"\n💾 Backup created: {backup_file}")

if __name__ == "__main__":
    create_patch()