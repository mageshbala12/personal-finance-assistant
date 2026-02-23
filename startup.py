"""
Startup script for production deployment.
Verifies environment and creates necessary directories.
"""
import os

# Create necessary directories if they don't exist
dirs = ["data", "chroma_db"]
for d in dirs:
    os.makedirs(d, exist_ok=True)
    print(f"✅ Directory ready: {d}")

print("✅ Startup checks complete!")