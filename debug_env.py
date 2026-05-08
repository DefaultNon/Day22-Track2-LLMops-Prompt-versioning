import os
from pathlib import Path
from dotenv import load_dotenv

def debug():
    print("--- Environment Debug V2 ---")
    env_path = Path(".env").absolute()
    print(f"Looking for .env at: {env_path}")
    print(f"File exists: {env_path.exists()}")
    
    if env_path.exists():
        content = env_path.read_text()
        print(f"File size: {len(content)} bytes")
        # Check first line
        first_line = content.splitlines()[0] if content.splitlines() else "EMPTY"
        print(f"First line: {first_line}")
    
    # Force load from specific path
    load_dotenv(dotenv_path=env_path, override=True)
    
    key = os.getenv("OPENAI_API_KEY")
    if key:
        masked = key[:10] + "..." + key[-4:] if len(key) > 15 else key
        print(f"OPENAI_API_KEY: {masked}")
    else:
        print("OPENAI_API_KEY NOT FOUND")

if __name__ == "__main__":
    debug()
