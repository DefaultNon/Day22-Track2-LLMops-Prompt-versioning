import os
from dotenv import load_dotenv

def load_config():
    """Load environment variables and return config dict."""
    load_dotenv(override=True)
    
    config = {
        "langsmith_project": os.getenv("LANGCHAIN_PROJECT", "default"),
        "openai_endpoint": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "default_llm": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        "api_key": os.getenv("OPENAI_API_KEY"),
        "langsmith_key": os.getenv("LANGCHAIN_API_KEY")
    }
    
    return config

def main():
    config = load_config()
    print("[OK] Config loaded successfully")
    print(f"   LangSmith project : {config['langsmith_project']}")
    print(f"   OpenAI endpoint   : {config['openai_endpoint']}")
    print(f"   Default LLM model : {config['default_llm']}")
    print(f"   Embedding model   : {config['embedding_model']}")
    
    if not config['api_key']:
        print("[WARN] Warning: OPENAI_API_KEY not found in .env")
    if not config['langsmith_key']:
        print("[WARN] Warning: LANGCHAIN_API_KEY not found in .env")

if __name__ == "__main__":
    main()
