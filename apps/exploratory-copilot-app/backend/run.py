import os
from dotenv import load_dotenv


def str2bool(v: str) -> bool:
    return str(v).lower() in {"1", "true", "yes", "y", "on"}


def main():
    # Load .env in current directory if present
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=False)

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8008"))
    reload = str2bool(os.getenv("RELOAD", "false"))
    log_level = os.getenv("LOG_LEVEL", "info")
    workers = int(os.getenv("WORKERS", "1"))

    # Prefer python -m to avoid PATH issues
    import uvicorn
    uvicorn.run("main:app", host=host, port=port, reload=reload, log_level=log_level, workers=workers)


if __name__ == "__main__":
    main()


