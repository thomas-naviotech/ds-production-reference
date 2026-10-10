"""scripts/serve.py — run the FastAPI service locally."""

import argparse

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the Titanic API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    # Points at the import-time `app` in titanic_prod.api (trained or fallback).
    uvicorn.run("titanic_prod.api:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
