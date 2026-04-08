"""Compatibility entrypoint that re-exports the FastAPI app."""

from server.app import app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=True)
