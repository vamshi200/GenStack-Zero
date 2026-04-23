from fastapi import FastAPI


# Create the FastAPI application instance.
app = FastAPI(title="GenStack-Zero API")


@app.get("/health")
def health() -> dict[str, str]:
    """Basic health check endpoint."""
    # This endpoint is useful for confirming the API is running.
    return {"status": "ok"}
