import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from agent import router as http_router # The batch API from before
from agent import router as agent_router
app = FastAPI(
    title="AI Agent (Batch + Streaming)",
    description="Hosts both a batch API and a real-time WebSocket agent.",
    version="2.0.0"
)

# Include the BATCH API routes (e.g., /api/process-audio/)
app.include_router(http_router, prefix="/api", tags=["Batch Agent"])

# Include the REAL-TIME WebSocket routes (e.g., /ws/agent)
app.include_router(agent_router, tags=["Streaming Agent"])


@app.get("/", response_class=HTMLResponse, tags=["Test Pages"])
async def get_batch_page():
    """Serves the test page for the batch upload API."""
    try:
        with open("index.html") as f:
            return f.read()
    except FileNotFoundError:
        return "Batch test page (index.html) not found."

@app.get("/stream", response_class=HTMLResponse, tags=["Test Pages"])
async def get_streaming_page():
    """Serves the test page for the real-time streaming agent."""
    try:
        with open("stream.html") as f:
            return f.read()
    except FileNotFoundError:
        return "Streaming test page (stream.html) not found."


if __name__ == "__main__":
    print("Starting FastAPI server...")
    print("Batch API test page at http://127.0.0.1:8000/")
    print("Streaming API test page at http://127.0.0.1:8000/stream")
    uvicorn.run(app, host="127.0.0.1", port=8000)