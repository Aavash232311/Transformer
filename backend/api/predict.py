import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.websockets import WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from src.interface_lm import * # Bad habit again but okay

app = FastAPI()


app.mount("/assets", StaticFiles(directory="client/dist/assets"), name="static")

''' Here, good habit is to never import torch and ML thing here,
we need to create an interface so that LLM talks to the backend.
If it was a microservice or anything we cannot just load torch in 
any other language. '''


@app.websocket("/ws/llm")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            # whatever everything is weird here no typecheck
            # This is a small model so I think I should write the entire prompt method here
            await websocket.send_json({
                "echo":data
            })

    except WebSocketDisconnect:
        print("Client disconnected or server reloaded.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


''' Point any request in this port to react '''
@app.get("/{rest_of_path:path}")
async def serve_react(rest_of_path: str):
    return FileResponse("client/dist/index.html")

# uvicorn backend.api.predict:app --reload