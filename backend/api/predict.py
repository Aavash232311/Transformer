from fastapi import FastAPI
from fastapi.websockets import WebSocket, WebSocketDisconnect

app = FastAPI()


@app.get("/")
async def read_main(): # basic get end point
    return {
        "msg": "OPEN TCP"
    }

@app.websocket("/ws/llm")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            # whatever everything is weird here no typecheck
            await websocket.send_json({
                "echo":data
            })

    except WebSocketDisconnect:
        print("Client disconnected or server reloaded.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")