import torch
from fastapi import FastAPI
from fastapi.websockets import WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from src.interface_lm import InterFaceLM, tokenizer # Bad habit again but okay

app = FastAPI()


app.mount("/assets", StaticFiles(directory="client/dist/assets"), name="static")

'''
Okay so Let's define few hyperparamaters here,
Others are in interface_lm.py
'''

BLOCK_SIZE = 128

device = torch.device("cpu")
if torch.cuda.is_available():
    evice = torch.device("cuda") 

GENERATE_LENGTH = 1024 

llm_interface = InterFaceLM()

''' 
Now we are facing a system design problem what to send and on what interval.
'''

@app.websocket("/ws/llm")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            # whatever everything is weird here no typecheck
            # This is a small model so I think I should write the entire prompt method here
            prompt = data["prompt"]
            prompt = torch.tensor(tokenizer.encode("for"), dtype=torch.long, device=device).unsqueeze(0)

            current_input = prompt

            for _ in range(GENERATE_LENGTH):
                next_token = llm_interface.token_ping(current_input) 
                await websocket.send_json({
                    "token": tokenizer.decoder([next_token.item()])
                }) # this is not the effective way of send tokens? we will figure out what next;
                current_input = torch.cat([current_input, next_token.unsqueeze(0)], dim=1) 
                if current_input.shape[1] > BLOCK_SIZE:
                     # if exceeded then take the last of wharever the block_size is
                    current_input = current_input[:, -BLOCK_SIZE:]       


            '''
            In short, most of the software people do things like this these days,
            This is a small one but idea is the same thing. 
            ''' 

    except WebSocketDisconnect:
        print("Client disconnected or server reloaded.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


''' Point any request in this port to react '''
@app.get("/{rest_of_path:path}")
async def serve_react(rest_of_path: str):
    return FileResponse("client/dist/index.html")

# uvicorn backend.api.predict:app --reload