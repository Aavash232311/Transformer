import os
import torch
from src.main import Main, tokenizer

'''
Load the skeleton for the LM, that main was just for testing.
'''

device = torch.device("cpu")
if torch.cuda.is_available():
    evice = torch.device("cuda") 


checkpoint_dir = 'checkpoint'
file_name = 'transfomer_v1.pth'
full_path = os.path.join(checkpoint_dir, file_name)

transfomer = Main(batch_size=120, 
        block_size=128,
        device=device,
        d_model=256,
        vocab_size=len(tokenizer.unique_characters())).to(device=device)

if os.path.exists(full_path):
    state_dict = torch.load(full_path, map_location=device)
    transfomer.load_state_dict(state_dict)
    transfomer.eval()


class InterFaceLM:

    def token_ping(self, prompt):
        prompt = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)

        embedding = transfomer.embedding(prompt) # There we go passed the encoded tensor!
        forward_pass = transfomer.block_transformer.forward(embedding)
        logits = transfomer.lm_head(forward_pass)

        next_token_logits = logits[:, -1 , :] # take only the last array
        next_token = torch.argmax(next_token_logits, dim=-1)

        return tokenizer.decoder([next_token.item()])




if os.path.exists(full_path):
    state_dict = torch.load(full_path, map_location=device)
    transfomer.load_state_dict(state_dict)
    transfomer.eval()



if __name__ == "__main__":
    lm_interface = InterFaceLM()
    print(lm_interface.token_ping("retur"))