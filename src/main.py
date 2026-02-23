import os
import torch
import torch.nn as nn
from src.tokenizer import Tokenizer
from src.batch import BatchLoader
from torch.utils.data import Dataset, DataLoader
from src.test.embedding import SinusoidalPositionalEncoding

device = torch.device("cpu")
if torch.cuda.is_available():
    evice = torch.device("cuda") 

with open('data/View.txt', 'r', encoding='utf-8') as f:
    text = f.read()

tokenizer = Tokenizer(text)

# fast math for rtx 30 series
torch.set_float32_matmul_precision('high')

# Data Plumbing 
split_data_index = int(len(text) * 0.9) 
train_data = torch.tensor(tokenizer.encode(text[:split_data_index]), dtype=torch.long).to(device)   
val_data = torch.tensor(tokenizer.encode(text[split_data_index:]), dtype=torch.long).to(device)    

class MLP(nn.Module):
    def __init__(self, d_model, n_neurons=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, n_neurons * d_model),
            nn.GELU(),
            nn.Linear(n_neurons * d_model, d_model)
        )

    def forward(self, x): # x = output from attention block
        return self.net(x)


class MultiHead(nn.Module):

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.key = nn.Linear(dim, dim) # it's learnable so using nn instead of matrix
        self.query = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.num_heads = num_heads

        self.head_dim = dim // num_heads # each head will have like dim // num_heads dimension

        self.output_proj = nn.Linear(dim, dim)
    
    ''' Attention(Q, K, V) = softmax(QK^T / √d_k) V '''
    def forward(self, x):
        B,T,C = x.shape
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        '''
        Assuming input dimensions is 512
        Let's say we want to split those into 8 heads then each head will have 64 dimensions.
        It will procress all of those in parallel, each head output will be in 64 dimensions.
        And the final output will be of 512 same dimension.
        '''

        # let's group these 512 numbers as 8 groups of 64.
        query = query.view(B, T, self.num_heads, self.head_dim)
        key = key.view(B, T, self.num_heads, self.head_dim)
        value = value.view(B, T, self.num_heads, self.head_dim)

        # we might want to re-order dimension so that head can procress in parallel
        # (Batch, Time, Heads, Features) -> (Batch, Heads, Time, Features) which let's torch procress all heads at once with matrix math
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # now we can do the attention computation
        # for a valid matrix mul (..., T, head_dim) @ (..., head_dim, T) = (..., T, T) so we need to swap the last two dims
        s = torch.matmul(query, key.transpose(-2, -1))
        s = s / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.long))

        # we need masking to make the model learn authentically
        mask = torch.tril(torch.ones(T, T)).view(1, 1, T, T).to(x.device)
        s = s.masked_fill(mask==0, -1e9) # set it to -Inf

        # now we can apply softmax
        attn_wts = torch.softmax(s, dim=-1) # make sure everythings sums up to a disto of 1
        out = torch.matmul(attn_wts, value)
        # shape here is ex [1, 8, 5, 64]
        # combine the heads back, contiguous() ensures data is stored sequently for the next view()
        out = out.transpose(1, 2).contiguous() # new shape ]1, 5, 8, 64]
        out = out.view(B, T, C) # here we reform our shape
        out = self.output_proj(out) # MultiHead(Q, K, V ) = Concat(head1, ..., headh)WO
        return out


class TransformerBlock(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attention = MultiHead(d_model, num_heads)
        self.mlp = MLP(d_model)

        self.norm1 = nn.LayerNorm(d_model) # normalzies each vector to have mean 0 and std dev 1, we might see problems like vanishing gradient
        self.norm2 = nn.LayerNorm(d_model)

    
    def forward(self, x):
        ''' I tried post norm, training look unstable let's try post norm '''
        temp = x # we need to store the original
        att_out = self.attention(self.norm1(x))
        x = temp + att_out

        # again storing second temp
        temp2 = x
        mlp_out = self.mlp(self.norm2(x))
        x = temp2 + mlp_out
        return x

class StackTransfomer(nn.Module):

    def __init__(self,  d_model, num_heads, num_layers=4):
        super().__init__()

        # now we need to stack the blocks
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads)
            for _ in range(num_layers)  # Creates N identical blocks
        ])

        self.final_norm = nn.LayerNorm(d_model) # normalizes output, the result that comes from being passed with n-blocks 

    def forward(self, x):
        for layer in self.layers: # passing through n-blocks
            x = layer(x)
        
        # final norm
        x = self.final_norm(x)
        return x
        
class Main(nn.Module):

    def __init__(self, batch_size, block_size, device, d_model, vocab_size, generate_length=1500, p_type="learned"):
        super().__init__()
        self.unique_characters = vocab_size
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
        self.d_model = d_model
        self.generate_length = generate_length

        self.token_embedding_table = nn.Embedding(self.unique_characters, d_model, device=self.device)

        if p_type == "learned":
            self.positional_embedding_table = nn.Embedding(self.block_size, d_model, device=self.device)
        else:
            self.positional_embedding_table = SinusoidalPositionalEncoding(
                d_model=d_model,
                max_len=block_size
            )

        self.block_transformer = StackTransfomer(
            d_model = d_model,
            num_heads=8
        ).to(device=device) 

        # we transfer to lm_head to turn the numbers into vocab predection!
        self.lm_head = nn.Linear(d_model, vocab_size) 



    def embedding(self, x):
        B, T = x.shape
        token_embedding_table = self.token_embedding_table(x) 

        pos = torch.arange(T, device=self.device)
  
        positional_embedding_table = self.positional_embedding_table(pos).unsqueeze(0)

        # adding those up like in the original paper
        res = token_embedding_table + positional_embedding_table
        return res
    
    def training_custom(self, train_data, num_epochs=12):
        '''
        optimizer slightly pulls every weight toward zero with weight_decay
        cause our model just memorized in one point leading to high accuracy.
        '''
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)

        dataset = BatchLoader(train_data, block_size=self.block_size)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,           
            drop_last=True,
        )

        ''' In my case CPU might be a bottleneck,
        NVIDA gpu handels math very fast, but it seems like the Sync is not so perfect.

        If I try to increase number of heads then it will be out of memory due to lack of GDDR6 RAM

        '''

        print(f"Training for {num_epochs} epochs")
        print(f"Batches per epoch: {len(dataloader)}")
        print("-" * 60)

        self.train()

        for epoch in range(num_epochs):
            epoch_loss = 0
            
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # accounting for all batches
            for batch_idx, (x, y) in enumerate(dataloader):
                x, y = x.to(self.device), y.to(self.device) # bring to NVIDA gpu is avalible
                
                embeddings = self.embedding(x)

                transformed = self.block_transformer(embeddings)
                logits = self.lm_head(transformed)
                
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, self.unique_characters), 
                    y.view(-1)
                )
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                if batch_idx % 20 == 0:
                    progress = (batch_idx + 1) / len(dataloader) * 100
                    print(f"  Batch {batch_idx+1}/{len(dataloader)} ({progress:.1f}%): loss = {loss.item():.4f}")
            
            avg_loss = epoch_loss / len(dataloader)
            print(f"epoch {epoch+1} complete! avg loss: {avg_loss:.4f}")
        
        print("training complete!")

    ''' Now we want to check how well hos our model done '''
    def evaulate(self, test_data):
        self.eval()
        test_data = test_data.to(self.device)

        total_loss = 0.0
        total_samples = 0

        # we need them ofc we do
        all_predictions = []
        all_targets = []

        # we need x, and y for cross entropy loss
        dataset = BatchLoader(test_data, block_size=self.block_size)
        test_loader = DataLoader(dataset, batch_size=32, shuffle=False)
        print("Evualating")
        print(f"Total characters: {len(test_data)}")

        with torch.no_grad(): # no gradient calculation in testing 
            for batch_idx, (x, y) in enumerate(test_loader):
                x, y = x.to(self.device), y.to(self.device)

                x = self.embedding(x)
                x = self.block_transformer(x) # this is the forward pass, passes through all the blocks
                
                logits = self.lm_head(x) 
                
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, self.unique_characters), 
                    y.view(-1)
                )

                batch_size = x.size(0)
                total_loss += loss.item() * batch_size # last batch might be smaller than the others
                total_samples += batch_size
                
                # get predictions
                predictions = torch.argmax(logits, dim=-1)

                all_predictions.append(predictions.cpu())
                all_targets.append(y.cpu())


        average_loss = total_loss / total_samples if total_samples > 0 else 0

        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # for accuracy 
        correct_mask = (all_predictions == all_targets) 
        accuracy = correct_mask.float().mean().item() * 100

        return average_loss, all_predictions, all_targets, accuracy
            
    def run(self, train_data, val_data):
        self.training_custom(train_data=train_data)
        average_loss, all_predictions, all_targets, accuracy = self.evaulate(val_data)
        print(f"avg loss: {average_loss} accuracy: {accuracy} ")
        return 

    ''' Here we want to expriement with simple prompt '''
    def prompt(self, prompt):

        prompt = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)


        generated_indices = []
        current_input = prompt

        for i in range(0, self.generate_length):
            embedding = self.embedding(current_input)
            out =  self.block_transformer.forward(embedding)
            logits = self.lm_head(out)

            next_token_logits = logits[:, -1, :] # ignore everything except last array, case we want to predit the next one

            next_token = torch.argmax(next_token_logits, dim=-1)

            generated_indices.append(next_token.item())

            current_input = torch.cat([current_input, next_token.unsqueeze(0)], dim=1) 

            if current_input.shape[1] > self.block_size:
                # making sure we don't exceed the block size
                current_input = current_input[:, -self.block_size:]
        all_indices = torch.cat([
            prompt[0], 
            torch.tensor(generated_indices, device=device)
        ]).tolist()
        return all_indices
    
if __name__ == "__name__":
    checkpoint_dir = 'checkpoint'
    file_name = 'transfomer_v1.pth'
    full_path = os.path.join(checkpoint_dir, file_name)


    prompt_lm = True


    transfomer = Main(batch_size=120, 
            block_size=128,
            device=device,
            d_model=256,
            vocab_size=len(tokenizer.unique_characters())).to(device=device)

    if prompt_lm == True:
        "Let's keep this like a way to poke this LM,"
        "Something what is more cool is to deply this"
        "even though this is like dumb."
        if os.path.exists(full_path):
            state_dict = torch.load(full_path, map_location=device)
            transfomer.load_state_dict(state_dict)
            transfomer.eval()
            prompt = str(input("Enter what you have to say: "))
            out = transfomer.prompt(prompt=prompt)
            print(tokenizer.decoder(out))
    else:
        "Let's keep this like a train mode!"
        transfomer.run(train_data=train_data, val_data=val_data)
        output = transfomer.prompt("for (let i in rootNode) {")
        print(tokenizer.decoder(output))

        total_params = sum(p.numel() for p in transfomer.parameters())
        print(f"Total paramaters: {total_params:,}")

        '''
            Let's save our trained params,
            if we have saved let's try using this model to prompt something!
        '''


        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            print(f"Created folder: {checkpoint_dir}")


        torch.save(transfomer.state_dict(), full_path)
        print(f"Saved successfully to {full_path}")