import torch
import torch.nn as nn
from tokenizer import Tokenizer
from batch import BatchLoader
from torch.utils.data import Dataset, DataLoader

device = torch.device("cpu")
if torch.cuda.is_available():
    print(f"{torch.cuda.get_device_name(0)}")
    device = torch.device("cuda") 

with open('data/alice.txt', 'r', encoding='utf-8') as f:
    text = f.read()

tokenizer = Tokenizer(text)

print(f"Dataset characters: {len(text)}")

# Data Plumbing 
split_data_index = int(len(text) * 0.9) 
train_data = torch.tensor(tokenizer.encode(text[:split_data_index]), dtype=torch.long)      
val_data = torch.tensor(tokenizer.encode(text[split_data_index:]), dtype=torch.long)    

class MLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.p = 4
        self.net = nn.Sequential(
            nn.Linear(d_model, self.p * d_model),
            nn.GELU(),
            nn.Linear(self.p*d_model, d_model)
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
        temp = x # we need to store the original
        att_out = self.attention(x)
        x = temp + att_out
        x = self.norm1(x) # post norm

        # again storing second temp
        temp2 = x
        mlp_out = self.mlp(x)
        x = temp2 + mlp_out
        x = self.norm2(x)
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

    def __init__(self, batch_size, block_size, device, d_model, vocab_size, generate_length=8):
        super().__init__()
        self.unique_characters = vocab_size
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
        self.d_model = d_model
        self.generate_length = 8


        self.positional_embedding_table = nn.Embedding(self.block_size, d_model, device=self.device)
        self.token_embedding_table = nn.Embedding(self.unique_characters, d_model, device=self.device)

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
    
    def training_custom(self, train_data, num_epochs=5):
        '''
        optimizer slightly pulls every weight toward zero with weight_decay
        cause our model just memorized in one point leading to high accuracy.
        '''
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=0.1)

        dataset = BatchLoader(train_data, block_size=self.block_size)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,           
            num_workers=2, 
            drop_last=True,
            pin_memory=True # speeds the trasnfer of data from DDR RAM to NVIDA gpu
        )

        ''' In my case CPU might be a bottleneck,
        NVIDA gpu handels math very fast, but it seems like the Sync is not so perfect.

        If I try to increase number of heads then it will be out of memory due to lack of GDDR6 RAM

        '''

        print(f"Training for {num_epochs} epochs")
        print(f"Batches per epoch: {len(dataloader)}")
        print("-" * 60)

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
        correct_mask = (all_predictions == all_predictions)
        accuracy = correct_mask.float().mean().item() * 100

        return average_loss, all_predictions, all_targets, accuracy
            
    def run(self, train_data, val_data):
        # prompt = torch.tensor(tokenizer.encode("Alice was beginning"), dtype=torch.long, device=device).unsqueeze(0)
        # embedding = self.embedding(prompt)

        # out =  self.block_transformer.forward(embedding)
        # print(prompt.shape)


        self.training_custom(train_data=train_data)
        average_loss, all_predictions, all_targets, accuracy = self.evaulate(val_data)
        print(f"avg loss: {average_loss} accuracy: {accuracy} ")
        return 

    ''' Here we want to expriement with simple prompt '''
    def prompt(self, prompt):
        # this is the initial prompt!
        prompt = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)

        for i in range(0, self.generate_length):
            embedding = self.embedding(prompt)
            out =  self.block_transformer.forward(embedding)
            logits = self.lm_head(out)

            indices = torch.argmax(logits, dim=-1)
            decoded_indices = indices[0].tolist()

            if i == 0: # replce that in first iteration
                prompt = decoded_indices
            else:
                prompt += decoded_indices # we add the result

        return prompt

transfomer = Main(batch_size=120, 
        block_size=128,
        device=device,
        d_model=256,
        vocab_size=len(tokenizer.unique_characters())).to(device=device)
transfomer.run(train_data=train_data, val_data=val_data)
output = transfomer.prompt("Alice")
print(tokenizer.decoder(output))

total_params = sum(p.numel() for p in transfomer.parameters())
print(f"Total paramaters: {total_params:,}")

