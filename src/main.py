
import torch
import torch.nn as nn
from tokenizer import Tokenizer
from batch import GetBatch

device = torch.device("cpu")
if torch.cuda.is_available():
    print(f"{torch.cuda.get_device_name(0)}")
    device = torch.device("cuda") 

with open('data/alice.txt', 'r', encoding='utf-8') as f:
    text = f.read()

tokenizer = Tokenizer(text)

print(f"Dataset characters: {len(text)}")

# Data Plumbing 
split_data_index = int(len(text) * 0.1)
train_data = torch.tensor(tokenizer.encode(text[:split_data_index]), dtype=torch.long)
test_data = torch.tensor(tokenizer.encode(text[split_data_index:]), dtype=torch.long)

class MLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.p = 4
        self.net = nn.Sequential(
            nn.Linear(d_model, self.p * d_model),
            nn.ReLU(),
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
        s = s / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

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


        
class TransfomerBlock:

    def __init__(self, batch_size, block_size, device, d_model):
        self.batch = GetBatch(train_data, batch_size, block_size)
        self.unique_characters = tokenizer.unique_characters()
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
        self.d_model = d_model


        self.positional_embedding_table = nn.Embedding(self.block_size, d_model, device=self.device)
        self.token_embedding_table = nn.Embedding(len(self.unique_characters), d_model, device=self.device)

    def mini_batch(self):
        x, y = self.batch.mini_batch()
        return x, y

    def forward(self, x):
        B, T = x.shape
        token_embedding_table = self.token_embedding_table(x) 

        pos = torch.arange(T, device=self.device)
  
        positional_embedding_table = self.positional_embedding_table(pos).unsqueeze(0)

        # adding those up like in the original paper
        res = token_embedding_table + positional_embedding_table
        return res
    

    def single_head(self):
        prompt = torch.tensor(tokenizer.encode("Alice"), dtype=torch.long, device=device).unsqueeze(0)
        embedding = transfomer.forward(prompt)

        # now we want to work in a single head of attention 
        single_headed = MultiHead(self.d_model).to(self.device) # bringing to NVIDA gpu if avalible 
        out = single_headed.forward(embedding)

        mlp = MLP(self.d_model).to(device=self.device)
        out_mlp = mlp.forward(out)
        print(out_mlp.shape)


transfomer = TransfomerBlock(batch_size=32, # for local hardware with 4GB GDDR6
        block_size=128,
        device=device,
        d_model=512)
transfomer.single_head()