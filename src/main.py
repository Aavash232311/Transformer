
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


class MultipleHead(nn.Module):

    def __init__(self, n_head):
        super().__init__()
        self.n_head = n_head

class SingleHeaded(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.key = nn.Linear(dim, dim) # it's learnable so using nn instead of matrix
        self.query = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
    
    ''' Attention(Q, K, V) = softmax(QK^T / √d_k) V '''
    def forward(self, x):
        B,T,C = x.shape
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        # from the paper we need to calculate attention now
        dk = query.size(-1)
        # Query times transpose of key
        # (B, C, T) instead of (B, T, C), new to torch keeping notes of small things
        q_k_t = torch.matmul(query, key.transpose(-2, -1)) 
        scores = q_k_t / torch.sqrt(torch.tensor(dk, dtype=torch.float32))

        # we need to do masking here before applying softmax
        mask = torch.tril(torch.ones(T, T)).view(1, T, T).to(x.device)
        scores = scores.masked_fill(mask == 0, -1e9)  # It's like a lower triangular matrix

        # finally apply softmax, dim=-1 probability distribution where each row sums to 1
        res = torch.matmul(torch.softmax(scores, dim=-1), value)
        print(res)
        return res

class Transfomer:

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
  
        positional_embedding_table = self.positional_embedding_table(pos)
        # adding those up like in the original paper
        res = token_embedding_table + positional_embedding_table
        return res
    

    def single_head(self):
        prompt = torch.tensor(tokenizer.encode("Alice"), dtype=torch.long, device=device).unsqueeze(0)
        embedding = transfomer.forward(prompt)

        # now we want to work in a single head of attention 
        single_headed = SingleHeaded(self.d_model).to(self.device) # bringing to NVIDA gpu if avalible 
        single_headed.forward(embedding)

transfomer = Transfomer(batch_size=4,
        block_size=8,
        device=device,
        d_model=512)
transfomer.single_head()