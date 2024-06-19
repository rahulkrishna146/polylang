import os
import math
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from mlm import BERTDataset
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
import time
# --------------------------------------

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.POLYLANG_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        #self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1,1,config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim= -1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.POLYLANG_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass 
class PolyLangConfig:
    block_size: int = 64 # max sequence length
    vocab_size: int = 1000 # got from running bpe
    n_layer: int = 12 # number of layers
    n_head: int = 8 # number of heads
    n_embd: int = 512 # embedding dimension, 2**9 = 512

class PolyLang(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight sharing schema 
        self.transformer.wte.weight = self.lm_head.weight

        #initialize params 
        self.apply(self._init_weights)

        if torch.cuda.is_available():
            device = "cuda:2"
        self.device = device
        print("Total Parameters:", sum([p.nelement() for p in self.parameters()]))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.04 #1/sqrt(512)
            if hasattr(module, 'POLYLANG_SCALE_INIT'):
                std *= (2*self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean= 0.0, std = std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.04)

    def forward(self, idx, targets=None):
        # idx is the shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        # For BERT calculate the loss on masked tokens 
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index = 0)
        return logits, loss

    @torch.no_grad()
    def generate_embedding(self, encoding):
        encoding = encoding.view(1, self.config.block_size)
        token_embedding = self.transformer.wte(encoding)
        position_embedding = self.transformer.wpe(torch.arange(self.config.block_size).to(self.device))

        x = token_embedding + position_embedding
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        emd = x.view(self.config.block_size, self.config.n_embd)
        return emd
    
    def mean_pooling(self,model_out,attention_mask):
        input_mask_expanded = (attention_mask.unsqueeze(-1).expand(model_out.size()).float())
        return torch.sum(model_out, 0) / torch.clamp(input_mask_expanded.sum(0), min = 1e-9)

    def get_psmile_embedding(self, text, tokenizer):
        # add special tokens begging and end 
        text = "<|SOS|>" + text + "<|EOS|>"
        out  = tokenizer.encode(text)
        encoding = out.ids
        # if the smile length is grater than bloack size truncate
        if len(encoding) > self.config.block_size:
            encoding = encoding[:self.config.block_size]
            attention_mask = [1 for _ in range(len(encoding))]
        else:
            padding = [0 for _ in range(self.config.block_size - len(encoding))]
            attention_mask = [1 for _ in range(len(encoding))]
            encoding = encoding + padding
            attention_mask = attention_mask + padding 
        encod = torch.tensor(encoding).to(self.device)
        attention_mask = torch.tensor(attention_mask).to(self.device)
        emb = self.generate_embedding(encod)
        return F.normalize(self.mean_pooling(emb, attention_mask).view(1, self.config.n_embd))

# ------------------------------------------------------
# tokenizer 

tok = Tokenizer.from_file("tokenizer_models/tokenizer-100m-HF-vocab1000.json")
print(f"Tokenizer loaded successfully, vocab_size = {tok.get_vocab_size()}")

#detect device 
device = "cpu"
if torch.cuda.is_available():
    device = "cuda:2"
print(f"using device: {device}")

# load the text file 
#print(f'Loading textfiles')
<<<<<<< HEAD
text = open("datasets/7m_traindata.txt", 'r')
train_set = [f"<|SOS|>{psmile}<|EOS|>" for psmile in text.read().splitlines()[:100000]]
=======
text = open("datasets/7k_psmiles.txt", 'r')
train_set = text.read().splitlines()
>>>>>>> a8ee2c4ff8e8d747fea3c2e4709dfaea71d9091f
print(f'Total number of lines in corpus in train: {len(train_set)}')

# initialize dataset 
#print(f"Loading training dataset")
<<<<<<< HEAD
block_size = 64 # set block size here 
=======
block_size = 64
>>>>>>> a8ee2c4ff8e8d747fea3c2e4709dfaea71d9091f
train_dataset = BERTDataset(data = train_set, 
    tokenizer = tok, 
    seq_len = block_size) # block_size
print(f"Loaded {len(train_set)*block_size}")

# initialize dataloader
<<<<<<< HEAD
batch_size = 128 # set batch size here
=======
batch_size = 64
>>>>>>> a8ee2c4ff8e8d747fea3c2e4709dfaea71d9091f
train_loader = DataLoader(dataset = train_dataset , 
    batch_size = batch_size, # set what fit on gpu, always a nice number
    shuffle=True)
print(f"1 epoch = {math.ceil(len(train_set)/batch_size)} batches")

#example batch 
#batch = next(iter(train_loader))
#x = batch['bert_input'].to(device)
#y = batch['bert_labels'].to(device)

# induce reproducability
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

#create model 
print("Building PolyLang..")
model = PolyLang(PolyLangConfig(block_size = block_size))
model.eval()
model.to(device)
#logits, loss = model(x['bert_input'].to(device),x['bert_labels'].to(device))
#print(logits.shape)
#print(loss.item())

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    t0 = time.time()
    batch = next(iter(train_loader))
    x = batch['bert_input'].to(device)
    y = batch['bert_labels'].to(device)
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1-t0)*1000 # time diff in milliseconds
<<<<<<< HEAD
    tokens_per_sec = (batch_size*block_size)/(t1-t0)
    print(f"step{i}, loss: {loss.item()}, dt: {dt:.2f}ms, tok/sec: {tokens_per_sec:.2f}")
=======
    print(f"step{i}, loss: {loss.item()}, dt: {dt:.2f}ms")
>>>>>>> a8ee2c4ff8e8d747fea3c2e4709dfaea71d9091f

import sys; sys.exit(0)
#import code; code.interact(local=locals())
# example psmile and embedding generation
psmile = '*CC(*)c1ccc(C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F)cc1'
embd = model.get_psmile_embedding(psmile, tokenizer = tok)
print(f'Embedding dimention : {embd.shape}')
print(f'The embedding looks like :....  {embd[0][:5]}')