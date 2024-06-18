import os
import math
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
# --------------------------------------

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
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
    block_size: int = 256 # max sequence length
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
        if torch.cuda.is_available():
            device = "cuda"
        self.device = device
        print("Total Parameters:", sum([p.nelement() for p in self.parameters()]))

    def forward(self, idx, targets):
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
from tokenizers import Tokenizer

tok = Tokenizer.from_file("tokenizer_models/tokenizer-100m-HF-vocab1000.json")

#detect device 
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")

#create model 
print("Building PolyLang..")
model = PolyLang(PolyLangConfig())
model.to(device)

# example psmile and embedding generation
psmile = '*CC(*)c1ccc(C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F)cc1'
embd = model.get_psmile_embedding(psmile, tokenizer = tok)
print(f'Embedding dimention : {embd.shape}')
print(f'The embedding looks like :....  {embd[0][:5]}')