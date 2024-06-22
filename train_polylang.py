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
import inspect
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
        
        # comment out next three lines and 
        #att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        #att = F.softmax(att, dim= -1)
        #y = att @ v
        y = F.scaled_dot_product_attention(q,k,v, is_causal= True)

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
        total_params = sum([p.nelement() for p in self.parameters()])
        #print(f"Total Parameters:{total_params}, {(total_params/1e+6):2f}M")

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

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        
        #print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        #print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        
        #print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

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


# ------------------------------------------------------
# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

# device vs. device_type distinction
device_type = "cuda" if device.startswith("cuda") else "cpu"

# tokenizer 
tok = Tokenizer.from_file("tokenizer_models/tokenizer-100m-HF-vocab1000.json")
if master_process:
    print(f"Tokenizer loaded successfully, vocab_size = {tok.get_vocab_size()}")
# load the text file 
#print(f'Loading textfiles')
text = open("datasets/7m_traindata.txt", 'r')
train_set = [f"<|SOS|>{psmile}<|EOS|>" for psmile in text.read().splitlines()[:500000]]
if master_process:
    print(f'Total number of lines in corpus in train: {len(train_set)}')
    print(f"Pretraning on {(len(train_set)/1e+6):2f}M PSMILES")

# initialize dataset 
#print(f"Loading training dataset")
block_size = 64 # set block size here 
train_dataset = BERTDataset(data = train_set, 
    tokenizer = tok, 
    seq_len = block_size) # block_size
if master_process:
    print(f"Loaded {len(train_set)*block_size} tokens")

# gradient accumulation
total_batch_size = 2**18 # 2**17 in terms of tokens 
# initialize dataloader
batch_size = 256 # set micro_batch size here,what fits max on gpu 
assert total_batch_size % (batch_size * block_size * ddp_world_size) == 0
grad_accum_steps = total_batch_size // (batch_size*block_size*ddp_world_size)
if master_process:
    print(f"Total desired batch size: {total_batch_size}")
    print(f"Micro batch size that fot the gpu: {batch_size}")
    print(f"=> calculate gradient accumulation step: {grad_accum_steps}")

train_loader = DataLoader(dataset = train_dataset , 
    batch_size = batch_size, # set what fit on gpu, always a nice number
    shuffle=True,
    num_workers= ddp_world_size)
if master_process:
    print(f"1 epoch = {len(train_set)//batch_size} batches")
    print(f"Maximum context length(block size):{block_size}")

#example batch 
#batch = next(iter(train_loader))
#x = batch['bert_input'].to(device)
#y = batch['bert_labels'].to(device)

# induce reproducability
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# A100
torch.set_float32_matmul_precision('high')

# DDP launch for 4 gpu
# torchrun --standalone --nproc_per_node=4 train_polylang.py


#create model 
model = PolyLang(PolyLangConfig(block_size = block_size, vocab_size=1024))
model.to(device)#
# torch.compile only works with python 3.8 to 3.11
model = torch.compile(model)
if ddp:
    model =DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model 

#logits, loss = model(x['bert_input'].to(device),x['bert_labels'].to(device))
#print(logits.shape)
#print(loss.item())

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

#opimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4,betas=(0.9, 0.95), eps = 1e-8)
optimizer= raw_model.configure_optimizers(weight_decay= 0.1,learning_rate=6e-4, device_type='cuda')
  
#iteration loop
for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        batch = next(iter(train_loader))
        x = batch['bert_input'].to(device)
        y = batch['bert_labels'].to(device)
        #a100  ampere
        with torch.autocast(device_type = 'cuda', dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss /grad_accum_steps
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps -1)
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op = dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and set the learing rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1-t0)*1000 # time diff in milliseconds
    tokens_per_sec = (batch_size*block_size*grad_accum_steps*ddp_world_size)/(t1-t0)
    if master_process:
        print(f"step{step:4d} | loss: {loss_accum.item():6f} | lr:{lr:4e} | norm: {norm:4f} | dt: {dt:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
if ddp:
    destroy_process_group()
import sys; sys.exit(0)



#import code; code.interact(local=locals())
# example psmile and embedding generation
psmile = '*CC(*)c1ccc(C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F)cc1'
embd = model.get_psmile_embedding(psmile, tokenizer = tok)
print(f'Embedding dimention : {embd.shape}')
print(f'The embedding looks like :....  {embd[0][:5]}')