import os
import math
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
# --------------------------------------

@dataclass 
class PolyLangConfig:
    block_size: int = 256 # max sequence length
    vocab_size: int = 1000 # got from running bpe
    n_layer: int = 12 # number of layers
    n_head: int = 8 # number of heads
    n_embd: int = 512 # embedding dimension, 2**9 = 512


