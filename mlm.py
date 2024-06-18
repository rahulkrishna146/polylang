"""
Dataset and Dataset Loader creation
MLM preprocessing and padding 
""" 

import torch
from torch.utils.data import Dataset, DataLoader
import itertools
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import tqdm
import copy

#hyper param 

# data ---> text 
class BERTDataset(Dataset):
    def __init__(self, data, tokenizer, seq_len):
        self.data = data
        self.tokenizer = tokenizer
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        psmile = self.data[idx]
        out = self.tokenizer.encode(psmile)
        encoding = out.ids
        input_ids , bert_labels = self.mlm(encoding)
        # truncate to max sequence length else padd it 
        if len(input_ids) > self.seq_len:
            input_ids = input_ids[:self.seq_len]
            bert_labels = bert_labels[:self.seq_len]
        else:
            input_ids = self.pad(input_ids)
            bert_labels = self.pad(bert_labels)
        output = {
            "bert_input" : input_ids,
            "bert_labels": bert_labels
        }
        return {key: torch.tensor(value) for key, value in output.items()}

    def mlm(self, encoding): # tokenizer output goes here
        input_ids = copy.copy(encoding)
        bert_labels = [0 for _ in range(len(encoding))]
        rand = torch.rand(len(encoding))
        #mask arr can be masked with any tokens ---other than 0--> pad ,1---> SOS, 2--> EOS
        mask_arr = (rand< 0.20)*(torch.tensor([False if (encod == 0 or encod == 1 or encod ==2) else True for encod in encoding]))
        for i in range(len(mask_arr)):
            if mask_arr[i] == True:
                input_ids[i] = 3 # masking token = 3
                bert_labels[i] = encoding[i]
        return input_ids, bert_labels

    def pad(self,encoding):
        padding = [0 for _ in range(self.seq_len - len(encoding))]
        padded = encoding + padding
        return padded
