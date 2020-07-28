import numpy as np
import pandas as pd
import os
import gc

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler

import transformers
from transformers import AutoModel, AutoTokenizer, DistilBertModel, DistilBertConfig
from transformers import AdamW, get_linear_schedule_with_warmup

from ToxicClassification import settings

from sklearn import metrics
import tqdm


class Bert(nn.Module):

    def __init__(self, bert):
        super(Bert, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(768, 1)

    def forward(self, xb):

        o = self.bert(xb)
        apool = torch.mean(o[0], 1)
        x = self.dropout(apool)
        return self.linear(x)


def classify(text):
    print('start')

    path = settings.MEDIA_ROOT + "\distilbert.bin"
    MODEL_PATH = 'distilbert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    encode = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=192,
        pad_to_max_length=True,
        truncation=True,
    )
    device = torch.device('cpu')
    tokens = encode['input_ids']
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    tokens = tokens.to(device)
    config = DistilBertConfig()
    model = Bert(DistilBertModel(config))

    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)

    output = model(tokens)
    output = output.cpu().detach().numpy()

    print(output)
    output = 0.0 if output < 0.5 else 1.0
    return output
