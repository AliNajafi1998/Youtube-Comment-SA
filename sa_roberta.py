# https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment?text=Like+I+care+about+you

import pandas as pd
import numpy as np
import os
from tqdm import tqdm

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import torch
import spacy
from scipy.special import softmax

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# loading the tokenizer and model
task = 'sentiment'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
model.to(device)

START_TOKEN_ID = 0
PAD_TOKEN_ID = 1
END_TOKEN_ID = 2


def get_chunks(token_ids, window_size=510, stride=211):
    total_length = len(token_ids)
    flag = True
    text_chunks = []
    start = 0
    while flag:
        end = start + window_size
        if end >= total_length:
            flag = False
            end = total_length

        # start token
        input_chunk_ids = torch.concat([torch.IntTensor([START_TOKEN_ID]), token_ids[start:end]])

        # end token
        input_chunk_ids = torch.concat([input_chunk_ids, torch.IntTensor([END_TOKEN_ID])])

        # padding
        padding_size = abs(window_size + 1 - len(input_chunk_ids))
        input_chunk_ids = torch.concat(
            [input_chunk_ids, torch.IntTensor([PAD_TOKEN_ID] * padding_size)])
        input_chunk_ids = torch.reshape(input_chunk_ids, [-1, 512])
        text_chunks.append(input_chunk_ids.to(device))
        start = start + stride
    return text_chunks


# load the texts
for root, dirs, files in os.walk(os.getcwd() + "\\data", topdown=False):
    for file_name in files:

        if file_name.endswith("_cleaned.tsv"):
            # segment it
            df = pd.read_csv(root + '//' + file_name, sep="\t")
            df = df[df['clean_comments'].isna() == False]
            comments = df['clean_comments'].tolist()
            for c in comments:
                input_ids = tokenizer.encode(c, return_tensors='pt', add_special_tokens=False)[0]

                chunks = get_chunks(input_ids)

                # for chunk in chunks:
                #     output = model(chunk)
                #     scores = output[0][0].detach().numpy()
                #     scores = softmax(scores)
