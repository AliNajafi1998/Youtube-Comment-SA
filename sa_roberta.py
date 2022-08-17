# https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment

import pandas as pd
import os
from tqdm import tqdm

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import torch

# loading the tokenizer and model
task = 'sentiment'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        padding_size = abs(window_size + 2 - len(input_chunk_ids))
        input_chunk_ids = torch.concat(
            [input_chunk_ids, torch.IntTensor([PAD_TOKEN_ID] * padding_size)])
        input_chunk_ids = torch.reshape(input_chunk_ids, [-1, 512])
        text_chunks.append(input_chunk_ids.to(device))
        start = start + stride
    return text_chunks


for root, dirs, files in os.walk(os.getcwd() + "\\data", topdown=False):
    for file_name in tqdm(files):

        if file_name.endswith("_cleaned.tsv"):

            # load the texts
            df = pd.read_csv(root + '//' + file_name, sep="\t")
            df = df[df['clean_comments'].isna() == False]
            df.reset_index(drop=True, inplace=True)
            comments = df['clean_comments'].tolist()
            predictions = []
            for c in comments:
                # encoding the texts
                input_ids = tokenizer.encode(c, return_tensors='pt', add_special_tokens=False)[0]

                # windowing
                chunks = get_chunks(input_ids)

                # preparing batch
                chunks = torch.concat(chunks, axis=0).to(device)

                # batch predicting
                output = model(chunks)[0].detach()

                # softmax scores
                batch_scores = torch.softmax(output[0], dim=-1)
                batch_scores = torch.reshape(batch_scores, [-1, 3])

                # mean probabilities
                scores = torch.mean(batch_scores, dim=0)

                # Labels: 0 -> Negative; 1 -> Neutral; 2 -> Positive
                pred = torch.argmax(scores).item()
                predictions.append(pred)

            df["predicted_label"] = predictions

            # saving the outputs
            file_path = f"{root}\\{file_name[:-4]}_predictions.tsv"
            df.to_csv(file_path, sep='\t', index=False)
