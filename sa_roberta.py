# https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest

import pandas as pd
import os
from tqdm import tqdm

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import torch

# loading the tokenizer and model
task = 'sentiment'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}-latest"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
model.to(device)

START_TOKEN_ID = 0
PAD_TOKEN_ID = 1
END_TOKEN_ID = 2

WINDOW_SIZE = 510
STRIDE = 211

for root, dirs, files in os.walk(os.getcwd() + "/data", topdown=False):
    for file_name in files:

        if file_name.endswith("_cleaned.tsv"):

            # load the texts
            df = pd.read_csv(root + '//' + file_name, sep="\t")
            df = df[df['clean_comments'].isna() == False]
            df.reset_index(drop=True, inplace=True)
            comments = df['clean_comments'].tolist()
            predictions = []

            for c in tqdm(comments):

                # encoding the texts
                tokens = tokenizer.encode_plus(c, return_tensors='pt', add_special_tokens=False)

                input_ids = tokens['input_ids'][0]
                attention_mask = tokens['attention_mask'][0]
                # windowing
                total_length = len(input_ids)
                flag = True
                text_chunks = []
                start = 0
                chunk_scores = []
                while flag:
                    end = start + WINDOW_SIZE
                    if end >= total_length:
                        flag = False
                        end = total_length

                    # start token
                    input_chunk_ids = torch.concat([torch.IntTensor([START_TOKEN_ID]), input_ids[start:end]])
                    attention_chunk_mask = torch.concat([torch.IntTensor([1]), attention_mask[start:end]])

                    # end token
                    input_chunk_ids = torch.concat([input_chunk_ids, torch.IntTensor([END_TOKEN_ID])])
                    attention_chunk_mask = torch.concat([attention_chunk_mask, torch.IntTensor([1])])

                    # padding
                    padding_size = abs(WINDOW_SIZE + 2 - len(input_chunk_ids))
                    input_chunk_ids = torch.concat([input_chunk_ids, torch.IntTensor([PAD_TOKEN_ID] * padding_size)])

                    attention_chunk_mask = torch.concat([attention_chunk_mask, torch.IntTensor([0] * padding_size)])

                    input_chunk_ids = torch.reshape(input_chunk_ids, [-1, 512])
                    attention_chunk_mask = torch.reshape(attention_chunk_mask, [-1, 512])

                    input_dict = {
                        'input_ids': input_chunk_ids.long().to(device),
                        'attention_mask': attention_chunk_mask.int().to(device)
                    }

                    output = model(**input_dict)[0][0].detach()
                    scores = torch.softmax(output, dim=-1)
                    chunk_scores.append(scores)

                    start = start + STRIDE

                # stacking the probabilities
                chunk_scores = torch.stack(chunk_scores)

                # mean probabilities
                final_scores = chunk_scores.mean(dim=0)

                # Labels: 0 -> Negative; 1 -> Neutral; 2 -> Positive
                pred = torch.argmax(final_scores).item()

                predictions.append(pred)

            df["predicted_label"] = predictions

            # saving the outputs
            file_path = f"{root}/{file_name[:-4]}_predictions.tsv"
            df.to_csv(file_path, sep='\t', index=False)
