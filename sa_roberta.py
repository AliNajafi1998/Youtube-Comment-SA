# https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment?text=Like+I+care+about+you

import pandas as pd
import numpy as np
import os
from tqdm import tqdm

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import spacy
from scipy.special import softmax

#
# nlp = spacy.load('en_core_web_md')
#
#
# def split_to_sentences(comment: str):
#     for i in nlp(comment).sents:
#         print(i)
#
#
# split_to_sentences("Love your videos so much that I watched all the ads lol.")


# loading the tokenizer and model
task = 'sentiment'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# load the texts
for root, dirs, files in os.walk(os.getcwd() + "\\data", topdown=False):
    for file_name in tqdm(files):
        if file_name.endswith("_cleaned.tsv"):
            # segment it
            pass

# segmenting the texts with the size of 512

# predicting

text = "so good"
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
scores = output[0][0].detach().numpy()
# Labels: 0 -> Negative; 1 -> Neutral; 2 -> Positive
scores = softmax(scores)

print(scores)

# saving the results
