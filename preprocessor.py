import html
import pandas as pd
from tqdm import tqdm
import os

from NLPKickStart.NlpKickStart import (Pipeline, UnicodeNormalizer, EmojiHandler, RegexHandler, HashTagHandler,
                                       URLHandler, HTMLHandler, MentionHandler, CaseFoldingNormalizer,
                                       NLTKSentenceTokenizer, EmailHandler, ExpandContractionHandler)

# building the preprocessing pipeline
pipeline = Pipeline([
    html.unescape,
    UnicodeNormalizer(),
    EmailHandler(),
    EmojiHandler(),
    RegexHandler("&#39;", "'"),
    ExpandContractionHandler(),
    HashTagHandler(keep=False),
    HTMLHandler(),
    URLHandler(keep=False),
    MentionHandler(keep=False),
    RegexHandler(r'\s+', ' '),
    RegexHandler(r"[^a-zA-Z0-9\. ]", ' '),
    CaseFoldingNormalizer(),
    RegexHandler(r"\.+", "."),
    RegexHandler(r"  +", " "),
    str.strip,
])


def preprocess(text: str):
    return pipeline([text])[0]


# iterating over the comment files
for root, dirs, files in os.walk(os.getcwd() + "\\data", topdown=False):
    for file_name in tqdm(files):
        if file_name.endswith("_comments.tsv"):
            print(root + '\\' + file_name)
            df = pd.read_csv(root + '\\' + file_name, sep='\t').drop("Unnamed: 0", axis=1)

            # dropping the nan values
            df = df[df['comment'].isna() == False]

            # cleaning
            df["clean_comments"] = df['comment'].apply(preprocess)

            # saving
            file_path = f"{root}\\{file_name[:-4]}_cleaned.tsv"
            df.to_csv(file_path, sep="\t", index=False)
