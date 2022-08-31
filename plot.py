import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
import os

result_folder = "/Results"

freq = defaultdict(int)

for root, dirs, files in os.walk(os.getcwd() + result_folder, topdown=False):
    for file_name in files:
        if file_name.endswith("_predictions.tsv"):
            df = pd.read_csv(root + "/" + file_name, sep="\t", lineterminator="\n")
            df = df[~df["predicted_label"].isna()]

            df["predicted_label"] = df["predicted_label"].astype(int)

            freq["negative"] += int((df["predicted_label"] == 0).sum())
            freq["neutral"] += int((df["predicted_label"] == 1).sum())
            freq["positive"] += int((df["predicted_label"] == 2).sum())

plt.bar(x=freq.keys(), height=freq.values(), color=["red", "gray", "green"])
plt.title("Predicted Labels Frequency")
plt.savefig(fname="Predicted_Labels_Frequency.pdf", format="pdf")
plt.show()

print(freq.items())
print(sum(freq.values()))
