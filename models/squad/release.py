# %%
from transformers import pipeline
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import json

# %%
model_name = "deepset/minilm-uncased-squad2"

# %%
device = torch.device('cuda:0' if torch.cuda.is_available()
                      else 'cpu')
print(device)

# %%
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name, device=device)

# %%
def get_predictions(question, context):
    QA_input = {
        'question': question,
        'context': context
    }
    return nlp(QA_input)

# %%
path = Path("squad/dev-v2.0.json")

with open(path, 'rb') as f:
    squad_dict = json.load(f)

texts = []
queries = []
answers = []
qid = []

# Search for each passage, its question and its answer
for group in squad_dict['data']:
    for passage in group['paragraphs']:
        context = passage['context']
        for qa in passage['qas']:
            question = qa['question']
            id = qa["id"]
            
            if(len(qa['answers']) == 0):
                texts.append(context)
                queries.append(question)
                qid.append(id)
                answers.append("")
            else:
                for answer in qa['answers']:
                    # Store every passage, query and its answer to the lists
                    texts.append(context)
                    queries.append(question)
                    qid.append(id)
                    answers.append(answer)


val_texts, val_queries, val_answers, val_qid = texts, queries, answers, qid

# %%
all_predictions = get_predictions(val_queries, val_texts)

# %%
predictions = {}
scores = {}

for i in range(len(all_predictions)):
    key = val_qid[i]

    prediction = all_predictions[i]

    predictions[key] = prediction["answer"]
    scores[key] = abs(all_predictions[i]["score"])
    if(scores[key] < 0.24):
        predictions[key] = ""

with open("predictions.json", "w") as outfile:
    json.dump(predictions, outfile)


with open("scores.json", "w") as outfile:
    json.dump(scores, outfile)


