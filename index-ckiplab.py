
from transformers import BertTokenizer, BertModel
import torch
import torch.nn.functional as F

# 選擇使用的模型
model_name = 'ckiplab/bert-base-chinese'

# 載入 tokenizer 和 model
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

def get_embeddings(text):
    inputs = tokenizer(text, max_length=512, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**inputs)

    embeddings = outputs.last_hidden_state

    first_token_embedding = embeddings[0][0]
    normalized_first_token_embedding = F.normalize(first_token_embedding, p=2, dim=0)

    return normalized_first_token_embedding.tolist()

# ----

from datasets import load_dataset
import requests
import json
from supabase import create_client, Client
import time
import os

print(model_name)

dataset = load_dataset("MediaTek-Research/TCEval-v2", "drcd")

supabase_url = 'https://xxx.supabase.co'
supabase_api_key = '...'

supabase: Client = create_client(supabase_url, supabase_api_key)

supabase.table('questions').delete().eq('model', model_name).execute()
supabase.table('paragraphs').delete().eq('model', model_name).execute()


# -----

current_paragraph = ''
current_paragraph_id = 0

for i in range(0, 3493):
    print(i)
    data = dataset["test"][i]
    if current_paragraph != data["paragraph"]:
      embedding = get_embeddings(data["paragraph"])
      response = supabase.table('paragraphs').upsert({"content": data["paragraph"], "embedding": embedding, "model": model_name }).execute()
      current_paragraph = data["paragraph"]
      current_paragraph_id = response.data[0]["id"]

    q_embedding = get_embeddings(data["question"])
    supabase.table('questions').upsert({"dataset_id": i, "content": data["question"], "embedding": q_embedding, "model": model_name, "paragraph_id": current_paragraph_id }).execute()





