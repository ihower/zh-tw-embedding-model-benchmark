from datasets import load_dataset
import requests
import json
from supabase import create_client, Client
import time
import os

import cohere

model_name = "embed-multilingual-light-v3.0"

dataset = load_dataset("MediaTek-Research/TCEval-v2", "drcd")

supabase_url = 'https://xxx.supabase.co'
supabase_api_key = ''

supabase: Client = create_client(supabase_url, supabase_api_key)

supabase.table('questions').delete().eq('model', model_name).execute()
supabase.table('paragraphs').delete().eq('model', model_name).execute()

# -----

cohere_key = "xxx"
co = cohere.Client(cohere_key)

# https://docs.cohere.com/reference/embed

def get_embeddings(input, model, input_type):
  doc_emb = co.embed( texts=[input], input_type=input_type, model=model).embeddings
  return doc_emb[0]

current_paragraph = ''
current_paragraph_id = 0

for i in range(0, 3493):
    print(i)
    data = dataset["test"][i]
    if current_paragraph != data["paragraph"]:
      embedding = get_embeddings( data["paragraph"], model_name, "search_document" )
      response = supabase.table('paragraphs').upsert({"content": data["paragraph"], "embedding": embedding, "model": model_name }).execute()
      current_paragraph = data["paragraph"]
      current_paragraph_id = response.data[0]["id"]

    q_embedding = get_embeddings( data["question"], model_name, "search_query" )
    supabase.table('questions').upsert({"dataset_id": i, "content": data["question"], "embedding": q_embedding, "model": model_name, "paragraph_id": current_paragraph_id }).execute()
