from datasets import load_dataset
import requests
import json
from supabase import create_client, Client
import time
import os

model_name = "text-embedding-004"

dataset = load_dataset("MediaTek-Research/TCEval-v2", "drcd")

supabase_url = 'https://xxx.supabase.co'
supabase_api_key = ''

supabase: Client = create_client(supabase_url, supabase_api_key)

supabase.table('questions').delete().eq('model', model_name).execute()
supabase.table('paragraphs').delete().eq('model', model_name).execute()


# -----

gemini_api_key = 'xxx'

def get_google_embeddings(input, model="text-embedding-004"):
  payload = { "model": model, "content": { "parts": [ { "text": input }]} }
  headers = { "Content-Type": "application/json" }
  response = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent?key={gemini_api_key}", headers = headers, data = json.dumps(payload) )
  obj = json.loads(response.text)
  if response.status_code == 200 :
    return obj["embedding"]["values"]
  else :
    return obj

current_paragraph = ''
current_paragraph_id = 0

for i in range(0, 3493):
    print(i)
    data = dataset["test"][i]
    if current_paragraph != data["paragraph"]:
      embedding = get_google_embeddings( data["paragraph"] )
      response = supabase.table('paragraphs').upsert({"content": data["paragraph"], "embedding": embedding, "model": model_name }).execute()
      current_paragraph = data["paragraph"]
      current_paragraph_id = response.data[0]["id"]

    q_embedding = get_google_embeddings( data["question"] )
    supabase.table('questions').upsert({"dataset_id": i, "content": data["question"], "embedding": q_embedding, "model": model_name, "paragraph_id": current_paragraph_id }).execute()
