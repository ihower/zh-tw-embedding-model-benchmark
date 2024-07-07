from datasets import load_dataset
import requests
import json
from supabase import create_client, Client
import time
import os


# 本來還想測試多模態的 multimodalembedding@001 (1408d)
# 不過碰到錯誤: "Multimodal embedding failed with the following error: Text field must be smaller than 1024 characters."

model_name = "text-multilingual-embedding-002"

dataset = load_dataset("MediaTek-Research/TCEval-v2", "drcd")

supabase_url = 'https://xxx.supabase.co'
supabase_api_key = ''

supabase: Client = create_client(supabase_url, supabase_api_key)

supabase.table('questions').delete().eq('model', model_name).execute()
supabase.table('paragraphs').delete().eq('model', model_name).execute()


# -----

gcp_project_id = 'xxx'

# gcloud auth print-access-token
gcp_access_token = 'yyy'

def get_vertex_text_embeddings(input):
  payload = {
    "instances": [
      {
        "content": input # text-multilingual-embedding-002 用這個
        # "text": input # multimodalembedding@001 用這個
      }
    ]
  }

  headers = { "Authorization": f'Bearer {gcp_access_token}', "Content-Type": "application/json" }
  response = requests.post(f"https://us-central1-aiplatform.googleapis.com/v1/projects/{gcp_project_id}/locations/us-central1/publishers/google/models/{model_name}:predict", headers = headers, data = json.dumps(payload) )
  obj = json.loads(response.text)

  if response.status_code == 200 :
    return obj["predictions"][0]["embeddings"]["values"] # text-multilingual-embedding-002 用這個
    # return obj["predictions"][0]["textEmbedding"] # multimodalembedding@001 用這個
  else :
    return obj

current_paragraph = ''
current_paragraph_id = 0

for i in range(0, 3493):
    print(i)
    data = dataset["test"][i]
    if current_paragraph != data["paragraph"]:
      embedding = get_vertex_text_embeddings( data["paragraph"] )
      response = supabase.table('paragraphs').upsert({"content": data["paragraph"], "embedding": embedding, "model": model_name }).execute()
      current_paragraph = data["paragraph"]
      current_paragraph_id = response.data[0]["id"]

    q_embedding = get_vertex_text_embeddings( data["question"] )
    supabase.table('questions').upsert({"dataset_id": i, "content": data["question"], "embedding": q_embedding, "model": model_name, "paragraph_id": current_paragraph_id }).execute()
