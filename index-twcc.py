from datasets import load_dataset
import requests
import json
from supabase import create_client, Client
import time
import os

model_name = "ffm-embedding"

dataset = load_dataset("MediaTek-Research/TCEval-v2", "drcd")

supabase_url = 'https://imcpayinnpcetclzvdfu.supabase.co'
supabase_api_key = ''

supabase: Client = create_client(supabase_url, supabase_api_key)

supabase.table('questions').delete().eq('model', model_name).execute()
supabase.table('paragraphs').delete().eq('model', model_name).execute()


# -----

twcc_api_key = os.environ["TWCC_API_KEY"]
twcc_api_url = 'https://api-ams.twcc.ai/api'

def get_twcc_embeddings(input):
  # https://member.twcc.ai/
  # https://www.twcc.ai/user/afsms/model/public/list
  payload = { "inputs": [input], "model": model_name }
  headers = { "X-API-KEY": twcc_api_key, "X-API-HOST": "afs-inference", "Content-Type": "application/json"  }
  response = requests.post(twcc_api_url + '/models/embeddings', headers = headers, data = json.dumps(payload) )
  obj = json.loads(response.text)
  if response.status_code == 200 :
    return obj["data"][0]["embedding"]
  else :
    return obj["error"]

current_paragraph = ''
current_paragraph_id = 0

for i in range(0, 3493):
    print(i)
    data = dataset["test"][i]
    if current_paragraph != data["paragraph"]:
      embedding = get_twcc_embeddings( data["paragraph"] )
      response = supabase.table('paragraphs').upsert({"content": data["paragraph"], "embedding": embedding, "model": model_name }).execute()
      current_paragraph = data["paragraph"]
      current_paragraph_id = response.data[0]["id"]

    q_embedding = get_twcc_embeddings( data["question"] )
    supabase.table('questions').upsert({"dataset_id": i, "content": data["question"], "embedding": q_embedding, "model": model_name, "paragraph_id": current_paragraph_id }).execute()
