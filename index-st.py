from datasets import load_dataset
import requests
import json
from supabase import create_client, Client
import time
import os

# sentence-transformers/all-MiniLM-L6-v2
# BAAI/bge-base-zh-v1.5
# BAAI/bge-small-zh-v1.5
# BAAI/bge-large-zh-v1.5
# BAAI/bge-m3
# jinaai/jina-embeddings-v2-base-zh
# thenlper/gte-small-zh
# thenlper/gte-base-zh
# thenlper/gte-large-zh
# maidalun1020/bce-embedding-base_v1
# nomic-ai/nomic-embed-text-v1.5
# intfloat/multilingual-e5-small
# intfloat/multilingual-e5-base
# intfloat/multilingual-e5-large
# aspire/acge_text_embedding
# DMetaSoul/Dmeta-embedding-zh
# infgrad/stella-mrl-large-zh-v3.5-1792d
# infgrad/stella-large-zh-v2
# infgrad/stella-base-zh-v2

model_name = 'sentence-transformers/all-MiniLM-L6-v2' # 填入要下載的模型
print(model_name)

dataset = load_dataset("MediaTek-Research/TCEval-v2", "drcd")

supabase_url = 'https://xxx.supabase.co'
supabase_api_key = ''

supabase: Client = create_client(supabase_url, supabase_api_key)

supabase.table('questions').delete().eq('model', model_name).execute()
supabase.table('paragraphs').delete().eq('model', model_name).execute()


# -----

from sentence_transformers import SentenceTransformer

model = SentenceTransformer(model_name, trust_remote_code=True)

current_paragraph = ''
current_paragraph_id = 0

for i in range(0, 3493):
    print(i)
    data = dataset["test"][i]
    if current_paragraph != data["paragraph"]:
      embedding = model.encode(data["paragraph"], normalize_embeddings=True).tolist()
      response = supabase.table('paragraphs').upsert({"content": data["paragraph"], "embedding": embedding, "model": model_name }).execute()
      current_paragraph = data["paragraph"]
      current_paragraph_id = response.data[0]["id"]

    q_embedding = model.encode(data["question"], normalize_embeddings=True).tolist()
    supabase.table('questions').upsert({"dataset_id": i, "content": data["question"], "embedding": q_embedding, "model": model_name, "paragraph_id": current_paragraph_id }).execute()
