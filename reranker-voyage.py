from supabase import create_client, Client

model_name = "text-embedding-3-small"

print( model_name )

supabase_url = 'https://xxx.supabase.co'
supabase_api_key = ''

supabase: Client = create_client(supabase_url, supabase_api_key)

response1 = supabase.table('questions').select("content, embedding, paragraph_id").eq("model", model_name).execute()

question_contents = [x["content"] for x in response1.data]
question_embeddings = [ eval(x["embedding"]) for x in response1.data]

gold_paragraph_ids = [x["paragraph_id"] for x in response1.data]

response2 = supabase.table('paragraphs').select("id, content, embedding").eq("model", model_name).execute()

paragraph_ids = [x["id"] for x in response2.data]

paragraph_embeddings = [ eval(x["embedding"]) for x in response2.data]
paragraph_contents = [x["content"] for x in response2.data]


print( len(question_embeddings) ) # 應該要是 3493
print( len(paragraph_embeddings) ) # 應該要是 1000

print("Dimension:")
print( len(question_embeddings[0]) ) # 這是向量維度

print("----------------")

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 參數 list_of_doc_vectors 是所有文件的 embeddings 向量
# 參數 query_vector 是查詢字串的 embedding 向量
# 參數 top_k 是回傳的比數
def get_top_k_indices(list_of_doc_vectors, query_vector, top_k):
  # 轉成 numpy arrays
  list_of_doc_vectors = np.array(list_of_doc_vectors)
  query_vector = np.array(query_vector)

  # 逐筆計算 cosine similarities
  similarities = cosine_similarity(query_vector.reshape(1, -1), list_of_doc_vectors).flatten()

  # 根據 cosine similarity 排序
  sorted_indices = np.argsort(similarities)[::-1]

  # 取出 top K 的索引編號
  top_k_indices = sorted_indices[:top_k]

  return top_k_indices

def find_index(arr, target):
  try:
      index = arr.index(target)
      return index
  except ValueError:
      return "not_found"

def calculate_average(arr):
    if len(arr) == 0:
        return 0  # 防止除以零錯誤
    return sum(arr) / len(arr)


hit_data = []
mmr_data = []

## voyage
voyage_api_key = 'xxx'
import os
os.environ['VOYAGE_API_KEY'] = voyage_api_key

import voyageai
vo = voyageai.Client()

import time

print("voyageai rerank-1")


for idx, question_embedding in enumerate(question_embeddings):

  print(idx)

  if idx % 95 == 0:
    print("wait 20s.....")
    time.sleep(20) # 有 rate limit of 100 requests per minute

  best_indexes = get_top_k_indices(paragraph_embeddings, question_embedding, 50) # 取出 top_k 的 indexes
  # ----- reranker 後取 top 5

  # voyageai
  rerank_docs = [paragraph_contents[i] for i in best_indexes]

  try:
    results = vo.rerank(question_contents[idx], rerank_docs, model="rerank-1", top_k=5)

  except Exception as e:
    print("api error.....")
    time.sleep(61)
    # retry
    results = vo.rerank(question_contents[idx], rerank_docs, model="rerank-1", top_k=5)

  rerank_indexes = [x.index for x in results.results]

  best_best_indexes = [best_indexes[i] for i in rerank_indexes]

  context_ids = [paragraph_ids[i] for i in best_best_indexes] # 找出對應的 paragraph_ids

  # -----
  hit_paragraph_id = gold_paragraph_ids[idx] # 這是黃金 paragraph_id

  position = find_index(context_ids, hit_paragraph_id)
  if position == "not_found":
    score = 0
  else:
    score = 1 / (position+1)

  mmr_data.append( score )
  hit_data.append( hit_paragraph_id in context_ids )

average_hit = sum(hit_data) / len(hit_data)

print("---------------------------")
print(average_hit)

average_mrr = calculate_average(mmr_data)

print("MRR score:")
print(average_mrr)
