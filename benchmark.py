import sqlite3
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

model_name = 'google/embeddinggemma-300m' # "ckiplab/bert-base-chinese"

print( model_name )

# 建立 SQLite 連接
conn = sqlite3.connect('data.db')
cursor = conn.cursor()

# 取得 questions 資料
cursor.execute("SELECT embedding, paragraph_id FROM questions WHERE model = ?", (model_name,))
questions_data = cursor.fetchall()

question_embeddings = [eval(row[0]) for row in questions_data]
gold_paragraph_ids = [row[1] for row in questions_data]

# 取得 paragraphs 資料
cursor.execute("SELECT id, embedding FROM paragraphs WHERE model = ?", (model_name,))
paragraphs_data = cursor.fetchall()

paragraph_ids = [row[0] for row in paragraphs_data]
paragraph_embeddings = [eval(row[1]) for row in paragraphs_data]

# 關閉資料庫連接
conn.close()

print( len(question_embeddings) ) # 應該要是 3493
print( len(paragraph_embeddings) ) # 應該要是 1000

print("Dimension:")
print( len(question_embeddings[0]) ) # 這是向量維度

print("----------------")

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

for idx, question_embedding in enumerate(question_embeddings):

  print(idx)

  best_indexes = get_top_k_indices(paragraph_embeddings, question_embedding, 5) # 取出 top_k 的 indexes
  context_ids = [paragraph_ids[i] for i in best_indexes] # 找出對應的 paragraph_ids
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
