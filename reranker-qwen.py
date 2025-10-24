import sqlite3
import json
from datetime import datetime

model_name = "text-embedding-3-small"

print(datetime.now())
print(model_name)

# 連接到 SQLite 資料庫
conn = sqlite3.connect('data.db')
cursor = conn.cursor()

# 從 questions 表獲取資料
cursor.execute("SELECT content, embedding, paragraph_id FROM questions WHERE model = ?", (model_name,))
response1 = cursor.fetchall()

question_contents = [x[0] for x in response1]
question_embeddings = [json.loads(x[1]) for x in response1]
gold_paragraph_ids = [x[2] for x in response1]

# 從 paragraphs 表獲取資料
cursor.execute("SELECT id, content, embedding FROM paragraphs WHERE model = ?", (model_name,))
response2 = cursor.fetchall()

paragraph_ids = [x[0] for x in response2]
paragraph_contents = [x[1] for x in response2]
paragraph_embeddings = [json.loads(x[2]) for x in response2]

# 關閉資料庫連接
conn.close()

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

# 移除 FlagReranker 相關的 import
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

def format_instruction(instruction, query, doc):
    output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(instruction=instruction,query=query, doc=doc)
    return output

def process_inputs(pairs):
    inputs = tokenizer(
        pairs, padding=False, truncation='longest_first',
        return_attention_mask=False, max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
    )
    for i, ele in enumerate(inputs['input_ids']):
        inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
    inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
    for key in inputs:
        inputs[key] = inputs[key].to(model.device)
    return inputs

@torch.no_grad()
def compute_logits(inputs, **kwargs):
    batch_scores = model(**inputs).logits[:, -1, :]
    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]
    batch_scores = torch.stack([false_vector, true_vector], dim=1)
    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
    scores = batch_scores[:, 1].exp().tolist()
    return scores

# 初始化 Qwen3-Reranker
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-0.6B", padding_side='left')
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-0.6B").eval()
token_false_id = tokenizer.convert_tokens_to_ids("no")
token_true_id = tokenizer.convert_tokens_to_ids("yes")
max_length = 8192

prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

task = 'Given a web search query, retrieve relevant passages that answer the query'

search_top_k = 50
print(search_top_k)

print(datetime.now())

for idx, question_embedding in enumerate(question_embeddings):

  print(idx)

  best_indexes = get_top_k_indices(paragraph_embeddings, question_embedding, search_top_k) # 取出 top_k 的 indexes
  # ----- reranker 後取 top 5

  rerank_docs = [paragraph_contents[i] for i in best_indexes]

  # 使用新的 reranker 格式
  pairs = [format_instruction(task, question_contents[idx], doc) for doc in rerank_docs]
  inputs = process_inputs(pairs)
  scores = compute_logits(inputs)
  
  ordered_index = np.argsort(scores)[::-1]
  rerank_indexes = ordered_index[0:5] # 取前5

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

print(datetime.now())
