from datasets import load_dataset

import jieba
from rank_bm25 import BM25Okapi
import numpy as np

import pdb


dataset = load_dataset("MediaTek-Research/TCEval-v2", "drcd")

# indexing
print("indexing")

current_paragraph = ''
current_paragraph_id = 0
answer_pairs = {}
corpus = []

for i in range(0, 3493):
    print(i, end=", ")
    data = dataset["test"][i]
    if current_paragraph != data["paragraph"]:
      corpus.append( data["paragraph"] )
      current_paragraph = data["paragraph"]
      current_paragraph_id = i
    answer_pairs[i] = current_paragraph_id


# 使用 jieba 進行分詞
corpus_tokenized = [list(jieba.cut(doc)) for doc in corpus]

# 建立 BM25 模型
bm25 = BM25Okapi(corpus_tokenized)

# benchmark
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

print("benchmarking")

for i in range(0, 3493):
  print(i, end=", ")

  data = dataset["test"][i]
  query = data["question"]

  query_tokenized = list(jieba.cut(query))

  scores = bm25.get_scores(query_tokenized)
  sorted_scores = np.argsort(scores)[::-1]
  best_indexes = sorted_scores[0:3493].tolist()

  #print(best_indexes)

  hit_answer_id = answer_pairs[i] # 這是黃金 paragraph_id

  #print(hit_answer_id)
  #pdb.set_trace()

  position = find_index(best_indexes, hit_answer_id)
  if position == "not_found":
    score = 0
  else:
    score = 1 / (position+1)

  mmr_data.append( score )
  hit_data.append( hit_answer_id in best_indexes )

average_hit = sum(hit_data) / len(hit_data)

print("---------------------------")
print(average_hit)

average_mrr = calculate_average(mmr_data)

print("MRR score:")
print(average_mrr)



