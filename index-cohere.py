from datasets import load_dataset
import sqlite3
import json
import time
import os

import cohere

model_name = "embed-v4.0"

dataset = load_dataset("MediaTek-Research/TCEval-v2", "drcd")

# 連接到 SQLite 資料庫
conn = sqlite3.connect('data.db')
cursor = conn.cursor()

# 清除現有的資料
cursor.execute('DELETE FROM questions WHERE model = ?', (model_name,))
cursor.execute('DELETE FROM paragraphs WHERE model = ?', (model_name,))
conn.commit()

# -----

cohere_key = os.environ["COHERE_API_KEY"]
co = cohere.Client(cohere_key)

# https://docs.cohere.com/reference/embed

# input_type = "search_document"
# input_type = "search_query"

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
      # 插入段落並獲取其 ID
      cursor.execute(
          'INSERT INTO paragraphs (content, embedding, model) VALUES (?, ?, ?)',
          (data["paragraph"], json.dumps(embedding), model_name)
      )
      conn.commit()
      current_paragraph = data["paragraph"]
      current_paragraph_id = cursor.lastrowid

    q_embedding = get_embeddings( data["question"], model_name, "search_query" )
    cursor.execute(
        'INSERT INTO questions (dataset_id, content, embedding, model, paragraph_id) VALUES (?, ?, ?, ?, ?)',
        (i, data["question"], json.dumps(q_embedding), model_name, current_paragraph_id)
    )
    conn.commit()

# 關閉資料庫連接
conn.close()
