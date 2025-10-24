from datasets import load_dataset
import requests
import json
import sqlite3
import time
import os

model_name = "text-embedding-3-small"

dataset = load_dataset("MediaTek-Research/TCEval-v2", "drcd")

# 連接到 SQLite 資料庫
conn = sqlite3.connect('data.db')
cursor = conn.cursor()

# 清除現有的資料
cursor.execute('DELETE FROM questions WHERE model = ?', (model_name,))
cursor.execute('DELETE FROM paragraphs WHERE model = ?', (model_name,))
conn.commit()

# -----

openai_api_key = os.environ['OPENAI_API_KEY']

def get_embeddings(input, model):
  payload = { "input": input, "model": model }
  headers = { "Authorization": f'Bearer {openai_api_key}', "Content-Type": "application/json" }
  response = requests.post('https://api.openai.com/v1/embeddings', headers = headers, data = json.dumps(payload) )
  obj = json.loads(response.text)
  if response.status_code == 200 :
    return obj["data"][0]["embedding"]
  else:
    time.sleep(3)
    print("embedding error..... retrying")
    # retry
    return get_embeddings(input, model)

current_paragraph = ''
current_paragraph_id = 0

for i in range(0, 3493):
    print(i)
    data = dataset["test"][i]
    if current_paragraph != data["paragraph"]:
      embedding = get_embeddings( data["paragraph"], model_name )
      # 插入段落並獲取其 ID
      cursor.execute(
          'INSERT INTO paragraphs (content, embedding, model) VALUES (?, ?, ?)',
          (data["paragraph"], json.dumps(embedding), model_name)
      )
      conn.commit()
      current_paragraph = data["paragraph"]
      current_paragraph_id = cursor.lastrowid

    q_embedding = get_embeddings( data["question"], model_name )
    cursor.execute(
        'INSERT INTO questions (dataset_id, content, embedding, model, paragraph_id) VALUES (?, ?, ?, ?, ?)',
        (i, data["question"], json.dumps(q_embedding), model_name, current_paragraph_id)
    )
    conn.commit()

# 關閉資料庫連接
conn.close()
