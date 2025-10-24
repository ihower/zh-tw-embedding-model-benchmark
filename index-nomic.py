from datasets import load_dataset
import sqlite3
import json
import time
import os
from sentence_transformers import SentenceTransformer

# https://www.nomic.ai/blog/posts/nomic-embed-text-v2

# https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe

model_name = 'nomic-ai/nomic-embed-text-v2-moe'
print(model_name)

dataset = load_dataset("MediaTek-Research/TCEval-v2", "drcd")

# 連接到 SQLite 資料庫
conn = sqlite3.connect('data.db')
cursor = conn.cursor()

# 清除現有的資料
cursor.execute('DELETE FROM questions WHERE model = ?', (model_name,))
cursor.execute('DELETE FROM paragraphs WHERE model = ?', (model_name,))
conn.commit()

# -----

model = SentenceTransformer(model_name, trust_remote_code=True)

current_paragraph = ''
current_paragraph_id = 0

for i in range(0, 3493):
    print(i)
    data = dataset["test"][i]
    if current_paragraph != data["paragraph"]:
        embedding = model.encode(data["paragraph"], prompt_name="passage").tolist()
        # 插入段落並獲取其 ID
        cursor.execute(
            'INSERT INTO paragraphs (content, embedding, model) VALUES (?, ?, ?)',
            (data["paragraph"], json.dumps(embedding), model_name)
        )
        conn.commit()
        current_paragraph = data["paragraph"]
        current_paragraph_id = cursor.lastrowid

    q_embedding = model.encode(data["question"], prompt_name="query").tolist()
    cursor.execute(
        'INSERT INTO questions (dataset_id, content, embedding, model, paragraph_id) VALUES (?, ?, ?, ?, ?)',
        (i, data["question"], json.dumps(q_embedding), model_name, current_paragraph_id)
    )
    conn.commit()

# 關閉資料庫連接
conn.close()
