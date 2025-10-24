from datasets import load_dataset
import sqlite3
import json
import time
import os
from FlagEmbedding import FlagModel

model_name = 'chuxin-llm/Chuxin-Embedding'
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

# 初始化模型
model = FlagModel(model_name,
                 query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                 use_fp16=True)

current_paragraph = ''
current_paragraph_id = 0

for i in range(0, 3493):
    print(i)
    data = dataset["test"][i]
    if current_paragraph != data["paragraph"]:
        embedding = model.encode([data["paragraph"]])[0].tolist()  # 注意這裡需要傳入列表
        # 插入段落並獲取其 ID
        cursor.execute(
            'INSERT INTO paragraphs (content, embedding, model) VALUES (?, ?, ?)',
            (data["paragraph"], json.dumps(embedding), model_name)
        )
        conn.commit()
        current_paragraph = data["paragraph"]
        current_paragraph_id = cursor.lastrowid

    q_embedding = model.encode([data["question"]])[0].tolist()  # 注意這裡需要傳入列表
    cursor.execute(
        'INSERT INTO questions (dataset_id, content, embedding, model, paragraph_id) VALUES (?, ?, ?, ?, ?)',
        (i, data["question"], json.dumps(q_embedding), model_name, current_paragraph_id)
    )
    conn.commit()

# 關閉資料庫連接
conn.close()
