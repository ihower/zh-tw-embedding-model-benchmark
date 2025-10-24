from datasets import load_dataset
import sqlite3
import json
import time
import os
from sentence_transformers import SentenceTransformer

model_name = 'nvidia/NV-Embed-v2'
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
model.max_seq_length = 32768
model.tokenizer.padding_side = "right"

# 定義指令前綴
task_instruction = "Given a question, retrieve passages that answer the question"
query_prefix = f"Instruct: {task_instruction}\nQuery: "

def add_eos(input_examples):
    return [input_example + model.tokenizer.eos_token for input_example in input_examples]

current_paragraph = ''
current_paragraph_id = 0

for i in range(0, 3493):
    print(i)
    data = dataset["test"][i]
    if current_paragraph != data["paragraph"]:
        # 為段落添加 EOS 標記並編碼
        paragraph_with_eos = add_eos([data["paragraph"]])[0]
        embedding = model.encode(paragraph_with_eos, normalize_embeddings=True).tolist()
        cursor.execute(
            'INSERT INTO paragraphs (content, embedding, model) VALUES (?, ?, ?)',
            (data["paragraph"], json.dumps(embedding), model_name)
        )
        conn.commit()
        current_paragraph = data["paragraph"]
        current_paragraph_id = cursor.lastrowid

    # 為問題添加 EOS 標記並編碼，同時加上指令前綴
    question_with_eos = add_eos([data["question"]])[0]
    q_embedding = model.encode(question_with_eos, normalize_embeddings=True, prompt=query_prefix).tolist()
    cursor.execute(
        'INSERT INTO questions (dataset_id, content, embedding, model, paragraph_id) VALUES (?, ?, ?, ?, ?)',
        (i, data["question"], json.dumps(q_embedding), model_name, current_paragraph_id)
    )
    conn.commit()

# 關閉資料庫連接
conn.close()
