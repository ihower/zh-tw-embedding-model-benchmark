# Benchmark Embedding model and Reranker model

Dataset: 台達閱讀理解資料集 drcd from https://huggingface.co/datasets/MediaTek-Research/TCEval-v2

* 使用繁體中文評測各家 Embedding 模型的檢索能力 https://ihower.tw/blog/archives/12167
* 使用繁體中文評測各家 Reranker 模型的重排能力: https://ihower.tw/blog/archives/12227


## Code

- `migrate.py` 建立 SQLite3 資料庫
- `index-*.py` 各家進行 embeddings 索引，向量存入 DB
- `benchmark.py` 向量檢索評測
- `reranker-*.py` 二階段檢索評測

早期版本使用 Supabase 遠端資料庫，後來改用 SQlite3 本機跑