# https://github.com/ckiplab/ckip-transformers

from transformers import BertTokenizer, BertModel
import torch
import torch.nn.functional as F

# 選擇使用的模型
model_name = 'ckiplab/bert-base-chinese'

# 載入 tokenizer 和 model
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# 輸入字串
text = "傅達仁今將執行安樂死"

# 將字串轉換成 token ids
# 最多只能 512 tokens
# RuntimeError: The size of tensor a (535) must match the size of tensor b (512) at non-singleton dimension 1
inputs = tokenizer(text, max_length=512, return_tensors='pt')

# 使用模型進行推理，獲取 embeddings 向量
with torch.no_grad():
    outputs = model(**inputs)

# 獲取最後一層的隱藏狀態
embeddings = outputs.last_hidden_state

# 查看 embeddings 向量形狀
print(embeddings.shape)  # (batch_size, sequence_length, hidden_size)

# 提取第一個 token 的 embeddings 向量
first_token_embedding = embeddings[0][0]

normalized_first_token_embedding = F.normalize(first_token_embedding, p=2, dim=0)

print( first_token_embedding.tolist() ) # 768 維度

print("----")

print( normalized_first_token_embedding.tolist() ) # 768 維度

