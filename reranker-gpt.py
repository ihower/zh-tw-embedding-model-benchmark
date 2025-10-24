import copy
from tqdm import tqdm
import time
import json


class OpenaiClient:
    def __init__(self, keys=None, start_id=None, proxy=None):
        from openai import OpenAI
        import openai
        if isinstance(keys, str):
            keys = [keys]
        if keys is None:
            raise "Please provide OpenAI Key."

        self.key = keys
        self.key_id = start_id or 0
        self.key_id = self.key_id % len(self.key)
        self.api_key = self.key[self.key_id % len(self.key)]
        self.client = OpenAI(api_key=self.api_key)

    def chat(self, *args, return_text=False, reduce_length=False, **kwargs):
        while True:
            try:
                completion = self.client.chat.completions.create(*args, **kwargs, timeout=30)
                break
            except Exception as e:
                print(str(e))
                if "This model's maximum context length is" in str(e):
                    print('reduce_length')
                    return 'ERROR::reduce_length'
                time.sleep(0.1)
        if return_text:
            completion = completion.choices[0].message.content
        return completion

    def text(self, *args, return_text=False, reduce_length=False, **kwargs):
        while True:
            try:
                completion = self.client.completions.create(
                    *args, **kwargs
                )
                break
            except Exception as e:
                print(e)
                if "This model's maximum context length is" in str(e):
                    print('reduce_length')
                    return 'ERROR::reduce_length'
                time.sleep(0.1)
        if return_text:
            completion = completion.choices[0].text
        return completion


class ClaudeClient:
    def __init__(self, keys):
        from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
        self.anthropic = Anthropic(api_key=keys)

    def chat(self, messages, return_text=True, max_tokens=300, *args, **kwargs):
        system = ' '.join([turn['content'] for turn in messages if turn['role'] == 'system'])
        messages = [turn for turn in messages if turn['role'] != 'system']
        if len(system) == 0:
            system = None
        completion = self.anthropic.beta.messages.create(messages=messages, system=system, max_tokens=max_tokens, *args, **kwargs)
        if return_text:
            completion = completion.content[0].text
        return completion

    def text(self, max_tokens=None, return_text=True, *args, **kwargs):
        completion = self.anthropic.beta.messages.create(max_tokens_to_sample=max_tokens, *args, **kwargs)
        if return_text:
            completion = completion.completion
        return completion


class LitellmClient:
    #  https://github.com/BerriAI/litellm
    def __init__(self, keys=None):
        self.api_key = keys

    def chat(self, return_text=True, *args, **kwargs):
        from litellm import completion
        response = completion(api_key=self.api_key, *args, **kwargs)
        if return_text:
            response = response.choices[0].message.content
        return response


def convert_messages_to_prompt(messages):
    #  convert chat message into a single prompt; used for completion model (eg davinci)
    prompt = ''
    for turn in messages:
        if turn['role'] == 'system':
            prompt += f"{turn['content']}\n\n"
        elif turn['role'] == 'user':
            prompt += f"{turn['content']}\n\n"
        else:  # 'assistant'
            pass
    prompt += "The ranking results of the 20 passages (only identifiers) is:"
    return prompt


def run_retriever(topics, searcher, qrels=None, k=100, qid=None):
    ranks = []
    if isinstance(topics, str):
        hits = searcher.search(topics, k=k)
        ranks.append({'query': topics, 'hits': []})
        rank = 0
        for hit in hits:
            rank += 1
            content = json.loads(searcher.doc(hit.docid).raw())
            if 'title' in content:
                content = 'Title: ' + content['title'] + ' ' + 'Content: ' + content['text']
            else:
                content = content['contents']
            content = ' '.join(content.split())
            ranks[-1]['hits'].append({
                'content': content,
                'qid': qid, 'docid': hit.docid, 'rank': rank, 'score': hit.score})
        return ranks[-1]

    for qid in tqdm(topics):
        if qid in qrels:
            query = topics[qid]['title']
            ranks.append({'query': query, 'hits': []})
            hits = searcher.search(query, k=k)
            rank = 0
            for hit in hits:
                rank += 1
                content = json.loads(searcher.doc(hit.docid).raw())
                if 'title' in content:
                    content = 'Title: ' + content['title'] + ' ' + 'Content: ' + content['text']
                else:
                    content = content['contents']
                content = ' '.join(content.split())
                ranks[-1]['hits'].append({
                    'content': content,
                    'qid': qid, 'docid': hit.docid, 'rank': rank, 'score': hit.score})
    return ranks


def get_prefix_prompt(query, num):
    return [{'role': 'system',
             'content': "You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query."},
            {'role': 'user',
             'content': f"I will provide you with {num} passages, each indicated by number identifier []. \nRank the passages based on their relevance to query: {query}."},
            {'role': 'assistant', 'content': 'Okay, please provide the passages.'}]


def get_post_prompt(query, num):
    # return f"Search Query: {query}. \nRank the {num} passages above based on their relevance to the search query. The passages should be listed in descending order using identifiers. The most relevant passages should be listed first. The output format should be [] > [], e.g., [1] > [2]. Only response the ranking results, do not say any word or explain."
    return f"Search Query: {query}. \nRank the {num} passages above based on their relevance to the search query. The passages should be listed in descending order using identifiers. The most relevant passages should be listed first. The output format should be JSON array. e.g., [1,3,2]. Only response the ranking results, do not say any word or explain."


def create_permutation_instruction(item=None, rank_start=0, rank_end=100, model_name='gpt-4o-mini'):
    query = item['query']
    num = len(item['hits'][rank_start: rank_end])

    max_length = 300

    messages = get_prefix_prompt(query, num)
    rank = 0
    for hit in item['hits'][rank_start: rank_end]:
        rank += 1
        content = hit['content']
        content = content.replace('Title: Content: ', '')
        content = content.strip()
        # For Japanese should cut by character: content = content[:int(max_length)]
        content = ' '.join(content.split()[:int(max_length)])
        messages.append({'role': 'user', 'content': f"[{rank}] {content}"})
        messages.append({'role': 'assistant', 'content': f'Received passage [{rank}].'})
    messages.append({'role': 'user', 'content': get_post_prompt(query, num)})

    return messages


def run_llm(messages, api_key=None, model_name="gpt-4o-mini"):
    if 'gpt' in model_name:
        Client = OpenaiClient
    elif 'claude' in model_name:
        Client = ClaudeClient
    else:
        Client = LitellmClient

    agent = Client(api_key)
    response = agent.chat(model=model_name, messages=messages, temperature=0, return_text=False, response_format={"type": "json_object"},)
    return response


def clean_response(response: str):
    new_response = ''
    for c in response:
        if not c.isdigit():
            new_response += ' '
        else:
            new_response += c
    new_response = new_response.strip()
    return new_response


def remove_duplicate(response):
    new_response = []
    for c in response:
        if c not in new_response:
            new_response.append(c)
    return new_response


def receive_permutation(item, permutation, rank_start=0, rank_end=100):
    response = clean_response(permutation)
    response = [int(x) - 1 for x in response.split()]
    response = remove_duplicate(response)
    cut_range = copy.deepcopy(item['hits'][rank_start: rank_end])
    original_rank = [tt for tt in range(len(cut_range))]
    response = [ss for ss in response if ss in original_rank]
    response = response + [tt for tt in original_rank if tt not in response]
    for j, x in enumerate(response):
        item['hits'][j + rank_start] = copy.deepcopy(cut_range[x])
        if 'rank' in item['hits'][j + rank_start]:
            item['hits'][j + rank_start]['rank'] = cut_range[j]['rank']
        if 'score' in item['hits'][j + rank_start]:
            item['hits'][j + rank_start]['score'] = cut_range[j]['score']
    return item


def permutation_pipeline(item=None, rank_start=0, rank_end=100, model_name='gpt-4o-mini', api_key=None):
    messages = create_permutation_instruction(item=item, rank_start=rank_start, rank_end=rank_end,
                                              model_name=model_name)  # chan
    permutation = run_llm(messages, api_key=api_key, model_name=model_name)
    item = receive_permutation(item, permutation, rank_start=rank_start, rank_end=rank_end)
    return item


def sliding_windows(item=None, rank_start=0, rank_end=100, window_size=20, step=10, model_name='gpt-4o-mini',
                    api_key=None):
    item = copy.deepcopy(item)
    end_pos = rank_end
    start_pos = rank_end - window_size
    while start_pos >= rank_start:
        start_pos = max(start_pos, rank_start)
        item = permutation_pipeline(item, start_pos, end_pos, model_name=model_name, api_key=api_key)
        end_pos = end_pos - step
        start_pos = start_pos - step
    return item


def write_eval_file(rank_results, file):
    with open(file, 'w') as f:
        for i in range(len(rank_results)):
            rank = 1
            hits = rank_results[i]['hits']
            for hit in hits:
                f.write(f"{hit['qid']} Q0 {hit['docid']} {rank} {hit['score']} rank\n")
                rank += 1
    return True

# ----

import os
openai_api_key = os.environ['OPENAI_API_KEY']

## quick example

# item = {
#     'query': 'How much impact do masks have on preventing the spread of the COVID-19?',
#     'hits': [
#         {'content': 'Title: To mask or not to mask: Modeling the potential for face mask use by the general public to curtail the COVID-19 pandemic Content: Face mask use by the general public for limiting the spread of the COVID-19 pandemic is controversial, though increasingly recommended, and the potential of this intervention is not well understood. We develop a compartmental model for assessing the community-wide impact of mask use by the general, asymptomatic public, a portion of which may be asymptomatically infectious. Model simulations, using data relevant to COVID-19 dynamics in the US states of New York and Washington, suggest that broad adoption of even relatively ineffective face masks may meaningfully reduce community transmission of COVID-19 and decrease peak hospitalizations and deaths. Moreover, mask use decreases the effective transmission rate in nearly linear proportion to the product of mask effectiveness (as a fraction of potentially infectious contacts blocked) and coverage rate (as'},
#         {'content': 'Title: Masking the general population might attenuate COVID-19 outbreaks Content: The effect of masking the general population on a COVID-19 epidemic is estimated by computer simulation using two separate state-of-the-art web-based softwares, one of them calibrated for the SARS-CoV-2 virus. The questions addressed are these: 1. Can mask use by the general population limit the spread of SARS-CoV-2 in a country? 2. What types of masks exist, and how elaborate must a mask be to be effective against COVID-19? 3. Does the mask have to be applied early in an epidemic? 4. A brief general discussion of masks and some possible future research questions regarding masks and SARS-CoV-2. Results are as follows: (1) The results indicate that any type of mask, even simple home-made ones, may be effective. Masks use seems to have an effect in lowering new patients even the protective effect of each mask (here dubbed"one-mask protection") is'},
#         {'content': 'Title: Universal Masking is Urgent in the COVID-19 Pandemic: SEIR and Agent Based Models, Empirical Validation, Policy Recommendations Content: We present two models for the COVID-19 pandemic predicting the impact of universal face mask wearing upon the spread of the SARS-CoV-2 virus--one employing a stochastic dynamic network based compartmental SEIR (susceptible-exposed-infectious-recovered) approach, and the other employing individual ABM (agent-based modelling) Monte Carlo simulation--indicating (1) significant impact under (near) universal masking when at least 80% of a population is wearing masks, versus minimal impact when only 50% or less of the population is wearing masks, and (2) significant impact when universal masking is adopted early, by Day 50 of a regional outbreak, versus minimal impact when universal masking is adopted late. These effects hold even at the lower filtering rates of homemade masks. To validate these theoretical models, we compare their predictions against a new empirical data set we have collected'},
#     ]
# }
#
# rank_start = 0
# rank_end = 20
# model_name = 'gpt-4o-mini'
#
# messages = create_permutation_instruction(item=item, rank_start=rank_start, rank_end=rank_end, model_name=model_name)
# print(messages)
# permutation = run_llm(messages, api_key=openai_api_key, model_name=model_name)
#
# print(permutation)


## ---

from supabase import create_client, Client

model_name = "text-embedding-3-small"

print( model_name )

supabase_url = 'https://imcpayinnpcetclzvdfu.supabase.co'
supabase_api_key = ''

supabase: Client = create_client(supabase_url, supabase_api_key)

response1 = supabase.table('questions').select("content, embedding, paragraph_id").eq("model", model_name).execute()

question_contents = [x["content"] for x in response1.data]
question_embeddings = [ eval(x["embedding"]) for x in response1.data]

gold_paragraph_ids = [x["paragraph_id"] for x in response1.data]

response2 = supabase.table('paragraphs').select("id, content, embedding").eq("model", model_name).execute()

paragraph_ids = [x["id"] for x in response2.data]

paragraph_embeddings = [ eval(x["embedding"]) for x in response2.data]
paragraph_contents = [x["content"] for x in response2.data]


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

def normalize_ranking(data):

    # 檢查 data 是否是字典並包含 'ranking' key
    if isinstance(data, dict) and 'ranking' in data:
        arr = data['ranking']
    elif isinstance(data, dict) and 'result' in data:
        arr = data['result']
    elif isinstance(data, list):
        arr = data
    elif data == {}:
        arr = [] # 暫時就當作沒結果
    else:
        raise ValueError("Input data is neither a dictionary with 'ranking' key nor a list")

    # 因為 LLM 回傳的 number 從 1 開始排名
    return [ i-1 for i in arr]

hit_data = []
mmr_data = []

import time
import pdb

# gpt-4o-mini 成本估計
# (5,000 × 3,493 / 1,000,000 )× 0.15
# (23 × 3,493 / 1,000,000) × 0.6
# 大約 2.66 美金 (top-k=10), 速度大約是 1秒1題
# 大約 13.3 美金 (top-k=50)
# 大約 26.6 美金 (top-k=100)
# 加起來的成本感覺跟 Voyage API 同等級?
# 不對啊，Voyage 1M tokens 是 $0.05 便宜三倍耶

for idx, question_embedding in enumerate(question_embeddings):

  print(idx)

  best_indexes = get_top_k_indices(paragraph_embeddings, question_embedding, 50) # 取出 top_k 的 indexes
  # ----- reranker 後取 top 5

  # reranker
  rerank_docs = [ { 'content': paragraph_contents[i] } for i in best_indexes]

  try:

    item = {
      'query': question_contents[idx],
       'hits': rerank_docs
    }

    messages = create_permutation_instruction(item=item, rank_start=0, rank_end=50, model_name='gpt-4o-mini')

    #pdb.set_trace()

    raw_results = run_llm(messages, api_key=openai_api_key, model_name='gpt-4o-mini')

    print(raw_results.usage)

    raw_content = raw_results.choices[0].message.content

    print(raw_content)

    results = json.loads(raw_content)
    results = normalize_ranking(results)

  except Exception as e:
    print("api error.....")
    time.sleep(3)
    # retry
    # TODO: 得增加溫度
    raw_results = run_llm(messages, api_key=openai_api_key, model_name='gpt-4o-mini')
    raw_content = raw_results.choices[0].message.content
    results = json.loads(raw_content)
    results = normalize_ranking(results)

  # print(raw_results)
  rerank_indexes = results[0:5]

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




