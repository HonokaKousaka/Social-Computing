import json

all_tweets_id = []
# 步骤 1: 读取 JSON 文件
with open('filtered_tweets_sorted_copy.jsonl', 'r', encoding='utf-8') as file:
    for line in file:
        tweet = json.loads(line)
        all_tweets_id.append(tweet['mblogid'])

# 步骤 3: （可选）将排序后的数据写回到文件
with open('filtered_tweets_sorted_mblogid.jsonl', 'w', encoding='utf-8') as file:
    
    json.dump(all_tweets_id, file, ensure_ascii=False)
