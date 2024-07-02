import json

all_tweets = []
# 步骤 1: 读取 JSON 文件
with open('filtered_tweets.jsonl', 'r', encoding='utf-8') as file:
    for line in file:
        tweet = json.loads(line)
        all_tweets.append(tweet)

# 步骤 2: 按 comments_count 降序排序
tweets_sorted = sorted(all_tweets, key=lambda x: x.get('comments_count', 0), reverse=True)

# 步骤 3: （可选）将排序后的数据写回到文件
with open('filtered_tweets_sorted.jsonl', 'w', encoding='utf-8') as file:
    for tweet in tweets_sorted:
        json.dump(tweet, file, ensure_ascii=False)
        file.write('\n')