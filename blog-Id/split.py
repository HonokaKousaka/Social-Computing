import json
import glob

# 步骤2: 找到所有以"tweet_spider_by_keyword"开头的json文件
file = 'tweet_spider_by_keyword_20240619154130.jsonl'

# 用于存储所有comment_count大于0的tweets
filtered_tweets = []

print(f"Processing file: {file}")
with open(file, 'r', encoding='utf-8') as f:
    for line in f:
        tweet = json.loads(line)
        if tweet.get('comments_count') is not None and tweet.get('comments_count') > 2:
            filtered_tweets.append(tweet)

print(f"Filtered tweets count: {len(filtered_tweets)}")

# 步骤4: 将筛选出的tweets写入到一个新的json文件中
with open('filtered_tweets.jsonl', 'w', encoding='utf-8') as f:
    for tweet in filtered_tweets:
        json.dump(tweet, f, ensure_ascii=False)
        f.write('\n')  # 为了保持原始文件的jsonl格式