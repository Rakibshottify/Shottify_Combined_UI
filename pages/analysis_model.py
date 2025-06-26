# Coordinated Amplification Detection using Groq LLM API

# --- Setup ---
from groq import Groq
import pandas as pd
import numpy as np
import networkx as nx
import community as community_louvain  # pip install python-louvain
from datetime import datetime
import re

# API Key
groq_api_key = "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"  # <- Replace this
client = Groq(api_key=groq_api_key)

# --- Load Data ---
CSV_FILENAME = "simulated_social_media_data.csv"
df = pd.read_csv(CSV_FILENAME)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['normalized_text'] = df['text'].str.lower().replace(r'[^\w\s]', '', regex=True).str.strip()

# --- 1. Batched Near-Duplicate Detection ---

# Sample 10 unique normalized texts
texts = df['normalized_text'].drop_duplicates().tolist()[:10]

# Generate unique text pairs (batched)
text_pairs = [(texts[i], texts[j]) for i in range(len(texts)) for j in range(i + 1, len(texts))]

def batch_duplicate_prompt(pairs):
    examples = "\n\n".join([f"Post 1: {a}\nPost 2: {b}" for a, b in pairs])
    return [
        {"role": "system", "content": "You are a social media analyst detecting coordinated bot campaigns."},
        {"role": "user", "content": f"""For the following post pairs, respond Yes/No if they are near-duplicates, with 1-line justification.

{examples}"""}
    ]

batch = text_pairs[:5]  # You can increase this up to 10 if needed
response = client.chat.completions.create(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    messages=batch_duplicate_prompt(batch)
)

print("=== Near-Duplicate Detection (Batched) ===")
print(response.choices[0].message.content)


# --- 2. Suspicious Account Evaluation ---

df['follower_following_ratio'] = df['user_followers'] / (df['user_following'] + 1e-6)

user_features = df.groupby('user_id').agg(
    total_posts=('post_id', 'count'),
    avg_likes_per_post=('likes', 'mean'),
    avg_account_age_days=('account_age_days', 'mean'),
    avg_follower_following_ratio=('follower_following_ratio', 'mean'),
    unique_hashtags_used=('hashtags', lambda x: len(set(' '.join(x.dropna()).split()))),
    first_post_time=('timestamp', 'min'),
    last_post_time=('timestamp', 'max')
).reset_index()
user_features['activity_span_days'] = (user_features['last_post_time'] - user_features['first_post_time']).dt.days

def summarize_user(row):
    return f"""User ID: {row['user_id']}
Posts: {row['total_posts']}
Likes/post: {row['avg_likes_per_post']:.2f}
Followers/Following: {row['avg_follower_following_ratio']:.2f}
Account Age: {row['avg_account_age_days']:.1f} days
Activity Span: {row['activity_span_days']} days
Hashtags used: {row['unique_hashtags_used']}"""

print("\n=== Suspicious Account Evaluation ===")
for _, row in user_features.head(3).iterrows():
    summary = summarize_user(row)
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {"role": "system", "content": "You are reviewing a user for suspicious behavior."},
            {"role": "user", "content": f"Is this account suspicious?\n\n{summary}"}
        ]
    )
    print(response.choices[0].message.content)
    print("---")


# --- 3. Community Detection & Explanation ---

df['text_hash'] = df['normalized_text'].apply(hash)
common_texts = df['text_hash'].value_counts()
coordinated_texts = common_texts[common_texts >= 3].index
df_coord = df[df['text_hash'].isin(coordinated_texts)]

# Build user-user graph via co-posting
G = nx.Graph()
for text_hash in coordinated_texts:
    users = df_coord[df_coord['text_hash'] == text_hash]['user_id'].unique()
    for i in range(len(users)):
        for j in range(i+1, len(users)):
            G.add_edge(users[i], users[j])

partition = community_louvain.best_partition(G)
user_features['community'] = user_features['user_id'].map(partition)

# Analyze a community
first_comm_id = list(partition.values())[0]
community_df = user_features[user_features['community'] == first_comm_id]

summary_text = f"""Community ID: {first_comm_id}
Users: {len(community_df)}
Mean Posts: {community_df['total_posts'].mean():.1f}
Avg Account Age: {community_df['avg_account_age_days'].mean():.1f} days
Suspicious indicators: check for coordinated behavior."""

response = client.chat.completions.create(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    messages=[
        {"role": "system", "content": "You are an analyst reviewing coordinated online communities."},
        {"role": "user", "content": f"What might be happening in this community?\n\n{summary_text}"}
    ]
)
print("\n=== Community Explanation ===")
print(response.choices[0].message.content)
