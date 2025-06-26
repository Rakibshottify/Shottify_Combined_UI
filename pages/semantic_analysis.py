import streamlit as st
import praw
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from scipy.special import softmax # For calculating probabilities

# --- Streamlit UI Configuration ---
st.set_page_config(
    page_title="Reddit Post Search & Sentiment (Local Transformer)",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Reddit Post Search & Sentiment Analysis (Local Transformer Model)")
st.markdown("Search for posts by keyword within a subreddit, or view Hot and Rising topics from a specified subreddit (or r/all), with sentiment scores using a locally loaded transformer model.")

# --- PRAW Authentication (using Streamlit Secrets for security) ---
try:
    reddit = praw.Reddit(
        client_id=st.secrets["client_id"],
        client_secret=st.secrets["client_secret"],
        user_agent="reddit_streamlit_local_transformer_sentiment_app_by_aijob876@gmail.com"
    )
except KeyError:
    st.error("Reddit API credentials not found in Streamlit Secrets. "
             "Please add `client_id` and `client_secret` to your `.streamlit/secrets.toml` file.")
    st.stop()

# --- Model Loading and Sentiment Functions ---
# Define your local model path
# IMPORTANT: Make sure this path is correct and accessible by your Streamlit app
# when it's run, especially if deploying.

MODEL_PATH = r"E:\SM Monitoring\models\twitter-roberta-base-sentiment-latest"

@st.cache_resource
def load_sentiment_model_and_tokenizer():
    st.spinner("Loading sentiment analysis model from local directory... This may take a moment.")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model from {MODEL_PATH}. Please ensure the path is correct and the model files are present. Error: {e}")
        st.stop()

tokenizer, model = load_sentiment_model_and_tokenizer()

# --- Preprocessing function (from your snippet) ---
def preprocess(text):
    new = []  # Initialize an empty list to store processed words
    for t in text.split(): # Split the input text into individual words/tokens
        t = '@user' if t.startswith('@') and len(t) > 1 else t  # Handle user mentions
        t = 'http' if t.startswith('http') else t            # Handle URLs
        new.append(t) # Add the (potentially modified) word to the new list
    return " ".join(new) # Join the processed words back into a single string

# --- Get sentiment function (adapted from your snippet) ---
# This function will now use the globally cached tokenizer and model
def get_sentiment(text):
    text = preprocess(text)
    # Ensure text is not empty to avoid tokenizer errors
    if not text.strip():
        return 'Neutral', 0.0 # Default for empty or just whitespace text

    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True) # Added padding=True
    
    # Move tokens to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokens = {k: v.to(device) for k, v in tokens.items()}
    model.to(device) # Move model to GPU

    with torch.no_grad():
        outputs = model(**tokens)
    
    scores = softmax(outputs.logits.cpu().numpy()[0]) # Move logits back to CPU for numpy operation
    
    # These labels correspond to 'cardiffnlp/twitter-roberta-base-sentiment-latest'
    # LABEL_0: Negative, LABEL_1: Neutral, LABEL_2: Positive
    labels = ['Negative', 'Neutral', 'Positive']
    
    top_label_index = scores.argmax()
    top_label = labels[top_label_index]
    confidence = float(scores[top_label_index])
    
    return top_label, confidence

# --- Function to fetch posts and calculate sentiment ---
@st.cache_data(ttl=600) # Cache data for 10 minutes
def get_reddit_posts(subreddit_name, query, post_type, limit):
    posts_data = []
    
    subreddit_name = subreddit_name.strip()
    target_subreddit = reddit.subreddit(subreddit_name) if subreddit_name else reddit.subreddit('all')

    fetch_limit = limit * 2 # Fetch more posts to filter out NSFW ones and ensure we hit the desired 'limit' for SFW posts
    
    try:
        submissions = []

        if query:
            st.info(f"Searching for '{query}' {'in r/' + subreddit_name if subreddit_name else 'across all of Reddit'}. "
                    "Note: 'Rising' sort for keyword searches is not directly supported by Reddit API; showing 'new' instead.")
            
            if post_type == "hot":
                submissions = target_subreddit.search(query=query, sort='hot', limit=fetch_limit)
            elif post_type == "rising":
                submissions = target_subreddit.search(query=query, sort='new', limit=fetch_limit) # Fallback for 'rising' with keyword
        else:
            st.info(f"No keyword provided. Displaying general {post_type} posts from r/{target_subreddit.display_name}.")
            if post_type == "hot":
                submissions = target_subreddit.hot(limit=fetch_limit)
            elif post_type == "rising":
                submissions = target_subreddit.rising(limit=fetch_limit)
            
        fetched_count = 0
        for post in submissions:
            if fetched_count >= limit:
                break

            if post.over_18:
                continue

            # --- Sentiment analysis using the local model ---
            sentiment, confidence = get_sentiment(post.title)
            
            image_url = None
            if hasattr(post, 'url') and (post.url.lower().endswith(('.jpg', '.png', '.gif', '.jpeg', '.webp'))):
                image_url = post.url
            elif hasattr(post, 'thumbnail') and post.thumbnail not in ['self', 'default', 'nsfw', 'spoiler']:
                image_url = post.thumbnail
            elif hasattr(post, 'preview') and 'images' in post.preview and len(post.preview['images']) > 0:
                image_url = post.preview['images'][0]['source']['url']
            
            posts_data.append({
                "Title": post.title,
                "Subreddit": post.subreddit.display_name,
                "Score": post.score,
                "Comments": post.num_comments,
                "Sentiment": f"{sentiment} ({confidence:.2f})", # Display sentiment and confidence
                "URL": post.url,
                "Permalink": f"https://www.reddit.com{post.permalink}",
                "Image": image_url
            })
            fetched_count += 1

    except Exception as e:
        st.error(f"Error fetching {post_type} posts: {e}")
        st.error("This might be due to Reddit API rate limits, an invalid subreddit name, or an invalid query.")
    return posts_data

# --- Sidebar for Keyword and Subreddit selection ---
st.sidebar.header("Search & Subreddit Settings")
input_subreddit = st.sidebar.text_input("Enter Subreddit (e.g., 'programming', 'science', leave empty for 'all')", "").strip()
search_keyword = st.sidebar.text_input("Enter Keyword (optional)", "").strip()
post_limit = st.sidebar.slider("Number of Posts to Display", 5, 50, 10)

# Determine what to display in the header based on inputs
display_location_text = ""
if input_subreddit:
    display_location_text = f"in r/{input_subreddit}"
else:
    display_location_text = "from r/all"

if search_keyword:
    display_keyword_text = f" with keyword: '{search_keyword}'"
else:
    display_keyword_text = ""

# --- Display Posts ---

tab1, tab2 = st.tabs(["🔥 Hot Topics", "⬆️ Rising Topics"])

with tab1:
    st.header(f"Hot Topics {display_location_text}{display_keyword_text}")
    hot_posts_data = get_reddit_posts(input_subreddit, search_keyword, "hot", post_limit)
    
    if hot_posts_data:
        df_hot = pd.DataFrame(hot_posts_data)
        
        for index, row in df_hot.iterrows():
            st.markdown(f"### [{row['Title']}]({row['URL']})")
            st.write(f"**Subreddit:** r/{row['Subreddit']} | **Score:** {row['Score']} | **Comments:** {row['Comments']} | **Sentiment:** **{row['Sentiment']}**")
            if row['Image']:
                st.image(row['Image'], width=300)
            st.markdown(f"[View on Reddit]({row['Permalink']})")
            st.divider()

    else:
        st.info("No SFW hot posts found based on your criteria or an error occurred. Try different inputs or increase the post limit.")

with tab2:
    st.header(f"Rising Topics {display_location_text}{display_keyword_text}")
    rising_posts_data = get_reddit_posts(input_subreddit, search_keyword, "rising", post_limit)

    if rising_posts_data:
        df_rising = pd.DataFrame(rising_posts_data)
        
        for index, row in df_rising.iterrows():
            st.markdown(f"### [{row['Title']}]({row['URL']})")
            st.write(f"**Subreddit:** r/{row['Subreddit']} | **Score:** {row['Score']} | **Comments:** {row['Comments']} | **Sentiment:** **{row['Sentiment']}**")
            if row['Image']:
                st.image(row['Image'], width=300)
            st.markdown(f"[View on Reddit]({row['Permalink']})")
            st.divider()
    else:
        st.info("No SFW rising posts found based on your criteria or an error occurred. Try different inputs or increase the post limit.")

st.markdown("---")
st.caption("Powered by PRAW, Streamlit, and Hugging Face Transformers (Local).")