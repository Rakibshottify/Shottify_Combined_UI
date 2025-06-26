import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime, timedelta, timezone
import pandas as pd
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
import plotly.express as px
from collections import Counter

# Load environment variables from a .env file
load_dotenv()

# --- SESSION STATE INITIALIZATION ---
# Initialize session state variables to store data across reruns
if 'corrections' not in st.session_state:
    st.session_state.corrections = []
if 'current_analysis' not in st.session_state:
    st.session_state.current_analysis = None
if 'original_analysis' not in st.session_state:
    st.session_state.original_analysis = None
if 'articles' not in st.session_state:
    st.session_state.articles = []
if 'correction_messages' not in st.session_state:
    st.session_state.correction_messages = []
if 'credibility_assessment' not in st.session_state:
    st.session_state.credibility_assessment = None
if 'similar_articles' not in st.session_state:
    st.session_state.similar_articles = []

# --- API CLIENT AND DATA FETCHING FUNCTIONS ---

def post_analysis_to_directus(analysis_data):
    """
    Posts the complete analysis results to the Directus 'news_analyses' collection.
    """
    directus_url = os.getenv("DIRECTUS_API_URL")
    token = os.getenv("DIRECTUS_TOKEN")
    if not directus_url or not token:
        st.error("Directus API URL or Token is missing from your environment variables.")
        return
    api_endpoint = f"{directus_url}/items/news_analyses"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    try:
        response = requests.post(api_endpoint, headers=headers, json=analysis_data)
        response.raise_for_status()
        st.success("Analysis was successfully posted to Directus!")
    except requests.exceptions.HTTPError as e:
        st.error(f"Failed to post data to Directus. Status code: {e.response.status_code}")
        st.error(f"Error details: {e.response.text}")
    except Exception as e:
        st.error(f"An unexpected error occurred when trying to post to Directus: {e}")

@st.cache_data(ttl=600) # Cache data for 10 minutes
def fetch_all_analyses():
    """Fetches all items from the 'news_analyses' collection in Directus."""
    directus_url = os.getenv("DIRECTUS_API_URL")
    token = os.getenv("DIRECTUS_TOKEN")
    if not directus_url or not token:
        st.error("Directus API URL or Token is missing from your environment variables.")
        return None

    api_endpoint = f"{directus_url}/items/news_analyses?limit=-1&fields=*,date_created"
    headers = {"Authorization": f"Bearer {token}"}

    try:
        response = requests.get(api_endpoint, headers=headers)
        response.raise_for_status()
        data = response.json().get('data', [])
        return data
    except Exception as e:
        st.error(f"Failed to fetch data from Directus: {e}")
        return None

def get_youtube_captions(video_url):
    """Fetches the full English captions from a YouTube video."""
    try:
        video_id = video_url.split("v=")[1].split("&")[0]
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        full_transcript = ' '.join([entry['text'] for entry in transcript_list])
        return {'title': f"YouTube Video: {video_id}", 'content': full_transcript, 'url': video_url}
    except Exception as e:
        return {'error': f"Could not retrieve YouTube captions: {str(e)}"}

def get_groq_client():
    """Initializes and returns the Groq client."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Missing GROQ_API_KEY in environment.")
    return Groq(api_key=api_key)

def fact_check_with_google(query_text):
    """Searches for fact checks using the Google Fact Check Tools API."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return {'error': "Missing GOOGLE_API_KEY in environment."}
    try:
        service = build("factchecktools", "v1alpha1", developerKey=api_key)
        request = service.claims().search(query=query_text, languageCode="en")
        response = request.execute()
        return response
    except Exception as e:
        return {'error': f"Error calling Google Fact Check API: {str(e)}"}

def search_google_news(query_text):
    """Searches Google for news articles using the Custom Search API."""
    api_key = os.getenv("GOOGLE_API_KEY")
    search_engine_id = os.getenv("GOOGLE_CSE_ID")
    if not api_key or not search_engine_id:
        return {'error': "Missing GOOGLE_API_KEY or GOOGLE_CSE_ID in environment."}
    try:
        service = build("customsearch", "v1", developerKey=api_key)
        res = service.cse().list(q=query_text, cx=search_engine_id, num=10).execute()
        return res.get('items', [])
    except Exception as e:
        return {'error': f"Error calling Google Custom Search API: {str(e)}"}

def scrape_webpage(url):
    """Scrapes the title and text content from a given URL."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.find('title').get_text().strip() if soup.find('title') else "No Title Found"
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        text = ' '.join(chunk.strip() for line in soup.get_text().splitlines() for chunk in line.split("  ") if chunk.strip())
        return {'title': title, 'content': text[:8000], 'url': url}
    except Exception as e:
        return {'error': f"Error scraping webpage: {str(e)}"}

@st.cache_data(show_spinner=False)
def load_and_fetch_headlines(uploaded_file):
    """Reads a CSV, extracts URLs, and scrapes their headlines."""
    if uploaded_file is None:
        return []
    try:
        df = pd.read_csv(uploaded_file)
        url_column = next((col for col in ['url', 'urls', 'link', 'links', 'href'] if col in df.columns), None)
        if not url_column:
            st.error("Error: CSV must contain a column named 'url', 'link', 'urls', 'links', or 'href'.")
            return []
        articles = []
        urls_to_process = df[url_column].dropna().unique()
        progress_bar = st.progress(0, "Fetching headlines...")
        for i, url in enumerate(urls_to_process):
            if isinstance(url, str) and url.startswith(('http://', 'https://')):
                try:
                    headers = {'User-Agent': 'Mozilla/5.0'}
                    response = requests.get(url, headers=headers, timeout=10)
                    response.raise_for_status()
                    soup = BeautifulSoup(response.content, 'html.parser')
                    title = soup.find('title').get_text().strip() if soup.find('title') else f"Untitled Article (URL: {url[:30]}...)"
                    articles.append({'headline': title, 'url': url})
                except Exception as e:
                    st.warning(f"Skipping URL (could not fetch title): {url} - Error: {e}")
            progress_value = (i + 1) / len(urls_to_process)
            progress_bar.progress(progress_value, f"Fetching headlines... {i+1}/{len(urls_to_process)}")
        progress_bar.empty()
        st.success(f"Successfully fetched titles from {len(articles)} articles!")
        return articles
    except Exception as e:
        st.error(f"Failed to process CSV file: {e}")
        return []

# --- AI AND PROMPT FUNCTIONS ---

def build_feedback_prompt(content_data):
    base_prompt = """
    Analyze the following news content and provide a comprehensive classification. Return your response as a JSON object with the following structure.
    For the "confidence_scores", you MUST provide your own estimated confidence level for each classification on a scale from 0.0 (not confident) to 1.0 (very confident). Do not use example values; generate your own assessment.
    {
        "headline": "extracted headline", "summary": "brief summary of the article", "topic": "one of: Politics, Economy, Technology, Health, Sports, Entertainment, Science, Environment, Crime, International, Education, Business, Social, Other",
        "type": "one of: Fact, Fake, Distorted, Opinion", "sentiment": "one of: Positive, Negative, Neutral", "urgency": "one of: Red, Orange, Yellow",
        "confidence_scores": { "topic": <float>, "type": <float>, "sentiment": <float>, "urgency": <float> },
        "key_points": ["point 1", "point 2"], "sources_mentioned": ["source 1", "source 2"],
        "reasoning": { "topic": "explanation", "type": "explanation", "sentiment": "explanation", "urgency": "explanation" }
    }
    """
    if st.session_state.corrections:
        feedback_section = "\n\nIMPORTANT: Learn from these previous corrections to improve accuracy:\n"
        for correction in st.session_state.corrections[-5:]:
            feedback_section += f"- {correction['field']}: AI classified as '{correction['original']}' but correct answer was '{correction['corrected']}'. Reason: {correction['reason']}\n"
        base_prompt += feedback_section
    
    base_prompt += f"\nTitle: {content_data.get('title', 'No title')}\nContent: {content_data.get('content', 'No content')}"
    return base_prompt

def classify_news_content(content_data):
    try:
        groq_client = get_groq_client()
        classification_prompt = build_feedback_prompt(content_data)
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are an expert news analyst and fact-checker. Provide accurate, objective analysis of news content. Always return valid JSON format. Learn from user feedback to improve accuracy."},
                {"role": "user", "content": classification_prompt}
            ],
            max_tokens=2000, temperature=0.1
        )
        if response.choices:
            result = response.choices[0].message.content
            try:
                json_start = result.find('{')
                json_end = result.rfind('}') + 1
                if json_start != -1 and json_end != -1:
                    return json.loads(result[json_start:json_end])
                else:
                    return {'error': 'Could not extract JSON from response', 'raw_response': result}
            except json.JSONDecodeError:
                return {'error': 'Invalid JSON in response', 'raw_response': result}
        else:
            return {'error': 'No response from AI'}
    except Exception as e:
        return {'error': f"Error in classification: {str(e)}"}

def generate_credibility_summary(original_summary, similar_articles):
    groq_client = get_groq_client()
    formatted_articles = ""
    for i, article in enumerate(similar_articles):
        formatted_articles += f"{i+1}. Title: {article.get('title', 'N/A')}\n"
        formatted_articles += f"   Source: {article.get('displayLink', 'N/A')}\n"
        formatted_articles += f"   Snippet: {article.get('snippet', 'N/A')}\n\n"
    prompt = f"""
    You are a meticulous and impartial news analyst. Your task is to provide a structured credibility and bias assessment of a primary news article by comparing it against a list of other articles found online.

    Here is the summary of the primary news article:
    "{original_summary}"

    Here is a list of other articles found online that may be related:
    {formatted_articles}

    Analyze the provided articles and return a JSON object with the following structure. Do not add any text or explanation outside of the JSON object. When assessing bias, consider the use of loaded language, emotional appeals, selective information, or a consistent political or ideological slant.

    {{
        "overall_credibility": "One of: High, Medium, Low, Uncertain",
        "bias_assessment": {{
            "rating": "One of: Low, Medium, High, Uncertain",
            "reasoning": "A brief, one-sentence explanation for the bias rating."
        }},
        "confidence_score": <float, your confidence in this assessment from 0.0 to 1.0>,
        "key_findings": [
            "A point about whether the main facts are corroborated by other sources.",
            "A point about the quality and diversity of the sources found (e.g., reputable news agencies, blogs, etc.).",
            "A point about any significant contradictions or lack of coverage found.",
            "Any other relevant observation, such as the general tone across sources."
        ],
        "summary": "A final, one-sentence concluding thought on the story's reliability and objectivity."
    }}

    Generate the JSON response based on your analysis.
    """
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a professional news analyst. Your task is to assess the credibility and potential bias of a primary news article by comparing it against a list of other articles. You must respond in the requested JSON format."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500, temperature=0.1, response_format={"type": "json_object"}
        )
        if response.choices:
            result_text = response.choices[0].message.content
            return json.loads(result_text)
        return {"error": "Could not generate a credibility summary."}
    except json.JSONDecodeError as e:
        return {'error': f"Failed to parse JSON response from AI. Error: {e}"}
    except Exception as e:
        return {'error': f"An error occurred while generating the summary: {e}"}

# --- UI DISPLAY AND INTERACTION FUNCTIONS ---

def handle_correction(field, original_value, corrected_value, reason):
    correction = {
        'timestamp': datetime.now().isoformat(), 'field': field, 'original': original_value,
        'corrected': corrected_value, 'reason': reason, 'url': st.session_state.current_analysis.get('url', 'Unknown')
    }
    st.session_state.corrections.append(correction)
    if st.session_state.current_analysis and field in st.session_state.current_analysis:
        st.session_state.current_analysis[field] = corrected_value

def display_correction_interface(classification):
    st.markdown("### 🔧 Manual Corrections")
    st.write("Help improve AI accuracy by correcting any misclassifications:")
    for message in st.session_state.correction_messages:
        st.success(message)
    st.session_state.correction_messages = []
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Current Classifications:**")
        st.write(f"📂 Topic: **{classification.get('topic', 'N/A')}**")
        st.write(f"🔍 Type: **{classification.get('type', 'N/A')}**")
        st.write(f"😊 Sentiment: **{classification.get('sentiment', 'N/A')}**")
        st.write(f"⏰ Urgency: **{classification.get('urgency', 'N/A')}**")
    with col2:
        with st.form(f"corrections_form_{hash(str(classification))}", clear_on_submit=True):
            new_topic = st.selectbox("Correct Topic:", ["No change"] + ["Politics", "Economy", "Technology", "Health", "Sports", "Entertainment", "Science", "Environment", "Crime", "International", "Education", "Business", "Social", "Other"])
            new_type = st.selectbox("Correct Type:", ["No change"] + ["Fact", "Fake", "Distorted", "Opinion"])
            new_sentiment = st.selectbox("Correct Sentiment:", ["No change"] + ["Positive", "Negative", "Neutral"])
            new_urgency = st.selectbox("Correct Urgency:", ["No change"] + ["Red", "Orange", "Yellow"])
            correction_reason = st.text_area("Reason for correction (optional):", placeholder="Explain why the AI was wrong...")
            if st.form_submit_button("💾 Submit Corrections", type="secondary"):
                corrections_made = []
                if new_topic != "No change": handle_correction("topic", classification.get('topic'), new_topic, correction_reason); corrections_made.append("Topic")
                if new_type != "No change": handle_correction("type", classification.get('type'), new_type, correction_reason); corrections_made.append("Type")
                if new_sentiment != "No change": handle_correction("sentiment", classification.get('sentiment'), new_sentiment, correction_reason); corrections_made.append("Sentiment")
                if new_urgency != "No change": handle_correction("urgency", classification.get('urgency'), new_urgency, correction_reason); corrections_made.append("Urgency")
                st.session_state.correction_messages.append(f"✅ Corrections saved for: {', '.join(corrections_made)}" if corrections_made else "ℹ️ No corrections were made.")
                st.rerun()

def display_feedback_stats():
    if st.session_state.corrections:
        st.markdown("### 📊 Learning Progress")
        st.metric("Total Corrections Made", len(st.session_state.corrections))
        correction_fields = {c['field']: st.session_state.corrections.count(c) for c in st.session_state.corrections}
        if correction_fields:
            st.write("**Corrections by Category:**")
            for field, count in correction_fields.items():
                st.write(f"• {field.title()}: {count} corrections")

def display_classification_results(classification):
    """Displays the full analysis, verification tools, and correction interface."""
    if 'error' in classification:
        st.error(f"Classification Error: {classification['error']}")
        if 'raw_response' in classification:
            st.text_area("Raw AI Response:", classification['raw_response'], height=200)
        return

    st.session_state.current_analysis = classification.copy()
    if not st.session_state.original_analysis:
        st.session_state.original_analysis = classification.copy()

    urgency = classification.get('urgency', 'Yellow')
    urgency_colors = {'Red': '🔴', 'Orange': '🟠', 'Yellow': '🟡'}
    headline_text = classification.get('headline', 'No headline')
    st.subheader(f"📰 {headline_text}")
    st.markdown(f"**Urgency:** {urgency_colors.get(urgency, '🟡')} **{urgency}**")
    st.write("**Summary:**")
    st.write(classification.get('summary', 'No summary available'))
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("📂 Topic", classification.get('topic', 'N/A'), f"{classification.get('confidence_scores', {}).get('topic', 0.0):.0%} confidence")
    with col2: st.metric("🔍 Type", classification.get('type', 'N/A'), f"{classification.get('confidence_scores', {}).get('type', 0.0):.0%} confidence")
    with col3: st.metric("😊 Sentiment", classification.get('sentiment', 'N/A'), f"{classification.get('confidence_scores', {}).get('sentiment', 0.0):.0%} confidence")
    with col4: st.metric("⏰ Urgency", urgency, f"{classification.get('confidence_scores', {}).get('urgency', 0.0):.0%} confidence")
    if classification.get('key_points'):
        st.write("**🔍 Key Points:**")
        for point in classification['key_points']: st.write(f"• {point}")
    if classification.get('sources_mentioned'):
        st.write("**📚 Sources Mentioned:**")
        for source in classification['sources_mentioned']: st.write(f"• {source}")

    st.markdown("---")
    st.subheader("🕵️‍♀️ Verification Tools")
    fact_check_tab, cross_reference_tab = st.tabs(["Google Fact-Check", "Cross-Reference Search"])

    with fact_check_tab:
        st.write("Search Google's database for published fact-checks related to the article's summary.")
        if st.button("Search for Fact-Checks", type="secondary"):
            if summary := classification.get("summary"):
                with st.spinner("Searching for published fact-checks..."):
                    results = fact_check_with_google(summary)
                    if 'error' in results: st.error(results['error'])
                    elif 'claims' in results and results['claims']:
                        st.success(f"Found {len(results['claims'])} published fact-check(s):")
                        for claim in results['claims']:
                            review = claim['claimReview'][0]
                            with st.container(border=True):
                                st.write(f"**Claim:** {claim.get('text', 'N/A')}")
                                st.write(f"**Publisher:** {review.get('publisher', {}).get('name', 'N/A')}")
                                st.write(f"**Rating:** {review.get('textualRating', 'N/A')}")
                                st.link_button("Read Fact-Check", review.get('url', '#'))
                    else: st.info("No published fact-checks were found matching the summary.")
            else: st.warning("No summary was generated to fact-check.")

    with cross_reference_tab:
        st.write("Verify the story by finding similar articles from other news portals using the headline.")
        
        if st.button("Find Similar News Articles", type="secondary"):
            st.session_state.credibility_assessment = None
            st.session_state.similar_articles = [] 

            if headline_text != 'No headline' and (original_summary := classification.get("summary")):
                with st.spinner(f"Searching for articles with keywords from: '{headline_text}'..."):
                    results = search_google_news(headline_text)
                
                if 'error' in results:
                    st.error(results['error'])
                elif results:
                    similar_articles = [res for res in results if res.get('link') != classification.get('url')]
                    st.session_state.similar_articles = similar_articles

                    if similar_articles:
                        with st.spinner("Analyzing cross-referenced articles to assess credibility..."):
                            assessment = generate_credibility_summary(original_summary, similar_articles)
                            if 'error' not in assessment:
                                st.session_state.credibility_assessment = assessment
                            else:
                                st.error(assessment['error'])
                    else:
                        st.info("The search returned results, but they all appear to be the original source.")
                else:
                    st.info("No other similar news articles were found.")
            else:
                st.warning("A headline and summary are needed to perform a cross-reference search.")
            
            st.rerun()

        if st.session_state.get('similar_articles'):
            st.subheader(f"Found {len(st.session_state.similar_articles)} Similar Articles:")
            for article in st.session_state.similar_articles:
                with st.container(border=True):
                    st.markdown(f"**[{article.get('title', 'No Title')}]({article.get('link', '#')})**")
                    st.write(article.get('snippet', 'No snippet available...'))
                    st.caption(f"Source: {article.get('displayLink', 'N/A')}")

        if st.session_state.credibility_assessment:
            st.markdown("---")
            st.subheader("🤖 AI Credibility Assessment")
            assessment = st.session_state.credibility_assessment
            verdict = assessment.get("overall_credibility", "Uncertain")
            confidence = assessment.get("confidence_score", 0.0)
            findings = assessment.get("key_findings", [])
            conclusion = assessment.get("summary", "No summary provided.")
            
            color_map = {"High": "green", "Medium": "orange", "Low": "red", "Uncertain": "grey"}
            verdict_color = color_map.get(verdict, "grey")

            st.markdown(f"#### Credibility Verdict: <span style='color:{verdict_color}; font-weight:bold;'>{verdict}</span>", unsafe_allow_html=True)
            st.progress(float(confidence), text=f"AI Confidence in this Assessment: {confidence:.0%}")
            
            st.markdown("##### Key Findings:")
            for finding in findings:
                st.markdown(f"- {finding}")
                
            if conclusion:
                st.markdown("##### Conclusion")
                st.write(conclusion)

            st.markdown("---")

            if st.button("🚀 Post Full Analysis to Directus", type="primary"):
                final_analysis_data = st.session_state.current_analysis.copy()
                final_analysis_data.update({
                    "overall_credibility": assessment.get("overall_credibility"),
                    "bias_assessment": assessment.get("bias_assessment"),
                    "credibility_confidence": assessment.get("confidence_score"),
                    "key_findings": assessment.get("key_findings"),
                    "credibility_summary": assessment.get("summary")
                })
                with st.spinner("Sending analysis to Directus..."):
                    post_analysis_to_directus(final_analysis_data)

    st.markdown("---")
    display_correction_interface(classification)


def display_live_analysis_page():
    """Renders the main page for analyzing new articles."""
    st.header("Analyze New Content")

    st.subheader("1. Upload News URLs")
    uploaded_file = st.file_uploader("Upload a CSV file containing a 'url' or 'link' column.", type=["csv"])
    if uploaded_file:
        st.session_state.articles = load_and_fetch_headlines(uploaded_file)
    
    if st.session_state.articles:
        st.subheader("2. Select an Article to Analyze")
        headlines = [article['headline'] for article in st.session_state.articles]
        selected_headline = st.selectbox("Choose an article from your list:", options=headlines, index=None, placeholder="Select a headline...")
        if selected_headline:
            selected_article = next((article for article in st.session_state.articles if article['headline'] == selected_headline), None)
            if selected_article:
                st.info(f"**Selected URL:** `{selected_article['url']}`")
                if st.button("🔍 Analyze Selected Article", type="primary"):
                    st.session_state.current_analysis = None
                    st.session_state.original_analysis = None
                    st.session_state.credibility_assessment = None
                    st.session_state.similar_articles = []
                    with st.spinner("Scraping and analyzing... This may take a moment."):
                        content_data = scrape_webpage(selected_article['url'])
                        if 'error' in content_data: st.error(content_data['error'])
                        else:
                            st.success("✅ Content scraped!")
                            with st.spinner("🤖 AI is classifying the content..."):
                                classification = classify_news_content(content_data)
                                if 'error' not in classification:
                                    classification['url'] = selected_article['url']
                                    st.session_state.current_analysis = classification
                                    st.rerun()

    st.markdown("---")
    st.subheader("Or, Analyze a YouTube Video")
    youtube_url = st.text_input("Enter a YouTube video URL:", placeholder="e.g., https://www.youtube.com/watch?v=...")
    if st.button("👁️‍🗨️ Analyze YouTube Video", type="primary"):
        if youtube_url:
            st.session_state.current_analysis = None
            st.session_state.original_analysis = None
            st.session_state.credibility_assessment = None
            st.session_state.similar_articles = []
            with st.spinner("Fetching YouTube captions and analyzing..."):
                caption_data = get_youtube_captions(youtube_url)
                if 'error' in caption_data: st.error(caption_data['error'])
                else:
                    st.success("✅ Captions fetched!")
                    with st.spinner("🤖 AI is classifying the captions..."):
                        classification = classify_news_content(caption_data)
                        if 'error' not in classification:
                            classification['url'] = youtube_url
                            st.session_state.current_analysis = classification
                            st.rerun()
        else:
            st.warning("Please enter a YouTube URL.")

    if st.session_state.current_analysis:
        st.markdown("---")
        st.header("📊 Analysis Results")
        display_classification_results(st.session_state.current_analysis)
        with st.expander("📄 View Raw Content"):
            if 'url' in st.session_state.current_analysis:
                with st.spinner("Fetching raw content for preview..."):
                    raw_content_data = scrape_webpage(st.session_state.current_analysis['url']) if "youtube.com" not in st.session_state.current_analysis['url'] else get_youtube_captions(st.session_state.current_analysis['url'])
                    if 'error' not in raw_content_data:
                        st.write(f"**Title:** {raw_content_data.get('title', 'No title')}")
                        st.text_area("Content Preview:", raw_content_data.get('content', '')[:2000] + "...", height=300)
                    else:
                        st.error(f"Could not fetch raw content: {raw_content_data['error']}")

# --- MAIN APP LAYOUT ---
# Note: The main title and page navigation is now handled by Streamlit's multipage app feature
# st.title("SHOTTIFY News Analyzer & Classifier") <--- REMOVED

with st.sidebar:
    st.title("⚙️ System Info")
    # app_mode = st.radio("Choose a page:", ["Live Analysis", "Dashboard"]) <--- REMOVED
    
    st.write("**Topic Categories:** Politics, Economy, Technology, etc.")
    st.write("**Content Types:** ✅ Fact, ❌ Fake, ⚠️ Distorted, 💭 Opinion")
    st.write("**Urgency Levels:** 🔴 Red, 🟠 Orange, 🟡 Yellow")
    st.markdown("---")
    display_feedback_stats()
    st.markdown("---")
    if st.button("🗑️ Clear All Corrections"):
        st.session_state.corrections = []
        st.success("All corrections cleared!")
        st.rerun()

# --- Page Routing ---
# Note: Page routing is now handled by placing this file in the `pages` directory
display_live_analysis_page()

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🔬 Powered by Groq & Google Cloud • Built with Streamlit • Enhanced with Learning</p>
    <p><small>Analysis results are AI-generated and should be verified independently for critical decisions.</small></p>
</div>
""", unsafe_allow_html=True)