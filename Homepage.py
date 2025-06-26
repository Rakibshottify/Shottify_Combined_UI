import streamlit as st

# Set the page configuration for the entire app
st.set_page_config(
    page_title="SHOTTIFY News Analyzer",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

def show_homepage():
    """Renders the homepage UI."""

    # --- Hero Section ---
    st.title("Welcome to the SHOTTIFY News Analyzer")
    st.markdown("#### Your intelligent partner for navigating the complex world of news. Uncover truth, detect bias, and analyze media with the power of AI.")
    st.markdown("---")

    # --- Key Features Section ---
    st.header("✨ Key Features")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🤖 AI-Powered Classification")
        st.write("Automatically categorize articles by topic (Politics, Tech, etc.), sentiment, and urgency level to quickly grasp the nature of the news.")

        st.subheader("⚖️ Bias & Credibility Assessment")
        st.write("Our AI analyzes the content and cross-references it with other sources to provide a credibility verdict and a bias rating, helping you see the bigger picture.")

    with col2:
        st.subheader("✅ Integrated Fact-Checking")
        st.write("Directly search the Google Fact Check Tools API to find verified fact-checks related to the article's claims without leaving the app.")

        st.subheader("🔗 Bulk URL & YouTube Analysis")
        st.write("Upload a CSV of news URLs for rapid, sequential analysis, or paste a YouTube link to analyze the content of video transcripts.")

    st.markdown("---")

    # --- How It Works Section ---
    st.header("🚀 How It Works")
    st.markdown("""
    **1. Navigate to the 'Live Analysis' Page:** Use the sidebar to go to the analysis tool.
    
    **2. Input Your Content:** Upload a CSV file with news article URLs or paste a link to a YouTube video.
    
    **3. Analyze:** Select an article from your uploaded list to initiate the AI analysis. The system will scrape the content and perform a deep-dive classification.
    
    **4. Verify & Correct:** Use the integrated verification tools to cross-reference and fact-check the story. You can also submit corrections to help the AI learn and improve.
    
    **5. Export (Optional):** Post the final, verified analysis to your Directus database with a single click.
    """)

    st.info("Navigate to the **🔬 Live Analysis** page from the sidebar to get started!")

    # --- Footer ---
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🔬 Powered by Groq & Google Cloud • Built with Streamlit • Enhanced with Learning</p>
        <p><small>Analysis results are AI-generated and should be verified independently for critical decisions.</small></p>
    </div>
    """, unsafe_allow_html=True)

# Run the homepage function
show_homepage()