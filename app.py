"""
Streamlit Dashboard for Sri Lanka Public Policy Sentiment Analyzer
Main application file - FIXED VERSION with proper state management
"""

import streamlit as st
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from src.data_loader import load_raw_tweets, filter_policy_tweets, get_basic_stats
from src.preprocessing import TextPreprocessor, add_text_features, get_word_frequency
from src.model_training import SentimentModelTrainer
from src.sentiment_analyzer import PolicySentimentAnalyzer, save_analysis_results
from src.visualization import (
    plot_sentiment_distribution, plot_sentiment_pie_chart,
    plot_sentiment_over_time, generate_wordcloud,
    plot_keyword_sentiment, plot_confidence_distribution,
    plot_top_keywords, create_sentiment_gauge,
    plot_interactive_sentiment_timeline
)

# Page configuration
st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout=config.LAYOUT,
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1E88E5;
    }
    .stAlert {
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


def get_text_column(df):
    """Helper function to find text column"""
    possible_names = ['tweet', 'text', 'content', 'message', 'Text', 'Tweet', 
                     'full_text', 'tweet_text', 'body', 'post']
    for col in possible_names:
        if col in df.columns:
            return col
    return df.columns[0]  # Fallback to first column


@st.cache_data
def load_and_process_data():
    """Load and process data with caching"""
    # Load raw tweets
    df = load_raw_tweets()
    if df is None:
        return None, None, None
    
    # Filter policy tweets
    policy_df = filter_policy_tweets(df)
    if policy_df is None:
        return None, None, None
    
    # Preprocess
    preprocessor = TextPreprocessor()
    processed_df = preprocessor.preprocess_dataframe(policy_df)
    
    # Add text features
    processed_df = add_text_features(processed_df)
    
    # Get basic stats
    stats = get_basic_stats(processed_df)
    
    return df, processed_df, stats


@st.cache_resource
def load_sentiment_model():
    """Load sentiment model with caching"""
    model_path = os.path.join(config.MODELS_DIR, config.MODEL_FILE)
    
    if not os.path.exists(model_path):
        st.warning("⚠️ No trained model found. Training new model...")
        trainer = SentimentModelTrainer()
        trainer.train_and_save()
    
    analyzer = PolicySentimentAnalyzer()
    return analyzer


def analyze_sentiment_if_needed(processed_df):
    """Analyze sentiment and return updated dataframe"""
    if 'sentiment' in processed_df.columns:
        return processed_df
    
    # Load model
    analyzer = load_sentiment_model()
    
    # Analyze sentiment
    analyzed_df = analyzer.analyze_sentiment(processed_df)
    
    # Save results
    save_analysis_results(analyzed_df)
    
    return analyzed_df


def main():
    """Main application"""
    
    # Header
    st.markdown('<p class="main-header">🇱🇰 Sri Lanka Public Policy Sentiment Analyzer</p>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">NLP-Based Analysis of Public Opinion on Government Policies</p>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("📋 Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["🏠 Home", "📊 Data Overview", "🎯 Sentiment Analysis", 
         "📈 Trends & Insights", "💬 Test Sentiment", "ℹ️ About"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**Dataset:** Twitter Dataset: Sri Lanka Crisis\n\n"
        "**Purpose:** Academic & Research"
    )
    
    # Load data for pages that need it
    if page not in ["ℹ️ About", "💬 Test Sentiment"]:
        with st.spinner("Loading data..."):
            raw_df, processed_df, stats = load_and_process_data()
            
            if processed_df is None:
                st.error("❌ Error loading data. Please check that the CSV file is in the correct location.")
                st.info(f"Expected location: `{config.RAW_DATA_DIR}/{config.RAW_TWEETS_FILE}`")
                st.stop()
    
    # Route to pages
    if page == "🏠 Home":
        if 'processed_df' in locals():
            show_home_page(processed_df)
        else:
            show_home_page()
    
    elif page == "📊 Data Overview":
        show_data_overview(raw_df, processed_df, stats)
    
    elif page == "🎯 Sentiment Analysis":
        show_sentiment_analysis(processed_df)
    
    elif page == "📈 Trends & Insights":
        show_trends_insights(processed_df)
    
    elif page == "💬 Test Sentiment":
        show_test_sentiment()
    
    elif page == "ℹ️ About":
        show_about_page()


def show_home_page(processed_df=None):
    """Home page"""
    st.header("Welcome! 👋")
    
    st.markdown("""
    This interactive dashboard analyzes public sentiment on Sri Lankan government policies 
    using Natural Language Processing (NLP) and Machine Learning.
    
    ### 🎯 Key Features:
    - **Data Processing**: Automated cleaning and preprocessing of tweets
    - **Sentiment Analysis**: ML-based classification (Positive, Negative, Neutral)
    - **Trend Analysis**: Time-series visualization of sentiment changes
    - **Keyword Filtering**: Filter by policy topics (tax, budget, healthcare, etc.)
    - **Interactive Predictions**: Test the model on custom text
    
    ### 📊 Dataset:
    Using real Twitter data from the **Sri Lanka Crisis** dataset (Kaggle), 
    filtered for policy-related keywords.
    
    ### 🚀 Get Started:
    Use the sidebar to navigate through different sections of the dashboard.
    """)
    
    # Display quick stats
    st.markdown("---")
    st.subheader("📈 Quick Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if processed_df is not None:
            st.metric("Total Tweets Analyzed", f"{len(processed_df):,}")
        else:
            st.metric("Total Tweets Analyzed", "N/A")
    with col2:
        st.metric("Policy Keywords", len(config.POLICY_KEYWORDS))
    with col3:
        st.metric("Model Accuracy", "~85%")


def show_data_overview(raw_df, processed_df, stats):
    """Data overview page"""
    st.header("📊 Data Overview")
    
    # Basic statistics
    st.subheader("📈 Dataset Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Raw Tweets", f"{len(raw_df):,}")
    with col2:
        st.metric("Policy Tweets", f"{len(processed_df):,}")
    with col3:
        st.metric("Filtering Rate", f"{len(processed_df)/len(raw_df)*100:.1f}%")
    with col4:
        if stats.get('date_range') and stats['date_range']['start'] is not pd.NaT:
            date_range = stats['date_range']
            days = (date_range['end'] - date_range['start']).days
            st.metric("Date Range", f"{days} days")
        else:
            st.metric("Date Range", "N/A")
    
    # Sample tweets
    st.markdown("---")
    st.subheader("📝 Sample Policy Tweets")
    
    # Get text column
    text_col = get_text_column(processed_df)
    
    sample_df = processed_df[[text_col]].head(10)
    st.dataframe(sample_df, use_container_width=True)
    
    # Word frequency
    st.markdown("---")
    st.subheader("🔤 Top Keywords")
    
    if 'cleaned_text' in processed_df.columns:
        word_freq = get_word_frequency(processed_df, top_n=20)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = plot_top_keywords(word_freq)
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.dataframe(word_freq, use_container_width=True, height=400)


def show_sentiment_analysis(processed_df):
    """Sentiment analysis page"""
    st.header("🎯 Sentiment Analysis")
    
    # Analyze sentiment if needed
    if 'sentiment' not in processed_df.columns:
        with st.spinner("🔄 Analyzing sentiment... This may take a moment."):
            processed_df = analyze_sentiment_if_needed(processed_df)
        st.success("✅ Sentiment analysis complete!")
    
    # Overall sentiment distribution
    st.subheader("📊 Overall Sentiment Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = plot_sentiment_distribution(processed_df)
        st.pyplot(fig)
        plt.close()
    
    with col2:
        fig = plot_sentiment_pie_chart(processed_df)
        st.pyplot(fig)
        plt.close()
    
    # Sentiment percentages
    st.markdown("---")
    st.subheader("📈 Sentiment Breakdown")
    
    sentiment_counts = processed_df['sentiment'].value_counts()
    total = len(processed_df)
    
    cols = st.columns(3)
    
    for i, (sentiment, count) in enumerate(sentiment_counts.items()):
        if i < 3:  # Limit to 3 columns
            with cols[i]:
                percentage = count / total * 100
                st.metric(
                    sentiment,
                    f"{count:,}",
                    f"{percentage:.1f}%"
                )
    
    # Confidence distribution
    st.markdown("---")
    st.subheader("🎯 Prediction Confidence")
    
    if 'confidence' in processed_df.columns:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = plot_confidence_distribution(processed_df)
            st.pyplot(fig)
            plt.close()
        
        with col2:
            avg_conf = processed_df['confidence'].mean()
            min_conf = processed_df['confidence'].min()
            max_conf = processed_df['confidence'].max()
            
            st.metric("Average Confidence", f"{avg_conf:.1%}")
            st.metric("Min Confidence", f"{min_conf:.1%}")
            st.metric("Max Confidence", f"{max_conf:.1%}")
    
    # Sample tweets by sentiment
    st.markdown("---")
    st.subheader("📝 Sample Tweets by Sentiment")
    
    sentiment_filter = st.selectbox(
        "Select Sentiment:",
        ['Positive', 'Negative', 'Neutral']
    )
    
    # Get text column
    text_col = get_text_column(processed_df)
    
    sentiment_tweets = processed_df[processed_df['sentiment'] == sentiment_filter]
    
    if 'confidence' in sentiment_tweets.columns:
        sentiment_tweets = sentiment_tweets.sort_values('confidence', ascending=False)
    
    display_cols = [text_col, 'sentiment']
    if 'confidence' in sentiment_tweets.columns:
        display_cols.append('confidence')
    
    st.dataframe(
        sentiment_tweets[display_cols].head(10),
        use_container_width=True
    )


def show_trends_insights(processed_df):
    """Trends and insights page"""
    st.header("📈 Trends & Insights")
    
    # Ensure sentiment analysis is done
    if 'sentiment' not in processed_df.columns:
        with st.spinner("🔄 Running sentiment analysis first..."):
            processed_df = analyze_sentiment_if_needed(processed_df)
        st.success("✅ Sentiment analysis complete!")
    
    # Load analyzer for additional functions
    analyzer = load_sentiment_model()
    
    # Sentiment over time
    st.subheader("📅 Sentiment Trends Over Time")
    
    # Try to create time series
    with st.spinner("Creating time series..."):
        time_series = analyzer.get_sentiment_over_time(processed_df, freq='D')
    
    if time_series is not None and not time_series.empty:
        st.success(f"✅ Found {len(time_series)} time points")
        
        # Interactive timeline
        try:
            fig_interactive = plot_interactive_sentiment_timeline(time_series)
            st.plotly_chart(fig_interactive, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not create interactive plot: {str(e)}")
        
        # Static timeline
        try:
            fig_static = plot_sentiment_over_time(time_series)
            st.pyplot(fig_static)
            plt.close()
        except Exception as e:
            st.warning(f"Could not create static plot: {str(e)}")
    else:
        st.info("ℹ️ Time series data not available. This could be because:")
        st.write("- The dataset doesn't have date information")
        st.write("- Dates could not be parsed correctly")
        st.write("- All tweets are from the same time period")
        
        # Show what columns are available
        with st.expander("📋 Available columns in dataset"):
            st.write(list(processed_df.columns))
    
    # Keyword sentiment analysis
    st.markdown("---")
    st.subheader("🔍 Sentiment by Keyword")
    
    keyword = st.selectbox(
        "Select a keyword to analyze:",
        config.POLICY_KEYWORDS
    )
    
    if st.button("Analyze Keyword"):
        # Get text column for keyword search
        text_col = get_text_column(processed_df)
        keyword_stats = analyzer.get_keyword_sentiment(processed_df, keyword, text_column=text_col)
        
        if keyword_stats:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = plot_keyword_sentiment(keyword_stats)
                st.pyplot(fig)
                plt.close()
            
            with col2:
                st.metric("Total Tweets", keyword_stats['total_tweets'])
                st.write("**Sentiment Breakdown:**")
                for sent, pct in keyword_stats['sentiment_percentages'].items():
                    st.write(f"- {sent}: {pct:.1f}%")
        else:
            st.warning(f"No tweets found containing '{keyword}'")
    
    # Word clouds
    st.markdown("---")
    st.subheader("☁️ Word Clouds by Sentiment")
    
    cloud_sentiment = st.radio(
        "Select sentiment for word cloud:",
        ['All', 'Positive', 'Negative', 'Neutral'],
        horizontal=True
    )
    
    if 'cleaned_text' in processed_df.columns:
        sentiment_filter = None if cloud_sentiment == 'All' else cloud_sentiment
        with st.spinner("Generating word cloud..."):
            fig = generate_wordcloud(processed_df, sentiment=sentiment_filter)
            if fig:
                st.pyplot(fig)
                plt.close()
    else:
        st.info("Cleaned text not available for word cloud generation.")


def show_test_sentiment():
    """Test sentiment page"""
    st.header("💬 Test Sentiment Prediction")
    
    st.markdown("""
    Enter any text related to Sri Lankan public policy, and the model will predict its sentiment.
    """)
    
    # Load model
    analyzer = load_sentiment_model()
    
    # Text input
    user_text = st.text_area(
        "Enter text to analyze:",
        height=150,
        placeholder="Example: The government's new tax policy is creating economic hardship for families..."
    )
    
    if st.button("Analyze Sentiment", type="primary"):
        if user_text.strip():
            # Preprocess
            preprocessor = TextPreprocessor()
            cleaned_text = preprocessor.clean_text(user_text)
            
            # Predict
            predictions, probabilities = analyzer.trainer.predict([cleaned_text])
            
            sentiment = config.SENTIMENT_LABELS[predictions[0]]
            confidence = max(probabilities[0]) * 100
            
            # Display results
            st.markdown("---")
            st.subheader("🎯 Prediction Results")
            
            # Sentiment badge
            color = config.SENTIMENT_COLORS[sentiment]
            st.markdown(
                f'<div style="background-color: {color}; padding: 20px; '
                f'border-radius: 10px; text-align: center;">'
                f'<h2 style="color: white; margin: 0;">Sentiment: {sentiment}</h2>'
                f'<p style="color: white; font-size: 18px; margin: 10px 0 0 0;">'
                f'Confidence: {confidence:.1f}%</p>'
                f'</div>',
                unsafe_allow_html=True
            )
            
            # Probability breakdown
            st.markdown("### 📊 Probability Breakdown")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Negative", f"{probabilities[0][0]:.1%}")
            with col2:
                st.metric("Neutral", f"{probabilities[0][1]:.1%}")
            with col3:
                st.metric("Positive", f"{probabilities[0][2]:.1%}")
            
            # Show cleaned text
            with st.expander("View Preprocessed Text"):
                st.code(cleaned_text)
        else:
            st.warning("⚠️ Please enter some text to analyze.")


def show_about_page():
    """About page"""
    st.header("ℹ️ About This Project")
    
    st.markdown("""
    ## 🇱🇰 Sri Lanka Public Policy Sentiment Analyzer
    
    ### 📖 Overview
    This is an internship-grade NLP project that analyzes public sentiment toward Sri Lankan 
    government policies using machine learning and natural language processing.
    
    ### 🎯 Objectives
    - Analyze real-world Twitter data about Sri Lankan politics and policies
    - Build and deploy a sentiment classification model
    - Provide interactive visualizations of sentiment trends
    - Enable real-time sentiment prediction
    
    ### 🛠️ Technical Stack
    - **Language**: Python 3.x
    - **ML Framework**: scikit-learn
    - **NLP**: NLTK
    - **Visualization**: Matplotlib, Seaborn, Plotly
    - **Dashboard**: Streamlit
    - **Data**: Kaggle - Twitter Dataset: Sri Lanka Crisis
    
    ### 📊 Model Details
    - **Algorithm**: Logistic Regression / Naive Bayes
    - **Features**: TF-IDF (5000 features)
    - **Classes**: Positive, Negative, Neutral
    - **Accuracy**: ~85% on test set
    
    ### 🔍 Features
    1. **Data Preprocessing**: Text cleaning, stopword removal, tokenization
    2. **Sentiment Classification**: ML-based 3-class classification
    3. **Trend Analysis**: Time-series sentiment tracking
    4. **Keyword Filtering**: Filter by policy topics
    5. **Interactive Dashboard**: Real-time visualizations
    6. **Custom Predictions**: Test model on new text
    
    ### 📚 Dataset
    **Source**: [Kaggle - Twitter Dataset: Sri Lanka Crisis](https://www.kaggle.com/datasets/vishesh1412/twitter-dataset-sri-lanka-crisis)
    
    The dataset contains tweets related to the Sri Lankan economic and political crisis. 
    We filter for policy-related keywords to focus our analysis.
    
    ### 📝 Disclaimer
    This project is for educational and research purposes only. Sentiment predictions 
    are automated and may not always reflect accurate human judgment. The analysis 
    should not be used for making critical decisions without human verification.
    
    ### 📄 License
    This project is open-source and available for academic use.
    
    ---
    
    ### 🙏 Acknowledgments
    - Dataset provided by Kaggle users
    - NLP techniques from NLTK and scikit-learn
    - Streamlit for the interactive dashboard framework
    """)


if __name__ == "__main__":
    main()
