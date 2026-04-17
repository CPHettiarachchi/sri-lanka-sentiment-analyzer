"""
Main Pipeline Script
Runs the complete data processing and sentiment analysis pipeline
"""

import os
import sys
import pandas as pd
import config
from src.data_loader import load_raw_tweets, filter_policy_tweets, save_processed_tweets, get_basic_stats
from src.preprocessing import TextPreprocessor, add_text_features, get_word_frequency
from src.model_training import SentimentModelTrainer
from src.sentiment_analyzer import PolicySentimentAnalyzer, save_analysis_results
from src.visualization import (
    plot_sentiment_distribution, plot_sentiment_pie_chart,
    generate_wordcloud, plot_top_keywords, plot_confidence_distribution
)
import matplotlib.pyplot as plt


def print_section_header(title):
    """Print formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def run_complete_pipeline():
    """
    Run the complete sentiment analysis pipeline
    """
    print_section_header("🇱🇰 SRI LANKA SENTIMENT ANALYZER - COMPLETE PIPELINE")
    
    # ========================================================================
    # STEP 1: Load Data
    # ========================================================================
    print_section_header("STEP 1: LOADING RAW DATA")
    
    raw_df = load_raw_tweets()
    
    if raw_df is None:
        print("❌ Error: Could not load data. Please check file location.")
        print(f"📁 Expected location: {os.path.join(config.RAW_DATA_DIR, config.RAW_TWEETS_FILE)}")
        return
    
    # Get basic statistics
    stats = get_basic_stats(raw_df)
    print(f"\n📊 Raw Data Statistics:")
    print(f"   Total tweets: {stats['total_tweets']:,}")
    print(f"   Unique tweets: {stats['unique_tweets']:,}")
    if stats['date_range']:
        print(f"   Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")
    
    # ========================================================================
    # STEP 2: Filter Policy Tweets
    # ========================================================================
    print_section_header("STEP 2: FILTERING POLICY-RELATED TWEETS")
    
    policy_df = filter_policy_tweets(raw_df)
    
    if policy_df is None or len(policy_df) == 0:
        print("❌ Error: No policy tweets found after filtering.")
        return
    
    print(f"\n✅ Filtered to {len(policy_df):,} policy-related tweets")
    
    # ========================================================================
    # STEP 3: Preprocess Text
    # ========================================================================
    print_section_header("STEP 3: TEXT PREPROCESSING")
    
    preprocessor = TextPreprocessor(remove_stopwords=True)
    processed_df = preprocessor.preprocess_dataframe(policy_df)
    
    # Add text features
    processed_df = add_text_features(processed_df)
    
    print(f"\n📝 Added text features:")
    print(f"   - Text length")
    print(f"   - Word count")
    print(f"   - Hashtag count")
    print(f"   - Mention count")
    
    # Save processed tweets
    save_processed_tweets(processed_df)
    
    # ========================================================================
    # STEP 4: Exploratory Data Analysis
    # ========================================================================
    print_section_header("STEP 4: EXPLORATORY DATA ANALYSIS")
    
    # Word frequency
    print("📊 Generating word frequency analysis...")
    word_freq = get_word_frequency(processed_df, top_n=20)
    print("\nTop 10 Keywords:")
    print(word_freq.head(10).to_string(index=False))
    
    # Create visualizations directory
    viz_dir = os.path.join(config.BASE_DIR, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Plot top keywords
    print("\n📊 Creating visualizations...")
    fig = plot_top_keywords(word_freq)
    fig.savefig(os.path.join(viz_dir, 'top_keywords.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: top_keywords.png")
    
    # Generate word cloud
    fig = generate_wordcloud(processed_df)
    if fig:
        fig.savefig(os.path.join(viz_dir, 'wordcloud_all.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✅ Saved: wordcloud_all.png")
    
    # ========================================================================
    # STEP 5: Train Sentiment Model
    # ========================================================================
    print_section_header("STEP 5: TRAINING SENTIMENT MODEL")
    
    model_path = os.path.join(config.MODELS_DIR, config.MODEL_FILE)
    
    if os.path.exists(model_path):
        print("ℹ️  Found existing model. Skipping training.")
        print(f"   To retrain, delete: {model_path}")
    else:
        print("🎯 Training new sentiment classification model...")
        trainer = SentimentModelTrainer(model_type='logistic_regression')
        results = trainer.train_and_save()
        
        print(f"\n✅ Model training complete!")
        print(f"   Accuracy: {results['accuracy']:.2%}")
    
    # ========================================================================
    # STEP 6: Sentiment Analysis
    # ========================================================================
    print_section_header("STEP 6: SENTIMENT ANALYSIS")
    
    analyzer = PolicySentimentAnalyzer()
    
    if not analyzer.model_loaded:
        print("❌ Error: Could not load model.")
        return
    
    # Analyze sentiment
    processed_df = analyzer.analyze_sentiment(processed_df)
    
    # Get overall statistics
    overall_stats = analyzer.get_overall_statistics(processed_df)
    
    print(f"\n📊 Overall Sentiment Statistics:")
    print(f"   Total analyzed: {overall_stats['total_tweets']:,}")
    print(f"   Average confidence: {overall_stats['average_confidence']:.2%}")
    print(f"\n   Sentiment Distribution:")
    for sentiment, count in overall_stats['sentiment_distribution'].items():
        percentage = overall_stats['sentiment_percentages'][sentiment]
        print(f"      {sentiment}: {count:,} ({percentage:.1f}%)")
    
    # Save analyzed data
    save_analysis_results(processed_df, filename='final_analysis.csv')
    
    # ========================================================================
    # STEP 7: Generate Visualizations
    # ========================================================================
    print_section_header("STEP 7: GENERATING VISUALIZATIONS")
    
    # Sentiment distribution
    fig = plot_sentiment_distribution(processed_df)
    fig.savefig(os.path.join(viz_dir, 'sentiment_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: sentiment_distribution.png")
    
    # Pie chart
    fig = plot_sentiment_pie_chart(processed_df)
    fig.savefig(os.path.join(viz_dir, 'sentiment_pie.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: sentiment_pie.png")
    
    # Confidence distribution
    fig = plot_confidence_distribution(processed_df)
    fig.savefig(os.path.join(viz_dir, 'confidence_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: confidence_distribution.png")
    
    # Word clouds by sentiment
    for sentiment in ['Positive', 'Negative', 'Neutral']:
        fig = generate_wordcloud(processed_df, sentiment=sentiment)
        if fig:
            filename = f'wordcloud_{sentiment.lower()}.png'
            fig.savefig(os.path.join(viz_dir, filename), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✅ Saved: {filename}")
    
    # ========================================================================
    # STEP 8: Sample Results
    # ========================================================================
    print_section_header("STEP 8: SAMPLE RESULTS")
    
    # Find text column
    text_col = 'text'
    if 'text' not in processed_df.columns:
        for col in ['tweet', 'content', 'message']:
            if col in processed_df.columns:
                text_col = col
                break
    
    print("\n📝 Sample Positive Tweets:")
    positive_tweets = analyzer.get_top_tweets_by_sentiment(processed_df, 'Positive', n=3)
    for i, row in positive_tweets.iterrows():
        print(f"   • {row[text_col][:100]}...")
        print(f"     Confidence: {row['confidence']:.1%}\n")
    
    print("📝 Sample Negative Tweets:")
    negative_tweets = analyzer.get_top_tweets_by_sentiment(processed_df, 'Negative', n=3)
    for i, row in negative_tweets.iterrows():
        print(f"   • {row[text_col][:100]}...")
        print(f"     Confidence: {row['confidence']:.1%}\n")
    
    # ========================================================================
    # STEP 9: Keyword Analysis
    # ========================================================================
    print_section_header("STEP 9: KEYWORD SENTIMENT ANALYSIS")
    
    # Analyze sentiment for key policy terms
    key_terms = ['tax', 'budget', 'government', 'protest', 'economy']
    
    print("\n📊 Sentiment by Policy Keyword:\n")
    for keyword in key_terms:
        keyword_stats = analyzer.get_keyword_sentiment(processed_df, keyword)
        if keyword_stats:
            print(f"   {keyword.upper()}:")
            print(f"      Total tweets: {keyword_stats['total_tweets']}")
            for sent, pct in keyword_stats['sentiment_percentages'].items():
                print(f"         {sent}: {pct:.1f}%")
            print()
    
    # ========================================================================
    # COMPLETION
    # ========================================================================
    print_section_header("✅ PIPELINE COMPLETE!")
    
    print("📁 Output Files:")
    print(f"   • Processed data: {os.path.join(config.PROCESSED_DATA_DIR, 'final_analysis.csv')}")
    print(f"   • Trained model: {os.path.join(config.MODELS_DIR, config.MODEL_FILE)}")
    print(f"   • Visualizations: {viz_dir}/")
    
    print("\n🚀 Next Steps:")
    print("   1. Review visualizations in the visualizations/ folder")
    print("   2. Run Streamlit dashboard: streamlit run app.py")
    print("   3. Explore interactive analysis in the web interface")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    try:
        run_complete_pipeline()
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
