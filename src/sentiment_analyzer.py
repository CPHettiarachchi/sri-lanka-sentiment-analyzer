"""
Sentiment Analyzer Module
Core functions for sentiment analysis on Sri Lankan policy tweets
FIXED VERSION - Proper time series grouping
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.model_training import SentimentModelTrainer
from src.preprocessing import TextPreprocessor


class PolicySentimentAnalyzer:
    """
    Main class for analyzing sentiment in policy-related tweets
    """
    
    def __init__(self, model_path=None, vectorizer_path=None):
        """
        Initialize the sentiment analyzer
        
        Parameters:
        -----------
        model_path : str, optional
            Path to trained model file
        vectorizer_path : str, optional
            Path to vectorizer file
        """
        self.preprocessor = TextPreprocessor()
        self.trainer = SentimentModelTrainer()
        
        # Try to load existing model, or train new one
        try:
            self.trainer.load_model(model_path, vectorizer_path)
            self.model_loaded = True
        except FileNotFoundError:
            print("⚠️  No trained model found. Please train a model first.")
            self.model_loaded = False
    
    def analyze_sentiment(self, df, text_column='cleaned_text'):
        """
        Analyze sentiment for all tweets in dataframe
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataframe with tweets
        text_column : str
            Column containing cleaned text
        
        Returns:
        --------
        pd.DataFrame : Dataframe with sentiment predictions added
        """
        if not self.model_loaded:
            print("❌ Error: Model not loaded. Cannot analyze sentiment.")
            return df
        
        print(f"\n🎯 Analyzing sentiment for {len(df)} tweets...")
        
        # Get predictions
        texts = df[text_column].tolist()
        predictions, probabilities = self.trainer.predict(texts)
        
        # Add predictions to dataframe
        df['sentiment_label'] = predictions
        df['sentiment'] = df['sentiment_label'].map(config.SENTIMENT_LABELS)
        
        # Add confidence scores
        df['confidence'] = [max(prob) for prob in probabilities]
        
        # Add individual probabilities
        df['prob_negative'] = probabilities[:, 0]
        df['prob_neutral'] = probabilities[:, 1]
        df['prob_positive'] = probabilities[:, 2]
        
        print(f"✅ Sentiment analysis complete!")
        
        # Print distribution
        sentiment_counts = df['sentiment'].value_counts()
        print(f"\n📊 Sentiment Distribution:")
        for sentiment, count in sentiment_counts.items():
            percentage = count / len(df) * 100
            print(f"   {sentiment}: {count} ({percentage:.1f}%)")
        
        return df
    
    def get_sentiment_over_time(self, df, date_column=None, freq='D'):
        """
        Aggregate sentiment counts over time - FIXED VERSION
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataframe with sentiment labels and dates
        date_column : str, optional
            Column containing dates (auto-detects if None)
        freq : str
            Frequency for aggregation ('D'=daily, 'W'=weekly, 'M'=monthly)
        
        Returns:
        --------
        pd.DataFrame : Time series of sentiment counts
        """
        print("\n📅 Creating time series...")
        
        # Auto-detect date column if not specified
        if date_column is None or date_column not in df.columns:
            possible_date_cols = ['date', 'created_at', 'timestamp', 'Date', 'Timestamp', 
                                 'time', 'datetime', 'tweet_date']
            for col in possible_date_cols:
                if col in df.columns:
                    date_column = col
                    print(f"ℹ️  Using '{date_column}' for time series")
                    break
        
        if date_column is None or date_column not in df.columns:
            print("⚠️  No date column found. Available columns:", list(df.columns))
            return None
        
        # Create a working copy
        df_time = df.copy()
        
        # Convert date column to datetime with multiple attempts
        print(f"📊 Parsing dates from '{date_column}'...")
        
        try:
            # First try: standard datetime
            df_time['parsed_date'] = pd.to_datetime(df_time[date_column], errors='coerce')
            
            # If most dates are NaT, try as timestamp
            if df_time['parsed_date'].isna().sum() > len(df_time) * 0.5:
                print("   Trying timestamp format...")
                df_time['parsed_date'] = pd.to_datetime(df_time[date_column], unit='ms', errors='coerce')
        except:
            print("⚠️  Could not parse dates")
            return None
        
        # Remove rows with invalid dates
        valid_dates = df_time['parsed_date'].notna()
        df_time = df_time[valid_dates].copy()
        
        if len(df_time) == 0:
            print("⚠️  No valid dates found after parsing")
            return None
        
        print(f"✅ Valid dates: {len(df_time)} tweets")
        print(f"📅 Date range: {df_time['parsed_date'].min()} to {df_time['parsed_date'].max()}")
        
        # Check if we have sentiment column
        if 'sentiment' not in df_time.columns:
            print("⚠️  No sentiment column found")
            return None
        
        # Create time series using proper groupby
        print(f"📈 Aggregating by {freq} frequency...")
        
        try:
            # Set parsed_date as index
            df_time = df_time.set_index('parsed_date')
            
            # Create time buckets
            df_time['time_bucket'] = df_time.index.to_period(freq)
            
            # Group by time bucket and sentiment, then count
            grouped = df_time.groupby(['time_bucket', 'sentiment']).size()
            
            # Pivot to get sentiments as columns
            time_series = grouped.unstack(fill_value=0)
            
            # Convert period index to timestamp for plotting
            time_series.index = time_series.index.to_timestamp()
            
            # Ensure all sentiment columns exist
            for sentiment in ['Positive', 'Negative', 'Neutral']:
                if sentiment not in time_series.columns:
                    time_series[sentiment] = 0
            
            print(f"✅ Created time series with {len(time_series)} time points")
            print(f"📊 Columns: {list(time_series.columns)}")
            
            # Show sample data
            if len(time_series) > 0:
                print(f"\n📋 Sample data:")
                print(time_series.head())
            
            return time_series
            
        except Exception as e:
            print(f"⚠️  Error creating time series: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_top_tweets_by_sentiment(self, df, sentiment='Negative', n=10):
        """
        Get top N tweets for a specific sentiment
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataframe with sentiment analysis
        sentiment : str
            Sentiment to filter by ('Positive', 'Negative', 'Neutral')
        n : int
            Number of tweets to return
        
        Returns:
        --------
        pd.DataFrame : Top N tweets
        """
        # Filter by sentiment
        sentiment_df = df[df['sentiment'] == sentiment].copy()
        
        # Sort by confidence
        sentiment_df = sentiment_df.sort_values('confidence', ascending=False)
        
        return sentiment_df.head(n)
    
    def get_keyword_sentiment(self, df, keyword, text_column=None):
        """
        Get sentiment distribution for tweets containing a specific keyword
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataframe with sentiment analysis
        keyword : str
            Keyword to filter by
        text_column : str, optional
            Column to search for keyword (auto-detects if None)
        
        Returns:
        --------
        dict : Sentiment distribution for keyword
        """
        # Auto-detect text column
        if text_column is None or text_column not in df.columns:
            possible_names = ['text', 'tweet', 'content', 'message', 'Text', 'Tweet', 
                            'full_text', 'tweet_text', 'body', 'post']
            for col in possible_names:
                if col in df.columns:
                    text_column = col
                    break
        
        if text_column is None:
            print("⚠️  Could not find text column")
            return None
        
        # Filter tweets containing keyword
        keyword_df = df[df[text_column].str.contains(keyword, case=False, na=False)]
        
        if len(keyword_df) == 0:
            return None
        
        # Get sentiment distribution
        sentiment_counts = keyword_df['sentiment'].value_counts().to_dict()
        total = len(keyword_df)
        
        result = {
            'keyword': keyword,
            'total_tweets': total,
            'sentiment_counts': sentiment_counts,
            'sentiment_percentages': {
                k: (v / total * 100) for k, v in sentiment_counts.items()
            }
        }
        
        return result
    
    def get_overall_statistics(self, df):
        """
        Get overall sentiment statistics
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataframe with sentiment analysis
        
        Returns:
        --------
        dict : Overall statistics
        """
        stats = {
            'total_tweets': len(df),
            'sentiment_distribution': df['sentiment'].value_counts().to_dict(),
            'average_confidence': df['confidence'].mean(),
            'sentiment_percentages': (df['sentiment'].value_counts() / len(df) * 100).to_dict()
        }
        
        # Add most common keywords by sentiment
        if 'sentiment' in df.columns:
            for sentiment in ['Positive', 'Negative', 'Neutral']:
                sentiment_df = df[df['sentiment'] == sentiment]
                if len(sentiment_df) > 0 and 'cleaned_text' in sentiment_df.columns:
                    all_words = ' '.join(sentiment_df['cleaned_text'].astype(str)).split()
                    from collections import Counter
                    word_freq = Counter(all_words).most_common(10)
                    stats[f'{sentiment.lower()}_keywords'] = word_freq
        
        return stats
    
    def detect_sentiment_spikes(self, time_series, threshold=2.0):
        """
        Detect unusual spikes in negative sentiment
        
        Parameters:
        -----------
        time_series : pd.DataFrame
            Time series of sentiment counts
        threshold : float
            Number of standard deviations to consider as spike
        
        Returns:
        --------
        pd.DataFrame : Dates with sentiment spikes
        """
        if time_series is None or 'Negative' not in time_series.columns:
            return None
        
        negative_series = time_series['Negative']
        
        # Calculate mean and std
        mean = negative_series.mean()
        std = negative_series.std()
        
        # Find spikes
        spikes = negative_series[negative_series > mean + threshold * std]
        
        return spikes


def save_analysis_results(df, filename='analyzed_tweets.csv'):
    """
    Save sentiment analysis results to CSV
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with analysis results
    filename : str
        Output filename
    """
    filepath = os.path.join(config.PROCESSED_DATA_DIR, filename)
    df.to_csv(filepath, index=False, encoding='utf-8')
    print(f"\n💾 Analysis results saved to: {filepath}")


if __name__ == "__main__":
    # Test the analyzer
    print("=" * 70)
    print("TESTING SENTIMENT ANALYZER MODULE")
    print("=" * 70)
    
    # Create sample data
    sample_data = {
        'text': [
            "Government's new policy is terrible",
            "Great initiative for economic recovery",
            "Parliament session scheduled for Monday",
            "Frustrated with tax increases",
            "Positive changes in healthcare system"
        ],
        'date': pd.date_range('2024-01-01', periods=5, freq='D')
    }
    
    df = pd.DataFrame(sample_data)
    
    # Preprocess
    preprocessor = TextPreprocessor()
    df = preprocessor.preprocess_dataframe(df)
    
    # Check if model exists
    model_path = os.path.join(config.MODELS_DIR, config.MODEL_FILE)
    if not os.path.exists(model_path):
        print("\n⚠️  No model found. Training new model...")
        trainer = SentimentModelTrainer()
        trainer.train_and_save()
    
    # Analyze sentiment
    analyzer = PolicySentimentAnalyzer()
    df = analyzer.analyze_sentiment(df)
    
    print("\n✅ Sample analysis:")
    print(df[['text', 'sentiment', 'confidence']])
    
    # Get statistics
    stats = analyzer.get_overall_statistics(df)
    print(f"\n📊 Overall Statistics:")
    print(f"   Total tweets: {stats['total_tweets']}")
    print(f"   Average confidence: {stats['average_confidence']:.2%}")
    print(f"   Sentiment distribution: {stats['sentiment_distribution']}")