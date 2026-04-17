"""
Data Loader Module
Handles loading CSV files and filtering tweets by policy keywords
"""

import pandas as pd
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def load_raw_tweets(filename=None):
    """Load raw tweets from CSV file"""
    if filename is None:
        filename = config.RAW_TWEETS_FILE

    filepath = os.path.join(config.RAW_DATA_DIR, filename)
    print(f"📂 Loading tweets from: {filepath}")

    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        print("💡 Make sure tweets.csv is inside data/raw/")
        return None

    try:
        try:
            df = pd.read_csv(filepath, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(filepath, encoding='latin-1')

        print(f"✅ Loaded {len(df)} tweets successfully")
        print(f"📊 Columns found: {list(df.columns)}")
        return df

    except Exception as e:
        print(f"❌ Error loading file: {str(e)}")
        return None


def filter_policy_tweets(df, text_column=None, keywords=None):
    """Filter tweets containing policy-related keywords"""
    if df is None or df.empty:
        print("❌ Cannot filter: DataFrame is empty or None")
        return None

    if keywords is None:
        keywords = config.POLICY_KEYWORDS

    print(f"\n🔍 Filtering tweets by {len(keywords)} policy keywords...")

    if text_column is None or text_column not in df.columns:
        possible_names = ['tweet', 'text', 'content', 'message', 'Text', 'Tweet',
                         'full_text', 'tweet_text', 'body', 'post']
        text_column = None
        for col in possible_names:
            if col in df.columns:
                text_column = col
                break

        if text_column is None:
            print(f"❌ Error: Could not find text column. Available columns: {list(df.columns)}")
            return None

    print(f"📝 Using column '{text_column}' for text analysis")

    df['text_lower'] = df[text_column].astype(str).str.lower()
    pattern = '|'.join(keywords)
    mask = df['text_lower'].str.contains(pattern, case=False, na=False, regex=True)
    filtered_df = df[mask].copy()
    filtered_df = filtered_df.drop('text_lower', axis=1, errors='ignore')

    print(f"✅ Found {len(filtered_df)} policy-related tweets ({len(filtered_df)/len(df)*100:.1f}% of total)")
    return filtered_df


def get_basic_stats(df, text_column=None):
    """Get basic statistics about the tweet dataset"""
    if df is None or df.empty:
        return {}

    if text_column is None or text_column not in df.columns:
        possible_names = ['tweet', 'text', 'content', 'message', 'Text', 'Tweet',
                         'full_text', 'tweet_text', 'body', 'post']
        for col in possible_names:
            if col in df.columns:
                text_column = col
                break

    stats = {
        'total_tweets': len(df),
        'unique_tweets': df[text_column].nunique() if text_column else 0,
        'columns': list(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'date_range': None
    }

    date_columns = ['date', 'created_at', 'timestamp', 'Date', 'Timestamp', 'time', 'datetime']
    for date_col in date_columns:
        if date_col in df.columns:
            try:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                stats['date_range'] = {
                    'start': df[date_col].min(),
                    'end': df[date_col].max()
                }
                break
            except:
                continue

    return stats


def save_processed_tweets(df, filename=None):
    """Save processed tweets to CSV"""
    if filename is None:
        filename = config.PROCESSED_TWEETS_FILE

    filepath = os.path.join(config.PROCESSED_DATA_DIR, filename)
    df.to_csv(filepath, index=False, encoding='utf-8')
    print(f"💾 Saved {len(df)} processed tweets to: {filepath}")
