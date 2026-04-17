"""
Text Preprocessing Module
Handles all text cleaning and preprocessing for NLP
"""

import re
import string
import pandas as pd
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nltk
from nltk.corpus import stopwords
import config

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class TextPreprocessor:

    def __init__(self, remove_stopwords=True, remove_numbers=False):
        self.remove_stopwords = remove_stopwords
        self.remove_numbers = remove_numbers
        self.stopwords_set = set(stopwords.words(config.STOPWORDS_LANGUAGE))
        self.stopwords_set.update(config.CUSTOM_STOPWORDS)

    def clean_text(self, text):
        if pd.isna(text) or not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = ' '.join(text.split())
        if self.remove_stopwords:
            text = self.remove_stopwords_from_text(text)
        return text.strip()

    def remove_stopwords_from_text(self, text):
        words = text.split()
        filtered_words = [word for word in words if word not in self.stopwords_set]
        return ' '.join(filtered_words)

    def preprocess_dataframe(self, df, text_column='tweet', new_column='cleaned_text'):
        print(f"\n🧹 Preprocessing {len(df)} tweets...")

        if text_column not in df.columns:
            possible_names = ['tweet', 'text', 'content', 'message', 'Text', 'Tweet',
                            'full_text', 'tweet_text', 'body', 'post']
            found = False
            for col in possible_names:
                if col in df.columns:
                    text_column = col
                    found = True
                    print(f"ℹ️  Auto-detected text column: '{text_column}'")
                    break

            if not found:
                for col in df.columns:
                    if df[col].dtype == 'object':
                        text_column = col
                        print(f"⚠️  Using '{col}' as text column (auto-detected)")
                        break
                else:
                    raise ValueError(f"Could not find text column. Available columns: {list(df.columns)}")

        print(f"📝 Processing column: '{text_column}'")

        if text_column != 'text':
            df['text'] = df[text_column]

        df[new_column] = df[text_column].apply(self.clean_text)

        initial_count = len(df)
        df = df[df[new_column].str.len() > 0].copy()
        removed = initial_count - len(df)

        if removed > 0:
            print(f"⚠️  Removed {removed} empty tweets after cleaning")

        print(f"✅ Preprocessing complete! {len(df)} tweets ready for analysis")
        return df


def get_word_frequency(df, text_column='cleaned_text', top_n=20):
    from collections import Counter
    all_text = ' '.join(df[text_column].astype(str))
    words = all_text.split()
    word_freq = Counter(words)
    top_words = word_freq.most_common(top_n)
    return pd.DataFrame(top_words, columns=['Word', 'Frequency'])


def extract_hashtags(text):
    if pd.isna(text):
        return []
    hashtags = re.findall(r'#\w+', str(text))
    return [tag.lower() for tag in hashtags]


def extract_mentions(text):
    if pd.isna(text):
        return []
    mentions = re.findall(r'@\w+', str(text))
    return [mention.lower() for mention in mentions]


def add_text_features(df, text_column=None):
    if text_column is None or text_column not in df.columns:
        possible_names = ['text', 'tweet', 'content', 'message', 'Text', 'Tweet',
                         'full_text', 'tweet_text', 'body', 'post']
        text_column = None
        for col in possible_names:
            if col in df.columns:
                text_column = col
                break

        if text_column is None:
            for col in df.columns:
                if df[col].dtype == 'object':
                    text_column = col
                    break

    if text_column is None:
        print("⚠️  Could not find text column for feature extraction")
        return df

    df['text_length'] = df[text_column].astype(str).apply(len)
    df['word_count'] = df[text_column].astype(str).apply(lambda x: len(x.split()))
    df['hashtags'] = df[text_column].apply(extract_hashtags)
    df['hashtag_count'] = df['hashtags'].apply(len)
    df['mentions'] = df[text_column].apply(extract_mentions)
    df['mention_count'] = df['mentions'].apply(len)
    df['has_url'] = df[text_column].astype(str).str.contains(
        r'http\S+|www\S+', regex=True, na=False
    )
    return df
