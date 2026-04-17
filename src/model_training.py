"""
Model Training Module
Trains sentiment classification model using scikit-learn
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import config


class SentimentModelTrainer:

    def __init__(self, model_type='logistic_regression'):
        self.model_type = model_type
        self.vectorizer = None
        self.model = None
        self.label_mapping = config.SENTIMENT_LABELS

    def create_training_data(self):
        print("\n📚 Creating synthetic training data...")

        positive_examples = [
            "excellent policy decision that will benefit everyone",
            "great initiative by the government",
            "wonderful progress on healthcare reforms",
            "impressive economic growth this quarter",
            "outstanding work on infrastructure development",
            "positive changes in education system",
            "happy to see improvements in public services",
            "fantastic budget allocation for social welfare",
            "proud of our country's achievements",
            "optimistic about future economic prospects",
            "commendable effort by officials",
            "amazing response to crisis situation",
            "grateful for government support programs",
            "encouraging news about tax reforms",
            "brilliant strategy for economic recovery"
        ]

        negative_examples = [
            "terrible policy that hurts common people",
            "worst decision ever by government",
            "disappointed with budget allocation",
            "frustrated by lack of action on crisis",
            "angry about fuel price increase",
            "horrible management of economy",
            "unacceptable tax burden on citizens",
            "failed policies causing suffering",
            "disgraceful handling of protests",
            "outraged by corruption scandals",
            "devastating impact on poor families",
            "pathetic response to public demands",
            "shameful neglect of healthcare",
            "appalling state of infrastructure",
            "disastrous economic policies"
        ]

        neutral_examples = [
            "government announced new budget today",
            "parliament session scheduled for next week",
            "minister visited affected areas",
            "policy document released for public review",
            "meeting held to discuss tax reforms",
            "officials presented quarterly report",
            "new legislation proposed in parliament",
            "committee reviewing healthcare policies",
            "data shows economic indicators",
            "president addressed nation on television",
            "cabinet meeting concluded yesterday",
            "ministry published annual statistics",
            "experts analyzing proposed reforms",
            "survey conducted among citizens",
            "report submitted to authorities"
        ]

        all_positive = positive_examples * 10
        all_negative = negative_examples * 10
        all_neutral = neutral_examples * 10

        texts = all_positive + all_negative + all_neutral
        labels = ([2] * len(all_positive) +
                  [0] * len(all_negative) +
                  [1] * len(all_neutral))

        training_df = pd.DataFrame({'text': texts, 'sentiment': labels})
        training_df = training_df.sample(frac=1, random_state=config.RANDOM_STATE).reset_index(drop=True)

        print(f"✅ Created {len(training_df)} training examples")
        return training_df

    def train_model(self, X_train, y_train):
        print(f"\n🎯 Training {self.model_type} model...")

        if self.model_type == 'logistic_regression':
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=config.RANDOM_STATE,
                class_weight='balanced'
            )
        elif self.model_type == 'naive_bayes':
            self.model = MultinomialNB(alpha=1.0)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self.model.fit(X_train, y_train)
        print(f"✅ Model trained successfully!")

    def evaluate_model(self, X_test, y_test):
        print("\n📊 Evaluating model performance...")
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        target_names = [self.label_mapping[i] for i in sorted(self.label_mapping.keys())]
        report = classification_report(y_test, y_pred, target_names=target_names)
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n✅ Accuracy: {accuracy:.2%}")
        print(f"\n📋 Classification Report:\n{report}")
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred
        }

    def train_and_save(self, texts=None, labels=None, save_path=None):
        if texts is None or labels is None:
            training_df = self.create_training_data()
            texts = training_df['text'].tolist()
            labels = training_df['sentiment'].tolist()

        print("\n🔤 Creating TF-IDF features...")
        self.vectorizer = TfidfVectorizer(
            max_features=config.MAX_FEATURES,
            min_df=config.MIN_DF,
            max_df=config.MAX_DF,
            ngram_range=(1, 2),
            strip_accents='unicode',
            lowercase=True
        )

        X = self.vectorizer.fit_transform(texts)
        y = np.array(labels)
        print(f"✅ Created TF-IDF matrix: {X.shape}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE,
            stratify=y
        )

        self.train_model(X_train, y_train)
        results = self.evaluate_model(X_test, y_test)

        if save_path is None:
            model_path = os.path.join(config.MODELS_DIR, config.MODEL_FILE)
            vectorizer_path = os.path.join(config.MODELS_DIR, config.VECTORIZER_FILE)
        else:
            model_path = save_path
            vectorizer_path = save_path.replace('.pkl', '_vectorizer.pkl')

        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"\n💾 Model saved to: {model_path}")

        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        print(f"💾 Vectorizer saved to: {vectorizer_path}")

        return results

    def load_model(self, model_path=None, vectorizer_path=None):
        if model_path is None:
            model_path = os.path.join(config.MODELS_DIR, config.MODEL_FILE)
        if vectorizer_path is None:
            vectorizer_path = os.path.join(config.MODELS_DIR, config.VECTORIZER_FILE)

        print(f"\n📂 Loading model from: {model_path}")
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        print(f"📂 Loading vectorizer from: {vectorizer_path}")
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)

        print("✅ Model and vectorizer loaded successfully!")

    def predict(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        X = self.vectorizer.transform(texts)
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        return predictions, probabilities
