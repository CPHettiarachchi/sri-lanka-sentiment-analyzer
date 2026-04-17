# Sri Lanka Public Policy Sentiment Analyzer

An NLP-based sentiment analysis system for analyzing public opinion on Sri Lankan government policies using Twitter data.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Technical Details](#technical-details)
- [Screenshots](#screenshots)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project analyzes public sentiment toward Sri Lankan government policies using Natural Language Processing (NLP) and Machine Learning. It processes real Twitter data, classifies tweets into sentiment categories (Positive, Negative, Neutral), and provides interactive visualizations through a Streamlit dashboard.

**Purpose**: Academic project / Internship portfolio demonstrating practical NLP and ML skills.

**Key Capabilities**:
- Process and clean real-world social media data
- Train custom sentiment classification models
- Analyze sentiment trends over time
- Filter analysis by policy keywords
- Interactive dashboard for real-time predictions

---

## Features

### Data Processing
- Automated text cleaning and preprocessing
- Stopword removal and normalization
- Keyword-based filtering for policy-related content
- Feature extraction (hashtags, mentions, URLs)

### Sentiment Analysis
- Machine Learning-based classification (Logistic Regression / Naive Bayes)
- TF-IDF vectorization
- 3-class sentiment prediction (Positive, Negative, Neutral)
- Confidence scores for predictions

### Visualizations
- Sentiment distribution charts (bar, pie)
- Time-series trend analysis
- Word clouds by sentiment
- Keyword-specific sentiment analysis
- Interactive Plotly visualizations

### Interactive Dashboard
- Web-based Streamlit interface
- Real-time sentiment prediction on custom text
- Dynamic keyword filtering
- Comprehensive data overview
- Downloadable results

---

## Project Structure

```
sri-lanka-sentiment-analyzer/
│
├── data/
│   ├── raw/                      # Place your Kaggle CSV here
│   └── processed/                # Cleaned and analyzed data
│
├── models/
│   ├── sentiment_model.pkl       # Trained ML model
│   └── tfidf_vectorizer.pkl      # TF-IDF vectorizer
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Data loading and filtering
│   ├── preprocessing.py          # Text cleaning and NLP
│   ├── model_training.py         # Model training pipeline
│   ├── sentiment_analyzer.py     # Core analysis functions
│   └── visualization.py          # Plotting functions
│
├── visualizations/               # Generated plots (created automatically)
│
├── app.py                        # Streamlit dashboard
├── main_pipeline.py              # Complete analysis pipeline
├── config.py                     # Configuration settings
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (optional)

### Step 1: Clone or Download

```bash
# Option 1: Clone with Git
git clone https://github.com/yourusername/sri-lanka-sentiment-analyzer.git
cd sri-lanka-sentiment-analyzer

# Option 2: Download ZIP and extract
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download NLTK Data

```python
# Run Python and execute:
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

---

## Dataset

### Download Dataset

1. Go to Kaggle: [Twitter Dataset: Sri Lanka Crisis](https://www.kaggle.com/datasets/vishesh1412/twitter-dataset-sri-lanka-crisis)
2. Download the CSV file
3. Place it in `data/raw/` folder
4. Rename it to `tweets.csv` (or update filename in `config.py`)

### Dataset Information

- **Source**: Kaggle
- **Content**: Tweets about Sri Lankan economic and political crisis
- **Size**: Varies (typically 10k-100k tweets)
- **Format**: CSV with columns for text, date, user info, etc.

---

## Usage

### Option 1: Run Complete Pipeline (Recommended for First-Time)

This runs the entire analysis pipeline from data loading to visualization generation:

```bash
python main_pipeline.py
```

**What it does**:
1. Loads raw tweets
2. Filters policy-related content
3. Preprocesses text
4. Trains sentiment model (if not exists)
5. Analyzes sentiment
6. Generates visualizations
7. Saves results

### Option 2: Launch Streamlit Dashboard

For interactive analysis and real-time predictions:

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

### Option 3: Use Individual Modules

```python
# Load and process data
from src.data_loader import load_raw_tweets, filter_policy_tweets
from src.preprocessing import TextPreprocessor

df = load_raw_tweets()
policy_df = filter_policy_tweets(df)

preprocessor = TextPreprocessor()
processed_df = preprocessor.preprocess_dataframe(policy_df)

# Train model
from src.model_training import SentimentModelTrainer

trainer = SentimentModelTrainer()
trainer.train_and_save()

# Analyze sentiment
from src.sentiment_analyzer import PolicySentimentAnalyzer

analyzer = PolicySentimentAnalyzer()
analyzed_df = analyzer.analyze_sentiment(processed_df)
```

---

## Technical Details

### Machine Learning Model

**Algorithm**: Logistic Regression (default) or Naive Bayes

**Features**: 
- TF-IDF vectorization
- Unigrams and bigrams (n-gram range: 1-2)
- Max features: 5000
- Min document frequency: 2
- Max document frequency: 0.8

**Training Data**:
- Synthetic training examples (450 total)
- Balanced across 3 classes
- 80-20 train-test split

**Performance**:
- Expected accuracy: ~85%
- Robust to class imbalance
- Confidence scoring for predictions

### Text Preprocessing Pipeline

1. **Lowercase conversion**
2. **URL removal** (`http://`, `https://`, `www.`)
3. **User mention removal** (`@username`)
4. **Hashtag symbol removal** (keeping the word)
5. **Punctuation removal**
6. **Stopword removal** (English + custom Twitter stopwords)
7. **Whitespace normalization**

### Sentiment Classes

- **Positive (2)**: Favorable opinions, support, praise
- **Neutral (1)**: Factual statements, announcements
- **Negative (0)**: Criticism, complaints, opposition

---

## Screenshots

### Dashboard Home
![Home Page](docs/screenshots/home.png)

### Sentiment Analysis
![Sentiment Analysis](docs/screenshots/analysis.png)

### Interactive Predictions
![Live Predictions](docs/screenshots/predict.png)

*Note: Screenshots will be generated after running the application*

---

## Future Enhancements

### Short-term
- [ ] Add more sophisticated models (BERT, RoBERTa)
- [ ] Implement real-time Twitter streaming
- [ ] Add multi-language support (Sinhala, Tamil)
- [ ] Export reports to PDF

### Long-term
- [ ] Integrate with news sources for context
- [ ] Add entity recognition for politicians/policies
- [ ] Implement topic modeling
- [ ] Create API for programmatic access
- [ ] Deploy on cloud (Heroku, AWS, GCP)

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Dataset**: Kaggle user vishesh1412 for the Sri Lanka Crisis Twitter dataset
- **Libraries**: scikit-learn, NLTK, Streamlit, Plotly
- **Inspiration**: Need for data-driven policy analysis in developing nations

---

## References

1. [NLTK Documentation](https://www.nltk.org/)
2. [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
3. [Streamlit Documentation](https://docs.streamlit.io/)
4. [Sentiment Analysis Best Practices](https://monkeylearn.com/sentiment-analysis/)

---

## Disclaimer

This project is for educational and research purposes only. Sentiment predictions are automated and may not always reflect accurate human judgment. The analysis should not be used for making critical decisions without human verification. All data analysis respects user privacy and does not store personally identifiable information.

---

