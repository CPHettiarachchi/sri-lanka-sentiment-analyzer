"""
Visualization Module
Contains all plotting and visualization functions
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import config


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = config.FIGURE_SIZE
plt.rcParams['figure.dpi'] = config.DPI


def plot_sentiment_distribution(df, save_path=None):
    """
    Plot sentiment distribution as bar chart
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with sentiment column
    save_path : str, optional
        Path to save the plot
    
    Returns:
    --------
    matplotlib.figure.Figure : The plot figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get sentiment counts
    sentiment_counts = df['sentiment'].value_counts()
    
    # Create colors based on sentiment
    colors = [config.SENTIMENT_COLORS[sent] for sent in sentiment_counts.index]
    
    # Create bar plot
    bars = ax.bar(sentiment_counts.index, sentiment_counts.values, color=colors, alpha=0.7)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Sentiment', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Tweets', fontsize=12, fontweight='bold')
    ax.set_title('Sentiment Distribution of Policy-Related Tweets', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_sentiment_pie_chart(df, save_path=None):
    """
    Plot sentiment distribution as pie chart
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with sentiment column
    save_path : str, optional
        Path to save the plot
    
    Returns:
    --------
    matplotlib.figure.Figure : The plot figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    sentiment_counts = df['sentiment'].value_counts()
    colors = [config.SENTIMENT_COLORS[sent] for sent in sentiment_counts.index]
    
    wedges, texts, autotexts = ax.pie(
        sentiment_counts.values,
        labels=sentiment_counts.index,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        textprops={'fontsize': 12, 'fontweight': 'bold'}
    )
    
    # Make percentage text white
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(14)
    
    ax.set_title('Overall Sentiment Distribution', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_sentiment_over_time(time_series, save_path=None):
    """
    Plot sentiment trends over time
    
    Parameters:
    -----------
    time_series : pd.DataFrame
        Time series dataframe with sentiment counts
    save_path : str, optional
        Path to save the plot
    
    Returns:
    --------
    matplotlib.figure.Figure : The plot figure
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for sentiment in time_series.columns:
        if sentiment in config.SENTIMENT_COLORS:
            ax.plot(time_series.index, time_series[sentiment],
                   marker='o', linewidth=2, label=sentiment,
                   color=config.SENTIMENT_COLORS[sentiment])
    
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Tweets', fontsize=12, fontweight='bold')
    ax.set_title('Sentiment Trends Over Time', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_interactive_sentiment_timeline(time_series):
    """
    Create interactive plotly timeline of sentiments
    
    Parameters:
    -----------
    time_series : pd.DataFrame
        Time series dataframe with sentiment counts
    
    Returns:
    --------
    plotly.graph_objects.Figure : Interactive plot
    """
    fig = go.Figure()
    
    for sentiment in time_series.columns:
        if sentiment in config.SENTIMENT_COLORS:
            fig.add_trace(go.Scatter(
                x=time_series.index,
                y=time_series[sentiment],
                mode='lines+markers',
                name=sentiment,
                line=dict(color=config.SENTIMENT_COLORS[sentiment], width=3),
                marker=dict(size=8)
            ))
    
    fig.update_layout(
        title='Interactive Sentiment Timeline',
        xaxis_title='Date',
        yaxis_title='Number of Tweets',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    return fig


def generate_wordcloud(df, sentiment=None, text_column='cleaned_text', save_path=None):
    """
    Generate word cloud from tweets
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with text
    sentiment : str, optional
        Filter by specific sentiment
    text_column : str
        Column containing text
    save_path : str, optional
        Path to save the plot
    
    Returns:
    --------
    matplotlib.figure.Figure : The plot figure
    """
    # Filter by sentiment if specified
    if sentiment:
        df = df[df['sentiment'] == sentiment]
    
    if len(df) == 0:
        print(f"⚠️  No tweets found for sentiment: {sentiment}")
        return None
    
    # Combine all text
    text = ' '.join(df[text_column].astype(str))
    
    # Generate word cloud
    wordcloud = WordCloud(
        width=1200,
        height=600,
        background_color='white',
        colormap='viridis' if not sentiment else None,
        max_words=100,
        relative_scaling=0.5,
        min_font_size=10
    ).generate(text)
    
    # Plot
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    
    title = f'Word Cloud - {sentiment} Tweets' if sentiment else 'Word Cloud - All Tweets'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_keyword_sentiment(keyword_stats, save_path=None):
    """
    Plot sentiment distribution for a specific keyword
    
    Parameters:
    -----------
    keyword_stats : dict
        Dictionary with keyword sentiment statistics
    save_path : str, optional
        Path to save the plot
    
    Returns:
    --------
    matplotlib.figure.Figure : The plot figure
    """
    if keyword_stats is None:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sentiments = list(keyword_stats['sentiment_counts'].keys())
    counts = list(keyword_stats['sentiment_counts'].values())
    colors = [config.SENTIMENT_COLORS[s] for s in sentiments]
    
    bars = ax.bar(sentiments, counts, color=colors, alpha=0.7)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Sentiment', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Tweets', fontsize=12, fontweight='bold')
    ax.set_title(f'Sentiment Distribution for "{keyword_stats["keyword"]}"',
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_confidence_distribution(df, save_path=None):
    """
    Plot distribution of confidence scores
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with confidence scores
    save_path : str, optional
        Path to save the plot
    
    Returns:
    --------
    matplotlib.figure.Figure : The plot figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histogram
    ax.hist(df['confidence'], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    
    # Add mean line
    mean_conf = df['confidence'].mean()
    ax.axvline(mean_conf, color='red', linestyle='--', linewidth=2,
               label=f'Mean: {mean_conf:.2f}')
    
    ax.set_xlabel('Confidence Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Sentiment Prediction Confidence',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_top_keywords(word_freq_df, save_path=None):
    """
    Plot top keywords as horizontal bar chart
    
    Parameters:
    -----------
    word_freq_df : pd.DataFrame
        Dataframe with Word and Frequency columns
    save_path : str, optional
        Path to save the plot
    
    Returns:
    --------
    matplotlib.figure.Figure : The plot figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Sort by frequency
    word_freq_df = word_freq_df.sort_values('Frequency', ascending=True)
    
    # Create horizontal bar chart
    bars = ax.barh(word_freq_df['Word'], word_freq_df['Frequency'],
                   color='teal', alpha=0.7)
    
    ax.set_xlabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_ylabel('Words', fontsize=12, fontweight='bold')
    ax.set_title('Top Keywords in Policy Tweets', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_sentiment_gauge(sentiment_percentages):
    """
    Create gauge chart showing overall sentiment
    
    Parameters:
    -----------
    sentiment_percentages : dict
        Dictionary with sentiment percentages
    
    Returns:
    --------
    plotly.graph_objects.Figure : Gauge chart
    """
    # Calculate sentiment score (-1 to 1)
    pos = sentiment_percentages.get('Positive', 0)
    neg = sentiment_percentages.get('Negative', 0)
    
    # Sentiment score: (Positive - Negative) / 100
    score = (pos - neg) / 100
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Overall Sentiment Score", 'font': {'size': 24}},
        delta={'reference': 0},
        gauge={
            'axis': {'range': [-1, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [-1, -0.3], 'color': config.SENTIMENT_COLORS['Negative']},
                {'range': [-0.3, 0.3], 'color': config.SENTIMENT_COLORS['Neutral']},
                {'range': [0.3, 1], 'color': config.SENTIMENT_COLORS['Positive']}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    
    fig.update_layout(height=400)
    
    return fig


if __name__ == "__main__":
    # Test visualizations
    print("=" * 70)
    print("TESTING VISUALIZATION MODULE")
    print("=" * 70)
    
    # Create sample data
    sample_data = {
        'sentiment': ['Positive'] * 50 + ['Negative'] * 30 + ['Neutral'] * 20,
        'confidence': np.random.uniform(0.6, 1.0, 100),
        'cleaned_text': ['policy reform economic growth'] * 100
    }
    
    df = pd.DataFrame(sample_data)
    
    print("\n📊 Generating test visualizations...")
    
    # Test sentiment distribution
    fig1 = plot_sentiment_distribution(df)
    print("✅ Created sentiment distribution plot")
    
    # Test pie chart
    fig2 = plot_sentiment_pie_chart(df)
    print("✅ Created pie chart")
    
    # Test word cloud
    fig3 = generate_wordcloud(df)
    print("✅ Created word cloud")
    
    plt.close('all')
    print("\n✅ All visualizations tested successfully!")
