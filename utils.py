import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def load_data(file_path):
    """
    Load data from an Excel file.

    Args:
        file_path (str or Path): Path to the Excel file.

    Returns:
        pd.DataFrame: Loaded dataframe or empty dataframe on error.
    """
    try:
        df = pd.read_excel(file_path)
        return preprocess_data(df)
    except FileNotFoundError:
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

def preprocess_data(df):
    """
    Preprocess data for better analysis and visualization.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Processed dataframe.
    """
    if df.empty:
        return df

    df_processed = df.copy()

    # Standardize column names to lowercase for consistent access
    df_processed.columns = [str(col).lower() for col in df_processed.columns]

    # Capitalize 'Close' only if it exists (for compatibility with price logic)
    # The price logic expects 'Close' (capitalized) or we should update price logic to use 'close'.
    # Looking at the code below, it uses 'Close', 'MA_7', 'MA_30', etc.
    # To keep things consistent, let's stick to lowercase for everything and update the rest of the functions?
    # Or just rename specific columns back to Title Case if needed?
    # Actually, the price logic uses 'Close', 'Daily_Return', 'Price_Change', etc.
    # So if I lowercased everything, I need to update those references or rename 'close' back to 'Close'.

    # Let's see... the previous code used `if str(col).lower() == 'date'`.
    # To be safe and consistent with the new requirement of fixing column naming regression,
    # I will lowercase everything but then capitalize specific known columns that other functions expect,
    # OR update other functions to use lowercase.
    # Updating other functions is cleaner.

    # Handle datetime columns
    if 'date' in df_processed.columns:
        df_processed['date'] = pd.to_datetime(df_processed['date'], errors='coerce')
        df_processed['date_display'] = df_processed['date'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # Also ensure 'Date' exists if expected by charts (charts use 'Date' capitalized in existing code?)
    # Let's check `create_advanced_price_chart`. It uses `df['Date']` and `df['Close']`.
    # So I should either update the charts or map back.
    # Mapping back is safer for now to minimize changes in logic I didn't write.
    # Wait, I wrote `create_advanced_price_chart` in `utils.py`. I can change it.

    # Let's stick to Title Case for Price data (Date, Close) as that seems to be the convention for financial data,
    # and lowercase for News data (date, title, source, sentiment).

    # Check if this is price data (has 'close' or 'Close')
    is_price_data = False
    for col in df_processed.columns:
        if col.lower() == 'close':
            is_price_data = True
            break

    if is_price_data:
        # Standardize Price Data columns to Title Case
        rename_map = {}
        for col in df_processed.columns:
            if col.lower() == 'date': rename_map[col] = 'Date'
            elif col.lower() == 'close': rename_map[col] = 'Close'
            elif col.lower() == 'high': rename_map[col] = 'High'
            elif col.lower() == 'low': rename_map[col] = 'Low'
            elif col.lower() == 'open': rename_map[col] = 'Open'
            elif col.lower() == 'volume': rename_map[col] = 'Volume'
        df_processed.rename(columns=rename_map, inplace=True)
    else:
        # For News data, use lowercase
        # But wait, `organize_news_by_date` uses lowercase 'date', 'title', 'source', 'sentiment'.
        # And `analyze_sentiment` uses 'sentiment'.
        pass

    # Handle datetime columns again with standardized names
    if 'Date' in df_processed.columns:
        df_processed['Date'] = pd.to_datetime(df_processed['Date'], errors='coerce')
        df_processed['Date_display'] = df_processed['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # Calculate additional metrics if we have price data
    if 'Close' in df_processed.columns:
        try:
            df_processed['Daily_Return'] = df_processed['Close'].pct_change() * 100
            df_processed['Price_Change'] = df_processed['Close'].diff()

            # Calculate moving averages
            if len(df_processed) >= 7:
                df_processed['MA_7'] = df_processed['Close'].rolling(window=7).mean()
            if len(df_processed) >= 30:
                df_processed['MA_30'] = df_processed['Close'].rolling(window=30).mean()

            if 'Daily_Return' in df_processed.columns and len(df_processed) >= 7:
                df_processed['Volatility'] = df_processed['Daily_Return'].rolling(window=7).std()
        except Exception:
            pass # Ignore errors during preprocessing

    return df_processed

def create_advanced_price_chart(df, commodity_name):
    """Create advanced price chart with multiple indicators"""
    if df.empty or 'Date' not in df.columns or 'Close' not in df.columns:
        fig = px.line(title=f'{commodity_name} Price Chart')
        fig.update_layout(height=400)
        return fig

    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(f'{commodity_name} Price Movement', 'Daily Returns'),
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )

    # Price line
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['Close'], name='Close Price',
                  line=dict(color='#1f77b4', width=2)),
        row=1, col=1
    )

    # Moving averages
    if 'MA_7' in df.columns and not df['MA_7'].isna().all():
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['MA_7'], name='7-Day MA',
                      line=dict(color='orange', width=1, dash='dash')),
            row=1, col=1
        )

    if 'MA_30' in df.columns and not df['MA_30'].isna().all():
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['MA_30'], name='30-Day MA',
                      line=dict(color='red', width=1, dash='dash')),
            row=1, col=1
        )

    # Daily returns
    if 'Daily_Return' in df.columns:
        colors = ['green' if x >= 0 else 'red' for x in df['Daily_Return']]
        fig.add_trace(
            go.Bar(x=df['Date'], y=df['Daily_Return'], name='Daily Return',
                  marker_color=colors, opacity=0.7),
            row=2, col=1
        )

    fig.update_layout(height=600, title_text=f"{commodity_name} Advanced Analysis")
    return fig

def create_correlation_heatmap(df, commodity_name):
    """Create correlation heatmap for numerical columns"""
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numerical_cols) < 2:
        fig = px.imshow([[0]], title="Not enough numerical data for correlation analysis")
        fig.update_layout(height=300)
        return fig

    try:
        correlation_matrix = df[numerical_cols].corr()

        fig = px.imshow(
            correlation_matrix,
            title=f"{commodity_name} Feature Correlation Heatmap",
            color_continuous_scale='RdBu_r',
            aspect="auto"
        )

        fig.update_layout(height=500)
        return fig
    except Exception:
        fig = px.imshow([[0]], title="Error creating correlation heatmap")
        fig.update_layout(height=300)
        return fig

def create_distribution_chart(df, title):
    """Create enhanced distribution chart"""
    if df.empty or 'Close' not in df.columns:
        fig = px.histogram(title=title)
        fig.update_layout(height=400)
        return fig

    fig = px.histogram(df, x='Close', nbins=50, title=title,
                      color_discrete_sequence=['#1f77b4'])
    fig.update_layout(showlegend=False, height=400)
    return fig

def analyze_sentiment(news_data):
    """Analyze sentiment from news data using existing sentiment scores"""
    if news_data.empty or 'sentiment' not in news_data.columns:
        return {"positive": 0, "negative": 0, "neutral": 0, "total": 0, "avg_sentiment": 0}

    try:
        sentiment_scores = news_data['sentiment'].dropna()

        positive = len(sentiment_scores[sentiment_scores > 0.1])
        negative = len(sentiment_scores[sentiment_scores < -0.1])
        neutral = len(sentiment_scores[(sentiment_scores >= -0.1) & (sentiment_scores <= 0.1)])

        avg_sentiment = sentiment_scores.mean() if len(sentiment_scores) > 0 else 0

        return {
            "positive": positive,
            "negative": negative,
            "neutral": neutral,
            "total": len(sentiment_scores),
            "avg_sentiment": avg_sentiment
        }
    except Exception:
        return {"positive": 0, "negative": 0, "neutral": 0, "total": 0, "avg_sentiment": 0}

def get_sentiment_category(score):
    """Convert numerical sentiment score to category"""
    if score > 0.1:
        return "positive"
    elif score < -0.1:
        return "negative"
    else:
        return "neutral"

def get_sentiment_color(score):
    """Get color based on sentiment score"""
    if score > 0.1:
        return "#28a745"  # Green
    elif score < -0.1:
        return "#dc3545"  # Red
    else:
        return "#6c757d"  # Gray

def organize_news_by_date(news_data):
    """
    Organize news data by date.
    Assumes columns: 'date', 'title', 'source' (link), 'sentiment'.
    """
    if news_data.empty:
        return {}

    # Ensure date column is datetime
    if 'date' in news_data.columns:
        news_data['date'] = pd.to_datetime(news_data['date'], errors='coerce')

    news_by_date = {}

    # Sort by date descending
    news_data = news_data.sort_values('date', ascending=False)

    # Iterate through rows
    # Note: Vectorization would be better for processing, but we need a dictionary structure output.
    # Given the likely size of news data (usually small enough for display), iterrows is acceptable here,
    # but we can improve robustness.

    for _, row in news_data.iterrows():
        date_val = row.get('date')
        if pd.isna(date_val):
            date_str = 'Unknown Date'
        else:
            date_str = date_val.strftime('%Y-%m-%d')

        if date_str not in news_by_date:
            news_by_date[date_str] = []

        title = row.get('title', 'No Title')

        # Robust link extraction: check common link column names
        link = ''
        for col_name in ['source', 'link', 'url', 'href']:
            val = row.get(col_name)
            if pd.notna(val) and str(val).strip():
                link = str(val).strip()
                break

        sentiment_score = row.get('sentiment', 0.0)

        # Basic link validation
        if isinstance(link, str) and link and not link.startswith(('http://', 'https://')):
             # If it looks like a url but misses protocol
             if re.search(r'[A-Za-z0-9-]+\.[A-Za-z]{2,6}(/|$)', link):
                 link = "https://" + link.strip()

        news_by_date[date_str].append({
            'title': title,
            'link': link,
            'original_date': str(date_val),
            'sentiment_score': sentiment_score,
            'sentiment_category': get_sentiment_category(sentiment_score)
        })

    return news_by_date
