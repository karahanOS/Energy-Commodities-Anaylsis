import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import datetime
from datetime import timedelta
import warnings
import io
warnings.filterwarnings('ignore')

# Configuration
st.set_page_config(
    page_title="Energy Data Analytics Dashboard", 
    layout="wide",
    page_icon="⛽",
    initial_sidebar_state="expanded"
)

# Constants and configuration - CORRECTED PATHS
DATA_PATHS = {
    "Crude Oil": {
        "price": Path("./data/crude_oil_2025_price.xlsx"),
        "news": Path("./data/crude_oil_sentiment_score_.xlsx")
    },
    "LNG": {
        "price": Path("./data/lng_2025_price.xlsx"),
        "news": Path("./data/lng_sentiment_score_.xlsx")
    }
}

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem !important;
        color: #1f77b4 !important;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .news-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #ff6b6b;
        transition: transform 0.2s ease-in-out;
    }
    .news-card:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .date-header {
        background: linear-gradient(90deg, #1f77b4, #4a90e2);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        font-weight: bold;
    }
    .positive-sentiment { color: #28a745; font-weight: bold; }
    .negative-sentiment { color: #dc3545; font-weight: bold; }
    .neutral-sentiment { color: #6c757d; font-weight: bold; }
    .news-date-badge {
        background-color: #6c757d;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 3px;
        font-size: 0.8rem;
        margin-right: 0.5rem;
    }
    .sentiment-badge {
        padding: 0.2rem 0.5rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
        margin-left: 0.5rem;
    }
    .commodity-selector {
        background: linear-gradient(90deg, #1f77b4, #4a90e2);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Set pandas/numpy options
def set_display_options():
    pd.options.plotting.backend = "plotly"
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.float_format', lambda x: '%.5f' % x)
    pd.set_option('mode.chained_assignment', None)
    np.set_printoptions(suppress=True, precision=5, threshold=10000, 
                       edgeitems=10, linewidth=200, legacy='1.13')

set_display_options()

@st.cache_data(ttl=3600)
def load_data(file_path, file_type="excel"):
    """Load data from file with caching and preprocessing"""
    try:
        if file_type == "excel":
            df = pd.read_excel(file_path)
        else:
            # For CSV files with encoding issues
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                # Try different encodings
                for encoding in ['latin-1', 'iso-8859-1', 'cp1252']:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        st.success(f"Successfully loaded with {encoding} encoding")
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    # If all encodings fail, try without specifying encoding
                    df = pd.read_csv(file_path)
        
        # Data preprocessing
        df = preprocess_data(df)
        return df
        
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info(f"Trying to load with different parameters...")
        
        # Fallback loading methods
        try:
            if file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path, engine='openpyxl')
            else:
                # Try multiple encodings for CSV files
                encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'windows-1252']
                for encoding in encodings:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        break
                    except (UnicodeDecodeError, Exception):
                        continue
                else:
                    df = pd.read_csv(file_path, encoding='latin-1', errors='ignore')
            
            df = preprocess_data(df)
            return df
        except Exception as e2:
            st.error(f"Failed to load data after retry: {e2}")
            return pd.DataFrame()

def preprocess_data(df):
    """Preprocess data for better analysis and visualization"""
    if df.empty:
        return df
        
    df_processed = df.copy()
    
    # Handle datetime columns
    for col in df_processed.columns:
        col_lower = str(col).lower()
        if 'date' in col_lower:
            try:
                df_processed[col] = pd.to_datetime(df_processed[col], errors='coerce')
                # For display purposes, create a string version
                df_processed[f'{col}_display'] = df_processed[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            except Exception as e:
                st.warning(f"Could not parse date column {col}: {e}")
    
    # Calculate additional metrics if we have price data
    if 'Close' in df_processed.columns:
        try:
            df_processed['Daily_Return'] = df_processed['Close'].pct_change() * 100
            df_processed['Price_Change'] = df_processed['Close'].diff()
            
            # Calculate moving averages only if we have enough data
            if len(df_processed) >= 7:
                df_processed['MA_7'] = df_processed['Close'].rolling(window=7).mean()
            if len(df_processed) >= 30:
                df_processed['MA_30'] = df_processed['Close'].rolling(window=30).mean()
            
            if 'Daily_Return' in df_processed.columns and len(df_processed) >= 7:
                df_processed['Volatility'] = df_processed['Daily_Return'].rolling(window=7).std()
        except Exception as e:
            st.warning(f"Could not calculate technical indicators: {e}")
    
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
    
    # Moving averages (if available)
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
    
    # Daily returns (if available)
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
        # Create empty heatmap with message
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
    except Exception as e:
        st.warning(f"Could not create correlation heatmap: {e}")
        fig = px.imshow([[0]], title="Error creating correlation heatmap")
        fig.update_layout(height=300)
        return fig

def create_metrics_dashboard(df, commodity_name):
    """Create a metrics dashboard with key statistics"""
    if df.empty or 'Close' not in df.columns:
        st.warning("No price data available for metrics")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        try:
            current_price = df['Close'].iloc[-1] if len(df) > 0 else 0
            price_change = df['Price_Change'].iloc[-1] if 'Price_Change' in df.columns and len(df) > 1 else 0
            change_percent = (price_change / df['Close'].iloc[-2] * 100) if len(df) > 1 else 0
            
            st.metric(
                label="Current Price",
                value=f"${current_price:.2f}" if current_price else "N/A",
                delta=f"{change_percent:+.2f}%" if len(df) > 1 else None
            )
        except Exception as e:
            st.metric(label="Current Price", value="N/A")
    
    with col2:
        try:
            volatility = df['Volatility'].iloc[-1] if 'Volatility' in df.columns and not pd.isna(df['Volatility'].iloc[-1]) else 0
            st.metric(
                label="7-Day Volatility",
                value=f"{volatility:.2f}%" if volatility else "N/A"
            )
        except:
            st.metric(label="7-Day Volatility", value="N/A")
    
    with col3:
        try:
            avg_price = df['Close'].mean()
            st.metric(
                label="Average Price",
                value=f"${avg_price:.2f}" if not pd.isna(avg_price) else "N/A"
            )
        except:
            st.metric(label="Average Price", value="N/A")
    
    with col4:
        try:
            total_return = ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100) if len(df) > 1 else 0
            st.metric(
                label="Total Return",
                value=f"{total_return:+.2f}%" if len(df) > 1 else "N/A"
            )
        except:
            st.metric(label="Total Return", value="N/A")

def analyze_sentiment(news_data):
    """Analyze sentiment from news data using existing sentiment scores"""
    if news_data.empty or 'sentiment' not in news_data.columns:
        return {"positive": 0, "negative": 0, "neutral": 0, "total": 0, "avg_sentiment": 0}
    
    try:
        # Use the existing sentiment scores from the file
        sentiment_scores = news_data['sentiment'].dropna()
        
        # Classify sentiments based on score ranges
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
    except Exception as e:
        st.warning(f"Sentiment analysis failed: {e}")
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
        return "#28a745"  # Green for positive
    elif score < -0.1:
        return "#dc3545"  # Red for negative
    else:
        return "#6c757d"  # Gray for neutral

def organize_news_by_date(news_data):
    """Organize news data by date for better navigation"""
    if news_data.empty:
        return {}
    
    # Find date and title columns
    date_col, title_col, link_col, sentiment_col = None, None, None, None
    
    for col in news_data.columns:
        col_lower = str(col).lower()
        if 'date' in col_lower:
            date_col = col
        elif 'title' in col_lower or 'headline' in col_lower:
            title_col = col
        elif 'link' in col_lower or 'url' in col_lower:
            link_col = col
        elif 'sentiment' in col_lower:
            sentiment_col = col
    
    # Use fallbacks if specific columns not found
    if date_col is None and len(news_data.columns) > 0:
        date_col = news_data.columns[0]
    if title_col is None and len(news_data.columns) > 1:
        title_col = news_data.columns[1]
    if link_col is None and len(news_data.columns) > 2:
        link_col = news_data.columns[2]
    if sentiment_col is None and 'sentiment' in news_data.columns:
        sentiment_col = 'sentiment'
    
    # Convert dates to proper format and group by date
    news_by_date = {}
    
    for _, row in news_data.iterrows():
        date_str = str(row[date_col]) if date_col else 'Unknown Date'
        title = str(row[title_col]) if title_col else 'No Title'
        link = str(row[link_col]) if link_col else ''
        sentiment_score = float(row[sentiment_col]) if sentiment_col and pd.notna(row[sentiment_col]) else 0
        
        # Try to parse and standardize the date
        try:
            # Remove time part if present and take only date portion
            date_key = date_str.split()[0] if ' ' in date_str else date_str
            # Convert to proper date format
            parsed_date = pd.to_datetime(date_key, errors='coerce')
            if pd.notna(parsed_date):
                date_key = parsed_date.strftime('%Y-%m-%d')
            else:
                date_key = date_key[:10]  # Take first 10 characters for YYYY-MM-DD
        except:
            date_key = date_str[:10] if len(date_str) >= 10 else date_str
        
        if date_key not in news_by_date:
            news_by_date[date_key] = []
        
        news_by_date[date_key].append({
            'title': title,
            'link': link,
            'original_date': date_str,
            'sentiment_score': sentiment_score,
            'sentiment_category': get_sentiment_category(sentiment_score)
        })
    
    # Sort dates in descending order (newest first)
    sorted_dates = sorted(news_by_date.keys(), reverse=True)
    return {date: news_by_date[date] for date in sorted_dates}

def display_news_with_sentiment(news_data, commodity_name):
    """Display news with sentiment analysis and date-based organization"""
    if news_data.empty:
        st.warning(f"No news data available for {commodity_name}")
        return
    
    # Sentiment analysis using existing scores
    sentiment = analyze_sentiment(news_data)
    
    # Sentiment summary
    st.markdown("---")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"<div class='metric-card'>📊 Total News: {sentiment['total']}</div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-card'><span class='positive-sentiment'>👍 Positive: {sentiment['positive']}</span></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-card'><span class='negative-sentiment'>👎 Negative: {sentiment['negative']}</span></div>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div class='metric-card'><span class='neutral-sentiment'>😐 Neutral: {sentiment['neutral']}</span></div>", unsafe_allow_html=True)
    with col5:
        avg_color = "#28a745" if sentiment['avg_sentiment'] > 0.1 else "#dc3545" if sentiment['avg_sentiment'] < -0.1 else "#000000"
        st.markdown(f"<div class='metric-card'>📈 Avg Score: <span style='color: {avg_color}; font-weight: bold;'>{sentiment['avg_sentiment']:.3f}</span></div>", unsafe_allow_html=True)
    
    # Organize news by date
    news_by_date = organize_news_by_date(news_data)
    
    if not news_by_date:
        st.warning("Could not organize news by date")
        return
    
    # News display with date sections
    st.markdown(f"### 📰 Recent {commodity_name} News (Using Pre-calculated Sentiment Scores)")
    
    # Add a slider to navigate through dates
    dates = list(news_by_date.keys())
    if len(dates) > 1:
        st.markdown("#### 📅 Navigate by Date")
        selected_date_index = st.slider(
            "Select date range:",
            min_value=0,
            max_value=len(dates)-1,
            value=0,
            format="Date: {}"
        )
        
        # Display news for the selected date and a few previous dates
        start_idx = max(0, selected_date_index)
        end_idx = min(len(dates), selected_date_index + 3)  # Show 3 days at a time
        
        dates_to_show = dates[start_idx:end_idx]
    else:
        dates_to_show = dates
    
    # Display news organized by date
    for date in dates_to_show:
        news_items = news_by_date[date]
        
        # Date header
        st.markdown(f"<div class='date-header'>📅 {date} ({len(news_items)} news items)</div>", unsafe_allow_html=True)
        
        # News items for this date
        for i, news_item in enumerate(news_items, 1):
            title = news_item['title']
            link = news_item['link']
            original_date = news_item['original_date']
            sentiment_score = news_item['sentiment_score']
            sentiment_category = news_item['sentiment_category']
            
            sentiment_class = f"{sentiment_category}-sentiment"
            sentiment_color = get_sentiment_color(sentiment_score)
            
            st.markdown(f"""
            <div class='news-card'>
                <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                    <span class='news-date-badge'>{i}</span>
                    <span style="font-size: 0.9rem; color: #6c757d;">{original_date}</span>
                    <span class='sentiment-badge' style="background-color: {sentiment_color}; color: white;">
                        {sentiment_score:.3f}
                    </span>
                </div>
                <div style="font-size: 1.1rem; margin-bottom: 0.5rem;">
                    <span class='{sentiment_class}'>{title}</span>
                </div>
                {f"<a href='{link}' target='_blank' style='color: #1f77b4; text-decoration: none;'>🔗 Read full article</a>" if link and link.strip() else ""}
            </div>
            """, unsafe_allow_html=True)
    
    # Show date navigation info
    if len(dates) > 1:
        st.info(f"📋 Showing {len(dates_to_show)} of {len(dates)} date(s). Use the slider above to navigate through different dates.")

def main():
    # Sidebar with enhanced options
    with st.sidebar:
        st.header("🔧 Dashboard Controls")
        
        # Replace option_menu with native Streamlit radio
        st.markdown('<div class="commodity-selector">🔋 Commodity Selection</div>', unsafe_allow_html=True)
        selected = st.radio(
            "Choose Commodity:",
            options=["Crude Oil", "LNG"],
            index=0,
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.subheader("📈 Analysis Options")
        
        show_advanced = st.checkbox("Show Advanced Analytics", value=True)
        show_correlation = st.checkbox("Show Correlation Analysis", value=True)
        show_news = st.checkbox("Show News Analysis", value=True)
        show_price_sentiment = st.checkbox("Show Price vs Sentiment Analysis", value=True)
        
        # News display options
        if show_news:
            st.markdown("---")
            st.subheader("📰 News Display")
            news_limit = st.slider("Maximum news items to show", 5, 50, 20)
            st.info(f"Will show up to {news_limit} news items")
        
        # Data info
        st.markdown("---")
        st.subheader("📊 Data Information")
        st.info(f"Selected: {selected}")
        st.info("All data files are in Excel format (.xlsx)")

    # Main content
    st.markdown(f"<div class='main-header'>🔋 {selected} Market Intelligence Dashboard</div>", unsafe_allow_html=True)
    
    # Load data with progress indicator
    with st.spinner(f"Loading {selected} data..."):
        price_data = load_data(DATA_PATHS[selected]["price"], "excel")
        news_data = load_data(DATA_PATHS[selected]["news"], "excel")
    
    # Data validation
    if price_data.empty:
        st.error(f"❌ Unable to load price data for {selected}")
        st.info("Please check if the Excel file exists and contains valid data")
        return
    
    st.success(f"✅ Successfully loaded {len(price_data)} price records")
    if not news_data.empty:
        st.success(f"✅ Successfully loaded {len(news_data)} news records")
        if 'sentiment' in news_data.columns:
            st.info(f"📊 Using pre-calculated sentiment scores (range: {news_data['sentiment'].min():.3f} to {news_data['sentiment'].max():.3f})")
    
    # Metrics Dashboard
    st.markdown("## 📊 Key Metrics")
    create_metrics_dashboard(price_data, selected)
    
    # Price Data Overview
    st.markdown("## 📋 Data Overview")
    with st.expander("View Raw Data", expanded=False):
        st.dataframe(price_data, use_container_width=True)
    
    # Price vs Sentiment Comparison Chart
    if show_price_sentiment and not news_data.empty and 'sentiment' in news_data.columns:
        st.markdown("## 📈 Price vs Sentiment Analysis")
        st.plotly_chart(
            create_price_sentiment_comparison_chart(price_data, news_data, selected), 
            use_container_width=True
        )
        
        # Add some insights
        with st.expander("💡 Analysis Insights", expanded=False):
            st.markdown("""
            **How to interpret this chart:**
            - **Blue line**: Commodity price movement over time
            - **Orange line**: Average daily sentiment score (right axis)
            - **Marker size/color**: Number of news articles per day
            - **Gray bars**: Daily news count (right axis)
            
            **Look for patterns like:**
            - Do price increases follow positive sentiment spikes?
            - Do negative sentiment periods precede price drops?
            - Is there a correlation between news volume and price volatility?
            """)
    
    # Advanced Charts
    if show_advanced:
        st.markdown("## 📈 Advanced Analytics")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.plotly_chart(create_advanced_price_chart(price_data, selected), 
                          use_container_width=True)
        
        with col2:
            st.plotly_chart(create_distribution_chart(price_data, selected), 
                          use_container_width=True)
    
    # Correlation Analysis
    if show_correlation:
        st.markdown("## 🔗 Correlation Analysis")
        st.plotly_chart(create_correlation_heatmap(price_data, selected), 
                      use_container_width=True)
    
    # Statistical Summary
    st.markdown("## 📊 Statistical Summary")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Descriptive Statistics:**")
        st.dataframe(price_data.describe(), use_container_width=True)
    
    with col2:
        st.write("**Data Information:**")
        buffer = io.StringIO()
        price_data.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
    
    # News Analysis
    if show_news:
        display_news_with_sentiment(news_data, selected)
    
    # Footer
    st.markdown("---")
    st.markdown("### 🌐 Data Sources & Information")
    st.markdown("""
    - **Price Data**: Yahoo Finance via yfinance library
    - **News Data**: NewsAPI and web scraping
    - **File Format**: Excel (.xlsx) files
    - **Last Updated**: Automatically refreshed hourly
    """)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Developed by Karahan Cetinkaya</p>
        <p>© 2024 Energy Analytics Dashboard - All rights reserved</p>
    </div>
    """, unsafe_allow_html=True)

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

if __name__ == "__main__":
    main()