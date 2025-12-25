import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import io

# Import helper functions
from utils import (
    load_data,
    create_advanced_price_chart,
    create_correlation_heatmap,
    create_distribution_chart,
    analyze_sentiment,
    get_sentiment_color,
    organize_news_by_date
)

warnings.filterwarnings('ignore')

# Configuration
st.set_page_config(
    page_title="Energy Data Analytics Dashboard", 
    layout="wide",
    page_icon="⛽",
    initial_sidebar_state="expanded"
)

# Constants and configuration
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

# Custom CSS
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

# Set display options
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
def load_data_cached(file_path):
    """Wrapper to cache data loading"""
    return load_data(file_path)

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
        except Exception:
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

def display_news_with_sentiment(news_data, commodity_name):
    """Display news with sentiment analysis"""
    if news_data.empty:
        st.warning(f"No news data available for {commodity_name}")
        return
    
    sentiment = analyze_sentiment(news_data)
    avg_color = "#28a745" if sentiment['avg_sentiment'] > 0.1 else "#dc3545" if sentiment['avg_sentiment'] < -0.1 else "#000000"

    st.markdown("---")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Corrected metrics and typos
    with col1:
        st.markdown(f"<div class='metric-card'><span style='color: #1f77b4; font-weight: bold;'> 📊 Total News: {sentiment['total']}</span></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-card'><span class='positive-sentiment'>👍 Positive: {sentiment['positive']}</span></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-card'><span class='negative-sentiment'>👎 Negative: {sentiment['negative']}</span></div>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div class='metric-card'><span class='neutral-sentiment'>😐 Neutral: {sentiment['neutral']}</span></div>", unsafe_allow_html=True)
    with col5:
        st.markdown(f"<div class='metric-card'><span style='color: {avg_color}; font-weight: bold;'>📈 Avg Score: {sentiment['avg_sentiment']:.3f}</span></div>", unsafe_allow_html=True)
    
    news_by_date = organize_news_by_date(news_data)
    
    if not news_by_date:
        st.warning("Could not organize news by date")
        return
    
    st.markdown(f"### 📰 Recent {commodity_name} News")
    
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
        
        start_idx = max(0, selected_date_index)
        end_idx = min(len(dates), selected_date_index + 3)
        dates_to_show = dates[start_idx:end_idx]
    else:
        dates_to_show = dates
    
    for date in dates_to_show:
        news_items = news_by_date[date]
        st.markdown(f"<div class='date-header'>📅 {date} ({len(news_items)} news items)</div>", unsafe_allow_html=True)
        
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
    
    if len(dates) > 1:
        st.info(f"📋 Showing {len(dates_to_show)} of {len(dates)} date(s). Use the slider above to navigate through different dates.")

def main():
    with st.sidebar:
        st.header("🔧 Dashboard Controls")
        
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
        
        if show_news:
            st.markdown("---")
            st.subheader("📰 News Display")
            news_limit = st.slider("Maximum news items to show", 5, 50, 20)
            st.info(f"Will show up to {news_limit} news items")
        
        st.markdown("---")
        st.subheader("📊 Data Information")
        st.info(f"Selected: {selected}")
        st.info("All data files are in Excel format (.xlsx)")

    st.markdown(f"<div class='main-header'>🔋 {selected} Market Intelligence Dashboard</div>", unsafe_allow_html=True)
    
    with st.spinner(f"Loading {selected} data..."):
        price_data = load_data_cached(DATA_PATHS[selected]["price"])
        news_data = load_data_cached(DATA_PATHS[selected]["news"])
    
    if price_data.empty:
        st.error(f"❌ Unable to load price data for {selected}")
        st.info("Please check if the Excel file exists and contains valid data")
        return
    
    st.success(f"✅ Successfully loaded {len(price_data)} price records")
    if not news_data.empty:
        st.success(f"✅ Successfully loaded {len(news_data)} news records")
        if 'sentiment' in news_data.columns:
            st.info(f"📊 Using pre-calculated sentiment scores (range: {news_data['sentiment'].min():.3f} to {news_data['sentiment'].max():.3f})")
    
    st.markdown("## 📊 Key Metrics")
    create_metrics_dashboard(price_data, selected)
    
    st.markdown("## 📋 Data Overview")
    with st.expander("View Raw Data", expanded=False):
        st.dataframe(price_data, use_container_width=True)
    
    if show_advanced:
        st.markdown("## 📈 Advanced Analytics")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(create_advanced_price_chart(price_data, selected), use_container_width=True)
        with col2:
            st.plotly_chart(create_distribution_chart(price_data, selected), use_container_width=True)
    
    if show_correlation:
        st.markdown("## 🔗 Correlation Analysis")
        st.plotly_chart(create_correlation_heatmap(price_data, selected), use_container_width=True)
    
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
    
    if show_news:
        display_news_with_sentiment(news_data, selected)
    
    st.markdown("---")
    st.markdown("### 🌐 Data Sources & Information")
    st.markdown("""
    - **Price Data**: Yahoo Finance via yfinance library
    - **News Data**: NewsAPI and web scraping
    - **File Format**: Excel (.xlsx) files
    - **Last Updated**: Refreshed Weekly
    - **Contact**: mehmetkarahanc@gmail.com
    - **GitHub**: [github.com/karahanos]
    - **LinkedIn**: [linkedin.com/in/karahan-cetinkaya]
    
    - **Disclaimer**: This dashboard is for educational purposes only. Not financial advice.
    """)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Developed by Karahan Cetinkaya</p>
        <p>© 2024 Energy Analytics Dashboard - All rights reserved</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()