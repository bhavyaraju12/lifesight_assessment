import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta

# Set page configuration for wide mode
st.set_page_config(
    page_title="Marketing Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Detect current theme and set colors accordingly
def get_theme_colors():
    # Try to detect theme from session state or use a more reliable method
    try:
        # Check if we can access theme from streamlit
        theme_name = st.get_option("theme.base") 
    except:
        theme_name = "light"  # Default fallback
    
    if theme_name == "dark":
        return {
            'primary': '#60A5FA',
            'secondary': '#34D399', 
            'accent': '#F59E0B',
            'success': '#10B981',
            'warning': '#F59E0B',
            'danger': '#EF4444',
            'background': '#0F172A',
            'surface': '#1E293B',
            'card': '#334155',
            'text': '#F8FAFC',
            'text_secondary': '#CBD5E1',
            'border': '#475569',
            'gradient_1': 'linear-gradient(135deg, #3B82F6 0%, #8B5CF6 100%)',
            'gradient_2': 'linear-gradient(135deg, #06B6D4 0%, #3B82F6 100%)',
            'card_shadow': '0 4px 6px -1px rgba(0, 0, 0, 0.3)',
            'hover_shadow': '0 8px 25px -5px rgba(0, 0, 0, 0.4)'
        }
    else:
        return {
            'primary': '#2563EB',
            'secondary': '#0891B2', 
            'accent': '#0D9488',
            'success': '#059669',
            'warning': '#D97706',
            'danger': '#DC2626',
            'background': '#FFFFFF',
            'surface': '#F8FAFC',
            'card': '#FFFFFF',
            'text': '#1F2937',
            'text_secondary': '#6B7280',
            'border': '#E5E7EB',
            'gradient_1': 'linear-gradient(135deg, #3B82F6 0%, #8B5CF6 100%)',
            'gradient_2': 'linear-gradient(135deg, #06B6D4 0%, #3B82F6 100%)',
            'card_shadow': '0 1px 3px 0 rgba(0, 0, 0, 0.1)',
            'hover_shadow': '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
        }

COLORS = get_theme_colors()

# Professional Brand Colors for Platforms
PLATFORM_COLORS = {
    'Facebook': '#1877F2',
    'Google': '#4285F4', 
    'TikTok': '#FF0050'
}

# Set plotly theme based on detected theme
PLOTLY_THEME = 'plotly_dark' if COLORS['background'] == '#0F172A' else 'plotly_white'

def get_custom_css():
    return f"""
<style>
    /* Enhanced Header */
    .dashboard-header {{
        background: {COLORS['gradient_1']};
        padding: 2.5rem 2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: {COLORS['hover_shadow']};
        border: 1px solid {COLORS['border']};
    }}
    
    .dashboard-header h1 {{
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        color: white !important;
    }}
    
    .dashboard-header p {{
        font-size: 1.3rem;
        opacity: 0.9;
        font-weight: 300;
        color: white !important;
    }}

    .dashboard-header small {{
        color: rgba(255, 255, 255, 0.8) !important;
    }}
    
    /* Section Headers */
    .section-header {{
        background: {COLORS['surface']};
        border-left: 4px solid {COLORS['primary']};
        padding: 1.5rem 2rem;
        border-radius: 8px;
        color: {COLORS['text']};
        font-weight: 600;
        font-size: 1.3rem;
        margin: 2rem 0 1.5rem 0;
        box-shadow: {COLORS['card_shadow']};
        border: 1px solid {COLORS['border']};
    }}
    
    /* Sidebar header */
    .sidebar-header {{
        background: {COLORS['gradient_2']};
        padding: 1.5rem;
        border-radius: 12px;
        color: white !important;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: {COLORS['card_shadow']};
    }}

    .sidebar-header h2 {{
        color: white !important;
        margin-bottom: 0.5rem;
    }}

    .sidebar-header p {{
        color: rgba(255, 255, 255, 0.9) !important;
        margin: 0;
    }}
    
    /* Alert message styling */
    .data-source-alert {{
        background: {COLORS['gradient_1']};
        color: white !important;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        text-align: center;
        font-weight: 500;
        box-shadow: {COLORS['card_shadow']};
    }}
    
    /* Executive summary footer */
    .executive-summary {{
        text-align: center;
        padding: 2rem;
        background: {COLORS['surface']};
        border-radius: 12px;
        border: 1px solid {COLORS['border']};
        margin-top: 2rem;
        box-shadow: {COLORS['card_shadow']};
    }}

    .executive-summary h3 {{
        color: {COLORS['text']};
        margin-bottom: 1rem;
    }}

    .executive-summary p {{
        color: {COLORS['text_secondary']};
    }}

    /* Custom styling for metric values in executive summary */
    .executive-summary .metric-value {{
        font-size: 1.5rem;
        font-weight: bold;
    }}

    .executive-summary .metric-label {{
        font-size: 0.9rem;
        color: {COLORS['text_secondary']};
    }}
</style>
"""

# Apply custom CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Data Source Alert
st.markdown("""
<div class="data-source-alert">
    📊 <strong>DEMO MODE:</strong> This dashboard uses simulated marketing data to demonstrate analytics capabilities. 
    In production, this would connect to live Facebook Ads API, Google Ads API, and TikTok for Business API.
</div>
""", unsafe_allow_html=True)

# Enhanced data generation function
@st.cache_data
def generate_comprehensive_demo_data():
    """
    Generate realistic marketing and business data that tells a story
    """
    # Date range for 120 days
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=120)
    date_range = pd.date_range(start_date, end_date, freq='D')
    
    # Business seasonality patterns
    base_revenue = 15000
    weekend_boost = 1.2
    month_trend = np.sin(np.arange(len(date_range)) * 2 * np.pi / 30) * 0.3 + 1
    
    # Generate business data
    business_data = []
    for i, date in enumerate(date_range):
        day_of_week = date.weekday()
        is_weekend = day_of_week >= 5
        
        # Revenue with seasonality
        daily_revenue = base_revenue * month_trend[i]
        if is_weekend:
            daily_revenue *= weekend_boost
        
        # Add some noise
        daily_revenue *= np.random.normal(1, 0.15)
        daily_revenue = max(daily_revenue, 5000)  # Floor
        
        orders = int(daily_revenue / np.random.normal(85, 10))  # AOV around $85
        orders = max(orders, 20)
        
        business_data.append({
            'date': date,
            'total_revenue': round(daily_revenue, 2),
            'orders': orders,
            'new_orders': int(orders * np.random.uniform(0.3, 0.7)),
            'new_customers': int(orders * np.random.uniform(0.25, 0.6)),
            'cogs': round(daily_revenue * 0.35, 2),  # 35% COGS
            'gross_profit': round(daily_revenue * 0.65, 2)
        })
    
    business_df = pd.DataFrame(business_data)
    
    # Generate marketing data
    platforms = ['Facebook', 'Google', 'TikTok']
    states = ['CA', 'TX', 'FL', 'NY', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI']
    tactics = ['Search', 'Display', 'Video', 'Shopping', 'Social']
    
    # Platform performance characteristics
    platform_characteristics = {
        'Facebook': {'base_ctr': 0.018, 'base_cpc': 0.85, 'roas_range': (2.8, 4.2)},
        'Google': {'base_ctr': 0.035, 'base_cpc': 1.25, 'roas_range': (3.2, 5.1)},
        'TikTok': {'base_ctr': 0.022, 'base_cpc': 0.65, 'roas_range': (2.1, 3.8)}
    }
    
    marketing_data = []
    for date in date_range:
        day_revenue = business_df[business_df['date'] == date]['total_revenue'].iloc[0]
        
        # Each platform gets 2-4 campaigns per day
        for platform in platforms:
            num_campaigns = np.random.randint(2, 5)
            platform_spend = day_revenue * np.random.uniform(0.12, 0.25) / len(platforms)
            
            for campaign_idx in range(num_campaigns):
                char = platform_characteristics[platform]
                
                # Campaign-level data
                campaign_spend = platform_spend / num_campaigns * np.random.uniform(0.5, 1.5)
                impressions = int(campaign_spend / char['base_cpc'] / char['base_ctr'])
                clicks = int(impressions * char['base_ctr'] * np.random.uniform(0.8, 1.2))
                
                # ROAS with some campaigns performing better/worse
                roas = np.random.uniform(char['roas_range'][0], char['roas_range'][1])
                attributed_revenue = campaign_spend * roas
                
                marketing_data.append({
                    'date': date,
                    'platform': platform,
                    'state': np.random.choice(states),
                    'tactic': np.random.choice(tactics),
                    'campaign': f"{platform}_{np.random.choice(tactics)}_{campaign_idx+1:02d}",
                    'spend': round(campaign_spend, 2),
                    'impression': impressions,
                    'clicks': clicks,
                    'attributed_revenue': round(attributed_revenue, 2)
                })
    
    marketing_df = pd.DataFrame(marketing_data)
    
    return business_df, marketing_df

def load_and_prepare_data():
    """
    Loads, cleans, merges, and calculates derived metrics from the marketing and business data.
    """
    business_df, marketing_df = generate_comprehensive_demo_data()
    
    # Aggregate marketing data by date to get daily totals
    marketing_agg = marketing_df.groupby('date').agg({
        'spend': 'sum',
        'impression': 'sum',
        'clicks': 'sum',
        'attributed_revenue': 'sum'
    }).reset_index()
    
    # Rename aggregated columns for clarity
    marketing_agg.rename(columns={
        'spend': 'total_spend', 
        'impression': 'total_impressions',
        'attributed_revenue': 'total_attributed_revenue'
        }, inplace=True)

    # Join the aggregated marketing data with the daily business data
    df = pd.merge(business_df, marketing_agg, on='date', how='left')
    df.fillna(0, inplace=True)

    # Create Derived Metrics
    df['roas'] = df.apply(lambda row: row['total_attributed_revenue'] / row['total_spend'] if row['total_spend'] > 0 else 0, axis=1)
    df['ctr'] = df.apply(lambda row: row['clicks'] / row['total_impressions'] if row['total_impressions'] > 0 else 0, axis=1)
    df['cpc'] = df.apply(lambda row: row['total_spend'] / row['clicks'] if row['clicks'] > 0 else 0, axis=1)
    df['mer'] = df.apply(lambda row: row['total_revenue'] / row['total_spend'] if row['total_spend'] > 0 else 0, axis=1)
    df['cpa'] = df.apply(lambda row: row['total_spend'] / row['orders'] if row['orders'] > 0 else 0, axis=1)

    return df, marketing_df

# Load the data
df_daily, marketing_df_raw = load_and_prepare_data()

# Dashboard Header
st.markdown(f"""
<div class="dashboard-header">
    <h1>📊 Marketing Intelligence Dashboard</h1>
    <p>Enterprise-grade analytics for data-driven marketing decisions</p>
    <small>Real-time insights • Performance optimization • Strategic intelligence</small>
</div>
""", unsafe_allow_html=True)

# Enhanced Sidebar
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <h2>🎛️ Dashboard Controls</h2>
        <p>Configure your analytics view</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Date Filter
    min_date = df_daily['date'].min().date()
    max_date = df_daily['date'].max().date()

    start_date, end_date = st.date_input(
        "📅 Date Range Analysis",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        help="Select the period for your analysis"
    )
    
    st.markdown("---")
    st.markdown("### 🎯 Granular Filters")

    # Get unique values for filters from the RAW marketing dataframe
    platforms = marketing_df_raw['platform'].unique()
    states = marketing_df_raw['state'].unique()
    tactics = marketing_df_raw['tactic'].unique()

    # Create multiselect widgets
    selected_platforms = st.multiselect("Platforms", options=platforms, default=platforms)
    selected_states = st.multiselect("States", options=states, default=states[:5])  # Default to first 5 states
    selected_tactics = st.multiselect("Tactics", options=tactics, default=tactics)

    # Dataset Overview
    st.markdown("---")
    st.markdown("### 📊 Dataset Overview")
    total_days = len(df_daily)
    st.info(f"**📈 Total Data Points:** {total_days} days")
    st.info(f"**📅 Coverage Period:** {min_date.strftime('%b %d, %Y')} - {max_date.strftime('%b %d, %Y')}")
    
    selected_days = len(pd.date_range(start_date, end_date))
    st.success(f"**🎯 Selected Period:** {selected_days} days")
    
    # Performance Summary in Sidebar
    st.markdown("---")
    st.markdown("### ⚡ Quick Stats")
    total_revenue_sidebar = df_daily['total_revenue'].sum()
    total_spend_sidebar = df_daily['total_spend'].sum()
    overall_mer_sidebar = total_revenue_sidebar / total_spend_sidebar if total_spend_sidebar > 0 else 0
    
    st.metric("Total Revenue", f"${total_revenue_sidebar/1000:.0f}K")
    st.metric("Marketing Spend", f"${total_spend_sidebar/1000:.0f}K")
    st.metric("Overall MER", f"{overall_mer_sidebar:.2f}x")

# Filter data based on all sidebar selections
date_mask = (df_daily['date'] >= pd.to_datetime(start_date)) & (df_daily['date'] <= pd.to_datetime(end_date))
filtered_df = df_daily[date_mask]

# Update the filtering logic for the detailed marketing dataframe to include ALL filters
filtered_marketing_df = marketing_df_raw[
    (marketing_df_raw['date'] >= pd.to_datetime(start_date)) &
    (marketing_df_raw['date'] <= pd.to_datetime(end_date)) &
    (marketing_df_raw['platform'].isin(selected_platforms)) &
    (marketing_df_raw['state'].isin(selected_states)) &
    (marketing_df_raw['tactic'].isin(selected_tactics))
]

# KPI Section
st.markdown('<div class="section-header">💎 Executive Performance Summary</div>', unsafe_allow_html=True)

# Calculate KPIs
total_revenue = filtered_df['total_revenue'].sum()
total_spend = filtered_df['total_spend'].sum()
total_orders = filtered_df['orders'].sum()
overall_mer = total_revenue / total_spend if total_spend > 0 else 0
overall_cpa = total_spend / total_orders if total_orders > 0 else 0

# Enhanced metrics display
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        label="💰 Total Revenue",
        value=f"${total_revenue:,.0f}",
        delta=f"+{total_revenue/len(filtered_df):,.0f}/day avg" if len(filtered_df) > 0 else None,
        help="Total revenue generated during the selected period"
    )

with col2:
    spend_percentage = (total_spend/total_revenue*100) if total_revenue > 0 else 0
    st.metric(
        label="🚀 Marketing Spend",
        value=f"${total_spend:,.0f}",
        delta=f"{spend_percentage:.1f}% of revenue",
        delta_color="inverse",
        help="Total marketing investment across all platforms"
    )

with col3:
    avg_order_value = total_revenue / total_orders if total_orders > 0 else 0
    st.metric(
        label="🛒 Total Orders",
        value=f"{total_orders:,}",
        delta=f"${avg_order_value:.0f} AOV",
        help="Total orders and average order value"
    )

with col4:
    st.metric(
        label="📈 Marketing Efficiency Ratio",
        value=f"{overall_mer:.2f}x",
        delta="Excellent" if overall_mer >= 4 else "Good" if overall_mer >= 2 else "Needs Improvement",
        delta_color="normal" if overall_mer >= 2 else "inverse",
        help="Total revenue divided by marketing spend"
    )

with col5:
    st.metric(
        label="🎯 Cost Per Acquisition",
        value=f"${overall_cpa:.2f}",
        delta="Low" if overall_cpa <= 50 else "Moderate" if overall_cpa <= 100 else "High",
        delta_color="normal" if overall_cpa <= 100 else "inverse",
        help="Average cost to acquire one customer"
    )

# Performance Trends Section
st.markdown('<div class="section-header">📈 Performance Trends & Analytics</div>', unsafe_allow_html=True)

# Create sophisticated multi-chart layout
fig_performance = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Revenue vs Spend Correlation', 'Efficiency Metrics Trend', 'Daily Orders & AOV', 'ROAS Performance'),
    specs=[[{"secondary_y": True}, {"secondary_y": False}],
           [{"secondary_y": True}, {"secondary_y": False}]],
    vertical_spacing=0.12,
    horizontal_spacing=0.08
)

# Chart 1: Revenue vs Spend
fig_performance.add_trace(
    go.Scatter(x=filtered_df['date'], y=filtered_df['total_revenue'], 
               name='Revenue', line=dict(color=COLORS['success'], width=3),
               hovertemplate='<b>Revenue</b><br>Date: %{x}<br>Amount: $%{y:,.0f}<extra></extra>'),
    row=1, col=1, secondary_y=False
)
fig_performance.add_trace(
    go.Scatter(x=filtered_df['date'], y=filtered_df['total_spend'], 
               name='Marketing Spend', line=dict(color=COLORS['danger'], width=3, dash='dash'),
               hovertemplate='<b>Marketing Spend</b><br>Date: %{x}<br>Amount: $%{y:,.0f}<extra></extra>'),
    row=1, col=1, secondary_y=True
)

# Chart 2: Efficiency Metrics
fig_performance.add_trace(
    go.Scatter(x=filtered_df['date'], y=filtered_df['mer'], 
               name='MER', line=dict(color=COLORS['primary'], width=2),
               hovertemplate='<b>MER</b><br>Date: %{x}<br>Ratio: %{y:.2f}x<extra></extra>'),
    row=1, col=2
)
fig_performance.add_trace(
    go.Scatter(x=filtered_df['date'], y=filtered_df['roas'], 
               name='ROAS', line=dict(color=COLORS['secondary'], width=2),
               hovertemplate='<b>ROAS</b><br>Date: %{x}<br>Ratio: %{y:.2f}x<extra></extra>'),
    row=1, col=2
)

# Chart 3: Orders & AOV
aov = filtered_df['total_revenue'] / filtered_df['orders']
aov = aov.fillna(0)

fig_performance.add_trace(
    go.Bar(x=filtered_df['date'], y=filtered_df['orders'], 
           name='Daily Orders', marker_color=COLORS['accent'], opacity=0.7,
           hovertemplate='<b>Orders</b><br>Date: %{x}<br>Count: %{y}<extra></extra>'),
    row=2, col=1, secondary_y=False
)
fig_performance.add_trace(
    go.Scatter(x=filtered_df['date'], y=aov, 
               name='AOV', line=dict(color=COLORS['warning'], width=3),
               hovertemplate='<b>AOV</b><br>Date: %{x}<br>Value: $%{y:.0f}<extra></extra>'),
    row=2, col=1, secondary_y=True
)

# Chart 4: ROAS Distribution
fig_performance.add_trace(
    go.Histogram(x=filtered_df['roas'], name='ROAS Distribution', 
                 marker_color=COLORS['primary'], opacity=0.7, nbinsx=20,
                 hovertemplate='<b>ROAS Range</b><br>%{x:.1f}x - %{x:.1f}x<br>Frequency: %{y}<extra></extra>'),
    row=2, col=2
)

# Update layout with appropriate theme
fig_performance.update_layout(
    height=700,
    showlegend=True,
    title_text="📊 Comprehensive Performance Analytics",
    title_x=0.5,
    template=PLOTLY_THEME,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

# Update axes
fig_performance.update_yaxes(title_text="Revenue ($)", row=1, col=1, secondary_y=False)
fig_performance.update_yaxes(title_text="Spend ($)", row=1, col=1, secondary_y=True)
fig_performance.update_yaxes(title_text="Efficiency Ratio", row=1, col=2)
fig_performance.update_yaxes(title_text="Orders", row=2, col=1, secondary_y=False)
fig_performance.update_yaxes(title_text="AOV ($)", row=2, col=1, secondary_y=True)
fig_performance.update_yaxes(title_text="Frequency", row=2, col=2)
fig_performance.update_xaxes(title_text="ROAS", row=2, col=2)

st.plotly_chart(fig_performance, use_container_width=True)

# Platform Performance Analysis
st.markdown('<div class="section-header">🏆 Multi-Platform Performance Intelligence</div>', unsafe_allow_html=True)

# Enhanced platform aggregation
platform_agg = filtered_marketing_df.groupby('platform').agg({
    'spend': 'sum',
    'attributed_revenue': 'sum',
    'clicks': 'sum',
    'impression': 'sum'
}).reset_index()

platform_agg['roas'] = platform_agg.apply(lambda row: row['attributed_revenue'] / row['spend'] if row['spend'] > 0 else 0, axis=1)
platform_agg['ctr'] = platform_agg.apply(lambda row: row['clicks'] / row['impression'] if row['impression'] > 0 else 0, axis=1)
platform_agg['cpc'] = platform_agg.apply(lambda row: row['spend'] / row['clicks'] if row['clicks'] > 0 else 0, axis=1)

# Sort platforms by performance metrics
spend_sorted = platform_agg.sort_values('spend', ascending=False)
roas_sorted = platform_agg.sort_values('roas', ascending=False)
ctr_sorted = platform_agg.sort_values('ctr', ascending=False)

# Create enhanced platform comparison charts
col1, col2, col3 = st.columns(3)

with col1:
    fig_spend = px.bar(
        spend_sorted, x='platform', y='spend',
        title='💸 Investment by Platform',
        text='spend',
        color='platform',
        color_discrete_map=PLATFORM_COLORS,
        template=PLOTLY_THEME
    )
    fig_spend.update_traces(
        texttemplate='$%{text:,.0f}', 
        textposition='outside',
        textfont_size=12
    )
    fig_spend.update_layout(
        showlegend=False, 
        yaxis_title="Investment ($)"
    )
    st.plotly_chart(fig_spend, use_container_width=True)

with col2:
    fig_roas = px.bar(
        roas_sorted, x='platform', y='roas',
        title='📈 Return on Ad Spend',
        text='roas',
        color='platform',
        color_discrete_map=PLATFORM_COLORS,
        template=PLOTLY_THEME
    )
    fig_roas.update_traces(
        texttemplate='%{text:.2f}x', 
        textposition='outside',
        textfont_size=12
    )
    fig_roas.update_layout(
        showlegend=False, 
        yaxis_title="ROAS Multiplier"
    )
    st.plotly_chart(fig_roas, use_container_width=True)

with col3:
    fig_ctr = px.bar(
        ctr_sorted, x='platform', y='ctr',
        title='🎯 Click-Through Rate',
        text='ctr',
        color='platform',
        color_discrete_map=PLATFORM_COLORS,
        template=PLOTLY_THEME
    )
    fig_ctr.update_traces(
        texttemplate='%{text:.2%}', 
        textposition='outside',
        textfont_size=12
    )
    fig_ctr.update_layout(
        showlegend=False, 
        yaxis_title="CTR (%)"
    )
    st.plotly_chart(fig_ctr, use_container_width=True)

# Advanced Analytics Section
st.markdown('<div class="section-header">🧠 Advanced Analytics & Cohort Analysis</div>', unsafe_allow_html=True)

# Create cohort analysis and attribution modeling
col1, col2 = st.columns(2)

with col1:
    # Weekly performance heatmap
    filtered_df_copy = filtered_df.copy()
    filtered_df_copy['week'] = filtered_df_copy['date'].dt.isocalendar().week
    filtered_df_copy['day_name'] = filtered_df_copy['date'].dt.day_name()
    
    weekly_heatmap = filtered_df_copy.groupby(['week', 'day_name'])['mer'].mean().reset_index()
    weekly_pivot = weekly_heatmap.pivot(index='week', columns='day_name', values='mer')
    
    # Reorder columns to show Monday-Sunday
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekly_pivot = weekly_pivot.reindex(columns=[day for day in day_order if day in weekly_pivot.columns])
    
    fig_heatmap = px.imshow(
        weekly_pivot.values,
        x=weekly_pivot.columns,
        y=[f"Week {int(w)}" for w in weekly_pivot.index],
        color_continuous_scale='RdYlGn',
        title='📅 Weekly MER Performance Heatmap',
        aspect='auto',
        template=PLOTLY_THEME
    )
    fig_heatmap.update_layout(height=400)
    st.plotly_chart(fig_heatmap, use_container_width=True)

with col2:
    # Channel attribution analysis
    if len(filtered_marketing_df) > 0:
        channel_attribution = filtered_marketing_df.groupby(['platform', 'tactic']).agg({
            'attributed_revenue': 'sum',
            'spend': 'sum'
        }).reset_index()
        
        channel_attribution['roas'] = channel_attribution['attributed_revenue'] / channel_attribution['spend']
        channel_attribution['channel'] = channel_attribution['platform'] + ' - ' + channel_attribution['tactic']
        
        fig_attribution = px.scatter(
            channel_attribution,
            x='spend',
            y='attributed_revenue',
            size='roas',
            color='platform',
            hover_data=['tactic', 'roas'],
            title='🎯 Channel Attribution & Performance',
            color_discrete_map=PLATFORM_COLORS,
            template=PLOTLY_THEME
        )
        fig_attribution.update_layout(height=400)
        st.plotly_chart(fig_attribution, use_container_width=True)

# Platform Performance Summary Table
st.markdown("### 📋 Platform Performance Executive Summary")

platform_summary = platform_agg.copy()
platform_summary['efficiency_score'] = (platform_summary['roas'] * platform_summary['ctr'] * 100).round(2)
platform_summary['spend_formatted'] = platform_summary['spend'].apply(lambda x: f"${x:,.0f}")
platform_summary['attributed_revenue_formatted'] = platform_summary['attributed_revenue'].apply(lambda x: f"${x:,.0f}")
platform_summary['roas_formatted'] = platform_summary['roas'].apply(lambda x: f"{x:.2f}x")
platform_summary['ctr_formatted'] = platform_summary['ctr'].apply(lambda x: f"{x:.2%}")
platform_summary['cpc_formatted'] = platform_summary['cpc'].apply(lambda x: f"${x:.2f}")

display_columns = ['platform', 'spend_formatted', 'attributed_revenue_formatted', 'roas_formatted', 'ctr_formatted', 'cpc_formatted', 'efficiency_score']
display_df = platform_summary[display_columns].copy()
display_df.columns = ['Platform', 'Investment', 'Revenue', 'ROAS', 'CTR', 'CPC', 'Efficiency Score']

# Sort by efficiency score for better insights
display_df = display_df.sort_values('Efficiency Score', ascending=False)

st.dataframe(display_df, use_container_width=True, hide_index=True)

# Predictive Analytics Section
st.markdown('<div class="section-header">🔮 Predictive Analytics & Forecasting</div>', unsafe_allow_html=True)

# Simple forecasting based on trends
if len(filtered_df) >= 7:
    # Calculate 7-day moving averages for trend analysis
    filtered_df_forecast = filtered_df.copy().sort_values('date')
    filtered_df_forecast['revenue_ma7'] = filtered_df_forecast['total_revenue'].rolling(7).mean()
    filtered_df_forecast['spend_ma7'] = filtered_df_forecast['total_spend'].rolling(7).mean()
    filtered_df_forecast['mer_ma7'] = filtered_df_forecast['mer'].rolling(7).mean()
    
    # Simple linear projection for next 7 days
    last_7_revenue = filtered_df_forecast['total_revenue'].tail(7).values
    last_7_spend = filtered_df_forecast['total_spend'].tail(7).values
    
    # Calculate trends
    revenue_trend = (last_7_revenue[-1] - last_7_revenue[0]) / 6 if len(last_7_revenue) > 1 else 0
    spend_trend = (last_7_spend[-1] - last_7_spend[0]) / 6 if len(last_7_spend) > 1 else 0
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        projected_revenue = last_7_revenue[-1] + (revenue_trend * 7)
        st.metric(
            "📈 7-Day Revenue Projection",
            f"${projected_revenue:,.0f}",
            delta=f"{revenue_trend:+.0f}/day trend"
        )
    
    with col2:
        projected_spend = last_7_spend[-1] + (spend_trend * 7)
        st.metric(
            "💸 7-Day Spend Projection", 
            f"${projected_spend:,.0f}",
            delta=f"{spend_trend:+.0f}/day trend"
        )
    
    with col3:
        projected_mer = projected_revenue / projected_spend if projected_spend > 0 else 0
        current_mer = filtered_df['mer'].tail(7).mean()
        mer_change = projected_mer - current_mer
        st.metric(
            "⚡ Projected MER",
            f"{projected_mer:.2f}x",
            delta=f"{mer_change:+.2f}x change"
        )

# Key Insights & Recommendations
st.markdown('<div class="section-header">💡 AI-Powered Insights & Recommendations</div>', unsafe_allow_html=True)

insights_col1, insights_col2 = st.columns(2)

with insights_col1:
    st.markdown("### 🔍 Performance Insights")
    
    # Calculate insights
    if len(platform_agg) > 0:
        best_platform = platform_agg.loc[platform_agg['roas'].idxmax(), 'platform']
        best_roas = platform_agg.loc[platform_agg['roas'].idxmax(), 'roas']
        
        worst_platform = platform_agg.loc[platform_agg['roas'].idxmin(), 'platform'] 
        worst_roas = platform_agg.loc[platform_agg['roas'].idxmin(), 'roas']
    else:
        best_platform = "N/A"
        best_roas = 0
        worst_platform = "N/A" 
        worst_roas = 0
    
    avg_daily_revenue = total_revenue / len(filtered_df) if len(filtered_df) > 0 else 0
    
    # Calculate week-over-week growth
    if len(filtered_df) >= 14:
        recent_week_revenue = filtered_df['total_revenue'].tail(7).mean()
        previous_week_revenue = filtered_df['total_revenue'].iloc[-14:-7].mean()
        revenue_growth = ((recent_week_revenue - previous_week_revenue) / previous_week_revenue * 100) if previous_week_revenue > 0 else 0
    else:
        revenue_growth = 0
    
    st.info(f"""
    **🏆 Top Performing Platform:** {best_platform} ({best_roas:.2f}x ROAS)  
    **⚠️ Underperforming Platform:** {worst_platform} ({worst_roas:.2f}x ROAS)  
    **📈 Average Daily Revenue:** ${avg_daily_revenue:,.0f}  
    **📊 Week-over-Week Growth:** {revenue_growth:+.1f}%
    """)
    
    # Performance recommendations
    if overall_mer < 2:
        st.warning("⚠️ **Optimization Opportunity:** MER below 2.0x suggests room for campaign optimization")
    elif overall_mer > 4:
        st.success("✅ **Excellent Performance:** MER above 4.0x indicates highly efficient marketing")
    else:
        st.info("📊 **Good Performance:** MER within healthy range (2-4x)")

with insights_col2:
    st.markdown("### 🎯 Strategic Recommendations")
    
    # Budget allocation recommendations
    if len(platform_agg) > 0:
        total_platform_spend = platform_agg['spend'].sum()
        recommendations = []
        
        platform_agg_sorted = platform_agg.sort_values('roas', ascending=False)
        
        for idx, platform in platform_agg_sorted.iterrows():
            current_share = platform['spend'] / total_platform_spend * 100
            if platform['roas'] > platform_agg['roas'].mean() * 1.2:
                recommendations.append(f"📈 **Scale {platform['platform']}** (Current: {current_share:.0f}%, ROAS: {platform['roas']:.2f}x)")
            elif platform['roas'] < platform_agg['roas'].mean() * 0.8:
                recommendations.append(f"🔍 **Optimize {platform['platform']}** (Current: {current_share:.0f}%, ROAS: {platform['roas']:.2f}x)")
        
        # Add tactical recommendations
        if overall_cpa > 75:
            recommendations.append("💰 **Focus on CPA Optimization** - Current acquisition cost is high")
        
        if len(filtered_df) > 0:
            avg_ctr = filtered_marketing_df['clicks'].sum() / filtered_marketing_df['impression'].sum() if filtered_marketing_df['impression'].sum() > 0 else 0
            if avg_ctr < 0.02:
                recommendations.append("🎯 **Improve Creative Performance** - CTR below industry average")
        
        for rec in recommendations[:4]:  # Show top 4 recommendations
            st.markdown(rec)
    
    # Performance consistency metric
    if len(filtered_df) > 0:
        high_performing_days = len(filtered_df[filtered_df['mer'] >= 3])
        total_days_analyzed = len(filtered_df)
        performance_consistency = high_performing_days / total_days_analyzed * 100
        
        st.metric(
            "🎯 High Performance Days", 
            f"{high_performing_days}/{total_days_analyzed}", 
            f"{performance_consistency:.0f}% consistency"
        )

# Detailed Data Analysis
st.markdown('<div class="section-header">📊 Granular Performance Data & Export</div>', unsafe_allow_html=True)

# Enhanced daily data with performance indicators
filtered_df_display = filtered_df.copy()
filtered_df_display['performance_indicator'] = filtered_df_display['mer'].apply(
    lambda x: '🟢 Excellent' if x >= 4 else '🟡 Good' if x >= 2.5 else '🔴 Optimize'
)

# Calculate additional insights
filtered_df_display['profit_margin'] = ((filtered_df_display['total_revenue'] - filtered_df_display['cogs'] - filtered_df_display['total_spend']) / filtered_df_display['total_revenue'] * 100).round(2)
filtered_df_display['efficiency_index'] = (filtered_df_display['roas'] * filtered_df_display['ctr'] * 1000).round(2)

# Select and reorder columns
column_order = [
    'date', 'performance_indicator', 'total_revenue', 'total_spend', 
    'gross_profit', 'orders', 'mer', 'roas', 'cpa', 'ctr', 'profit_margin', 'efficiency_index'
]

display_data = filtered_df_display[column_order].copy()

# Format the data for display
styled_df = display_data.style.format({
    'total_revenue': '${:,.0f}',
    'gross_profit': '${:,.0f}',
    'total_spend': '${:,.0f}',
    'roas': '{:.2f}x',
    'ctr': '{:.2%}',
    'mer': '{:.2f}x',
    'cpa': '${:.2f}',
    'orders': '{:,}',
    'profit_margin': '{:.1f}%',
    'efficiency_index': '{:.0f}'
}).background_gradient(subset=['mer', 'roas'], cmap='RdYlGn')

st.dataframe(styled_df, use_container_width=True, height=400)

# Export functionality
col1, col2, col3 = st.columns([2, 1, 1])

with col2:
    csv_data = display_data.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv_data,
        file_name=f"marketing_performance_{start_date}_{end_date}.csv",
        mime="text/csv",
        help="Download the current filtered data as CSV"
    )

with col3:
    if st.button("🔄 Refresh Data", help="Regenerate demo data with new patterns"):
        st.cache_data.clear()
        st.rerun()

# Executive Summary Footer
st.markdown("---")
st.markdown(f"""
<div class="executive-summary">
    <h3>📊 Enterprise Marketing Intelligence Platform</h3>
    <p style='font-size: 1.1rem; margin-bottom: 1rem;'>
        Powered by advanced analytics • Real-time data processing • Strategic insights • Predictive modeling
    </p>
    <div style='display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap; margin-top: 1.5rem;'>
        <div style='text-align: center;'>
            <div class='metric-value' style='color: {COLORS['primary']};'>${total_revenue:,.0f}</div>
            <div class='metric-label'>Total Revenue</div>
        </div>
        <div style='text-align: center;'>
            <div class='metric-value' style='color: {COLORS['secondary']};'>{overall_mer:.2f}x</div>
            <div class='metric-label'>Marketing Efficiency</div>
        </div>
        <div style='text-align: center;'>
            <div class='metric-value' style='color: {COLORS['success']};'>{len(filtered_df)}</div>
            <div class='metric-label'>Days Analyzed</div>
        </div>
        <div style='text-align: center;'>
            <div class='metric-value' style='color: {COLORS['warning']};'>{len(platform_agg)}</div>
            <div class='metric-label'>Active Platforms</div>
        </div>
    </div>
    <p style='font-size: 0.9rem; margin-top: 1.5rem; font-style: italic;'>
        Built with Streamlit & Plotly • Optimized for executive decision-making • Ready for production deployment
    </p>
</div>
""", unsafe_allow_html=True)