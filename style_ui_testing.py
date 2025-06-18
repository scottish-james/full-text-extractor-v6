import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Configure page
st.set_page_config(
    page_title="Analytics Dashboard",
    page_icon="●",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Eduard Bodak inspired CSS
st.markdown("""
<style>
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Hide Streamlit branding */
    .stApp > header {
        background-color: transparent;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Main container styling */
    .main .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* Root variables - Bodak inspired palette */
    :root {
        --yellow: #FFE32E;
        --black: #000000;
        --white: #FFFFFF;
        --grey-100: #F5F5F5;
        --grey-200: #E8E8E8;
        --grey-800: #2A2A2A;
        --beige: #F2E7D6;
        --light-purple: #E6E1F0;
    }

    /* Typography system */
    .hero-title {
        font-family: 'Inter', sans-serif;
        font-size: 4.5rem;
        font-weight: 300;
        color: var(--black);
        line-height: 1.1;
        margin: 0;
        letter-spacing: -0.02em;
    }

    .hero-subtitle {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        font-weight: 400;
        color: var(--grey-800);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 1rem;
        margin-bottom: 3rem;
    }

    .section-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        font-weight: 500;
        color: var(--grey-800);
        text-transform: uppercase;
        letter-spacing: 0.15em;
        margin-bottom: 0.5rem;
    }

    .metric-title {
        font-family: 'Inter', sans-serif;
        font-size: 2.5rem;
        font-weight: 600;
        color: var(--black);
        line-height: 1;
        margin: 0;
    }

    .metric-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        font-weight: 400;
        color: var(--grey-800);
        margin-top: 0.25rem;
    }

    /* Card system */
    .minimal-card {
        background: var(--white);
        border: 1px solid var(--grey-200);
        border-radius: 8px;
        padding: 2rem;
        margin-bottom: 2rem;
        transition: all 0.2s ease;
    }

    .minimal-card:hover {
        border-color: var(--yellow);
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
    }

    .accent-card {
        background: var(--yellow);
        border: 1px solid var(--black);
        border-radius: 8px;
        padding: 2rem;
        margin-bottom: 2rem;
        color: var(--black);
    }

    .subtle-card {
        background: var(--beige);
        border: 1px solid rgba(0,0,0,0.1);
        border-radius: 8px;
        padding: 2rem;
        margin-bottom: 2rem;
    }

    /* Button styling */
    .stButton > button {
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        background: var(--black) !important;
        color: var(--white) !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.75rem 2rem !important;
        font-size: 0.9rem !important;
        transition: all 0.2s ease !important;
        letter-spacing: 0.01em !important;
    }

    .stButton > button:hover {
        background: var(--grey-800) !important;
        transform: translateY(-1px);
    }

    /* Select box styling */
    .stSelectbox > div > div {
        border-color: var(--grey-200) !important;
        border-radius: 6px !important;
    }

    .stSelectbox > div > div:focus-within {
        border-color: var(--yellow) !important;
        box-shadow: 0 0 0 2px rgba(255, 227, 46, 0.2) !important;
    }

    /* Metrics override */
    [data-testid="metric-container"] {
        background: var(--white);
        border: 1px solid var(--grey-200);
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: none;
    }

    [data-testid="metric-container"]:hover {
        border-color: var(--yellow);
    }

    /* Sidebar */
    .css-1d391kg {
        background: var(--grey-100);
    }

    /* Abstract decorative elements */
    .abstract-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--yellow);
        display: inline-block;
        margin-right: 1rem;
    }

    .abstract-line {
        width: 60px;
        height: 2px;
        background: var(--black);
        margin: 2rem 0;
    }

    /* Data table styling */
    .dataframe {
        border: 1px solid var(--grey-200) !important;
        border-radius: 8px !important;
        font-family: 'JetBrains Mono', monospace !important;
    }

    .dataframe th {
        background: var(--grey-100) !important;
        color: var(--black) !important;
        font-weight: 500 !important;
        border-bottom: 1px solid var(--grey-200) !important;
        font-size: 0.8rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }

    .dataframe td {
        border-bottom: 1px solid var(--grey-200) !important;
        font-size: 0.85rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Header section
st.markdown("""
<div class="minimal-card">
    <div class="abstract-dot"></div>
    <h1 class="hero-title">Analytics<br>Platform</h1>
    <p class="hero-subtitle">Contemporary data insights with precision</p>
</div>
""", unsafe_allow_html=True)

# Create elegant metrics section
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<p class="section-label">Revenue</p>', unsafe_allow_html=True)
    st.metric("", "£245.6K", "12.3%")

with col2:
    st.markdown('<p class="section-label">Users</p>', unsafe_allow_html=True)
    st.metric("", "8,924", "5.7%")

with col3:
    st.markdown('<p class="section-label">Conversion</p>', unsafe_allow_html=True)
    st.metric("", "3.8%", "-0.2%")

with col4:
    st.markdown('<p class="section-label">Retention</p>', unsafe_allow_html=True)
    st.metric("", "67.2%", "2.1%")

# Abstract divider
st.markdown('<div class="abstract-line"></div>', unsafe_allow_html=True)

# Main content area
col_left, col_right = st.columns([2, 1])

with col_left:
    st.markdown("""
    <div class="minimal-card">
        <p class="section-label">Performance Overview</p>
    </div>
    """, unsafe_allow_html=True)

    # Create clean, minimal chart
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    data = {
        'Date': dates,
        'Revenue': np.cumsum(np.random.normal(1000, 200, 30)) + 50000,
        'Users': np.cumsum(np.random.normal(50, 10, 30)) + 5000
    }
    df = pd.DataFrame(data)

    # Plotly chart with Bodak-inspired styling
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Revenue'],
        mode='lines',
        name='Revenue',
        line=dict(color='#000000', width=2),
        fill='tonexty',
        fillcolor='rgba(255, 227, 46, 0.1)'
    ))

    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_family='Inter',
        font_color='#000000',
        showlegend=False,
        margin=dict(l=0, r=0, t=20, b=0),
        xaxis=dict(
            showgrid=False,
            showline=True,
            linecolor='#E8E8E8',
            linewidth=1
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#F5F5F5',
            showline=False,
            tickformat=',.0f'
        ),
        height=300
    )

    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.markdown("""
    <div class="accent-card">
        <p class="section-label" style="color: black;">Key Insights</p>
        <div class="metric-title">↗ 15.7%</div>
        <p class="metric-subtitle" style="color: black;">Growth this quarter</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="subtle-card">
        <p class="section-label">Target Progress</p>
        <div class="metric-title">67%</div>
        <p class="metric-subtitle">of annual goal achieved</p>
    </div>
    """, unsafe_allow_html=True)

# Interactive section
st.markdown('<div class="abstract-line"></div>', unsafe_allow_html=True)

st.markdown("""
<div class="minimal-card">
    <p class="section-label">Data Explorer</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 2])

with col1:
    view_type = st.selectbox(
        "Select view",
        ["Overview", "Detailed", "Trends", "Comparative"],
        index=0
    )

    time_range = st.selectbox(
        "Time period",
        ["Last 7 days", "Last 30 days", "Last quarter", "Year to date"],
        index=1
    )

    if st.button("Generate Report"):
        st.success("Report generated successfully")

with col2:
    # Sample data table with clean styling
    sample_data = pd.DataFrame({
        'Metric': ['Sessions', 'Page Views', 'Bounce Rate', 'Avg Duration'],
        'Current': ['12.4K', '28.7K', '42.3%', '2m 34s'],
        'Previous': ['11.8K', '26.1K', '45.1%', '2m 18s'],
        'Change': ['+5.1%', '+10.0%', '-2.8%', '+11.6%']
    })

    st.dataframe(sample_data, hide_index=True, use_container_width=True)

# Footer
st.markdown("""
<div style="margin-top: 4rem; text-align: center;">
    <p class="section-label">Powered by contemporary design principles</p>
</div>
""", unsafe_allow_html=True)