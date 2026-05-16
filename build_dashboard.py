# ============================================================
# DEMANDSENSE — File 3: Interactive Streamlit Dashboard
# PM Purpose: This IS the product. A live, interactive
# revenue intelligence dashboard that a real operations
# team would use daily. This is the link on your resume.
# ============================================================

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="DemandSense | Revenue Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM STYLING ---
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .metric-label  { font-size: 0.78rem; color: #64748B; font-weight: 500; }
    .metric-value  { font-size: 1.6rem; color: #1E293B; font-weight: 700; }
    .metric-delta  { font-size: 0.78rem; color: #16A34A; }
    .section-title { font-size: 1.1rem; font-weight: 600;
                     color: #1E293B; margin-bottom: 0.5rem; }
    .alert-red    { background:#FEF2F2; border-left:4px solid #DC2626;
                    padding:10px 14px; border-radius:6px; margin:6px 0;
                    font-size:0.85rem; }
    .alert-yellow { background:#FFFBEB; border-left:4px solid #D97706;
                    padding:10px 14px; border-radius:6px; margin:6px 0;
                    font-size:0.85rem; }
    .alert-green  { background:#F0FDF4; border-left:4px solid #16A34A;
                    padding:10px 14px; border-radius:6px; margin:6px 0;
                    font-size:0.85rem; }
    .tag-red    { background:#DC2626; color:white; padding:2px 8px;
                  border-radius:4px; font-size:0.72rem; font-weight:600; }
    .tag-yellow { background:#D97706; color:white; padding:2px 8px;
                  border-radius:4px; font-size:0.72rem; font-weight:600; }
    .tag-green  { background:#16A34A; color:white; padding:2px 8px;
                  border-radius:4px; font-size:0.72rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_csv('train.csv', encoding='latin-1')
    df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y')
    df['Ship Date']  = pd.to_datetime(df['Ship Date'],  format='%d/%m/%Y')
    df['Month']      = df['Order Date'].dt.to_period('M').astype(str)
    df['Year']       = df['Order Date'].dt.year
    return df

@st.cache_data
def run_forecast(category, months=6):
    df = load_data()
    cat_df = df[df['Category'] == category].copy()
    monthly = cat_df.groupby(
        cat_df['Order Date'].dt.to_period('M')
    )['Sales'].sum().reset_index()
    monthly['Order Date'] = monthly['Order Date'].dt.to_timestamp()
    monthly.columns = ['ds', 'y']

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode='multiplicative',
        interval_width=0.80
    )
    model.fit(monthly)
    future   = model.make_future_dataframe(periods=months, freq='MS')
    forecast = model.predict(future)

    # Floor negative lower bounds — revenue cannot be negative
    forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)

    return monthly, forecast


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## ⚙️ DemandSense Controls")
    st.markdown("---")

    selected_category = st.selectbox(
        "Forecast Category",
        ["Technology", "Furniture", "Office Supplies"],
        index=0
    )

    forecast_horizon = st.slider(
        "Forecast Horizon (months)",
        min_value=3, max_value=12, value=6, step=1
    )

    selected_region = st.selectbox(
        "Filter Region (Charts)",
        ["All Regions", "West", "East", "Central", "South"],
        index=0
    )

    st.markdown("---")
    st.markdown("### 🎛️ Alert Thresholds")
    upper_multiplier = st.slider(
        "Flag if order exceeds forecast by (%)",
        min_value=10, max_value=100, value=30, step=5
    )
    st.markdown("""
    <div style='font-size:0.78rem; color:#64748B;'>
    Orders exceeding forecast + threshold
    are routed to human review queue.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:#94A3B8;'>
    <b>DemandSense v1.0</b><br>
    Powered by Prophet + Plotly<br>
    Portfolio Project — AI PM Portfolio
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# LOAD DATA & FORECAST
# ============================================================
df = load_data()

if selected_region != "All Regions":
    df_filtered = df[df['Region'] == selected_region]
else:
    df_filtered = df.copy()

with st.spinner(f"Training forecast model for {selected_category}..."):
    monthly_actual, forecast_df = run_forecast(
        selected_category, forecast_horizon
    )

future_only = forecast_df[
    forecast_df['ds'] > monthly_actual['ds'].max()
].copy()


# ============================================================
# HEADER
# ============================================================
col_title, col_badge = st.columns([4, 1])
with col_title:
    st.markdown("# 📈 DemandSense")
    st.markdown(
        "**Revenue Intelligence Platform** — "
        "Probabilistic demand forecasting for enterprise retail operations"
    )
with col_badge:
    st.markdown("<br>", unsafe_allow_html=True)
    st.success("Model Live ✅")

st.markdown("---")


# ============================================================
# KPI METRICS ROW
# ============================================================
total_rev    = df_filtered['Sales'].sum()
total_orders = df_filtered['Order ID'].nunique()
avg_monthly  = df_filtered.groupby('Month')['Sales'].sum().mean()
forecast_6m  = future_only['yhat'].sum()
hist_avg     = monthly_actual['y'].mean()
growth_pct   = (future_only['yhat'].mean() - hist_avg) / hist_avg * 100

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric(
        label="💰 Total Historical Revenue",
        value=f"${total_rev:,.0f}",
        delta="4 years of data"
    )
with k2:
    st.metric(
        label="📦 Total Orders Processed",
        value=f"{total_orders:,}",
        delta=f"{df_filtered['Customer Name'].nunique()} customers"
    )
with k3:
    st.metric(
        label="📅 Avg Monthly Revenue",
        value=f"${avg_monthly:,.0f}",
        delta="Baseline for forecasting"
    )
with k4:
    st.metric(
        label=f"🔮 {forecast_horizon}-Month Forecast",
        value=f"${forecast_6m:,.0f}",
        delta=f"{growth_pct:+.1f}% vs historical avg"
    )

st.markdown("---")


# ============================================================
# MAIN CHARTS — ROW 1
# ============================================================
chart_left, chart_right = st.columns([3, 2])

# --- FORECAST CHART ---
with chart_left:
    st.markdown(
        f'<div class="section-title">'
        f'🔮 {selected_category} Revenue Forecast — '
        f'Next {forecast_horizon} Months</div>',
        unsafe_allow_html=True
    )

    fig_forecast = go.Figure()

    # Historical actuals
    fig_forecast.add_trace(go.Scatter(
        x=monthly_actual['ds'], y=monthly_actual['y'],
        name='Actual Revenue',
        line=dict(color='#2563EB', width=2.5),
        mode='lines+markers', marker=dict(size=5)
    ))

    # Forecast line
    fig_forecast.add_trace(go.Scatter(
        x=forecast_df['ds'], y=forecast_df['yhat'],
        name='Forecast',
        line=dict(color='#16A34A', width=2, dash='dot')
    ))

    # Confidence band
    fig_forecast.add_trace(go.Scatter(
        x=pd.concat([forecast_df['ds'], forecast_df['ds'][::-1]]),
        y=pd.concat([forecast_df['yhat_upper'],
                     forecast_df['yhat_lower'][::-1]]),
        fill='toself',
        fillcolor='rgba(22,163,74,0.10)',
        line=dict(color='rgba(0,0,0,0)'),
        name='80% Confidence Band'
    ))

    # Forecast start line
    split_str = monthly_actual['ds'].max().strftime('%Y-%m-%d')
    fig_forecast.add_shape(
        type='line',
        x0=split_str, x1=split_str, y0=0, y1=1,
        xref='x', yref='paper',
        line=dict(color='#DC2626', width=1.5, dash='dash')
    )
    fig_forecast.add_annotation(
        x=split_str, y=0.97,
        xref='x', yref='paper',
        text='Forecast Start',
        showarrow=False,
        font=dict(color='#DC2626', size=11)
    )

    fig_forecast.update_layout(
        template='plotly_white',
        height=350,
        margin=dict(t=20, b=40, l=10, r=10),
        legend=dict(orientation='h', yanchor='bottom',
                    y=1.0, font=dict(size=11)),
        hovermode='x unified',
        yaxis_title='Revenue ($)',
        xaxis_title=''
    )
    st.plotly_chart(fig_forecast, use_container_width=True)

# --- CATEGORY BREAKDOWN ---
with chart_right:
    st.markdown(
        '<div class="section-title">📊 Revenue by Category</div>',
        unsafe_allow_html=True
    )
    cat_data = df_filtered.groupby('Category')['Sales'].sum().reset_index()
    fig_pie = px.pie(
        cat_data, names='Category', values='Sales',
        color_discrete_sequence=['#2563EB', '#16A34A', '#D97706'],
        hole=0.4
    )
    fig_pie.update_layout(
        height=180,
        margin=dict(t=10, b=10, l=10, r=10),
        showlegend=True,
        legend=dict(font=dict(size=11))
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown(
        '<div class="section-title">🗺️ Revenue by Region</div>',
        unsafe_allow_html=True
    )
    reg_data = df_filtered.groupby('Region')['Sales'].sum().reset_index()
    reg_data  = reg_data.sort_values('Sales', ascending=True)
    fig_bar   = px.bar(
        reg_data, x='Sales', y='Region',
        orientation='h',
        color='Sales', color_continuous_scale='Blues',
        text_auto='.2s'
    )
    fig_bar.update_layout(
        height=155,
        margin=dict(t=5, b=5, l=5, r=5),
        showlegend=False,
        coloraxis_showscale=False,
        xaxis_title='Revenue ($)',
        yaxis_title=''
    )
    st.plotly_chart(fig_bar, use_container_width=True)


st.markdown("---")


# ============================================================
# HUMAN-IN-THE-LOOP ALERT PANEL
# This is the core AI PM differentiator — showing that
# you designed probabilistic guardrails into the product
# ============================================================
st.markdown(
    '<div class="section-title">'
    '🚦 Human-in-the-Loop Alert Panel — '
    'Forecast Deviation Monitor</div>',
    unsafe_allow_html=True
)
st.markdown(
    "<div style='font-size:0.82rem; color:#64748B; margin-bottom:12px;'>"
    "Incoming procurement orders are automatically validated against "
    "the forecast confidence band. Orders outside thresholds are "
    "flagged for buyer review before approval."
    "</div>",
    unsafe_allow_html=True
)

# Simulate incoming orders using forecast values with variation
import random
random.seed(42)

alert_cols = st.columns(3)
alert_data = []

for i, row in future_only.iterrows():
    month_label = row['ds'].strftime('%b %Y')
    forecast_val = row['yhat']
    lower = max(row['yhat_lower'], 0)
    upper = row['yhat_upper']

    # Simulate a procurement order (±40% of forecast)
    variation   = random.uniform(-0.40, 0.55)
    order_value = max(forecast_val * (1 + variation), 0)

    # Classify
    threshold_upper = forecast_val * (1 + upper_multiplier / 100)
    if order_value > threshold_upper:
        status = "REVIEW"
        reason = "Order exceeds upper forecast bound"
    elif order_value < lower * 0.7:
        status = "ALERT"
        reason = "Order far below minimum expected demand"
    else:
        status = "OK"
        reason = "Within confidence band — auto-approved"

    alert_data.append({
        'Month': month_label,
        'Forecast': forecast_val,
        'Lower': lower,
        'Upper': upper,
        'Order Value': order_value,
        'Status': status,
        'Reason': reason
    })

# Display alerts
for item in alert_data:
    tag  = item['Status']
    if tag == "REVIEW":
        css_class = "alert-yellow"
        badge     = '<span class="tag-yellow">⚠️ NEEDS REVIEW</span>'
    elif tag == "ALERT":
        css_class = "alert-red"
        badge     = '<span class="tag-red">🔴 ALERT</span>'
    else:
        css_class = "alert-green"
        badge     = '<span class="tag-green">✅ AUTO-APPROVED</span>'

    st.markdown(f"""
    <div class="{css_class}">
        <b>{item['Month']}</b> &nbsp; {badge} &nbsp;
        Order: <b>${item['Order Value']:,.0f}</b> |
        Forecast: <b>${item['Forecast']:,.0f}</b>
        [${item['Lower']:,.0f} — ${item['Upper']:,.0f}] |
        {item['Reason']}
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")


# ============================================================
# FOOTER — MODEL CARD
# ============================================================
st.markdown(
    '<div class="section-title">📋 Model Card & PM Decisions</div>',
    unsafe_allow_html=True
)

info1, info2, info3 = st.columns(3)

with info1:
    st.markdown("""
    **Model Details**
    - Algorithm: Facebook Prophet
    - Seasonality: Yearly (multiplicative)
    - Training data: 48 months
    - Confidence interval: 80%
    """)

with info2:
    st.markdown("""
    **Key PM Trade-offs Made**
    - Prophet vs ARIMA: Prophet handles gaps natively
    - 80% CI vs 95% CI: Fewer false positives for ops team
    - Monthly vs weekly: Matches procurement cycle
    """)

with info3:
    st.markdown("""
    **Known Limitations**
    - Forecast accuracy degrades beyond 6 months
    - Negative lower bounds floored to $0
    - Does not account for external shocks
    - Retrain recommended quarterly
    """)