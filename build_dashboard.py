import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go
import plotly.express as px
from io import StringIO
import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ────────────────────────────────────────────────
st.set_page_config(
    page_title="DemandSense — Revenue Forecasting",
    page_icon="📈",
    layout="wide"
)

# ── STYLES ─────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #06090F; }
    .stApp { background-color: #06090F; }
    section[data-testid="stSidebar"] { background-color: #0B1019; border-right: 1px solid #1a2535; }
    .metric-box {
        background: linear-gradient(145deg, #0B1019, #0F1622);
        border: 1px solid #1a2535;
        border-radius: 12px;
        padding: 18px 20px;
        text-align: center;
    }
    .metric-val { font-size: 1.8rem; font-weight: 800; color: #60A5FA; font-family: monospace; }
    .metric-lbl { font-size: 0.72rem; color: #6B82A0; letter-spacing: 0.1em; text-transform: uppercase; margin-top: 4px; }
    .gate-auto   { background: rgba(16,185,129,0.08); border: 1px solid rgba(16,185,129,0.2); border-radius: 10px; padding: 14px 18px; }
    .gate-flag   { background: rgba(245,158,11,0.08); border: 1px solid rgba(245,158,11,0.2); border-radius: 10px; padding: 14px 18px; }
    .gate-alert  { background: rgba(239,68,68,0.08);  border: 1px solid rgba(239,68,68,0.2);  border-radius: 10px; padding: 14px 18px; }
    .upload-box  { background: rgba(59,130,246,0.04); border: 1px dashed rgba(59,130,246,0.2); border-radius: 12px; padding: 24px; }
    .sample-note { background: rgba(59,130,246,0.06); border: 1px solid rgba(59,130,246,0.15); border-radius: 8px; padding: 12px 16px; font-size: 0.82rem; color: #B8C8E0; }
    h1, h2, h3 { color: #F0F6FF !important; }
    p, li { color: #B8C8E0; }
    .stSelectbox label, .stRadio label { color: #B8C8E0 !important; }
</style>
""", unsafe_allow_html=True)

# ── HELPERS ────────────────────────────────────────────────────
SAMPLE_CSV = """date,sales
2021-01-01,14200
2021-02-01,13800
2021-03-01,16500
2021-04-01,15900
2021-05-01,17200
2021-06-01,18800
2021-07-01,16100
2021-08-01,17500
2021-09-01,19200
2021-10-01,21000
2021-11-01,26500
2021-12-01,31200
2022-01-01,15100
2022-02-01,14600
2022-03-01,17800
2022-04-01,16900
2022-05-01,18500
2022-06-01,20100
2022-07-01,17200
2022-08-01,18900
2022-09-01,20800
2022-10-01,22700
2022-11-01,28900
2022-12-01,34500
2023-01-01,16200
2023-02-01,15800
2023-03-01,19100
2023-04-01,18200
2023-05-01,20100
2023-06-01,21900
2023-07-01,18600
2023-08-01,20300
2023-09-01,22500
2023-10-01,24800
2023-11-01,31200
2023-12-01,37100"""

@st.cache_data
def load_superstore():
    url = "https://raw.githubusercontent.com/snehajai907-sys/demandsense/main/train.csv"
    try:
        df = pd.read_csv(url)
        df['Order Date'] = pd.to_datetime(df['Order Date'])
        monthly = df.groupby(pd.Grouper(key='Order Date', freq='MS'))['Sales'].sum().reset_index()
        monthly.columns = ['ds', 'y']
        monthly = monthly[monthly['y'] > 0]
        return monthly, df
    except:
        return None, None

def detect_columns(df):
    date_cols, num_cols = [], []
    for col in df.columns:
        try:
            pd.to_datetime(df[col].dropna().head(20))
            date_cols.append(col)
        except: pass
        if pd.api.types.is_numeric_dtype(df[col]):
            num_cols.append(col)
    return date_cols, num_cols

def run_forecast(monthly_df, label="Revenue"):
    m = Prophet(
        interval_width=0.80,
        seasonality_mode='multiplicative',
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False
    )
    m.fit(monthly_df)
    future = m.make_future_dataframe(periods=6, freq='MS')
    forecast = m.predict(future)
    return forecast

def hitl_gate(order_val, lower, upper):
    if order_val < lower:   return "alert",  "🔴 ALERT",  "Below expected range — investigate before cutting procurement"
    if order_val <= upper:  return "auto",   "✅ AUTO-APPROVE", "Within expected range — low risk"
    return "flag", "🟡 FLAG FOR REVIEW", "Above expected range — verify before committing budget"

# ── SIDEBAR ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 DemandSense")
    st.markdown("<p style='color:#6B82A0;font-size:0.78rem'>Probabilistic Revenue Forecasting</p>", unsafe_allow_html=True)
    st.divider()

    data_source = st.radio(
        "**Data Source**",
        ["🏪 Demo Data (Retail)", "📂 Upload Your CSV"],
        index=0
    )

    st.divider()

    if data_source == "🏪 Demo Data (Retail)":
        st.markdown("**Filter by Category**")
        category = st.selectbox("Category", ["Technology", "Furniture", "Office Supplies", "All Categories"])
        st.markdown("**Filter by Region**")
        region = st.selectbox("Region", ["All Regions", "West", "East", "Central", "South"])
    else:
        st.markdown("**Forecast Settings**")
        horizon = st.slider("Forecast horizon (months)", 3, 12, 6)

    st.divider()
    st.markdown("""
    <div style='font-size:0.72rem;color:#3D5070;line-height:1.7'>
    <b style='color:#6B82A0'>How the HITL Gate works:</b><br>
    ✅ Within 80% CI → Auto-approve<br>
    🟡 Above upper bound → Flag for review<br>
    🔴 Below lower bound → Alert ops team
    </div>
    """, unsafe_allow_html=True)

# ── MAIN HEADER ────────────────────────────────────────────────
st.markdown("## 📈 DemandSense")
st.markdown("<p style='color:#6B82A0'>Probabilistic revenue forecasting with Human-in-the-Loop procurement gates</p>", unsafe_allow_html=True)
st.divider()

# ── DATA SOURCE: UPLOAD ────────────────────────────────────────
if data_source == "📂 Upload Your CSV":
    st.markdown("### Upload Your Sales Data")

    col_info, col_dl = st.columns([3, 1])
    with col_info:
        st.markdown("""
        <div class='sample-note'>
        <b>Expected format:</b> A CSV with one date column (monthly or daily) and one numeric sales/revenue column.
        Minimum 12 months of data required for a reliable forecast.
        </div>
        """, unsafe_allow_html=True)
    with col_dl:
        st.download_button(
            label="⬇ Download Sample CSV",
            data=SAMPLE_CSV,
            file_name="sample_sales_data.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.markdown("")

    # ── ONE-CLICK DEMO BUTTON (same as FinDoc) ──
    col_demo, col_spacer = st.columns([1, 2])
    with col_demo:
        load_demo = st.button(
            "⚡ Load Demo Data",
            use_container_width=True,
            help="Instantly load 3 years of sample retail data — no upload needed"
        )

    if load_demo:
        st.session_state['use_demo_csv'] = True
    if 'use_demo_csv' not in st.session_state:
        st.session_state['use_demo_csv'] = False

    st.markdown("<p style='color:#6B82A0;font-size:0.78rem;margin-top:4px'>— or upload your own file below —</p>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop your CSV here or click to browse",
        type=["csv"],
        help="CSV with a date column and a numeric sales/revenue column"
    )

    # Determine data source: demo button > uploaded file > empty state
    if st.session_state['use_demo_csv'] and uploaded_file is None:
        df_raw = pd.read_csv(StringIO(SAMPLE_CSV))
        st.success("⚡ Demo data loaded — 3 years of monthly retail sales (36 months). Upload your own CSV above to replace it.")
    elif uploaded_file is not None:
        st.session_state['use_demo_csv'] = False
        try:
            df_raw = pd.read_csv(uploaded_file)
            st.success(f"✅ File loaded — {len(df_raw):,} rows, {len(df_raw.columns)} columns")
        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.stop()
    else:
        st.markdown("""
        <div class='upload-box' style='text-align:center;margin-top:20px'>
            <div style='font-size:2rem'>📂</div>
            <div style='color:#B8C8E0;margin-top:10px;font-size:0.95rem'>Click <b>⚡ Load Demo Data</b> for instant preview</div>
            <div style='color:#6B82A0;font-size:0.8rem;margin-top:6px'>or upload your own CSV to forecast your data</div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # Detect columns
    date_cols, num_cols = detect_columns(df_raw)

    if not date_cols:
        st.error("No date column detected. Make sure one column contains dates (e.g. 2023-01-01 or Jan 2023).")
        st.stop()
    if not num_cols:
        st.error("No numeric column detected. Make sure one column contains your sales or revenue numbers.")
        st.stop()

    # Column mapping
    col_map1, col_map2 = st.columns(2)
    with col_map1:
        date_col = st.selectbox("📅 Date column", date_cols, index=0)
    with col_map2:
        num_col  = st.selectbox("💰 Sales / Revenue column", num_cols, index=0)

    # Build Prophet-ready dataframe
    try:
        df_raw[date_col] = pd.to_datetime(df_raw[date_col])
        df_raw[num_col]  = pd.to_numeric(df_raw[num_col], errors='coerce')
        df_raw = df_raw.dropna(subset=[date_col, num_col])

        # Aggregate to monthly
        monthly = (
            df_raw
            .groupby(pd.Grouper(key=date_col, freq='MS'))[num_col]
            .sum()
            .reset_index()
        )
        monthly.columns = ['ds', 'y']
        monthly = monthly[monthly['y'] > 0].sort_values('ds').reset_index(drop=True)

        if len(monthly) < 12:
            st.warning(f"Only {len(monthly)} months of data found. At least 12 months recommended for a reliable forecast. Proceeding but results may be less accurate.")
        if len(monthly) < 4:
            st.error("Not enough data to forecast. Please provide at least 4 months of data.")
            st.stop()

        currency_label = num_col
        use_upload = True

    except Exception as e:
        st.error(f"Error processing data: {e}")
        st.stop()

    # Run forecast
    with st.spinner("Training forecasting model on your data..."):
        try:
            forecast = run_forecast(monthly, label=num_col)
        except Exception as e:
            st.error(f"Forecasting error: {e}")
            st.stop()

    hist = forecast[forecast['ds'] <= monthly['ds'].max()].copy()
    fut  = forecast[forecast['ds'] >  monthly['ds'].max()].copy()
    last_actual = float(monthly['y'].iloc[-1])
    avg_actual  = float(monthly['y'].mean())
    proj_total  = float(fut['yhat'].sum())
    growth_pct  = ((fut['yhat'].mean() - avg_actual) / avg_actual * 100)

# ── DATA SOURCE: DEMO ──────────────────────────────────────────
else:
    with st.spinner("Loading retail dataset..."):
        monthly_full, df_raw = load_superstore()

    if monthly_full is None:
        st.error("Could not load demo data. Please check your connection and try again.")
        st.stop()

    # Apply filters
    df_f = df_raw.copy()
    if 'category' in locals() and category != "All Categories":
        df_f = df_f[df_f['Category'] == category]
    if 'region' in locals() and region != "All Regions":
        df_f = df_f[df_f['Region'] == region]

    monthly = df_f.groupby(pd.Grouper(key='Order Date', freq='MS'))['Sales'].sum().reset_index()
    monthly.columns = ['ds', 'y']
    monthly = monthly[monthly['y'] > 0].sort_values('ds').reset_index(drop=True)
    currency_label = "Sales ($)"
    horizon = 6

    with st.spinner("Training Prophet forecasting model..."):
        forecast = run_forecast(monthly)

    hist = forecast[forecast['ds'] <= monthly['ds'].max()].copy()
    fut  = forecast[forecast['ds'] >  monthly['ds'].max()].copy()
    last_actual = float(monthly['y'].iloc[-1])
    avg_actual  = float(monthly['y'].mean())
    proj_total  = float(fut['yhat'].sum())
    growth_pct  = ((fut['yhat'].mean() - avg_actual) / avg_actual * 100)

# ── KEY METRICS ────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)

def mbox(val, lbl):
    return f"<div class='metric-box'><div class='metric-val'>{val}</div><div class='metric-lbl'>{lbl}</div></div>"

with m1: st.markdown(mbox(f"${last_actual:,.0f}", "Last Month Actual"), unsafe_allow_html=True)
with m2: st.markdown(mbox(f"${fut['yhat'].iloc[0]:,.0f}", "Next Month Forecast"), unsafe_allow_html=True)
with m3: st.markdown(mbox(f"{growth_pct:+.1f}%", "Projected Growth vs Avg"), unsafe_allow_html=True)
with m4: st.markdown(mbox(f"80%", "Confidence Interval"), unsafe_allow_html=True)

st.markdown("")

# ── FORECAST CHART ─────────────────────────────────────────────
fig = go.Figure()

# CI band (forecast)
fig.add_trace(go.Scatter(
    x=pd.concat([fut['ds'], fut['ds'][::-1]]),
    y=pd.concat([fut['yhat_upper'], fut['yhat_lower'][::-1]]),
    fill='toself',
    fillcolor='rgba(59,130,246,0.08)',
    line=dict(color='rgba(255,255,255,0)'),
    name='80% Confidence Band',
    showlegend=True
))

# Historical line
fig.add_trace(go.Scatter(
    x=monthly['ds'], y=monthly['y'],
    mode='lines+markers',
    name='Actual', line=dict(color='#60A5FA', width=2),
    marker=dict(size=4)
))

# Forecast line
fig.add_trace(go.Scatter(
    x=fut['ds'], y=fut['yhat'],
    mode='lines+markers',
    name='Forecast', line=dict(color='#818CF8', width=2, dash='dash'),
    marker=dict(size=6, symbol='circle-open')
))

# Upper / lower bounds
fig.add_trace(go.Scatter(
    x=fut['ds'], y=fut['yhat_upper'],
    mode='lines', name='Upper Bound (80%)',
    line=dict(color='rgba(59,130,246,0.3)', width=1, dash='dot'), showlegend=False
))
fig.add_trace(go.Scatter(
    x=fut['ds'], y=fut['yhat_lower'],
    mode='lines', name='Lower Bound (80%)',
    line=dict(color='rgba(59,130,246,0.3)', width=1, dash='dot'), showlegend=False
))

# Divider line
fig.add_vline(
    x=monthly['ds'].max().timestamp() * 1000,
    line_dash="dash", line_color="rgba(255,255,255,0.15)",
    annotation_text="Forecast →",
    annotation_font_color="#6B82A0"
)

fig.update_layout(
    plot_bgcolor='#0B1019', paper_bgcolor='#0B1019',
    font=dict(color='#B8C8E0', family='sans-serif'),
    xaxis=dict(gridcolor='#1a2535', showgrid=True, zeroline=False, title=''),
    yaxis=dict(gridcolor='#1a2535', showgrid=True, zeroline=False, title=currency_label),
    legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='#1a2535', borderwidth=1),
    margin=dict(t=20, b=20, l=10, r=10),
    height=400,
    hovermode='x unified'
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── HITL PROCUREMENT GATE ──────────────────────────────────────
st.markdown("### 🛡️ Human-in-the-Loop Procurement Gate")
st.markdown("<p style='color:#6B82A0;font-size:0.85rem'>Simulate an incoming procurement order. The system checks it against the 80% confidence interval and recommends an action.</p>", unsafe_allow_html=True)

next_lower = float(max(0, fut['yhat_lower'].iloc[0]))
next_upper = float(fut['yhat_upper'].iloc[0])
next_yhat  = float(fut['yhat'].iloc[0])

col_slider, col_result = st.columns([1, 1])

with col_slider:
    order_min = int(next_lower * 0.5)
    order_max = int(next_upper * 1.5)
    order_default = int(next_yhat)

    order_value = st.number_input(
        "Enter procurement order value ($)",
        min_value=0,
        max_value=order_max * 2,
        value=order_default,
        step=max(100, int((order_max - order_min) / 50)),
        help="Simulate any order value to see the gate recommendation"
    )

    st.markdown(f"""
    <div style='margin-top:12px;font-size:0.8rem;color:#6B82A0;line-height:1.9'>
    <b style='color:#B8C8E0'>Expected range (80% CI):</b><br>
    Lower: <span style='color:#60A5FA'>${next_lower:,.0f}</span><br>
    Forecast: <span style='color:#818CF8'>${next_yhat:,.0f}</span><br>
    Upper: <span style='color:#60A5FA'>${next_upper:,.0f}</span>
    </div>
    """, unsafe_allow_html=True)

with col_result:
    gate_type, gate_label, gate_msg = hitl_gate(order_value, next_lower, next_upper)
    css_class = f"gate-{gate_type}"
    col_icon = {"auto": "#10B981", "flag": "#F59E0B", "alert": "#EF4444"}[gate_type]

    st.markdown(f"""
    <div class='{css_class}' style='margin-top:8px'>
        <div style='font-size:1.1rem;font-weight:700;color:{col_icon};margin-bottom:8px'>{gate_label}</div>
        <div style='font-size:0.85rem;color:#B8C8E0'>{gate_msg}</div>
        <div style='font-size:0.78rem;color:#6B82A0;margin-top:10px'>
            Order: <b style='color:#F0F6FF'>${order_value:,.0f}</b> &nbsp;|&nbsp;
            {"Within" if gate_type == "auto" else ("Above" if gate_type == "flag" else "Below")} 80% CI
        </div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ── FORECAST TABLE ─────────────────────────────────────────────
with st.expander("📋 View Full Forecast Table"):
    table = fut[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    table.columns = ['Month', 'Forecast', 'Lower (80%)', 'Upper (80%)']
    table['Month'] = table['Month'].dt.strftime('%b %Y')
    table['Floor'] = table['Lower (80%)'].apply(lambda x: max(0, x))
    for c in ['Forecast', 'Lower (80%)', 'Upper (80%)', 'Floor']:
        table[c] = table[c].apply(lambda x: f"${max(0,x):,.0f}")
    st.dataframe(table.drop('Floor', axis=1), use_container_width=True, hide_index=True)

# ── FOOTER ─────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;margin-top:40px;padding:20px;border-top:1px solid #1a2535'>
    <span style='font-size:0.72rem;color:#3D5070;letter-spacing:0.08em'>
    DEMANDSENSE · PROBABILISTIC REVENUE FORECASTING · BUILT BY SNEHA JAISWAL · 2025
    </span>
</div>
""", unsafe_allow_html=True)
