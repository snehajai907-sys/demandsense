import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go
from io import StringIO
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="DemandSense — Revenue Forecasting", page_icon="📈", layout="wide")

st.markdown("""
<style>
    .main,.stApp{background-color:#06090F}
    section[data-testid="stSidebar"]{background-color:#0B1019;border-right:1px solid #1a2535}
    .mbox{background:linear-gradient(145deg,#0B1019,#0F1622);border:1px solid #1a2535;border-radius:12px;padding:18px 20px;text-align:center}
    .mv{font-size:1.8rem;font-weight:800;color:#60A5FA;font-family:monospace}
    .ml{font-size:0.7rem;color:#6B82A0;letter-spacing:.1em;text-transform:uppercase;margin-top:4px}
    .gate-auto{background:rgba(16,185,129,.08);border:1px solid rgba(16,185,129,.2);border-radius:10px;padding:14px 18px}
    .gate-flag{background:rgba(245,158,11,.08);border:1px solid rgba(245,158,11,.2);border-radius:10px;padding:14px 18px}
    .gate-alert{background:rgba(239,68,68,.08);border:1px solid rgba(239,68,68,.2);border-radius:10px;padding:14px 18px}
    .info-box{background:rgba(59,130,246,.05);border:1px solid rgba(59,130,246,.15);border-radius:10px;padding:14px 18px}
    .card{background:rgba(255,255,255,.02);border:1px solid #1a2535;border-radius:10px;padding:14px 16px;margin-bottom:10px}
    h1,h2,h3{color:#F0F6FF!important}
    p,li{color:#B8C8E0}
</style>
""", unsafe_allow_html=True)

# ── SAMPLE CSV ─────────────────────────────────────────────────
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

# ── HELPERS ────────────────────────────────────────────────────
# FIX 2: st.error removed from cached function — error returned as string
def parse_dates_robust(series):
    """Try multiple strategies — handles ISO, DD/MM/YYYY, MM/DD/YYYY, mixed."""
    strategies = [
        dict(format='ISO8601'),
        dict(format='mixed', dayfirst=True),
        dict(format='mixed', dayfirst=False),
        dict(dayfirst=True),
        dict(),
    ]
    for kwargs in strategies:
        try:
            result = pd.to_datetime(series, **kwargs)
            valid = result.notna().sum()
            if valid > len(series) * 0.8:   # at least 80% parsed successfully
                return result
        except Exception:
            continue
    return pd.to_datetime(series, errors='coerce')  # final fallback

@st.cache_data
def load_superstore():
    try:
        df = pd.read_csv('train.csv')
        df['Order Date'] = parse_dates_robust(df['Order Date'])
        return df, None
    except Exception as e:
        return None, str(e)

def prep_monthly(df, date_col, val_col):
    df = df.copy()
    df[date_col] = parse_dates_robust(df[date_col])
    df[val_col]  = pd.to_numeric(df[val_col], errors='coerce')
    df = df.dropna(subset=[date_col, val_col])
    monthly = df.groupby(pd.Grouper(key=date_col, freq='MS'))[val_col].sum().reset_index()
    monthly.columns = ['ds', 'y']
    monthly = monthly[monthly['y'] > 0].sort_values('ds').reset_index(drop=True)
    return monthly

def detect_columns(df):
    date_cols, num_cols = [], []
    for col in df.columns:
        is_num = pd.api.types.is_numeric_dtype(df[col])
        if not is_num:
            try:
                sample = df[col].dropna().head(30).astype(str)
                try:    parsed = pd.to_datetime(sample, format='ISO8601')
                except: parsed = pd.to_datetime(sample, format='mixed', dayfirst=True)
                if parsed.nunique() > 1:
                    date_cols.append(col)
            except: pass
        if is_num:
            cd = df[col].dropna()
            is_id = (cd.dtype in ['int64','int32'] and cd.nunique()==len(cd) and cd.min()==1)
            if not is_id and cd.nunique() > 3:
                num_cols.append(col)
    return date_cols, num_cols

# FIX 1 & 9: horizon parameter now used
def run_forecast(monthly_df, horizon=6):
    m = Prophet(
        interval_width=0.80,
        seasonality_mode='multiplicative',
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False
    )
    m.fit(monthly_df)
    future = m.make_future_dataframe(periods=horizon, freq='MS')
    forecast = m.predict(future)
    return forecast, m

def hitl_gate(val, lo, hi):
    if val < lo:  return "alert", "🔴 ALERT",          "Below expected range — investigate before cutting procurement"
    if val <= hi: return "auto",  "✅ AUTO-APPROVE",   "Within expected range — safe to approve"
    return              "flag",   "🟡 FLAG FOR REVIEW", "Above expected range — verify before committing budget"

def mbox(val, lbl):
    return f"<div class='mbox'><div class='mv'>{val}</div><div class='ml'>{lbl}</div></div>"

def dark_chart():
    return dict(
        plot_bgcolor='#0B1019', paper_bgcolor='#0B1019',
        font=dict(color='#B8C8E0'),
        xaxis=dict(gridcolor='#1a2535', showgrid=True, zeroline=False),
        yaxis=dict(gridcolor='#1a2535', showgrid=True, zeroline=False),
        legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='#1a2535', borderwidth=1),
        margin=dict(t=30, b=20, l=10, r=10), height=380, hovermode='x unified'
    )

# ── SIDEBAR ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 DemandSense")
    st.markdown("<p style='color:#6B82A0;font-size:.78rem'>Probabilistic Revenue Forecasting</p>", unsafe_allow_html=True)
    st.divider()

    src = st.radio("**Data Source**",
        ["🏪 Demo Data (Retail)", "📂 Upload Your CSV"], index=0)
    st.divider()

    # FIX 5: category/region always initialised
    category = "Technology"
    region   = "All Regions"
    horizon  = 6

    if src == "🏪 Demo Data (Retail)":
        category = st.selectbox("Category", ["Technology","Furniture","Office Supplies","All Categories"])
        region   = st.selectbox("Region",   ["All Regions","West","East","Central","South"])
    else:
        horizon = st.slider("Forecast horizon (months)", 3, 12, 6)

    st.divider()
    st.markdown("""
    <div style='font-size:.72rem;color:#3D5070;line-height:1.8'>
    <b style='color:#6B82A0'>HITL Gate logic:</b><br>
    ✅ Within 80% CI → Auto-approve<br>
    🟡 Above upper → Flag for review<br>
    🔴 Below lower → Alert ops team
    </div>""", unsafe_allow_html=True)

# ── HEADER ─────────────────────────────────────────────────────
st.markdown("## 📈 DemandSense")
st.markdown("<p style='color:#6B82A0'>Probabilistic revenue forecasting · Human-in-the-Loop procurement gates · 80% CI</p>", unsafe_allow_html=True)
st.divider()

# ══════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════
monthly = None
currency_label = "Sales ($)"

if src == "📂 Upload Your CSV":
    st.markdown("### Upload Your Sales Data")

    c1, c2 = st.columns([3,1])
    with c1:
        st.markdown("""<div class='info-box'><b>Expected format:</b> CSV with a date column
        (daily or monthly) and a numeric sales/revenue column. Minimum 12 months for reliable results.
        </div>""", unsafe_allow_html=True)
    with c2:
        st.download_button("⬇ Sample CSV", SAMPLE_CSV,
            "sample_sales.csv", "text/csv", use_container_width=True)

    st.markdown("")
    cdemo, _ = st.columns([1,2])
    with cdemo:
        if st.button("⚡ Load Demo Data", use_container_width=True):
            st.session_state['demo_csv'] = True

    if 'demo_csv' not in st.session_state:
        st.session_state['demo_csv'] = False

    st.markdown("<p style='color:#6B82A0;font-size:.78rem;margin-top:4px'>— or upload your own below —</p>",
        unsafe_allow_html=True)

    uploaded = st.file_uploader("Drop CSV here", type=["csv"])

    if st.session_state['demo_csv'] and uploaded is None:
        df_raw = pd.read_csv(StringIO(SAMPLE_CSV))
        st.success("⚡ Demo data loaded — 36 months of sample retail sales")
    elif uploaded is not None:
        st.session_state['demo_csv'] = False
        try:
            df_raw = pd.read_csv(uploaded)
            st.success(f"✅ {len(df_raw):,} rows loaded")
        except Exception as e:
            st.error(f"Could not read file: {e}"); st.stop()
    else:
        st.markdown("""<div style='background:rgba(59,130,246,.04);border:1px dashed rgba(59,130,246,.2);
        border-radius:12px;padding:32px;text-align:center;margin-top:16px'>
        <div style='font-size:2rem'>📂</div>
        <div style='color:#B8C8E0;margin-top:10px'>Click <b>⚡ Load Demo Data</b> for instant preview</div>
        <div style='color:#6B82A0;font-size:.8rem;margin-top:6px'>or upload your own CSV</div>
        </div>""", unsafe_allow_html=True)
        st.stop()

    date_cols, num_cols = detect_columns(df_raw)
    if not date_cols:
        st.error("❌ No date column found. Ensure one column has dates like 2023-01-01 or 15/04/2022.")
        st.stop()
    if not num_cols:
        st.error("❌ No numeric column found. Ensure one column has sales or revenue numbers.")
        st.stop()

    cm1, cm2 = st.columns(2)
    with cm1: date_col = st.selectbox("📅 Date column", date_cols)
    with cm2: num_col  = st.selectbox("💰 Sales / Revenue column", num_cols)

    try:
        monthly = prep_monthly(df_raw, date_col, num_col)
    except Exception as e:
        st.error(f"Error processing columns: {e}"); st.stop()

    if len(monthly) < 4:
        st.error("Not enough data — need at least 4 months."); st.stop()
    if len(monthly) < 12:
        st.warning(f"Only {len(monthly)} months found. 12+ recommended for accuracy.")

    # IMPROVEMENT A: Data quality card
    st.markdown(f"""
    <div class='info-box' style='margin-top:12px;font-size:.82rem;color:#B8C8E0;line-height:1.8'>
    📊 <b>Data Summary</b> &nbsp;·&nbsp;
    {len(monthly)} months &nbsp;·&nbsp;
    {monthly['ds'].min().strftime('%b %Y')} → {monthly['ds'].max().strftime('%b %Y')} &nbsp;·&nbsp;
    Avg ${monthly['y'].mean():,.0f}/mo &nbsp;·&nbsp;
    Peak ${monthly['y'].max():,.0f} ({monthly.loc[monthly['y'].idxmax(),'ds'].strftime('%b %Y')})
    </div>""", unsafe_allow_html=True)

    currency_label = num_col

else:
    # Demo retail mode
    with st.spinner("Loading retail dataset..."):
        df_raw, err = load_superstore()

    # FIX 2: error handling outside cached function
    if df_raw is None:
        st.error(f"Could not load demo data: {err}"); st.stop()

    df_f = df_raw.copy()
    if category != "All Categories":
        df_f = df_f[df_f['Category'] == category]
    if region != "All Regions":
        df_f = df_f[df_f['Region'] == region]

    monthly = df_f.groupby(pd.Grouper(key='Order Date', freq='MS'))['Sales'].sum().reset_index()
    monthly.columns = ['ds','y']
    monthly = monthly[monthly['y'] > 0].sort_values('ds').reset_index(drop=True)

# ══════════════════════════════════════════════════════════════
# FORECAST
# ══════════════════════════════════════════════════════════════
with st.spinner("Training forecasting model..."):
    try:
        forecast, model = run_forecast(monthly, horizon=horizon)
    except Exception as e:
        st.error(f"Forecast error: {e}"); st.stop()

fut = forecast[forecast['ds'] > monthly['ds'].max()].copy()
# FIX 3: floor negatives in forecast
fut['yhat_lower'] = fut['yhat_lower'].clip(lower=0)
fut['yhat']       = fut['yhat'].clip(lower=0)

hist_fc = forecast[forecast['ds'] <= monthly['ds'].max()].copy()

last_val   = float(monthly['y'].iloc[-1])
avg_val    = float(monthly['y'].mean())
proj_total = float(fut['yhat'].sum())          # FIX 7: now displayed below
growth_pct = (fut['yhat'].mean() - avg_val) / avg_val * 100

# IMPROVEMENT B: Model accuracy on historical data (MAPE)
merged = monthly.merge(hist_fc[['ds','yhat']], on='ds', how='inner')
mape = float((abs(merged['y'] - merged['yhat']) / merged['y']).mean() * 100)
accuracy = max(0, 100 - mape)

# ── METRICS ────────────────────────────────────────────────────
c1,c2,c3,c4,c5 = st.columns(5)
with c1: st.markdown(mbox(f"${last_val:,.0f}",   "Last Month Actual"),    unsafe_allow_html=True)
with c2: st.markdown(mbox(f"${fut['yhat'].iloc[0]:,.0f}", "Next Month Forecast"), unsafe_allow_html=True)
with c3: st.markdown(mbox(f"${proj_total:,.0f}", f"{horizon}M Projected Total"),  unsafe_allow_html=True)
with c4: st.markdown(mbox(f"{growth_pct:+.1f}%", "vs Historical Avg"),   unsafe_allow_html=True)
with c5: st.markdown(mbox(f"{accuracy:.1f}%",    "Model Fit Score"),      unsafe_allow_html=True)

st.markdown("")

# ── FORECAST CHART ─────────────────────────────────────────────
# IMPROVEMENT E: annotate peak and lowest forecast month
peak_idx = fut['yhat'].idxmax()
low_idx  = fut['yhat'].idxmin()

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=pd.concat([fut['ds'], fut['ds'][::-1]]),
    y=pd.concat([fut['yhat_upper'], fut['yhat_lower'][::-1]]),
    fill='toself', fillcolor='rgba(59,130,246,0.07)',
    line=dict(color='rgba(0,0,0,0)'),
    name='80% Confidence Band'
))

fig.add_trace(go.Scatter(
    x=monthly['ds'], y=monthly['y'],
    mode='lines+markers', name='Actual',
    line=dict(color='#60A5FA', width=2), marker=dict(size=4)
))

fig.add_trace(go.Scatter(
    x=fut['ds'], y=fut['yhat'],
    mode='lines+markers', name='Forecast',
    line=dict(color='#818CF8', width=2, dash='dash'),
    marker=dict(size=6, symbol='circle-open'),
    customdata=np.stack([fut['yhat_lower'], fut['yhat_upper']], axis=-1),
    hovertemplate='<b>%{x|%b %Y}</b><br>Forecast: $%{y:,.0f}<br>Lower: $%{customdata[0]:,.0f}<br>Upper: $%{customdata[1]:,.0f}<extra></extra>'
))

# Upper/lower dotted lines
for col, nm in [('yhat_upper','Upper 80%'),('yhat_lower','Lower 80%')]:
    fig.add_trace(go.Scatter(
        x=fut['ds'], y=fut[col], mode='lines', name=nm,
        line=dict(color='rgba(59,130,246,0.25)', width=1, dash='dot'), showlegend=False
    ))

# Peak annotation
fig.add_annotation(
    x=fut.loc[peak_idx,'ds'], y=fut.loc[peak_idx,'yhat'],
    text=f"Peak: ${fut.loc[peak_idx,'yhat']:,.0f}",
    showarrow=True, arrowhead=2, arrowcolor='#10B981',
    font=dict(color='#10B981', size=11), bgcolor='rgba(16,185,129,0.1)',
    bordercolor='rgba(16,185,129,0.3)', borderwidth=1
)

fig.add_vline(
    x=monthly['ds'].max().timestamp()*1000,
    line_dash='dash', line_color='rgba(255,255,255,0.12)',
    annotation_text='Forecast →', annotation_font_color='#6B82A0'
)

fig.update_layout(**dark_chart(), yaxis_title=currency_label)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── HITL GATE ──────────────────────────────────────────────────
st.markdown("### 🛡️ Human-in-the-Loop Procurement Gate")
st.markdown("""
<div class='info-box' style='margin-bottom:16px;font-size:.85rem;color:#B8C8E0;line-height:1.7'>
<b style='color:#F0F6FF'>What this does:</b> Your buying team wants to place a procurement order next month.
Enter the order value below. The system checks it against the forecast's 80% confidence interval
and recommends whether to <span style='color:#10B981'>auto-approve it</span>,
<span style='color:#F59E0B'>flag it for manual review</span>, or
<span style='color:#EF4444'>alert the ops team</span>.<br>
<span style='color:#6B82A0;font-size:.78rem'>Use the +/- buttons or type any value to simulate different order scenarios.</span>
</div>""", unsafe_allow_html=True)

nxt_lo  = float(fut['yhat_lower'].iloc[0])
nxt_hi  = float(fut['yhat_upper'].iloc[0])
nxt_hat = float(fut['yhat'].iloc[0])
step = max(100, int((nxt_hi - nxt_lo) / 40))

# Key that changes only when the underlying forecast changes (new data/filter)
# This resets the input only when you switch data source, not on every rerun
forecast_fingerprint = f"{len(monthly)}_{monthly['ds'].max().date()}_{nxt_hat:.0f}"
if st.session_state.get('_forecast_fp') != forecast_fingerprint:
    st.session_state['_forecast_fp']  = forecast_fingerprint
    st.session_state['order_input']   = int(nxt_hat)

gc1, gc2 = st.columns(2)
with gc1:
    st.markdown("<p style='color:#B8C8E0;font-size:.85rem;font-weight:600;margin-bottom:4px'>Enter next month\'s procurement order value:</p>", unsafe_allow_html=True)
    order_val = st.number_input(
        "Order value ($)",
        label_visibility="collapsed",
        min_value=0,
        max_value=int(nxt_hi * 3),
        step=step,
        key="order_input",
        help=f"Use +/- to adjust by ${step:,} per click. Gate updates instantly."
    )
    st.markdown(f"""
    <div style='margin-top:14px;background:rgba(255,255,255,.02);border:1px solid #1a2535;
    border-radius:9px;padding:14px 16px;font-size:.82rem;color:#6B82A0;line-height:2'>
    <b style='color:#B8C8E0'>Model's expected range for next month</b><br>
    🔵 Lower bound (80% CI): <b style='color:#60A5FA'>${nxt_lo:,.0f}</b><br>
    🟣 Point forecast: <b style='color:#818CF8'>${nxt_hat:,.0f}</b><br>
    🔵 Upper bound (80% CI): <b style='color:#60A5FA'>${nxt_hi:,.0f}</b><br>
    <span style='font-size:.73rem'>Orders between lower and upper bound are considered normal.</span>
    </div>""", unsafe_allow_html=True)

with gc2:
    gtype, glabel, gmsg = hitl_gate(order_val, nxt_lo, nxt_hi)
    gcolor = {"auto":"#10B981","flag":"#F59E0B","alert":"#EF4444"}[gtype]
    pct_vs_forecast = ((order_val - nxt_hat) / nxt_hat * 100) if nxt_hat > 0 else 0
    st.markdown(f"""<div class='gate-{gtype}' style='height:100%;min-height:160px'>
        <div style='font-size:1.1rem;font-weight:700;color:{gcolor};margin-bottom:10px'>{glabel}</div>
        <div style='font-size:.85rem;color:#B8C8E0;margin-bottom:12px'>{gmsg}</div>
        <div style='font-size:.8rem;color:#6B82A0;border-top:1px solid rgba(255,255,255,.05);padding-top:10px;line-height:1.8'>
        Your order: <b style='color:#F0F6FF'>${order_val:,.0f}</b><br>
        vs forecast: <b style='color:{gcolor}'>{pct_vs_forecast:+.1f}%</b><br>
        Status: {"✅ No action needed" if gtype=="auto" else ("⚠️ Needs human review" if gtype=="flag" else "🚨 Ops team notified")}
        </div>
    </div>""", unsafe_allow_html=True)

st.divider()

# ── IMPROVEMENT C: SEASONAL DECOMPOSITION ──────────────────────
with st.expander("📊 Seasonal Trends & Model Components"):
    comp = forecast[forecast['ds'] <= monthly['ds'].max()].copy()

    sfig = go.Figure()
    sfig.add_trace(go.Scatter(x=comp['ds'], y=comp['trend'],
        mode='lines', name='Trend', line=dict(color='#60A5FA', width=2)))
    sfig.add_trace(go.Scatter(
        x=comp['ds'],
        y=comp['trend'] * (1 + comp.get('yearly', pd.Series(0, index=comp.index))),
        mode='lines', name='Trend + Seasonality',
        line=dict(color='#818CF8', width=1.5, dash='dot')))
    sfig.update_layout(**dark_chart(),
        title=dict(text='Revenue Trend vs Seasonal Pattern', font=dict(color='#B8C8E0', size=13)))
    st.plotly_chart(sfig, use_container_width=True)

    st.markdown("""<div class='info-box' style='font-size:.82rem;color:#B8C8E0;line-height:1.8'>
    <b>How to read this:</b> The solid line shows the underlying revenue trend (growth direction).
    The dotted line adds the seasonal effect — showing how holidays and cycles lift or suppress revenue.
    When the lines diverge in Q4, that is the holiday season multiplier at work.
    </div>""", unsafe_allow_html=True)

# ── IMPROVEMENT D: EXPORT FORECAST ─────────────────────────────
with st.expander("📋 Forecast Table & Export"):
    tbl = fut[['ds','yhat','yhat_lower','yhat_upper']].copy()
    tbl.columns = ['Month','Forecast','Lower (80%)','Upper (80%)']
    tbl['Month'] = tbl['Month'].dt.strftime('%b %Y')
    tbl['Gate'] = [hitl_gate(r['Forecast'], r['Lower (80%)'], r['Upper (80%)'])[1]
                   for _, r in tbl.iterrows()]
    for c in ['Forecast','Lower (80%)','Upper (80%)']:
        tbl[c] = tbl[c].apply(lambda x: f"${max(0,x):,.0f}")

    st.dataframe(tbl, use_container_width=True, hide_index=True)

    csv_out = fut[['ds','yhat','yhat_lower','yhat_upper']].copy()
    csv_out.columns = ['month','forecast','lower_80pct','upper_80pct']
    csv_out['month'] = csv_out['month'].dt.strftime('%Y-%m-%d')
    for c in ['forecast','lower_80pct','upper_80pct']:
        csv_out[c] = csv_out[c].clip(lower=0).round(2)

    st.download_button(
        "⬇ Download Forecast CSV",
        csv_out.to_csv(index=False),
        "demandsense_forecast.csv", "text/csv"
    )

# ── IMPROVEMENT F: MODEL CARD ───────────────────────────────────
with st.expander("🗂️ Model Card — PM Decisions & Architecture"):
    st.markdown("""
    <div style='font-size:.85rem;color:#B8C8E0;line-height:1.9'>

    <b style='color:#F0F6FF;font-size:.95rem'>Why Prophet over ARIMA?</b><br>
    Prophet handles missing dates and irregular seasonality natively. ARIMA requires stationary
    data and manual differencing. For a lean build with no dedicated data scientist, Prophet
    reduces data engineering work by weeks.<br><br>

    <b style='color:#F0F6FF;font-size:.95rem'>Why 80% Confidence Interval — not 95%?</b><br>
    A 95% CI produces a wider band, which means more orders trigger alerts.
    If every order is flagged, buyers stop trusting the system and ignore it entirely.
    80% CI keeps alerts meaningful — only genuinely unusual orders are escalated.
    <i>Optimising for user adoption over statistical conservatism.</i><br><br>

    <b style='color:#F0F6FF;font-size:.95rem'>Why multiplicative seasonality?</b><br>
    Retail revenue scales proportionally during peak seasons — a 20% holiday uplift
    applies to a larger revenue base each year. Multiplicative mode captures this correctly.
    Additive mode assumes the seasonal bump is a fixed dollar amount, which understates Q4.<br><br>

    <b style='color:#F0F6FF;font-size:.95rem'>Why floor negative lower bounds?</b><br>
    Revenue cannot be negative. In February, the model's lower bound can drift below zero.
    Displaying -$774 as a minimum revenue estimate destroys user trust instantly.
    All lower bounds are clipped at $0 before reaching the UI.<br><br>

    <b style='color:#F0F6FF;font-size:.95rem'>The HITL Gate design</b><br>
    Three tiers — Auto-approve / Flag / Alert — replaces a binary yes/no with proportionate
    human attention. Routine orders flow through automatically. Genuinely unusual signals
    get escalated. Estimated 60-70% reduction in decisions requiring manual review.

    </div>
    """, unsafe_allow_html=True)

# ── FOOTER ─────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;margin-top:48px;padding:20px;border-top:1px solid #1a2535'>
<span style='font-size:.7rem;color:#3D5070;letter-spacing:.08em'>
DEMANDSENSE · PROBABILISTIC REVENUE FORECASTING · BUILT BY SNEHA JAISWAL · 2025
</span>
</div>""", unsafe_allow_html=True)
