import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from pathlib import Path

BASE = Path(__file__).parent
MAE_TEST = 225895          # XGBoost test MAE, used for the forecast error band
INDIGO, TEAL, RED, GREEN = "#4F46E5", "#14B8A6", "#EF4444", "#22C55E"

st.set_page_config(page_title="ATM Cash Predictor", page_icon="🏧", layout="wide")

# ---------- styling ----------
st.markdown(f"""
<style>
.block-container {{padding-top: 1.5rem; max-width: 1200px;}}
.hero {{
    background: linear-gradient(120deg, {INDIGO} 0%, #7C3AED 55%, {TEAL} 100%);
    padding: 28px 34px; border-radius: 18px; margin-bottom: 18px; color: white;
}}
.hero h1 {{margin: 0; font-size: 2.1rem; color: white;}}
.hero p {{margin: 6px 0 0 0; opacity: 0.92; font-size: 1.02rem;}}
.kpi {{
    border-radius: 14px; padding: 16px 20px;
    background: rgba(128,128,128,0.08);
    border: 1px solid rgba(128,128,128,0.22);
    border-left: 6px solid var(--c);
    min-height: 108px;
}}
.kpi .t {{font-size: 0.82rem; opacity: 0.7; text-transform: uppercase; letter-spacing: 0.04em;}}
.kpi .v {{font-size: 1.75rem; font-weight: 700; margin-top: 4px;}}
.kpi .s {{font-size: 0.82rem; opacity: 0.7; margin-top: 2px;}}
.stTabs [data-baseweb="tab-list"] {{gap: 6px;}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 10px 10px 0 0; padding: 10px 18px; font-weight: 600;
}}
</style>
""", unsafe_allow_html=True)


def kpi(title, value, sub="", color=INDIGO):
    return (f'<div class="kpi" style="--c:{color}"><div class="t">{title}</div>'
            f'<div class="v">{value}</div><div class="s">{sub}</div></div>')


def style_fig(fig, height=400):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified", legend=dict(orientation="h", y=1.12, x=0),
        yaxis_title="Cash withdrawn (₦)",
    )
    fig.update_yaxes(tickformat=",.0f")
    return fig


# ---------- load model and data ----------
@st.cache_resource
def load_model():
    b = joblib.load(BASE / "models" / "atm_model.pkl")
    return b["model"], b["feature_cols"]


@st.cache_data
def load_data():
    hist = pd.read_csv(BASE / "data" / "processed" / "daily_history.csv", parse_dates=["date"])
    locs = pd.read_csv(BASE / "data" / "processed" / "locations.csv")
    return hist, locs


model, feature_cols = load_model()
hist, locs = load_data()


# ---------- prediction helpers ----------
def build_features(h, loc_id, d, n_atms):
    d = pd.Timestamp(d)
    prev = h.loc[d - pd.Timedelta(days=7): d - pd.Timedelta(days=1)]
    if len(prev) < 7:
        return None
    row = {
        "dayofweek": d.dayofweek, "day": d.day,
        "is_weekend": int(d.dayofweek >= 5),
        "is_salary_week": int(d.day <= 7 or d.day >= 28),
        "lag_1": prev["total_amount"].iloc[-1],
        "lag_7": prev["total_amount"].iloc[0],
        "roll_7": prev["total_amount"].mean(),
        "txn_lag_1": prev["txn_count"].iloc[-1],
        "n_atms": n_atms,
    }
    X = pd.DataFrame([row]).reindex(columns=feature_cols, fill_value=0)
    col = f"loc_{loc_id}"
    if col in X.columns:
        X[col] = 1
    return X


def predict_one(h, loc_id, d, n_atms):
    X = build_features(h, loc_id, d, n_atms)
    return None if X is None else max(0.0, float(model.predict(X)[0]))


def forecast(h, loc_id, n_atms, days):
    hh = h.copy()
    out = []
    for _ in range(days):
        d = hh.index[-1] + pd.Timedelta(days=1)
        p = predict_one(hh, loc_id, d, n_atms)
        hh.loc[d] = [p, hh["txn_count"].iloc[-7:].mean()]
        out.append((d, p))
    return pd.DataFrame(out, columns=["date", "predicted"]).set_index("date")


# ---------- sidebar ----------
st.sidebar.markdown("## 🏧 Controls")
locs["label"] = locs["location_name"] + " · " + locs["City"] + " (" + locs["locationid"] + ")"
label = st.sidebar.selectbox("ATM location", locs["label"].tolist())
loc_row = locs[locs["label"] == label].iloc[0]
loc_id, n_atms = loc_row["locationid"], int(loc_row["n_atms"])

buffer_pct = st.sidebar.slider("Safety buffer (%)", 0, 30, 10,
                               help="Extra cash on top of the prediction so the ATM does not run out.")

h = (hist[hist["locationid"] == loc_id].set_index("date").sort_index()[["total_amount", "txn_count"]])

st.sidebar.markdown("---")
st.sidebar.markdown(f"**{loc_row['location_name']}**")
st.sidebar.caption(f"{loc_row['City']} · {n_atms} ATMs")
st.sidebar.metric("Average per day", f"₦{h['total_amount'].mean():,.0f}")
st.sidebar.metric("Busiest day on record", f"₦{h['total_amount'].max():,.0f}")

# ---------- header ----------
st.markdown("""
<div class="hero">
  <h1>ATM Cash Requirement Prediction</h1>
  <p>Machine learning forecasts of daily ATM cash demand, so every ATM has enough cash without tying up too much.</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📅 Predict a day", "🔮 Forecast ahead", "📈 Insights", "📊 Model performance", "ℹ️ About"])

# ===== Tab 1: single day =====
with tab1:
    min_d = (h.index[0] + pd.Timedelta(days=7)).date()
    max_d = (h.index[-1] + pd.Timedelta(days=1)).date()
    d_sel = st.date_input("Pick a date", value=h.index[-1].date(), min_value=min_d, max_value=max_d)
    d_ts = pd.Timestamp(d_sel)

    pred = predict_one(h, loc_id, d_ts, n_atms)
    actual = float(h.loc[d_ts, "total_amount"]) if d_ts in h.index else None

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi("Predicted need", f"₦{pred:,.0f}", d_ts.strftime("%A, %d %b %Y"), INDIGO), unsafe_allow_html=True)
    c2.markdown(kpi("Cash to load", f"₦{pred * (1 + buffer_pct / 100):,.0f}", f"Includes {buffer_pct}% buffer", TEAL),
                unsafe_allow_html=True)
    if actual is not None:
        acc = max(0.0, 100 - abs(pred - actual) / max(actual, 1) * 100)
        c3.markdown(kpi("Actual withdrawn", f"₦{actual:,.0f}", "From the dataset", GREEN), unsafe_allow_html=True)
        c4.markdown(kpi("Accuracy for this day", f"{acc:.0f}%", f"Error: ₦{pred - actual:+,.0f}", RED),
                    unsafe_allow_html=True)
    else:
        c3.markdown(kpi("Actual withdrawn", "N/A", "Date is beyond the dataset", GREEN), unsafe_allow_html=True)
        c4.markdown(kpi("Accuracy for this day", "N/A", "No actual to compare", RED), unsafe_allow_html=True)

    past = h.loc[d_ts - pd.Timedelta(days=30): d_ts - pd.Timedelta(days=1)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=past.index, y=past["total_amount"], mode="lines+markers",
                             name="Actual (previous 30 days)", line=dict(color=INDIGO, width=2),
                             fill="tozeroy", fillcolor="rgba(79,70,229,0.08)",
                             hovertemplate="₦%{y:,.0f}"))
    if actual is not None:
        fig.add_trace(go.Scatter(x=[d_ts], y=[actual], mode="markers", name="Actual on this day",
                                 marker=dict(color=GREEN, size=13), hovertemplate="₦%{y:,.0f}"))
    fig.add_trace(go.Scatter(x=[d_ts], y=[pred], mode="markers", name="Predicted",
                             marker=dict(color=RED, size=15, symbol="diamond"), hovertemplate="₦%{y:,.0f}"))
    st.plotly_chart(style_fig(fig))

# ===== Tab 2: forecast =====
with tab2:
    days = st.slider("How many days ahead?", 1, 14, 7)
    fc = forecast(h, loc_id, n_atms, days)
    total = fc["predicted"].sum()

    c1, c2, c3 = st.columns(3)
    c1.markdown(kpi(f"Total, next {days} days", f"₦{total:,.0f}", "Predicted withdrawals", INDIGO), unsafe_allow_html=True)
    c2.markdown(kpi("Cash to load", f"₦{total * (1 + buffer_pct / 100):,.0f}", f"Includes {buffer_pct}% buffer", TEAL),
                unsafe_allow_html=True)
    peak = fc["predicted"].idxmax()
    c3.markdown(kpi("Busiest predicted day", peak.strftime("%a, %d %b"), f"₦{fc['predicted'].max():,.0f}", RED),
                unsafe_allow_html=True)

    recent = h.iloc[-30:]
    upper = fc["predicted"] + MAE_TEST
    lower = (fc["predicted"] - MAE_TEST).clip(lower=0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recent.index, y=recent["total_amount"], mode="lines+markers",
                             name="Actual (last 30 days of data)", line=dict(color=INDIGO, width=2),
                             hovertemplate="₦%{y:,.0f}"))
    fig.add_trace(go.Scatter(x=fc.index, y=upper, mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fc.index, y=lower, mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(239,68,68,0.15)", name="Typical error range (±MAE)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fc.index, y=fc["predicted"], mode="lines+markers", name="Forecast",
                             line=dict(color=RED, width=3), hovertemplate="₦%{y:,.0f}"))
    st.plotly_chart(style_fig(fig))

    table = fc.copy()
    table.insert(0, "Day", table.index.strftime("%A"))
    table["Cash to load (₦)"] = (table["predicted"] * (1 + buffer_pct / 100)).round(0).astype(int)
    table["predicted"] = table["predicted"].round(0).astype(int)
    table.index = table.index.strftime("%d %b %Y")
    table = table.rename(columns={"predicted": "Predicted (₦)"})
    st.dataframe(table)
    st.download_button("⬇️ Download forecast as CSV", table.to_csv().encode("utf-8"),
                       file_name=f"forecast_{loc_id}.csv", mime="text/csv")
    st.caption("Each forecast day uses earlier forecasts as history, so uncertainty grows the further ahead you look.")

# ===== Tab 3: insights =====
with tab3:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Average withdrawals by weekday")
        names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        wd = h.groupby(h.index.dayofweek)["total_amount"].mean().reindex(range(7))
        colors = [TEAL if i >= 5 else INDIGO for i in range(7)]
        fig = go.Figure(go.Bar(x=names, y=wd.values, marker_color=colors, hovertemplate="₦%{y:,.0f}"))
        st.plotly_chart(style_fig(fig, 340))

    with c2:
        st.markdown("#### Daily trend (7-day average)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=h.index, y=h["total_amount"], mode="lines", name="Daily",
                                 line=dict(color="rgba(79,70,229,0.3)", width=1), hovertemplate="₦%{y:,.0f}"))
        fig.add_trace(go.Scatter(x=h.index, y=h["total_amount"].rolling(7).mean(), mode="lines",
                                 name="7-day average", line=dict(color=INDIGO, width=3), hovertemplate="₦%{y:,.0f}"))
        st.plotly_chart(style_fig(fig, 340))

    st.markdown("#### Top 10 locations by average daily withdrawals")
    top = (hist.groupby("locationid")["total_amount"].mean().reset_index()
           .merge(locs[["locationid", "location_name", "City"]], on="locationid")
           .sort_values("total_amount", ascending=False).head(10).iloc[::-1])
    top["name"] = top["location_name"] + " (" + top["City"] + ")"
    fig = go.Figure(go.Bar(x=top["total_amount"], y=top["name"], orientation="h",
                           marker_color=[RED if i == loc_id else INDIGO for i in top["locationid"]],
                           hovertemplate="₦%{x:,.0f}"))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Average daily withdrawals (₦)")
    fig.update_xaxes(tickformat=",.0f")
    st.plotly_chart(fig)
    st.caption("The selected location is highlighted in red if it appears in the top 10.")

# ===== Tab 4: model performance =====
with tab4:
    results = pd.DataFrame({
        "Model": ["XGBoost", "Random Forest", "Linear Regression", "Decision Tree", "Baseline (same as last week)"],
        "MAE": [225895, 226979, 237758, 236948, 311663],
        "RMSE": [310874, 311218, 323863, 332770, 424138],
        "R2": [0.473, 0.472, 0.428, 0.396, 0.019],
    })
    improvement = (311663 - 225895) / 311663 * 100

    c1, c2, c3 = st.columns(3)
    c1.markdown(kpi("Best model", "XGBoost", "Lowest error on the test set", INDIGO), unsafe_allow_html=True)
    c2.markdown(kpi("Error vs baseline", f"{improvement:.0f}% lower", "MAE vs 'same as last week'", GREEN),
                unsafe_allow_html=True)
    c3.markdown(kpi("R² score", "0.47", "Share of daily variation explained", TEAL), unsafe_allow_html=True)

    st.markdown("")
    c1, c2 = st.columns(2)
    bar_colors = [INDIGO, INDIGO, INDIGO, INDIGO, "#9CA3AF"]
    with c1:
        st.markdown("#### RMSE (lower is better)")
        fig = go.Figure(go.Bar(x=results["RMSE"], y=results["Model"], orientation="h",
                               marker_color=bar_colors, hovertemplate="%{x:,.0f}"))
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig)
    with c2:
        st.markdown("#### R² (higher is better)")
        fig = go.Figure(go.Bar(x=results["R2"], y=results["Model"], orientation="h",
                               marker_color=bar_colors, hovertemplate="%{x:.3f}"))
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig)

    st.dataframe(results.set_index("Model"))

    if hasattr(model, "feature_importances_"):
        st.markdown("#### What the model relies on most")
        imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(10).iloc[::-1]
        fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=TEAL))
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig)

# ===== Tab 5: about =====
with tab5:
    st.markdown("""
### About this project

**Problem:** ATMs that run out of cash frustrate customers, while ATMs loaded with too much cash tie up money and raise costs.
Predicting daily demand helps banks load the right amount.

**Data:** ATM transaction records from Wisabi Bank (Kaggle), covering 50 ATM locations across 5 Nigerian states for 2022.
Only withdrawals were used, summed per location per day.

**Features:** day of week, day of month, weekend flag, salary-week flag, previous-day withdrawals, same-day-last-week
withdrawals, 7-day average, previous-day transaction count, number of ATMs, and the location.

**Models compared:** Linear Regression, Decision Tree, Random Forest and XGBoost, against a naive baseline. XGBoost performed best.

**Limitations:** one year of data only (no seasonal or holiday effects), and data from a Nigerian bank, so amounts are in Naira.
The pipeline can be retrained on any ATM transaction data, including Indian bank data when available.
""")