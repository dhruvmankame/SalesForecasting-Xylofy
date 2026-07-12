"""
Xylofy AI — Week 3 & 4 Internship Project
Sales Forecasting & Analytics Dashboard
Author: Dhruv
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# Page Config & Custom CSS
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Xylofy AI — Sales Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Colour palette (consistent throughout)
COLORS = {
    "primary": "#6366F1",    # indigo
    "secondary": "#8B5CF6",  # violet
    "accent": "#EC4899",     # pink
    "success": "#10B981",    # emerald
    "warning": "#F59E0B",    # amber
    "danger": "#EF4444",     # red
    "bg_card": "#1E1E2E",
    "text": "#E2E8F0",
}

CATEGORY_COLORS = {
    "Furniture": "#6366F1",
    "Technology": "#EC4899",
    "Office Supplies": "#10B981",
}

REGION_COLORS = {
    "West": "#6366F1",
    "East": "#EC4899",
    "Central": "#10B981",
    "South": "#F59E0B",
}

PLOTLY_TEMPLATE = "plotly_dark"

st.markdown(
    """
    <style>
    /* Global */
    .stApp {background-color: #0F0F1A;}
    section[data-testid="stSidebar"] {background-color: #161625;}

    /* KPI cards */
    .kpi-card {
        background: linear-gradient(135deg, #1E1E2E 0%, #252540 100%);
        border: 1px solid #2D2D4A;
        border-radius: 14px;
        padding: 1.3rem 1.5rem;
        text-align: center;
    }
    .kpi-value {font-size: 2rem; font-weight: 800; color: #E2E8F0;}
    .kpi-label {font-size: 0.85rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 1px;}

    /* Section header */
    .section-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: #C4B5FD;
        margin-top: 1.6rem;
        margin-bottom: 0.5rem;
        border-left: 4px solid #6366F1;
        padding-left: 0.7rem;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #64748B;
        font-size: 0.78rem;
        padding: 2rem 0 1rem;
        border-top: 1px solid #1E293B;
        margin-top: 3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset …")
def load_data():
    """Load and preprocess the Superstore Sales dataset."""
    df = pd.read_csv("train.csv", parse_dates=["Order Date", "Ship Date"], dayfirst=True)
    df.columns = df.columns.str.strip()
    # Derived columns
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    df["Quarter"] = df["Order Date"].dt.quarter
    df["YearMonth"] = df["Order Date"].dt.to_period("M").astype(str)
    return df


def render_footer():
    st.markdown(
        '<div class="footer">Xylofy AI — Week 3 &amp; 4 Internship Project &nbsp;|&nbsp; Built by <b>Dhruv</b></div>',
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
st.sidebar.image(
    "https://img.icons8.com/fluency/96/combo-chart.png",
    width=72,
)
st.sidebar.markdown("## 📊 Sales Analytics")
page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Sales Overview",
        "🔮 Forecast Explorer",
        "🚨 Anomaly Report",
        "📦 Product Demand Segments",
    ],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.caption("Author: **Dhruv**")
st.sidebar.caption("Xylofy AI — Week 3 & 4")

# Load data once
try:
    df = load_data()
except Exception as e:
    st.error(f"❌ Could not load `train.csv`: {e}")
    st.stop()


# ══════════════════════════════════════════════
# PAGE 1 — Sales Overview Dashboard
# ══════════════════════════════════════════════
if page == "🏠 Sales Overview":
    st.markdown("# 🏠 Sales Overview Dashboard")
    st.caption("High-level KPIs and interactive breakdowns of Superstore sales data.")

    # ── KPI Cards ─────────────────────────────
    total_revenue = df["Sales"].sum()
    total_orders = df["Order ID"].nunique()
    avg_order = total_revenue / total_orders if total_orders else 0
    total_customers = df["Customer ID"].nunique()

    k1, k2, k3, k4 = st.columns(4)
    for col, emoji, label, value in [
        (k1, "💰", "Total Revenue", f"${total_revenue:,.0f}"),
        (k2, "📦", "Total Orders", f"{total_orders:,}"),
        (k3, "🧾", "Avg Order Value", f"${avg_order:,.2f}"),
        (k4, "👥", "Total Customers", f"{total_customers:,}"),
    ]:
        col.markdown(
            f'<div class="kpi-card"><div class="kpi-label">{emoji} {label}</div>'
            f'<div class="kpi-value">{value}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")  # spacer

    # ── Yearly Sales Bar Chart ────────────────
    st.markdown('<div class="section-header">Total Sales by Year</div>', unsafe_allow_html=True)
    yearly = df.groupby("Year", as_index=False)["Sales"].sum()
    fig_year = px.bar(
        yearly,
        x="Year",
        y="Sales",
        text_auto="$.2s",
        color_discrete_sequence=[COLORS["primary"]],
        template=PLOTLY_TEMPLATE,
    )
    fig_year.update_layout(
        yaxis_title="Sales ($)",
        xaxis_title="",
        xaxis=dict(dtick=1),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=380,
    )
    st.plotly_chart(fig_year, use_container_width=True)

    # ── Monthly Trend ─────────────────────────
    st.markdown('<div class="section-header">Monthly Sales Trend</div>', unsafe_allow_html=True)
    monthly = df.groupby("YearMonth", as_index=False)["Sales"].sum().sort_values("YearMonth")
    fig_month = px.line(
        monthly,
        x="YearMonth",
        y="Sales",
        markers=True,
        color_discrete_sequence=[COLORS["accent"]],
        template=PLOTLY_TEMPLATE,
    )
    fig_month.update_layout(
        yaxis_title="Sales ($)",
        xaxis_title="",
        xaxis=dict(tickangle=-45),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=400,
    )
    st.plotly_chart(fig_month, use_container_width=True)

    # ── Region × Category Breakdown ──────────
    st.markdown('<div class="section-header">Sales by Region & Category</div>', unsafe_allow_html=True)

    fc1, fc2 = st.columns(2)
    with fc1:
        sel_regions = st.multiselect(
            "Filter Regions",
            options=sorted(df["Region"].unique()),
            default=sorted(df["Region"].unique()),
        )
    with fc2:
        sel_categories = st.multiselect(
            "Filter Categories",
            options=sorted(df["Category"].unique()),
            default=sorted(df["Category"].unique()),
        )

    filtered = df[df["Region"].isin(sel_regions) & df["Category"].isin(sel_categories)]

    c1, c2 = st.columns(2)
    with c1:
        region_sales = filtered.groupby("Region", as_index=False)["Sales"].sum()
        fig_r = px.bar(
            region_sales,
            x="Region",
            y="Sales",
            color="Region",
            color_discrete_map=REGION_COLORS,
            text_auto="$.2s",
            template=PLOTLY_TEMPLATE,
        )
        fig_r.update_layout(
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=370,
            yaxis_title="Sales ($)",
        )
        st.plotly_chart(fig_r, use_container_width=True)

    with c2:
        cat_sales = filtered.groupby("Category", as_index=False)["Sales"].sum()
        fig_c = px.bar(
            cat_sales,
            x="Category",
            y="Sales",
            color="Category",
            color_discrete_map=CATEGORY_COLORS,
            text_auto="$.2s",
            template=PLOTLY_TEMPLATE,
        )
        fig_c.update_layout(
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=370,
            yaxis_title="Sales ($)",
        )
        st.plotly_chart(fig_c, use_container_width=True)

    # Heatmap — Region × Category
    pivot = filtered.pivot_table(values="Sales", index="Region", columns="Category", aggfunc="sum", fill_value=0)
    fig_heat = px.imshow(
        pivot,
        text_auto="$.2s",
        color_continuous_scale="Viridis",
        template=PLOTLY_TEMPLATE,
        aspect="auto",
    )
    fig_heat.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        height=340,
        title="Region × Category Heatmap",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    render_footer()


# ══════════════════════════════════════════════
# PAGE 2 — Forecast Explorer
# ══════════════════════════════════════════════
elif page == "🔮 Forecast Explorer":
    st.markdown("# 🔮 Forecast Explorer")
    st.caption("XGBoost-based monthly sales forecasting with lag features.")

    # ── Controls ──────────────────────────────
    col_a, col_b = st.columns([2, 1])
    with col_a:
        segment_options = (
            ["Overall"]
            + [f"Category: {c}" for c in sorted(df["Category"].unique())]
            + [f"Region: {r}" for r in sorted(df["Region"].unique())]
        )
        segment = st.selectbox("Select Segment", segment_options)
    with col_b:
        horizon = st.slider("Forecast Horizon (months)", min_value=1, max_value=3, value=2)

    # ── Filter data ───────────────────────────
    if segment == "Overall":
        sub = df.copy()
    elif segment.startswith("Category:"):
        sub = df[df["Category"] == segment.split(": ", 1)[1]]
    else:
        sub = df[df["Region"] == segment.split(": ", 1)[1]]

    monthly = (
        sub.groupby(sub["Order Date"].dt.to_period("M"))["Sales"]
        .sum()
        .sort_index()
        .reset_index()
    )
    monthly.columns = ["Period", "Sales"]
    monthly["ds"] = monthly["Period"].dt.to_timestamp()

    # ── Feature engineering ───────────────────
    def build_features(series: pd.DataFrame) -> pd.DataFrame:
        """Create lag + rolling features for XGBoost."""
        frame = series.copy()
        frame["Lag1"] = frame["Sales"].shift(1)
        frame["Lag2"] = frame["Sales"].shift(2)
        frame["Lag3"] = frame["Sales"].shift(3)
        frame["RollingMean3"] = frame["Sales"].shift(1).rolling(3).mean()
        frame["Month"] = frame["ds"].dt.month
        frame["Quarter"] = frame["ds"].dt.quarter
        return frame

    feat = build_features(monthly).dropna()

    if len(feat) < 8:
        st.warning("⚠️ Not enough data points to train a reliable model for this segment.")
        st.stop()

    feature_cols = ["Lag1", "Lag2", "Lag3", "RollingMean3", "Month", "Quarter"]

    # Train / test split (last 6 months = test)
    test_size = min(6, len(feat) // 3)
    train_df = feat.iloc[:-test_size]
    test_df = feat.iloc[-test_size:]

    X_train, y_train = train_df[feature_cols], train_df["Sales"]
    X_test, y_test = test_df[feature_cols], test_df["Sales"]

    # ── Model training ────────────────────────
    from xgboost import XGBRegressor  # noqa: E402  (deferred import for speed)

    @st.cache_data(show_spinner="Training XGBoost …")
    def train_xgb(_X_train, _y_train, _X_test, _y_test, _seg, _hor):
        model = XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.8,
            random_state=42,
        )
        model.fit(_X_train, _y_train, eval_set=[(_X_test, _y_test)], verbose=False)
        preds_test = model.predict(_X_test)
        mae = mean_absolute_error(_y_test, preds_test)
        rmse = np.sqrt(mean_squared_error(_y_test, preds_test))
        return model, preds_test, mae, rmse

    model, preds_test, mae, rmse = train_xgb(
        X_train.values, y_train.values, X_test.values, y_test.values, segment, horizon
    )

    # ── Recursive future forecast ─────────────
    last_vals = list(feat["Sales"].values[-3:])  # last 3 sales for lags
    last_date = feat["ds"].iloc[-1]
    future_rows = []
    for i in range(1, horizon + 1):
        next_date = last_date + pd.DateOffset(months=i)
        lag1 = last_vals[-1]
        lag2 = last_vals[-2]
        lag3 = last_vals[-3]
        rm3 = np.mean(last_vals[-3:])
        row = [lag1, lag2, lag3, rm3, next_date.month, (next_date.month - 1) // 3 + 1]
        pred = model.predict(np.array([row]))[0]
        pred = max(pred, 0)  # clamp
        future_rows.append({"ds": next_date, "Forecast": pred})
        last_vals.append(pred)

    future_df = pd.DataFrame(future_rows)

    # ── Chart ─────────────────────────────────
    fig = go.Figure()
    # Historical
    fig.add_trace(
        go.Scatter(
            x=feat["ds"],
            y=feat["Sales"],
            mode="lines+markers",
            name="Historical",
            line=dict(color=COLORS["primary"], width=2),
            marker=dict(size=5),
        )
    )
    # Test predictions overlay
    fig.add_trace(
        go.Scatter(
            x=test_df["ds"],
            y=preds_test,
            mode="lines+markers",
            name="Test Prediction",
            line=dict(color=COLORS["success"], dash="dash", width=2),
            marker=dict(size=6),
        )
    )
    # Future forecast
    bridge_x = [feat["ds"].iloc[-1]] + list(future_df["ds"])
    bridge_y = [feat["Sales"].iloc[-1]] + list(future_df["Forecast"])
    fig.add_trace(
        go.Scatter(
            x=bridge_x,
            y=bridge_y,
            mode="lines+markers",
            name="Forecast",
            line=dict(color=COLORS["accent"], width=3),
            marker=dict(size=8, symbol="diamond"),
        )
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=480,
        yaxis_title="Sales ($)",
        xaxis_title="",
        legend=dict(orientation="h", y=-0.15),
        title=f"Sales Forecast — {segment}",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Metrics ───────────────────────────────
    m1, m2, m3 = st.columns(3)
    m1.metric("📏 MAE (test)", f"${mae:,.0f}")
    m2.metric("📐 RMSE (test)", f"${rmse:,.0f}")
    m3.metric("📅 Months Forecasted", horizon)

    # Forecast table
    st.markdown('<div class="section-header">Forecasted Values</div>', unsafe_allow_html=True)
    display_future = future_df.copy()
    display_future["Month"] = display_future["ds"].dt.strftime("%B %Y")
    display_future["Forecast ($)"] = display_future["Forecast"].apply(lambda x: f"${x:,.0f}")
    st.dataframe(
        display_future[["Month", "Forecast ($)"]].reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )

    render_footer()


# ══════════════════════════════════════════════
# PAGE 3 — Anomaly Report
# ══════════════════════════════════════════════
elif page == "🚨 Anomaly Report":
    st.markdown("# 🚨 Anomaly Report")
    st.caption("Isolation Forest anomaly detection on weekly aggregated sales.")

    # ── Weekly aggregation ────────────────────
    weekly = (
        df.set_index("Order Date")
        .resample("W")["Sales"]
        .sum()
        .reset_index()
    )
    weekly.columns = ["Week", "Sales"]

    # ── Isolation Forest ──────────────────────
    @st.cache_data(show_spinner="Detecting anomalies …")
    def detect_anomalies(sales_values):
        iso = IsolationForest(
            n_estimators=200,
            contamination=0.07,
            random_state=42,
        )
        labels = iso.fit_predict(sales_values.reshape(-1, 1))
        scores = iso.decision_function(sales_values.reshape(-1, 1))
        return labels, scores

    labels, scores = detect_anomalies(weekly["Sales"].values)
    weekly["Anomaly"] = labels
    weekly["Score"] = scores
    anomalies = weekly[weekly["Anomaly"] == -1].copy()

    # ── Explanation heuristic ─────────────────
    median_sales = weekly["Sales"].median()
    q75 = weekly["Sales"].quantile(0.75)

    def explain(row):
        if row["Sales"] > q75 * 1.3:
            return "🔺 Unusually high sales — possible seasonal peak or large bulk order"
        elif row["Sales"] < median_sales * 0.35:
            return "🔻 Unusually low sales — possible holiday lull or data gap"
        elif row["Sales"] > q75:
            return "🔺 Above-normal spike"
        else:
            return "⚠️ Unusual pattern detected"

    anomalies["Explanation"] = anomalies.apply(explain, axis=1)

    # ── Chart ─────────────────────────────────
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=weekly["Week"],
            y=weekly["Sales"],
            mode="lines",
            name="Weekly Sales",
            line=dict(color=COLORS["primary"], width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=anomalies["Week"],
            y=anomalies["Sales"],
            mode="markers",
            name="Anomaly",
            marker=dict(color=COLORS["danger"], size=10, symbol="x", line=dict(width=2, color="white")),
        )
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=480,
        yaxis_title="Sales ($)",
        xaxis_title="",
        title="Weekly Sales with Anomalies Highlighted",
        legend=dict(orientation="h", y=-0.12),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Summary metrics ───────────────────────
    s1, s2, s3 = st.columns(3)
    s1.metric("Total Weeks", len(weekly))
    s2.metric("Anomalies Detected", len(anomalies))
    s3.metric("Anomaly Rate", f"{len(anomalies)/len(weekly)*100:.1f}%")

    # ── Anomaly Table ─────────────────────────
    st.markdown('<div class="section-header">Detected Anomaly Weeks</div>', unsafe_allow_html=True)
    display_anom = anomalies[["Week", "Sales", "Score", "Explanation"]].copy()
    display_anom["Week"] = display_anom["Week"].dt.strftime("%d %b %Y")
    display_anom["Sales"] = display_anom["Sales"].apply(lambda x: f"${x:,.0f}")
    display_anom["Score"] = display_anom["Score"].round(3)
    st.dataframe(display_anom.reset_index(drop=True), use_container_width=True, hide_index=True)

    # ── Interpretation ────────────────────────
    st.markdown('<div class="section-header">Interpretation</div>', unsafe_allow_html=True)
    st.info(
        "**Isolation Forest** identifies data points that are easy to isolate from the rest, "
        "i.e., they deviate significantly from normal patterns.\n\n"
        f"• **{len(anomalies[anomalies['Sales'] > median_sales])}** anomalies are **high-sales spikes** — "
        "these often correspond to seasonal promotions, bulk orders, or year-end pushes.\n\n"
        f"• **{len(anomalies[anomalies['Sales'] <= median_sales])}** anomalies are **low-sales dips** — "
        "these may indicate holiday periods, supply disruptions, or data collection gaps.\n\n"
        "Use these flags to investigate root causes and refine inventory planning."
    )

    render_footer()


# ══════════════════════════════════════════════
# PAGE 4 — Product Demand Segments
# ══════════════════════════════════════════════
elif page == "📦 Product Demand Segments":
    st.markdown("# 📦 Product Demand Segments")
    st.caption("K-Means clustering on sub-category sales behaviour with PCA visualisation.")

    # ── Sub-category features ─────────────────
    @st.cache_data(show_spinner="Building cluster features …")
    def build_cluster_data(dataframe):
        monthly_sub = (
            dataframe.groupby([dataframe["Order Date"].dt.to_period("M"), "Sub-Category"])["Sales"]
            .sum()
            .reset_index()
        )
        monthly_sub.columns = ["Period", "Sub-Category", "Sales"]

        agg = dataframe.groupby("Sub-Category").agg(
            TotalSales=("Sales", "sum"),
            AvgOrderValue=("Sales", "mean"),
            OrderCount=("Sales", "count"),
        ).reset_index()

        # Growth rate (last 12 months vs first 12 months)
        monthly_sub["PeriodTs"] = monthly_sub["Period"].dt.to_timestamp()
        mid = monthly_sub["PeriodTs"].quantile(0.5)
        first_half = monthly_sub[monthly_sub["PeriodTs"] <= mid].groupby("Sub-Category")["Sales"].sum()
        second_half = monthly_sub[monthly_sub["PeriodTs"] > mid].groupby("Sub-Category")["Sales"].sum()
        growth = ((second_half - first_half) / first_half.replace(0, np.nan)).fillna(0).reset_index()
        growth.columns = ["Sub-Category", "GrowthRate"]

        # Volatility (CV)
        vol = monthly_sub.groupby("Sub-Category")["Sales"].std() / monthly_sub.groupby("Sub-Category")["Sales"].mean()
        vol = vol.fillna(0).reset_index()
        vol.columns = ["Sub-Category", "Volatility"]

        features = agg.merge(growth, on="Sub-Category").merge(vol, on="Sub-Category")
        return features

    cluster_df = build_cluster_data(df)
    feature_cols_cl = ["TotalSales", "GrowthRate", "Volatility", "AvgOrderValue"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(cluster_df[feature_cols_cl])

    # ── Elbow method ──────────────────────────
    @st.cache_data(show_spinner="Running elbow analysis …")
    def elbow_analysis(X):
        inertias = []
        K_range = range(2, min(8, len(X)))
        for k in K_range:
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            km.fit(X)
            inertias.append(km.inertia_)
        # Simple elbow: largest drop-off
        diffs = [inertias[i] - inertias[i + 1] for i in range(len(inertias) - 1)]
        best_k = list(K_range)[np.argmax(diffs) + 1] if diffs else 3
        return best_k, list(K_range), inertias

    best_k, k_range, inertias = elbow_analysis(X_scaled)

    # Show elbow chart
    with st.expander("📐 Elbow Curve (click to expand)"):
        fig_elbow = px.line(
            x=k_range,
            y=inertias,
            markers=True,
            labels={"x": "Number of Clusters (K)", "y": "Inertia"},
            template=PLOTLY_TEMPLATE,
        )
        fig_elbow.add_vline(x=best_k, line_dash="dash", line_color=COLORS["accent"])
        fig_elbow.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=320,
            title=f"Optimal K = {best_k}",
        )
        st.plotly_chart(fig_elbow, use_container_width=True)

    # ── K-Means ───────────────────────────────
    km = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    cluster_df["Cluster"] = km.fit_predict(X_scaled)

    # ── Meaningful labels ─────────────────────
    cluster_profiles = cluster_df.groupby("Cluster")[feature_cols_cl].mean()

    def label_cluster(row):
        if row["TotalSales"] > cluster_profiles["TotalSales"].median() and row["GrowthRate"] > cluster_profiles["GrowthRate"].median():
            return "⭐ High Demand, High Growth"
        elif row["TotalSales"] > cluster_profiles["TotalSales"].median():
            return "💎 High Demand, Stable"
        elif row["GrowthRate"] > cluster_profiles["GrowthRate"].median() and row["Volatility"] < cluster_profiles["Volatility"].median():
            return "🌱 Emerging, Steady"
        elif row["Volatility"] > cluster_profiles["Volatility"].median():
            return "⚡ Volatile / Seasonal"
        else:
            return "📉 Low Demand, Slow Growth"

    label_map = {idx: label_cluster(row) for idx, row in cluster_profiles.iterrows()}
    cluster_df["Segment"] = cluster_df["Cluster"].map(label_map)

    # ── Stocking strategy ─────────────────────
    def stocking_strategy(segment):
        if "High Demand, High Growth" in segment:
            return "Aggressive stocking — increase safety stock & reorder frequency"
        elif "High Demand, Stable" in segment:
            return "Maintain healthy stock levels — reliable demand, standard reorder"
        elif "Emerging" in segment:
            return "Moderate stocking with upward trend monitoring"
        elif "Volatile" in segment:
            return "Flexible / JIT stocking — demand is unpredictable, avoid overstocking"
        else:
            return "Lean inventory — low priority, reduce holding costs"

    cluster_df["Stocking Strategy"] = cluster_df["Segment"].apply(stocking_strategy)

    # ── PCA Scatter ───────────────────────────
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    cluster_df["PC1"] = coords[:, 0]
    cluster_df["PC2"] = coords[:, 1]

    fig_pca = px.scatter(
        cluster_df,
        x="PC1",
        y="PC2",
        color="Segment",
        hover_data=["Sub-Category", "TotalSales", "GrowthRate"],
        text="Sub-Category",
        template=PLOTLY_TEMPLATE,
        title="Product Demand Clusters (PCA Projection)",
    )
    fig_pca.update_traces(textposition="top center", marker=dict(size=12, line=dict(width=1, color="white")))
    fig_pca.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=560,
        legend=dict(orientation="h", y=-0.18),
    )
    st.plotly_chart(fig_pca, use_container_width=True)

    # Explained variance
    ev1, ev2 = pca.explained_variance_ratio_
    st.caption(f"PCA explains **{(ev1+ev2)*100:.1f}%** of variance (PC1: {ev1*100:.1f}%, PC2: {ev2*100:.1f}%)")

    # ── Cluster Table ─────────────────────────
    st.markdown('<div class="section-header">Cluster Membership & Strategies</div>', unsafe_allow_html=True)
    display_cl = cluster_df[
        ["Sub-Category", "Segment", "TotalSales", "GrowthRate", "Volatility", "AvgOrderValue", "Stocking Strategy"]
    ].copy()
    display_cl["TotalSales"] = display_cl["TotalSales"].apply(lambda x: f"${x:,.0f}")
    display_cl["GrowthRate"] = display_cl["GrowthRate"].apply(lambda x: f"{x*100:.1f}%")
    display_cl["Volatility"] = display_cl["Volatility"].apply(lambda x: f"{x:.2f}")
    display_cl["AvgOrderValue"] = display_cl["AvgOrderValue"].apply(lambda x: f"${x:,.0f}")
    display_cl = display_cl.sort_values("Segment")
    st.dataframe(display_cl.reset_index(drop=True), use_container_width=True, hide_index=True)

    render_footer()
