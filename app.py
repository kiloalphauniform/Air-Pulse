import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go

from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------------
# AIRPULSE - AI AIR QUALITY INTELLIGENCE
# --------------------------------------------------

st.set_page_config(
    page_title="AirPulse | Air Quality Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif !important;
    }

    /* Tighten top whitespace */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }

    /* Hide Deploy button & footer for a clean app feel */
    .stDeployButton, footer {
        display: none !important;
    }

    /* Subtle hover highlight on KPI metric cards */
    [data-testid="stMetric"] {
        transition: border-color 0.2s ease, transform 0.2s ease;
    }
    [data-testid="stMetric"]:hover {
        border-color: #E07A5F !important;
        transform: translateY(-2px);
    }

    /* Cleaner, modern tab bar */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1.5rem;
        border-bottom: 1px solid #1F2937;
    }
    .stTabs [data-baseweb="tab"] {
        padding-bottom: 0.75rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

LOCATIONS = {
    "Pune": (18.5204, 73.8567),
    "Mumbai": (19.0760, 72.8777),
    "Delhi": (28.6139, 77.2090),
    "Jalgaon": (21.0077, 75.5626),
    "Bengaluru": (12.9716, 77.5946)
}

POLLUTANTS = [
    "pm2_5",
    "pm10",
    "nitrogen_dioxide",
    "carbon_monoxide",
    "ozone"
]

POLLUTANT_NAMES = {
    "pm2_5": "PM2.5",
    "pm10": "PM10",
    "nitrogen_dioxide": "NO₂",
    "carbon_monoxide": "CO",
    "ozone": "O₃"
}

FEATURES = POLLUTANTS + ["pm2_5_24h", "pm10_24h", "hour", "day", "month", "day_of_week"]


# --------------------------------------------------
# AQI CLASSIFICATION & HEALTH ADVISORIES
# --------------------------------------------------

def classify_aqi(aqi):
    if aqi <= 50:
        return "Good", "🟢", "#22C55E"
    elif aqi <= 100:
        return "Moderate", "🟡", "#EAB308"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "🟠", "#F97316"
    elif aqi <= 200:
        return "Unhealthy", "🔴", "#EF4444"
    elif aqi <= 300:
        return "Very Unhealthy", "🟣", "#A855F7"
    else:
        return "Hazardous", "🟤", "#991B1B"


def get_health_advisory(aqi):
    if aqi <= 50:
        return {
            "level": "success",
            "title": "GOOD — Safe for Outdoor Activity",
            "general": "Air quality is satisfactory with little to no health risk.",
            "sensitive": "No restrictions needed for children, elderly, or asthma patients.",
            "action": "Great conditions for outdoor workouts and ventilation."
        }
    elif aqi <= 100:
        return {
            "level": "info",
            "title": "MODERATE — Acceptable Air Quality",
            "general": "Air quality is acceptable for most people.",
            "sensitive": "Unusually sensitive individuals should watch for mild respiratory symptoms.",
            "action": "Normal outdoor activities are fine; keep windows open during low-traffic hours."
        }
    elif aqi <= 150:
        return {
            "level": "warning",
            "title": "UNHEALTHY FOR SENSITIVE GROUPS",
            "general": "General public is less likely to be affected.",
            "sensitive": "Children, older adults, and people with asthma/heart disease should limit prolonged outdoor exertion.",
            "action": "Consider wearing an N95 mask near heavy traffic and keep rescue inhalers handy."
        }
    elif aqi <= 200:
        return {
            "level": "error",
            "title": "UNHEALTHY — Limit Outdoor Exposure",
            "general": "Everyone may begin to experience respiratory irritation.",
            "sensitive": "Avoid prolonged outdoor exertion; stay indoors with air filtration.",
            "action": "Shift workouts indoors, close windows, and use N95 masks outdoors."
        }
    elif aqi <= 300:
        return {
            "level": "error",
            "title": "VERY UNHEALTHY — Health Alert",
            "general": "Significant health risk for everyone.",
            "sensitive": "Avoid all outdoor physical activity.",
            "action": "Keep doors/windows sealed, run air purifiers, and avoid unnecessary travel."
        }
    else:
        return {
            "level": "error",
            "title": "HAZARDOUS — Emergency Conditions",
            "general": "Emergency conditions affecting the entire population.",
            "sensitive": "Strict indoor isolation recommended.",
            "action": "Avoid all outdoor exposure and follow local public health advisories."
        }


# --------------------------------------------------
# DATA COLLECTION - OPEN-METEO AIR QUALITY + WEATHER
# --------------------------------------------------

@st.cache_data(ttl=1800)
def fetch_air_quality(city):
    latitude, longitude = LOCATIONS[city]
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(POLLUTANTS + ["us_aqi"]),
        "current": ",".join(POLLUTANTS + ["us_aqi"]),
        "past_days": 90,
        "forecast_days": 2,
        "timezone": "Asia/Kolkata"
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    if "hourly" not in data:
        raise ValueError("No hourly air-quality data returned.")

    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df["us_aqi"] = pd.to_numeric(df["us_aqi"], errors="coerce")

    for col in POLLUTANTS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["hour"] = df["time"].dt.hour
    df["day"] = df["time"].dt.day
    df["month"] = df["time"].dt.month
    df["day_of_week"] = df["time"].dt.dayofweek

    df["pm2_5_24h"] = df["pm2_5"].rolling(window=24, min_periods=1).mean()
    df["pm10_24h"] = df["pm10"].rolling(window=24, min_periods=1).mean()

    return df, data.get("current", {})


@st.cache_data(ttl=1800)
def fetch_weather(city):
    latitude, longitude = LOCATIONS[city]
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,relative_humidity_2m,apparent_temperature,wind_speed_10m,surface_pressure",
        "timezone": "Asia/Kolkata"
    }
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    return response.json().get("current", {})


# --------------------------------------------------
# MULTI-MODEL MACHINE LEARNING ENGINE
# --------------------------------------------------

@st.cache_resource(ttl=1800)
def train_and_compare_models(df_hash, df):
    now = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None)
    historical = df[(df["time"] < now) & (df["us_aqi"].notna())].copy()

    if len(historical) < 100:
        raise ValueError("Not enough historical observations to train models.")

    X = historical[FEATURES]
    y = historical["us_aqi"]

    # Chronological 80:20 split
    split = int(len(historical) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model_specs = {
        "Random Forest": RandomForestRegressor(
            n_estimators=150, max_depth=12, min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.08, max_depth=5, random_state=42
        ),
        "Linear Regression": LinearRegression()
    }

    trained_models = {}
    metrics_rows = []

    for name, estimator in model_specs.items():
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", estimator)
        ])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        r2 = r2_score(y_test, preds)

        trained_models[name] = pipe
        metrics_rows.append({
            "Model": name,
            "RMSE": round(rmse, 2),
            "MAE": round(mae, 2),
            "R² Score": round(r2, 4)
        })

    metrics_df = pd.DataFrame(metrics_rows).sort_values("RMSE").reset_index(drop=True)
    return trained_models, metrics_df, len(historical)


# --------------------------------------------------
# SIDEBAR CONTROLS
# --------------------------------------------------

with st.sidebar:
    st.title("🌍 AirPulse")
    st.caption("MIT-ADT University • Group SY617")
    st.divider()

    city = st.selectbox("Monitoring Location", list(LOCATIONS.keys()), index=0)
    selected_model_name = st.selectbox(
        "Active Forecasting Model",
        ["Random Forest", "Gradient Boosting", "Linear Regression"],
        index=0
    )

    st.divider()
    st.markdown("**System Config**")
    st.caption("• Forecast Horizon: 24 Hours")
    st.caption("• Training Window: 90 Days (80:20 Split)")
    st.caption("• Data Source: Open-Meteo / CAMS")
    st.caption("• Scale: US EPA AQI")

    st.divider()
    if st.button("🔄 Refresh Live Data", width="stretch", type="primary"):
        fetch_air_quality.clear()
        fetch_weather.clear()
        train_and_compare_models.clear()
        st.rerun()


# --------------------------------------------------
# FETCH DATA & TRAIN MODELS
# --------------------------------------------------

try:
    with st.spinner(f"Fetching live air quality & weather for {city}..."):
        df, current_aq = fetch_air_quality(city)
        weather = fetch_weather(city)

    with st.spinner("Training & benchmarking ML models..."):
        df_hash = f"{city}_{len(df)}_{df['time'].max()}"
        trained_models, metrics_df, training_rows = train_and_compare_models(df_hash, df)

except Exception as error:
    st.error(f"Unable to initialize AirPulse: {error}")
    st.info("Check your internet connection and click Refresh Live Data.")
    st.stop()


# --------------------------------------------------
# PREPARE CURRENT & FORECAST DATA
# --------------------------------------------------

now = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None)
past_data = df[(df["time"] <= now) & (df["us_aqi"].notna())].copy()

if past_data.empty:
    st.error("Current AQI information is unavailable.")
    st.stop()

latest = past_data.iloc[-1]
prev_hour = past_data.iloc[-2] if len(past_data) > 1 else latest

current_aqi = float(latest["us_aqi"])
aqi_delta = round(current_aqi - float(prev_hour["us_aqi"]), 1)
category, emoji, badge_color = classify_aqi(current_aqi)

active_model = trained_models[selected_model_name]
future = df[df["time"] > now].sort_values("time").head(24).copy()

if not future.empty:
    future["predicted_aqi"] = np.clip(active_model.predict(future[FEATURES]), 0, 500)
    future["category"] = future["predicted_aqi"].apply(lambda x: classify_aqi(x)[0])
    peak_row = future.loc[future["predicted_aqi"].idxmax()]
    avg_24h = float(future["predicted_aqi"].mean())
    peak_24h = float(peak_row["predicted_aqi"])
else:
    peak_row, avg_24h, peak_24h = None, 0.0, 0.0


# --------------------------------------------------
# HEADER & TOP KPI ROW
# --------------------------------------------------

col_title, col_status = st.columns([3, 1.5], vertical_alignment="bottom")
with col_title:
    st.title(f"AirPulse — {city}")
    st.caption(
        f"Latest reading: {latest['time'].strftime('%d %b %Y, %I:%M %p')} IST • "
        f"Active Model: **{selected_model_name}**"
    )
with col_status:
    st.markdown(
        f"""
        <div style="text-align: right; padding-bottom: 6px;">
            <span style="
                display: inline-block;
                padding: 6px 14px;
                border-radius: 999px;
                background-color: {badge_color}22;
                border: 1px solid {badge_color};
                color: {badge_color};
                font-weight: 600;
                font-size: 0.82rem;
                letter-spacing: 0.03em;
                white-space: nowrap;
            ">
                {emoji} {category.upper()}
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

st.divider()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Current US AQI", f"{round(current_aqi)}", delta=f"{aqi_delta:+.1f} vs 1h ago", delta_color="inverse", border=True)
k2.metric("PM2.5", f"{latest['pm2_5']:.1f} µg/m³" if pd.notna(latest["pm2_5"]) else "N/A", border=True)
k3.metric("PM10", f"{latest['pm10']:.1f} µg/m³" if pd.notna(latest["pm10"]) else "N/A", border=True)
k4.metric(
    "24h Predicted Peak",
    f"{round(peak_24h)} AQI" if peak_row is not None else "N/A",
    delta=f"at {peak_row['time'].strftime('%I:%M %p')}" if peak_row is not None else None,
    delta_color="off",
    border=True
)


# --------------------------------------------------
# MAIN NAVIGATION TABS
# --------------------------------------------------

tab_overview, tab_forecast, tab_models, tab_alerts = st.tabs([
    "Live Overview & Weather",
    "24h Forecast & Trends",
    "Model Comparison",
    "Health Alerts & Export"
])

# --- TAB 1: LIVE OVERVIEW & WEATHER ---
with tab_overview:
    left_col, right_col = st.columns([1.75, 1.25])

    with left_col:
        with st.container(border=True):
            st.subheader("🧪 Live Pollutant Breakdown")
            p_cols = st.columns(5)
            for i, p in enumerate(POLLUTANTS):
                val = latest[p]
                p_cols[i].metric(POLLUTANT_NAMES[p], f"{val:.1f}" if pd.notna(val) else "N/A", help="µg/m³")

            pollutant_df = pd.DataFrame({
                "Pollutant": [POLLUTANT_NAMES[p] for p in POLLUTANTS],
                "Concentration (µg/m³)": [float(latest[p]) if pd.notna(latest[p]) else 0.0 for p in POLLUTANTS]
            })
            bar_fig = px.bar(
                pollutant_df,
                x="Pollutant",
                y="Concentration (µg/m³)",
                color="Concentration (µg/m³)",
                color_continuous_scale="Sunsetdark"
            )
            bar_fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=300,
                margin=dict(l=10, r=10, t=20, b=10),
                coloraxis_showscale=False
            )
            st.plotly_chart(bar_fig, width="stretch")

    with right_col:
        with st.container(border=True):
            st.subheader("🌤️ Live Weather Context")
            if weather:
                w1, w2 = st.columns(2)
                w1.metric("Temperature", f"{weather.get('temperature_2m', 'N/A')} °C", border=True)
                w2.metric("Feels Like", f"{weather.get('apparent_temperature', 'N/A')} °C", border=True)

                w3, w4 = st.columns(2)
                w3.metric("Humidity", f"{weather.get('relative_humidity_2m', 'N/A')}%", border=True)
                w4.metric("Wind (km/h)", f"{weather.get('wind_speed_10m', 'N/A')}", border=True)

                st.metric("Surface Pressure", f"{weather.get('surface_pressure', 'N/A')} hPa", border=True)
                st.caption("Low wind speeds and high humidity typically trap particulate matter near ground level.")
            else:
                st.info("Weather telemetry temporarily unavailable.")


# --- TAB 2: 24H FORECAST & HISTORICAL TRENDS ---
with tab_forecast:
    with st.container(border=True):
        st.subheader(f"🤖 24-Hour AQI Forecast ({selected_model_name})")
        if future.empty:
            st.warning("Future forecast data is currently unavailable.")
        else:
            f_col1, f_col2, f_col3 = st.columns(3)
            f_col1.metric("24h Average AQI", round(avg_24h), border=True)
            f_col2.metric("24h Peak AQI", round(peak_24h), border=True)
            f_col3.metric("Peak Time", peak_row["time"].strftime("%I:%M %p"), border=True)

            forecast_fig = px.line(
                future,
                x="time",
                y="predicted_aqi",
                markers=True,
                labels={"time": "Time (IST)", "predicted_aqi": "Predicted US AQI"}
            )
            forecast_fig.update_traces(line=dict(color="#E07A5F", width=3))
            forecast_fig.add_hline(y=100, line_dash="dash", line_color="#EAB308", annotation_text="Moderate (100)")
            forecast_fig.add_hline(y=150, line_dash="dash", line_color="#EF4444", annotation_text="Unhealthy (150)")
            forecast_fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=380,
                hovermode="x unified",
                margin=dict(l=10, r=10, t=20, b=10)
            )
            st.plotly_chart(forecast_fig, width="stretch")

            with st.expander("View Hourly Prediction Table"):
                st.dataframe(
                    future[["time", "predicted_aqi", "category"]].rename(columns={
                        "time": "Timestamp",
                        "predicted_aqi": "Predicted AQI",
                        "category": "Risk Category"
                    }),
                    width="stretch",
                    hide_index=True
                )

    with st.container(border=True):
        st.subheader("📈 Historical AQI Trend (Last 7 Days)")
        hist_7d = past_data.tail(168)
        hist_fig = px.area(
            hist_7d,
            x="time",
            y="us_aqi",
            labels={"time": "Date & Time", "us_aqi": "US AQI"}
        )
        hist_fig.update_traces(line=dict(color="#38BDF8", width=2), fillcolor="rgba(56, 189, 248, 0.15)")
        hist_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=340,
            margin=dict(l=10, r=10, t=20, b=10)
        )
        st.plotly_chart(hist_fig, width="stretch")


# --- TAB 3: MODEL COMPARISON ---
with tab_models:
    with st.container(border=True):
        st.subheader("🧠 Multi-Model Benchmark (RMSE, MAE, R²)")
        best_model = metrics_df.iloc[0]["Model"]
        st.success(f"**Top Performer (Lowest RMSE):** {best_model} — evaluated on {training_rows:,} hourly records (80:20 chronological split).")

        m_left, m_right = st.columns([1.2, 1.8])
        with m_left:
            st.dataframe(metrics_df, width="stretch", hide_index=True)

        with m_right:
            melted = metrics_df.melt(id_vars="Model", value_vars=["RMSE", "MAE"], var_name="Metric", value_name="Error")
            comp_fig = px.bar(
                melted,
                x="Model",
                y="Error",
                color="Metric",
                barmode="group",
                color_discrete_sequence=["#E07A5F", "#38BDF8"]
            )
            comp_fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=300,
                margin=dict(l=10, r=10, t=20, b=10)
            )
            st.plotly_chart(comp_fig, width="stretch")


# --- TAB 4: HEALTH ALERTS & DOWNLOADABLE REPORTS ---
with tab_alerts:
    adv = get_health_advisory(current_aqi)

    with st.container(border=True):
        st.subheader("🚨 Active Health Advisory")
        getattr(st, adv["level"])(f"**{adv['title']}**")

        a1, a2, a3 = st.columns(3)
        with a1:
            st.markdown("**General Public**")
            st.write(adv["general"])
        with a2:
            st.markdown("**Sensitive Groups**")
            st.write(adv["sensitive"])
        with a3:
            st.markdown("**Recommended Action**")
            st.write(adv["action"])

        if peak_24h > 150:
            st.warning(f"⚠️ **24h Forecast Alert:** AQI is projected to reach **{round(peak_24h)}** around **{peak_row['time'].strftime('%I:%M %p')}**.")
        else:
            st.info("No AQI spikes above 150 are projected in the next 24 hours.")

    with st.container(border=True):
        st.subheader("📥 Downloadable Reports")
        dl1, dl2 = st.columns(2)

        if not future.empty:
            csv_data = future[["time", "pm2_5", "pm10", "nitrogen_dioxide", "carbon_monoxide", "ozone", "predicted_aqi", "category"]].to_csv(index=False).encode("utf-8")
            dl1.download_button(
                label="Download 24h Forecast (CSV)",
                data=csv_data,
                file_name=f"AirPulse_{city}_24h_Forecast.csv",
                mime="text/csv",
                width="stretch"
            )

        summary_txt = f"""AIRPULSE — AIR QUALITY REPORT
Generated: {datetime.now().strftime('%d %b %Y, %I:%M %p')} IST
Location: {city}
Active ML Model: {selected_model_name}

CURRENT READINGS
- US AQI: {round(current_aqi)} ({category})
- PM2.5: {latest['pm2_5']:.1f} µg/m³
- PM10: {latest['pm10']:.1f} µg/m³
- Temperature: {weather.get('temperature_2m', 'N/A')} °C
- Humidity: {weather.get('relative_humidity_2m', 'N/A')}%

24-HOUR FORECAST
- Average Predicted AQI: {round(avg_24h)}
- Peak Predicted AQI: {round(peak_24h)}

MODEL BENCHMARK
{metrics_df.to_string(index=False)}

HEALTH ADVISORY
{adv['title']}
Action: {adv['action']}
"""
        dl2.download_button(
            label="Download Executive Summary (TXT)",
            data=summary_txt.encode("utf-8"),
            file_name=f"AirPulse_{city}_Summary.txt",
            mime="text/plain",
            width="stretch"
        )

st.divider()
st.caption(
    "DISCLAIMER: AirPulse is an academic prototype (MIT-ADT University, Group SY617). "
    "Data is sourced from Open-Meteo/CAMS modelled feeds (US EPA AQI scale) and is not a substitute for official CPCB stations."
)