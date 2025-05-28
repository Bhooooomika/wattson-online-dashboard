import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib

st.set_page_config(layout="wide", page_title="Smart Home Energy Dashboard")

# Load data
@st.cache_data
def load_data():
    return joblib.load("df_cleaned.pkl")

data = load_data()

st.title("🏠 Smart Home Energy Usage Dashboard")
st.markdown("Cleaned and enriched dataset loaded from `df_cleaned.pkl`")

# Tabs for sections
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Overview", "📈 Visualizations", "📊 Correlations", "📆 Grouped Insights"])

with tab1:
    st.subheader("Data Preview")
    st.dataframe(data.head())

    st.write("Columns:")
    st.write(data.columns.tolist())

    st.markdown("**Shape:**")
    st.write(data.shape)

    st.markdown("**Datetime Range:**")
    st.write(f"{data.index.min()} → {data.index.max()}")

with tab2:
    st.subheader("Daily Energy and Weather Visualizations")

    energy_cols = data.columns[0:13].tolist()
    weather_cols = data.columns[13:-5].tolist()

    selected_energy = st.multiselect("Select Energy Columns", energy_cols, default=energy_cols[:6])
    selected_weather = st.multiselect("Select Weather Columns", weather_cols, default=weather_cols[:6])

    if selected_energy:
        st.markdown("### Energy Consumption")
        data[selected_energy].resample("D").mean().plot(subplots=True, layout=(-1, 3), figsize=(20, 10), grid=True)
        st.pyplot(plt.gcf())

    if selected_weather:
        st.markdown("### Weather Parameters")
        data[selected_weather].resample("D").mean().plot(subplots=True, layout=(-1, 3), figsize=(20, 10), grid=True)
        st.pyplot(plt.gcf())

with tab3:
    st.subheader("Correlations")

    st.markdown("### 🔄 Energy Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(data[data.columns[0:13]].corr(), annot=True, vmin=-1, vmax=1, center=0, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.markdown("### 🌤️ Weather Correlation Heatmap")
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    sns.heatmap(data[data.columns[13:-5]].corr(), annot=True, vmin=-1, vmax=1, center=0, cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)

    st.markdown("### 🌡️ Temperature Difference Correlation")
    data['Tdiff'] = data['apparentTemperature'] - data['temperature']
    clima = data.columns[13:-5].tolist()
    tdiff_corrs = {col: data[col].corr(data['Tdiff']) for col in clima}
    tdiff_df = pd.DataFrame.from_dict(tdiff_corrs, orient='index', columns=["Tdiff Correlation"])
    st.dataframe(tdiff_df.style.background_gradient(cmap="RdBu", axis=0))

with tab4:
    st.subheader("Grouped Energy Insights")

    st.markdown("### 📅 Average Monthly Consumption")
    data['month'] = data.index.month
    mean_month = data.groupby('month').mean()
    mean_month[data.columns[0:13]].plot(subplots=True, layout=(-1, 3), figsize=(18, 12), grid=True, marker='o')
    st.pyplot(plt.gcf())

    st.markdown("### 📆 Average Consumption per Weekday")
    data['weekday'] = data.index.day_name()
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    mean_weekday = data.groupby('weekday').mean().reindex(weekdays)
    mean_weekday[data.columns[0:13]].plot(subplots=True, layout=(-1, 3), figsize=(18, 12), grid=True, marker='o')
    st.pyplot(plt.gcf())

    st.markdown("### 🕒 Average Hourly Consumption")
    data['hour'] = data.index.hour
    mean_hour = data.groupby('hour').mean()
    mean_hour[data.columns[0:13]].plot(subplots=True, layout=(-1, 3), figsize=(18, 12), grid=True, marker='o')
    st.pyplot(plt.gcf())
