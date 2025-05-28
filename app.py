# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
LOGO_PATH = "logo.png" # Path to your logo image
IMAGES_DIR = "images" # Directory where plot images are stored
PASSWORD = "welcome" # Fixed password for login

# --- Helper function to load images ---
def load_image(image_path):
    if os.path.exists(image_path):
        return st.image(image_path, use_column_width=True)
    else:
        st.warning(f"Image not found: {image_path}. Please ensure it's in the correct directory.")
        st.info(f"Expected path: {os.path.abspath(image_path)}")
        st.write("Placeholder image:")
        st.image("https://placehold.co/600x200/cccccc/ffffff?text=Logo+Placeholder", use_column_width=True) # Placeholder

# --- Data for Model Comparison Table (reconstructed from notebook) ---
# This DataFrame is small enough to be hardcoded or loaded from a small CSV if preferred.
model_metrics_data = {
    'Model': ['Baseline', 'Arima_basic', 'Arima_dynamic', 'Sarima', 'Sarimax', 'LSTM_Univar', 'LSTM_Multivar'],
    'MSE': [0.071, 0.259, 0.069, 0.107, 0.101, 0.068, 0.022],
    'RMSE': [0.266, 0.509, 0.263, 0.327, 0.317, 0.261, 0.15],
    'MAE': [0.177, 0.463, 0.176, 0.266, 0.243, 0.173, 0.11],
    'MAPE': [0.236, 0.722, 0.229, 0.397, 0.363, 0.307, 0.173],
    'R^2': [0.077, -2.379, 0.094, -0.399, -0.318, 0.106, 0.7]
}
model_comparison_df = pd.DataFrame(data=model_metrics_data).set_index('Model')

# --- Data for Temperature Difference Correlation Table (reconstructed from notebook) ---
# This DataFrame is small enough to be hardcoded.
temp_diff_corr_data = {
    'weather': ['temperature', 'humidity', 'visibility', 'apparentTemperature', 'pressure', 'windSpeed',
                'cloudCover', 'windBearing', 'precipIntensity', 'dewPoint', 'precipProbability'],
    'Tdiff_corr': [-0.992, -0.281, 0.009, 1.000, 0.005, 0.490, -0.052, 0.006, 0.006, -0.963, 0.006]
}
temp_diff_corr_df = pd.DataFrame(temp_diff_corr_data).set_index('weather')

# --- Data for Energy-Weather Correlation Matrix (reconstructed from notebook) ---
# This DataFrame is small enough to be hardcoded.
energy_weather_corr_data = {
    'consumtions': ['House overall', 'Dishwasher', 'Home office', 'Fridge', 'Wine cellar', 'Garage door', 'Barn',
                    'Well', 'Microwave', 'Living room', 'Furnace', 'Kitchen', 'Solar'],
    'temperature_corr': [0.141, -0.007, 0.015, -0.012, 0.001, -0.000, 0.001, 0.000, 0.000, 0.002, 0.001, 0.002, 0.001],
    'humidity_corr': [-0.076, 0.000, -0.001, -0.000, -0.000, 0.000, -0.000, -0.000, -0.000, -0.001, -0.000, -0.000, -0.000],
    'visibility_corr': [0.009, -0.000, 0.000, -0.000, -0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    'apparentTemperature_corr': [0.141, -0.007, 0.015, -0.012, 0.001, -0.000, 0.001, 0.000, 0.000, 0.002, 0.001, 0.002, 0.001],
    'pressure_corr': [0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    'windSpeed_corr': [0.038, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    'cloudCover_corr': [-0.001, 0.000, -0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    'windBearing_corr': [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    'precipIntensity_corr': [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    'dewPoint_corr': [0.136, -0.007, 0.015, -0.012, 0.001, -0.000, 0.001, 0.000, 0.000, 0.002, 0.001, 0.002, 0.001],
    'precipProbability_corr': [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
}
energy_weather_corr_df = pd.DataFrame(energy_weather_corr_data).set_index('consumtions')

# --- Streamlit App ---

def show_login_page():
    st.markdown("""
        <style>
            .stApp {
                background-color: #f0f2f6; /* Light gray background */
            }
            .stTextInput label {
                font-size: 1.2em;
                color: #333;
            }
            .stButton button {
                background-color: #4CAF50; /* Green */
                color: white;
                padding: 10px 20px;
                border-radius: 8px;
                border: none;
                font-size: 1.1em;
                cursor: pointer;
                box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
                transition: all 0.3s ease;
            }
            .stButton button:hover {
                background-color: #45a049;
                transform: translateY(-2px);
                box-shadow: 4px 4px 10px rgba(0,0,0,0.3);
            }
            .login-container {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                min-height: 80vh;
                text-align: center;
            }
            .login-box {
                background-color: white;
                padding: 40px;
                border-radius: 15px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.1);
                max-width: 400px;
                width: 90%;
            }
            h1 {
                color: #2c3e50;
                font-size: 2.5em;
                margin-bottom: 20px;
            }
        </style>
    """, unsafe_allow_html=True)

    st.container()
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("<div class='login-container'>", unsafe_allow_html=True)
        st.markdown("<div class='login-box'>", unsafe_allow_html=True)
        st.title("Welcome to Wattson")
        st.subheader("Your Smart Energy Companion")

        username = st.text_input("Enter your name:")
        password = st.text_input("Enter password:", type="password")

        if st.button("Login"):
            if password == PASSWORD:
                st.session_state['logged_in'] = True
                st.session_state['username'] = username if username else "Guest"
                st.success("Logged in successfully!")
                st.experimental_rerun() # Rerun to switch to main app
            else:
                st.error("Incorrect password. Please try again.")
        st.markdown("</div></div>", unsafe_allow_html=True)


def show_welcome_tab():
    st.markdown(f"""
        <style>
            .welcome-container {{
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                min-height: 70vh;
                text-align: center;
                padding: 20px;
            }}
            .welcome-card {{
                background: linear-gradient(135deg, #e0f7fa, #b2ebf2);
                padding: 40px;
                border-radius: 20px;
                box-shadow: 0 8px 30px rgba(0,0,0,0.15);
                max-width: 700px;
                width: 95%;
                animation: fadeIn 1s ease-out;
            }}
            @keyframes fadeIn {{
                from {{ opacity: 0; transform: translateY(20px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
            .welcome-title {{
                color: #00796b;
                font-size: 3em;
                margin-bottom: 15px;
                font-weight: bold;
            }}
            .welcome-tagline {{
                color: #004d40;
                font-size: 1.5em;
                margin-bottom: 30px;
                font-style: italic;
            }}
            .stImage {{
                border-radius: 10px;
                margin-bottom: 30px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            }}
            .stButton button {{
                background-color: #00796b;
                color: white;
                padding: 15px 30px;
                border-radius: 10px;
                border: none;
                font-size: 1.3em;
                cursor: pointer;
                box-shadow: 3px 3px 8px rgba(0,0,0,0.3);
                transition: all 0.3s ease;
            }}
            .stButton button:hover {{
                background-color: #004d40;
                transform: scale(1.05);
            }}
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='welcome-container'>", unsafe_allow_html=True)
    st.markdown("<div class='welcome-card'>", unsafe_allow_html=True)
    st.markdown(f"<h1 class='welcome-title'>Welcome to Wattson, {st.session_state.get('username', 'User')}!</h1>", unsafe_allow_html=True)

    load_image(LOGO_PATH)

    st.markdown("<p class='welcome-tagline'>It’s All About the Smart Clues</p>", unsafe_allow_html=True)

    if st.button("See Your Energy Usage"):
        st.session_state['current_tab'] = 'Visualizations'
        st.experimental_rerun()
    st.markdown("</div></div>", unsafe_allow_html=True)


def show_visualizations_tab():
    st.markdown("""
        <style>
            .viz-container {
                padding: 20px;
                background-color: #f9f9f9;
                border-radius: 15px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.05);
            }
            .viz-header {
                color: #2c3e50;
                font-size: 2.2em;
                margin-bottom: 25px;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
            }
            .stButton button {
                background-color: #3498db; /* Blue */
                color: white;
                padding: 12px 25px;
                border-radius: 8px;
                border: none;
                font-size: 1.05em;
                cursor: pointer;
                margin: 5px;
                box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
                transition: all 0.3s ease;
            }
            .stButton button:hover {
                background-color: #2980b9;
                transform: translateY(-2px);
                box-shadow: 4px 4px 10px rgba(0,0,0,0.3);
            }
            .stExpander {
                border: 1px solid #ddd;
                border-radius: 10px;
                margin-bottom: 15px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            }
            .stExpander > div > div > p {
                font-weight: bold;
                color: #2c3e50;
            }
            .dataframe-style {
                border-collapse: collapse;
                width: 100%;
                margin-top: 15px;
                font-size: 0.9em;
            }
            .dataframe-style th, .dataframe-style td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }
            .dataframe-style th {
                background-color: #f2f2f2;
                color: #333;
            }
            .dataframe-style tr:nth-child(even) {
                background-color: #f8f8f8;
            }
            .highlight-red { background-color: #ffcccc; } /* Light red */
            .highlight-orange { background-color: #ffe0b2; } /* Light orange */
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='viz-container'>", unsafe_allow_html=True)
    st.markdown("<h2 class='viz-header'>Energy Usage Visualizations</h2>", unsafe_allow_html=True)

    # --- Energy Correlations ---
    with st.expander("Appliance Energy Correlations"):
        st.write("This heatmap illustrates the correlation between different energy consumption categories.")
        load_image(os.path.join(IMAGES_DIR, "energy_corr_heatmap.png"))

    # --- Weather Correlations ---
    with st.expander("Weather Feature Correlations"):
        st.write("This heatmap shows the correlation among various weather features.")
        load_image(os.path.join(IMAGES_DIR, "weather_corr_heatmap.png"))

    # --- Temperature Difference vs. Weather Factors ---
    with st.expander("Temperature Difference vs. Weather Factors"):
        st.write("Correlation between the difference in apparent and actual temperature, and other weather variables.")
        st.dataframe(temp_diff_corr_df.style.format("{:.3f}"))

    # --- Daily Average Appliance Usage ---
    with st.expander("Daily Average Appliance Usage (Part 1)"):
        st.write("Daily mean consumption for various home appliances.")
        load_image(os.path.join(IMAGES_DIR, "daily_avg_appliances_p1.png"))

    with st.expander("Daily Average Appliance Usage (Part 2)"):
        st.write("Daily mean consumption for additional home appliances.")
        load_image(os.path.join(IMAGES_DIR, "daily_avg_appliances_p2.png"))

    # --- Daily Average Weather Trends ---
    with st.expander("Daily Average Weather Trends"):
        st.write("Daily mean trends for key weather parameters.")
        load_image(os.path.join(IMAGES_DIR, "daily_avg_weather.png"))

    # --- Monthly Energy Consumption Patterns ---
    with st.expander("Monthly Energy Consumption Patterns"):
        st.write("Average energy consumption for different appliances across months.")
        load_image(os.path.join(IMAGES_DIR, "monthly_avg_consumption.png"))

    # --- Weekly Energy Consumption Patterns ---
    with st.expander("Weekly Energy Consumption Patterns"):
        st.write("Average energy consumption for different appliances across days of the week.")
        load_image(os.path.join(IMAGES_DIR, "weekly_avg_consumption.png"))

    # --- Hourly Energy Consumption Patterns ---
    with st.expander("Hourly Energy Consumption Patterns"):
        st.write("Average energy consumption for different appliances throughout the day (hourly).")
        load_image(os.path.join(IMAGES_DIR, "hourly_avg_consumption.png"))

    # --- Energy-Weather Correlation Matrix ---
    with st.expander("Energy-Weather Correlation Matrix"):
        st.write("Correlations between various energy consumption categories and weather features.")
        # Function to apply color based on correlation value
        def highlight_corr(s):
            is_red = s > 0.1
            is_orange = s < -0.1
            return ['background-color: #ffcccc' if v else 'background-color: #ffe0b2' if w else ''
                    for v, w in zip(is_red, is_orange)]

        st.dataframe(energy_weather_corr_df.style.apply(highlight_corr, axis=1).format("{:.3f}"))

    # --- Overall House Usage vs. Appliance Sum ---
    with st.expander("Overall House Usage vs. Appliance Sum"):
        st.write("Comparison of 'House overall' consumption against the sum of individual appliance usages, and their difference.")
        load_image(os.path.join(IMAGES_DIR, "house_usage_sum_diff.png"))

    st.markdown("<h3 class='viz-header'>Anomaly Detection Insights</h3>", unsafe_allow_html=True)

    # --- Anomaly Detection: Moving Average ---
    with st.expander("Anomaly Detection: Moving Average"):
        st.write("Anomalies detected using a moving average approach with confidence intervals.")
        load_image(os.path.join(IMAGES_DIR, "anomaly_ma.png"))

    # --- Anomaly Detection: ARIMA (Residuals) ---
    with st.expander("Anomaly Detection: ARIMA (Residuals)"):
        st.write("Anomalies detected based on the residuals of the ARIMA model.")
        load_image(os.path.join(IMAGES_DIR, "anomaly_arima_resid.png"))

    # --- Anomaly Detection: ARIMA (Confidence Intervals) ---
    with st.expander("Anomaly Detection: ARIMA (Confidence Intervals)"):
        st.write("Anomalies detected using confidence intervals from the ARIMA model's predictions.")
        load_image(os.path.join(IMAGES_DIR, "anomaly_arima_conf_int.png"))

    # --- Anomaly Detection: LSTM Multivariate ---
    with st.expander("Anomaly Detection: LSTM Multivariate"):
        st.write("Anomalies detected by the Multivariate LSTM model, highlighting unusual energy usage patterns.")
        load_image(os.path.join(IMAGES_DIR, "anomaly_lstm_multivar.png"))

    st.markdown("</div>", unsafe_allow_html=True)


def show_predictions_tab():
    st.markdown("""
        <style>
            .pred-container {
                padding: 20px;
                background-color: #f9f9f9;
                border-radius: 15px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.05);
            }
            .pred-header {
                color: #2c3e50;
                font-size: 2.2em;
                margin-bottom: 25px;
                border-bottom: 2px solid #27ae60;
                padding-bottom: 10px;
            }
            .stButton button {
                background-color: #2ecc71; /* Green */
                color: white;
                padding: 12px 25px;
                border-radius: 8px;
                border: none;
                font-size: 1.05em;
                cursor: pointer;
                margin: 5px;
                box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
                transition: all 0.3s ease;
            }
            .stButton button:hover {
                background-color: #27ae60;
                transform: translateY(-2px);
                box-shadow: 4px 4px 10px rgba(0,0,0,0.3);
            }
            .stExpander {
                border: 1px solid #ddd;
                border-radius: 10px;
                margin-bottom: 15px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            }
            .stExpander > div > div > p {
                font-weight: bold;
                color: #2c3e50;
            }
            .metrics-table {
                font-size: 0.9em;
                margin-top: 15px;
                width: 100%;
            }
            .metrics-table th, .metrics-table td {
                padding: 8px;
                border: 1px solid #ddd;
                text-align: left;
            }
            .metrics-table th {
                background-color: #f2f2f2;
                color: #333;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='pred-container'>", unsafe_allow_html=True)
    st.markdown("<h2 class='pred-header'>Energy Usage Predictions</h2>", unsafe_allow_html=True)

    st.markdown("### Baseline Models")
    with st.expander("Moving Average Forecast"):
        st.write("Forecast using a simple moving average, serving as a baseline.")
        load_image(os.path.join(IMAGES_DIR, "forecast_ma.png"))
        st.subheader("Performance Metrics:")
        st.dataframe(model_comparison_df.loc[['Baseline']].style.format("{:.3f}"))

    with st.expander("Persistence Algorithm Forecast"):
        st.write("Forecast using the persistence algorithm, where the next value is predicted to be the same as the current value.")
        load_image(os.path.join(IMAGES_DIR, "forecast_persistence.png"))
        st.subheader("Performance Metrics:")
        # Note: Persistence metrics are calculated in your notebook but not explicitly in the 'df' table under a distinct row.
        # Assuming 'Baseline' row covers the MA, we can add a custom display or prompt user to add to df.
        st.write("Metrics for Persistence Algorithm are usually compared against other baselines.")
        st.markdown("""
            <table class="metrics-table">
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>RMSE</td><td>0.244</td></tr>
                <tr><td>MAE</td><td>0.170</td></tr>
                <tr><td>MAPE</td><td>0.224</td></tr>
                <tr><td>MASE</td><td>0.771</td></tr>
                <tr><td>R^2</td><td>0.133</td></tr>
            </table>
        """, unsafe_allow_html=True)


    st.markdown("### Advanced Time Series Models")
    with st.expander("ARIMA Single Step Forecast"):
        st.write("Forecast using the Autoregressive Integrated Moving Average (ARIMA) model.")
        load_image(os.path.join(IMAGES_DIR, "forecast_arima_single.png"))
        st.subheader("Performance Metrics:")
        st.dataframe(model_comparison_df.loc[['Arima_basic']].style.format("{:.3f}"))

    with st.expander("ARIMA Rolling Forecast (Cross-Validation)"):
        st.write("ARIMA model's performance evaluated using a rolling forecast cross-validation technique.")
        load_image(os.path.join(IMAGES_DIR, "forecast_arima_rolling.png"))
        st.subheader("Performance Metrics:")
        st.dataframe(model_comparison_df.loc[['Arima_dynamic']].style.format("{:.3f}"))

    with st.expander("SARIMAX Forecast with Exogenous Variables"):
        st.write("Seasonal Autoregressive Integrated Moving Average with Exogenous Regressors (SARIMAX) model forecast, incorporating external factors.")
        load_image(os.path.join(IMAGES_DIR, "forecast_sarimax.png"))
        st.subheader("Performance Metrics:")
        st.dataframe(model_comparison_df.loc[['Sarimax']].style.format("{:.3f}"))

    with st.expander("LSTM Univariate Forecast"):
        st.write("Long Short-Term Memory (LSTM) neural network model for univariate time series forecasting.")
        load_image(os.path.join(IMAGES_DIR, "forecast_lstm_univariate.png"))
        st.subheader("Performance Metrics:")
        st.dataframe(model_comparison_df.loc[['LSTM_Univar']].style.format("{:.3f}"))

    with st.expander("LSTM Multivariate Forecast"):
        st.write("Long Short-Term Memory (LSTM) neural network model for multivariate time series forecasting, considering multiple input features.")
        load_image(os.path.join(IMAGES_DIR, "forecast_lstm_multivariate.png"))
        st.subheader("Performance Metrics:")
        st.dataframe(model_comparison_df.loc[['LSTM_Multivar']].style.format("{:.3f}"))

    st.markdown("### Overall Model Performance Comparison")
    with st.expander("View All Model Metrics"):
        st.write("A comprehensive comparison of all forecasting models based on various error metrics.")
        st.dataframe(model_comparison_df.style.format("{:.3f}"))

    st.markdown("</div>", unsafe_allow_html=True)


# --- Main App Logic ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'current_tab' not in st.session_state:
    st.session_state['current_tab'] = 'Welcome' # Default tab after login

if not st.session_state['logged_in']:
    show_login_page()
else:
    st.sidebar.title("Navigation")
    # Use st.radio for tab-like navigation in the sidebar
    st.session_state['current_tab'] = st.sidebar.radio(
        "Go to",
        ('Welcome', 'Visualizations', 'Predictions'),
        index=('Welcome', 'Visualizations', 'Predictions').index(st.session_state['current_tab'])
    )

    st.sidebar.markdown("---")
    if st.sidebar.button("Logout"):
        st.session_state['logged_in'] = False
        st.session_state['username'] = None
        st.session_state['current_tab'] = 'Welcome'
        st.experimental_rerun()

    if st.session_state['current_tab'] == 'Welcome':
        show_welcome_tab()
    elif st.session_state['current_tab'] == 'Visualizations':
        show_visualizations_tab()
    elif st.session_state['current_tab'] == 'Predictions':
        show_predictions_tab()
