import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import os
import datetime
import plotly.graph_objs as go

# Streamlit page config
st.set_page_config(page_title="Stock Market Predictor", page_icon="📈", layout="wide")

# Model path
model_path = r"C:\Users\adity\OneDrive\Documents\Desktop\Jupyter Projects\adityaa\models\StockPricePrediction.h5"
if not os.path.exists(model_path):
    st.error(f"⚠️ Model file not found at: {model_path}")
    st.stop()

model = load_model(model_path)

# Sidebar: About only
st.sidebar.title("ℹ️ About This Project")
with st.sidebar.expander("Project Info"):
    st.markdown(
        """
        ### 📈 Stock Market Predictor  
        **Developed by: Aditya Nandal**  
        This application uses a trained LSTM Machine Learning model to predict stock prices based on historical data.  

        **🔧 Tech Stack:**  
        - Python  
        - Machine Learning (LSTM)  
        - TensorFlow / Keras  
        - yFinance API  
        - Streamlit  
        - Plotly  
        - Scikit-learn  

        _Responsive design: Works on mobile, tablets, and desktops._  

        🔗 [LinkedIn](https://www.linkedin.com/in/adityanandalwork)  
        🔗 [GitHub](https://github.com/aditya1461)  
        """,
        unsafe_allow_html=True
    )

# App title
st.title("📈 Stock Market Predictor")

# Date selectors
start_date = st.date_input('🗕️ Start Date', datetime.date(2012, 1, 1))
end_date = st.date_input('🗕️ End Date', datetime.date.today())

# Stock input
stock = st.text_input('Enter Stock Symbol (e.g., AAPL, TSLA, GOOG)', 'GOOG').upper()

# Download data
try:
    data = yf.download(stock, start=start_date, end=end_date)
    if data.empty:
        st.error(f"⚠️ No data found for {stock}.")
        st.stop()
except Exception as e:
    st.error(f"⚠️ Error fetching stock data: {e}")
    st.stop()

st.subheader(f'📊 Stock Data for {stock}')
st.dataframe(data, use_container_width=True)

# Preprocess
lookback_period = 30
scaler = MinMaxScaler(feature_range=(0,1))
data_full = data['Close']
data_full_scaled = scaler.fit_transform(data_full.values.reshape(-1,1))

x_full = [data_full_scaled[i-lookback_period:i] for i in range(lookback_period, len(data_full_scaled))]
x_full = np.array(x_full)

# Predict
predicted_prices = model.predict(x_full)
predicted_prices = scaler.inverse_transform(predicted_prices).flatten()
prediction_dates = data.index[lookback_period:]

# Stats
total_days = (data.index[-1] - data.index[0]).days + 1
highest_price = float(data['Close'].max())
lowest_price = float(data['Close'].min())

# Summary section
st.subheader("📌 Summary Stats")
st.markdown(f"""
- Total Days: **{total_days}**
- Highest Closing Price: **${highest_price:.2f}**
- Lowest Closing Price: **${lowest_price:.2f}**
""")

# Prediction period
st.subheader("🗕️ Prediction Period")
st.markdown(f"""
- From **{prediction_dates[0].strftime('%Y-%m-%d')}** to **{prediction_dates[-1].strftime('%Y-%m-%d')}**
- Total Prediction Days: **{len(prediction_dates)}**
""")

# Plot actual predictions
pred_df = pd.DataFrame({'Date': prediction_dates, 'Predicted Price': predicted_prices})
fig = go.Figure()
fig.add_trace(go.Scatter(x=pred_df['Date'], y=pred_df['Predicted Price'], mode='lines', name='Predicted Price', line=dict(color='royalblue')))
fig.update_layout(
    title=f'📉 Predicted Stock Prices for {stock}',
    xaxis_title='Date',
    yaxis_title='Price (USD)',
    hovermode='x unified',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)
st.plotly_chart(fig, use_container_width=True)

# CSV Download
st.subheader("📅 Download Stock Data")
csv = data.to_csv(index=True)
st.download_button("Download CSV", csv, file_name=f"{stock}_stock_data.csv", mime="text/csv")

# Future Prediction
st.subheader("🔮 Future Stock Prediction (Next 30 Business Days)")
future_days = 30
last_lookback_days = data_full.tail(lookback_period).values.reshape(-1,1)
future_data = np.copy(last_lookback_days)
future_predictions = []

for _ in range(future_days):
    future_scaled = scaler.transform(future_data)
    future_input = future_scaled.reshape(1, lookback_period, 1)
    predicted_price = model.predict(future_input)
    future_predictions.append(predicted_price[0][0])
    future_data = np.append(future_data, predicted_price[0][0]).reshape(-1, 1)[1:]

future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()
future_dates = pd.date_range(data.index[-1], periods=future_days + 1, freq='B')[1:]
future_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_predictions})

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=future_df['Date'], y=future_df['Predicted Price'], mode='lines', name='Future Price', line=dict(color='green')))
fig2.update_layout(
    title=f'🔮 Future Stock Prices for {stock}',
    xaxis_title='Date',
    yaxis_title='Price (USD)',
    hovermode='x unified',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)
st.plotly_chart(fig2, use_container_width=True)

# Refresh
if st.button("🔄 Refresh Data"):
    st.experimental_rerun()