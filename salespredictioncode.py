import pandas as pd

data = {
    "Date": [
        "2021-01-01","2021-01-02","2021-01-03","2021-01-04","2021-01-05",
        "2021-01-06","2021-01-07","2021-01-08","2021-01-09","2021-01-10",
        "2021-01-11","2021-01-12","2021-01-13","2021-01-14","2021-01-15",
        "2021-01-16","2021-01-17","2021-01-18","2021-01-19","2021-01-20",
        "2021-01-21","2021-01-22","2021-01-23","2021-01-24","2021-01-25",
        "2021-01-26","2021-01-27","2021-01-28","2021-01-29","2021-01-30",
        "2021-02-01","2021-03-01","2021-04-01","2021-05-01","2021-06-01",
        "2021-07-01","2021-08-01","2021-09-01","2021-10-01","2021-11-01",
        "2021-12-01","2022-01-01","2022-02-01","2022-03-01","2022-04-01",
        "2022-05-01","2022-06-01","2022-07-01","2022-08-01","2022-09-01",
        "2022-10-01","2022-11-01","2022-12-01","2023-01-01","2023-02-01",
        "2023-03-01","2023-04-01","2023-05-01","2023-06-01","2023-07-01",
        "2023-08-01","2023-09-01","2023-10-01","2023-11-01","2023-12-01"
    ],
    "Sales": [
        2054,2130,1988,2201,2244,2199,2305,2401,2440,2322,
        2460,2501,2490,2588,2604,2680,2555,2702,2766,2804,
        2877,2902,2988,2866,3020,3099,3110,3188,3221,3304,
        3455,3510,3604,3720,3899,4010,4122,4208,4290,4411,
        4504,4620,4701,4822,4955,5099,5204,5310,5440,5524,
        5630,5740,5855,5999,6100,6210,6304,6455,6501,6620,
        6750,6880,6999,7104,7255
    ]
}

df = pd.DataFrame(data)

df.to_csv("sales.csv", index=False)

df.head(), "Dataset saved as sales.csv"

import pandas as pd

df = pd.read_csv("sales.csv")

# Prophet requires these column names
df['ds'] = pd.to_datetime(df['Date'])
df['y'] = df['Sales']

df = df[['ds', 'y']]

df.head()

from prophet import Prophet

model = Prophet()
model.fit(df)

future = model.make_future_dataframe(periods=30)  # forecast next 30 days
forecast = model.predict(future)

model.plot(forecast)
model.plot_components(forecast)

#writefile app.py
import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

st.title("📈 Sales Forecasting Dashboard")
st.write("Powered by Prophet — Future Interns ML Task 1")

# Upload CSV
uploaded_file = st.file_uploader("Upload your sales.csv file", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # Preprocess
    df['ds'] = pd.to_datetime(df['Date'])
    df['y'] = df['Sales']
    data = df[['ds', 'y']]

    st.subheader("📊 Raw Data")
    st.dataframe(data)

    # Model
    model = Prophet()
    model.fit(data)

    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    st.subheader("📈 Sales Forecast (Next 30 Days)")

    # Plot forecast using Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['ds'], y=data['y'], mode='lines', name='Historical Sales'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📉 Trend & Seasonality Components")
    st.write("Use the Prophet built-in plots below:")

    # Show Prophet components using matplotlib
    from prophet.plot import plot_components
    import matplotlib.pyplot as plt

    fig2 = plot_components(model, forecast)
    st.pyplot(fig2)

