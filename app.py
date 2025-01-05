import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Streamlit interface title and user input
st.title("Stock Price Prediction App")
st.markdown("""
This app predicts whether a stock price will increase or decrease tomorrow based on historical data.
The model uses technical indicators like **SMA (Simple Moving Average)** and **RSI (Relative Strength Index)** for prediction.
""")

# User inputs: Stock ticker and date range
ticker = st.text_input('Enter Stock Ticker', 'AAPL')  # Default is Apple
start_date = st.date_input('Start Date', pd.to_datetime('2020-01-01'))
end_date = st.date_input('End Date', pd.to_datetime('2024-01-01'))

# Fetch stock data from Yahoo Finance
st.write(f"Fetching stock data for **{ticker}** from {start_date} to {end_date}...")
stock_data = yf.download(ticker, start=start_date, end=end_date)

# Check if data is fetched successfully
if stock_data.empty:
    st.error("No data found for the given stock ticker and date range. Please try again.")
else:
    # Show the first few rows of the fetched data
    st.write("### Stock Data Overview:")
    st.write(stock_data.head())

    # Plot the closing price of the stock
    st.write(f"### {ticker} Stock Price (Closing) Over Time")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(stock_data['Close'], label="Closing Price", color='blue')
    ax.set_title(f'{ticker} Stock Price (Close)', fontsize=14)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price (USD)', fontsize=12)
    ax.grid(True)
    st.pyplot(fig)

    # Check and handle missing values
    st.write("### Checking for Missing Values:")
    missing_values = stock_data.isnull().sum()
    st.write(missing_values)
    
    if missing_values.any():
        st.write("Filling missing values with forward fill...")
        stock_data.fillna(method='ffill', inplace=True)

    # Feature Scaling using Min-Max Scaler
    st.write("### Scaling Data...")
    scaler = MinMaxScaler()
    stock_data[['Open', 'High', 'Low', 'Close', 'Volume']] = scaler.fit_transform(
        stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
    )

    # Calculate 30-day and 100-day Simple Moving Averages (SMA)
    st.write("### Adding 30-day and 100-day Moving Averages (SMA)...")
    stock_data['SMA_30'] = stock_data['Close'].rolling(window=30).mean()
    stock_data['SMA_100'] = stock_data['Close'].rolling(window=100).mean()

    # Show the most recent data with moving averages
    st.write("### Recent Data with Moving Averages:")
    st.write(stock_data[['Close', 'SMA_30', 'SMA_100']].tail())

    # Calculate RSI (Relative Strength Index)
    st.write("### Calculating RSI (Relative Strength Index)...")
    delta = stock_data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    stock_data['RSI'] = rsi

    # Show the latest data with RSI
    st.write("### Latest Data with RSI:")
    st.write(stock_data[['Close', 'RSI']].tail())

    # Generate target column for prediction (1 if price goes up the next day, 0 if down)
    st.write("### Generating Target Column...")
    stock_data['Target'] = (stock_data['Close'].shift(-1) > stock_data['Close']).astype(int)
    stock_data.dropna(inplace=True)

    # Display the data with the target column
    st.write("### Data with Target Column:")
    st.write(stock_data[['Close', 'Target']].tail())

    # Creating lag features (previous day's closing price)
    stock_data['Lag_1'] = stock_data['Close'].shift(1)
    stock_data.dropna(inplace=True)

    # Define features (X) and target (y)
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_30', 'SMA_100', 'RSI', 'Lag_1']
    X = stock_data[features]
    y = stock_data['Target']

    # Split the data into training and testing sets (80% train, 20% test)
    st.write("### Splitting Data into Training and Testing Sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    st.write(f"Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")

    # Train the Random Forest Classifier
    st.write("### Training the Random Forest Classifier Model...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Save the trained model (optional)
    joblib.dump(rf, 'stock_price_model.pkl')

    # Make predictions and evaluate the model
    st.write("### Evaluating the Model...")
    y_pred_rf = rf.predict(X_test)

    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
    class_report_rf = classification_report(y_test, y_pred_rf)

    # Display evaluation metrics
    st.write(f"**Random Forest Accuracy**: {accuracy_rf:.4f}")
    st.write("Confusion Matrix:")
    st.write(conf_matrix_rf)
    st.write("Classification Report:")
    st.write(class_report_rf)

    # Predict the next day's price movement
    st.write("### Predicting Next Day's Price Movement...")
    latest_data = stock_data.iloc[-1][features].values.reshape(1, -1)
    latest_prediction = rf.predict(latest_data)
    prediction = "up" if latest_prediction == 1 else "down"
    st.write(f"The model predicts that the stock price will go **{prediction}** tomorrow!")