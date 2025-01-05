Stock Price Prediction App
This is a Stock Price Prediction App built using Streamlit, yfinance, and Random Forest Classifier. The app allows users to enter a stock ticker and select a date range, fetch historical stock data, and make predictions on whether the stock price will go up or down the following day. The model uses technical indicators such as Moving Averages (SMA) and Relative Strength Index (RSI) to make predictions.

Features
Fetches stock data from Yahoo Finance based on user input for ticker symbol and date range.
Visualizes the closing price of the stock over time.
Computes and displays technical indicators like 30-day and 100-day Simple Moving Averages (SMA) and Relative Strength Index (RSI).
Trains a Random Forest Classifier model to predict whether the stock price will increase or decrease the next day.
Displays model evaluation metrics including accuracy, confusion matrix, and classification report.
Provides a prediction for the next day's stock price movement (up or down).
Technologies Used
Python
Streamlit: For creating the interactive web app.
yfinance: To fetch historical stock data.
scikit-learn: For model training and evaluation.
Matplotlib: For plotting stock price data.
Pandas: For data manipulation.
Joblib: For saving the trained model.
Installation
Prerequisites
Make sure you have Python 3.x installed. Then, install the required dependencies using pip:

bash
Copy code
pip install streamlit yfinance pandas matplotlib scikit-learn joblib
How to Run the App
Clone the repository or create a directory for your project.

Save the code into a Python file (e.g., app.py).

Run the Streamlit app using the following command in your terminal:

bash
Copy code
streamlit run app.py
After running the above command, Streamlit will automatically open the app in your default web browser.

Features Explained
Stock Ticker and Date Range: Users can enter a stock ticker (e.g., AAPL for Apple) and specify the start and end dates to fetch stock data from Yahoo Finance.
Stock Price Plot: The app visualizes the stock's closing price over time.
Technical Indicators: The app calculates and displays the 30-day and 100-day simple moving averages (SMA) and the 14-day Relative Strength Index (RSI).
Prediction: Based on the data, the app trains a Random Forest model to predict if the stock price will go up or down on the next day. The prediction is displayed in the app interface.
Model Evaluation: The app provides the accuracy, confusion matrix, and classification report of the trained Random Forest model.
Example Output
User Input: Stock ticker = AAPL, Date range = 2020-01-01 to 2024-01-01.
Model Output: "The model predicts that the stock price will go up tomorrow!".
Evaluation Metrics:
Random Forest Accuracy: XX.XX%
Confusion Matrix:
lua
Copy code
[[TN, FP],
 [FN, TP]]
Classification Report:
python
Copy code
precision    recall  f1-score   support
...
Troubleshooting
If the app doesn't load: Make sure you have installed all dependencies and that your internet connection is stable for fetching data from Yahoo Finance.
Missing Data: If the app displays missing data or doesn't generate the plot, verify the stock ticker and date range to ensure valid data is available.
Future Enhancements
Support for additional technical indicators.
Use more advanced models like LSTM (Long Short-Term Memory) for time series prediction.
Real-time predictions using live stock data.
License
This project is licensed under the MIT License - see the LICENSE file for details.