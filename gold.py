import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

# Step 1: Get Gold Price Data
gold_data = yf.download("GC=F", start="2020-01-01", end=datetime.today().strftime('%Y-%m-%d'))
gold_data.reset_index(inplace=True)

# Step 2: Prepare Data
gold_data['5_day_avg'] = gold_data['Close'].rolling(5).mean()  # 5-day average
gold_data['20_day_avg'] = gold_data['Close'].rolling(20).mean()  # 20-day average
gold_data['Next_Close'] = gold_data['Close'].shift(-1)  # What we want to predict
gold_data.dropna(inplace=True)  # Remove empty rows

# Step 3: Train Prediction Model
features = ['Open', 'High', 'Low', 'Close', '5_day_avg', '20_day_avg']
X = gold_data[features]
y = gold_data['Next_Close']

# Use last 100 days for testing, rest for training
train_size = len(gold_data) - 100
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# Step 4: Make Predictions
last_data = gold_data.iloc[-1][features]
future_dates = [datetime.today() + timedelta(days=i) for i in range(1, 31)]
future_prices = []

for _ in range(30):
    pred = model.predict([last_data])[0]
    future_prices.append(pred)
    # Update the features for next prediction
    last_data['5_day_avg'] = (last_data['5_day_avg'] * 4 + pred) / 5
    last_data['20_day_avg'] = (last_data['20_day_avg'] * 19 + pred) / 20
    last_data['Open'] = pred
    last_data['High'] = pred * 1.01
    last_data['Low'] = pred * 0.99
    last_data['Close'] = pred

# Step 5: Show Results
plt.figure(figsize=(12, 6))
plt.plot(gold_data['Date'], gold_data['Close'], label='Historical Prices', color='blue')
plt.plot(future_dates, future_prices, label='Predicted Prices', color='red', linestyle='--')
plt.title('Gold Price Prediction (Next 30 Days)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()

print("\nPredicted Prices for Next 7 Days:")
for i, (date, price) in enumerate(zip(future_dates[:7], future_prices[:7]), 1):
    print(f"Day {i}: {date.strftime('%Y-%m-%d')} - ${price:.2f}")