import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

# 1. Download silver price data
print("Downloading silver price data...")
silver_data = yf.download("SI=F", start="2020-01-01", end=datetime.today().strftime('%Y-%m-%d'))
silver_data.reset_index(inplace=True)

# 2. Prepare the data
print("Preparing data...")
silver_data['5_day_avg'] = silver_data['Close'].rolling(5).mean()
silver_data['20_day_avg'] = silver_data['Close'].rolling(20).mean()
silver_data['Next_Close'] = silver_data['Close'].shift(-1)  # What we want to predict
silver_data.dropna(inplace=True)  # Remove rows with missing values

# 3. Create and train the model
print("Training prediction model...")
features = ['Open', 'High', 'Low', 'Close', '5_day_avg', '20_day_avg']
X = silver_data[features]
y = silver_data['Next_Close']

# Use last 100 days for testing
train_size = len(silver_data) - 100
X_train, y_train = X[:train_size], y[:train_size]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Make future predictions
print("Making predictions...")
last_data = silver_data.iloc[-1][features]
future_dates = [datetime.today() + timedelta(days=i) for i in range(1, 31)]
future_prices = []

for _ in range(30):
    pred = model.predict([last_data])[0]
    future_prices.append(pred)
    # Update features for next prediction
    last_data['5_day_avg'] = (last_data['5_day_avg'] * 4 + pred) / 5
    last_data['20_day_avg'] = (last_data['20_day_avg'] * 19 + pred) / 20
    last_data['Open'] = pred
    last_data['High'] = pred * 1.01  # Small random variation
    last_data['Low'] = pred * 0.99
    last_data['Close'] = pred

# 5. Show results
plt.figure(figsize=(12, 6))
plt.plot(silver_data['Date'], silver_data['Close'], label='Historical Prices', color='blue')
plt.plot(future_dates, future_prices, label='Predicted Prices', color='red', linestyle='--')
plt.title('Silver Price Prediction (Next 30 Days)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()

print("\nPredicted Silver Prices:")
for i, (date, price) in enumerate(zip(future_dates[:7], future_prices[:7]), 1):
    print(f"Day {i} ({date.strftime('%Y-%m-%d')}): ${price:.2f}")