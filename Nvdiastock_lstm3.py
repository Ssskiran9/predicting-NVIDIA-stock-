import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import plotly.graph_objects as go
from tensorflow.keras.metrics import MeanAbsoluteError, MeanSquaredError, RootMeanSquaredError
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer

# Download stock data
def get_stock_data(symbol, start_date, end_date):
    stock = yf.download(symbol, start=start_date, end=end_date)
    # Flatten the MultiIndex columns
    stock.columns = ['_'.join(col) for col in stock.columns]
    return stock

# Add technical indicators using pandas_ta
def add_indicators(df):
    # Ensure single-level column index
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col) for col in df.columns]
    
    # Add RSI
    df.ta.rsi(length=14, append=True)

    #Add VWAP
    df.ta.vwap(append = True)
    
    # Add Bollinger Bands
    df.ta.bbands(length=20, append=True)
    
    #Add EMA 20
    df.ta.ema(length=9, append=True)
    
    print(df.columns)  # Debugging: Check the column names
    return df

# Prepare data for LSTM
def prepare_data(df, lookback=60):
    # Select features
    features = ['Close_NVDA', 'Volume_NVDA', 'RSI_14', 'VWAP_D', 
                'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0',  'EMA_9']
    
    # Impute missing values using mean imputation
    imputer = SimpleImputer(strategy='mean')
    df[features] = imputer.fit_transform(df[features])
    
    # Scale the features
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])
    
    X, y = [], []
    for i in range(lookback, len(scaled_data) - 1):
        X.append(scaled_data[i-lookback:i])
        y.append(scaled_data[i + 1, 0])  # 0 = Close price
        
    return np.array(X), np.array(y), scaler

# Modified LSTM model using Functional API with single LSTM layer
def create_model(lookback, n_features):
    # Input layer
    inputs = Input(shape=(lookback, n_features))
    
    # Single LSTM layer with increased units to maintain model capacity
    x = LSTM(units=200)(inputs)
    x = Dropout(0.2)(x)
    
    # Output layer
    outputs = Dense(1)(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', metrics=[
            MeanAbsoluteError(name='mae'),
            MeanSquaredError(name='mse'),
            RootMeanSquaredError(name='rmse')
        ])
    
    return model

# Main execution
def main():
    # Get stock data
    symbol = 'NVDA'
    df = get_stock_data(symbol, '2020-01-01', '2024-12-31')
    
    # Add technical indicators
    df = add_indicators(df)
    
    # Prepare data
    X, y, scaler = prepare_data(df)
    
    # Split data
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Create and train model
    model = create_model(lookback=60, n_features=X.shape[2])
    
    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Make predictions
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    print("train predictions:", train_predictions.shape)
    print("test predictions:", test_predictions.shape)

    train_predictions = train_predictions.reshape(-1,1)
    test_predictions = test_predictions.reshape(-1,1)
    y_train = y_train.reshape(-1,1)
    y_test = y_test.reshape(-1,1)
    
    # Inverse transform predictions
    n_features = scaler.n_features_in_
    train_predictions = scaler.inverse_transform(
        np.concatenate([train_predictions, np.zeros((len(train_predictions), n_features - 1))], axis=1)
    )[:, 0]
    
    test_predictions = scaler.inverse_transform(
        np.concatenate([test_predictions, np.zeros((len(test_predictions), n_features - 1))], axis=1)
    )[:, 0]

    # Inverse-transform y_train and y_test
    y_train_original = scaler.inverse_transform(
        np.concatenate([y_train, np.zeros((len(y_train), n_features - 1))], axis=1)
    )[:, 0]
    y_test_original = scaler.inverse_transform(
        np.concatenate([y_test, np.zeros((len(y_test), n_features - 1))], axis=1)
    )[:, 0]

    # Calculate R² score
    train_r2 = r2_score(y_train_original, train_predictions)
    test_r2 = r2_score(y_test_original, test_predictions)

    print(f"Training R² Score: {train_r2:.4f}")
    print(f"Testing R² Score: {test_r2:.4f}")
    
    # Plot results using plotly
    fig = go.Figure()
    
    # Add actual prices
    fig.add_trace(go.Scatter(
        x=df.index[60:train_size+60],
        y=df['Close_NVDA'][60:train_size+60],
        name='Training Actual',
        mode='lines'
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index[train_size+60:],
        y=df['Close_NVDA'][train_size+60:],
        name='Testing Actual',
        mode='lines'
    ))
    
    # Add predictions
    fig.add_trace(go.Scatter(
        x=df.index[60:train_size+60],
        y=train_predictions,
        name='Training Predictions',
        mode='lines'
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index[train_size+60:],
        y=test_predictions,
        name='Testing Predictions',
        mode='lines'
    ))
    
    fig.update_layout(
        title=f'{symbol} Stock Price Prediction',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_dark'
    )
    
    fig.show()

if __name__ == "__main__":
    main()