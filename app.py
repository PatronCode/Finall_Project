from flask import Flask, render_template, request, redirect, url_for, flash, session
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from datetime import datetime
from functools import wraps
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import bcrypt
from binance.client import Client

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')


MONGODB_URI = os.getenv('MONGODB_URI')
client = MongoClient(MONGODB_URI)
db = client.get_database('crypto_prediction')
users = db['users']


api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')
binance_client = Client(api_key, api_secret)


model = load_model('lstm_binance_model.h5')
scaler = joblib.load('scaler_binance.pkl')

def get_current_bitcoin_price():
    try:
        
        ticker = binance_client.get_symbol_ticker(symbol="BTCUSDT")
       
        current_price = float(ticker['price'])
        last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return current_price, last_update
    except Exception as e:
        print(f"Error fetching price: {e}")
        return None, None


klines = binance_client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1DAY, "1 Jan, 2017")
data = pd.DataFrame(klines, columns=[
    'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
    'Close Time', 'Quote Asset Volume', 'Number of Trades',
    'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'
])
data['Date'] = pd.to_datetime(data['Open Time'], unit='ms')
data['Close'] = data['Close'].astype(float)
data = data[['Date', 'Close']]


scaled_data = scaler.transform(data['Close'].values.reshape(-1, 1))
best_timestep = 30
last_sequence = scaled_data[-best_timestep:]
last_known_date = data['Date'].iloc[-1]

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            flash('Please login first.')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def predict_future(target_date):
    target_date = pd.to_datetime(target_date)
    days_ahead = (target_date - last_known_date).days
    
    if days_ahead <= 0:
        return None, "Target date must be in the future!"

    current_input = last_sequence.copy()
    prediction = None
    
    for _ in range(days_ahead):
        prediction = model.predict(current_input.reshape(1, best_timestep, 1))
        print(prediction)
        current_input = np.append(current_input[1:], prediction, axis=0)

    predicted_price = scaler.inverse_transform(prediction)[0, 0]
    print(predicted_price)
    return predicted_price, None

def predict_historical(target_date):
    target_date = pd.to_datetime(target_date)
    historical_data = data[data['Date'] <= target_date].copy()
  
    if len(historical_data) < best_timestep:
        return None, None, "Not enough historical data available for this date"
    
    actual_price = None
    if not historical_data[historical_data['Date'] == target_date].empty:
        actual_price = historical_data[historical_data['Date'] == target_date]['Close'].values[0]

    historical_scaled = scaler.transform(historical_data['Close'].values.reshape(-1, 1))
    input_sequence = historical_scaled[-best_timestep-1:-1]
    
    prediction = model.predict(input_sequence.reshape(1, best_timestep, 1))
    predicted_price = scaler.inverse_transform(prediction)[0, 0]
    
    # Calculate accuracy
    accuracy = None
    if actual_price is not None:
        accuracy = (1 - abs(predicted_price - actual_price) / actual_price) * 100
    
    return predicted_price, actual_price, accuracy

@app.route('/')
def landing():
    current_price, last_update = get_current_bitcoin_price()
    if current_price is None:
        current_price = 0
        last_update = "Failed to fetch price"
    return render_template('landing.html', current_price=current_price, last_update=last_update)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        
        if users.find_one({'email': email}):
            flash('Email already exists')
            return redirect(url_for('signup'))
        
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        
       
        users.insert_one({
            'email': email,
            'password': hashed_password
        })
    
        
        flash('Registration successful')
        return redirect(url_for('login'))
    
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        user = users.find_one({'email': email})
        
        if user and bcrypt.checkpw(password.encode('utf-8'), user['password']):
            session['user'] = email
            return redirect(url_for('dashboard'))
        
        flash('Invalid email or password')
        return redirect(url_for('login'))
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('landing'))

@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    prediction = None
    error = None
    input_date = None
    current_price, last_update = get_current_bitcoin_price()
    
    if request.method == 'POST':
        input_date = request.form['date']
        try:
            prediction, error = predict_future(input_date)
        except Exception as e:
            error = str(e)
    
    return render_template('dashboard.html', 
                         prediction=prediction, 
                         error=error, 
                         input_date=input_date,
                         last_known_date=last_known_date.strftime('%Y-%m-%d'),
                         user=session['user'],
                         current_price=current_price,
                         last_update=last_update)

@app.route('/histdashboard', methods=['GET', 'POST'])
@login_required
def histdashboard():
    prediction = None
    actual_price = None
    accuracy = None
    error = None
    input_date = None
    current_price, last_update = get_current_bitcoin_price()
    
    if request.method == 'POST':
        input_date = request.form['date']
        try:
            prediction, actual_price, accuracy = predict_historical(input_date)
            if isinstance(accuracy, str):  
                error = accuracy
                prediction = None
                actual_price = None
                accuracy = None
        except Exception as e:
            error = str(e)

    earliest_date = data['Date'].iloc[best_timestep].strftime('%Y-%m-%d')
    
    return render_template('histdashboard.html', 
                         prediction=prediction,
                         actual_price=actual_price,
                         accuracy=accuracy,
                         error=error,
                         input_date=input_date,
                         earliest_date=earliest_date,
                         last_known_date=last_known_date.strftime('%Y-%m-%d'),
                         user=session['user'],
                         current_price=current_price,
                         last_update=last_update)

if __name__ == '__main__':
    app.run(debug=True)
