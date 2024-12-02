<<<<<<< HEAD
# Bitcoin Price Prediction Web Application

A Flask-based web application that provides real-time Bitcoin price tracking and future price predictions using machine learning. The application features user authentication, real-time price updates from Binance, and a machine learning model for price predictions.

## Features

### 1. Real-Time Price Tracking
- Live Bitcoin price updates from Binance API
- Automatic price refresh every 30 seconds
- Clean and modern UI with real-time display

### 2. Price Prediction
- Machine learning-based price prediction
- LSTM model for accurate forecasting
- User-friendly date selection interface
- Visual representation of predictions

### 3. User Authentication
- Secure user registration and login
- MongoDB Atlas database integration
- Password hashing with bcrypt
- Protected dashboard routes

### 4. Modern UI/UX
- Responsive design
- Glassmorphism effects
- Gradient color schemes
- Intuitive navigation
- Bootstrap integration

## Technology Stack

- **Backend**: Python Flask
- **Database**: MongoDB Atlas
- **Machine Learning**: TensorFlow (LSTM model)
- **Frontend**: HTML, CSS, Bootstrap
- **APIs**: Binance API for real-time data
- **Authentication**: bcrypt for password hashing
- **Environment Variables**: python-dotenv

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Unix/macOS
   venv\Scripts\activate     # For Windows    <this command>
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables**
   Create a `.env` file in the root directory with the following:
   ```
   MONGODB_URI=your_mongodb_connection_string
   BINANCE_API_KEY=your_binance_api_key
   BINANCE_API_SECRET=your_binance_api_secret
   SECRET_KEY=your_flask_secret_key
   ```

5. **Database Setup**
   - Create a MongoDB Atlas account
   - Create a new cluster
   - Add your connection string to `.env`
   - The application will automatically create required collections

6. **Run the Application**
   <!-- ```bash                       <this command> -->
   python app.py 
   ```
   The application will be available at `http://127.0.0.1:5000`

## Project Structure

```
├── app.py                 # Main Flask application
├── templates/            # HTML templates
│   ├── landing.html     # Landing page
│   ├── login.html       # Login page
│   ├── signup.html      # Registration page
│   └── dashboard.html   # Main dashboard
├── lstm_binance_model.h5 # Trained LSTM model
├── scaler_binance.pkl    # Data scaler
├── .env                  # Environment variables
└── requirements.txt      # Python dependencies
```

## Usage

1. **Landing Page**
   - View real-time Bitcoin price
   - Access login/signup options
   - View feature highlights

2. **Registration/Login**
   - Create a new account
   - Login with existing credentials
   - Secure password handling

3. **Dashboard**
   - View real-time Bitcoin price
   - Select future dates for prediction
   - View prediction results
   - Access historical data

## Security Features

- Password hashing using bcrypt
- Secure session management
- Protected routes with login_required decorator
- Environment variable protection
- MongoDB Atlas security features

## API Integration

### Binance API
- Real-time price data
- Historical data for model training
- Automatic updates

### MongoDB Atlas
- User data storage
- Secure connection
- Scalable database solution

## Error Handling

- Form validation
- API error handling
- Database connection error handling
- User-friendly error messages

## Contributing

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

#
