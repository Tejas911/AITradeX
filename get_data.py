import yfinance as yf
import datetime
import pandas as pd
from datetime import timezone, timedelta

def get_user_input():
    # Collect stock ticker
    ticker = input("Enter stock ticker (e.g., AAPL, TSLA): ").strip().upper()
    
    # Collect investment duration in days
    while True:
        try:
            invest_duration = int(input("Enter investment duration [Range: 1-30] (in days): "))
            if invest_duration <= 0:
                print("Please enter a positive number for the duration.")
            else:
                break
        except ValueError:
            print("Invalid input. Please enter a valid number.")
    
    # Select ML training period (1, 2, or 3 years)
    print("Select ML Model Training Period:")
    print("1: 1 year")
    print("2: 2 years")
    print("3: 3 years (default)")
    
    training_choice = input("Enter choice (1, 2, or 3): ").strip()
    # Map the choice to days (approximately)
    training_period_days = {
        "1": 365,
        "2": 730,
        "3": 1095
    }.get(training_choice, 1095)  # Defaults to 3 years if invalid choice
    
    return ticker, invest_duration, training_period_days

def fetch_stock_data(ticker, period_days):
    # Get current date in US timezone
    us_timezone = timezone(timedelta(hours=-4))  # EDT (UTC-4)
    end_date = datetime.datetime.now(us_timezone).date()
    start_date = end_date - datetime.timedelta(days=period_days)
    
    # Ensure we're not trying to fetch future data
    if start_date > end_date:
        print("Error: Start date cannot be in the future.")
        return pd.DataFrame()
    
    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    
    try:
        # Fetch historical data from Yahoo Finance
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        
        if stock_data.empty:
            print("No data found for the ticker. Please check the stock symbol and try again.")
        else:
            print("Data retrieval successful!")
            print(f"Retrieved {len(stock_data)} data points")
        
        return stock_data
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        print("Please try again with a valid stock symbol.")
        return pd.DataFrame()

if __name__ == '__main__':
    # Step 1: Get user inputs
    ticker, invest_duration, training_period_days = get_user_input()
    
    # Display user inputs for confirmation
    print(f"\nUser Input Summary:")
    print(f"Stock Ticker: {ticker}")
    print(f"Investment Duration: {invest_duration} days")
    print(f"ML Model Training Period: {training_period_days // 365} year(s)")
    
    # Step 2: Fetch stock data using Yahoo Finance
    data = fetch_stock_data(ticker, training_period_days)
    
    # Show a preview of the fetched data
    if not data.empty:
        print("\nPreview of fetched data:")
        print(data.head())
