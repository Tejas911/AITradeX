from get_data import get_user_input, fetch_stock_data
from sentiment import perform_sentiment_analysis_multi
from fundamental_analysis import fundamental_analyzer
from technical_analysis import technical_analyzer
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np

def load_api_keys():
    """Load API keys from .env file"""
    load_dotenv()
    google_api_key = os.getenv('GOOGLE_API_KEY')
    groq_api_key = os.getenv('GROQ_API_KEY')
    
    if not google_api_key or not groq_api_key:
        raise ValueError("API keys not found in .env file. Please check your .env file configuration.")
    
    return google_api_key, groq_api_key

def main():
    print("Welcome to Stock Analysis System")
    print("=" * 30)
    
    try:
        # Load API keys
        google_api_key, groq_api_key = load_api_keys()
        print("API keys loaded successfully")
        
        # Get user inputs for stock analysis
        ticker, invest_duration, training_period_days = get_user_input()
        
        # Fetch historical stock data
        stock_data = fetch_stock_data(ticker, training_period_days)
        stock_data.to_csv('stock_data.csv')
        if not stock_data.empty:
            print("\nData Analysis Summary:")
            print(f"Total data points: {len(stock_data)}")
            print(f"Date range: {stock_data.index.min().strftime('%Y-%m-%d')} to {stock_data.index.max().strftime('%Y-%m-%d')}")
            print("\nLatest Stock Data:")
            print(stock_data.tail())
            
            # Get last 3 days' prices with proper formatting
            last_3_days_data = stock_data.tail(3)
            price_lines = []
            for index, row in last_3_days_data.iterrows():
                try:
                    date_str = index.strftime('%Y-%m-%d')
                    price = float(row['Close'])
                    price_lines.append(f"- {date_str}: ${price:.2f}")
                except (ValueError, TypeError) as e:
                    print(f"Warning: Error formatting price data: {e}")
                    continue
            last_3_days_prices = "\n".join(price_lines)
            
            # Perform sentiment analysis
            print("\nPerforming Sentiment Analysis...")
            sentiment_result = perform_sentiment_analysis_multi(ticker, invest_duration, last_3_days_prices)
            print("\nSentiment Analysis Results:")
            print(sentiment_result)
            
            # Perform fundamental analysis
            print("\nPerforming Fundamental Analysis...")
            metrics, fundamental_result = fundamental_analyzer.analyze_fundamentals(ticker, invest_duration, last_3_days_prices)
            print("\nKey Financial Metrics:")
            for key, value in metrics.items():
                if value != 'N/A':
                    try:
                        if isinstance(value, (int, float)):
                            print(f"{key}: {value:.2f}")
                        else:
                            print(f"{key}: {value}")
                    except (ValueError, TypeError):
                        print(f"{key}: {value}")
            print("\nFundamental Analysis Results:")
            print(fundamental_result)
            
            # Perform technical analysis
            print("\nPerforming Technical Analysis...")
            indicators, technical_result = technical_analyzer.analyze_technicals(ticker, invest_duration)
            print("\nKey Technical Indicators:")
            for key, value in indicators.items():
                try:
                    if isinstance(value, (list, np.ndarray)) and len(value) > 0:
                        print(f"{key}: {value[-1]:.2f}")
                    elif isinstance(value, dict):
                        print(f"{key}: {', '.join([f'{k}' for k, v in value.items() if v])}")
                    elif isinstance(value, (int, float)):
                        print(f"{key}: {value:.2f}")
                    else:
                        print(f"{key}: {value}")
                except (ValueError, TypeError):
                    print(f"{key}: {value}")
            print("\nTechnical Analysis Results:")
            print(technical_result)
            
    except ValueError as ve:
        print(f"Configuration Error: {str(ve)}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please try again with valid inputs.")

if __name__ == "__main__":
    main()
