import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, Any, Tuple, List
import talib
from datetime import datetime, timedelta
from llm import sentiment_analyzer

class TechnicalAnalyzer:
    def __init__(self):
        self.llm = sentiment_analyzer
    
    def fetch_historical_data(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        """
        Fetch historical price data for technical analysis
        """
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval="1d")
            if df.empty:
                raise ValueError(f"No data found for ticker {ticker}")
            return df
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate various technical indicators with error handling for missing or non-numeric data.
        """
        indicators = {}

        try:
            # Ensure the dataframe is not empty
            if df.empty:
                raise ValueError("Empty dataframe provided")
            
            # Convert columns to numeric and handle non-numeric values
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            df['High'] = pd.to_numeric(df['High'], errors='coerce')
            df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
            df['Open'] = pd.to_numeric(df['Open'], errors='coerce')

            # Drop rows with NaN values
            df.dropna(subset=['Close', 'High', 'Low', 'Volume', 'Open'], inplace=True)

            # Convert to numpy arrays with explicit float64 type
            close = df['Close'].values.astype(np.float64)
            high = df['High'].values.astype(np.float64)
            low = df['Low'].values.astype(np.float64)
            volume = df['Volume'].values.astype(np.float64)
            open_price = df['Open'].values.astype(np.float64)

            # Ensure data is available before calculations
            if len(close) == 0 or len(high) == 0 or len(low) == 0 or len(volume) == 0:
                raise ValueError("Insufficient data after cleaning")

            # Helper function to safely get last value
            def safe_get_last(arr):
                if isinstance(arr, (list, np.ndarray)):
                    return float(arr[-1]) if len(arr) > 0 else None
                return float(arr) if arr is not None else None

            try:
                # Trend Indicators
                indicators['SMA_20'] = safe_get_last(talib.SMA(close, timeperiod=20))
                indicators['SMA_50'] = safe_get_last(talib.SMA(close, timeperiod=50))
                indicators['SMA_200'] = safe_get_last(talib.SMA(close, timeperiod=200))
                indicators['EMA_20'] = safe_get_last(talib.EMA(close, timeperiod=20))
                indicators['EMA_50'] = safe_get_last(talib.EMA(close, timeperiod=50))

                # Momentum Indicators
                indicators['RSI'] = safe_get_last(talib.RSI(close, timeperiod=14))
                macd, macd_signal, macd_hist = talib.MACD(close)
                indicators['MACD'] = safe_get_last(macd)
                indicators['MACD_Signal'] = safe_get_last(macd_signal)
                indicators['MACD_Hist'] = safe_get_last(macd_hist)
                
                stoch_k, stoch_d = talib.STOCH(high, low, close)
                indicators['Stoch_K'] = safe_get_last(stoch_k)
                indicators['Stoch_D'] = safe_get_last(stoch_d)

                # Volume Indicators
                indicators['OBV'] = safe_get_last(talib.OBV(close, volume))
                indicators['AD'] = safe_get_last(talib.AD(high, low, close, volume))

                # Volatility Indicators
                indicators['ATR'] = safe_get_last(talib.ATR(high, low, close, timeperiod=14))
                bb_upper, bb_middle, bb_lower = talib.BBANDS(close)
                indicators['BB_Upper'] = safe_get_last(bb_upper)
                indicators['BB_Middle'] = safe_get_last(bb_middle)
                indicators['BB_Lower'] = safe_get_last(bb_lower)

                # Support & Resistance Levels
                indicators['Support'] = self._calculate_support_levels(df)[:3]
                indicators['Resistance'] = self._calculate_resistance_levels(df)[:3]

                # Pattern Recognition
                patterns = {}
                patterns['Doji'] = talib.CDLDOJI(open=open_price, high=high, low=low, close=close)[-1] != 0
                patterns['Engulfing'] = talib.CDLENGULFING(open=open_price, high=high, low=low, close=close)[-1] != 0
                patterns['Morning Star'] = talib.CDLMORNINGSTAR(open=open_price, high=high, low=low, close=close)[-1] != 0
                patterns['Evening Star'] = talib.CDLEVENINGSTAR(open=open_price, high=high, low=low, close=close)[-1] != 0
                patterns['Hammer'] = talib.CDLHAMMER(open=open_price, high=high, low=low, close=close)[-1] != 0
                patterns['Shooting Star'] = talib.CDLSHOOTINGSTAR(open=open_price, high=high, low=low, close=close)[-1] != 0
                indicators['Patterns'] = patterns

            except Exception as e:
                print(f"Error in technical calculation: {e}")
                return {}

            return indicators

        except Exception as e:
            print(f"Error calculating technical indicators: {e}")
            return {}
    
    def _calculate_support_levels(self, df: pd.DataFrame) -> List[float]:
        """
        Calculate support levels using pivot points and recent lows
        """
        try:
            # Calculate pivot points
            pivot = (df['High'] + df['Low'] + df['Close']) / 3
            s1 = 2 * pivot - df['High']
            s2 = pivot - (df['High'] - df['Low'])
            
            # Get recent lows
            recent_lows = df['Low'].rolling(window=20).min().dropna()
            
            # Combine and get unique levels
            support_levels = list(set(s1.tolist() + s2.tolist() + recent_lows.tolist()))
            support_levels.sort(reverse=True)
            
            return support_levels[:3]  # Return top 3 support levels
        except Exception as e:
            print(f"Error calculating support levels: {e}")
            return []
    
    def _calculate_resistance_levels(self, df: pd.DataFrame) -> List[float]:
        """
        Calculate resistance levels using pivot points and recent highs
        """
        try:
            # Calculate pivot points
            pivot = (df['High'] + df['Low'] + df['Close']) / 3
            r1 = 2 * pivot - df['Low']
            r2 = pivot + (df['High'] - df['Low'])
            
            # Get recent highs
            recent_highs = df['High'].rolling(window=20).max().dropna()
            
            # Combine and get unique levels
            resistance_levels = list(set(r1.tolist() + r2.tolist() + recent_highs.tolist()))
            resistance_levels.sort()
            
            return resistance_levels[:3]  # Return top 3 resistance levels
        except Exception as e:
            print(f"Error calculating resistance levels: {e}")
            return []
    
    def _identify_patterns(self, df: pd.DataFrame) -> Dict[str, bool]:
        """
        Identify common chart patterns
        """
        patterns = {}
        try:
            # Convert price data to numpy arrays
            close = df['Close'].values
            high = df['High'].values
            low = df['Low'].values
            open_price = df['Open'].values
            
            # Identify various patterns
            patterns['Doji'] = talib.CDLDOJI(open=open_price, high=high, low=low, close=close)[-1] != 0
            patterns['Engulfing'] = talib.CDLENGULFING(open=open_price, high=high, low=low, close=close)[-1] != 0
            patterns['Morning Star'] = talib.CDLMORNINGSTAR(open=open_price, high=high, low=low, close=close)[-1] != 0
            patterns['Evening Star'] = talib.CDLEVENINGSTAR(open=open_price, high=high, low=low, close=close)[-1] != 0
            patterns['Hammer'] = talib.CDLHAMMER(open=open_price, high=high, low=low, close=close)[-1] != 0
            patterns['Shooting Star'] = talib.CDLSHOOTINGSTAR(open=open_price, high=high, low=low, close=close)[-1] != 0
            
            return patterns
        except Exception as e:
            print(f"Error identifying patterns: {e}")
            return {}
    
    def analyze_technicals(self, ticker: str, invest_duration: int) -> Tuple[Dict[str, Any], str]:
        """
        Perform comprehensive technical analysis
        
        Args:
            ticker (str): Stock symbol
            invest_duration (int): Investment duration in days
            
        Returns:
            Tuple[Dict[str, Any], str]: Technical indicators and LLM analysis
        """
        # Fetch historical data
        df = self.fetch_historical_data(ticker)
        
        if df.empty:
            return {}, "Error: Unable to fetch historical data"
        
        # Calculate technical indicators
        indicators = self.calculate_technical_indicators(df)
        
        if not indicators:
            return {}, "Error: Unable to calculate technical indicators"
        
        # Get current price and recent data
        current_price = df['Close'].iloc[-1]
        price_change = ((current_price - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
        
        # Get last 3 days' prices
        last_3_days = df['Close'].tail(3)
        last_3_days_dates = df.index.tail(3)
        last_3_days_prices = "\n".join([
            f"- {date.strftime('%Y-%m-%d')}: ${price:.2f}"
            for date, price in zip(last_3_days_dates, last_3_days)
        ])
        
        # Create analysis text for LLM
        analysis_text = f"""
        Technical Analysis Data for {ticker}:
        
        Current Price: {current_price:.2f} (Change: {price_change:.2f}%)
        
        Last 3 Days Prices:
        {last_3_days_prices}
        
        Moving Averages:
        - 20-day SMA: {indicators.get('SMA_20', 'N/A')}
        - 50-day SMA: {indicators.get('SMA_50', 'N/A')}
        - 200-day SMA: {indicators.get('SMA_200', 'N/A')}
        
        Momentum Indicators:
        - RSI: {indicators.get('RSI', 'N/A')}
        - MACD: {indicators.get('MACD', 'N/A')}
        - MACD Signal: {indicators.get('MACD_Signal', 'N/A')}
        - Stochastic K: {indicators.get('Stoch_K', 'N/A')}
        - Stochastic D: {indicators.get('Stoch_D', 'N/A')}
        
        Volatility Indicators:
        - ATR: {indicators.get('ATR', 'N/A')}
        - Bollinger Bands:
          Upper: {indicators.get('BB_Upper', 'N/A')}
          Middle: {indicators.get('BB_Middle', 'N/A')}
          Lower: {indicators.get('BB_Lower', 'N/A')}
        
        Support Levels:
        {', '.join([f'{level:.2f}' for level in indicators.get('Support', [])])}
        
        Resistance Levels:
        {', '.join([f'{level:.2f}' for level in indicators.get('Resistance', [])])}
        
        Chart Patterns:
        {', '.join([f'{pattern}' for pattern, detected in indicators.get('Patterns', {}).items() if detected])}
        """
        
        # Get LLM analysis
        llm_analysis = self.llm.analyze_technicals(ticker, analysis_text, invest_duration)
        
        return indicators, llm_analysis

# Create a singleton instance
technical_analyzer = TechnicalAnalyzer()

if __name__ == "__main__":
    # Example usage
    ticker = "RELIANCE.NS"  # Example Indian stock ticker
    invest_duration = 30
    indicators, analysis = technical_analyzer.analyze_technicals(ticker, invest_duration)
    print("\nTechnical Indicators:")
    print(indicators)
    print("\nTechnical Analysis:")
    print(analysis) 

    