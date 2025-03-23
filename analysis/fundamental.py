import yfinance as yf
import pandas as pd
import numpy as np
from utils.llm_handler import sentiment_analyzer
from config.settings import *
from typing import Dict, Any, Tuple

class FundamentalAnalyzer:
    def __init__(self):
        self.llm = sentiment_analyzer
    
    def get_financial_metrics(self, ticker: str) -> Dict[str, Any]:
        """
        Fetch and calculate key financial metrics for the stock
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Basic Financial Metrics
            metrics = {
                "Market Cap": info.get('marketCap', 'N/A'),
                "P/E Ratio": info.get('forwardPE', 'N/A'),
                "P/B Ratio": info.get('priceToBook', 'N/A'),
                "Dividend Yield": info.get('dividendYield', 'N/A'),
                "52 Week High": info.get('fiftyTwoWeekHigh', 'N/A'),
                "52 Week Low": info.get('fiftyTwoWeekLow', 'N/A'),
                "Current Price": info.get('currentPrice', 'N/A'),
                "Volume": info.get('volume', 'N/A'),
                "Beta": info.get('beta', 'N/A'),
                "ROE": info.get('returnOnEquity', 'N/A'),
                "ROA": info.get('returnOnAssets', 'N/A'),
                "Debt to Equity": info.get('debtToEquity', 'N/A'),
                "Profit Margins": info.get('profitMargins', 'N/A'),
                "Operating Margins": info.get('operatingMargins', 'N/A'),
                "Revenue Growth": info.get('revenueGrowth', 'N/A'),
                "Earnings Growth": info.get('earningsGrowth', 'N/A'),
                "Free Cash Flow": info.get('freeCashflow', 'N/A'),
                "Enterprise Value": info.get('enterpriseValue', 'N/A'),
                "Book Value": info.get('bookValue', 'N/A'),
                "Price to Sales": info.get('priceToSalesTrailing12Months', 'N/A')
            }
            
            # Get financial statements
            balance_sheet = stock.balance_sheet
            income_stmt = stock.income_stmt
            cash_flow = stock.cash_flow
            
            # Add financial statement data if available
            if not balance_sheet.empty:
                metrics["Total Assets"] = balance_sheet.iloc[0].get('Total Assets', 'N/A')
                metrics["Total Liabilities"] = balance_sheet.iloc[0].get('Total Liabilities', 'N/A')
                metrics["Total Equity"] = balance_sheet.iloc[0].get('Total Stockholder Equity', 'N/A')
            
            if not income_stmt.empty:
                metrics["Revenue"] = income_stmt.iloc[0].get('Total Revenue', 'N/A')
                metrics["Net Income"] = income_stmt.iloc[0].get('Net Income', 'N/A')
                metrics["Operating Income"] = income_stmt.iloc[0].get('Operating Income', 'N/A')
            
            if not cash_flow.empty:
                metrics["Operating Cash Flow"] = cash_flow.iloc[0].get('Total Cash From Operating Activities', 'N/A')
                metrics["Investing Cash Flow"] = cash_flow.iloc[0].get('Total Cashflows From Investing Activities', 'N/A')
                metrics["Financing Cash Flow"] = cash_flow.iloc[0].get('Total Cash From Financing Activities', 'N/A')
            
            return metrics
            
        except Exception as e:
            print(f"Error fetching financial metrics: {e}")
            return {}
    def calculate_ratios(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate important financial ratios, ensuring numerical values are used.
        """
        ratios = {}

        try:
            # Helper function to safely convert values to float
            def safe_float(value):
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return None  # Return None if conversion fails
            
            # Convert necessary financial values to float
            total_current_assets = safe_float(metrics.get('Total Current Assets'))
            total_current_liabilities = safe_float(metrics.get('Total Current Liabilities'))
            inventory = safe_float(metrics.get('Inventory'))
            total_liabilities = safe_float(metrics.get('Total Liabilities'))
            total_equity = safe_float(metrics.get('Total Equity'))
            net_income = safe_float(metrics.get('Net Income'))
            total_assets = safe_float(metrics.get('Total Assets'))
            operating_income = safe_float(metrics.get('Operating Income'))
            revenue = safe_float(metrics.get('Revenue'))

            # Current Ratio
            if total_current_assets is not None and total_current_liabilities is not None and total_current_liabilities != 0:
                ratios['Current Ratio'] = total_current_assets / total_current_liabilities
            
            # Quick Ratio
            if total_current_assets is not None and inventory is not None and total_current_liabilities is not None and total_current_liabilities != 0:
                ratios['Quick Ratio'] = (total_current_assets - inventory) / total_current_liabilities
            
            # Debt to Equity Ratio
            if total_liabilities is not None and total_equity is not None and total_equity != 0:
                ratios['Debt to Equity'] = total_liabilities / total_equity
            
            # Return on Equity (ROE)
            if net_income is not None and total_equity is not None and total_equity != 0:
                ratios['ROE'] = net_income / total_equity
            
            # Return on Assets (ROA)
            if net_income is not None and total_assets is not None and total_assets != 0:
                ratios['ROA'] = net_income / total_assets
            
            # Operating Margin
            if operating_income is not None and revenue is not None and revenue != 0:
                ratios['Operating Margin'] = operating_income / revenue
            
            # Net Profit Margin
            if net_income is not None and revenue is not None and revenue != 0:
                ratios['Net Profit Margin'] = net_income / revenue
            
            return ratios

        except Exception as e:
            print(f"Error calculating ratios: {e}")
            return {}

    
    def analyze_fundamentals(self, ticker: str, invest_duration: int, last_3_days_prices: str = "") -> Tuple[Dict[str, Any], str]:
        """
        Perform comprehensive fundamental analysis
        
        Args:
            ticker (str): Stock symbol
            invest_duration (int): Investment duration in days
            last_3_days_prices (str): String containing last 3 days' prices
            
        Returns:
            Tuple[Dict[str, Any], str]: Financial metrics and LLM analysis
        """
        # Get financial metrics
        metrics = self.get_financial_metrics(ticker)
        
        # Calculate ratios
        ratios = self.calculate_ratios(metrics)
        
        # Combine metrics and ratios for analysis
        analysis_data = {**metrics, **ratios}
        
        # Helper function to format values
        def format_value(value):
            if isinstance(value, (int, float)):
                return f"{value:.2f}"
            return str(value)
        
        # Create analysis text for LLM
        analysis_text = f"""
        Fundamental Analysis Data for {ticker}:
        
        Recent Price History:
        {last_3_days_prices}
        
        Key Metrics:
        - Market Cap: {format_value(analysis_data.get('Market Cap', 'N/A'))}
        - P/E Ratio: {format_value(analysis_data.get('P/E Ratio', 'N/A'))}
        - P/B Ratio: {format_value(analysis_data.get('P/B Ratio', 'N/A'))}
        - Dividend Yield: {format_value(analysis_data.get('Dividend Yield', 'N/A'))}
        - Current Price: {format_value(analysis_data.get('Current Price', 'N/A'))}
        
        Financial Ratios:
        - Current Ratio: {format_value(analysis_data.get('Current Ratio', 'N/A'))}
        - Quick Ratio: {format_value(analysis_data.get('Quick Ratio', 'N/A'))}
        - Debt to Equity: {format_value(analysis_data.get('Debt to Equity', 'N/A'))}
        - ROE: {format_value(analysis_data.get('ROE', 'N/A'))}
        - ROA: {format_value(analysis_data.get('ROA', 'N/A'))}
        - Operating Margin: {format_value(analysis_data.get('Operating Margin', 'N/A'))}
        - Net Profit Margin: {format_value(analysis_data.get('Net Profit Margin', 'N/A'))}
        
        Growth Metrics:
        - Revenue Growth: {format_value(analysis_data.get('Revenue Growth', 'N/A'))}
        - Earnings Growth: {format_value(analysis_data.get('Earnings Growth', 'N/A'))}
        
        Cash Flow:
        - Operating Cash Flow: {format_value(analysis_data.get('Operating Cash Flow', 'N/A'))}
        - Free Cash Flow: {format_value(analysis_data.get('Free Cash Flow', 'N/A'))}
        """
        
        # Get LLM analysis
        llm_analysis = self.llm.analyze_fundamentals(ticker, analysis_text, invest_duration)
        
        return analysis_data, llm_analysis

# Create a singleton instance
fundamental_analyzer = FundamentalAnalyzer()

if __name__ == "__main__":
    # Example usage
    ticker = "RELIANCE.NS"  # Example Indian stock ticker
    invest_duration = 30
    metrics, analysis = fundamental_analyzer.analyze_fundamentals(ticker, invest_duration)
    print("\nFinancial Metrics:")
    print(metrics)
    print("\nFundamental Analysis:")
    print(analysis) 