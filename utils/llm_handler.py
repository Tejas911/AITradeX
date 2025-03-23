from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

class LLMHandler:
    def __init__(self):
        self.api_key = self._load_api_key()
        self.llm = self._initialize_llm()
    
    def _load_api_key(self):
        """Load Groq API key from .env file"""
        load_dotenv()
        api_key = os.getenv('GROQ_API_KEY')
        
        if not api_key:
            raise ValueError("GROQ API key not found in .env file. Please check your .env file configuration.")
        
        return api_key
    
    def _initialize_llm(self):
        """Initialize the Groq LLM with default settings"""
        return ChatGroq(
            # model="llama-3.1-8b-instant",
            # model="llama3-70b-8192",
            model="deepseek-r1-distill-llama-70b",
            temperature=0.0,
            api_key=self.api_key
        )
    
    def create_chain(self, prompt_template: str, input_variables: list):
        """Create a RunnableSequence with the given prompt template"""
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=input_variables
        )
        return prompt | self.llm
    
    def run_chain(self, chain, **kwargs):
        """Run the chain with the given inputs"""
        return chain.invoke(kwargs)

# Sentiment Analysis specific functionality
class SentimentAnalyzer(LLMHandler):
    def __init__(self):
        super().__init__()
        self.sentiment_prompt_template = """
        You are an expert financial analyst with decades of experience in stock market analysis.
        Your recommendations are highly respected and have consistently outperformed the market.
        Analyze the following aggregated text about the stock {stock_ticker} from multiple sources.
        The investment duration is {invest_duration} days.
        
        Based on your expert analysis of the sentiment and recent price history, provide:
        1. Clear and decisive overall sentiment (positive, negative, or neutral)
        2. Key factors driving the sentiment
        3. Price impact assessment
        4. Precise price movement prediction
        
        Format your response as follows:
        
        --- Expert Sentiment Analysis Result ---
        **Overall Sentiment:** [Strongly Positive/Positive/Neutral/Negative/Strongly Negative]
        
        **Key Factors Driving Sentiment:**
        - [Factor 1]
        - [Factor 2]
        - [Factor 3]
        
        **Recent Price Analysis:**
        [Expert analysis of recent price movements and their significance]
        
        **Price Impact Assessment:**
        - Short-term impact: [Clear description]
        - Medium-term outlook: [Clear description]
        
        **Expert Price Prediction ({invest_duration} days):**
        - Predicted Direction: [Strongly Upward/Upward/Sideways/Downward/Strongly Downward]
        - Precise Price Range: ${{[Lower bound]}} to ${{[Upper bound]}}
        - Confidence Level: [Very High/High/Moderate/Low]
        
        **Expert Investment Recommendation:**
        [STRONG BUY/BUY/HOLD/SELL/STRONG SELL]
        
        **Risk Factors:**
        - [Risk 1]
        - [Risk 2]
        - [Risk 3]
        
        **Expert's Final Word:**
        [Clear, decisive statement about the investment opportunity]
        
        Text to analyze:
        {text}
        """
        
        self.fundamental_prompt_template = """
        You are a world-renowned fundamental analyst with exceptional track record in stock market analysis.
        Your recommendations are highly sought after by institutional investors.
        Analyze the following financial data for the stock {stock_ticker} and provide a comprehensive analysis.
        The investment duration is {invest_duration} days.
        
        Based on your expert analysis of the fundamental metrics, provide:
        1. Precise valuation assessment
        2. Growth potential evaluation
        3. Risk analysis
        4. Exact price target prediction
        
        Format your response as follows:
        
        --- Expert Fundamental Analysis Result ---
        **Valuation Summary:**
        - Current Valuation: [Significantly Overvalued/Overvalued/Fairly Valued/Undervalued/Significantly Undervalued]
        - Key Valuation Metrics Analysis
        - Industry Comparison
        
        **Growth Analysis:**
        - Revenue Growth Trajectory: [Strong/Moderate/Weak]
        - Earnings Growth Potential: [High/Medium/Low]
        - Market Position: [Market Leader/Strong Contender/Challenger]
        
        **Risk Assessment:**
        - Financial Health: [Excellent/Good/Fair/Poor]
        - Market Risks: [High/Medium/Low]
        - Company-Specific Risks: [High/Medium/Low]
        
        **Expert Price Target Analysis ({invest_duration} days):**
        - Fair Value: ${{[Exact Value]}}
        - Price Target Range: ${{[Lower bound]}} to ${{[Upper bound]}}
        - Key Catalysts:
          * [Catalyst 1]
          * [Catalyst 2]
        
        **Expert Investment Thesis:**
        [Clear, decisive investment reasoning]
        
        **Expert's Final Recommendation:**
        [STRONG BUY/BUY/HOLD/SELL/STRONG SELL]
        
        **Price Prediction Confidence:**
        - Confidence Level: [Very High/High/Moderate/Low]
        - Supporting Factors:
          * [Factor 1]
          * [Factor 2]
        
        **Risk Management:**
        - Stop Loss: ${{[Price]}}
        - Take Profit: ${{[Price]}}
        - Position Size: [Percentage of Portfolio]
        
        Financial Data:
        {text}
        """
        
        self.technical_prompt_template = """
        You are a legendary technical analyst with exceptional accuracy in predicting market movements.
        Your technical analysis has consistently identified major market trends and reversals.
        Analyze the following technical data for the stock {stock_ticker} and provide a comprehensive analysis.
        The investment duration is {invest_duration} days.
        
        Based on your expert analysis of the technical indicators, provide:
        1. Clear trend identification
        2. Precise support/resistance levels
        3. Momentum analysis
        4. Exact price target prediction
        
        Format your response as follows:
        
        --- Expert Technical Analysis Result ---
        **Trend Analysis:**
        - Primary Trend: [Strong Uptrend/Uptrend/Sideways/Downtrend/Strong Downtrend]
        - Trend Strength: [Very Strong/Strong/Moderate/Weak]
        - Moving Average Analysis
        
        **Price Levels:**
        - Key Support Levels: ${{[Level 1]}}, ${{[Level 2]}}, ${{[Level 3]}}
        - Key Resistance Levels: ${{[Level 1]}}, ${{[Level 2]}}, ${{[Level 3]}}
        - Breakout/Breakdown Points: ${{[Price]}}
        
        **Momentum Indicators:**
        - RSI Analysis: [Overbought/Neutral/Oversold]
        - MACD Signals: [Strong Bullish/Bullish/Neutral/Bearish/Strong Bearish]
        - Volume Analysis: [High/Moderate/Low]
        
        **Pattern Recognition:**
        - Identified Patterns: [List patterns]
        - Pattern Reliability: [High/Medium/Low]
        - Potential Targets: ${{[Price]}}
        
        **Expert Price Prediction ({invest_duration} days):**
        - Technical Target Range: ${{[Lower bound]}} to ${{[Upper bound]}}
        - Key Levels to Watch:
          * Upside Target: ${{[Price]}}
          * Stop Loss: ${{[Price]}}
        
        **Expert's Final Recommendation:**
        [STRONG BUY/BUY/HOLD/SELL/STRONG SELL]
        
        **Confidence Analysis:**
        - Signal Strength: [Very Strong/Strong/Moderate/Weak]
        - Confirming Indicators: [List indicators]
        - Contradicting Indicators: [List indicators]
        
        **Risk Management:**
        - Entry Price: ${{[Price]}}
        - Stop Loss: ${{[Price]}}
        - Take Profit: ${{[Price]}}
        - Position Size: [Percentage of Portfolio]
        
        Technical Data:
        {text}
        """
    
    def analyze_sentiment(self, stock_ticker: str, text: str, invest_duration: int) -> str:
        """Perform sentiment analysis on the given text"""
        chain = self.create_chain(
            prompt_template=self.sentiment_prompt_template,
            input_variables=["stock_ticker", "text", "invest_duration"]
        )
        result = self.run_chain(chain, 
                              stock_ticker=stock_ticker, 
                              text=text,
                              invest_duration=invest_duration)
        return result.content
    
    def analyze_fundamentals(self, stock_ticker: str, text: str, invest_duration: int) -> str:
        """Perform fundamental analysis on the given financial data"""
        chain = self.create_chain(
            prompt_template=self.fundamental_prompt_template,
            input_variables=["stock_ticker", "text", "invest_duration"]
        )
        result = self.run_chain(chain,
                              stock_ticker=stock_ticker,
                              text=text,
                              invest_duration=invest_duration)
        return result.content
    
    def analyze_technicals(self, stock_ticker: str, text: str, invest_duration: int) -> str:
        """Perform technical analysis on the given technical data"""
        chain = self.create_chain(
            prompt_template=self.technical_prompt_template,
            input_variables=["stock_ticker", "text", "invest_duration"]
        )
        result = self.run_chain(chain,
                              stock_ticker=stock_ticker,
                              text=text,
                              invest_duration=invest_duration)
        return result.content

# Create a singleton instance for easy access
sentiment_analyzer = SentimentAnalyzer()

# Example usage:
if __name__ == "__main__":
    # Example of using the sentiment analyzer
    test_text = "Sample text for testing sentiment analysis"
    result = sentiment_analyzer.analyze_sentiment("AAPL", test_text, 30)
    print(result)
