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
        You are a financial sentiment analysis expert.
        Analyze the following aggregated text about the stock {stock_ticker} from multiple sources.
        The investment duration is {invest_duration} days.
        
        Based on the sentiment analysis and recent price history, provide:
        1. Overall sentiment (positive, negative, or neutral)
        2. Key factors driving the sentiment
        3. Price impact assessment
        4. Potential price movement range for the next {invest_duration} days
        
        Format your response as follows:
        
        --- Sentiment Analysis Result ---
        **Overall Sentiment:** [Positive/Negative/Neutral]
        
        **Key Factors Driving Sentiment:**
        - [Factor 1]
        - [Factor 2]
        - [Factor 3]
        
        **Recent Price Analysis:**
        [Analyze the recent price movements and their significance]
        
        **Price Impact Assessment:**
        - Short-term impact: [Description]
        - Medium-term outlook: [Description]
        
        **Price Movement Prediction ({invest_duration} days):**
        - Predicted Direction: [Upward/Downward/Sideways]
        - Potential Range: [Lower bound] to [Upper bound]
        - Confidence Level: [High/Medium/Low]
        
        **Investment Recommendation:**
        [Clear recommendation with risk factors]
        
        Text to analyze:
        {text}
        """
        
        self.fundamental_prompt_template = """
        You are a fundamental analysis expert specializing in stock markets.
        Analyze the following financial data for the stock {stock_ticker} and provide a comprehensive analysis.
        The investment duration is {invest_duration} days.
        
        Based on the fundamental metrics, recent price history, and market conditions, provide:
        1. Comprehensive valuation analysis
        2. Growth potential assessment
        3. Risk evaluation
        4. Price target prediction
        
        Format your response as follows:
        
        --- Fundamental Analysis Result ---
        **Valuation Summary:**
        - Current Valuation: [Fair/Overvalued/Undervalued]
        - Key Valuation Metrics Analysis
        - Industry Comparison
        
        **Growth Analysis:**
        - Revenue Growth Trajectory
        - Earnings Growth Potential
        - Market Position
        
        **Risk Assessment:**
        - Financial Health Indicators
        - Market Risks
        - Company-Specific Risks
        
        **Price Target Analysis ({invest_duration} days):**
        - Fair Value Estimate: [Calculate based on fundamentals]
        - Price Target Range: [Lower bound] to [Upper bound]
        - Key Catalysts:
          * [Catalyst 1]
          * [Catalyst 2]
        
        **Investment Thesis:**
        [Detailed investment reasoning]
        
        **Price Prediction Confidence:**
        - Confidence Level: [High/Medium/Low]
        - Supporting Factors:
          * [Factor 1]
          * [Factor 2]
        
        Financial Data:
        {text}
        """
        
        self.technical_prompt_template = """
        You are a technical analysis expert specializing in stock markets.
        Analyze the following technical data for the stock {stock_ticker} and provide a comprehensive analysis.
        The investment duration is {invest_duration} days.
        
        Based on the technical indicators, chart patterns, and price action, provide:
        1. Trend analysis
        2. Support/Resistance levels
        3. Momentum assessment
        4. Price target prediction
        
        Format your response as follows:
        
        --- Technical Analysis Result ---
        **Trend Analysis:**
        - Primary Trend: [Uptrend/Downtrend/Sideways]
        - Trend Strength: [Strong/Moderate/Weak]
        - Moving Average Analysis
        
        **Price Levels:**
        - Key Support Levels: [List levels]
        - Key Resistance Levels: [List levels]
        - Breakout/Breakdown Points
        
        **Momentum Indicators:**
        - RSI Analysis
        - MACD Signals
        - Volume Analysis
        
        **Pattern Recognition:**
        - Identified Patterns
        - Pattern Reliability
        - Potential Targets
        
        **Price Prediction ({invest_duration} days):**
        - Technical Target Range: [Lower bound] to [Upper bound]
        - Key Levels to Watch:
          * Upside Target: [Level]
          * Stop Loss: [Level]
        
        **Confidence Analysis:**
        - Signal Strength: [Strong/Moderate/Weak]
        - Confirming Indicators
        - Contradicting Indicators
        
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
