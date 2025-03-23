from langchain.document_loaders import WebBaseLoader
from llm import sentiment_analyzer

def load_documents_from_urls(urls):
    """
    Given a list of URLs, load documents from each using WebBaseLoader.
    Returns a list of documents (each with a .page_content attribute).
    """
    documents = []
    for url in urls:
        try:
            print(f"Loading data from: {url}")
            loader = WebBaseLoader(url)
            docs = loader.load()
            documents.extend(docs)
        except Exception as e:
            print(f"Error loading {url}: {e}")
    return documents

def perform_sentiment_analysis_multi(stock_ticker: str, invest_duration: int, last_3_days_prices: str = "") -> str:
    """
    Performs sentiment analysis for a given stock ticker by loading news content
    from multiple Indian financial websites and then using an LLM (via LangChain) to generate a sentiment summary.
    
    Args:
        stock_ticker (str): The stock symbol (e.g., "RELIANCE", "TCS").
        invest_duration (int): Investment duration in days
        last_3_days_prices (str): String containing last 3 days' prices
    
    Returns:
        str: The sentiment analysis result provided by the LLM.
    """
    # Define a list of URLs for different Indian financial news sources related to the stock.
    urls = [
        f"https://finance.yahoo.com/quote/{stock_ticker}/news",
        f"https://www.marketwatch.com/investing/stock/{stock_ticker}",
        f"https://www.cnbc.com/quotes/{stock_ticker}"

        # Major Indian Financial News Sources
        # f"https://economictimes.indiatimes.com/markets/stocks/news/{stock_ticker}",
        # f"https://www.moneycontrol.com/india/stockpricequote/{stock_ticker}",
        # f"https://www.business-standard.com/markets/stocks/{stock_ticker}",
        
        # Additional Indian Financial News Sources
        # f"https://www.livemint.com/market/stock-market-news/{stock_ticker}",
        # f"https://www.financialexpress.com/market/{stock_ticker}",
        # f"https://www.ndtv.com/business/stock/{stock_ticker}",
        
        # Indian Stock Analysis Websites
        # f"https://www.screener.in/company/{stock_ticker}",
        # f"https://www.tickertape.in/stocks/{stock_ticker}",
        # f"https://www.5paisa.com/stock-market/{stock_ticker}",
        
        # Indian Financial Analysis Platforms
        # f"https://www.ambit.co/stock/{stock_ticker}",
        # f"https://www.angelone.in/stock/{stock_ticker}",
        # f"https://www.nseindia.com/get-quotes/equity?symbol={stock_ticker}"
    ]
    
    # Load documents from all defined URLs
    documents = load_documents_from_urls(urls)
    
    if not documents:
        return "No documents could be loaded from the provided sources."
    
    # Combine text content from all loaded documents
    combined_text = " ".join(doc.page_content for doc in documents)
    
    # Add price history to the combined text
    if last_3_days_prices:
        combined_text = f"Recent Price History:\n{last_3_days_prices}\n\nNews and Analysis:\n{combined_text}"
    
    # Use the sentiment analyzer from llm.py
    return sentiment_analyzer.analyze_sentiment(stock_ticker, combined_text, invest_duration)

if __name__ == "__main__":
    ticker = "RELIANCE"  # Example Indian stock ticker
    invest_duration = 30  # Example investment duration
    sentiment_result = perform_sentiment_analysis_multi(ticker, invest_duration)
    print("\n--- Sentiment Analysis Result ---")
    print(sentiment_result)
