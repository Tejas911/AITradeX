# AITradeX - Advanced Stock Analysis System

A comprehensive stock analysis system that combines technical, fundamental, and predictive analysis with AI-powered insights.

## Features

- Technical Analysis
- Fundamental Analysis
- Sentiment Analysis
- Predictive Modeling (ARIMA, SARIMA, Prophet)
- AI-Powered Recommendations
- Interactive Visualization

## Project Structure

```
AITradeX/
├── data/               # Data storage and management
├── models/            # Predictive models
├── analysis/          # Analysis modules
├── utils/             # Utility functions
├── config/            # Configuration files
└── main.py            # Main application entry point
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/AITradeX.git
cd AITradeX
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file with your API keys:
```
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
```

## Usage

Run the main script:
```bash
python main.py
```

Follow the prompts to:
1. Enter stock ticker
2. Specify investment duration
3. Set training period

## Analysis Components

1. **Technical Analysis**
   - Moving averages
   - RSI
   - MACD
   - Volume analysis

2. **Fundamental Analysis**
   - Financial metrics
   - Valuation ratios
   - Growth indicators

3. **Sentiment Analysis**
   - News sentiment
   - Market sentiment
   - Social media sentiment

4. **Predictive Analysis**
   - ARIMA model
   - SARIMA model
   - Prophet model
   - Combined forecasts

## Output

The system generates:
- Detailed analysis reports
- Price predictions
- Risk assessments
- Investment recommendations
- Visualization plots

## License

This project is licensed under the MIT License - see the LICENSE file for details. 