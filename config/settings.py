import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Model Parameters
ARIMA_ORDER = (1, 1, 1)
SARIMA_ORDER = (1, 1, 1)
SARIMA_SEASONAL_ORDER = (1, 1, 1, 12)

# Analysis Parameters
TRAINING_SPLIT = 0.8
CONFIDENCE_THRESHOLD = 0.7

# File Paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
STOCK_DATA_FILE = os.path.join(DATA_DIR, 'stock_data.csv')
FORECAST_PLOT_FILE = 'forecast_plot.png'

# LLM Settings
LLM_MODEL = "deepseek-r1-distill-llama-70b"
LLM_TEMPERATURE = 0.0

# Analysis Settings
DEFAULT_TRAINING_PERIOD = 730  # 2 years in days
MAX_TRAINING_PERIOD = 3650     # 10 years in days
MIN_TRAINING_PERIOD = 30       # 1 month in days

# Technical Analysis Parameters
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
MA_PERIODS = [20, 50, 200]

# Risk Management
DEFAULT_STOP_LOSS = 0.05  # 5%
DEFAULT_TAKE_PROFIT = 0.15  # 15%
MAX_POSITION_SIZE = 0.1  # 10% of portfolio 