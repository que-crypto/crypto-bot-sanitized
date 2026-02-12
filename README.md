# Crypto Sentiment Trading Bot

An educational cryptocurrency trading bot that analyzes market sentiment across multiple data sources and generates trading signals using machine learning and multi-source signal aggregation.

## ⚠️ Disclaimer

**This is an educational project for learning purposes only.**

- Not financial advice
- Use at your own risk
- Cryptocurrency trading carries significant risk
- Past performance does not guarantee future results
- Always test with paper trading (dry-run mode) before using real capital

## 📊 Overview

This bot combines sentiment analysis, on-chain data, and derivative market signals to generate informed trading decisions. It uses explicit weighting to balance different data sources and employs risk management strategies to protect capital.

### Key Features

- **Multi-Source Sentiment Analysis**
  - Twitter/X real-time social sentiment
  - Reddit community discussions
  - Professional news articles
  - Fine-tuned BERT model for crypto-specific sentiment

- **On-Chain Data Integration**
  - Whale wallet movements
  - Exchange inflows/outflows
  - Stablecoin flow tracking
  - Network activity metrics

- **Derivative Market Signals**
  - Funding rates (contrarian indicator)
  - Open interest changes
  - Liquidation data
  - Options flow (put/call ratios)

- **Explicit Weighting System**
  - Sentiment: 10-15% (noisiest, easiest to manipulate)
  - On-Chain: 50-60% (hardest to fake, represents actual capital movement)
  - Derivatives: 30-35% (forward-looking positioning)

- **Risk Management**
  - Kelly Criterion position sizing
  - Dynamic leverage based on signal confidence
  - Automatic stop-loss and take-profit
  - Kill switch at 15% drawdown
  - 5-minute cooldown between trades

## 🏗️ Architecture

```
Data Sources → BERT Sentiment → Multi-Source Detector → LLM Interpreter → Risk Engine → Exchange
     ↓              ↓                    ↓                      ↓              ↓           ↓
  Twitter        Scores            Explicit             Reasoning      Position      Binance
  Reddit         Messages          Weighting            Context         Sizing       Kraken
  News                             Composite                           Leverage      etc.
  On-Chain                         Score
  Derivatives
```

### Components

1. **signal_layer.py** - BERT sentiment classifier and sentiment tracking
2. **multi_source_signal_layer.py** - Weighted signal aggregation with configurable thresholds
3. **onchain_data_layer.py** - Blockchain data integration (whale movements, exchange flows)
4. **derivative_signals_layer.py** - Futures/options market data (funding, OI, liquidations)
5. **execution_layer.py** - Trade execution with risk management
6. **ingestion_layer.py** - Main event loop orchestrating all components

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- Ubuntu/Linux (recommended) or macOS/Windows
- 8GB RAM minimum (16GB recommended for BERT model)
- API keys for data sources (see Setup below)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/crypto-sentiment-bot.git
   cd crypto-sentiment-bot
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

5. **Run in dry-run mode (recommended for testing)**
   ```bash
   python ingestion_layer.py
   ```

### API Keys Required

See `.env.example` for the complete list. At minimum, you need:

- **ANTHROPIC_API_KEY** - For LLM signal interpretation (Claude)
- **At least one sentiment source:**
  - TWITTER_BEARER_TOKEN, or
  - REDDIT_CLIENT_ID + REDDIT_CLIENT_SECRET, or
  - NEWSAPI_KEY
- **EXCHANGE_API_KEY + EXCHANGE_API_SECRET** - For trading (Binance, Kraken, etc.)

Optional but recommended:
- On-chain data: CRYPTOQUANT_API_KEY, WHALE_ALERT_API_KEY
- Derivatives are free (Binance public API, Deribit public API)

For detailed API setup instructions, see [API Setup Guide](docs/API_SETUP_GUIDE.md) (if available).

## ⚙️ Configuration

### Adjusting Signal Weights

Edit the weights in `signal_layer.py`:

```python
pipeline = SentimentPipeline(
    asset="BTC",
    sentiment_weight=0.15,    # 15% - adjust to your preference
    onchain_weight=0.55,      # 55%
    derivative_weight=0.30,   # 30%
)
```

Weights must sum to 1.0.

### Trigger Thresholds

Modify thresholds in `multi_source_signal_layer.py`:

```python
thresholds=TriggerThresholds(
    sentiment_delta=0.15,       # Sentiment shift threshold
    onchain_extreme=0.75,       # On-chain triggers independently above this
    derivative_extreme=0.70,    # Derivative triggers independently above this
    composite_score=0.50,       # Minimum composite score to trigger
)
```

### Risk Parameters

Adjust risk settings in `execution_layer.py`:

```python
risk_config = {
    "max_position_pct": 0.05,   # Risk 5% of capital per trade
    "stop_loss_pct": 0.03,      # 3% stop loss
    "take_profit_pct": 0.06,    # 6% take profit
    "max_leverage": 10,         # Maximum leverage
    "kill_switch_drawdown": 0.15,  # Stop all trading at 15% drawdown
}
```

## 📈 Usage Examples

### Basic Usage

```python
from signal_layer import SentimentPipeline, RawMessage

# Initialize pipeline
pipeline = SentimentPipeline(asset="BTC")

# Process a message
message = RawMessage(
    text="Bitcoin just broke above $50k with massive volume!",
    source="twitter",
    timestamp=time.time(),
)

signal = pipeline.process(message)
if signal:
    print(f"Signal: {signal.direction} | Confidence: {signal.confidence}")
```

### Running the Full Bot

```bash
# Dry-run mode (no real trades)
python ingestion_layer.py

# Live trading (after thorough testing!)
python ingestion_layer.py --live
```

### Monitoring

```bash
# View live logs
tail -f logs/bot.log

# Check bot status (if running as systemd service)
sudo systemctl status crypto-bot
```

## 🧪 Testing

The bot includes dry-run mode by default. Always test for at least 1-2 weeks before live trading.

**Testing checklist:**
- ✅ Run in dry-run mode for 2+ weeks
- ✅ Verify signals fire appropriately
- ✅ Check LLM reasoning makes sense
- ✅ Monitor for false positives
- ✅ Validate composite score calculations
- ✅ Test with small capital first when going live

## 🔒 Security Best Practices

1. **Never commit your `.env` file to Git**
   - It's in `.gitignore` by default
   - Always use `.env.example` as template

2. **Secure your API keys**
   - Use IP whitelist on exchange APIs
   - Disable withdrawals on exchange API keys
   - Store `.env` with restricted permissions: `chmod 600 .env`

3. **Start with dry-run**
   - Keep `DRY_RUN=true` until thoroughly tested
   - Test all features before enabling live trading

4. **Monitor actively**
   - Set up alerts for unexpected behavior
   - Check logs regularly
   - Use kill switch (15% drawdown stops all trading)

## 📚 Documentation

- **Installation Guide** - See above
- **API Setup Guide** - Detailed instructions for obtaining all required API keys
- **Migration Guide** - How to upgrade from older versions
- **Ubuntu Server Setup** - Instructions for running on a dedicated server

## 🛠️ Development

### Project Structure

```
crypto-sentiment-bot/
├── signal_layer.py                 # BERT sentiment + sentiment tracking
├── multi_source_signal_layer.py    # Weighted signal aggregation
├── onchain_data_layer.py           # On-chain data sources
├── derivative_signals_layer.py     # Derivative market data
├── execution_layer.py              # Trade execution + risk management
├── ingestion_layer.py              # Main event loop
├── crypto_bert_finetuner.py        # Optional: Fine-tune BERT for crypto
├── .env.example                    # Environment variables template
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

### Fine-Tuning BERT

For better sentiment accuracy, you can fine-tune BERT on crypto-specific data:

```bash
python crypto_bert_finetuner.py --run full
```

This trains a custom model that understands crypto slang, whale activity terminology, and domain-specific sentiment patterns.

### Backtesting & Optimization

The `WeightOptimizer` class allows you to find optimal weights via backtesting:

```python
from multi_source_signal_layer import WeightOptimizer

optimizer = WeightOptimizer()
best = optimizer.grid_search(historical_signals)
print(f"Optimal weights: {best['weights']}")
```

## 🤝 Contributing

This is a personal educational project, but feedback and suggestions are welcome via issues.

## 📄 License

This project is for educational and personal use only. Not licensed for commercial use.

## 🙏 Acknowledgments

- **Anthropic** - Claude API for LLM interpretation
- **Hugging Face** - Transformers library for BERT
- **ccxt** - Unified exchange API
- Various data providers (Twitter, Reddit, NewsAPI, Glassnode, CryptoQuant, etc.)

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Check existing documentation
- Review logs for error messages

## ⚖️ Legal

By using this software, you acknowledge that:
- Cryptocurrency trading is risky
- You are responsible for your own trading decisions
- The author is not liable for any financial losses
- This is not financial advice
- Always comply with local regulations regarding automated trading

---

**Built for educational purposes. Trade responsibly. 🚀**
