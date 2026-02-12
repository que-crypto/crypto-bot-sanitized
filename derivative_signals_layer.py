"""
Derivative Market Signals Layer
=================================
Tracks futures and options market data that provides forward-looking
signals about where the market is headed:

    - Funding rates (extreme values predict reversals)
    - Open interest (new money entering vs positions closing)
    - Liquidation levels (where leveraged traders are vulnerable)
    - Options flow (put/call ratio, max pain levels)

Derivative markets are forward-looking — traders put capital at risk
based on their expectations. This makes derivative data a leading
indicator compared to spot price or even sentiment.

Data Sources:
    - Binance Futures API (funding, OI, liquidations) — free, real-time
    - Deribit API (options flow, IV) — free for market data
    - Coinglass API (aggregated derivative metrics) — free tier available

This layer outputs DerivativeSignal objects that augment the trading
decision process — the LLM considers derivative positioning when
interpreting sentiment shifts.

Dependencies:
    pip install requests python-dotenv ccxt

Setup (.env file):
    COINGLASS_API_KEY=your_key  # optional, for aggregated data
    # Binance/Deribit APIs are public (no auth needed for market data)

Usage:
    # Standalone
    python derivative_signals_layer.py

    # Integrated with signal pipeline
    from derivative_signals_layer import DerivativeMonitor
    deriv = DerivativeMonitor(asset="BTC")
    signal = deriv.get_latest_signal()
"""

import os
import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from dotenv import load_dotenv
import requests

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG = {
    # API keys
    "coinglass_key": os.getenv("COINGLASS_API_KEY"),  # Optional aggregator

    # Asset
    "asset": "BTC",

    # Thresholds for signal generation
    "funding_rate_extreme_long":  0.01,   # 1%+ funding = extremely bullish positioning
    "funding_rate_extreme_short": -0.01,  # -1% funding = extremely bearish positioning
    "oi_change_threshold":        0.15,   # 15% change in OI is significant
    "liquidation_cluster_threshold": 100_000_000,  # $100M in liquidations clustered

    # Poll interval
    "poll_interval": 300,  # 5 minutes
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class DerivativeSignalType(Enum):
    """Types of derivative market signals."""
    FUNDING_RATE_EXTREME_LONG   = "funding_extreme_long"   # Bearish: overheated longs, reversal likely
    FUNDING_RATE_EXTREME_SHORT  = "funding_extreme_short"  # Bullish: overheated shorts, reversal likely
    OPEN_INTEREST_SURGE         = "oi_surge"               # Neutral: new leverage entering
    OPEN_INTEREST_DECLINE       = "oi_decline"             # Neutral: leverage unwinding
    LIQUIDATION_CASCADE_LONG    = "liquidation_cascade_long"   # Bearish: longs getting liquidated
    LIQUIDATION_CASCADE_SHORT   = "liquidation_cascade_short"  # Bullish: shorts getting liquidated
    OPTIONS_SKEW_BULLISH        = "options_skew_bullish"   # Bullish: call volume > put volume
    OPTIONS_SKEW_BEARISH        = "options_skew_bearish"   # Bearish: put volume > call volume


@dataclass
class DerivativeSignal:
    """
    Represents a derivative market observation.
    Augments sentiment and on-chain data with forward-looking positioning.
    """
    signal_type: DerivativeSignalType
    magnitude: float         # Significance (0.0 to 1.0)
    raw_value: float         # Actual measured value
    metric_name: str         # Human-readable description
    timestamp: float
    confidence: float = 0.8  # Derivative data is strong but not as ironclad as on-chain
    source: str = "derivative"

    def to_dict(self) -> dict:
        return {
            "signal_type":  self.signal_type.value,
            "magnitude":    self.magnitude,
            "raw_value":    self.raw_value,
            "metric_name":  self.metric_name,
            "timestamp":    self.timestamp,
            "confidence":   self.confidence,
            "source":       self.source,
        }


@dataclass
class DerivativeContext:
    """
    Aggregated derivative market state.
    """
    signals: list[DerivativeSignal] = field(default_factory=list)
    net_sentiment: float = 0.0   # -1.0 (bearish) to +1.0 (bullish)
    timestamp: float = field(default_factory=time.time)

    def add_signal(self, signal: DerivativeSignal):
        self.signals.append(signal)
        self._recalculate_net_sentiment()

    def _recalculate_net_sentiment(self):
        """
        Compute overall derivative market sentiment.
        
        Key principle: extreme positioning in one direction typically precedes
        a reversal in the opposite direction. Funding rate extremes are contrarian.
        """
        if not self.signals:
            self.net_sentiment = 0.0
            return

        score = 0.0
        for sig in self.signals:
            # Funding rate extremes are CONTRARIAN
            if sig.signal_type == DerivativeSignalType.FUNDING_RATE_EXTREME_LONG:
                direction = -1.0  # Overheated longs → expect reversal down
            elif sig.signal_type == DerivativeSignalType.FUNDING_RATE_EXTREME_SHORT:
                direction = 1.0   # Overheated shorts → expect reversal up
            # Liquidations indicate which side is in trouble
            elif sig.signal_type == DerivativeSignalType.LIQUIDATION_CASCADE_LONG:
                direction = -1.0  # Longs liquidating → bearish
            elif sig.signal_type == DerivativeSignalType.LIQUIDATION_CASCADE_SHORT:
                direction = 1.0   # Shorts liquidating → bullish
            # Options flow is directional
            elif sig.signal_type == DerivativeSignalType.OPTIONS_SKEW_BULLISH:
                direction = 1.0
            elif sig.signal_type == DerivativeSignalType.OPTIONS_SKEW_BEARISH:
                direction = -1.0
            else:
                direction = 0.0  # OI changes are neutral in isolation

            score += direction * sig.magnitude * sig.confidence

        self.net_sentiment = max(-1.0, min(1.0, score / len(self.signals) if self.signals else 0))


# ---------------------------------------------------------------------------
# Data Source: Binance Futures (funding rate, OI, liquidations)
# ---------------------------------------------------------------------------

class BinanceFuturesSource:
    """
    Tracks perpetual futures data from Binance.
    All endpoints are public (no API key needed).

    API: https://binance-docs.github.io/apidocs/futures/en/
    """

    def __init__(self, asset: str = "BTC"):
        self.asset = asset.upper()
        self.symbol = f"{self.asset}USDT"  # Perpetual symbol on Binance
        self.base_url = "https://fapi.binance.com"
        self.prev_oi = None  # Track OI for delta calculation
        logger.info("BinanceFuturesSource initialized.")

    def fetch_funding_rate(self) -> Optional[DerivativeSignal]:
        """
        Fetch the current funding rate.
        Extreme funding rates (> 1% or < -1%) signal overheated positioning.
        """
        try:
            response = requests.get(
                f"{self.base_url}/fapi/v1/premiumIndex",
                params={"symbol": self.symbol},
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

            funding_rate = float(data.get("lastFundingRate", 0))

            # Funding rates are usually small (0.0001 = 0.01%)
            # Binance returns them as decimals, so 0.01 = 1%
            if abs(funding_rate) < CONFIG["funding_rate_extreme_long"]:
                return None  # Not extreme

            if funding_rate > CONFIG["funding_rate_extreme_long"]:
                # Very positive funding = longs paying shorts = overheated longs
                return DerivativeSignal(
                    signal_type  = DerivativeSignalType.FUNDING_RATE_EXTREME_LONG,
                    magnitude    = min(1.0, abs(funding_rate) / (CONFIG["funding_rate_extreme_long"] * 2)),
                    raw_value    = funding_rate * 100,  # Convert to percentage
                    metric_name  = f"Funding rate: {funding_rate * 100:.3f}% (extreme long)",
                    timestamp    = time.time(),
                    source       = "binance_futures",
                )
            else:
                # Very negative funding = shorts paying longs = overheated shorts
                return DerivativeSignal(
                    signal_type  = DerivativeSignalType.FUNDING_RATE_EXTREME_SHORT,
                    magnitude    = min(1.0, abs(funding_rate) / abs(CONFIG["funding_rate_extreme_short"] * 2)),
                    raw_value    = funding_rate * 100,
                    metric_name  = f"Funding rate: {funding_rate * 100:.3f}% (extreme short)",
                    timestamp    = time.time(),
                    source       = "binance_futures",
                )

        except Exception as e:
            logger.warning(f"Binance funding rate fetch failed: {e}")
            return None

    def fetch_open_interest(self) -> Optional[DerivativeSignal]:
        """
        Fetch open interest and detect significant changes.
        Rising OI = new leverage entering.
        Falling OI = positions being closed.
        """
        try:
            response = requests.get(
                f"{self.base_url}/fapi/v1/openInterest",
                params={"symbol": self.symbol},
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

            current_oi = float(data.get("openInterest", 0))

            if self.prev_oi is None:
                self.prev_oi = current_oi
                return None  # Need at least one historical point

            oi_change_pct = (current_oi - self.prev_oi) / self.prev_oi if self.prev_oi > 0 else 0
            self.prev_oi = current_oi

            if abs(oi_change_pct) < CONFIG["oi_change_threshold"]:
                return None

            if oi_change_pct > 0:
                return DerivativeSignal(
                    signal_type  = DerivativeSignalType.OPEN_INTEREST_SURGE,
                    magnitude    = min(1.0, abs(oi_change_pct) / (CONFIG["oi_change_threshold"] * 2)),
                    raw_value    = oi_change_pct * 100,
                    metric_name  = f"Open interest surged {oi_change_pct * 100:+.1f}%",
                    timestamp    = time.time(),
                    source       = "binance_futures",
                )
            else:
                return DerivativeSignal(
                    signal_type  = DerivativeSignalType.OPEN_INTEREST_DECLINE,
                    magnitude    = min(1.0, abs(oi_change_pct) / (CONFIG["oi_change_threshold"] * 2)),
                    raw_value    = oi_change_pct * 100,
                    metric_name  = f"Open interest declined {oi_change_pct * 100:.1f}%",
                    timestamp    = time.time(),
                    source       = "binance_futures",
                )

        except Exception as e:
            logger.warning(f"Binance OI fetch failed: {e}")
            return None

    def fetch_recent_liquidations(self) -> list[DerivativeSignal]:
        """
        Fetch recent liquidation orders.
        Large liquidation cascades indicate one side getting wiped out.
        
        Note: Binance doesn't expose historical liquidations via public API easily.
        This is a simplified implementation — production would use WebSocket
        streams or a third-party aggregator like Coinglass.
        """
        # Binance's public API doesn't provide liquidation history directly.
        # You'd typically use:
        #   1. WebSocket stream: wss://fstream.binance.com/ws/!forceOrder@arr
        #   2. Third-party aggregator (Coinglass, CryptoQuant)
        
        # For this implementation, we'll return empty and note it as a TODO.
        # In production, integrate a WebSocket listener or use Coinglass API.
        return []


# ---------------------------------------------------------------------------
# Data Source: Deribit (options flow)
# ---------------------------------------------------------------------------

class DeribitOptionsSource:
    """
    Tracks options market data from Deribit (largest BTC/ETH options exchange).
    Free for public market data.

    API: https://docs.deribit.com/
    """

    def __init__(self, asset: str = "BTC"):
        self.asset = asset.upper()
        self.base_url = "https://www.deribit.com/api/v2/public"
        logger.info("DeribitOptionsSource initialized.")

    def fetch_put_call_ratio(self) -> Optional[DerivativeSignal]:
        """
        Fetch the put/call open interest ratio.
        High put/call ratio = bearish positioning.
        Low put/call ratio = bullish positioning.
        """
        try:
            # Get open interest for all BTC options
            response = requests.get(
                f"{self.base_url}/get_book_summary_by_currency",
                params={"currency": self.asset, "kind": "option"},
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

            if data.get("result") is None:
                return None

            put_oi = 0.0
            call_oi = 0.0

            for instrument in data["result"]:
                # instrument_name format: BTC-31JAN25-50000-P (Put) or -C (Call)
                if instrument["instrument_name"].endswith("-P"):
                    put_oi += float(instrument.get("open_interest", 0))
                elif instrument["instrument_name"].endswith("-C"):
                    call_oi += float(instrument.get("open_interest", 0))

            if call_oi == 0:
                return None

            put_call_ratio = put_oi / call_oi

            # Typical neutral ratio is around 0.7-1.0
            # > 1.2 = bearish skew
            # < 0.6 = bullish skew
            if put_call_ratio > 1.2:
                return DerivativeSignal(
                    signal_type  = DerivativeSignalType.OPTIONS_SKEW_BEARISH,
                    magnitude    = min(1.0, (put_call_ratio - 1.0) / 0.5),
                    raw_value    = put_call_ratio,
                    metric_name  = f"Put/Call ratio: {put_call_ratio:.2f} (bearish skew)",
                    timestamp    = time.time(),
                    source       = "deribit_options",
                )
            elif put_call_ratio < 0.6:
                return DerivativeSignal(
                    signal_type  = DerivativeSignalType.OPTIONS_SKEW_BULLISH,
                    magnitude    = min(1.0, (1.0 - put_call_ratio) / 0.4),
                    raw_value    = put_call_ratio,
                    metric_name  = f"Put/Call ratio: {put_call_ratio:.2f} (bullish skew)",
                    timestamp    = time.time(),
                    source       = "deribit_options",
                )

            return None  # Ratio in neutral range

        except Exception as e:
            logger.warning(f"Deribit options fetch failed: {e}")
            return None


# ---------------------------------------------------------------------------
# Data Source: Coinglass (aggregated derivative metrics)
# ---------------------------------------------------------------------------

class CoinglassSource:
    """
    Aggregates derivative data across multiple exchanges.
    Provides liquidation heatmaps, aggregated funding, etc.

    API: https://coinglass.github.io/API-Reference/
    Free tier available with limits.
    """

    def __init__(self, api_key: str, asset: str = "BTC"):
        if not api_key:
            raise ValueError("COINGLASS_API_KEY not set.")
        self.api_key = api_key
        self.asset = asset.upper()
        self.base_url = "https://open-api.coinglass.com/public/v2"
        logger.info("CoinglassSource initialized.")

    def fetch_liquidation_heatmap(self) -> list[DerivativeSignal]:
        """
        Fetch recent liquidations across all exchanges.
        Identifies cascades (large clustered liquidations).
        """
        try:
            response = requests.get(
                f"{self.base_url}/indicator/liquidation_history",
                headers={"coinglassSecret": self.api_key},
                params={"symbol": f"{self.asset}USDT", "time_type": "h1"},  # Last 1h
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

            if data.get("code") != "0" or not data.get("data"):
                return []

            signals = []
            for entry in data["data"]:
                long_liq = float(entry.get("longLiquidation", 0))
                short_liq = float(entry.get("shortLiquidation", 0))

                if long_liq > CONFIG["liquidation_cluster_threshold"]:
                    signals.append(DerivativeSignal(
                        signal_type  = DerivativeSignalType.LIQUIDATION_CASCADE_LONG,
                        magnitude    = min(1.0, long_liq / (CONFIG["liquidation_cluster_threshold"] * 2)),
                        raw_value    = long_liq,
                        metric_name  = f"Long liquidations: ${long_liq / 1e6:.1f}M",
                        timestamp    = entry.get("createTime", time.time()) / 1000,
                        source       = "coinglass",
                    ))

                if short_liq > CONFIG["liquidation_cluster_threshold"]:
                    signals.append(DerivativeSignal(
                        signal_type  = DerivativeSignalType.LIQUIDATION_CASCADE_SHORT,
                        magnitude    = min(1.0, short_liq / (CONFIG["liquidation_cluster_threshold"] * 2)),
                        raw_value    = short_liq,
                        metric_name  = f"Short liquidations: ${short_liq / 1e6:.1f}M",
                        timestamp    = entry.get("createTime", time.time()) / 1000,
                        source       = "coinglass",
                    ))

            logger.info(f"Coinglass: found {len(signals)} liquidation signals.")
            return signals

        except Exception as e:
            logger.warning(f"Coinglass fetch failed: {e}")
            return []


# ---------------------------------------------------------------------------
# Orchestrator: DerivativeMonitor
# ---------------------------------------------------------------------------

class DerivativeMonitor:
    """
    Central coordinator for all derivative market data sources.
    Polls each source and aggregates into a unified view.

    Usage:
        monitor = DerivativeMonitor(asset="BTC")
        context = monitor.get_latest_context()
        print(context.net_sentiment)
    """

    def __init__(self, asset: str = "BTC"):
        self.asset = asset
        self.sources = []
        self.last_context = DerivativeContext()

        self._initialize_sources()

        if not self.sources:
            logger.warning("No derivative data sources available. Using Binance public API as fallback.")
            # Binance is always available (no API key needed)
            self.sources.append(("binance_futures", BinanceFuturesSource(asset)))
        else:
            logger.info(f"DerivativeMonitor initialized with {len(self.sources)} sources.")

    def _initialize_sources(self):
        """Initialize available sources."""
        # Binance Futures (always available)
        self.sources.append(("binance_futures", BinanceFuturesSource(self.asset)))

        # Deribit Options (always available for BTC/ETH)
        if self.asset in ("BTC", "ETH"):
            self.sources.append(("deribit_options", DeribitOptionsSource(self.asset)))

        # Coinglass (requires API key)
        if CONFIG["coinglass_key"]:
            try:
                self.sources.append(("coinglass", CoinglassSource(CONFIG["coinglass_key"], self.asset)))
            except Exception as e:
                logger.warning(f"Coinglass init failed: {e}")

    def collect_signals(self) -> DerivativeContext:
        """
        Poll all sources and aggregate into a DerivativeContext.
        Call this periodically from your main loop.
        """
        context = DerivativeContext()

        for name, source in self.sources:
            try:
                if name == "binance_futures":
                    # Funding rate
                    funding_sig = source.fetch_funding_rate()
                    if funding_sig:
                        context.add_signal(funding_sig)
                    # Open interest
                    oi_sig = source.fetch_open_interest()
                    if oi_sig:
                        context.add_signal(oi_sig)
                    # Liquidations (placeholder — would use WebSocket or Coinglass)
                    # liq_sigs = source.fetch_recent_liquidations()
                    # for sig in liq_sigs:
                    #     context.add_signal(sig)

                elif name == "deribit_options":
                    pc_sig = source.fetch_put_call_ratio()
                    if pc_sig:
                        context.add_signal(pc_sig)

                elif name == "coinglass":
                    liq_sigs = source.fetch_liquidation_heatmap()
                    for sig in liq_sigs:
                        context.add_signal(sig)

            except Exception as e:
                logger.error(f"Error collecting from {name}: {e}")

        self.last_context = context
        logger.info(f"Collected {len(context.signals)} derivative signals | net_sentiment: {context.net_sentiment:+.2f}")
        return context

    def get_latest_context(self) -> DerivativeContext:
        """Return the most recently collected context (cached)."""
        return self.last_context


# ---------------------------------------------------------------------------
# Integration with signal_layer.py
# ---------------------------------------------------------------------------

def augment_llm_prompt_with_derivatives(prompt: str, deriv_context: DerivativeContext) -> str:
    """
    Helper to augment the LLM prompt with derivative market data.

    Usage in signal_layer.py's LLMInterpreter.interpret():
        from derivative_signals_layer import augment_llm_prompt_with_derivatives
        deriv = deriv_monitor.get_latest_context()
        augmented = augment_llm_prompt_with_derivatives(original_prompt, deriv)
    """
    if not deriv_context.signals:
        return prompt

    deriv_summary = f"\n\nDERIVATIVE MARKET CONTEXT (forward-looking positioning):\n"
    deriv_summary += f"- Net derivative sentiment: {deriv_context.net_sentiment:+.2f}\n"
    deriv_summary += f"- Key signals:\n"

    for sig in deriv_context.signals[:5]:
        deriv_summary += f"  • {sig.metric_name} (magnitude: {sig.magnitude:.2f})\n"

    deriv_summary += "\nConsider whether derivative positioning confirms or contradicts the sentiment shift."

    return prompt + deriv_summary


# ---------------------------------------------------------------------------
# Demo / standalone mode
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Derivative Market Signals Monitor")
    parser.add_argument("--asset", default="BTC", help="Asset to monitor (BTC or ETH)")
    parser.add_argument("--interval", type=int, default=300, help="Poll interval in seconds")
    args = parser.parse_args()

    monitor = DerivativeMonitor(asset=args.asset)

    print(f"\n{'=' * 70}")
    print(f"  DERIVATIVE MARKET MONITOR — {args.asset}")
    print(f"  Poll interval: {args.interval}s")
    print(f"  Active sources: {len(monitor.sources)}")
    print(f"{'=' * 70}\n")

    try:
        while True:
            context = monitor.collect_signals()

            print(f"\n--- Derivative Update ---")
            print(f"Timestamp:       {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(context.timestamp))}")
            print(f"Net Sentiment:   {context.net_sentiment:+.3f}")
            print(f"Signals:         {len(context.signals)}")

            if context.signals:
                print("\nTop signals:")
                for i, sig in enumerate(context.signals[:5], 1):
                    print(f"  {i}. [{sig.signal_type.value:25s}] {sig.metric_name}")

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\nMonitor stopped by user.")
