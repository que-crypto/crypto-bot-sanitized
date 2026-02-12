"""
On-Chain Data Integration Layer
=================================
Tracks real blockchain activity that can't be faked or manipulated:
    - Exchange inflows/outflows (whale movements signaling intent)
    - Whale wallet activity (large holders becoming active)
    - Stablecoin flows (capital positioning ahead of buys/sells)
    - Network activity metrics (active addresses, transaction volume)

These signals are fundamentally stronger than sentiment because they
represent *actual capital movement*, not just people talking.

Data Sources:
    - Glassnode API (on-chain metrics) — premium, most comprehensive
    - CryptoQuant API (exchange flows) — good free tier
    - Whale Alert API (large transaction tracking) — free with limits
    - Etherscan / Blockchain.com APIs (raw blockchain data) — free

This layer outputs OnChainSignal objects that augment the sentiment
pipeline — the LLM interpreter considers both sentiment AND on-chain
context when making trading decisions.

Dependencies:
    pip install requests python-dotenv

Setup (.env file):
    GLASSNODE_API_KEY=your_key       # optional, premium service
    CRYPTOQUANT_API_KEY=your_key     # recommended, good free tier
    WHALE_ALERT_API_KEY=your_key     # optional, free tier available
    ETHERSCAN_API_KEY=your_key       # optional, for ETH on-chain data

Usage:
    # Standalone
    python onchain_data_layer.py

    # Integrated with sentiment pipeline
    from onchain_data_layer import OnChainMonitor
    from signal_layer import SentimentPipeline

    onchain = OnChainMonitor()
    sentiment = SentimentPipeline()

    # Get current on-chain context
    onchain_signal = onchain.get_latest_signal()

    # Pass to LLM for interpretation (modify signal_layer.py to accept this)
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
    # API keys (all optional — system works with whatever is available)
    "glassnode_key":    os.getenv("GLASSNODE_API_KEY"),
    "cryptoquant_key":  os.getenv("CRYPTOQUANT_API_KEY"),
    "whale_alert_key":  os.getenv("WHALE_ALERT_API_KEY"),
    "etherscan_key":    os.getenv("ETHERSCAN_API_KEY"),

    # Asset to monitor
    "asset":            "BTC",  # BTC, ETH supported

    # Thresholds for signal generation
    "whale_threshold_btc":  100,    # Transactions above this are "whale" moves
    "whale_threshold_eth":  1000,
    "exchange_flow_threshold": 5000,  # Net flow (in/out) threshold for significance
    "stablecoin_flow_threshold": 100_000_000,  # $100M USDT movement is significant

    # Poll intervals (seconds)
    "poll_interval_whale":      300,   # 5 min
    "poll_interval_exchange":   600,   # 10 min
    "poll_interval_network":    3600,  # 1 hour (less time-sensitive)
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class OnChainSignalType(Enum):
    """Types of on-chain signals that can be generated."""
    WHALE_ACCUMULATION   = "whale_accumulation"    # Bullish: large wallets buying
    WHALE_DISTRIBUTION   = "whale_distribution"    # Bearish: large wallets selling
    EXCHANGE_INFLOW      = "exchange_inflow"       # Bearish: coins moving to exchanges (sell setup)
    EXCHANGE_OUTFLOW     = "exchange_outflow"      # Bullish: coins leaving exchanges (accumulation)
    STABLECOIN_INFLOW    = "stablecoin_inflow"     # Bullish: capital entering, buying power up
    STABLECOIN_OUTFLOW   = "stablecoin_outflow"    # Bearish: capital leaving
    NETWORK_SURGE        = "network_surge"         # Neutral-bullish: increased activity
    NETWORK_DECLINE      = "network_decline"       # Neutral-bearish: decreased activity


@dataclass
class OnChainSignal:
    """
    Represents a single on-chain observation.
    This augments sentiment signals with hard, unfakeable data.
    """
    signal_type: OnChainSignalType
    magnitude: float         # How significant (0.0 to 1.0, normalized)
    raw_value: float         # Actual measured value (e.g. 5000 BTC, $200M USDT)
    metric_name: str         # Human-readable description
    timestamp: float
    confidence: float = 1.0  # On-chain data is high-confidence by default
    source: str = "onchain"  # Where this came from (e.g. "whale_alert", "glassnode")

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
class OnChainContext:
    """
    Aggregated on-chain state at a point in time.
    Multiple signals rolled up into a single view.
    """
    signals: list[OnChainSignal] = field(default_factory=list)
    net_sentiment: float = 0.0   # Aggregate: -1.0 (bearish) to +1.0 (bullish)
    timestamp: float = field(default_factory=time.time)

    def add_signal(self, signal: OnChainSignal):
        self.signals.append(signal)
        self._recalculate_net_sentiment()

    def _recalculate_net_sentiment(self):
        """
        Compute an overall on-chain sentiment score.
        Bullish signals (outflows, accumulation, stablecoin inflows) → positive
        Bearish signals (inflows, distribution, stablecoin outflows) → negative
        """
        if not self.signals:
            self.net_sentiment = 0.0
            return

        score = 0.0
        for sig in self.signals:
            direction = 1.0 if "accumulation" in sig.signal_type.value or "outflow" in sig.signal_type.value else -1.0
            # Exception: stablecoin_inflow is bullish (buying power), stablecoin_outflow is bearish
            if sig.signal_type == OnChainSignalType.STABLECOIN_INFLOW:
                direction = 1.0
            elif sig.signal_type == OnChainSignalType.STABLECOIN_OUTFLOW:
                direction = -1.0

            score += direction * sig.magnitude * sig.confidence

        # Normalize to [-1, 1]
        self.net_sentiment = max(-1.0, min(1.0, score / len(self.signals)))


# ---------------------------------------------------------------------------
# Data Source: Whale Alert (free tier — tracks large transactions)
# ---------------------------------------------------------------------------

class WhaleAlertSource:
    """
    Monitors large crypto transactions in real-time.
    Free tier: 10 requests/minute, tracks tx > threshold.

    API: https://docs.whale-alert.io/
    """

    def __init__(self, api_key: str, asset: str = "BTC"):
        if not api_key:
            raise ValueError("WHALE_ALERT_API_KEY not set.")
        self.api_key = api_key
        self.asset = asset.lower()
        self.base_url = "https://api.whale-alert.io/v1"
        self.last_check = time.time() - 3600  # Start 1hr back to get recent history
        logger.info("WhaleAlertSource initialized.")

    def fetch_recent_transactions(self) -> list[OnChainSignal]:
        """
        Fetch large transactions from the last poll interval.
        Returns OnChainSignal objects for whale movements.
        """
        now = int(time.time())
        start = int(self.last_check)

        try:
            response = requests.get(
                f"{self.base_url}/transactions",
                params={
                    "api_key":   self.api_key,
                    "start":     start,
                    "end":       now,
                    "currency":  self.asset,
                    "min_value": CONFIG[f"whale_threshold_{self.asset.lower()}"] * 1_000_000,  # API uses USD value
                },
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

            signals = []
            for tx in data.get("transactions", []):
                # Determine if this is exchange-related
                from_type = tx.get("from", {}).get("owner_type")
                to_type   = tx.get("to", {}).get("owner_type")

                amount = float(tx.get("amount", 0))
                if amount < CONFIG[f"whale_threshold_{self.asset.lower()}"]:
                    continue

                # Classify the movement
                if to_type == "exchange":
                    # Coins moving TO exchange → potential sell setup
                    signals.append(OnChainSignal(
                        signal_type  = OnChainSignalType.EXCHANGE_INFLOW,
                        magnitude    = min(1.0, amount / (CONFIG[f"whale_threshold_{self.asset.lower()}"] * 10)),
                        raw_value    = amount,
                        metric_name  = f"{amount:.2f} {self.asset.upper()} moved to exchange",
                        timestamp    = tx.get("timestamp", now),
                        source       = "whale_alert",
                    ))
                elif from_type == "exchange":
                    # Coins leaving exchange → accumulation
                    signals.append(OnChainSignal(
                        signal_type  = OnChainSignalType.EXCHANGE_OUTFLOW,
                        magnitude    = min(1.0, amount / (CONFIG[f"whale_threshold_{self.asset.lower()}"] * 10)),
                        raw_value    = amount,
                        metric_name  = f"{amount:.2f} {self.asset.upper()} withdrawn from exchange",
                        timestamp    = tx.get("timestamp", now),
                        source       = "whale_alert",
                    ))
                elif from_type == "unknown" and to_type == "unknown":
                    # Whale-to-whale transfer (accumulation by large holder)
                    signals.append(OnChainSignal(
                        signal_type  = OnChainSignalType.WHALE_ACCUMULATION,
                        magnitude    = min(1.0, amount / (CONFIG[f"whale_threshold_{self.asset.lower()}"] * 20)),
                        raw_value    = amount,
                        metric_name  = f"{amount:.2f} {self.asset.upper()} whale transfer",
                        timestamp    = tx.get("timestamp", now),
                        source       = "whale_alert",
                    ))

            self.last_check = now
            logger.info(f"WhaleAlert: fetched {len(signals)} signals from {len(data.get('transactions', []))} transactions.")
            return signals

        except Exception as e:
            logger.warning(f"WhaleAlert fetch failed: {e}")
            return []


# ---------------------------------------------------------------------------
# Data Source: CryptoQuant (exchange flow data)
# ---------------------------------------------------------------------------

class CryptoQuantSource:
    """
    Monitors exchange reserves, inflows, and outflows.
    Free tier: limited requests/day, still very usable.

    API: https://docs.cryptoquant.com/
    """

    def __init__(self, api_key: str, asset: str = "BTC"):
        if not api_key:
            raise ValueError("CRYPTOQUANT_API_KEY not set.")
        self.api_key = api_key
        self.asset = asset.upper()
        self.base_url = "https://api.cryptoquant.com/v1"
        logger.info("CryptoQuantSource initialized.")

    def fetch_exchange_flows(self) -> list[OnChainSignal]:
        """
        Fetch net exchange flow (inflow - outflow).
        Positive net = coins flowing IN (bearish).
        Negative net = coins flowing OUT (bullish).
        """
        signals = []

        try:
            # Get exchange inflow
            inflow_response = requests.get(
                f"{self.base_url}/{self.asset.lower()}/exchange-flows/inflow",
                headers={"Authorization": f"Bearer {self.api_key}"},
                params={"window": "day", "limit": 1},  # Last 24h aggregated
                timeout=10,
            )
            inflow_response.raise_for_status()
            inflow_data = inflow_response.json()
            inflow = float(inflow_data.get("result", {}).get("data", [{}])[0].get("value", 0))

            # Get exchange outflow
            outflow_response = requests.get(
                f"{self.base_url}/{self.asset.lower()}/exchange-flows/outflow",
                headers={"Authorization": f"Bearer {self.api_key}"},
                params={"window": "day", "limit": 1},
                timeout=10,
            )
            outflow_response.raise_for_status()
            outflow_data = outflow_response.json()
            outflow = float(outflow_data.get("result", {}).get("data", [{}])[0].get("value", 0))

            net_flow = inflow - outflow

            if abs(net_flow) > CONFIG["exchange_flow_threshold"]:
                if net_flow > 0:
                    # Net inflow → bearish
                    signals.append(OnChainSignal(
                        signal_type  = OnChainSignalType.EXCHANGE_INFLOW,
                        magnitude    = min(1.0, abs(net_flow) / (CONFIG["exchange_flow_threshold"] * 2)),
                        raw_value    = net_flow,
                        metric_name  = f"Net exchange inflow: {net_flow:.2f} {self.asset}",
                        timestamp    = time.time(),
                        source       = "cryptoquant",
                    ))
                else:
                    # Net outflow → bullish
                    signals.append(OnChainSignal(
                        signal_type  = OnChainSignalType.EXCHANGE_OUTFLOW,
                        magnitude    = min(1.0, abs(net_flow) / (CONFIG["exchange_flow_threshold"] * 2)),
                        raw_value    = abs(net_flow),
                        metric_name  = f"Net exchange outflow: {abs(net_flow):.2f} {self.asset}",
                        timestamp    = time.time(),
                        source       = "cryptoquant",
                    ))

            logger.info(f"CryptoQuant: net flow = {net_flow:.2f} {self.asset} | signals: {len(signals)}")
            return signals

        except Exception as e:
            logger.warning(f"CryptoQuant fetch failed: {e}")
            return []


# ---------------------------------------------------------------------------
# Data Source: Glassnode (premium on-chain metrics)
# ---------------------------------------------------------------------------

class GlassnodeSource:
    """
    Premium on-chain analytics. Requires paid subscription for most metrics.
    Free tier gives limited access to some indicators.

    API: https://docs.glassnode.com/
    """

    def __init__(self, api_key: str, asset: str = "BTC"):
        if not api_key:
            raise ValueError("GLASSNODE_API_KEY not set.")
        self.api_key = api_key
        self.asset = asset.upper()
        self.base_url = "https://api.glassnode.com/v1/metrics"
        logger.info("GlassnodeSource initialized.")

    def fetch_active_addresses(self) -> Optional[OnChainSignal]:
        """
        Fetch the number of active addresses (24h).
        Surge in activity can indicate heightened interest.
        """
        try:
            response = requests.get(
                f"{self.base_url}/addresses/active_count",
                params={
                    "a":       self.asset,
                    "api_key": self.api_key,
                    "i":       "24h",
                    "c":       "native",
                },
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

            if not data:
                return None

            latest = data[-1]
            value = float(latest.get("v", 0))
            timestamp = latest.get("t", time.time())

            # We'd need historical baseline to compute surge/decline properly.
            # For now, just log the value — a full implementation would track
            # a rolling average and flag deviations.
            logger.info(f"Glassnode: active addresses = {value:.0f}")

            # Placeholder: return a neutral network signal
            return OnChainSignal(
                signal_type  = OnChainSignalType.NETWORK_SURGE,
                magnitude    = 0.5,  # Neutral — need historical context
                raw_value    = value,
                metric_name  = f"Active addresses: {value:.0f}",
                timestamp    = timestamp,
                source       = "glassnode",
            )

        except Exception as e:
            logger.warning(f"Glassnode fetch failed: {e}")
            return None


# ---------------------------------------------------------------------------
# Stablecoin Flow Monitor (composite — uses Etherscan for USDT on Tron/Eth)
# ---------------------------------------------------------------------------

class StablecoinFlowMonitor:
    """
    Tracks stablecoin movements (primarily USDT).
    Large inflows to exchanges = buying power entering.
    Large outflows = capital exiting the market.

    Uses Etherscan API for Ethereum-based USDT tracking.
    (Tron-based USDT would require Tronscan API — similar pattern.)
    """

    def __init__(self, etherscan_key: str):
        if not etherscan_key:
            raise ValueError("ETHERSCAN_API_KEY not set.")
        self.etherscan_key = etherscan_key
        self.usdt_contract = "0xdac17f958d2ee523a2206206994597c13d831ec7"  # USDT on Ethereum
        self.base_url = "https://api.etherscan.io/api"
        logger.info("StablecoinFlowMonitor initialized.")

    def fetch_recent_large_transfers(self) -> list[OnChainSignal]:
        """
        Fetch large USDT transfers in the last 24h.
        Classify as inflow/outflow based on destination.
        """
        # This is a simplified implementation — a production version would
        # maintain a list of known exchange addresses and classify transfers.
        # For now, we'll just flag large movements.

        try:
            # Get recent transactions for the USDT contract
            response = requests.get(
                self.base_url,
                params={
                    "module":     "account",
                    "action":     "tokentx",
                    "contractaddress": self.usdt_contract,
                    "page":       1,
                    "offset":     100,
                    "sort":       "desc",
                    "apikey":     self.etherscan_key,
                },
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

            signals = []
            if data.get("status") != "1":
                return signals

            for tx in data.get("result", []):
                value_raw = int(tx.get("value", 0))
                value_usdt = value_raw / 1e6  # USDT has 6 decimals

                if value_usdt < CONFIG["stablecoin_flow_threshold"]:
                    continue

                # Simplified: we don't have exchange address mappings here,
                # so we'll just flag large movements as noteworthy.
                # In production, check if 'to' is a known exchange address.
                signals.append(OnChainSignal(
                    signal_type  = OnChainSignalType.STABLECOIN_INFLOW,  # Assume inflow for now
                    magnitude    = min(1.0, value_usdt / (CONFIG["stablecoin_flow_threshold"] * 2)),
                    raw_value    = value_usdt,
                    metric_name  = f"Large USDT transfer: ${value_usdt:,.0f}",
                    timestamp    = int(tx.get("timeStamp", time.time())),
                    source       = "etherscan",
                ))

            logger.info(f"StablecoinFlow: found {len(signals)} large USDT transfers.")
            return signals[:5]  # Limit to top 5 to avoid spam

        except Exception as e:
            logger.warning(f"StablecoinFlow fetch failed: {e}")
            return []


# ---------------------------------------------------------------------------
# Orchestrator: OnChainMonitor
# ---------------------------------------------------------------------------

class OnChainMonitor:
    """
    Central coordinator for all on-chain data sources.
    Polls each source on its own schedule, aggregates signals,
    and provides a unified view of on-chain state.

    Usage:
        monitor = OnChainMonitor()
        context = monitor.get_latest_context()
        print(context.net_sentiment)  # -1.0 to +1.0
    """

    def __init__(self, asset: str = "BTC"):
        self.asset = asset
        self.sources = []
        self.last_context = OnChainContext()

        # Initialize available sources
        self._initialize_sources()

        if not self.sources:
            logger.warning("No on-chain data sources available. Check API keys in .env")
        else:
            logger.info(f"OnChainMonitor initialized with {len(self.sources)} sources.")

    def _initialize_sources(self):
        """Try to init each source. Skip gracefully if API keys are missing."""
        # Whale Alert
        if CONFIG["whale_alert_key"]:
            try:
                self.sources.append(("whale_alert", WhaleAlertSource(CONFIG["whale_alert_key"], self.asset)))
            except Exception as e:
                logger.warning(f"WhaleAlert init failed: {e}")

        # CryptoQuant
        if CONFIG["cryptoquant_key"]:
            try:
                self.sources.append(("cryptoquant", CryptoQuantSource(CONFIG["cryptoquant_key"], self.asset)))
            except Exception as e:
                logger.warning(f"CryptoQuant init failed: {e}")

        # Glassnode
        if CONFIG["glassnode_key"]:
            try:
                self.sources.append(("glassnode", GlassnodeSource(CONFIG["glassnode_key"], self.asset)))
            except Exception as e:
                logger.warning(f"Glassnode init failed: {e}")

        # Stablecoin Flow
        if CONFIG["etherscan_key"]:
            try:
                self.sources.append(("stablecoin", StablecoinFlowMonitor(CONFIG["etherscan_key"])))
            except Exception as e:
                logger.warning(f"StablecoinFlow init failed: {e}")

    def collect_signals(self) -> OnChainContext:
        """
        Poll all sources and aggregate into a single OnChainContext.
        Call this periodically (e.g. every 10 minutes) from your main loop.
        """
        context = OnChainContext()

        for name, source in self.sources:
            try:
                if name == "whale_alert":
                    signals = source.fetch_recent_transactions()
                elif name == "cryptoquant":
                    signals = source.fetch_exchange_flows()
                elif name == "glassnode":
                    signal = source.fetch_active_addresses()
                    signals = [signal] if signal else []
                elif name == "stablecoin":
                    signals = source.fetch_recent_large_transfers()
                else:
                    signals = []

                for sig in signals:
                    context.add_signal(sig)

            except Exception as e:
                logger.error(f"Error collecting from {name}: {e}")

        self.last_context = context
        logger.info(f"Collected {len(context.signals)} on-chain signals | net_sentiment: {context.net_sentiment:+.2f}")
        return context

    def get_latest_context(self) -> OnChainContext:
        """Return the most recently collected context (cached)."""
        return self.last_context


# ---------------------------------------------------------------------------
# Integration with signal_layer.py
# ---------------------------------------------------------------------------

def augment_llm_prompt_with_onchain(sentiment_context: str, onchain_context: OnChainContext) -> str:
    """
    Helper function to augment the LLM prompt in signal_layer.py with on-chain data.

    Usage in signal_layer.py's LLMInterpreter.interpret():
        from onchain_data_layer import augment_llm_prompt_with_onchain
        onchain = onchain_monitor.get_latest_context()
        augmented_prompt = augment_llm_prompt_with_onchain(original_prompt, onchain)
    """
    if not onchain_context.signals:
        return sentiment_context

    onchain_summary = f"\n\nON-CHAIN CONTEXT (hard data, cannot be faked):\n"
    onchain_summary += f"- Net on-chain sentiment: {onchain_context.net_sentiment:+.2f} (-1.0 bearish → +1.0 bullish)\n"
    onchain_summary += f"- Signals detected:\n"

    for sig in onchain_context.signals[:5]:  # Top 5 most recent
        onchain_summary += f"  • {sig.metric_name} (magnitude: {sig.magnitude:.2f}, source: {sig.source})\n"

    onchain_summary += "\nConsider this on-chain data as confirmation or contradiction of the sentiment shift."

    return sentiment_context + onchain_summary


# ---------------------------------------------------------------------------
# Demo / standalone mode
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="On-Chain Data Monitor")
    parser.add_argument("--asset", default="BTC", help="Asset to monitor (BTC or ETH)")
    parser.add_argument("--interval", type=int, default=600, help="Poll interval in seconds")
    args = parser.parse_args()

    monitor = OnChainMonitor(asset=args.asset)

    if not monitor.sources:
        print("\n⚠️  No on-chain data sources available.")
        print("Add at least one API key to your .env file:")
        print("  - WHALE_ALERT_API_KEY (free tier available)")
        print("  - CRYPTOQUANT_API_KEY (recommended)")
        print("  - GLASSNODE_API_KEY (premium)")
        print("  - ETHERSCAN_API_KEY (for stablecoin tracking)")
        exit(1)

    print(f"\n{'=' * 70}")
    print(f"  ON-CHAIN MONITOR — {args.asset}")
    print(f"  Poll interval: {args.interval}s")
    print(f"  Active sources: {len(monitor.sources)}")
    print(f"{'=' * 70}\n")

    try:
        while True:
            context = monitor.collect_signals()

            print(f"\n--- On-Chain Update ---")
            print(f"Timestamp:       {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(context.timestamp))}")
            print(f"Net Sentiment:   {context.net_sentiment:+.3f}")
            print(f"Signals:         {len(context.signals)}")

            if context.signals:
                print("\nTop signals:")
                for i, sig in enumerate(context.signals[:5], 1):
                    print(f"  {i}. [{sig.signal_type.value:20s}] {sig.metric_name} (mag={sig.magnitude:.2f})")

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\nMonitor stopped by user.")
