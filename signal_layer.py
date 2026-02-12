"""
Crypto Sentiment Signal/Decision Layer — UPGRADED
===================================================
Hybrid multi-source pipeline with explicit weighting:
    - Sentiment (Twitter/Reddit/News): 10-15% weight
    - On-Chain (whale activity, exchange flows): 50-60% weight
    - Derivatives (funding, OI, liquidations): 30-35% weight

BERT handles high-volume real-time sentiment scoring.
Multi-source detector combines all three data sources with explicit weights.
LLM provides final interpretation and sanity-checking.

Dependencies:
    pip install transformers torch requests python-dotenv

Setup:
    Create a .env file with:
        ANTHROPIC_API_KEY=your_key_here
        
        # On-chain data (at least one recommended)
        WHALE_ALERT_API_KEY=your_key
        CRYPTOQUANT_API_KEY=your_key
        GLASSNODE_API_KEY=your_key
        ETHERSCAN_API_KEY=your_key
        
        # Derivative data (optional — Binance works without API key)
        COINGLASS_API_KEY=your_key

Architecture Change from v1:
    OLD: Sentiment triggers → LLM weights all sources implicitly
    NEW: Any source can trigger → Explicit math weights → LLM sanity-checks
"""

import os
import time
import json
import logging
from dataclasses import dataclass, field
from collections import deque
from dotenv import load_dotenv
import requests
from transformers import pipeline

# Import the multi-source weighted detection system
from multi_source_signal_layer import (
    MultiSourceSignalDetector,
    WeightedLLMInterpreter,
    WeightConfig,
    TriggerThresholds,
)

# Import on-chain and derivative monitors
try:
    from onchain_data_layer import OnChainMonitor
    ONCHAIN_AVAILABLE = True
except ImportError:
    ONCHAIN_AVAILABLE = False
    logger.warning("onchain_data_layer.py not found — on-chain signals disabled")

try:
    from derivative_signals_layer import DerivativeMonitor
    DERIVATIVE_AVAILABLE = True
except ImportError:
    DERIVATIVE_AVAILABLE = False
    logger.warning("derivative_signals_layer.py not found — derivative signals disabled")

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RawMessage:
    """A single piece of incoming text (tweet, article, etc.)."""
    text: str
    source: str          # e.g. "twitter", "reddit", "news"
    timestamp: float     # unix epoch
    source_weight: float = 1.0  # higher = more trusted source


@dataclass
class BertScore:
    """Output from the BERT classification layer."""
    positive: float
    negative: float
    neutral: float
    label: str           # dominant label
    confidence: float
    raw_message: RawMessage


@dataclass
class SignalEvent:
    """A flagged event that gets escalated to the LLM layer."""
    messages: list[BertScore]
    sentiment_delta: float   # how much sentiment shifted
    volume_spike: bool       # whether discussion volume spiked
    timestamp: float


@dataclass
class TradingSignal:
    """Final output: what the system recommends."""
    direction: str           # "BUY", "SELL", or "HOLD"
    confidence: float        # 0.0 - 1.0
    reasoning: str           # LLM's explanation
    sentiment_score: float   # composite score (now from multi-source detector)
    timestamp: float


# ---------------------------------------------------------------------------
# Layer 1: BERT Sentiment Classifier (high-volume, real-time)
# ---------------------------------------------------------------------------

class BertSentimentClassifier:
    """
    Loads a fine-tuned sentiment model and scores messages in bulk.
    Uses 'distilbert-base-uncased-finetuned-sst-2-english' as a default.
    For crypto-specific accuracy, swap in a model like:
        - 'mrm8488/distilbert-base-finetuned-finance-sentiment'
        - Your custom fine-tuned model from crypto_bert_finetuner.py
    
    Usage:
        classifier = BertSentimentClassifier(model_name="./crypto_bert_model")
    """

    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        logger.info(f"Loading BERT model: {model_name}")
        self.classifier = pipeline("sentiment-analysis", model=model_name, top_k=None)

    def score(self, message: RawMessage) -> BertScore:
        """Score a single message. Returns structured output."""
        try:
            results = self.classifier(message.text[:512])  # BERT token limit
            # Normalize into positive/negative/neutral
            scores = {r["label"].lower(): r["score"] for r in results}
            positive = scores.get("positive", scores.get("pos", 0.0))
            negative = scores.get("negative", scores.get("neg", 0.0))
            neutral  = 1.0 - positive - negative

            label = max(
                [("positive", positive), ("negative", negative), ("neutral", neutral)],
                key=lambda x: x[1]
            )[0]

            return BertScore(
                positive=positive,
                negative=negative,
                neutral=neutral,
                label=label,
                confidence=max(positive, negative, neutral),
                raw_message=message,
            )
        except Exception as e:
            logger.warning(f"BERT scoring failed: {e}")
            return BertScore(0.33, 0.33, 0.34, "neutral", 0.0, message)

    def score_batch(self, messages: list[RawMessage]) -> list[BertScore]:
        """Score a batch of messages."""
        return [self.score(m) for m in messages]


# ---------------------------------------------------------------------------
# Layer 2: Sentiment Tracking (for delta calculation)
# ---------------------------------------------------------------------------

class SignalDetector:
    """
    Watches the stream of BERT scores for meaningful shifts.
    
    NOTE: In the upgraded system, this class is ONLY used to track
    sentiment delta over time. The actual trigger decision is handled
    by MultiSourceSignalDetector, which combines sentiment with
    on-chain and derivative data.
    """

    def __init__(self, window_size: int = 200, delta_threshold: float = 0.15, volume_spike_multiplier: float = 2.0):
        self.window = deque(maxlen=window_size)       # rolling window of BertScores
        self.baseline_volume = 0                      # avg messages per window
        self.prev_sentiment = 0.0                     # last computed weighted sentiment
        self.delta_threshold = delta_threshold        # how big a shift must be to flag
        self.volume_spike_multiplier = volume_spike_multiplier

    def _weighted_sentiment(self, scores: list[BertScore]) -> float:
        """
        Compute a single sentiment value from a list of scores.
        Range: -1.0 (fully negative) to +1.0 (fully positive).
        Weighted by source importance.
        """
        if not scores:
            return 0.0
        total_weight = 0.0
        weighted_sum = 0.0
        for s in scores:
            w = s.raw_message.source_weight
            # Map to -1 to +1 scale
            sentiment = (s.positive - s.negative) * w
            weighted_sum += sentiment
            total_weight += w
        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def ingest(self, score: BertScore) -> SignalEvent | None:
        """
        Feed a new BertScore into the detector.
        Returns a SignalEvent with current sentiment delta.
        
        In the upgraded system, this returns an event even if threshold
        is NOT crossed — the MultiSourceSignalDetector will make the
        final trigger decision by combining this with on-chain/derivative data.
        """
        self.window.append(score)

        if len(self.window) < self.window.maxlen:
            # Not enough data yet to establish a baseline
            self.prev_sentiment = self._weighted_sentiment(list(self.window))
            self.baseline_volume = len(self.window)
            return None

        current_sentiment = self._weighted_sentiment(list(self.window))
        delta = current_sentiment - self.prev_sentiment

        # Check for volume spike
        volume_spike = len(self.window) >= self.baseline_volume * self.volume_spike_multiplier

        self.prev_sentiment = current_sentiment

        # ALWAYS return an event (multi-source detector will decide if it's actionable)
        return SignalEvent(
            messages=list(self.window),
            sentiment_delta=delta,
            volume_spike=volume_spike,
            timestamp=time.time(),
        )


# ---------------------------------------------------------------------------
# Orchestrator: Multi-Source Pipeline
# ---------------------------------------------------------------------------

class SentimentPipeline:
    """
    Main pipeline with multi-source weighted detection.
    
    Data flow:
        RawMessage → BERT → SignalDetector (sentiment delta tracking)
                                    ↓
                    MultiSourceSignalDetector ← OnChainMonitor
                                    ↓           ← DerivativeMonitor
                            (explicit weighting)
                                    ↓
                        WeightedLLMInterpreter → TradingSignal
    
    Usage:
        pipeline = SentimentPipeline(
            asset="BTC",
            sentiment_weight=0.15,
            onchain_weight=0.55,
            derivative_weight=0.30,
        )
        
        signal = pipeline.process(RawMessage(text="...", source="twitter", timestamp=time.time()))
        if signal:
            print(signal.direction, signal.confidence)
    """

    def __init__(
        self,
        asset: str = "BTC",
        sentiment_weight: float = 0.15,
        onchain_weight: float = 0.55,
        derivative_weight: float = 0.30,
        bert_model: str = "distilbert-base-uncased-finetuned-sst-2-english",
    ):
        """
        Initialize the multi-source pipeline.
        
        Args:
            asset: Which asset to monitor (BTC, ETH, etc.)
            sentiment_weight: Weight for sentiment signals (0.10-0.20 recommended)
            onchain_weight: Weight for on-chain signals (0.50-0.60 recommended)
            derivative_weight: Weight for derivative signals (0.25-0.35 recommended)
            bert_model: Path to BERT model (use "./crypto_bert_model" for custom)
        """
        logger.info(f"Initializing SentimentPipeline for {asset}")
        
        # Layer 1: BERT sentiment classifier
        self.bert = BertSentimentClassifier(model_name=bert_model)
        
        # Layer 2a: Sentiment delta tracker
        self.sentiment_detector = SignalDetector(window_size=200, delta_threshold=0.15)
        
        # Layer 2b: Multi-source detector with explicit weights
        self.multi_detector = MultiSourceSignalDetector(
            weights=WeightConfig(
                sentiment_weight=sentiment_weight,
                onchain_weight=onchain_weight,
                derivative_weight=derivative_weight,
            ),
            thresholds=TriggerThresholds(
                sentiment_delta=0.15,       # Sentiment must shift by 15%
                onchain_extreme=0.75,       # On-chain net ≥ ±0.75 triggers independently
                derivative_extreme=0.70,    # Derivative net ≥ ±0.70 triggers independently
                composite_score=0.50,       # Composite must exceed 0.50 to trigger
            ),
        )
        
        # Layer 3: Weighted LLM interpreter
        self.llm = WeightedLLMInterpreter()
        
        # Layer 2c: On-chain monitor
        self.onchain_monitor = None
        if ONCHAIN_AVAILABLE:
            try:
                self.onchain_monitor = OnChainMonitor(asset=asset)
                logger.info("On-chain monitor initialized.")
            except Exception as e:
                logger.warning(f"On-chain monitor failed to initialize: {e}")
        else:
            logger.info("On-chain monitor not available (onchain_data_layer.py missing).")
        
        # Layer 2d: Derivative monitor
        self.derivative_monitor = None
        if DERIVATIVE_AVAILABLE:
            try:
                self.derivative_monitor = DerivativeMonitor(asset=asset)
                logger.info("Derivative monitor initialized.")
            except Exception as e:
                logger.warning(f"Derivative monitor failed to initialize: {e}")
        else:
            logger.info("Derivative monitor not available (derivative_signals_layer.py missing).")
        
        logger.info(
            f"Pipeline initialized | Weights: sentiment={sentiment_weight:.2f}, "
            f"onchain={onchain_weight:.2f}, derivative={derivative_weight:.2f}"
        )

    def process(self, message: RawMessage) -> TradingSignal | None:
        """
        Process a single incoming message through the full multi-source pipeline.
        Returns a TradingSignal only if a trigger condition is met.
        """
        # Layer 1: BERT scores the message
        bert_score = self.bert.score(message)
        logger.debug(f"BERT: {bert_score.label} (conf={bert_score.confidence:.2f})")

        # Layer 2a: Feed into sentiment detector (for delta tracking)
        sentiment_event = self.sentiment_detector.ingest(bert_score)
        
        if sentiment_event is None:
            # Not enough data yet to compute delta
            return None

        # Layer 2b: Collect on-chain context
        onchain_context = None
        if self.onchain_monitor:
            try:
                onchain_context = self.onchain_monitor.collect_signals()
            except Exception as e:
                logger.warning(f"On-chain collection failed: {e}")
        
        # Layer 2c: Collect derivative context
        derivative_context = None
        if self.derivative_monitor:
            try:
                derivative_context = self.derivative_monitor.collect_signals()
            except Exception as e:
                logger.warning(f"Derivative collection failed: {e}")
        
        # Layer 2d: Multi-source evaluation with explicit weighting
        multi_signal = self.multi_detector.evaluate(
            sentiment_context=sentiment_event,
            onchain_context=onchain_context,
            derivative_context=derivative_context,
        )
        
        if multi_signal is None:
            # No trigger condition met
            return None
        
        # Layer 3: LLM interpretation
        logger.info("Escalating to LLM for final interpretation...")
        trading_signal = self.llm.interpret(multi_signal)
        
        logger.info(
            f"Signal: {trading_signal.direction} | "
            f"conf={trading_signal.confidence:.2f} | "
            f"composite={multi_signal.composite_score:+.3f} | "
            f"{trading_signal.reasoning}"
        )
        
        return trading_signal


# ---------------------------------------------------------------------------
# Demo / test harness
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  MULTI-SOURCE SENTIMENT PIPELINE — DEMO")
    print("=" * 70 + "\n")
    
    # Simulated incoming messages to demonstrate the pipeline
    demo_messages = [
        # Noise / neutral
        RawMessage("Just checking out the new crypto app", "twitter", time.time(), 1.0),
        RawMessage("Anyone else using MetaMask lately?", "reddit", time.time(), 1.2),
        RawMessage("Crypto is interesting I guess", "twitter", time.time(), 1.0),
    ] * 60 + [
        # Sudden bullish spike (simulating a real event)
        RawMessage("HUGE news: major exchange just listed the new token!", "news", time.time(), 2.0),
        RawMessage("This is massive — institutional money is flowing in!", "twitter", time.time(), 1.0),
        RawMessage("Just saw the listing announcement. This is going to moon.", "reddit", time.time(), 1.2),
        RawMessage("Bullish signal confirmed by on-chain data. Buy the dip.", "news", time.time(), 2.0),
        RawMessage("Everyone is talking about the new partnership deal. Very bullish!", "twitter", time.time(), 1.0),
    ] * 20

    pipeline = SentimentPipeline(
        asset="BTC",
        sentiment_weight=0.15,
        onchain_weight=0.55,
        derivative_weight=0.30,
    )

    signals_detected = 0
    for msg in demo_messages:
        signal = pipeline.process(msg)
        if signal:
            signals_detected += 1
            print("\n" + "=" * 70)
            print(f"  TRADING SIGNAL #{signals_detected}")
            print("=" * 70)
            print(f"  Direction:        {signal.direction}")
            print(f"  Confidence:       {signal.confidence:.2f}")
            print(f"  Composite Score:  {signal.sentiment_score:+.3f}")
            print(f"  Reasoning:        {signal.reasoning}")
            print("=" * 70 + "\n")
    
    if signals_detected == 0:
        print("No signals detected in demo run.")
        print("Note: On-chain and derivative monitors may not be initialized in test mode.")
