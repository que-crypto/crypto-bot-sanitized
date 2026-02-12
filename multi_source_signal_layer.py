"""
Multi-Source Signal Layer with Explicit Weighting
===================================================
Redesigned signal detection and interpretation layer that:
    1. Uses explicit, tunable weights for each data source
    2. Allows ANY source to trigger independently on extreme values
    3. Weights on-chain data most heavily (0.5-0.55)
    4. Weights derivative data second (0.3-0.35)
    5. Weights sentiment least (0.1-0.15)
    6. Provides a clear scoring mechanism that can be backtested and optimized

This replaces the parts of signal_layer.py that handle signal detection
and LLM interpretation with a more transparent, tunable system.

Architecture:
    - MultiSourceSignalDetector: aggregates all three data sources into a single score
    - WeightedLLMInterpreter: uses the weighted score to make trading decisions
    - The score is transparent and loggable, making it easy to tune via backtesting

Usage:
    Replace the signal detection in signal_layer.py with this module:
    
    from multi_source_signal_layer import MultiSourceSignalDetector, WeightedLLMInterpreter
    
    detector = MultiSourceSignalDetector(
        sentiment_weight=0.15,
        onchain_weight=0.55,
        derivative_weight=0.30,
    )
    
    llm = WeightedLLMInterpreter()
    
    # Feed in all three contexts
    signal = detector.evaluate(sentiment_ctx, onchain_ctx, derivative_ctx)
    if signal:
        trading_signal = llm.interpret(signal)
"""

import os
import time
import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import requests
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class WeightConfig:
    """
    Tunable weights for each data source.
    
    Guidelines:
        - Must sum to 1.0
        - On-chain should be heaviest (0.5-0.6) — hardest to fake
        - Derivatives second (0.25-0.35) — forward-looking but can be gamed
        - Sentiment lowest (0.1-0.2) — noisiest, easiest to manipulate
    
    These can be optimized via backtesting.
    """
    sentiment_weight: float = 0.15      # Default: 15%
    onchain_weight: float = 0.55        # Default: 55%
    derivative_weight: float = 0.30     # Default: 30%

    def __post_init__(self):
        total = self.sentiment_weight + self.onchain_weight + self.derivative_weight
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {total:.3f}")


@dataclass
class TriggerThresholds:
    """
    Thresholds that determine when each source can trigger independently.
    
    Any source can trigger a signal if its value exceeds its threshold,
    even if the other sources are neutral.
    """
    # Sentiment delta threshold (same as before — rate of change in sentiment)
    sentiment_delta: float = 0.15       # 15% shift in sentiment window
    
    # On-chain net sentiment threshold (absolute value)
    onchain_extreme: float = 0.75       # On-chain net above ±0.75 triggers independently
    
    # Derivative net sentiment threshold (absolute value)
    derivative_extreme: float = 0.70    # Derivative net above ±0.70 triggers independently
    
    # Composite score threshold (when no single source is extreme)
    composite_score: float = 0.50       # Weighted score must exceed this to trigger


class TriggerSource(Enum):
    """Which data source triggered the signal."""
    SENTIMENT = "sentiment"
    ONCHAIN = "onchain"
    DERIVATIVE = "derivative"
    COMPOSITE = "composite"  # Multiple sources combined above threshold


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MultiSourceSignal:
    """
    Output from the multi-source detector.
    Contains the weighted composite score and metadata about what triggered it.
    """
    composite_score: float           # -1.0 to +1.0, weighted aggregate
    trigger_source: TriggerSource    # What caused this signal to fire
    
    # Individual component scores (for transparency)
    sentiment_score: float
    onchain_score: float
    derivative_score: float
    
    # Raw contexts (passed through for LLM interpretation)
    sentiment_context: any
    onchain_context: any
    derivative_context: any
    
    timestamp: float
    
    def __repr__(self):
        return (
            f"MultiSourceSignal(score={self.composite_score:+.3f}, "
            f"trigger={self.trigger_source.value}, "
            f"sentiment={self.sentiment_score:+.2f}, "
            f"onchain={self.onchain_score:+.2f}, "
            f"derivative={self.derivative_score:+.2f})"
        )


# ---------------------------------------------------------------------------
# Multi-Source Signal Detector
# ---------------------------------------------------------------------------

class MultiSourceSignalDetector:
    """
    Combines sentiment, on-chain, and derivative data into a single weighted score.
    
    Key differences from the original signal_layer.py:
        1. Explicit weights for each source (tunable)
        2. Any source can trigger independently on extreme values
        3. Composite score is transparent and loggable
        4. Designed for backtesting and optimization
    
    Usage:
        detector = MultiSourceSignalDetector()
        signal = detector.evaluate(sentiment_ctx, onchain_ctx, deriv_ctx)
        if signal:
            print(signal.composite_score)  # Use this for trade sizing
    """
    
    def __init__(
        self,
        weights: Optional[WeightConfig] = None,
        thresholds: Optional[TriggerThresholds] = None,
    ):
        self.weights = weights or WeightConfig()
        self.thresholds = thresholds or TriggerThresholds()
        
        logger.info(
            f"MultiSourceSignalDetector initialized | "
            f"weights: sentiment={self.weights.sentiment_weight:.2f}, "
            f"onchain={self.weights.onchain_weight:.2f}, "
            f"derivative={self.weights.derivative_weight:.2f}"
        )
        
        # Track previous sentiment for delta calculation
        self.prev_sentiment_score = 0.0
    
    def evaluate(
        self,
        sentiment_context,      # From sentiment pipeline (SignalEvent or similar)
        onchain_context,        # From OnChainContext
        derivative_context,     # From DerivativeContext
    ) -> Optional[MultiSourceSignal]:
        """
        Evaluate all three data sources and determine if a signal should fire.
        
        Returns:
            MultiSourceSignal if any trigger condition is met, else None.
        """
        # --- Extract scores from each context ---
        
        # Sentiment: use the delta (rate of change) as the score
        # sentiment_context is expected to have a sentiment_delta attribute
        if hasattr(sentiment_context, 'sentiment_delta'):
            sentiment_delta = sentiment_context.sentiment_delta
        else:
            sentiment_delta = 0.0
        
        sentiment_score = sentiment_delta  # Already in [-1, 1] range typically
        
        # On-chain: use net_sentiment directly
        if onchain_context and hasattr(onchain_context, 'net_sentiment'):
            onchain_score = onchain_context.net_sentiment
        else:
            onchain_score = 0.0
        
        # Derivative: use net_sentiment directly
        if derivative_context and hasattr(derivative_context, 'net_sentiment'):
            derivative_score = derivative_context.net_sentiment
        else:
            derivative_score = 0.0
        
        # --- Compute weighted composite score ---
        composite_score = (
            sentiment_score * self.weights.sentiment_weight +
            onchain_score * self.weights.onchain_weight +
            derivative_score * self.weights.derivative_weight
        )
        
        # Clamp to [-1, 1]
        composite_score = max(-1.0, min(1.0, composite_score))
        
        # --- Determine if any trigger condition is met ---
        
        trigger_source = None
        
        # Check for independent extremes (any single source can trigger)
        if abs(onchain_score) >= self.thresholds.onchain_extreme:
            trigger_source = TriggerSource.ONCHAIN
            logger.info(f"ON-CHAIN EXTREME triggered | score={onchain_score:+.3f}")
        
        elif abs(derivative_score) >= self.thresholds.derivative_extreme:
            trigger_source = TriggerSource.DERIVATIVE
            logger.info(f"DERIVATIVE EXTREME triggered | score={derivative_score:+.3f}")
        
        elif abs(sentiment_delta) >= self.thresholds.sentiment_delta:
            # Sentiment triggered, but check if composite score is actionable
            if abs(composite_score) >= self.thresholds.composite_score:
                trigger_source = TriggerSource.SENTIMENT
                logger.info(f"SENTIMENT triggered | delta={sentiment_delta:+.3f}, composite={composite_score:+.3f}")
            else:
                # Sentiment moved but composite score is weak (other sources contradicting)
                logger.debug(
                    f"Sentiment delta {sentiment_delta:+.3f} crossed threshold, "
                    f"but composite score {composite_score:+.3f} below threshold — signal suppressed."
                )
                return None
        
        elif abs(composite_score) >= self.thresholds.composite_score:
            # No single source is extreme, but aggregate is strong
            trigger_source = TriggerSource.COMPOSITE
            logger.info(f"COMPOSITE triggered | score={composite_score:+.3f}")
        
        else:
            # No trigger condition met
            return None
        
        # --- Build and return the signal ---
        signal = MultiSourceSignal(
            composite_score=composite_score,
            trigger_source=trigger_source,
            sentiment_score=sentiment_score,
            onchain_score=onchain_score,
            derivative_score=derivative_score,
            sentiment_context=sentiment_context,
            onchain_context=onchain_context,
            derivative_context=derivative_context,
            timestamp=time.time(),
        )
        
        logger.info(f"Signal generated: {signal}")
        return signal


# ---------------------------------------------------------------------------
# Weighted LLM Interpreter
# ---------------------------------------------------------------------------

class WeightedLLMInterpreter:
    """
    Sends MultiSourceSignal to the LLM for final interpretation.
    
    Differences from the original LLMInterpreter:
        1. Receives a weighted composite score (transparent)
        2. Prompt explicitly mentions the weighting system
        3. LLM's job is to sanity-check and add reasoning, not to re-weight
    
    The LLM can still adjust confidence based on context, but the heavy
    lifting of combining the signals is done by the explicit weighting.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-5-20250929",
    ):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set in .env")
    
    def interpret(self, signal: MultiSourceSignal):
        """
        Sends the multi-source signal to the LLM for interpretation.
        
        Returns:
            TradingSignal (direction, confidence, reasoning)
        """
        # Import here to avoid circular dependency
        from signal_layer import TradingSignal
        
        # --- Build the prompt ---
        
        # Determine suggested direction based on composite score
        if signal.composite_score > 0.3:
            suggested_direction = "BUY"
        elif signal.composite_score < -0.3:
            suggested_direction = "SELL"
        else:
            suggested_direction = "HOLD"
        
        prompt = f"""You are analyzing a multi-source trading signal for cryptocurrency.

COMPOSITE SIGNAL SCORE: {signal.composite_score:+.3f} (range: -1.0 bearish to +1.0 bullish)
SUGGESTED DIRECTION: {suggested_direction}
TRIGGERED BY: {signal.trigger_source.value}

COMPONENT BREAKDOWN (weighted):
- Sentiment score:    {signal.sentiment_score:+.3f}  (weight: 15%)
- On-chain score:     {signal.onchain_score:+.3f}  (weight: 55%)
- Derivative score:   {signal.derivative_score:+.3f}  (weight: 30%)

"""
        
        # Add sentiment context if available
        if hasattr(signal.sentiment_context, 'messages'):
            top_messages = sorted(
                signal.sentiment_context.messages,
                key=lambda s: s.confidence,
                reverse=True
            )[:10]  # Reduced from 20 since sentiment is lower priority
            
            prompt += "SENTIMENT CONTEXT (what people are saying):\n"
            for msg in top_messages:
                prompt += f"  • [{msg.raw_message.source}] {msg.raw_message.text[:100]}...\n"
            prompt += "\n"
        
        # Add on-chain context if available
        if signal.onchain_context and signal.onchain_context.signals:
            prompt += "ON-CHAIN CONTEXT (hard data, cannot be faked):\n"
            for sig in signal.onchain_context.signals[:5]:
                prompt += f"  • {sig.metric_name} (magnitude: {sig.magnitude:.2f})\n"
            prompt += "\n"
        
        # Add derivative context if available
        if signal.derivative_context and signal.derivative_context.signals:
            prompt += "DERIVATIVE CONTEXT (forward-looking positioning):\n"
            for sig in signal.derivative_context.signals[:5]:
                prompt += f"  • {sig.metric_name} (magnitude: {sig.magnitude:.2f})\n"
            prompt += "\n"
        
        prompt += f"""
WEIGHTING PHILOSOPHY:
- On-chain data (55%) is weighted most heavily because it represents actual capital movement and cannot be faked.
- Derivative data (30%) is weighted second because it shows where leverage is positioned and vulnerable.
- Sentiment data (15%) is weighted lowest because it's the noisiest and easiest to manipulate.

YOUR TASK:
1. Review the composite score and component breakdown.
2. Sanity-check: Does the suggested direction make sense given all the context?
3. Assess confidence: How strong is this signal? Consider:
   - Are all three sources aligned (high confidence) or contradicting (low confidence)?
   - Is there a clear narrative/catalyst, or is this just noise?
   - Are there any red flags that would warrant reducing confidence or flipping to HOLD?

Respond ONLY with valid JSON in this exact format:
{{
    "direction": "BUY" | "SELL" | "HOLD",
    "confidence": <float 0.0 to 1.0>,
    "reasoning": "<2-4 sentences explaining your decision, mentioning key factors from each data source>"
}}
"""
        
        # --- Call the LLM ---
        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "content-type": "application/json",
                },
                json={
                    "model": self.model,
                    "max_tokens": 400,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=30,
            )
            response.raise_for_status()
            raw_text = response.json()["content"][0]["text"].strip()
            
            # Strip markdown fences if present
            if raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
            
            result = json.loads(raw_text)
            
            trading_signal = TradingSignal(
                direction=result["direction"],
                confidence=float(result["confidence"]),
                reasoning=result["reasoning"],
                sentiment_score=signal.composite_score,  # Use composite as the score
                timestamp=time.time(),
            )
            
            logger.info(
                f"LLM decision: {trading_signal.direction} | "
                f"conf={trading_signal.confidence:.2f} | "
                f"{trading_signal.reasoning}"
            )
            
            return trading_signal
        
        except Exception as e:
            logger.error(f"LLM interpretation failed: {e}")
            # Fallback: use composite score to make a mechanical decision
            if abs(signal.composite_score) < 0.3:
                direction = "HOLD"
                confidence = 0.0
            elif signal.composite_score > 0:
                direction = "BUY"
                confidence = min(0.7, abs(signal.composite_score))
            else:
                direction = "SELL"
                confidence = min(0.7, abs(signal.composite_score))
            
            return TradingSignal(
                direction=direction,
                confidence=confidence,
                reasoning=f"LLM unavailable — mechanical decision based on composite score {signal.composite_score:+.3f}",
                sentiment_score=signal.composite_score,
                timestamp=time.time(),
            )


# ---------------------------------------------------------------------------
# Integration Example
# ---------------------------------------------------------------------------

def example_usage():
    """
    Shows how to use the multi-source signal detector with all three contexts.
    """
    # Mock contexts for demonstration
    from dataclasses import dataclass as dc
    
    @dc
    class MockSentimentContext:
        sentiment_delta: float = 0.25
        messages: list = None
    
    @dc
    class MockOnChainContext:
        net_sentiment: float = 0.68
        signals: list = None
    
    @dc
    class MockDerivativeContext:
        net_sentiment: float = -0.42
        signals: list = None
    
    # Initialize detector with custom weights
    detector = MultiSourceSignalDetector(
        weights=WeightConfig(
            sentiment_weight=0.10,
            onchain_weight=0.60,
            derivative_weight=0.30,
        )
    )
    
    # LLM interpreter would be initialized here, but skipped for demo
    # llm = WeightedLLMInterpreter()
    
    # Create mock contexts
    print("Scenario 1: On-chain extreme (should trigger independently)")
    sentiment = MockSentimentContext(sentiment_delta=0.08, messages=[])  # Below threshold
    onchain = MockOnChainContext(net_sentiment=0.82, signals=[])  # EXTREME - triggers alone
    derivative = MockDerivativeContext(net_sentiment=-0.15, signals=[])  # Neutral
    
    signal = detector.evaluate(sentiment, onchain, derivative)
    
    if signal:
        print(f"\n{'=' * 70}")
        print(f"  SIGNAL DETECTED")
        print(f"{'=' * 70}")
        print(f"  Composite Score:    {signal.composite_score:+.3f}")
        print(f"  Triggered By:       {signal.trigger_source.value}")
        print(f"  Component Scores:")
        print(f"    - Sentiment:      {signal.sentiment_score:+.3f} (weight: 10%)")
        print(f"    - On-Chain:       {signal.onchain_score:+.3f} (weight: 60%)")
        print(f"    - Derivative:     {signal.derivative_score:+.3f} (weight: 30%)")
        print(f"{'=' * 70}\n")
    else:
        print("  → No signal (unexpected)\n")
    
    print("\nScenario 2: Sentiment triggers, but contradicted by derivatives")
    sentiment2 = MockSentimentContext(sentiment_delta=0.35, messages=[])  # Strong sentiment
    onchain2 = MockOnChainContext(net_sentiment=0.45, signals=[])  # Moderately bullish
    derivative2 = MockDerivativeContext(net_sentiment=-0.65, signals=[])  # Strong bearish
    
    signal2 = detector.evaluate(sentiment2, onchain2, derivative2)
    
    if signal2:
        print(f"\n{'=' * 70}")
        print(f"  SIGNAL DETECTED")
        print(f"{'=' * 70}")
        print(f"  Composite Score:    {signal2.composite_score:+.3f}")
        print(f"  Triggered By:       {signal2.trigger_source.value}")
        print(f"  Component Scores:")
        print(f"    - Sentiment:      {signal2.sentiment_score:+.3f} (weight: 10%)")
        print(f"    - On-Chain:       {signal2.onchain_score:+.3f} (weight: 60%)")
        print(f"    - Derivative:     {signal2.derivative_score:+.3f} (weight: 30%)")
        print(f"  Note: Sentiment is bullish but derivatives show bearish positioning.")
        print(f"        The composite score reflects this conflict (on-chain dominates).")
        print(f"{'=' * 70}\n")
    else:
        print("  → No signal (sentiment contradicted by derivatives)\n")
    
    print("\nScenario 3: All sources aligned (highest confidence)")
    sentiment3 = MockSentimentContext(sentiment_delta=0.22, messages=[])
    onchain3 = MockOnChainContext(net_sentiment=0.71, signals=[])
    derivative3 = MockDerivativeContext(net_sentiment=0.58, signals=[])
    
    signal3 = detector.evaluate(sentiment3, onchain3, derivative3)
    
    if signal3:
        print(f"\n{'=' * 70}")
        print(f"  SIGNAL DETECTED")
        print(f"{'=' * 70}")
        print(f"  Composite Score:    {signal3.composite_score:+.3f}")
        print(f"  Triggered By:       {signal3.trigger_source.value}")
        print(f"  Component Scores:")
        print(f"    - Sentiment:      {signal3.sentiment_score:+.3f} (weight: 10%)")
        print(f"    - On-Chain:       {signal3.onchain_score:+.3f} (weight: 60%)")
        print(f"    - Derivative:     {signal3.derivative_score:+.3f} (weight: 30%)")
        print(f"  Note: All three sources are bullish and aligned.")
        print(f"        This would generate a high-confidence BUY signal.")
        print(f"{'=' * 70}\n")
    else:
        print("  → No signal (unexpected)\n")


# ---------------------------------------------------------------------------
# Backtesting utilities
# ---------------------------------------------------------------------------

class WeightOptimizer:
    """
    Helper class for backtesting different weight configurations.
    
    Usage:
        optimizer = WeightOptimizer()
        best_weights = optimizer.grid_search(historical_data)
    """
    
    def __init__(self):
        self.results = []
    
    def evaluate_weights(
        self,
        weights: WeightConfig,
        historical_signals: list,  # List of (sentiment, onchain, derivative, actual_outcome)
    ) -> dict:
        """
        Test a weight configuration on historical data.
        
        Returns:
            Performance metrics (accuracy, sharpe, win_rate, etc.)
        """
        detector = MultiSourceSignalDetector(weights=weights)
        
        correct_direction = 0
        total_signals = 0
        pnls = []
        
        for sentiment_ctx, onchain_ctx, deriv_ctx, actual_outcome in historical_signals:
            signal = detector.evaluate(sentiment_ctx, onchain_ctx, deriv_ctx)
            
            if signal is None:
                continue
            
            total_signals += 1
            
            # Predicted direction based on composite score
            predicted = 1 if signal.composite_score > 0 else -1
            
            # Check if direction was correct
            if (predicted > 0 and actual_outcome > 0) or (predicted < 0 and actual_outcome < 0):
                correct_direction += 1
            
            # Simulated PnL (sign-weighted by composite score)
            pnl = signal.composite_score * actual_outcome
            pnls.append(pnl)
        
        if total_signals == 0:
            return {"accuracy": 0, "sharpe": 0, "total_signals": 0}
        
        accuracy = correct_direction / total_signals
        sharpe = (sum(pnls) / len(pnls)) / (max(1e-6, __import__('statistics').stdev(pnls))) if len(pnls) > 1 else 0
        
        return {
            "weights": weights,
            "accuracy": accuracy,
            "sharpe": sharpe,
            "total_signals": total_signals,
            "avg_pnl": sum(pnls) / len(pnls) if pnls else 0,
        }
    
    def grid_search(
        self,
        historical_signals: list,
        sentiment_range: tuple = (0.05, 0.20, 0.05),  # (min, max, step)
        onchain_range: tuple = (0.45, 0.65, 0.05),
    ) -> dict:
        """
        Grid search over weight combinations.
        Derivative weight is computed as 1.0 - sentiment - onchain.
        
        Returns:
            Best performing weight configuration.
        """
        best_result = None
        best_sharpe = -999
        
        s_min, s_max, s_step = sentiment_range
        o_min, o_max, o_step = onchain_range
        
        sentiment_vals = [round(s_min + i * s_step, 2) for i in range(int((s_max - s_min) / s_step) + 1)]
        onchain_vals = [round(o_min + i * o_step, 2) for i in range(int((o_max - o_min) / o_step) + 1)]
        
        for s_weight in sentiment_vals:
            for o_weight in onchain_vals:
                d_weight = round(1.0 - s_weight - o_weight, 2)
                
                # Skip invalid combinations
                if d_weight < 0.1 or d_weight > 0.5:
                    continue
                
                weights = WeightConfig(
                    sentiment_weight=s_weight,
                    onchain_weight=o_weight,
                    derivative_weight=d_weight,
                )
                
                result = self.evaluate_weights(weights, historical_signals)
                self.results.append(result)
                
                if result["sharpe"] > best_sharpe:
                    best_sharpe = result["sharpe"]
                    best_result = result
                
                logger.info(
                    f"Tested weights: S={s_weight:.2f} O={o_weight:.2f} D={d_weight:.2f} | "
                    f"Sharpe={result['sharpe']:.3f} Acc={result['accuracy']:.2f}"
                )
        
        logger.info(f"\nBest configuration: {best_result}")
        return best_result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  MULTI-SOURCE SIGNAL LAYER — DEMO")
    print("=" * 70 + "\n")
    
    example_usage()
    
    print("\n" + "=" * 70)
    print("  Weight optimization example:")
    print("=" * 70)
    print("""
To backtest and optimize weights:

1. Collect historical data:
   historical_signals = [
       (sentiment_ctx, onchain_ctx, deriv_ctx, actual_price_move),
       ...
   ]

2. Run grid search:
   optimizer = WeightOptimizer()
   best = optimizer.grid_search(historical_signals)
   
3. Use the best weights in production:
   detector = MultiSourceSignalDetector(weights=best["weights"])
""")
