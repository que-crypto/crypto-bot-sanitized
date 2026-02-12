"""
Crypto Sentiment Trading — Execution Layer
============================================
Consumes TradingSignal outputs from signal_layer.py and converts them
into actual (or simulated) leveraged trades on a crypto exchange.

Responsibilities:
    - Position sizing (Kelly Criterion-based)
    - Risk management (stop-loss, take-profit, max drawdown kill switch)
    - Leverage selection based on signal confidence
    - Exchange communication via ccxt (supports Binance, Kraken, etc.)
    - Dry-run mode for paper trading (no real money at risk)
    - Full trade lifecycle: open → monitor → close

Dependencies:
    pip install ccxt python-dotenv

Setup:
    Add to your .env file:
        EXCHANGE_API_KEY=your_exchange_api_key
        EXCHANGE_API_SECRET=your_exchange_api_secret
        EXCHANGE_NAME=binance          # or kraken, coinbase, etc.
        TRADING_PAIR=BTC/USDT
        DRY_RUN=true                   # set to false ONLY when ready for live trading
"""

import os
import time
import json
import logging
from enum import Enum
from dataclasses import dataclass, field
from dotenv import load_dotenv

try:
    import ccxt
except ImportError:
    raise ImportError("Run: pip install ccxt")

# Import the TradingSignal dataclass from the signal layer
# If running standalone for testing, the mock version below is used instead.
try:
    from signal_layer import TradingSignal
except ImportError:
    @dataclass
    class TradingSignal:
        direction: str
        confidence: float
        reasoning: str
        sentiment_score: float
        timestamp: float

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ExecutionConfig:
    """All tunable parameters for the execution layer in one place."""

    # --- Account & exchange ---
    exchange_name: str = os.getenv("EXCHANGE_NAME", "binance")
    api_key: str        = os.getenv("EXCHANGE_API_KEY", "")
    api_secret: str     = os.getenv("EXCHANGE_API_SECRET", "")
    trading_pair: str   = os.getenv("TRADING_PAIR", "BTC/USDT")
    dry_run: bool       = os.getenv("DRY_RUN", "true").lower() == "true"

    # --- Position sizing ---
    # Total capital (in quote currency, e.g. USDT) available for this strategy.
    # In dry-run mode this is your simulated balance.
    total_capital: float = 1000.0

    # Maximum fraction of total_capital risked on any single trade (e.g. 0.05 = 5%)
    max_risk_per_trade: float = 0.05

    # Kelly fraction cap — even if Kelly says bet more, never exceed this
    kelly_cap: float = 0.20

    # --- Leverage ---
    # Leverage is mapped from signal confidence, clamped to [min, max].
    min_leverage: int = 2
    max_leverage: int = 10

    # --- Stop-loss & take-profit (as fraction of entry price) ---
    stop_loss_pct: float   = 0.03   # 3% adverse move → close
    take_profit_pct: float = 0.06   # 6% favorable move → close

    # --- Kill switch ---
    # If total portfolio drawdown from starting capital hits this, stop ALL trading.
    max_drawdown_pct: float = 0.15   # 15% drawdown kills the bot

    # --- Cooldown ---
    # Minimum seconds between two consecutive trade opens (prevents over-trading)
    cooldown_seconds: int = 300      # 5 minutes


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class TradeStatus(Enum):
    OPEN   = "open"
    CLOSED = "closed"
    FAILED = "failed"


class TradeDirection(Enum):
    LONG  = "long"
    SHORT = "short"


@dataclass
class Trade:
    """Represents a single trade from open to close."""
    id: str                         # unique identifier
    direction: TradeDirection
    entry_price: float
    quantity: float                 # in base currency (e.g. BTC)
    leverage: int
    stop_loss: float                # absolute price
    take_profit: float              # absolute price
    status: TradeStatus = TradeStatus.OPEN
    exit_price: float | None = None
    pnl: float | None = None       # profit/loss in quote currency
    opened_at: float = field(default_factory=time.time)
    closed_at: float | None = None
    signal: TradingSignal | None = None
    exchange_order_id: str | None = None


# ---------------------------------------------------------------------------
# Risk & Sizing Engine
# ---------------------------------------------------------------------------

class RiskEngine:
    """
    Computes position size and leverage from a TradingSignal.

    Position sizing uses a simplified Kelly Criterion:
        kelly_fraction = (win_rate * avg_win) - (loss_rate * avg_loss)
                         -----------------------------------------------
                                        avg_win

    Since we don't have historical win/loss stats yet, we bootstrap
    by treating signal confidence as a proxy for win probability.
    Once you have live trade history, feed real stats in via update_stats().
    """

    def __init__(self, config: ExecutionConfig):
        self.config = config
        # Running stats — updated after each closed trade
        self.total_trades   = 0
        self.winning_trades = 0
        self.avg_win        = 0.0   # average PnL of winning trades
        self.avg_loss       = 0.0   # average PnL of losing trades (stored as positive)

    def update_stats(self, trade: Trade):
        """Call this after every trade closes to keep Kelly stats fresh."""
        self.total_trades += 1
        if trade.pnl and trade.pnl > 0:
            self.winning_trades += 1
            # Running average of wins
            self.avg_win = (
                (self.avg_win * (self.winning_trades - 1) + trade.pnl)
                / self.winning_trades
            )
        elif trade.pnl and trade.pnl < 0:
            losers = self.total_trades - self.winning_trades
            self.avg_loss = (
                (self.avg_loss * (losers - 1) + abs(trade.pnl))
                / losers
            )

    def _kelly_fraction(self, confidence: float) -> float:
        """
        Compute Kelly fraction.
        If no historical data yet, use signal confidence as win_rate
        and assume avg_win / avg_loss = 2:1 (common starting assumption).
        """
        if self.total_trades < 10 or self.avg_win == 0 or self.avg_loss == 0:
            # Bootstrap: use confidence as win_rate, assume 2:1 payoff ratio
            win_rate  = confidence
            loss_rate = 1 - confidence
            payoff_ratio = 2.0
            kelly = win_rate - (loss_rate / payoff_ratio)
        else:
            win_rate  = self.winning_trades / self.total_trades
            loss_rate = 1 - win_rate
            kelly = (win_rate * self.avg_win - loss_rate * self.avg_loss) / self.avg_win

        # Clamp: never negative, never above the cap
        return max(0.0, min(kelly, self.config.kelly_cap))

    def compute_leverage(self, confidence: float) -> int:
        """
        Map signal confidence [0, 1] linearly onto [min_leverage, max_leverage].
        Higher confidence → more leverage. Low confidence → stay conservative.
        """
        span = self.config.max_leverage - self.config.min_leverage
        leverage = self.config.min_leverage + int(span * confidence)
        return min(leverage, self.config.max_leverage)

    def size_position(self, signal: TradingSignal, current_price: float, available_capital: float) -> dict:
        """
        Given a signal and current market price, compute:
            - How much capital to allocate
            - How much base currency to buy/sell
            - What leverage to use
            - Where to place stop-loss and take-profit

        Returns a dict with all the parameters needed to open a trade.
        Returns None if the signal doesn't justify a trade.
        """
        if signal.direction == "HOLD":
            return None

        kelly = self._kelly_fraction(signal.confidence)
        if kelly <= 0:
            logger.info("Kelly fraction <= 0 — skipping trade.")
            return None

        # Capital to risk on this trade (capped by max_risk_per_trade)
        risk_cap    = available_capital * self.config.max_risk_per_trade
        kelly_alloc = available_capital * kelly
        capital_to_use = min(kelly_alloc, risk_cap)

        # Leverage
        leverage = self.compute_leverage(signal.confidence)

        # Notional position size (capital * leverage)
        notional = capital_to_use * leverage

        # Quantity in base currency
        quantity = notional / current_price

        # Stop-loss & take-profit prices
        is_long = signal.direction == "BUY"
        stop_loss = (
            current_price * (1 - self.config.stop_loss_pct) if is_long
            else current_price * (1 + self.config.stop_loss_pct)
        )
        take_profit = (
            current_price * (1 + self.config.take_profit_pct) if is_long
            else current_price * (1 - self.config.take_profit_pct)
        )

        return {
            "direction":    TradeDirection.LONG if is_long else TradeDirection.SHORT,
            "capital":      capital_to_use,
            "notional":     notional,
            "quantity":     quantity,
            "leverage":     leverage,
            "entry_price":  current_price,
            "stop_loss":    stop_loss,
            "take_profit":  take_profit,
        }


# ---------------------------------------------------------------------------
# Exchange Client (ccxt wrapper with dry-run support)
# ---------------------------------------------------------------------------

class ExchangeClient:
    """
    Thin wrapper around ccxt. All methods check dry_run first —
    in dry-run mode, no real API calls are made for order placement.
    Market data (price fetching) is always live even in dry-run.
    """

    def __init__(self, config: ExecutionConfig):
        self.config = config
        exchange_class = getattr(ccxt, config.exchange_name, None)
        if exchange_class is None:
            raise ValueError(f"Unsupported exchange: {config.exchange_name}")

        self.exchange = exchange_class({
            "apiKey":   config.api_key,
            "secret":   config.api_secret,
            "enableRateLimit": True,
        })

        # Enable sandbox/testnet if available and in dry-run
        if config.dry_run and hasattr(self.exchange, "set_sandbox_mode"):
            try:
                self.exchange.set_sandbox_mode(True)
                logger.info("Exchange sandbox mode enabled.")
            except Exception:
                logger.warning("Sandbox mode not available — will simulate orders locally.")

        logger.info(f"Exchange client initialized: {config.exchange_name} | dry_run={config.dry_run}")

    def get_price(self, pair: str) -> float:
        """Fetch the current market price. Always live, even in dry-run."""
        try:
            ticker = self.exchange.fetch_ticker(pair)
            return float(ticker["last"])
        except Exception as e:
            logger.error(f"Failed to fetch price for {pair}: {e}")
            raise

    def open_position(self, trade: Trade) -> str | None:
        """
        Place an order on the exchange.
        In dry-run mode, logs the order and returns a fake order ID.
        Returns the exchange order ID on success, None on failure.
        """
        side = "buy" if trade.direction == TradeDirection.LONG else "sell"

        if self.config.dry_run:
            fake_id = f"dry_{int(time.time() * 1000)}"
            logger.info(
                f"[DRY RUN] Would open {side} | qty={trade.quantity:.6f} "
                f"| leverage={trade.leverage}x | entry={trade.entry_price:.2f} "
                f"| SL={trade.stop_loss:.2f} | TP={trade.take_profit:.2f}"
            )
            return fake_id

        # --- LIVE MODE ---
        try:
            # Place the market order
            order = self.exchange.create_order(
                symbol   = self.config.trading_pair,
                type     = "market",
                side     = side,
                amount   = trade.quantity,
                params   = {"leverage": trade.leverage},
            )
            order_id = order["id"]
            logger.info(f"[LIVE] Order placed: {order_id} | side={side} | qty={trade.quantity:.6f}")

            # Place stop-loss order
            self.exchange.create_order(
                symbol = self.config.trading_pair,
                type   = "stop",
                side   = "sell" if side == "buy" else "buy",
                amount = trade.quantity,
                price  = trade.stop_loss,
                params = {"stopPrice": trade.stop_loss, "reduceOnly": True},
            )
            logger.info(f"[LIVE] Stop-loss set at {trade.stop_loss:.2f}")

            # Place take-profit order
            self.exchange.create_order(
                symbol = self.config.trading_pair,
                type   = "limit",
                side   = "sell" if side == "buy" else "buy",
                amount = trade.quantity,
                price  = trade.take_profit,
                params = {"reduceOnly": True},
            )
            logger.info(f"[LIVE] Take-profit set at {trade.take_profit:.2f}")

            return order_id

        except Exception as e:
            logger.error(f"[LIVE] Failed to open position: {e}")
            return None

    def close_position(self, trade: Trade) -> bool:
        """
        Close an open position at market price.
        In dry-run mode, simulates the close.
        """
        side = "sell" if trade.direction == TradeDirection.LONG else "buy"

        if self.config.dry_run:
            logger.info(f"[DRY RUN] Would close position {trade.id} at market.")
            return True

        try:
            self.exchange.create_order(
                symbol = self.config.trading_pair,
                type   = "market",
                side   = side,
                amount = trade.quantity,
                params = {"reduceOnly": True},
            )
            logger.info(f"[LIVE] Position {trade.id} closed.")
            return True
        except Exception as e:
            logger.error(f"[LIVE] Failed to close position {trade.id}: {e}")
            return False


# ---------------------------------------------------------------------------
# Trade Lifecycle Manager
# ---------------------------------------------------------------------------

class TradeManager:
    """
    Owns the full lifecycle of every trade: open → monitor → close.
    Also enforces the kill switch (max drawdown) and cooldown timer.
    """

    def __init__(self, config: ExecutionConfig):
        self.config     = config
        self.risk       = RiskEngine(config)
        self.exchange   = ExchangeClient(config)
        self.open_trades: list[Trade] = []
        self.closed_trades: list[Trade] = []
        self.capital    = config.total_capital
        self.starting_capital = config.total_capital
        self.last_trade_time  = 0.0          # epoch of last opened trade
        self._trade_counter   = 0

    # --- Internal helpers ---

    def _next_id(self) -> str:
        self._trade_counter += 1
        return f"trade_{self._trade_counter:06d}"

    def _current_drawdown(self) -> float:
        """Drawdown as a fraction (0.0 to 1.0) from starting capital."""
        if self.starting_capital == 0:
            return 0.0
        return (self.starting_capital - self.capital) / self.starting_capital

    def _is_cooldown_active(self) -> bool:
        return (time.time() - self.last_trade_time) < self.config.cooldown_seconds

    def _compute_pnl(self, trade: Trade, exit_price: float) -> float:
        """
        Compute PnL for a trade.
        PnL = quantity * (exit - entry) * leverage   (for long)
        PnL = quantity * (entry - exit) * leverage   (for short)
        """
        if trade.direction == TradeDirection.LONG:
            return trade.quantity * (exit_price - trade.entry_price) * trade.leverage
        else:
            return trade.quantity * (trade.entry_price - exit_price) * trade.leverage

    # --- Public interface ---

    def handle_signal(self, signal: TradingSignal) -> Trade | None:
        """
        Main entry point. Takes a TradingSignal, decides whether to trade,
        and if so opens a position. Returns the Trade object or None.
        """
        # --- Guard rails ---
        if signal.direction == "HOLD":
            logger.info("Signal is HOLD — no action.")
            return None

        if self._current_drawdown() >= self.config.max_drawdown_pct:
            logger.warning("KILL SWITCH ACTIVE — max drawdown reached. No new trades.")
            return None

        if self._is_cooldown_active():
            remaining = self.config.cooldown_seconds - (time.time() - self.last_trade_time)
            logger.info(f"Cooldown active — {remaining:.0f}s remaining. Skipping.")
            return None

        if len(self.open_trades) > 0:
            logger.info("Already have an open position — skipping to avoid over-exposure.")
            return None

        # --- Get current price ---
        try:
            current_price = self.exchange.get_price(self.config.trading_pair)
        except Exception as e:
            logger.error(f"Could not fetch price: {e}")
            return None

        # --- Size the position ---
        sizing = self.risk.size_position(signal, current_price, self.capital)
        if sizing is None:
            logger.info("Risk engine declined to size a position.")
            return None

        # --- Build the Trade object ---
        trade = Trade(
            id             = self._next_id(),
            direction      = sizing["direction"],
            entry_price    = sizing["entry_price"],
            quantity       = sizing["quantity"],
            leverage       = sizing["leverage"],
            stop_loss      = sizing["stop_loss"],
            take_profit    = sizing["take_profit"],
            signal         = signal,
        )

        # --- Place the order ---
        order_id = self.exchange.open_position(trade)
        if order_id is None:
            trade.status = TradeStatus.FAILED
            logger.error(f"Trade {trade.id} failed to open.")
            return trade

        trade.exchange_order_id = order_id
        self.open_trades.append(trade)
        self.last_trade_time = time.time()

        logger.info(
            f"Trade {trade.id} opened | {sizing['direction'].value.upper()} "
            f"| {sizing['quantity']:.6f} {self.config.trading_pair.split('/')[0]} "
            f"| {sizing['leverage']}x leverage | capital at risk: {sizing['capital']:.2f}"
        )
        return trade

    def monitor_positions(self) -> list[Trade]:
        """
        Check all open positions against current price.
        Closes any that have hit stop-loss or take-profit.
        Call this on a regular interval (e.g. every 10 seconds).
        Returns a list of any trades that were closed this cycle.
        """
        closed_this_cycle = []

        if not self.open_trades:
            return closed_this_cycle

        try:
            current_price = self.exchange.get_price(self.config.trading_pair)
        except Exception as e:
            logger.error(f"Could not fetch price during monitoring: {e}")
            return closed_this_cycle

        for trade in list(self.open_trades):  # iterate over a copy
            hit_sl = False
            hit_tp = False

            if trade.direction == TradeDirection.LONG:
                hit_sl = current_price <= trade.stop_loss
                hit_tp = current_price >= trade.take_profit
            else:
                hit_sl = current_price >= trade.stop_loss
                hit_tp = current_price <= trade.take_profit

            if hit_sl or hit_tp:
                reason = "STOP-LOSS" if hit_sl else "TAKE-PROFIT"
                self._close_trade(trade, current_price, reason)
                closed_this_cycle.append(trade)

        return closed_this_cycle

    def _close_trade(self, trade: Trade, exit_price: float, reason: str):
        """Internal: close a trade, update bookkeeping."""
        success = self.exchange.close_position(trade)

        trade.exit_price = exit_price
        trade.pnl        = self._compute_pnl(trade, exit_price)
        trade.status     = TradeStatus.CLOSED if success else TradeStatus.FAILED
        trade.closed_at  = time.time()

        # Update capital and stats
        if trade.pnl is not None:
            self.capital += trade.pnl
            self.risk.update_stats(trade)

        self.open_trades.remove(trade)
        self.closed_trades.append(trade)

        logger.info(
            f"Trade {trade.id} closed — reason: {reason} | "
            f"entry: {trade.entry_price:.2f} | exit: {exit_price:.2f} | "
            f"PnL: {trade.pnl:+.4f} | capital: {self.capital:.2f}"
        )

    def force_close_all(self):
        """Emergency: close every open position immediately."""
        logger.warning("FORCE CLOSING ALL POSITIONS")
        try:
            price = self.exchange.get_price(self.config.trading_pair)
        except Exception:
            price = 0.0
            logger.error("Could not fetch price for emergency close — PnL will be inaccurate.")

        for trade in list(self.open_trades):
            self._close_trade(trade, price, "FORCE_CLOSE")

    def status(self) -> dict:
        """Return a summary of the current state."""
        return {
            "capital":          self.capital,
            "starting_capital": self.starting_capital,
            "drawdown_pct":     self._current_drawdown(),
            "open_trades":      len(self.open_trades),
            "closed_trades":    len(self.closed_trades),
            "total_pnl":        self.capital - self.starting_capital,
            "kill_switch":      self._current_drawdown() >= self.config.max_drawdown_pct,
        }


# ---------------------------------------------------------------------------
# Demo / integration test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Config (dry-run by default) ---
    config = ExecutionConfig(
        exchange_name   = "binance",
        trading_pair    = "BTC/USDT",
        dry_run         = True,
        total_capital   = 1000.0,
        max_risk_per_trade = 0.05,
        min_leverage    = 2,
        max_leverage    = 10,
        stop_loss_pct   = 0.03,
        take_profit_pct = 0.06,
        cooldown_seconds = 5,  # short cooldown for demo
    )

    manager = TradeManager(config)

    # --- Simulate a sequence of signals ---
    mock_signals = [
        TradingSignal("HOLD",  0.3, "Sentiment shift is noise — meme-driven, no real catalyst.",     -0.02, time.time()),
        TradingSignal("BUY",   0.82, "Major exchange listing confirmed. Institutional inflows detected. Bullish.", 0.38, time.time()),
        TradingSignal("BUY",   0.6, "Continued positive sentiment but no new catalyst.",              0.12, time.time()),  # should be blocked by cooldown / open position
        TradingSignal("SELL",  0.75, "Regulatory crackdown announced. Sharp sentiment reversal.",    -0.41, time.time()),
    ]

    print("\n" + "=" * 70)
    print("  EXECUTION LAYER — DRY RUN DEMO")
    print("=" * 70 + "\n")

    for i, signal in enumerate(mock_signals):
        print(f"\n--- Signal {i+1}: {signal.direction} (confidence={signal.confidence}) ---")
        print(f"    Reasoning: {signal.reasoning}\n")

        trade = manager.handle_signal(signal)

        if trade:
            print(f"    ✓ Trade opened: {trade.id}")
            print(f"      Direction:    {trade.direction.value.upper()}")
            print(f"      Leverage:     {trade.leverage}x")
            print(f"      Quantity:     {trade.quantity:.6f}")
            print(f"      Stop-loss:    {trade.stop_loss:.2f}")
            print(f"      Take-profit:  {trade.take_profit:.2f}")

            # Simulate time passing so cooldown expires for next signal
            time.sleep(config.cooldown_seconds + 1)

            # Simulate a price move that triggers stop-loss or take-profit
            if trade.direction == TradeDirection.LONG:
                sim_price = trade.take_profit  # pretend price hit TP
            else:
                sim_price = trade.take_profit  # for shorts, TP is below entry

            # Manually trigger close for demo (normally monitor_positions does this)
            manager._close_trade(trade, sim_price, "DEMO_TP_HIT")
            print(f"    ✓ Trade closed: PnL = {trade.pnl:+.4f}")

    print(f"\n{'=' * 70}")
    print("  FINAL STATUS")
    print(f"{'=' * 70}")
    status = manager.status()
    for k, v in status.items():
        print(f"    {k:.<25} {v}")
    print(f"{'=' * 70}\n")
