"""
Crypto Sentiment Trading — Data Ingestion Layer
=================================================
Pulls live text data from Twitter/X, Reddit, and NewsAPI,
deduplicates and normalizes it, then feeds RawMessage objects
into the signal pipeline (signal_layer.py), which in turn
feeds TradingSignals into the execution layer (execution_layer.py).

This file also contains the main event loop that ties all three
layers together into a single running bot.

Dependencies:
    pip install tweepy praw requests python-dotenv

Setup:
    Add to your .env file:
        # Twitter / X (v2 API — requires a developer account)
        TWITTER_BEARER_TOKEN=your_bearer_token

        # Reddit (register an app at reddit.com/prefs/apps)
        REDDIT_CLIENT_ID=your_client_id
        REDDIT_CLIENT_SECRET=your_client_secret
        REDDIT_USER_AGENT=your_app_name/1.0

        # NewsAPI (free tier at newsapi.org — 100 req/day)
        NEWSAPI_KEY=your_api_key

        # From previous layers
        ANTHROPIC_API_KEY=your_key
        EXCHANGE_API_KEY=your_key
        EXCHANGE_API_SECRET=your_secret
        EXCHANGE_NAME=binance
        TRADING_PAIR=BTC/USDT
        DRY_RUN=true
"""

import os
import time
import hashlib
import logging
import asyncio
from dataclasses import dataclass, field
from collections import deque
from dotenv import load_dotenv
import requests

# ---------------------------------------------------------------------------
# Conditional imports — each source is optional.
# The bot runs with whatever sources are configured; missing API keys
# just disable that source with a warning.
# ---------------------------------------------------------------------------
try:
    import tweepy
    TWEEPY_AVAILABLE = True
except ImportError:
    TWEEPY_AVAILABLE = False

try:
    import praw
    PRAW_AVAILABLE = True
except ImportError:
    PRAW_AVAILABLE = False

# Import from sibling modules
try:
    from signal_layer import RawMessage, SentimentPipeline
except ImportError:
    # Fallback dataclass for standalone testing
    @dataclass
    class RawMessage:
        text: str
        source: str
        timestamp: float
        source_weight: float = 1.0

    class SentimentPipeline:
        def process(self, msg):
            return None

try:
    from execution_layer import TradeManager, ExecutionConfig
except ImportError:
    TradeManager = None
    ExecutionConfig = None

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Source configuration & keyword targeting
# ---------------------------------------------------------------------------

# These are the keywords/hashtags the scrapers will target.
# Customize per the asset you're trading.
CRYPTO_KEYWORDS = [
    "bitcoin", "BTC", "ethereum", "ETH", "crypto", "cryptocurrency",
    "altcoin", "DeFi", "web3", "blockchain",
]

# Subreddits to monitor
TARGET_SUBREDDITS = [
    "bitcoin", "ethereum", "cryptocurrency", "CryptoMarkets",
    "altcoin", "defi", "Web3",
]

# Source weights — how much each platform's data counts in sentiment scoring.
# News articles carry more signal than a random tweet.
SOURCE_WEIGHTS = {
    "twitter": 1.0,
    "reddit":  1.2,   # Reddit tends to have longer, more analytical posts
    "news":    2.0,   # Verified news sources are highest weight
}


# ---------------------------------------------------------------------------
# Deduplication filter
# ---------------------------------------------------------------------------

class DeduplicationFilter:
    """
    Prevents the same piece of content from being processed twice.
    Uses a hash of the normalized text + source as a fingerprint.
    Also catches near-duplicates by checking if a new message's text
    is a substring of (or contains) any recent message from the same source.

    window_size: how many recent messages to keep in memory for comparison.
    """

    def __init__(self, window_size: int = 500):
        self.seen_hashes: set[str] = set()
        self.recent_texts: deque[tuple[str, str]] = deque(maxlen=window_size)  # (source, text)

    def _fingerprint(self, text: str, source: str) -> str:
        normalized = text.strip().lower()
        return hashlib.sha256(f"{source}:{normalized}".encode()).hexdigest()

    def is_duplicate(self, text: str, source: str) -> bool:
        fp = self._fingerprint(text, source)

        # Exact duplicate
        if fp in self.seen_hashes:
            return True

        # Near-duplicate check: is this text contained in a recent message
        # from the same source (or vice versa)?
        normalized = text.strip().lower()
        for prev_source, prev_text in self.recent_texts:
            if prev_source != source:
                continue
            if normalized in prev_text or prev_text in normalized:
                return True

        return False

    def register(self, text: str, source: str):
        """Call this after accepting a message to add it to the seen set."""
        fp = self._fingerprint(text, source)
        self.seen_hashes.add(fp)
        self.recent_texts.append((source, text.strip().lower()))


# ---------------------------------------------------------------------------
# Source: Twitter / X
# ---------------------------------------------------------------------------

class TwitterSource:
    """
    Streams tweets matching crypto keywords using the Twitter v2 API.
    Requires a Bearer Token (available on free tier, but with strict
    rate limits — 1 request per 15 minutes on free, more on paid tiers).

    Because of rate limits, this uses polling rather than streaming
    on the free tier. On paid tiers, swap to tweepy.StreamingClient
    for true real-time streaming.
    """

    def __init__(self):
        token = os.getenv("TWITTER_BEARER_TOKEN")
        if not token:
            raise ValueError("TWITTER_BEARER_TOKEN not set.")
        if not TWEEPY_AVAILABLE:
            raise ImportError("tweepy not installed. Run: pip install tweepy")

        self.client = tweepy.Client(bearer_token=token, wait_on_rate_limit=True)
        self.weight = SOURCE_WEIGHTS["twitter"]
        # Twitter free tier: ~1 req/15 min. Paid tiers allow more.
        self.poll_interval = 900  # seconds (15 min) — reduce if on a paid tier
        logger.info("TwitterSource initialized.")

    async def fetch(self) -> list[RawMessage]:
        """
        Poll for recent tweets matching crypto keywords.
        Returns a list of RawMessage objects.
        """
        query = " OR ".join(CRYPTO_KEYWORDS) + " lang:en -is:retweet"
        messages = []

        try:
            # Run the blocking tweepy call in a thread so it doesn't block the event loop
            loop = asyncio.get_event_loop()
            tweets = await loop.run_in_executor(
                None,
                lambda: self.client.search_recent_tweets(
                    query=query,
                    max_results=100,
                    tweet_fields=["created_at", "public_metrics"],
                )
            )

            if tweets.data is None:
                return messages

            for tweet in tweets.data:
                # Filter out very low-engagement tweets (likely noise)
                metrics = tweet.public_metrics or {}
                engagement = metrics.get("like_count", 0) + metrics.get("retweet_count", 0)
                if engagement < 2:
                    continue

                messages.append(RawMessage(
                    text=tweet.text,
                    source="twitter",
                    timestamp=tweet.created_at.timestamp() if tweet.created_at else time.time(),
                    source_weight=self.weight,
                ))

            logger.info(f"Twitter: fetched {len(messages)} messages.")
            return messages

        except Exception as e:
            logger.warning(f"Twitter fetch failed: {e}")
            return messages


# ---------------------------------------------------------------------------
# Source: Reddit
# ---------------------------------------------------------------------------

class RedditSource:
    """
    Polls specified crypto subreddits for new posts and comments.
    Uses PRAW (Python Reddit API Wrapper).
    Reddit's API is fairly generous — 60 requests/min authenticated.
    """

    def __init__(self):
        client_id     = os.getenv("REDDIT_CLIENT_ID")
        client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        user_agent    = os.getenv("REDDIT_USER_AGENT", "CryptoSentimentBot/1.0")

        if not client_id or not client_secret:
            raise ValueError("REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET not set.")
        if not PRAW_AVAILABLE:
            raise ImportError("praw not installed. Run: pip install praw")

        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
        )
        self.weight = SOURCE_WEIGHTS["reddit"]
        self.poll_interval = 60  # seconds
        self.seen_post_ids: set[str] = set()
        logger.info("RedditSource initialized.")

    async def fetch(self) -> list[RawMessage]:
        """
        Fetch new posts from target subreddits.
        Skips posts we've already seen.
        """
        messages = []
        loop = asyncio.get_event_loop()

        for sub_name in TARGET_SUBREDDITS:
            try:
                subreddit = await loop.run_in_executor(None, lambda sn=sub_name: self.reddit.subreddit(sn))
                hot_posts = await loop.run_in_executor(None, lambda sr=subreddit: list(sr.hot(limit=25)))

                for post in hot_posts:
                    if post.id in self.seen_post_ids:
                        continue
                    self.seen_post_ids.add(post.id)

                    # Skip very low-engagement posts
                    if post.score < 5 and post.num_comments < 3:
                        continue

                    # Use title + selftext for content
                    text = post.title
                    if post.is_self and post.selftext:
                        text += " " + post.selftext[:500]  # cap selftext length

                    messages.append(RawMessage(
                        text=text.strip(),
                        source="reddit",
                        timestamp=post.created_utc,
                        source_weight=self.weight,
                    ))

            except Exception as e:
                logger.warning(f"Reddit fetch failed for r/{sub_name}: {e}")

        # Trim seen_post_ids to prevent unbounded growth
        if len(self.seen_post_ids) > 5000:
            self.seen_post_ids = set(list(self.seen_post_ids)[-2500:])

        logger.info(f"Reddit: fetched {len(messages)} messages.")
        return messages


# ---------------------------------------------------------------------------
# Source: NewsAPI
# ---------------------------------------------------------------------------

class NewsAPISource:
    """
    Fetches crypto news headlines and descriptions from NewsAPI.
    Free tier: 100 requests/day. Use sparingly — poll every 10 minutes.

    This is often the highest-signal source because verified news
    outlets are weighted 2x in the sentiment scoring.
    """

    def __init__(self):
        self.api_key = os.getenv("NEWSAPI_KEY")
        if not self.api_key:
            raise ValueError("NEWSAPI_KEY not set.")

        self.weight = SOURCE_WEIGHTS["news"]
        self.poll_interval = 600  # 10 minutes (conserves free-tier quota)
        self.seen_urls: set[str] = set()
        logger.info("NewsAPISource initialized.")

    async def fetch(self) -> list[RawMessage]:
        """
        Fetch top crypto news articles. Uses title + description as text.
        """
        messages = []
        query = " OR ".join(CRYPTO_KEYWORDS[:5])  # NewsAPI free tier limits query length

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.get(
                    "https://newsapi.org/v2/everything",
                    params={
                        "q":          query,
                        "language":   "en",
                        "sortBy":     "publishedAt",
                        "pageSize":   20,
                        "apiKey":     self.api_key,
                    },
                    timeout=10,
                )
            )
            response.raise_for_status()
            data = response.json()

            for article in data.get("articles", []):
                url = article.get("url", "")
                if url in self.seen_urls:
                    continue
                self.seen_urls.add(url)

                title       = article.get("title", "")
                description = article.get("description", "")
                text        = f"{title}. {description}".strip() if description else title

                if not text or text == "[Removed]":
                    continue

                # Parse publishedAt (ISO 8601)
                pub_date = article.get("publishedAt")
                if pub_date:
                    from datetime import datetime, timezone
                    ts = datetime.fromisoformat(pub_date.replace("Z", "+00:00")).timestamp()
                else:
                    ts = time.time()

                messages.append(RawMessage(
                    text=text,
                    source="news",
                    timestamp=ts,
                    source_weight=self.weight,
                ))

            # Trim seen_urls
            if len(self.seen_urls) > 2000:
                self.seen_urls = set(list(self.seen_urls)[-1000:])

            logger.info(f"NewsAPI: fetched {len(messages)} messages.")
            return messages

        except Exception as e:
            logger.warning(f"NewsAPI fetch failed: {e}")
            return messages


# ---------------------------------------------------------------------------
# Source Manager — orchestrates all sources on their own schedules
# ---------------------------------------------------------------------------

class SourceManager:
    """
    Manages multiple data sources, each with its own poll interval.
    Tracks when each source was last polled and only fetches when
    the interval has elapsed. This respects rate limits per-source
    without requiring a separate timer per source.
    """

    def __init__(self):
        self.sources: list[tuple[object, int]] = []   # (source_instance, poll_interval_seconds)
        self.last_polled: dict[str, float] = {}       # source_name -> last poll timestamp
        self._register_sources()

    def _register_sources(self):
        """Try to initialize each source. Skip gracefully if API keys are missing."""
        source_classes = [
            ("twitter", TwitterSource),
            ("reddit",  RedditSource),
            ("news",    NewsAPISource),
        ]
        for name, cls in source_classes:
            try:
                source = cls()
                self.sources.append((name, source))
                self.last_polled[name] = 0.0
                logger.info(f"Source registered: {name}")
            except (ValueError, ImportError) as e:
                logger.warning(f"Source '{name}' not available: {e}")

        if not self.sources:
            raise RuntimeError(
                "No data sources could be initialized. "
                "Check that at least one set of API credentials is in your .env file."
            )

    async def collect(self) -> list[RawMessage]:
        """
        Poll all sources that are due for a refresh.
        Returns the combined list of new RawMessages.
        """
        all_messages = []
        now = time.time()

        for name, source in self.sources:
            interval = source.poll_interval
            if (now - self.last_polled[name]) < interval:
                continue  # not time yet

            messages = await source.fetch()
            all_messages.extend(messages)
            self.last_polled[name] = now

        return all_messages


# ---------------------------------------------------------------------------
# Main Event Loop — ties all three layers together
# ---------------------------------------------------------------------------

async def run_bot(dry_run: bool = True):
    """
    The main async event loop.

    Every iteration:
        1. SourceManager polls any sources that are due.
        2. New messages are deduplicated.
        3. Each unique message is fed into the SentimentPipeline.
        4. If the pipeline produces a TradingSignal, it's handed to the TradeManager.
        5. TradeManager.monitor_positions() checks open trades against current price.

    The loop runs forever until interrupted (Ctrl+C).
    """

    logger.info("=" * 60)
    logger.info("  CRYPTO SENTIMENT TRADING BOT — STARTING")
    logger.info(f"  Mode: {'DRY RUN' if dry_run else 'LIVE TRADING'}")
    logger.info("=" * 60)

    # --- Initialize all layers ---
    source_manager = SourceManager()
    dedup_filter   = DeduplicationFilter(window_size=500)
    signal_pipeline = SentimentPipeline()

    # Execution layer
    if TradeManager and ExecutionConfig:
        exec_config = ExecutionConfig(dry_run=dry_run)
        trade_manager = TradeManager(exec_config)
        logger.info("Execution layer initialized.")
    else:
        trade_manager = None
        logger.warning("Execution layer not available — signals will be logged but not traded.")

    # --- Counters for runtime stats ---
    stats = {
        "total_fetched":     0,
        "total_duplicates":  0,
        "total_processed":   0,
        "total_signals":     0,
        "total_trades":      0,
    }

    # --- Main loop ---
    monitor_interval = 10  # seconds between position checks
    last_monitor_time = 0.0

    while True:
        now = time.time()

        # --- 1. Fetch new messages ---
        raw_messages = await source_manager.collect()
        stats["total_fetched"] += len(raw_messages)

        # --- 2. Deduplicate ---
        unique_messages = []
        for msg in raw_messages:
            if dedup_filter.is_duplicate(msg.text, msg.source):
                stats["total_duplicates"] += 1
                continue
            dedup_filter.register(msg.text, msg.source)
            unique_messages.append(msg)

        # --- 3 & 4. Signal pipeline + execution ---
        for msg in unique_messages:
            stats["total_processed"] += 1
            signal = signal_pipeline.process(msg)

            if signal is None:
                continue

            stats["total_signals"] += 1
            logger.info(
                f"Signal #{stats['total_signals']}: {signal.direction} "
                f"(conf={signal.confidence:.2f}) — {signal.reasoning}"
            )

            if trade_manager:
                trade = trade_manager.handle_signal(signal)
                if trade and trade.status.value == "open":
                    stats["total_trades"] += 1

        # --- 5. Monitor open positions ---
        if trade_manager and (now - last_monitor_time) >= monitor_interval:
            trade_manager.monitor_positions()
            last_monitor_time = now

            # Check kill switch
            status = trade_manager.status()
            if status["kill_switch"]:
                logger.critical("KILL SWITCH TRIGGERED — forcing close of all positions.")
                trade_manager.force_close_all()
                logger.critical("All positions closed. Bot will continue monitoring but will not open new trades.")

        # --- Log periodic status ---
        if stats["total_processed"] % 100 == 0 and stats["total_processed"] > 0:
            logger.info(f"Stats: {stats}")
            if trade_manager:
                logger.info(f"Portfolio: {trade_manager.status()}")

        # --- Sleep briefly to avoid a tight loop when no sources are due ---
        await asyncio.sleep(2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Crypto Sentiment Trading Bot")
    parser.add_argument(
        "--live", action="store_true",
        help="Run in LIVE mode (real trades). Omit for dry-run (default).",
    )
    args = parser.parse_args()

    dry_run = not args.live

    if not dry_run:
        # Safety confirmation for live mode
        confirm = input("\n⚠️  LIVE TRADING MODE — real money will be used.\nType 'yes' to confirm: ").strip()
        if confirm != "yes":
            print("Aborted.")
            exit(0)

    try:
        asyncio.run(run_bot(dry_run=dry_run))
    except KeyboardInterrupt:
        logger.info("Bot stopped by user.")
