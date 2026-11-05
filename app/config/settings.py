"""Central configuration loaded from environment / .env.

Risk limits live here and are read once at start-up. The LLM layer has no write path to them.
"""

from __future__ import annotations

from decimal import Decimal
from enum import Enum
from functools import lru_cache

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.errors import ConfigError
from app.core.types import TradingMode


class RiskSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RISK_", env_file=".env", extra="ignore")

    max_daily_drawdown_pct: Decimal = Field(Decimal("0.03"), gt=0, le=Decimal("0.5"))
    max_leverage: int = Field(5, ge=1, le=20)
    max_risk_per_trade_pct: Decimal = Field(Decimal("0.005"), gt=0, le=Decimal("0.05"))  # equity fraction lost if SL hit
    max_position_notional_pct: Decimal = Field(Decimal("0.5"), gt=0, le=Decimal("5"))  # notional / equity, per position
    max_total_exposure_pct: Decimal = Field(Decimal("1.5"), gt=0, le=Decimal("20"))  # sum notional / equity
    max_open_positions: int = Field(3, ge=1, le=50)
    min_fee_cover_multiple: Decimal = Field(Decimal("2"), ge=1)  # expected gross profit must exceed fees * this
    min_net_profit_usd: Decimal = Field(Decimal("0.01"), ge=0)
    default_slippage_bps: Decimal = Field(Decimal("2"), ge=0)  # 2 bps = 0.02% per fill
    max_margin_ratio: Decimal = Field(Decimal("0.8"), gt=0, lt=1)  # watchdog flattens above this
    max_stop_distance_pct: Decimal = Field(Decimal("0.05"), gt=0, le=Decimal("0.5"))
    min_stop_distance_pct: Decimal = Field(Decimal("0.001"), ge=0)
    trailing_activation_r: Decimal = Field(Decimal("1"), gt=0)  # move SL to breakeven after +1R
    watchdog_interval_seconds: float = Field(1.0, gt=0, le=60)
    max_consecutive_losses: int = Field(5, ge=1)
    blocked_symbols: list[str] = Field(default_factory=list)


class DataSettings(BaseSettings):
    """Market-data ingestion. Every buffer is bounded (24/7 memory safety)."""

    model_config = SettingsConfigDict(env_prefix="DATA_", env_file=".env", extra="ignore")

    timeframe: str = "1m"
    stale_after_seconds: float = Field(2.0, gt=0, le=60)  # rule 1: >2s without update -> STALE -> PAUSE
    health_check_interval_seconds: float = Field(0.25, gt=0, le=5)
    candle_buffer_size: int = Field(1000, ge=100, le=20_000)  # rule 2: deque(maxlen)
    orderbook_depth: int = Field(20, ge=5, le=500)
    liquidation_buffer_size: int = Field(500, ge=10, le=10_000)
    reconnect_base_delay_seconds: float = Field(0.5, gt=0)
    reconnect_max_delay_seconds: float = Field(30.0, gt=0)
    reconnect_jitter: float = Field(0.2, ge=0, le=1)
    resume_after_live_seconds: float = Field(1.0, ge=0)  # stay LIVE this long before un-pausing
    bootstrap_candles: int = Field(200, ge=0, le=1500)  # REST backfill at start so indicators are warm


class QuantSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="QUANT_", env_file=".env", extra="ignore")

    rsi_period: int = 14
    ema_fast: int = 20
    ema_slow: int = 50
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    volume_z_period: int = 20
    obi_levels: int = Field(10, ge=1, le=100)
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    obi_threshold: float = Field(0.25, gt=0, lt=1)
    min_confluence_score: float = Field(0.7, gt=0, le=1)  # rule 3: only high-probability setups
    atr_stop_multiple: Decimal = Field(Decimal("1.5"), gt=0)
    atr_target_multiple: Decimal = Field(Decimal("3.0"), gt=0)
    max_spread_bps: float = Field(5.0, gt=0)
    reference_notional_usd: Decimal = Field(Decimal("1000"), gt=0)  # notional used for the fee pre-check
    trigger_cooldown_seconds: float = Field(300, ge=0)  # per symbol+direction
    max_triggers_per_hour: int = Field(12, ge=1)  # global Claude cost cap
    expected_holding_hours: Decimal = Field(Decimal("2"), gt=0)
    news_lookback_minutes: int = Field(120, ge=1)
    news_in_payload: int = Field(8, ge=0, le=50)


class NewsSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NEWS_", env_file=".env", extra="ignore")

    poll_interval_seconds: float = Field(30, ge=5)
    request_timeout_seconds: float = Field(10, gt=0)
    cache_max_items: int = Field(2000, ge=100, le=100_000)  # rule 2/4: bounded in-memory cache
    cache_ttl_hours: float = Field(24, gt=0)
    dedup_window: int = Field(20_000, ge=1000)  # bounded set of seen hashes
    title_similarity_threshold: float = Field(0.85, gt=0, le=1)  # near-duplicate across sources
    max_age_hours: float = Field(12, gt=0)  # ignore stale articles on startup
    cryptopanic_token: SecretStr = SecretStr("")
    x_bearer_token: SecretStr = SecretStr("")
    x_accounts: list[str] = Field(default_factory=lambda: ["federalreserve", "SECGov", "whale_alert", "WatcherGuru"])
    rss_feeds: dict[str, str] = Field(
        default_factory=lambda: {
            "coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
            "cointelegraph": "https://cointelegraph.com/rss",
            "theblock": "https://www.theblock.co/rss.xml",
            "decrypt": "https://decrypt.co/feed",
            "sec": "https://www.sec.gov/news/pressreleases.rss",
            "fed": "https://www.federalreserve.gov/feeds/press_all.xml",
        }
    )
    symbol_keywords: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "BTC": ["bitcoin", "btc"],
            "ETH": ["ethereum", "ether", "eth"],
            "SOL": ["solana", "sol"],
            "BNB": ["bnb", "binance coin"],
            "XRP": ["xrp", "ripple"],
        }
    )
    macro_keywords: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "FED": ["fed", "fomc", "federal reserve", "powell", "rate cut", "rate hike", "interest rate"],
            "CPI": ["cpi", "inflation", "pce", "jobs report", "nonfarm", "payrolls"],
            "SEC": ["sec", "gensler", "atkins", "securities and exchange commission"],
            "ETF": ["etf", "spot etf", "etf inflow", "etf outflow"],
            "HACK": ["hack", "exploit", "drained", "stolen", "breach", "vulnerability"],
            "REGULATION": ["regulation", "lawsuit", "ban", "sanction", "stablecoin bill", "mica"],
            "EXCHANGE": ["binance", "coinbase", "bybit", "okx", "withdrawals halted", "insolvency"],
        }
    )


class ClaudeSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CLAUDE_", env_file=".env", extra="ignore")

    model: str = "claude-sonnet-4-5"  # set CLAUDE_MODEL to the model id you want; fast models suit the 6s budget
    timeout_seconds: float = Field(6.0, gt=0, le=60)  # rule 1: >6s -> HOLD
    max_output_tokens: int = Field(1200, ge=200, le=8000)
    temperature: float | None = Field(None, ge=0, le=1)  # CLAUDE_TEMPERATURE; unset = model default
    circuit_breaker_failures: int = Field(5, ge=1)  # consecutive failures before calls are suspended
    circuit_breaker_cooldown_seconds: float = Field(300, ge=1)
    max_news_chars: int = Field(600, ge=50, le=4000)  # per news item, after sanitising


class ExecutionSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EXEC_", env_file=".env", extra="ignore")

    max_drift_atr: Decimal = Field(Decimal("0.3"), gt=0)  # rule 3: adverse move since signal > 0.3 ATR -> abort
    max_abs_drift_atr: Decimal = Field(Decimal("1.0"), gt=0)  # any-direction jump this large -> abort (anomaly)
    max_book_age_seconds: float = Field(2.0, gt=0)
    limit_entry_timeout_seconds: float = Field(5.0, gt=0)
    entry_fill_poll_seconds: float = Field(0.2, gt=0)
    stop_place_retries: int = Field(3, ge=1, le=10)  # rule 4: 3 fast retries then emergency close
    stop_retry_backoff_seconds: float = Field(0.1, ge=0)
    emergency_close_retries: int = Field(3, ge=1, le=10)
    breakeven_trigger_fee_multiple: Decimal = Field(Decimal("2"), gt=0)  # rule 5: net >= 2x fees -> BE+fee
    breakeven_buffer_bps: Decimal = Field(Decimal("2"), ge=0)  # extra margin above BE+fee for spread at the fill
    trail_atr_multiple: Decimal = Field(Decimal("1.0"), gt=0)
    trail_min_step_atr: Decimal = Field(Decimal("0.1"), gt=0)  # don't spam the exchange with tiny moves
    trail_min_interval_seconds: float = Field(1.0, ge=0)
    manager_poll_seconds: float = Field(1.0, gt=0)


class OrphanAction(str, Enum):
    ADOPT_AND_PROTECT = "ADOPT_AND_PROTECT"  # take over an unmanaged exchange position and put a hard stop on it
    CLOSE = "CLOSE"  # flatten every unmanaged exchange position at market


class ReconciliationSettings(BaseSettings):
    """Exchange <-> local state reconciliation and the LIVE start-up guard thresholds."""

    model_config = SettingsConfigDict(env_prefix="RECON_", env_file=".env", extra="ignore")

    interval_seconds: float = Field(60.0, gt=0, le=3600)
    orphan_stop_pct: Decimal = Field(Decimal("0.02"), gt=0, lt=1)  # hard stop distance from entry for unmanaged positions
    orphan_action: OrphanAction = OrphanAction.ADOPT_AND_PROTECT
    max_orphan_notional_pct: Decimal = Field(Decimal("0.6"), gt=0, le=Decimal("20"))  # notional / equity above this -> close
    close_if_cross_margin: bool = True
    live_min_balance: Decimal = Field(Decimal("50"), ge=0)
    live_max_existing_leverage: int = Field(5, ge=1, le=125)


class ApiSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="API_", env_file=".env", extra="ignore")

    host: str = "127.0.0.1"  # local only by default; put a reverse proxy with TLS + auth in front for remote use
    port: int = Field(8000, ge=1, le=65535)
    control_token: SecretStr = SecretStr("")  # X-Control-Token for resume / manual close (REQUIRED in LIVE)
    active_symbol: str | None = None  # default chart symbol (first of SYMBOLS when unset)
    ws_max_clients: int = Field(20, ge=1, le=500)
    ws_send_queue: int = Field(500, ge=50, le=10_000)
    ws_snapshot_deadline_seconds: float = Field(0.08, gt=0, le=2)  # INITIAL_SNAPSHOT must go out < 100 ms
    ws_throttle_seconds: float = Field(0.2, ge=0.05, le=5)  # rule 3: 5 Hz
    allowed_origins: list[str] = Field(default_factory=lambda: ["http://127.0.0.1:8000", "http://localhost:8000"])


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    trading_mode: TradingMode = TradingMode.PAPER
    live_trading_confirmation: str = ""  # must equal LIVE_CONFIRMATION_PHRASE to run LIVE

    exchange_id: str = "binance"
    exchange_market: str = "PERPETUAL"  # SPOT | PERPETUAL
    exchange_testnet: bool = False  # derived from TRADING_MODE (TESTNET -> True); kept for backwards compatibility
    exchange_api_key: SecretStr = SecretStr("")
    exchange_api_secret: SecretStr = SecretStr("")
    exchange_api_password: SecretStr = SecretStr("")  # OKX passphrase etc.

    paper_starting_balance: Decimal = Decimal("10000")
    quote_currency: str = "USDT"
    symbols: list[str] = Field(default_factory=lambda: ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"])

    database_url: str = "postgresql+asyncpg://quant:quant@localhost:5432/quant"
    redis_url: str = "redis://localhost:6379/0"

    anthropic_api_key: SecretStr = SecretStr("")

    # TODO: Phase 2 - US Equities (Alpaca/IBKR)
    alpaca_api_key: SecretStr = SecretStr("")
    alpaca_api_secret: SecretStr = SecretStr("")
    alpaca_paper: bool = True

    risk: RiskSettings = Field(default_factory=RiskSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    quant: QuantSettings = Field(default_factory=QuantSettings)
    news: NewsSettings = Field(default_factory=NewsSettings)
    claude: ClaudeSettings = Field(default_factory=ClaudeSettings)
    execution: ExecutionSettings = Field(default_factory=ExecutionSettings)
    api: ApiSettings = Field(default_factory=ApiSettings)
    recon: ReconciliationSettings = Field(default_factory=ReconciliationSettings)

    @property
    def uses_exchange_testnet(self) -> bool:
        """Orders AND market data go to the testnet only in TESTNET mode. PAPER uses mainnet public data."""
        return self.trading_mode is TradingMode.TESTNET

    @model_validator(mode="after")
    def _live_guard(self) -> "Settings":
        if self.trading_mode is TradingMode.TESTNET:
            if not self.exchange_api_key.get_secret_value() or not self.exchange_api_secret.get_secret_value():
                raise ConfigError("TESTNET mode requires the exchange TESTNET API key and secret "
                                  "(EXCHANGE_API_KEY / EXCHANGE_API_SECRET created on the testnet site).")
            self.exchange_testnet = True
        if self.trading_mode is TradingMode.LIVE:
            if self.exchange_testnet:
                raise ConfigError("LIVE mode cannot run with EXCHANGE_TESTNET=true; use TRADING_MODE=TESTNET instead.")
            if self.live_trading_confirmation != LIVE_CONFIRMATION_PHRASE:
                raise ConfigError(
                    "TRADING_MODE=LIVE requires LIVE_TRADING_CONFIRMATION to be set to the exact confirmation phrase."
                )
            if not self.exchange_api_key.get_secret_value() or not self.exchange_api_secret.get_secret_value():
                raise ConfigError("LIVE mode requires EXCHANGE_API_KEY and EXCHANGE_API_SECRET.")
            if len(self.api.control_token.get_secret_value()) < 16:
                raise ConfigError("LIVE mode requires API_CONTROL_TOKEN (min 16 chars) to protect the control API.")
        return self


LIVE_CONFIRMATION_PHRASE = "I_UNDERSTAND_REAL_MONEY_IS_AT_RISK"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
