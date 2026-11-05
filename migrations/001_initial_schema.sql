-- 001_initial_schema.sql
-- Initial PostgreSQL 16 + TimescaleDB schema for quant-core.
-- Hand-written and idempotent; mirrors app/db/models.py (constraint and index names match
-- the SQLAlchemy naming convention). Money = NUMERIC(38,18), timestamps = TIMESTAMPTZ (UTC).
-- Enums are VARCHAR(32) + CHECK, matching SQLAlchemy Enum(native_enum=False).

-- TimescaleDB is optional: on plain PostgreSQL the tables below simply stay regular tables.
-- gen_random_uuid() is built in since PG13, so pgcrypto is not needed.
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_available_extensions WHERE name = 'timescaledb') THEN
        CREATE EXTENSION IF NOT EXISTS timescaledb;
    ELSE
        RAISE NOTICE 'timescaledb extension not available: time-series tables stay regular tables';
    END IF;
END
$$;

-- ---------------------------------------------------------------------------------------
-- Reference data
-- ---------------------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS accounts (
    id             SERIAL NOT NULL,
    broker         VARCHAR(32) NOT NULL,
    asset_class    VARCHAR(32) NOT NULL,
    market_type    VARCHAR(32) NOT NULL,
    trading_mode   VARCHAR(32) NOT NULL,
    quote_currency VARCHAR(16) NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT pk_accounts PRIMARY KEY (id),
    CONSTRAINT uq_accounts_broker_class_market_mode UNIQUE (broker, asset_class, market_type, trading_mode),
    -- US_EQUITY / EQUITY values are reserved for Phase 2 (Alpaca/IBKR)
    CONSTRAINT ck_accounts_asset_class CHECK (asset_class IN ('CRYPTO', 'US_EQUITY')),
    CONSTRAINT ck_accounts_market_type CHECK (market_type IN ('SPOT', 'PERPETUAL', 'EQUITY')),
    CONSTRAINT ck_accounts_trading_mode CHECK (trading_mode IN ('PAPER', 'TESTNET', 'LIVE'))
);

CREATE TABLE IF NOT EXISTS instruments (
    id           SERIAL NOT NULL,
    account_id   INTEGER NOT NULL,
    symbol       VARCHAR(64) NOT NULL,
    asset_class  VARCHAR(32) NOT NULL,
    market_type  VARCHAR(32) NOT NULL,
    base         VARCHAR(32) NOT NULL,
    quote        VARCHAR(32) NOT NULL,
    tick_size    NUMERIC(38, 18) NOT NULL,
    step_size    NUMERIC(38, 18) NOT NULL,
    min_qty      NUMERIC(38, 18) NOT NULL,
    min_notional NUMERIC(38, 18) NOT NULL,
    maker_fee    NUMERIC(38, 18) NOT NULL,
    taker_fee    NUMERIC(38, 18) NOT NULL,
    max_leverage INTEGER NOT NULL DEFAULT 1,
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT pk_instruments PRIMARY KEY (id),
    CONSTRAINT uq_instruments_account_symbol UNIQUE (account_id, symbol),
    CONSTRAINT ck_instruments_max_leverage_positive CHECK (max_leverage >= 1),
    CONSTRAINT fk_instruments_account_id_accounts FOREIGN KEY (account_id) REFERENCES accounts (id) ON DELETE CASCADE,
    CONSTRAINT ck_instruments_asset_class CHECK (asset_class IN ('CRYPTO', 'US_EQUITY')),
    CONSTRAINT ck_instruments_market_type CHECK (market_type IN ('SPOT', 'PERPETUAL', 'EQUITY'))
);

-- ---------------------------------------------------------------------------------------
-- Signals and Claude decisions
-- ---------------------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS signals (
    id         UUID NOT NULL DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    symbol     VARCHAR(64) NOT NULL,
    strategy   VARCHAR(64) NOT NULL,
    direction  VARCHAR(32) NOT NULL,
    timeframe  VARCHAR(16) NOT NULL,
    score      NUMERIC(18, 8) NOT NULL,
    features   JSONB NOT NULL,
    status     VARCHAR(32) NOT NULL DEFAULT 'NEW',
    CONSTRAINT pk_signals PRIMARY KEY (id),
    CONSTRAINT ck_signals_direction CHECK (direction IN ('LONG', 'SHORT')),
    CONSTRAINT ck_signals_status CHECK (status IN ('NEW', 'SENT_TO_CLAUDE', 'ACTED', 'DISCARDED', 'EXPIRED'))
);
CREATE INDEX IF NOT EXISTS ix_signals_symbol_created_at ON signals (symbol, created_at);

CREATE TABLE IF NOT EXISTS claude_decisions (
    id             UUID NOT NULL DEFAULT gen_random_uuid(),
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    signal_id      UUID,
    symbol         VARCHAR(64) NOT NULL,
    model          VARCHAR(64) NOT NULL,
    action         VARCHAR(32) NOT NULL,
    confidence     NUMERIC(5, 4) NOT NULL,
    leverage       INTEGER,
    stop_loss      NUMERIC(38, 18),
    take_profit    NUMERIC(38, 18),
    reasoning      TEXT NOT NULL,          -- Turkish explanation: "Claude neden bu işlemi açtı?"
    input_context  JSONB,                  -- technical signal + news + portfolio snapshot sent to Claude
    raw_response   JSONB,
    latency_ms     INTEGER,
    input_tokens   INTEGER,
    output_tokens  INTEGER,
    risk_verdict   VARCHAR(32),
    risk_reasons   JSONB,
    final_leverage INTEGER,
    final_quantity NUMERIC(38, 18),
    fallback_reason  VARCHAR(32),
    execution_status VARCHAR(32),
    execution_detail TEXT,
    CONSTRAINT pk_claude_decisions PRIMARY KEY (id),
    CONSTRAINT ck_claude_decisions_confidence_range CHECK (confidence >= 0 AND confidence <= 1),
    CONSTRAINT ck_claude_decisions_leverage_range CHECK (leverage IS NULL OR (leverage >= 1 AND leverage <= 20)),
    CONSTRAINT ck_claude_decisions_final_leverage_range CHECK (final_leverage IS NULL OR (final_leverage >= 1 AND final_leverage <= 20)),
    CONSTRAINT fk_claude_decisions_signal_id_signals FOREIGN KEY (signal_id) REFERENCES signals (id) ON DELETE SET NULL,
    CONSTRAINT ck_claude_decisions_action CHECK (action IN ('LONG', 'SHORT', 'HOLD', 'CLOSE')),
    CONSTRAINT ck_claude_decisions_risk_verdict CHECK (risk_verdict IN ('APPROVED', 'REJECTED', 'REDUCED'))
);
CREATE INDEX IF NOT EXISTS ix_claude_decisions_created_at_desc ON claude_decisions (created_at DESC);
CREATE INDEX IF NOT EXISTS ix_claude_decisions_symbol_created_at ON claude_decisions (symbol, created_at);

-- ---------------------------------------------------------------------------------------
-- Orders, fills, trades
-- ---------------------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS orders (
    id                 UUID NOT NULL DEFAULT gen_random_uuid(),
    account_id         INTEGER NOT NULL,
    decision_id        UUID,
    exchange_order_id  VARCHAR(128),
    client_order_id    VARCHAR(128),
    symbol             VARCHAR(64) NOT NULL,
    side               VARCHAR(32) NOT NULL,
    order_type         VARCHAR(32) NOT NULL,
    status             VARCHAR(32) NOT NULL,
    purpose            VARCHAR(32) NOT NULL,
    quantity           NUMERIC(38, 18) NOT NULL,
    price              NUMERIC(38, 18),
    stop_price         NUMERIC(38, 18),
    filled_quantity    NUMERIC(38, 18) NOT NULL DEFAULT 0,
    average_fill_price NUMERIC(38, 18),
    reduce_only        BOOLEAN NOT NULL DEFAULT false,
    fee_paid           NUMERIC(38, 18) NOT NULL DEFAULT 0,
    fee_currency       VARCHAR(16),
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    raw                JSONB,
    CONSTRAINT pk_orders PRIMARY KEY (id),
    CONSTRAINT uq_orders_client_order_id UNIQUE (client_order_id),
    CONSTRAINT uq_orders_account_exchange_order_id UNIQUE (account_id, exchange_order_id),
    CONSTRAINT ck_orders_quantity_non_negative CHECK (quantity >= 0),
    CONSTRAINT ck_orders_filled_quantity_non_negative CHECK (filled_quantity >= 0),
    CONSTRAINT fk_orders_account_id_accounts FOREIGN KEY (account_id) REFERENCES accounts (id),
    CONSTRAINT fk_orders_decision_id_claude_decisions FOREIGN KEY (decision_id) REFERENCES claude_decisions (id) ON DELETE SET NULL,
    CONSTRAINT ck_orders_side CHECK (side IN ('BUY', 'SELL')),
    CONSTRAINT ck_orders_order_type CHECK (order_type IN ('MARKET', 'LIMIT', 'STOP_MARKET', 'TAKE_PROFIT_MARKET', 'TRAILING_STOP_MARKET')),
    CONSTRAINT ck_orders_status CHECK (status IN ('NEW', 'PARTIALLY_FILLED', 'FILLED', 'CANCELED', 'REJECTED', 'EXPIRED')),
    CONSTRAINT ck_orders_purpose CHECK (purpose IN ('ENTRY', 'STOP_LOSS', 'TAKE_PROFIT', 'TRAILING_STOP', 'EXIT', 'KILL_SWITCH'))
);
CREATE INDEX IF NOT EXISTS ix_orders_account_status ON orders (account_id, status);
CREATE INDEX IF NOT EXISTS ix_orders_decision_id ON orders (decision_id);
CREATE INDEX IF NOT EXISTS ix_orders_symbol_created_at ON orders (symbol, created_at);

CREATE TABLE IF NOT EXISTS fills (
    id                BIGINT GENERATED BY DEFAULT AS IDENTITY,
    order_id          UUID NOT NULL,
    exchange_trade_id VARCHAR(128) NOT NULL,
    symbol            VARCHAR(64) NOT NULL,
    side              VARCHAR(32) NOT NULL,
    price             NUMERIC(38, 18) NOT NULL,
    quantity          NUMERIC(38, 18) NOT NULL,
    fee               NUMERIC(38, 18) NOT NULL DEFAULT 0,
    fee_currency      VARCHAR(16) NOT NULL,
    liquidity         VARCHAR(32) NOT NULL,
    ts                TIMESTAMPTZ NOT NULL,
    CONSTRAINT pk_fills PRIMARY KEY (id),
    CONSTRAINT uq_fills_order_exchange_trade UNIQUE (order_id, exchange_trade_id),
    CONSTRAINT ck_fills_quantity_positive CHECK (quantity > 0),
    CONSTRAINT fk_fills_order_id_orders FOREIGN KEY (order_id) REFERENCES orders (id) ON DELETE CASCADE,
    CONSTRAINT ck_fills_side CHECK (side IN ('BUY', 'SELL')),
    CONSTRAINT ck_fills_liquidity CHECK (liquidity IN ('MAKER', 'TAKER'))
);
CREATE INDEX IF NOT EXISTS ix_fills_order_id ON fills (order_id);
CREATE INDEX IF NOT EXISTS ix_fills_symbol_ts ON fills (symbol, ts);

CREATE TABLE IF NOT EXISTS trades (
    id            UUID NOT NULL DEFAULT gen_random_uuid(),
    account_id    INTEGER NOT NULL,
    decision_id   UUID,
    symbol        VARCHAR(64) NOT NULL,
    side          VARCHAR(32) NOT NULL,
    leverage      INTEGER NOT NULL DEFAULT 1,
    margin_mode   VARCHAR(32) NOT NULL,
    opened_at     TIMESTAMPTZ NOT NULL,
    closed_at     TIMESTAMPTZ,
    entry_price   NUMERIC(38, 18) NOT NULL,
    exit_price    NUMERIC(38, 18),
    quantity      NUMERIC(38, 18) NOT NULL,
    gross_pnl     NUMERIC(38, 18),
    fees_paid     NUMERIC(38, 18) NOT NULL DEFAULT 0,
    funding_paid  NUMERIC(38, 18) NOT NULL DEFAULT 0,
    net_pnl       NUMERIC(38, 18),
    r_multiple    NUMERIC(38, 18),
    initial_stop  NUMERIC(38, 18),
    exit_reason   VARCHAR(32),
    status        VARCHAR(32) NOT NULL DEFAULT 'OPEN',
    CONSTRAINT pk_trades PRIMARY KEY (id),
    CONSTRAINT ck_trades_leverage_positive CHECK (leverage >= 1),
    CONSTRAINT ck_trades_quantity_positive CHECK (quantity > 0),
    CONSTRAINT ck_trades_closed_has_closed_at CHECK (status = 'OPEN' OR closed_at IS NOT NULL),
    CONSTRAINT fk_trades_account_id_accounts FOREIGN KEY (account_id) REFERENCES accounts (id),
    CONSTRAINT fk_trades_decision_id_claude_decisions FOREIGN KEY (decision_id) REFERENCES claude_decisions (id) ON DELETE SET NULL,
    CONSTRAINT ck_trades_side CHECK (side IN ('LONG', 'SHORT')),
    CONSTRAINT ck_trades_margin_mode CHECK (margin_mode IN ('ISOLATED', 'CROSS')),
    CONSTRAINT ck_trades_exit_reason CHECK (exit_reason IN ('TAKE_PROFIT', 'STOP_LOSS', 'TRAILING_STOP', 'MANUAL', 'KILL_SWITCH', 'SIGNAL_EXIT', 'LIQUIDATION')),
    CONSTRAINT ck_trades_status CHECK (status IN ('OPEN', 'CLOSED'))
);
CREATE INDEX IF NOT EXISTS ix_trades_account_status ON trades (account_id, status);
CREATE INDEX IF NOT EXISTS ix_trades_symbol_opened_at ON trades (symbol, opened_at);
CREATE INDEX IF NOT EXISTS ix_trades_opened_at_desc ON trades (opened_at DESC);
CREATE INDEX IF NOT EXISTS ix_trades_decision_id ON trades (decision_id);

-- ---------------------------------------------------------------------------------------
-- News, risk events, kill switch
-- ---------------------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS news_items (
    id              UUID NOT NULL DEFAULT gen_random_uuid(),
    source          VARCHAR(64) NOT NULL,
    external_id     VARCHAR(256) NOT NULL,
    url             TEXT,
    title           TEXT NOT NULL,
    body            TEXT,
    published_at    TIMESTAMPTZ NOT NULL,
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    symbols         JSONB NOT NULL DEFAULT '[]'::jsonb,
    sentiment_score NUMERIC(5, 4),
    sentiment_label VARCHAR(32),
    importance      SMALLINT,
    news_hash       VARCHAR(64),
    macro_tags      JSONB NOT NULL DEFAULT '[]'::jsonb,
    summary         TEXT,
    duplicate_sources JSONB NOT NULL DEFAULT '[]'::jsonb,
    CONSTRAINT pk_news_items PRIMARY KEY (id),
    CONSTRAINT uq_news_items_source_external_id UNIQUE (source, external_id),
    CONSTRAINT ck_news_items_sentiment_range CHECK (sentiment_score IS NULL OR (sentiment_score >= -1 AND sentiment_score <= 1)),
    CONSTRAINT ck_news_items_importance_range CHECK (importance IS NULL OR (importance >= 0 AND importance <= 10)),
    CONSTRAINT ck_news_items_sentiment_label CHECK (sentiment_label IN ('POSITIVE', 'NEUTRAL', 'NEGATIVE'))
);
CREATE INDEX IF NOT EXISTS ix_news_items_published_at_desc ON news_items (published_at DESC);
CREATE INDEX IF NOT EXISTS ix_news_items_news_hash ON news_items (news_hash);

CREATE TABLE IF NOT EXISTS risk_events (
    id      UUID NOT NULL DEFAULT gen_random_uuid(),
    ts      TIMESTAMPTZ NOT NULL DEFAULT now(),
    level   VARCHAR(32) NOT NULL,
    code    VARCHAR(64) NOT NULL,
    message TEXT NOT NULL,
    details JSONB,
    CONSTRAINT pk_risk_events PRIMARY KEY (id),
    CONSTRAINT ck_risk_events_level CHECK (level IN ('INFO', 'WARN', 'CRITICAL'))
);
CREATE INDEX IF NOT EXISTS ix_risk_events_ts_desc ON risk_events (ts DESC);
CREATE INDEX IF NOT EXISTS ix_risk_events_code_ts ON risk_events (code, ts);

CREATE TABLE IF NOT EXISTS kill_switch_state (
    id               SMALLINT NOT NULL,
    engaged          BOOLEAN NOT NULL DEFAULT false,
    reason           TEXT,
    engaged_at       TIMESTAMPTZ,
    trading_day      DATE,
    day_start_equity NUMERIC(38, 18),
    engaged_day      DATE,
    peak_equity      NUMERIC(38, 18),
    consecutive_losses INTEGER NOT NULL DEFAULT 0,
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT pk_kill_switch_state PRIMARY KEY (id),
    CONSTRAINT ck_kill_switch_state_singleton CHECK (id = 1)
);

INSERT INTO kill_switch_state (id, engaged) VALUES (1, false) ON CONFLICT (id) DO NOTHING;

-- ---------------------------------------------------------------------------------------
-- Time series (TimescaleDB hypertables; the partitioning column is part of every PK)
-- ---------------------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS equity_snapshots (
    ts                 TIMESTAMPTZ NOT NULL,
    account_id         INTEGER NOT NULL,
    equity             NUMERIC(38, 18) NOT NULL,
    wallet_balance     NUMERIC(38, 18) NOT NULL,
    unrealized_pnl     NUMERIC(38, 18) NOT NULL,
    margin_ratio       NUMERIC(38, 18),
    daily_pnl          NUMERIC(38, 18),
    daily_drawdown_pct NUMERIC(38, 18),
    CONSTRAINT pk_equity_snapshots PRIMARY KEY (ts, account_id),
    CONSTRAINT fk_equity_snapshots_account_id_accounts FOREIGN KEY (account_id) REFERENCES accounts (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS market_candles (
    ts        TIMESTAMPTZ NOT NULL,
    symbol    VARCHAR(64) NOT NULL,
    timeframe VARCHAR(16) NOT NULL,
    o         NUMERIC(38, 18) NOT NULL,
    h         NUMERIC(38, 18) NOT NULL,
    l         NUMERIC(38, 18) NOT NULL,
    c         NUMERIC(38, 18) NOT NULL,
    v         NUMERIC(38, 18) NOT NULL,
    CONSTRAINT pk_market_candles PRIMARY KEY (ts, symbol, timeframe)
);

CREATE TABLE IF NOT EXISTS position_snapshots (
    ts                TIMESTAMPTZ NOT NULL,
    account_id        INTEGER NOT NULL,
    symbol            VARCHAR(64) NOT NULL,
    side              VARCHAR(32) NOT NULL,
    quantity          NUMERIC(38, 18) NOT NULL,
    entry_price       NUMERIC(38, 18) NOT NULL,
    mark_price        NUMERIC(38, 18) NOT NULL,
    unrealized_pnl    NUMERIC(38, 18) NOT NULL,
    liquidation_price NUMERIC(38, 18),
    CONSTRAINT pk_position_snapshots PRIMARY KEY (ts, account_id, symbol),
    CONSTRAINT fk_position_snapshots_account_id_accounts FOREIGN KEY (account_id) REFERENCES accounts (id) ON DELETE CASCADE,
    CONSTRAINT ck_position_snapshots_side CHECK (side IN ('LONG', 'SHORT'))
);

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        PERFORM create_hypertable('equity_snapshots',   'ts', chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE, migrate_data => TRUE);
        PERFORM create_hypertable('market_candles',     'ts', chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE, migrate_data => TRUE);
        PERFORM create_hypertable('position_snapshots', 'ts', chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE, migrate_data => TRUE);
    END IF;
END
$$;

CREATE INDEX IF NOT EXISTS ix_market_candles_symbol_timeframe_ts ON market_candles (symbol, timeframe, ts DESC);
CREATE INDEX IF NOT EXISTS ix_position_snapshots_account_symbol_ts ON position_snapshots (account_id, symbol, ts DESC);

-- Optional: compression and retention for candles (enable once volume justifies it).
-- ALTER TABLE market_candles SET (
--     timescaledb.compress,
--     timescaledb.compress_segmentby = 'symbol, timeframe',
--     timescaledb.compress_orderby = 'ts DESC'
-- );
-- SELECT add_compression_policy('market_candles', INTERVAL '7 days', if_not_exists => TRUE);
-- SELECT add_retention_policy('market_candles', INTERVAL '2 years', if_not_exists => TRUE);
-- ALTER TABLE equity_snapshots SET (timescaledb.compress, timescaledb.compress_segmentby = 'account_id', timescaledb.compress_orderby = 'ts DESC');
-- SELECT add_compression_policy('equity_snapshots', INTERVAL '30 days', if_not_exists => TRUE);

-- ---------------------------------------------------------------------------------------
-- Terminal event bridge: NOTIFY 'terminal_events' (consumed by the Phase 4 WebSocket bridge)
-- Payloads stay small (pg_notify limit is 8000 bytes): ids and headline fields only.
-- ---------------------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION notify_terminal_event() RETURNS trigger AS $$
DECLARE
    payload jsonb;
BEGIN
    IF TG_TABLE_NAME = 'claude_decisions' THEN
        payload := jsonb_build_object(
            'id', NEW.id, 'symbol', NEW.symbol, 'action', NEW.action,
            'confidence', NEW.confidence, 'risk_verdict', NEW.risk_verdict,
            'created_at', NEW.created_at);
    ELSIF TG_TABLE_NAME = 'trades' THEN
        payload := jsonb_build_object(
            'id', NEW.id, 'symbol', NEW.symbol, 'side', NEW.side, 'status', NEW.status,
            'net_pnl', NEW.net_pnl, 'exit_reason', NEW.exit_reason,
            'opened_at', NEW.opened_at, 'closed_at', NEW.closed_at);
    ELSIF TG_TABLE_NAME = 'risk_events' THEN
        payload := jsonb_build_object(
            'id', NEW.id, 'level', NEW.level, 'code', NEW.code,
            'message', left(NEW.message, 500), 'ts', NEW.ts);
    ELSE
        payload := jsonb_build_object('id', to_jsonb(NEW) -> 'id');
    END IF;

    PERFORM pg_notify(
        'terminal_events',
        jsonb_build_object('table', TG_TABLE_NAME, 'op', TG_OP, 'data', payload)::text);
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Decisions and trades also fire on UPDATE: risk verdict arrives after insert, trades close later.
DROP TRIGGER IF EXISTS trg_claude_decisions_notify ON claude_decisions;
CREATE TRIGGER trg_claude_decisions_notify
    AFTER INSERT OR UPDATE ON claude_decisions
    FOR EACH ROW EXECUTE FUNCTION notify_terminal_event();

DROP TRIGGER IF EXISTS trg_trades_notify ON trades;
CREATE TRIGGER trg_trades_notify
    AFTER INSERT OR UPDATE ON trades
    FOR EACH ROW EXECUTE FUNCTION notify_terminal_event();

DROP TRIGGER IF EXISTS trg_risk_events_notify ON risk_events;
CREATE TRIGGER trg_risk_events_notify
    AFTER INSERT ON risk_events
    FOR EACH ROW EXECUTE FUNCTION notify_terminal_event();
