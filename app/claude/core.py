"""Claude Decision Core: payload in, validated ClaudeDecision out. Never raises.

The LLM proposes; the Hard-Risk Engine disposes. Every failure mode degrades to an automatic HOLD
carrying a FallbackReason, and every success is semantically checked against the quant payload so the
model can only confirm, reject (HOLD) or CLOSE - never flip or widen the trade.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from collections.abc import Callable
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import anthropic

from app.claude.contracts import (
    ClaudeDecision,
    DecisionAction,
    DecisionResult,
    EntryType,
    FallbackReason,
)
from app.claude.prompt import SYSTEM_PROMPT, build_user_message
from app.claude.tool import TOOL_NAME, decision_tool_definition, parse_tool_input
from app.claude.transport import LLMTransport
from app.config.settings import ClaudeSettings
from app.core.types import utcnow
from app.quant.contracts import SignalPayload

if TYPE_CHECKING:  # pragma: no cover
    from app.claude.decision_log import DecisionLogger

logger = logging.getLogger(__name__)

MAX_STOP_ATR_MULTIPLE = Decimal("5")
MAX_LIMIT_DEVIATION = Decimal("0.005")
_SINK_TIMEOUT_SECONDS = 5.0

_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_\-]{6,}"),
    re.compile(r"(?i)(x-api-key|api[_-]?key|authorization)\s*[:=]\s*\S+"),
    re.compile(r"(?i)bearer\s+\S+"),
)

_HOLD_TEXT = {
    FallbackReason.TIMEOUT: "Claude yanıtı zaman aşımına uğradı",
    FallbackReason.RATE_LIMITED: "Claude API istek sınırına takıldı (429)",
    FallbackReason.API_ERROR: "Claude API hatası veya bağlantı sorunu",
    FallbackReason.INVALID_OUTPUT: "Claude çıktısı geçersiz veya kurallara aykırı",
    FallbackReason.CIRCUIT_OPEN: "Claude devre kesici açık, art arda hata nedeniyle API çağrıları askıda",
    FallbackReason.NO_API_KEY: "Claude API anahtarı tanımlı değil",
}


def redact(text: str, limit: int = 300) -> str:
    out = text
    for pat in _SECRET_PATTERNS:
        out = pat.sub("[REDACTED]", out)
    return out[:limit]


def _json_safe(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, default=str, ensure_ascii=False))
    except Exception:  # noqa: BLE001
        return {"unserialisable": redact(repr(value))}


def validate_semantics(decision: ClaudeDecision, payload: SignalPayload) -> str | None:
    """Return a Turkish violation message, or None when the decision is consistent with the payload."""
    action = decision.action
    if action in (DecisionAction.HOLD, DecisionAction.CLOSE):
        return None
    if action.value != payload.direction.value:
        return f"yön değiştirme yasak: sinyal {payload.direction.value}, Claude {action.value} döndürdü"

    ref = payload.proposed_entry
    sl, tp = decision.stop_loss, decision.take_profit
    if sl is None or tp is None:  # also guarded by the contract; defensive
        return "stop_loss/take_profit eksik"
    if action is DecisionAction.LONG:
        if not sl < ref < tp:
            return f"LONG için stop_loss < giriş < take_profit olmalı (sl={sl}, giriş={ref}, tp={tp})"
    else:
        if not tp < ref < sl:
            return f"SHORT için take_profit < giriş < stop_loss olmalı (sl={sl}, giriş={ref}, tp={tp})"

    atr = payload.indicators.atr
    if atr is None or atr <= 0:
        return "ATR değeri yok, stop mesafesi doğrulanamadı"
    max_dist = MAX_STOP_ATR_MULTIPLE * Decimal(str(atr))
    if abs(ref - sl) > max_dist:
        return f"stop mesafesi 5 x ATR sınırını aşıyor (mesafe={abs(ref - sl)}, sınır={max_dist})"

    if decision.entry_type is EntryType.LIMIT:
        price = decision.entry_price
        if price is None:
            return "LIMIT giriş için entry_price eksik"
        if ref <= 0 or abs(price - ref) / ref > MAX_LIMIT_DEVIATION:
            return f"LIMIT fiyatı önerilen girişten %0.5'ten fazla uzak (fiyat={price}, giriş={ref})"
        ok = (sl < price < tp) if action is DecisionAction.LONG else (tp < price < sl)
        if not ok:
            return "LIMIT fiyatı stop_loss ile take_profit arasında değil"
    return None


class ClaudeDecisionCore:
    def __init__(
        self,
        settings: ClaudeSettings,
        transport: LLMTransport | None,
        *,
        api_key_present: bool = True,
        logger_sink: "DecisionLogger | None" = None,
        clock: Callable[[], Any] = utcnow,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._settings = settings
        self._transport = transport
        self._api_key_present = api_key_present
        self._sink = logger_sink
        self._clock = clock
        self._mono = monotonic
        self._consecutive_failures = 0
        self._open_until: float | None = None
        self._stats: dict[str, int] = {"ok": 0, **{r.value: 0 for r in FallbackReason}}
        self._tool = decision_tool_definition()

    # ------------------------------------------------------------------ public

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    @property
    def circuit_open(self) -> bool:
        return self._open_until is not None and self._mono() < self._open_until

    async def decide(self, payload: SignalPayload, portfolio_context: dict[str, Any]) -> DecisionResult:
        started = self._mono()
        decision_id = str(uuid.uuid4())
        try:
            result = await self._decide(payload, portfolio_context, decision_id, started)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - the core must never raise
            logger.exception("claude core internal error")
            result = self._fallback(
                payload, decision_id, started, FallbackReason.API_ERROR,
                f"iç hata: {type(exc).__name__}", redact(f"{type(exc).__name__}: {exc}"), count_failure=True,
            )
        self._stats[result.fallback.value if result.fallback else "ok"] += 1
        await self._log(result, payload, portfolio_context)
        return result

    # ------------------------------------------------------------------ internals

    def _latency_ms(self, started: float) -> int:
        return max(0, int(round((self._mono() - started) * 1000)))

    def _fallback(
        self,
        payload: SignalPayload,
        decision_id: str,
        started: float,
        reason: FallbackReason,
        detail_tr: str,
        error: str,
        *,
        count_failure: bool,
        response: Any = None,
        raw: dict[str, Any] | None = None,
    ) -> DecisionResult:
        if count_failure:
            self._record_failure()
        text = f"{_HOLD_TEXT[reason]}: {detail_tr}. İşlem açılmadı (HOLD)."
        decision = ClaudeDecision.hold(text, [f"FALLBACK_{reason.value}"])
        usage = getattr(response, "usage", None)
        return DecisionResult(
            decision_id=decision_id,
            signal_id=str(getattr(payload, "signal_id", "unknown")),
            symbol=str(getattr(payload, "symbol", "UNKNOWN")),
            decision=decision,
            model=self._settings.model,
            created_at=self._clock(),
            latency_ms=self._latency_ms(started),
            fallback=reason,
            error=redact(error),
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            raw_response=raw or {},
        )

    def _record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._settings.circuit_breaker_failures:
            self._open_until = self._mono() + self._settings.circuit_breaker_cooldown_seconds

    def _record_success(self) -> None:
        self._consecutive_failures = 0
        self._open_until = None

    async def _decide(
        self, payload: SignalPayload, portfolio_context: dict[str, Any], decision_id: str, started: float
    ) -> DecisionResult:
        s = self._settings
        if not self._api_key_present or self._transport is None:
            return self._fallback(payload, decision_id, started, FallbackReason.NO_API_KEY,
                                  "API çağrısı yapılmadı", "no api key configured", count_failure=False)

        if self._open_until is not None:
            if self._mono() < self._open_until:
                return self._fallback(payload, decision_id, started, FallbackReason.CIRCUIT_OPEN,
                                      f"{s.circuit_breaker_cooldown_seconds:g} sn bekleme süresi dolmadı",
                                      "circuit breaker open", count_failure=False)
            # Cooldown over: half-open. One failed probe re-opens immediately.
            self._open_until = None
            self._consecutive_failures = max(0, s.circuit_breaker_failures - 1)

        user_message = build_user_message(payload, portfolio_context, s.max_news_chars)
        kwargs: dict[str, Any] = {
            "model": s.model,
            "max_tokens": s.max_output_tokens,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_message}],
            "tools": [self._tool],
            "tool_choice": {"type": "tool", "name": TOOL_NAME},
        }
        if s.temperature is not None:  # some newer models reject sampling params; leave unset for those
            kwargs["temperature"] = s.temperature

        try:
            response = await asyncio.wait_for(self._transport.create_message(**kwargs), timeout=s.timeout_seconds)
        except (asyncio.TimeoutError, anthropic.APITimeoutError) as exc:
            return self._fallback(payload, decision_id, started, FallbackReason.TIMEOUT,
                                  f"{s.timeout_seconds:g} sn içinde yanıt gelmedi",
                                  f"timeout: {type(exc).__name__}", count_failure=True)
        except anthropic.RateLimitError as exc:
            return self._fallback(payload, decision_id, started, FallbackReason.RATE_LIMITED,
                                  "daha sonra tekrar denenecek", f"rate limited: {redact(str(exc))}", count_failure=True)
        except anthropic.APIStatusError as exc:
            reason = FallbackReason.RATE_LIMITED if getattr(exc, "status_code", None) == 429 else FallbackReason.API_ERROR
            return self._fallback(payload, decision_id, started, reason, f"HTTP {getattr(exc, 'status_code', '?')}",
                                  f"api status {getattr(exc, 'status_code', '?')}: {redact(str(exc))}", count_failure=True)
        except anthropic.APIConnectionError as exc:
            return self._fallback(payload, decision_id, started, FallbackReason.API_ERROR, "bağlantı kurulamadı",
                                  f"connection error: {redact(str(exc))}", count_failure=True)
        except Exception as exc:  # noqa: BLE001
            return self._fallback(payload, decision_id, started, FallbackReason.API_ERROR,
                                  f"beklenmeyen hata ({type(exc).__name__})",
                                  f"{type(exc).__name__}: {redact(str(exc))}", count_failure=True)

        return self._handle_response(response, payload, decision_id, started)

    def _handle_response(self, response: Any, payload: SignalPayload, decision_id: str, started: float) -> DecisionResult:
        stop_reason = getattr(response, "stop_reason", None)
        raw: dict[str, Any] = {"id": getattr(response, "id", None), "stop_reason": stop_reason}
        content = getattr(response, "content", None) or []
        block = next(
            (b for b in content if getattr(b, "type", None) == "tool_use" and getattr(b, "name", None) == TOOL_NAME),
            None,
        )
        if block is None:
            return self._fallback(payload, decision_id, started, FallbackReason.INVALID_OUTPUT,
                                  "araç çağrısı (tool_use) bulunamadı", "missing tool_use block",
                                  count_failure=True, response=response, raw=_json_safe(raw))
        tool_input = getattr(block, "input", None)
        raw["tool_input"] = _json_safe(tool_input)

        try:
            decision = parse_tool_input(tool_input)  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001 - ValueError / ValidationError / anything odd
            return self._fallback(payload, decision_id, started, FallbackReason.INVALID_OUTPUT,
                                  "şema doğrulaması başarısız", f"schema violation: {redact(str(exc), 400)}",
                                  count_failure=True, response=response, raw=raw)

        violation = validate_semantics(decision, payload)
        if violation is not None:
            return self._fallback(payload, decision_id, started, FallbackReason.INVALID_OUTPUT,
                                  violation, f"semantic violation: {violation}",
                                  count_failure=True, response=response, raw=raw)

        self._record_success()
        usage = getattr(response, "usage", None)
        return DecisionResult(
            decision_id=decision_id,
            signal_id=payload.signal_id,
            symbol=payload.symbol,
            decision=decision,
            model=self._settings.model,
            created_at=self._clock(),
            latency_ms=self._latency_ms(started),
            fallback=None,
            error=None,
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            raw_response=raw,
        )

    async def _log(self, result: DecisionResult, payload: SignalPayload, portfolio_context: dict[str, Any]) -> None:
        if self._sink is None:
            return
        try:
            await asyncio.wait_for(self._sink.log(result, payload, portfolio_context), timeout=_SINK_TIMEOUT_SECONDS)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - sink failures must never break decide()
            logger.warning("decision log sink failed: %s", redact(f"{type(exc).__name__}: {exc}"))
