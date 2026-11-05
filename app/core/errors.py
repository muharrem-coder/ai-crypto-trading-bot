"""Exception hierarchy. Every layer raises subclasses of QuantCoreError."""


class QuantCoreError(Exception):
    """Base error."""


class ConfigError(QuantCoreError):
    """Invalid or unsafe configuration."""


class BrokerError(QuantCoreError):
    """Broker / exchange communication failed."""


class BrokerNotSupportedError(BrokerError):
    """Feature not available on this broker or market type."""


class InsufficientFundsError(BrokerError):
    pass


class OrderRejectedError(BrokerError):
    pass


class RiskViolation(QuantCoreError):
    """Raised by the hard-risk engine. Never caught-and-ignored by strategy or LLM code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


class KillSwitchEngaged(RiskViolation):
    def __init__(self, message: str) -> None:
        super().__init__("KILL_SWITCH", message)
