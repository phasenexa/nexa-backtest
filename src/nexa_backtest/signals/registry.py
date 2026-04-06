"""Signal registry for registering and looking up signal providers."""

from __future__ import annotations

from nexa_backtest.exceptions import SignalError
from nexa_backtest.signals.base import SignalProvider


class SignalRegistry:
    """Registry for :class:`~nexa_backtest.signals.base.SignalProvider` instances.

    Holds registered providers and provides lookup by name. The backtest
    engine populates the registry from explicitly-passed providers and from
    auto-discovered CSV files.
    """

    def __init__(self) -> None:
        self._providers: dict[str, SignalProvider] = {}

    def register(self, provider: SignalProvider) -> None:
        """Register a signal provider.

        If a provider with the same name is already registered it is
        silently replaced.

        Args:
            provider: The provider to register.
        """
        self._providers[provider.name] = provider

    def get(self, name: str) -> SignalProvider:
        """Return the provider registered under ``name``.

        Args:
            name: Signal name.

        Returns:
            The registered :class:`~nexa_backtest.signals.base.SignalProvider`.

        Raises:
            :class:`~nexa_backtest.exceptions.SignalError`: If no provider
                is registered with ``name``.
        """
        if name not in self._providers:
            available = ", ".join(sorted(self._providers)) or "(none)"
            raise SignalError(
                f"Signal '{name}' not found in registry. Available signals: {available}"
            )
        return self._providers[name]

    def has(self, name: str) -> bool:
        """Return ``True`` if a provider is registered under ``name``.

        Args:
            name: Signal name to check.

        Returns:
            Whether the name is registered.
        """
        return name in self._providers

    def list_signals(self) -> list[str]:
        """Return a sorted list of all registered signal names.

        Returns:
            Alphabetically sorted list of signal names.
        """
        return sorted(self._providers)
