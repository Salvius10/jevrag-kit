from __future__ import annotations


class JevragKitError(Exception):
    """Base class for errors raised by jevrag-kit."""


class ConfigError(JevragKitError, ValueError):
    """A configuration file or mapping is invalid. The message names every problem."""


class MissingCredentials(JevragKitError, RuntimeError):
    """An API key the configuration needs is not set."""


class MissingDependency(JevragKitError, ImportError):
    """An optional SDK the configuration needs is not installed. The message names the install command."""
