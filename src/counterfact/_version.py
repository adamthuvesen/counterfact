"""Installed package version, importable without loading the package API."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("counterfact")
except PackageNotFoundError:
    __version__ = "0.1.0"
