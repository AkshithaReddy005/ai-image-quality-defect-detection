"""Shared test fixtures and configuration."""
import pytest
import asyncio


# Ensure each test function gets its own event loop for pytest-asyncio
pytest_plugins = ('pytest_asyncio',)
