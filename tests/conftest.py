"""
Test configuration for spekk test suite.
"""
import os
import pytest


def pytest_configure(config):
    """Configure pytest settings."""
    # Set Array API tests module for compliance testing
    os.environ.setdefault("ARRAY_API_TESTS_MODULE", "spekk.ops")
    os.environ.setdefault("ARRAY_API_TESTS_VERSION", "2023.12")