"""Test package for vaul."""

import logging
import pytest

logger = logging.getLogger(__name__)


class BaseTest:
    """Base class for all tests"""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Set up test environment before each test and tear down after"""
        logger.info(f"Setting up {self.__class__.__name__}")

        yield

        logger.info(f"Tearing down {self.__class__.__name__}")
