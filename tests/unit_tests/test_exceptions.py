"""Unit tests for the exception hierarchy."""

from __future__ import annotations

import pytest

from langchain_dockersandbox.exceptions import (
    GVisorError,
    LangChainSandboxError,
    SandboxContainerError,
    SandboxFileError,
    SandboxNotOpenError,
    SandboxTimeoutError,
)


class TestExceptionHierarchy:
    """Tests that exceptions inherit correctly."""

    def test_all_inherit_from_base(self) -> None:
        exceptions = [
            SandboxNotOpenError,
            SandboxContainerError,
            SandboxTimeoutError,
            SandboxFileError,
            GVisorError,
        ]
        for exc_class in exceptions:
            assert issubclass(exc_class, LangChainSandboxError), (
                f"{exc_class.__name__} does not inherit from LangChainSandboxError"
            )

    def test_base_inherits_from_exception(self) -> None:
        assert issubclass(LangChainSandboxError, Exception)

class TestExceptionCatchability:
    """Tests that exceptions can be caught at various levels."""

    def test_catch_sandbox_not_open_as_base(self) -> None:
        msg = "test"
        with pytest.raises(LangChainSandboxError):
            raise SandboxNotOpenError(msg)

class TestSandboxTimeoutError:
    """Tests for SandboxTimeoutError with timeout_duration."""

    def test_stores_timeout_duration(self) -> None:
        exc = SandboxTimeoutError("timed out", timeout_duration=30.0)
        assert exc.timeout_duration == 30.0

    def test_default_timeout_duration_is_none(self) -> None:
        exc = SandboxTimeoutError("timed out")
        assert exc.timeout_duration is None

    def test_message_preserved(self) -> None:
        exc = SandboxTimeoutError("custom message")
        assert str(exc) == "custom message"
