"""Custom exceptions for langchain-dockersandbox.

Provides a hierarchy of descriptive, catchable exception types so callers
can distinguish container failures, timeouts, file failures, and gVisor
configuration problems without parsing error message strings.
"""

from __future__ import annotations


class LangChainSandboxError(Exception):
    """Base exception for all langchain-dockersandbox errors.

    Catch this to handle any error originating from the sandbox layer.
    """


class SandboxNotOpenError(LangChainSandboxError):
    """Raised when an operation requires an open sandbox session.

    Call ``sandbox.open()`` or use ``with DockerSandbox(...) as sb:``
    before invoking methods that interact with the container.
    """


class SandboxContainerError(LangChainSandboxError):
    """Raised when a container-level operation fails.

    Covers container creation, startup, image pull, and exec failures.
    Check that Docker is running and the image is accessible.
    """


class SandboxTimeoutError(LangChainSandboxError):
    """Raised or returned when a command exceeds the configured timeout.

    Increase the ``timeout`` parameter, or check the command for
    infinite loops / blocking I/O.
    """

    def __init__(self, message: str, timeout_duration: float | None = None) -> None:
        """Initialize with optional timeout duration in seconds."""
        super().__init__(message)
        self.timeout_duration = timeout_duration


class SandboxFileError(LangChainSandboxError):
    """Raised when a file upload, download, or path operation fails.

    Verify the path is an absolute POSIX path (starts with ``/``),
    the container is running, and the target directory is writable.
    """


class GVisorError(LangChainSandboxError):
    """Raised when gVisor runtime detection or configuration fails.

    Ensure gVisor (runsc) is installed on the host and registered
    as a Docker runtime in ``/etc/docker/daemon.json``.
    See https://gvisor.dev/docs/user_guide/quick_start/docker/
    """
