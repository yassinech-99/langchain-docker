"""gVisor (runsc) runtime detection and configuration.

gVisor provides kernel-level sandboxing by intercepting application system calls
in a user-space Sentry process. When enabled, containers run with ``--runtime=runsc``
instead of the default ``runc``.

Requires gVisor to be installed on the Docker host and registered as a runtime
in ``/etc/docker/daemon.json``. See https://gvisor.dev/docs/user_guide/quick_start/docker/
"""

from __future__ import annotations

import logging
import shutil
import subprocess

from langchain_dockersandbox.exceptions import GVisorError

logger = logging.getLogger(__name__)

GVISOR_RUNTIME_NAME = "runsc"
GVISOR_KERNEL_MARKER = "gvisor"


def is_runsc_installed() -> bool:
    """Check if the runsc binary is available on the host PATH."""
    return shutil.which(GVISOR_RUNTIME_NAME) is not None


def is_gvisor_docker_runtime_available(client: object | None = None) -> bool:
    """Check if gVisor is registered as a Docker runtime.

    Args:
        client: Optional ``docker.DockerClient`` instance. If None, attempts
            to detect via ``docker info`` CLI.

    Returns:
        True if ``runsc`` is available as a Docker runtime.
    """
    if client is not None:
        return _check_via_client(client)
    return _check_via_cli()


def _check_via_client(client: object) -> bool:
    """Check gVisor availability via Docker SDK client."""
    try:
        info = client.info()  # type: ignore[union-attr]
        runtimes = info.get("Runtimes", {})
    except (AttributeError, ConnectionError, OSError) as exc:
        logger.warning(
            "Failed to query Docker runtimes via SDK: %s. "
            "Cannot determine if gVisor is available.",
            exc,
        )
        return False
    except RuntimeError as exc:
        logger.warning(
            "Unexpected error querying Docker runtimes via SDK: %s. "
            "Cannot determine if gVisor is available.",
            exc, exc_info=True,
        )
        return False
    else:
        return GVISOR_RUNTIME_NAME in runtimes


def _check_via_cli() -> bool:
    """Check gVisor availability via docker info CLI."""
    docker_cmd = shutil.which("docker")
    if docker_cmd is None:
        logger.warning(
            "Docker CLI not found on PATH. "
            "Cannot determine if gVisor runtime is available. "
            "Ensure Docker is installed and the 'docker' command is accessible."
        )
        return False

    try:
        result = subprocess.run(  # noqa: S603
            [docker_cmd, "info", "--format", "{{json .Runtimes}}"],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        logger.warning(
            "Docker info command timed out after 10s. "
            "Cannot determine if gVisor runtime is available."
        )
        return False
    except OSError as exc:
        logger.warning(
            "Failed to run docker info CLI: %s. "
            "Cannot determine if gVisor runtime is available.",
            exc,
        )
        return False
    else:
        return GVISOR_RUNTIME_NAME in result.stdout


def validate_gvisor_container(container: object) -> bool:
    """Verify that a running container is actually using the gVisor runtime.

    gVisor reports a synthetic kernel version containing "gvisor" in ``uname -r``.

    Args:
        container: A running Docker container object.

    Returns:
        True if the container appears to be running under gVisor.
    """
    try:
        result = container.exec_run("uname -r")  # type: ignore[union-attr]
        exit_code = result.exit_code or 0
        output = result.output
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        return exit_code == 0 and GVISOR_KERNEL_MARKER in output.lower()
    except (AttributeError, RuntimeError, OSError) as exc:
        logger.warning(
            "Failed to verify gVisor runtime in container: %s. "
            "The container may not be running or 'uname' is not available.",
            exc, exc_info=True,
        )
        return False


def configure_gvisor_runtime(
    runtime_configs: dict,
    *,
    gvisor_args: list[str] | None = None,
    client: object | None = None,
    strict: bool = False,
) -> dict:
    """Apply gVisor runtime configuration to Docker runtime_configs.

    Args:
        runtime_configs: Existing Docker runtime configuration dict (modified in place).
        gvisor_args: Optional additional arguments for the runsc runtime.
        client: Optional Docker client for availability check.
        strict: If True, raise GVisorError when gVisor is unavailable.
            If False, log a warning and fall back to default runtime.

    Returns:
        The updated runtime_configs dict.

    Raises:
        GVisorError: If strict=True and gVisor is not available.
    """
    available = is_gvisor_docker_runtime_available(client)

    if not available:
        if strict:
            msg = (
                "gVisor runtime (runsc) is not available as a Docker runtime. "
                "Install gVisor and register it in /etc/docker/daemon.json. "
                "See https://gvisor.dev/docs/user_guide/quick_start/docker/ "
                "If you want to fall back to the default runtime instead, "
                "set strict=False or omit gvisor=True."
            )
            raise GVisorError(msg)
        logger.warning(
            "gVisor runtime (runsc) not available. Falling back to default "
            "Docker runtime. For enhanced isolation, install gVisor: "
            "https://gvisor.dev/docs/user_guide/quick_start/docker/"
        )
        return runtime_configs

    runtime_configs["runtime"] = GVISOR_RUNTIME_NAME
    logger.info("gVisor runtime (runsc) enabled for container isolation")

    if gvisor_args:
        logger.debug("gVisor extra args: %s", gvisor_args)

    return runtime_configs
