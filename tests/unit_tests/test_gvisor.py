"""Unit tests for gVisor detection and configuration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from langchain_dockersandbox.exceptions import GVisorError
from langchain_dockersandbox.gvisor import (
    configure_gvisor_runtime,
    is_gvisor_docker_runtime_available,
    is_runsc_installed,
    validate_gvisor_container,
)


class TestRunscInstalled:
    """Tests for runsc binary detection."""

    @patch(
        "langchain_dockersandbox.gvisor.shutil.which",
        return_value="/usr/local/bin/runsc",
    )
    def test_runsc_found(self, mock_which: MagicMock) -> None:
        assert is_runsc_installed() is True
        mock_which.assert_called_once_with("runsc")

    @patch("langchain_dockersandbox.gvisor.shutil.which", return_value=None)
    def test_runsc_not_found(self, mock_which: MagicMock) -> None:
        assert is_runsc_installed() is False


class TestGvisorDockerRuntime:
    """Tests for Docker runtime availability check."""

    def test_available_via_client(self) -> None:
        client = MagicMock()
        client.info.return_value = {"Runtimes": {"runc": {}, "runsc": {}}}

        assert is_gvisor_docker_runtime_available(client) is True

    def test_not_available_via_client(self) -> None:
        client = MagicMock()
        client.info.return_value = {"Runtimes": {"runc": {}}}

        assert is_gvisor_docker_runtime_available(client) is False

    def test_client_connection_error(self) -> None:
        client = MagicMock()
        client.info.side_effect = ConnectionError("connection refused")

        assert is_gvisor_docker_runtime_available(client) is False

    def test_client_unexpected_error(self) -> None:
        client = MagicMock()
        client.info.side_effect = RuntimeError("unexpected")

        assert is_gvisor_docker_runtime_available(client) is False

    @patch("langchain_dockersandbox.gvisor.subprocess.run")
    def test_available_via_cli(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(stdout='{"runc":{},"runsc":{}}')

        assert is_gvisor_docker_runtime_available(None) is True

    @patch("langchain_dockersandbox.gvisor.subprocess.run")
    def test_not_available_via_cli(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(stdout='{"runc":{}}')

        assert is_gvisor_docker_runtime_available(None) is False


class TestValidateGvisorContainer:
    """Tests for post-start gVisor verification."""

    def test_gvisor_kernel(self) -> None:
        container = MagicMock()
        container.exec_run.return_value = MagicMock(
            exit_code=0,
            output=b"4.4.0-gvisor\n",
        )
        assert validate_gvisor_container(container) is True

    def test_non_gvisor_kernel(self) -> None:
        container = MagicMock()
        container.exec_run.return_value = MagicMock(
            exit_code=0,
            output=b"5.15.0-88-generic\n",
        )
        assert validate_gvisor_container(container) is False

    def test_exec_failure(self) -> None:
        container = MagicMock()
        container.exec_run.side_effect = RuntimeError("not running")

        assert validate_gvisor_container(container) is False


class TestConfigureGvisorRuntime:
    """Tests for applying gVisor to runtime_configs."""

    @patch(
        "langchain_dockersandbox.gvisor.is_gvisor_docker_runtime_available",
        return_value=True,
    )
    def test_gvisor_available(self, mock_check: MagicMock) -> None:
        configs: dict = {}
        result = configure_gvisor_runtime(configs)

        assert result["runtime"] == "runsc"

    @patch(
        "langchain_dockersandbox.gvisor.is_gvisor_docker_runtime_available",
        return_value=False,
    )
    def test_gvisor_unavailable_fallback(self, mock_check: MagicMock) -> None:
        configs: dict = {}
        result = configure_gvisor_runtime(configs)

        assert "runtime" not in result

    @patch(
        "langchain_dockersandbox.gvisor.is_gvisor_docker_runtime_available",
        return_value=False,
    )
    def test_gvisor_unavailable_strict_raises_gvisor_error(
        self,
        mock_check: MagicMock,
    ) -> None:
        configs: dict = {}

        with pytest.raises(GVisorError, match="not available as a Docker runtime"):
            configure_gvisor_runtime(configs, strict=True)

    @patch(
        "langchain_dockersandbox.gvisor.is_gvisor_docker_runtime_available",
        return_value=False,
    )
    def test_gvisor_strict_error_suggests_install(self, mock_check: MagicMock) -> None:
        with pytest.raises(GVisorError, match=r"daemon\.json"):
            configure_gvisor_runtime({}, strict=True)

    @patch(
        "langchain_dockersandbox.gvisor.is_gvisor_docker_runtime_available",
        return_value=False,
    )
    def test_gvisor_strict_error_has_docs_link(self, mock_check: MagicMock) -> None:
        with pytest.raises(GVisorError, match=r"gvisor\.dev"):
            configure_gvisor_runtime({}, strict=True)

    @patch(
        "langchain_dockersandbox.gvisor.is_gvisor_docker_runtime_available",
        return_value=False,
    )
    def test_gvisor_strict_error_suggests_fallback(self, mock_check: MagicMock) -> None:
        with pytest.raises(GVisorError, match="strict=False"):
            configure_gvisor_runtime({}, strict=True)
