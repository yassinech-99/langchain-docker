"""Unit tests for the exported DockerSandbox backend."""

from __future__ import annotations

import io
import tarfile
from unittest.mock import MagicMock, patch

import pytest

from langchain_dockersandbox.exceptions import (
    SandboxContainerError,
    SandboxNotOpenError,
)
from langchain_dockersandbox.sandbox import (
    _FALLBACK_LARGE_OUTPUT_BASE,
    COMMAND_BLOCKED_EXIT_CODE,
    DEFAULT_MAX_OUTPUT_BYTES,
    TIMEOUT_EXIT_CODE,
    DockerSandbox,
)


class TestDockerSandboxInit:
    """Constructor and argument validation."""

    def test_default_init_uses_image_defaults(self) -> None:
        sandbox = DockerSandbox()

        assert sandbox.is_open is False
        assert sandbox._max_output_bytes == DEFAULT_MAX_OUTPUT_BYTES
        assert sandbox._upload_fallback is True

    def test_upload_fallback_can_disable(self) -> None:
        sandbox = DockerSandbox(upload_fallback=False)
        assert sandbox._upload_fallback is False

    def test_image_and_dockerfile_conflict(self) -> None:
        with pytest.raises(
            ValueError, match="Cannot specify both 'image' and 'dockerfile'"
        ):
            DockerSandbox(image="python:3.11", dockerfile="Dockerfile")

    def test_invalid_timeout_cleanup(self) -> None:
        with pytest.raises(ValueError, match="Unsupported timeout_cleanup"):
            DockerSandbox(timeout_cleanup="detach")


class TestFromContainer:
    """The factory keeps the public attach-to-existing-container path simple."""

    def test_from_container_defaults(self) -> None:
        sandbox = DockerSandbox.from_container("abc123")

        assert sandbox._container_id_arg == "abc123"
        assert sandbox._using_existing is True


class TestNotOpenErrors:
    """Public operations should fail clearly before open()."""

    def _make_closed_sandbox(self) -> DockerSandbox:
        sandbox = DockerSandbox.__new__(DockerSandbox)
        sandbox._container = None
        sandbox._container_id_resolved = None
        sandbox._is_open = False
        return sandbox

    def test_id_raises_sandbox_not_open(self) -> None:
        with pytest.raises(SandboxNotOpenError, match="Call open\\(\\)"):
            _ = self._make_closed_sandbox().id

    def test_execute_raises_sandbox_not_open(self) -> None:
        with pytest.raises(SandboxNotOpenError, match="is not open"):
            self._make_closed_sandbox().execute("echo hello")

    def test_upload_raises_sandbox_not_open(self) -> None:
        with pytest.raises(SandboxNotOpenError, match="is not open"):
            self._make_closed_sandbox().upload_files([("/test.txt", b"content")])

    def test_download_raises_sandbox_not_open(self) -> None:
        with pytest.raises(SandboxNotOpenError, match="is not open"):
            self._make_closed_sandbox().download_files(["/test.txt"])


class TestExecuteHooks:
    """Command validator and audit hooks run around execute()."""

    def _make_open_sandbox(self) -> DockerSandbox:
        sandbox = DockerSandbox.__new__(DockerSandbox)
        sandbox._is_open = True
        sandbox._container = MagicMock()
        sandbox._container_id_resolved = "container-1"
        sandbox._default_timeout = 30
        sandbox._encoding_errors = "strict"
        sandbox._max_output_bytes = DEFAULT_MAX_OUTPUT_BYTES
        sandbox._timeout_cleanup = "kill_container"
        return sandbox

    def test_command_validator_can_block_execution(self) -> None:
        sandbox = self._make_open_sandbox()
        sandbox._command_validator = lambda command: (
            "network disabled" if "curl" in command else None
        )
        sandbox._audit_callback = MagicMock()

        result = sandbox.execute("curl https://example.com")

        assert result.exit_code == COMMAND_BLOCKED_EXIT_CODE
        assert "network disabled" in result.output
        sandbox._container.exec_run.assert_not_called()
        sandbox._audit_callback.assert_called_once()

    def test_audit_callback_receives_success_event(self) -> None:
        sandbox = self._make_open_sandbox()
        sandbox._command_validator = None
        sandbox._audit_callback = MagicMock()
        raw = MagicMock(exit_code=0, output=(b"ok\n", None))
        sandbox._container.exec_run.return_value = raw

        result = sandbox.execute("echo ok")

        assert result.exit_code == 0
        sandbox._audit_callback.assert_called_once()
        event = sandbox._audit_callback.call_args.args[0]
        assert event["command"] == "echo ok"
        assert event["exit_code"] == 0
        assert event["truncated"] is False


class TestUploadDownload:
    """Native Docker file transfer APIs are for application-side file movement."""

    def _make_open_sandbox(self) -> DockerSandbox:
        sandbox = DockerSandbox.__new__(DockerSandbox)
        sandbox._is_open = True
        sandbox._container = MagicMock()
        sandbox._encoding_errors = "strict"
        return sandbox

    def test_upload_rejects_relative_path(self) -> None:
        sandbox = self._make_open_sandbox()

        responses = sandbox.upload_files([("relative/path.txt", b"data")])

        assert responses[0].error is not None
        assert responses[0].error == "invalid_path"

    def test_upload_rejects_parent_traversal(
        self,
    ) -> None:
        sandbox = self._make_open_sandbox()

        responses = sandbox.upload_files([("/tmp/../test.txt", b"data")])

        assert responses[0].error is not None
        assert responses[0].error == "invalid_path"
        sandbox._container.exec_run.assert_not_called()

    def test_upload_permission_error_uses_image_default_hint(self) -> None:
        sandbox = self._make_open_sandbox()
        sandbox._container.exec_run.return_value = MagicMock(
            exit_code=1,
            output=(b"Permission denied", None),
        )

        responses = sandbox.upload_files([("/usr/test.txt", b"data")])

        assert responses[0].error is not None
        assert "writable by the container image" in responses[0].error
        assert "/sandbox" not in responses[0].error

    def test_upload_checks_put_archive_result(self) -> None:
        sandbox = self._make_open_sandbox()
        sandbox._upload_fallback = False
        sandbox._container.exec_run.return_value = MagicMock(
            exit_code=0,
            output=(b"", None),
        )
        sandbox._container.put_archive.return_value = False

        responses = sandbox.upload_files([("/sandbox/test.txt", b"data")])

        assert responses[0].error is not None
        assert "put_archive failed" in responses[0].error
        assert "upload_fallback=False" in responses[0].error

    def test_upload_tar_entry_does_not_force_sandbox_user(self) -> None:
        sandbox = self._make_open_sandbox()
        sandbox._container.exec_run.return_value = MagicMock(
            exit_code=0,
            output=(b"", None),
        )
        sandbox._container.put_archive.return_value = True

        responses = sandbox.upload_files([("/sandbox/test.txt", b"data")])

        assert responses[0].error is None
        _parent, archive_bytes = sandbox._container.put_archive.call_args.args
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r") as tar:
            member = tar.getmember("test.txt")
        assert member.uid == 0
        assert member.gid == 0
        assert member.mode == 0o644

    def test_upload_verifies_file_after_archive(self) -> None:
        sandbox = self._make_open_sandbox()
        sandbox._container.exec_run.side_effect = [
            MagicMock(exit_code=0, output=(b"", None)),
            MagicMock(exit_code=1, output=(b"missing", None)),
        ]
        sandbox._container.put_archive.return_value = True

        responses = sandbox.upload_files([("/sandbox/test.txt", b"data")])

        assert responses[0].error is not None
        assert "could not be verified" in responses[0].error

    def test_download_extracts_first_regular_file(self) -> None:
        sandbox = self._make_open_sandbox()
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            info = tarfile.TarInfo(name="output.txt")
            content = b"hello"
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
        buf.seek(0)
        sandbox._container.get_archive.return_value = ([buf.getvalue()], {})

        responses = sandbox.download_files(["/sandbox/output.txt"])

        assert responses[0].content == b"hello"
        assert responses[0].error is None


class TestExecutionInternals:
    """Execution output, timeout, and Docker error handling."""

    def _make_sandbox(self) -> DockerSandbox:
        sandbox = DockerSandbox.__new__(DockerSandbox)
        sandbox._encoding_errors = "strict"
        sandbox._max_output_bytes = DEFAULT_MAX_OUTPUT_BYTES
        sandbox._timeout_cleanup = "kill_container"
        sandbox._container = None
        sandbox._is_open = True
        return sandbox

    def test_demux_merges_stdout_and_stderr(self) -> None:
        sandbox = self._make_sandbox()

        assert sandbox._demux((b"out\n", b"err\n")) == "out\n\nerr\n"

    def test_large_output_is_saved_and_truncated(self) -> None:
        sandbox = self._make_sandbox()
        sandbox._max_output_bytes = 5
        container = MagicMock()
        container.exec_run.return_value = MagicMock(exit_code=0, output=(b"", None))
        container.put_archive.return_value = True

        result = sandbox._format_execute_response(
            container=container,
            command="printf large",
            output="hello world",
            exit_code=0,
        )

        assert result.truncated is True
        assert (
            f"Full output saved to {_FALLBACK_LARGE_OUTPUT_BASE}/.deepagents/outputs/"
            in result.output
        )
        container.put_archive.assert_called_once()

    def test_exec_uses_image_default_without_environment(self) -> None:
        sandbox = self._make_sandbox()
        container = MagicMock()
        container.exec_run.return_value = MagicMock(exit_code=0, output=(b"ok", None))

        result = sandbox._run_in_container(container, "pwd", timeout_seconds=1)

        assert result.output == "ok"
        kwargs = container.exec_run.call_args.kwargs
        assert "environment" not in kwargs

    def test_timeout_kills_container_and_returns_124(self) -> None:
        import time

        sandbox = self._make_sandbox()
        container = MagicMock()
        container.exec_run.side_effect = lambda **kwargs: time.sleep(10)
        sandbox._container = container

        result = sandbox._run_in_container(container, "sleep 10", timeout_seconds=1)

        assert result.exit_code == TIMEOUT_EXIT_CODE
        assert "timed out" in result.output
        container.kill.assert_called_once()
        assert sandbox.is_open is False

    def test_timeout_stops_container_when_configured(self) -> None:
        import time

        sandbox = self._make_sandbox()
        sandbox._timeout_cleanup = "stop_container"
        container = MagicMock()
        container.exec_run.side_effect = lambda **kwargs: time.sleep(10)
        sandbox._container = container

        result = sandbox._run_in_container(container, "sleep 10", timeout_seconds=1)

        assert result.exit_code == TIMEOUT_EXIT_CODE
        container.stop.assert_called_once()
        container.kill.assert_not_called()
        assert sandbox.is_open is False

    def test_timeout_none_skips_container_shutdown(self) -> None:
        import time

        sandbox = self._make_sandbox()
        sandbox._timeout_cleanup = "none"
        container = MagicMock()
        container.exec_run.side_effect = lambda **kwargs: time.sleep(10)
        sandbox._container = container

        result = sandbox._run_in_container(container, "sleep 10", timeout_seconds=1)

        assert result.exit_code == TIMEOUT_EXIT_CODE
        container.kill.assert_not_called()
        container.stop.assert_not_called()
        assert sandbox.is_open is True
        assert "Timeout cleanup is disabled" in result.output

    def test_execute_exception_mentions_docker_ps(self) -> None:
        sandbox = self._make_sandbox()
        container = MagicMock()
        container.exec_run.side_effect = RuntimeError("Docker error")

        result = sandbox._run_in_container(container, "echo hello", timeout_seconds=30)

        assert result.exit_code == 1
        assert "Docker error" in result.output
        assert "docker ps" in result.output


class TestRuntimeConfigHelpers:
    """Runtime config helpers."""

    def test_runtime_configs_are_user_configs_only(self) -> None:
        sandbox = DockerSandbox.__new__(DockerSandbox)
        sandbox._user_runtime_configs = {"network_mode": "none"}
        sandbox._gvisor = False

        assert sandbox._resolve_runtime_configs() == {"network_mode": "none"}

    def test_gvisor_request_configures_runtime_when_not_overridden(self) -> None:
        sandbox = DockerSandbox.__new__(DockerSandbox)
        sandbox._user_runtime_configs = {}
        sandbox._gvisor = True
        sandbox._client = MagicMock()

        with patch(
            "langchain_dockersandbox.sandbox.configure_gvisor_runtime",
        ) as configure:
            configure.side_effect = lambda configs, **_kwargs: configs.update(
                {"runtime": "runsc"}
            )
            assert sandbox._resolve_runtime_configs() == {"runtime": "runsc"}

    def test_runtime_configs_runtime_overrides_gvisor_request(self) -> None:
        sandbox = DockerSandbox.__new__(DockerSandbox)
        sandbox._user_runtime_configs = {"runtime": "runc"}
        sandbox._gvisor = True
        sandbox._client = MagicMock()

        with patch(
            "langchain_dockersandbox.sandbox.configure_gvisor_runtime",
        ) as configure:
            assert sandbox._resolve_runtime_configs() == {"runtime": "runc"}
            configure.assert_not_called()

    def test_no_container_raises_descriptive_error(self) -> None:
        sandbox = DockerSandbox.__new__(DockerSandbox)
        sandbox._container = None

        with pytest.raises(SandboxContainerError, match="docker info"):
            sandbox._get_container()


class TestPoolLifecycle:
    """Container pool acquire/release wiring."""

    def test_open_acquires_container_from_pool_and_close_releases(self) -> None:
        mock_container = MagicMock()
        mock_client = MagicMock()
        mock_client.containers.get.return_value = mock_container
        pooled = MagicMock()
        pooled.container_id = "pool-cid-9"
        pool = MagicMock()
        pool.acquire.return_value = pooled

        sandbox = DockerSandbox(client=mock_client, pool=pool)
        sandbox.open()

        assert sandbox._container is mock_container
        assert sandbox._container_id_resolved == "pool-cid-9"
        pool.acquire.assert_called_once()
        mock_client.containers.get.assert_called_once_with("pool-cid-9")

        sandbox.close()

        pool.release.assert_called_once_with(pooled)
