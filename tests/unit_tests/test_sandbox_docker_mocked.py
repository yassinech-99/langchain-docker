"""Heavy mocking against Docker SDK boundaries for coverage above daemon-less CI."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from docker.errors import APIError, ImageNotFound, NotFound

from langchain_dockersandbox.exceptions import (
    SandboxContainerError,
    SandboxNotOpenError,
)
from langchain_dockersandbox.gvisor import configure_gvisor_runtime
from langchain_dockersandbox.sandbox import COMMAND_BLOCKED_EXIT_CODE, DockerSandbox


def test_open_pull_image_create_and_close_removes_container() -> None:
    mock_client = MagicMock()
    mock_image = MagicMock()
    mock_client.images.get.side_effect = ImageNotFound("missing")
    mock_client.images.pull.return_value = mock_image
    mock_container = MagicMock()
    mock_container.id = "container-abcd"
    mock_client.containers.create.return_value = mock_container

    sandbox = DockerSandbox(image="python:unit-mock", client=mock_client)
    sandbox.open()
    assert sandbox.is_open
    assert sandbox.id == "container-abcd"
    mock_client.containers.create.assert_called_once()

    sandbox.close()
    mock_container.stop.assert_called_once()
    mock_container.wait.assert_called_once()
    mock_container.remove.assert_called_once_with(force=True)


def test_open_when_already_open_is_noop() -> None:
    mock_client = MagicMock()
    mock_image = MagicMock()
    mock_client.images.get.return_value = mock_image
    mock_container = MagicMock()
    mock_container.id = "c1"
    mock_client.containers.create.return_value = mock_container

    sandbox = DockerSandbox(image="python:unit-mock", client=mock_client)
    sandbox.open()
    sandbox.open()
    mock_client.containers.create.assert_called_once()


def test_open_failure_wraps_unknown_exceptions() -> None:
    with patch(
        "langchain_dockersandbox.sandbox.docker.from_env",
        side_effect=RuntimeError("transport broken"),
    ):
        sandbox = DockerSandbox(image="python:unit-mock")
        with pytest.raises(SandboxContainerError, match="Failed to open sandbox"):
            sandbox.open()


def test_attach_to_existing_running_container_sets_id() -> None:
    mock_client = MagicMock()
    mock_container = MagicMock()
    mock_container.status = "running"
    mock_container.id = "full-id"
    mock_client.containers.get.return_value = mock_container

    sandbox = DockerSandbox(container_id="short-name", client=mock_client)
    sandbox.open()
    assert sandbox.id == "full-id"
    mock_container.start.assert_not_called()


def test_attach_to_existing_stopped_container_starts_it() -> None:
    mock_client = MagicMock()
    mock_container = MagicMock()
    mock_container.status = "exited"
    mock_container.id = "full-id"
    mock_client.containers.get.return_value = mock_container

    sandbox = DockerSandbox(container_id="short-name", client=mock_client)
    sandbox.open()
    mock_container.start.assert_called_once()


def test_attach_missing_container_raises_sandbox_container_error() -> None:
    mock_client = MagicMock()
    mock_client.containers.get.side_effect = NotFound("nope")

    sandbox = DockerSandbox(container_id="ghost", client=mock_client)
    with pytest.raises(SandboxContainerError, match="not found"):
        sandbox.open()


def test_close_on_existing_container_only_detaches() -> None:
    mock_client = MagicMock()
    mock_container = MagicMock()
    mock_container.status = "running"
    mock_container.id = "full-id"
    mock_client.containers.get.return_value = mock_container

    sandbox = DockerSandbox(container_id="named", client=mock_client)
    sandbox.open()
    sandbox.close()

    mock_container.stop.assert_not_called()
    assert sandbox.is_open is False


def test_close_commits_when_requested() -> None:
    mock_client = MagicMock()
    mock_image = MagicMock()
    mock_image.tags = ["repo:tag"]
    mock_client.images.get.return_value = mock_image
    mock_container = MagicMock()
    mock_container.id = "cid"
    mock_client.containers.create.return_value = mock_container

    sandbox = DockerSandbox(
        image="python:unit-mock",
        client=mock_client,
        commit_container=True,
    )
    sandbox.open()
    sandbox.close()

    mock_container.commit.assert_called_once_with(repository="repo", tag="tag")


def test_stop_and_remove_logs_when_removal_errors() -> None:
    mock_client = MagicMock()
    mock_image = MagicMock()
    mock_client.images.get.return_value = mock_image
    mock_container = MagicMock()
    mock_container.id = "cid"
    mock_container.stop.side_effect = RuntimeError("stop boom")
    mock_client.containers.create.return_value = mock_container

    sandbox = DockerSandbox(image="python:unit-mock", client=mock_client)
    sandbox.open()
    sandbox.close()


def test_open_with_gvisor_warns_when_validation_fails() -> None:
    mock_client = MagicMock()
    mock_image = MagicMock()
    mock_client.images.get.return_value = mock_image
    mock_container = MagicMock()
    mock_container.id = "cid"
    mock_client.containers.create.return_value = mock_container

    with patch(
        "langchain_dockersandbox.sandbox.configure_gvisor_runtime",
        lambda configs, client=None: configs.update({"runtime": "runsc"}) or configs,
    ), patch(
        "langchain_dockersandbox.sandbox.validate_gvisor_container",
        return_value=False,
    ):
        sandbox = DockerSandbox(
            image="python:unit-mock",
            client=mock_client,
            gvisor=True,
        )
        sandbox.open()
        sandbox.close()


def test_context_manager_enter_exit() -> None:
    mock_client = MagicMock()
    mock_image = MagicMock()
    mock_client.images.get.return_value = mock_image
    mock_container = MagicMock()
    mock_container.id = "cid"
    mock_client.containers.create.return_value = mock_container

    with DockerSandbox(image="python:unit-mock", client=mock_client) as sandbox:
        assert sandbox.is_open
    assert sandbox.is_open is False


def test_container_property_requires_open_state() -> None:
    sandbox = DockerSandbox.__new__(DockerSandbox)
    sandbox._is_open = False
    sandbox._container = MagicMock()
    with pytest.raises(SandboxNotOpenError):
        _ = sandbox.container


def test_id_resolves_from_container_or_fallback_id() -> None:
    sandbox = DockerSandbox.__new__(DockerSandbox)
    sandbox._container = MagicMock()
    sandbox._container.id = "from-live-ref"
    sandbox._container_id_resolved = None
    assert sandbox.id == "from-live-ref"

    sandbox._container = None
    sandbox._container_id_resolved = "preresolved"
    assert sandbox.id == "preresolved"


def test_download_files_rejects_relative_paths() -> None:
    sandbox = DockerSandbox.__new__(DockerSandbox)
    sandbox._is_open = True
    sandbox._container = MagicMock()
    responses = sandbox.download_files(["relative.txt"])
    assert responses[0].error is not None
    assert responses[0].error == "invalid_path"


def test_download_files_get_archive_notfound_uses_docker_cp() -> None:
    sandbox = DockerSandbox.__new__(DockerSandbox)
    sandbox._is_open = True
    sandbox._upload_fallback = True
    sandbox._container = MagicMock()
    sandbox._container.id = "cid9"
    sandbox._container.get_archive.side_effect = NotFound("no archive")

    def fake_run(
        cmd: list[str],
        **_kwargs: object,
    ) -> MagicMock:
        if len(cmd) >= 4 and cmd[1] == "cp" and cmd[2].startswith("cid9:"):
            Path(cmd[3]).write_bytes(b"cli-bytes")
        return MagicMock(returncode=0, stdout="", stderr="")

    with (
        patch(
            "langchain_dockersandbox.sandbox.shutil.which",
            return_value="/bin/docker",
        ),
        patch(
            "langchain_dockersandbox.sandbox.subprocess.run",
            side_effect=fake_run,
        ),
    ):
        responses = sandbox.download_files(["/r/file.bin"])

    assert responses[0].error is None
    assert responses[0].content == b"cli-bytes"


def test_download_files_get_archive_apierror_uses_docker_cp() -> None:
    sandbox = DockerSandbox.__new__(DockerSandbox)
    sandbox._is_open = True
    sandbox._upload_fallback = True
    sandbox._container = MagicMock()
    sandbox._container.id = "z1"
    sandbox._container.get_archive.side_effect = APIError("500 boom")

    def fake_run(
        cmd: list[str],
        **_kwargs: object,
    ) -> MagicMock:
        if len(cmd) >= 4 and cmd[1] == "cp" and cmd[2].startswith("z1:"):
            Path(cmd[3]).write_bytes(b"z")
        return MagicMock(returncode=0, stdout="", stderr="")

    with (
        patch(
            "langchain_dockersandbox.sandbox.shutil.which",
            return_value="/bin/docker",
        ),
        patch(
            "langchain_dockersandbox.sandbox.subprocess.run",
            side_effect=fake_run,
        ),
    ):
        responses = sandbox.download_files(["/x"])

    assert responses[0].error is None
    assert responses[0].content == b"z"


def test_download_files_get_archive_notfound_uses_exec_when_cp_fails() -> None:
    sandbox = DockerSandbox.__new__(DockerSandbox)
    sandbox._is_open = True
    sandbox._upload_fallback = True
    sandbox._container = MagicMock()
    sandbox._container.id = "cidexec"
    sandbox._container.get_archive.side_effect = NotFound("no archive")

    def fake_run(
        cmd: list[str],
        **_kwargs: object,
    ) -> MagicMock:
        if len(cmd) >= 4 and cmd[1] == "cp":
            return MagicMock(returncode=1, stdout="", stderr="cp failed")
        if cmd[1] == "exec" and "python3" in cmd:
            return MagicMock(returncode=0, stdout=b"exec-payload", stderr=b"")
        return MagicMock(returncode=0, stdout="", stderr="")

    with (
        patch(
            "langchain_dockersandbox.sandbox.shutil.which",
            return_value="/bin/docker",
        ),
        patch(
            "langchain_dockersandbox.sandbox.subprocess.run",
            side_effect=fake_run,
        ),
    ):
        responses = sandbox.download_files(["/remote/f.txt"])

    assert responses[0].error is None
    assert responses[0].content == b"exec-payload"


def test_download_files_not_found_from_container() -> None:
    sandbox = DockerSandbox.__new__(DockerSandbox)
    sandbox._is_open = True
    sandbox._container = MagicMock()
    sandbox._container.get_archive.side_effect = NotFound("missing")

    with patch("langchain_dockersandbox.sandbox.shutil.which", return_value=None):
        responses = sandbox.download_files(["/tmp/x.bin"])
    assert responses[0].error is not None
    assert responses[0].error == "file_not_found"


def test_download_files_handles_generic_archive_errors() -> None:
    sandbox = DockerSandbox.__new__(DockerSandbox)
    sandbox._is_open = True
    sandbox._container = MagicMock()
    sandbox._container.get_archive.side_effect = ValueError("boom")

    responses = sandbox.download_files(["/tmp/x.bin"])
    assert responses[0].error is not None
    assert "Failed to download" in responses[0].error


def test_download_files_empty_tar_members_reports_missing() -> None:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        info = tarfile.TarInfo(name="nodir")
        info.type = tarfile.DIRTYPE
        tar.addfile(info)
    buf.seek(0)

    sandbox = DockerSandbox.__new__(DockerSandbox)
    sandbox._is_open = True
    sandbox._container = MagicMock()
    sandbox._container.get_archive.return_value = ([buf.read()], {})

    responses = sandbox.download_files(["/tmp/missing"])
    assert responses[0].error is not None


def test_download_artifacts_empty_returns_without_download() -> None:
    sandbox = DockerSandbox.__new__(DockerSandbox)
    sandbox._is_open = True
    sandbox._artifacts = []
    sandbox.download_files = MagicMock()
    assert sandbox.download_artifacts() == []
    sandbox.download_files.assert_not_called()


def test_upload_files_wraps_unexpected_errors() -> None:
    sandbox = DockerSandbox.__new__(DockerSandbox)
    sandbox._is_open = True
    sandbox._container = MagicMock()
    sandbox._upload_fallback = False
    sandbox._exec_container_shell = MagicMock(
        return_value=MagicMock(exit_code=0, output=(b"", None)),
    )
    sandbox._container.put_archive.side_effect = RuntimeError("boom")

    responses = sandbox.upload_files([("/sandbox/a.txt", b"hi")])
    assert responses[0].error is not None
    assert "Sandbox" in responses[0].error or "writable" in responses[0].error.lower()


def test_upload_put_archive_false_uses_docker_cp_fallback() -> None:
    sandbox = DockerSandbox.__new__(DockerSandbox)
    sandbox._is_open = True
    sandbox._upload_fallback = True
    sandbox._container = MagicMock()
    sandbox._container.id = "abc123"
    sandbox._container.put_archive.return_value = False
    sandbox._exec_container_shell = MagicMock(
        return_value=MagicMock(exit_code=0, output=(b"", None)),
    )
    with (
        patch(
            "langchain_dockersandbox.sandbox.shutil.which",
            return_value="/bin/docker",
        ),
        patch("langchain_dockersandbox.sandbox.subprocess.run") as run_mock,
    ):
        run_mock.return_value = MagicMock(returncode=0, stdout="", stderr="")
        responses = sandbox.upload_files([("/workspace/a.txt", b"hello")])

        assert responses[0].error is None
        assert run_mock.call_count == 2
        cp_calls = [c for c in run_mock.call_args_list if c[0][0][1] == "cp"]
        assert len(cp_calls) == 1
        argv = cp_calls[0][0][0]
        assert argv[0] == "/bin/docker"
        assert argv[1] == "cp"


def test_upload_put_archive_apierror_uses_cli_fallback() -> None:
    sandbox = DockerSandbox.__new__(DockerSandbox)
    sandbox._is_open = True
    sandbox._upload_fallback = True
    sandbox._container = MagicMock()
    sandbox._container.id = "cid42"
    sandbox._container.put_archive.side_effect = APIError("404 Client Error")
    sandbox._exec_container_shell = MagicMock(
        return_value=MagicMock(exit_code=0, output=(b"", None)),
    )
    with (
        patch(
            "langchain_dockersandbox.sandbox.shutil.which",
            return_value="/bin/docker",
        ),
        patch("langchain_dockersandbox.sandbox.subprocess.run") as run_mock,
    ):
        run_mock.return_value = MagicMock(returncode=0, stdout="", stderr="")
        responses = sandbox.upload_files([("/tmp/x.bin", b"zz")])

    assert responses[0].error is None


def test_upload_fallback_disabled_after_put_archive_false() -> None:
    sandbox = DockerSandbox.__new__(DockerSandbox)
    sandbox._is_open = True
    sandbox._upload_fallback = False
    sandbox._container = MagicMock()
    sandbox._container.put_archive.return_value = False
    sandbox._exec_container_shell = MagicMock(
        return_value=MagicMock(exit_code=0, output=(b"", None)),
    )
    responses = sandbox.upload_files([("/workspace/a.txt", b"x")])
    assert responses[0].error is not None
    assert "upload_fallback=False" in responses[0].error


def test_save_large_output_cli_fallback_when_put_archive_false() -> None:
    sandbox = DockerSandbox.__new__(DockerSandbox)
    sandbox._upload_fallback = True
    container = MagicMock()
    container.id = "bigout"
    container.put_archive.return_value = False
    sandbox._exec_container_shell = MagicMock(
        return_value=MagicMock(exit_code=0, output=(b"", None)),
    )
    with (
        patch(
            "langchain_dockersandbox.sandbox.shutil.which",
            return_value="/usr/bin/docker",
        ),
        patch("langchain_dockersandbox.sandbox.subprocess.run") as run_mock,
    ):
        run_mock.return_value = MagicMock(returncode=0, stdout="", stderr="")
        saved = sandbox._save_large_output(container, b"truncated-payload")

    assert saved is not None
    assert "langchain-dockersandbox" in saved
    assert run_mock.call_count == 2


def test_upload_uses_exec_when_docker_cp_fails() -> None:
    sandbox = DockerSandbox.__new__(DockerSandbox)
    sandbox._is_open = True
    sandbox._upload_fallback = True
    sandbox._container = MagicMock()
    sandbox._container.id = "cid-exec"
    sandbox._container.put_archive.return_value = False
    sandbox._exec_container_shell = MagicMock(
        return_value=MagicMock(exit_code=0, output=(b"", None)),
    )
    mock_proc = MagicMock()
    mock_proc.communicate.return_value = (b"", b"")
    mock_proc.returncode = 0
    with (
        patch(
            "langchain_dockersandbox.sandbox.shutil.which",
            return_value="/bin/docker",
        ),
        patch("langchain_dockersandbox.sandbox.subprocess.run") as run_mock,
        patch("langchain_dockersandbox.sandbox.subprocess.Popen") as popen_mock,
    ):
        run_mock.side_effect = [
            MagicMock(returncode=0, stdout="", stderr=""),
            MagicMock(returncode=1, stdout="out", stderr="err"),
        ]
        popen_mock.return_value = mock_proc
        responses = sandbox.upload_files([("/w/fallback-exec.txt", b"body")])

    assert responses[0].error is None
    assert run_mock.call_count == 2
    popen_mock.assert_called_once()
    mock_proc.communicate.assert_called_once()


def test_execute_validator_exception_surfaces_as_blocked() -> None:
    sandbox = DockerSandbox.__new__(DockerSandbox)
    sandbox._is_open = True
    sandbox._container = MagicMock()
    sandbox._container_id_resolved = "cid"
    sandbox._default_timeout = 30
    sandbox._command_validator = MagicMock(side_effect=RuntimeError("bad validator"))
    sandbox._audit_callback = None

    result = sandbox.execute("echo hi")
    assert result.exit_code == COMMAND_BLOCKED_EXIT_CODE
    assert "command validator failed" in result.output


def test_audit_callback_failure_is_non_fatal() -> None:
    sandbox = DockerSandbox.__new__(DockerSandbox)
    sandbox._is_open = True
    sandbox._container = MagicMock()
    sandbox._container_id_resolved = "cid"
    sandbox._default_timeout = 30
    sandbox._encoding_errors = "strict"
    sandbox._max_output_bytes = 100_000
    sandbox._command_validator = None
    sandbox._audit_callback = MagicMock(side_effect=RuntimeError("audit boom"))
    sandbox._container.exec_run.return_value = MagicMock(
        exit_code=0,
        output=(b"ok", None),
    )

    result = sandbox.execute("echo hi")
    assert result.exit_code == 0


def test_cleanup_after_timeout_stop_failure_returns_message() -> None:
    sandbox = DockerSandbox.__new__(DockerSandbox)
    sandbox._timeout_cleanup = "stop_container"
    container = MagicMock()
    container.stop.side_effect = RuntimeError("stop failed")
    msg = sandbox._cleanup_after_timeout(container)
    assert "Failed to stop" in msg


def test_build_image_from_dockerfile(tmp_path: Path) -> None:
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM alpine:latest\nCMD echo hi\n", encoding="utf-8")

    mock_client = MagicMock()
    built_image = MagicMock()
    mock_client.images.build.return_value = (built_image, [])
    mock_container = MagicMock()
    mock_container.id = "cid"
    mock_client.containers.create.return_value = mock_container

    sandbox = DockerSandbox(dockerfile=str(dockerfile), client=mock_client)
    sandbox.open()
    sandbox.close()

    mock_client.images.build.assert_called_once()
    build_kw = mock_client.images.build.call_args.kwargs
    assert build_kw["dockerfile"] == "Dockerfile"


def test_configure_gvisor_logs_extra_args_when_present() -> None:
    client = MagicMock()
    client.info.return_value = {"Runtimes": {"runsc": {"path": "/usr/bin/runsc"}}}
    configs: dict[str, object] = {}
    configure_gvisor_runtime(
        configs,
        client=client,
        gvisor_args=["--platform=kvm"],
    )
    assert configs.get("runtime") == "runsc"
