"""Docker sandbox backend for LangChain Deep Agents.

Language-agnostic Docker sandbox that implements the ``BaseSandbox`` protocol
for use with ``create_deep_agent(backend=sandbox)``.

The Deep Agents sandbox contract requires ``id``, ``execute()``,
``upload_files()``, and ``download_files()``. ``BaseSandbox`` builds filesystem
tools (``ls``, ``read_file``, ``write_file``, ``edit_file``, ``glob``, ``grep``)
on top of those methods, and provides async wrappers for them.

Typical usage::

    from langchain_dockersandbox import DockerSandbox
    from deepagents import create_deep_agent

    sandbox = DockerSandbox(
        image="python:3.11-slim",
        keep_template=True,
        verbose=True,
    )
    sandbox.open()
    try:
        agent = create_deep_agent(
            model="anthropic:claude-sonnet-4-6",
            backend=sandbox,
            system_prompt="You are a coding assistant with sandbox access.",
        )
        result = agent.invoke({"messages": [{"role": "user", "content": "..."}]})
    finally:
        sandbox.close()

Prerequisites: Docker must be installed and running on the host.
"""

from __future__ import annotations

import io
import logging
import os
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import threading
import uuid
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

import docker
from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox
from docker.errors import APIError, ImageNotFound, NotFound

from langchain_dockersandbox.exceptions import (
    SandboxContainerError,
    SandboxNotOpenError,
)
from langchain_dockersandbox.gvisor import (
    configure_gvisor_runtime,
    validate_gvisor_container,
)

logger = logging.getLogger(__name__)

# Sensible default for BaseSandbox: ships python3 plus common POSIX tools.
DEFAULT_IMAGE = os.getenv("DEFAULT_IMAGE", "python:3.11-slim") 

# Mirrors Deep Agents guidance: large command output should not flood context.
DEFAULT_MAX_OUTPUT_BYTES = 1_000_000

# Matches the POSIX `timeout` command exit-code convention.
TIMEOUT_EXIT_CODE = 124

# Shell convention for "command invoked cannot execute".
COMMAND_BLOCKED_EXIT_CODE = 126

# Large-output persistence location inside the container.
_FALLBACK_LARGE_OUTPUT_BASE = "/tmp/langchain-dockersandbox"  # noqa: S108

_DEMUX_LEN = 2
CommandValidator = Callable[[str], str | None]
AuditCallback = Callable[[dict[str, Any]], None]


class DockerSandbox(BaseSandbox):
    """Language-agnostic Docker sandbox for LangChain Deep Agents.

    Implements the ``BaseSandbox`` interface so it can be passed directly
    to ``create_deep_agent(backend=sandbox)``.

    The sandbox is **completely language-agnostic** — the agent decides what
    to run via shell commands.  There is no venv, no pip, no language handler.

    Artifacts (files generated inside the container that you want back) can be
    declared with the ``artifacts`` constructor argument and retrieved via
    ``download_artifacts()`` once the agent has finished.

    Example::

        sandbox = DockerSandbox(
            image="python:3.11-slim",
            keep_template=True,
            artifacts=["/workspace/report.pdf"],
        )
        sandbox.open()
        try:
            agent = create_deep_agent(backend=sandbox, model=..., system_prompt=...)
            agent.invoke({"messages": [...]})
            for r in sandbox.download_artifacts():
                if r.content:
                    Path(r.path.lstrip("/")).write_bytes(r.content)
        finally:
            sandbox.close()
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        # -- Docker -----------------------------------------------------------
        client: docker.DockerClient | None = None,
        image: str | None = None,
        dockerfile: str | None = None,
        container_id: str | None = None,
        # -- Container options ------------------------------------------------
        runtime_configs: dict[str, Any] | None = None,
        keep_template: bool = False,
        commit_container: bool = False,
        # -- Security ---------------------------------------------------------
        gvisor: bool = False,
        upload_fallback: bool = True,
        # -- Timeout ----------------------------------------------------------
        timeout: int = 30 * 60,
        # -- Logging ----------------------------------------------------------
        verbose: bool = False,
        encoding_errors: str = "strict",
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
        command_validator: CommandValidator | None = None,
        audit_callback: AuditCallback | None = None,
        timeout_cleanup: str = "kill_container",
        # -- Artifacts --------------------------------------------------------
        artifacts: list[str] | None = None,
        # -- Pooling ----------------------------------------------------------
        pool: Any | None = None,
    ) -> None:
        """Create a Docker sandbox.

        Args:
            client: Docker client.  Defaults to ``docker.from_env()``.
            image: Docker image to use (e.g. ``"python:3.11-slim"``).  When
                neither ``image`` nor ``dockerfile`` is given, falls back to
                ``python:3.11-slim`` so BaseSandbox filesystem tools can use
                ``python3``.
            dockerfile: Path to a Dockerfile to build an image from.  Mutually
                exclusive with ``image``.
            container_id: ID of an already-running container to attach to.
                Skips image pull/build and container creation entirely.
            runtime_configs: Extra Docker ``containers.create()`` keyword
                arguments (mounts, env vars, resource limits, etc.).
            keep_template: Keep the Docker image after ``close()``.  Avoids
                re-pulling the same image on every session.
            commit_container: Commit the container to a new image tag on
                ``close()``, preserving installed packages for the next session.
            gvisor: Explicitly request gVisor (``runsc``) kernel-level
                sandboxing. Daemon-level ``default-runtime: runsc`` is the
                recommended production setup.
            upload_fallback: When the Engine API archive endpoints fail
                (``put_archive`` / ``get_archive`` — common on some gVisor/runsc
                setups, and ``docker cp`` may fail too), retry uploads and downloads
                using the Docker CLI: ``docker cp``, ``docker exec`` upload writer,
                and for downloads ``docker exec`` + ``python3`` streaming reads.
                Requires ``docker`` on ``PATH`` beside the Python API client.
            timeout: Default per-command timeout in seconds for ``execute()``.
            verbose: Enable DEBUG logging.
            encoding_errors: Error-handling mode when decoding container output
                bytes (forwarded to ``bytes.decode()``).
            max_output_bytes: Maximum UTF-8 bytes to return directly from
                ``execute()``. Larger outputs are written into the sandbox and
                replaced with a pointer that the agent can read incrementally.
            command_validator: Optional hook called before execution. Return an
                error string to block a command; return ``None`` to allow it.
            audit_callback: Optional callback that receives structured command
                events for logging or metrics.
            timeout_cleanup: Timeout cleanup strategy: ``"kill_container"``
                (default), ``"stop_container"``, or ``"none"``.
            artifacts: Absolute paths **inside the container** to collect after
                the agent finishes.  Retrieve them with ``download_artifacts()``.

                Example::

                    DockerSandbox(artifacts=["/workspace/out.csv", "/tmp/report.pdf"])

            pool: A ``ContainerPoolManager`` for pre-warmed container reuse.
                When provided, ``open()`` acquires a container from the pool
                and ``close()`` returns it.

        Raises:
            ValueError: If ``image`` and ``dockerfile`` are both given, or if
                ``container_id`` and ``dockerfile`` are both given.
        """
        _validate_args(image, dockerfile, container_id)
        _validate_timeout_cleanup(timeout_cleanup)

        self._client_arg: docker.DockerClient | None = client
        self._client: docker.DockerClient | None = None
        self._image_name: str | None = image
        self._dockerfile: str | None = dockerfile
        self._container_id_arg: str | None = container_id
        self._user_runtime_configs: dict[str, Any] = runtime_configs or {}
        self._keep_template: bool = keep_template
        self._do_commit: bool = commit_container
        self._gvisor: bool = gvisor
        self._upload_fallback: bool = upload_fallback
        self._default_timeout: int = timeout
        self._verbose: bool = verbose
        self._encoding_errors: str = encoding_errors
        self._max_output_bytes: int = max_output_bytes
        self._command_validator: CommandValidator | None = command_validator
        self._audit_callback: AuditCallback | None = audit_callback
        self._timeout_cleanup: str = timeout_cleanup
        self._artifacts: list[str] = artifacts or []
        self._pool: Any = pool

        # Runtime state — populated by open()
        self._container: Any = None
        self._docker_image: Any = None
        self._is_create_template: bool = False
        self._using_existing: bool = container_id is not None
        self._container_id_resolved: str | None = container_id
        self._pooled_container: Any = None
        self._is_open: bool = False
        self._lock: threading.Lock = threading.Lock()

        if verbose:
            logging.basicConfig(level=logging.DEBUG)

    # -- Factory --------------------------------------------------------------

    @classmethod
    def from_container(
        cls,
        container_id: str,
        *,
        client: docker.DockerClient | None = None,
        artifacts: list[str] | None = None,
        **kwargs: Any,
    ) -> DockerSandbox:
        """Attach to an already-running container (zero cold-start cost).

        Args:
            container_id: Docker container ID or name.
            client: Docker client.  Defaults to ``docker.from_env()``.
            artifacts: Container paths to collect with ``download_artifacts()``.
            **kwargs: Extra arguments forwarded to ``DockerSandbox.__init__``.

        Returns:
            A ``DockerSandbox`` ready to ``open()``.

        Example::

            sandbox = DockerSandbox.from_container("my-dev-container")
            sandbox.open()
            try:
                agent = create_deep_agent(backend=sandbox, ...)
                agent.invoke(...)
            finally:
                sandbox.close()
        """
        return cls(
            container_id=container_id,
            client=client,
            artifacts=artifacts,
            **kwargs,
        )

    # -- BaseSandbox protocol -------------------------------------------------

    @property
    def id(self) -> str:
        """Unique sandbox identifier — the Docker container ID.

        Raises:
            SandboxNotOpenError: If the sandbox has not been opened yet.
        """
        if self._container is not None:
            return str(self._container.id)
        if self._container_id_resolved:
            return self._container_id_resolved
        raise SandboxNotOpenError(
            "Sandbox is not open.  Call open() or use the context manager "
            "before accessing the container id."
        )

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        """Execute a shell command inside the container.

        ``BaseSandbox`` uses this method, along with ``upload_files()`` and
        ``download_files()``, to provide filesystem tools (``ls``, ``read_file``,
        ``write_file``, ``edit_file``, ``glob``, ``grep``).

        Args:
            command: Shell command string.
            timeout: Per-call timeout override in seconds.

        Returns:
            ``ExecuteResponse`` with ``output``, ``exit_code``, and
            ``truncated`` fields.

        Raises:
            SandboxNotOpenError: If the sandbox is not open.
            SandboxContainerError: If no container is available.
        """
        self._ensure_open()
        container = self._get_container()
        effective_timeout = timeout if timeout is not None else self._default_timeout

        blocked_reason = self._validate_command(command)
        if blocked_reason is not None:
            self._audit_command(
                command=command,
                timeout_seconds=effective_timeout,
                exit_code=COMMAND_BLOCKED_EXIT_CODE,
                blocked_reason=blocked_reason,
                truncated=False,
            )
            return ExecuteResponse(
                output=f"Command blocked by DockerSandbox validator: {blocked_reason}",
                exit_code=COMMAND_BLOCKED_EXIT_CODE,
                truncated=False,
            )

        result = self._run_in_container(container, command, effective_timeout)
        self._audit_command(
            command=command,
            timeout_seconds=effective_timeout,
            exit_code=result.exit_code,
            truncated=result.truncated,
        )
        return result

    def upload_files(
        self,
        files: list[tuple[str, bytes]],
    ) -> list[FileUploadResponse]:
        """Upload files into the container.

        Use this from your application code to **seed** the sandbox with
        source code, configuration, or data before the agent runs.
        Paths must be absolute POSIX paths; parent directories are created
        automatically.

        Args:
            files: List of ``(absolute_container_path, content_bytes)`` tuples.

        Returns:
            One ``FileUploadResponse`` per input file.

        Raises:
            SandboxNotOpenError: If the sandbox is not open.

        Example::

            sandbox.upload_files(
                [
                    ("/workspace/main.py", b"print('hello')\\n"),
                    ("/workspace/config.json", config_bytes),
                ]
            )
        """
        self._ensure_open()
        container = self._get_container()
        responses: list[FileUploadResponse] = []

        for path, content in files:
            validation_error = self._validate_upload_path(path)
            if validation_error is not None:
                responses.append(FileUploadResponse(path=path, error=validation_error))
                continue

            try:
                posix = PurePosixPath(path)
                parent = str(posix.parent)
                mkdir_result = self._exec_container_shell(
                    container,
                    f"mkdir -p {shlex.quote(parent)}",
                )
                if _exec_exit_code(mkdir_result) != 0:
                    error = (
                        f"Failed to create parent directory '{parent}': "
                        f"{self._demux(_exec_output(mkdir_result))}"
                    )
                    responses.append(
                        FileUploadResponse(
                            path=path,
                            error=self._append_writable_paths_hint(error),
                        )
                    )
                    continue

                buf = io.BytesIO()
                with tarfile.open(fileobj=buf, mode="w") as tar:
                    info = tarfile.TarInfo(name=posix.name)
                    info.size = len(content)
                    info.mode = 0o644
                    tar.addfile(info, io.BytesIO(content))
                buf.seek(0)
                tar_bytes = buf.getvalue()

                api_ok = self._try_put_archive(container, parent, tar_bytes)
                if api_ok:
                    verify_error = self._verify_uploaded_file(
                        container, path, len(content)
                    )
                    if verify_error is not None and getattr(
                        self, "_upload_fallback", True
                    ):
                        logger.info(
                            "put_archive reported success but verify failed for %s; "
                            "retrying via Docker CLI",
                            path,
                        )
                        if self._docker_cp_to_container(
                            container, path, content
                        ) or self._docker_exec_stdin_upload(
                            container, path, content
                        ):
                            verify_error = self._verify_uploaded_file(
                                container, path, len(content)
                            )
                    if verify_error is not None:
                        responses.append(
                            FileUploadResponse(path=path, error=verify_error)
                        )
                        continue
                    responses.append(FileUploadResponse(path=path, error=None))
                    continue

                if not getattr(self, "_upload_fallback", True):
                    err_detail = (
                        "Docker put_archive failed and upload_fallback=False."
                    )
                    responses.append(
                        FileUploadResponse(
                            path=path,
                            error=f"Failed to upload '{path}': {err_detail}",
                        )
                    )
                    continue

                logger.info(
                    "put_archive unusable for %s; trying Docker CLI fallback",
                    path,
                )
                if not (
                    self._docker_cp_to_container(container, path, content)
                    or self._docker_exec_stdin_upload(container, path, content)
                ):
                    err_detail = (
                        "Docker put_archive failed and CLI fallback "
                        "(docker cp / docker exec) also failed or docker is "
                        "not on PATH."
                    )
                    responses.append(
                        FileUploadResponse(
                            path=path,
                            error=f"Failed to upload '{path}': {err_detail}",
                        )
                    )
                    continue

                verify_error = self._verify_uploaded_file(container, path, len(content))
                if verify_error is not None:
                    responses.append(FileUploadResponse(path=path, error=verify_error))
                    continue

                responses.append(FileUploadResponse(path=path, error=None))

            except Exception as exc:
                logger.warning(
                    "upload_files: '%s' failed: %s", path, exc, exc_info=True
                )
                error = (
                    f"Failed to upload '{path}': {exc}.  "
                    f"Verify the container is running and the target directory is writable."
                )
                responses.append(
                    FileUploadResponse(
                        path=path,
                        error=self._append_writable_paths_hint(error),
                    )
                )

        return responses

    def download_files(
        self,
        paths: list[str],
    ) -> list[FileDownloadResponse]:
        """Download files from the container.

        Use this from your application code to **retrieve outputs** after
        the agent finishes — generated reports, compiled binaries, CSV exports, etc.

        Args:
            paths: Absolute container paths to download.

        Returns:
            One ``FileDownloadResponse`` per input path.  Always check
            ``result.error`` before using ``result.content``.

        Raises:
            SandboxNotOpenError: If the sandbox is not open.

        Example::

            results = sandbox.download_files(["/workspace/output.csv"])
            for r in results:
                if r.content:
                    Path("output.csv").write_bytes(r.content)
                else:
                    print(f"Error: {r.error}")
        """
        self._ensure_open()
        container = self._get_container()
        responses: list[FileDownloadResponse] = []

        for path in paths:
            if not path.startswith("/"):
                responses.append(
                    FileDownloadResponse(
                        path=path,
                        content=None,
                        error="invalid_path",
                    )
                )
                continue

            try:
                stream, _stat = container.get_archive(path)
                raw = b"".join(stream)

                with tarfile.open(fileobj=io.BytesIO(raw), mode="r") as tar:
                    members = [m for m in tar.getmembers() if m.isfile()]
                    if not members:
                        responses.append(
                            FileDownloadResponse(
                                path=path,
                                content=None,
                                error="is_directory",
                            )
                        )
                        continue

                    fobj = tar.extractfile(members[0])
                    if fobj is None:
                        responses.append(
                            FileDownloadResponse(
                                path=path,
                                content=None,
                                error=f"Could not read content at '{path}'.",
                            )
                        )
                        continue

                    responses.append(
                        FileDownloadResponse(path=path, content=fobj.read(), error=None)
                    )

            except NotFound:
                cli_bytes = self._docker_download_cli_bytes(container, path)
                if cli_bytes is not None:
                    responses.append(
                        FileDownloadResponse(
                            path=path, content=cli_bytes, error=None
                        )
                    )
                else:
                    responses.append(
                        FileDownloadResponse(
                            path=path,
                            content=None,
                            error="file_not_found",
                        )
                    )
            except APIError as exc:
                logger.info(
                    "get_archive API error for %s (%s); may retry CLI if enabled",
                    path,
                    exc,
                )
                cli_bytes = self._docker_download_cli_bytes(container, path)
                if cli_bytes is not None:
                    responses.append(
                        FileDownloadResponse(
                            path=path, content=cli_bytes, error=None
                        )
                    )
                else:
                    responses.append(
                        FileDownloadResponse(
                            path=path,
                            content=None,
                            error="file_not_found",
                        )
                    )
            except Exception as exc:
                logger.warning(
                    "download_files: '%s' failed: %s", path, exc, exc_info=True
                )
                responses.append(
                    FileDownloadResponse(
                        path=path,
                        content=None,
                        error=f"Failed to download '{path}': {exc}.",
                    )
                )

        return responses

    # -- Artifact helpers -----------------------------------------------------

    def download_artifacts(self) -> list[FileDownloadResponse]:
        """Download every path declared in the ``artifacts`` constructor arg.

        Call this after the agent has finished to retrieve the files it generated.

        Returns:
            One ``FileDownloadResponse`` per configured artifact path.

        Example::

            sandbox = DockerSandbox(
                image="python:3.11-slim",
                artifacts=["/workspace/report.pdf", "/workspace/data.csv"],
            )
            sandbox.open()
            try:
                agent.invoke(...)
                for result in sandbox.download_artifacts():
                    if result.content:
                        Path(result.path.lstrip("/")).write_bytes(result.content)
            finally:
                sandbox.close()
        """
        if not self._artifacts:
            logger.debug("No artifacts configured — returning empty list.")
            return []
        return self.download_files(self._artifacts)

    # -- Lifecycle ------------------------------------------------------------

    def open(self) -> None:
        """Create (or attach to) a Docker container and mark the sandbox open.

        Raises:
            SandboxContainerError: If container creation or image pull fails.
        """
        if self._is_open:
            return

        try:
            self._client = self._client_arg or docker.from_env()

            if self._pool is not None:
                self._open_from_pool()
            elif self._using_existing and self._container_id_arg:
                self._attach_to_existing(self._container_id_arg)
            else:
                self._create_container()

        except SandboxContainerError:
            raise
        except Exception as exc:
            raise SandboxContainerError(
                f"Failed to open sandbox: {exc}.  "
                "Ensure Docker is running ('docker info') and the image is accessible."
            ) from exc

        self._is_open = True
        logger.debug("Sandbox open — container %s", self._container_id_resolved)

        if self._wants_gvisor and self._container is not None:
            if validate_gvisor_container(self._container):
                logger.info(
                    "gVisor confirmed for container %s", self._container_id_resolved
                )
            else:
                logger.warning(
                    "gVisor requested but not confirmed for container %s.  "
                    "Ensure 'runsc' is registered as a Docker runtime.",
                    self._container_id_resolved,
                )

    def close(self) -> None:
        """Stop the container and release all resources."""
        if not self._is_open:
            return

        try:
            if self._container is not None:
                if self._do_commit and self._docker_image is not None:
                    self._commit_image()

                if self._pool is not None:
                    self._pool.release(self._pooled_container)
                    self._pooled_container = None
                elif not self._using_existing:
                    self._stop_and_remove()
                else:
                    logger.debug("Detached from existing container (not stopped).")
        finally:
            self._container = None
            self._is_open = False

        if (
            self._is_create_template
            and not self._keep_template
            and self._docker_image is not None
        ):
            self._cleanup_image()

    def __enter__(self) -> DockerSandbox:
        """Open the sandbox (supports ``with`` statement)."""
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        """Close the sandbox (supports ``with`` statement)."""
        self.close()

    # -- Properties -----------------------------------------------------------

    @property
    def is_open(self) -> bool:
        """Whether the sandbox is currently open."""
        return self._is_open

    @property
    def container(self) -> Any:
        """Raw Docker container object (for advanced / debugging use).

        Raises:
            SandboxNotOpenError: If the sandbox is not open.
        """
        self._ensure_open()
        return self._container

    @property
    def configured_artifacts(self) -> list[str]:
        """The artifact paths declared at construction time."""
        return list(self._artifacts)

    # -- Internal: lifecycle --------------------------------------------------

    def _open_from_pool(self) -> None:
        self._pooled_container = self._pool.acquire()
        cid = self._pooled_container.container_id
        self._container = self._client.containers.get(cid)  # type: ignore[union-attr]
        self._container_id_resolved = cid
        logger.debug("Acquired pooled container %s", cid)

    def _attach_to_existing(self, container_id: str) -> None:
        try:
            self._container = self._client.containers.get(container_id)  # type: ignore[union-attr]
            if self._container.status != "running":
                logger.debug("Container %s not running — starting.", container_id)
                self._container.start()
            self._container_id_resolved = str(self._container.id)
        except NotFound as exc:
            raise SandboxContainerError(
                f"Container '{container_id}' not found."
            ) from exc

    def _create_container(self) -> None:
        if self._dockerfile:
            self._docker_image = self._build_image()
            self._is_create_template = True
        elif self._image_name:
            self._docker_image = self._pull_or_get(self._image_name)
        else:
            self._docker_image = self._pull_or_get(DEFAULT_IMAGE)

        create_kwargs: dict[str, Any] = {
            "image": self._docker_image,
            "detach": True,
            "tty": True,
        }
        create_kwargs.update(self._resolve_runtime_configs())

        self._container = self._client.containers.create(**create_kwargs)  # type: ignore[union-attr]
        self._container.start()
        self._container_id_resolved = str(self._container.id)

    def _build_image(self) -> Any:
        p = Path(self._dockerfile)  # type: ignore[arg-type]
        logger.info("Building image from Dockerfile: %s", p)
        image, _ = self._client.images.build(  # type: ignore[union-attr]
            path=str(p.parent),
            dockerfile=p.name,
            tag=f"docker-sandbox-{p.parent.name}",
        )
        return image

    def _pull_or_get(self, image: str) -> Any:
        try:
            img = self._client.images.get(image)  # type: ignore[union-attr]
            logger.debug("Using cached local image '%s'", image)
            return img
        except ImageNotFound:
            logger.info("Pulling image '%s' …", image)
            img = self._client.images.pull(image)  # type: ignore[union-attr]
            self._is_create_template = True
            return img

    def _stop_and_remove(self) -> None:
        try:
            self._container.stop()
            self._container.wait()
            self._container.remove(force=True)
            logger.debug(
                "Stopped and removed container %s", self._container_id_resolved
            )
        except Exception as exc:
            logger.warning(
                "Error removing container %s: %s", self._container_id_resolved, exc
            )

    def _commit_image(self) -> None:
        if self._docker_image and self._docker_image.tags:
            full = self._docker_image.tags[-1]
            repo, tag = full.rsplit(":", 1) if ":" in full else (full, "latest")
            try:
                self._container.commit(repository=repo, tag=tag)
                logger.info("Committed container as %s:%s", repo, tag)
            except Exception as exc:
                logger.warning("Failed to commit container: %s", exc)

    def _cleanup_image(self) -> None:
        try:
            if not self._client:
                return
            in_use = self._client.containers.list(
                all=True, filters={"ancestor": self._docker_image.id}
            )
            if not in_use:
                self._docker_image.remove(force=True)
                logger.debug("Removed image %s", self._docker_image.id)
            else:
                logger.debug("Image still in use — skipping removal.")
        except Exception as exc:
            logger.warning("Failed to remove image: %s", exc)

    def _exec_container_shell(
        self,
        container: Any,
        command: str,
        *,
        user: str | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "cmd": ["bash", "-c", command],
            "demux": True,
            "tty": False,
            "stderr": True,
            "stdout": True,
        }
        if user is not None:
            kwargs["user"] = user
        return container.exec_run(**kwargs)

    def _validate_upload_path(self, path: str) -> str | None:
        if not path.startswith("/"):
            return "invalid_path"

        posix = PurePosixPath(path)
        if posix.name in {"", ".", ".."}:
            return "invalid_path"
        if ".." in posix.parts:
            return "invalid_path"
        return None

    def _verify_uploaded_file(
        self,
        container: Any,
        path: str,
        expected_size: int,
    ) -> str | None:
        quoted_path = shlex.quote(path)
        command = (
            f"test -f {quoted_path} && "
            f'test "$(wc -c < {quoted_path})" -eq {expected_size}'
        )
        result = self._exec_container_shell(container, command)
        if _exec_exit_code(result) == 0:
            return None
        output = self._demux(_exec_output(result)).strip()
        detail = f": {output}" if output else ""
        return f"Uploaded file '{path}' could not be verified inside the sandbox{detail}."

    def _try_put_archive(
        self,
        container: Any,
        parent: str,
        tar_bytes: bytes,
    ) -> bool:
        try:
            return bool(container.put_archive(parent, tar_bytes))
        except APIError as exc:
            logger.info(
                "put_archive failed (%s); may retry with CLI fallback if enabled",
                exc,
            )
            return False

    def _docker_cli_mkdir_p(self, container: Any, posix_dir: str) -> bool:
        docker_bin = shutil.which("docker")
        if not docker_bin:
            return False
        cid = str(container.id)
        completed = subprocess.run(
            [docker_bin, "exec", cid, "mkdir", "-p", posix_dir],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if completed.returncode != 0:
            logger.warning(
                "docker exec mkdir -p failed: stderr=%s stdout=%s",
                completed.stderr,
                completed.stdout,
            )
            return False
        return True

    def _docker_cp_to_container(
        self,
        container: Any,
        dest_path: str,
        content: bytes,
    ) -> bool:
        docker_bin = shutil.which("docker")
        if not docker_bin:
            logger.warning("docker CLI not found on PATH; cannot use cp fallback")
            return False
        parent = str(PurePosixPath(dest_path).parent)
        if not self._docker_cli_mkdir_p(container, parent):
            return False
        cid = str(container.id)
        tmp_path: str | None = None
        try:
            fd, tmp_path_str = tempfile.mkstemp(
                prefix="lc-dockersandbox-",
                suffix=".upload",
            )
            tmp_path = tmp_path_str
            os.close(fd)
            Path(tmp_path).write_bytes(content)
            try:
                os.chmod(tmp_path, 0o600)
            except OSError:
                pass
            dest_spec = f"{cid}:{dest_path}"
            completed = subprocess.run(
                [docker_bin, "cp", tmp_path, dest_spec],
                check=False,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if completed.returncode == 0:
                return True
            logger.warning(
                "docker cp upload fallback failed: stderr=%s stdout=%s",
                completed.stderr,
                completed.stdout,
            )
        except OSError as exc:
            logger.warning("docker cp temp file error: %s", exc)
        except subprocess.TimeoutExpired:
            logger.warning("docker cp upload fallback timed out")
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)
        return False

    def _docker_exec_stdin_upload(
        self,
        container: Any,
        dest_path: str,
        content: bytes,
    ) -> bool:
        docker_bin = shutil.which("docker")
        if not docker_bin:
            return False
        cid = str(container.id)
        cmd = [
            docker_bin,
            "exec",
            "-i",
            cid,
            "python3",
            "-c",
            "import sys; open(sys.argv[1],'wb').write(sys.stdin.buffer.read())",
            dest_path,
        ]
        proc: subprocess.Popen[bytes] | None = None
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            out, err = proc.communicate(input=content, timeout=300)
            if proc.returncode == 0:
                return True
            logger.warning(
                "docker exec stdin upload failed: rc=%s stderr=%s stdout=%s",
                proc.returncode,
                err,
                out,
            )
        except OSError as exc:
            logger.warning("docker exec stdin upload error: %s", exc)
        except subprocess.TimeoutExpired:
            logger.warning("docker exec stdin upload timed out")
            if proc is not None:
                try:
                    proc.kill()
                except OSError:
                    pass
        return False

    def _push_bytes_with_optional_fallback(
        self,
        container: Any,
        container_file_path: str,
        raw_content: bytes,
        tar_bytes: bytes,
    ) -> bool:
        parent = str(PurePosixPath(container_file_path).parent)
        if self._try_put_archive(container, parent, tar_bytes):
            return True
        if not getattr(self, "_upload_fallback", True):
            logger.debug("upload_fallback disabled; skipping CLI upload")
            return False
        logger.info(
            "put_archive unusable for %s; trying Docker CLI fallback",
            container_file_path,
        )
        if self._docker_cp_to_container(container, container_file_path, raw_content):
            return True
        return self._docker_exec_stdin_upload(
            container, container_file_path, raw_content
        )

    def _docker_cp_from_container_optional(
        self,
        container: Any,
        src_path: str,
    ) -> bytes | None:
        if not getattr(self, "_upload_fallback", True):
            return None
        docker_bin = shutil.which("docker")
        if not docker_bin:
            logger.warning(
                "docker CLI not on PATH; cannot use download cp fallback"
            )
            return None
        cid = str(container.id)
        tmp_path: str | None = None
        try:
            fd, tmp_path_str = tempfile.mkstemp(
                prefix="lc-dockersandbox-",
                suffix=".download",
            )
            tmp_path = tmp_path_str
            os.close(fd)
            Path(tmp_path).unlink(missing_ok=True)
            completed = subprocess.run(
                [docker_bin, "cp", f"{cid}:{src_path}", tmp_path],
                check=False,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if completed.returncode != 0:
                logger.warning(
                    "docker cp download fallback failed: stderr=%s stdout=%s",
                    completed.stderr,
                    completed.stdout,
                )
                return None
            return Path(tmp_path).read_bytes()
        except OSError as exc:
            logger.warning("docker cp download temp file error: %s", exc)
            return None
        except subprocess.TimeoutExpired:
            logger.warning("docker cp download fallback timed out")
            return None
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

    def _docker_exec_cat_download(
        self,
        container: Any,
        src_path: str,
    ) -> bytes | None:
        if not getattr(self, "_upload_fallback", True):
            return None
        docker_bin = shutil.which("docker")
        if not docker_bin:
            return None
        cid = str(container.id)
        try:
            completed = subprocess.run(
                [
                    docker_bin,
                    "exec",
                    cid,
                    "python3",
                    "-c",
                    "import sys; sys.stdout.buffer.write(open(sys.argv[1],'rb').read())",
                    src_path,
                ],
                check=False,
                capture_output=True,
                text=False,
                timeout=300,
            )
            if completed.returncode != 0:
                logger.warning(
                    "docker exec download fallback failed: rc=%s stderr=%s",
                    completed.returncode,
                    completed.stderr,
                )
                return None
            return completed.stdout
        except OSError as exc:
            logger.warning("docker exec download error: %s", exc)
            return None
        except subprocess.TimeoutExpired:
            logger.warning("docker exec download fallback timed out")
            return None

    def _docker_download_cli_bytes(
        self,
        container: Any,
        path: str,
    ) -> bytes | None:
        data = self._docker_cp_from_container_optional(container, path)
        if data is not None:
            return data
        return self._docker_exec_cat_download(container, path)

    # -- Internal: execution --------------------------------------------------

    def _run_in_container(
        self,
        container: Any,
        command: str,
        timeout_seconds: int,
    ) -> ExecuteResponse:
        """Run a shell command inside the container with thread-based timeout."""
        result_holder: dict[str, Any] = {}
        error_holder: dict[str, Any] = {}

        def _worker() -> None:
            try:
                kwargs: dict[str, Any] = {
                    "cmd": ["bash", "-c", command],
                    "demux": True,
                    "tty": False,
                    "stderr": True,
                    "stdout": True,
                }
                result = container.exec_run(**kwargs)
                result_holder["result"] = result
            except Exception as exc:
                error_holder["error"] = exc
                logger.warning("exec_run failed: %s", exc, exc_info=True)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        thread.join(timeout=timeout_seconds)

        if thread.is_alive():
            logger.warning(
                "Command timed out after %ds: %.80s", timeout_seconds, command
            )
            cleanup = self._cleanup_after_timeout(container)
            return ExecuteResponse(
                output=(
                    f"Command timed out after {timeout_seconds}s.  "
                    "The sandbox attempted to stop the timed-out work. "
                    "Increase the timeout or check for infinite loops / blocking I/O. "
                    f"{cleanup}"
                ),
                exit_code=TIMEOUT_EXIT_CODE,
                truncated=False,
            )

        if "error" in error_holder:
            return ExecuteResponse(
                output=(
                    f"Execution error: {error_holder['error']}. "
                    "Verify Docker is running and inspect the container with 'docker ps'."
                ),
                exit_code=1,
                truncated=False,
            )

        raw = result_holder.get("result")
        if raw is None:
            return ExecuteResponse(
                output=(
                    "No result returned from Docker exec. This usually indicates an "
                    "unexpected Docker API response."
                ),
                exit_code=1,
                truncated=False,
            )

        output = self._append_writable_paths_hint(self._demux(raw.output))
        return self._format_execute_response(
            container=container,
            command=command,
            output=output,
            exit_code=raw.exit_code or 0,
        )

    def _demux(self, output: Any) -> str:
        """Merge Docker's demuxed (stdout_bytes, stderr_bytes) into one string."""
        if output is None:
            return ""
        if isinstance(output, tuple) and len(output) == _DEMUX_LEN:
            stdout_b, stderr_b = output
            parts: list[str] = []
            if stdout_b:
                parts.append(
                    stdout_b.decode("utf-8", errors=self._encoding_errors)
                    if isinstance(stdout_b, bytes)
                    else str(stdout_b)
                )
            if stderr_b:
                decoded = (
                    stderr_b.decode("utf-8", errors=self._encoding_errors)
                    if isinstance(stderr_b, bytes)
                    else str(stderr_b)
                )
                if decoded.strip():
                    parts.append(decoded)
            return "\n".join(parts)
        if isinstance(output, bytes):
            return output.decode("utf-8", errors=self._encoding_errors)
        return str(output) if output else ""

    def _format_execute_response(
        self,
        *,
        container: Any,
        command: str,
        output: str,
        exit_code: int,
    ) -> ExecuteResponse:
        max_bytes = getattr(self, "_max_output_bytes", DEFAULT_MAX_OUTPUT_BYTES)
        encoded = output.encode(
            "utf-8", errors=getattr(self, "_encoding_errors", "strict")
        )
        if max_bytes <= 0 or len(encoded) <= max_bytes:
            return ExecuteResponse(output=output, exit_code=exit_code, truncated=False)

        saved_path = self._save_large_output(container, encoded)
        preview = encoded[:max_bytes].decode("utf-8", errors="replace")
        if saved_path is None:
            message = (
                f"{preview}\n\n[Output truncated after {max_bytes} bytes; "
                "failed to save full output inside the sandbox.]"
            )
        else:
            message = (
                f"{preview}\n\n[Output truncated after {max_bytes} bytes. "
                f"Full output saved to {saved_path}. Use read_file to inspect it incrementally.]"
            )
        logger.info("Command output truncated: %.80s", command)
        return ExecuteResponse(output=message, exit_code=exit_code, truncated=True)

    def _save_large_output(self, container: Any, content: bytes) -> str | None:
        base_dir = PurePosixPath(_FALLBACK_LARGE_OUTPUT_BASE)
        output_dir = base_dir / ".deepagents" / "outputs"
        output_path = output_dir / f"execute-{uuid.uuid4().hex}.txt"

        try:
            mkdir_result = self._exec_container_shell(
                container,
                f"mkdir -p {shlex.quote(str(output_dir))}",
            )
            if _exec_exit_code(mkdir_result) != 0:
                return None

            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w") as tar:
                info = tarfile.TarInfo(name=output_path.name)
                info.size = len(content)
                info.mode = 0o644
                tar.addfile(info, io.BytesIO(content))
            buf.seek(0)
            tar_bytes = buf.getvalue()
            out_str = str(output_path)

            if not self._push_bytes_with_optional_fallback(
                container, out_str, content, tar_bytes
            ):
                return None
            return out_str
        except Exception as exc:
            logger.warning(
                "Failed to persist large command output: %s", exc, exc_info=True
            )
            return None

    def _validate_command(self, command: str) -> str | None:
        validator = getattr(self, "_command_validator", None)
        if validator is None:
            return None
        try:
            return validator(command)
        except Exception as exc:
            logger.warning("Command validator failed closed: %s", exc, exc_info=True)
            return f"command validator failed: {exc}"

    def _audit_command(
        self,
        *,
        command: str,
        timeout_seconds: int,
        exit_code: int,
        truncated: bool,
        blocked_reason: str | None = None,
    ) -> None:
        callback = getattr(self, "_audit_callback", None)
        if callback is None:
            return
        event = {
            "command": command,
            "timeout_seconds": timeout_seconds,
            "exit_code": exit_code,
            "truncated": truncated,
            "blocked_reason": blocked_reason,
            "container_id": self._container_id_resolved,
        }
        try:
            callback(event)
        except Exception as exc:
            logger.warning("Audit callback failed: %s", exc, exc_info=True)

    def _cleanup_after_timeout(self, container: Any) -> str:
        strategy = getattr(self, "_timeout_cleanup", "kill_container")
        if strategy == "none":
            return "Timeout cleanup is disabled; the command may still be running."

        action = "kill" if strategy == "kill_container" else "stop"
        try:
            getattr(container, action)()
            if container is self._container:
                self._container = None
                self._is_open = False
            past_tense = "killed" if action == "kill" else "stopped"
            return f"Timed-out container was {past_tense}."
        except Exception as exc:
            logger.warning(
                "Failed to %s timed-out container: %s", action, exc, exc_info=True
            )
            return f"Failed to {action} the timed-out container: {exc}."

    def _append_writable_paths_hint(self, message: str) -> str:
        lowered = message.lower()
        if "read-only" not in lowered and "permission denied" not in lowered:
            return message
        return (
            f"{message} Choose a path writable by the container image, "
            "or pass a custom image, user, or runtime_configs."
        )

    # -- Internal: runtime configuration -------------------------------------

    @property
    def _wants_gvisor(self) -> bool:
        return self._gvisor

    def _resolve_runtime_configs(self) -> dict[str, Any]:
        """Merge user-supplied Docker options with optional gVisor runtime."""
        configs: dict[str, Any] = {}
        configs.update(self._user_runtime_configs)
        if self._wants_gvisor and "runtime" not in configs:
            configure_gvisor_runtime(
                configs,
                client=self._client,
            )
        return configs

    # -- Internal: guards -----------------------------------------------------

    def _ensure_open(self) -> None:
        if not self._is_open:
            raise SandboxNotOpenError(
                "Sandbox is not open.  Call sandbox.open() before performing operations, "
                "or use 'with DockerSandbox(...) as sandbox:'."
            )

    def _get_container(self) -> Any:
        if self._container is None:
            raise SandboxContainerError(
                "No container available.  Run 'docker info' to verify Docker is running "
                "and the image is accessible."
            )
        return self._container


# -- Module-level helpers -----------------------------------------------------


def _validate_args(
    image: str | None,
    dockerfile: str | None,
    container_id: str | None,
) -> None:
    if image and dockerfile:
        raise ValueError(
            "Cannot specify both 'image' and 'dockerfile'.  "
            "Use 'image' to pull an existing image, or 'dockerfile' to build one."
        )
    if container_id and dockerfile:
        raise ValueError(
            "Cannot specify both 'container_id' and 'dockerfile'.  "
            "Use 'container_id' to attach to an existing container."
        )


def _validate_timeout_cleanup(timeout_cleanup: str) -> None:
    allowed = {"kill_container", "stop_container", "none"}
    if timeout_cleanup not in allowed:
        raise ValueError(
            f"Unsupported timeout_cleanup: {timeout_cleanup!r}. "
            f"Expected one of: {sorted(allowed)}."
        )


def _exec_exit_code(result: Any) -> int:
    return int(getattr(result, "exit_code", 0) or 0)


def _exec_output(result: Any) -> Any:
    return getattr(result, "output", None)
