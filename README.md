# langchain-dockersandbox


`langchain-dockersandbox` implements Deep Agents `BaseSandbox` contract:
- `id`
- `execute()`
- `upload_files()`
- `download_files()`

## Install

```bash
pip install langchain-dockersandbox
```

or

```bash
uv add langchain-dockersandbox
```

Requirements:
- Docker daemon running
- Docker CLI available on `PATH` (needed for fallback paths)
- gVisor installed and configured as runtime for docker. `check Security Model Section`
- Envars: DOCKERSANDBOX_TEST_IMAGE 



## Quick Start


```python
from deepagents import create_deep_agent
from langchain_anthropic import ChatAnthropic
from langchain_dockersandbox import DockerSandbox

sandbox = DockerSandbox(image="python:3.11-slim", verbose=True)
sandbox.open()
try:
    agent = create_deep_agent(
        model=ChatAnthropic(model="claude-sonnet-4-20250514"),
        system_prompt="Use sandbox for shell and files.",
        backend=sandbox,
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Create main.py and run it"}]}
    )
finally:
    sandbox.close()
```

### Attach Existing Container (No Context Manager)

```python
from langchain_dockersandbox import DockerSandbox

sandbox = DockerSandbox.from_container("assistant-workspace")  # or container ID
sandbox.open()
try:
    out = sandbox.execute("python --version")
    print(out.output.strip())
finally:
    sandbox.close()
```

## Thread-scoped sandboxes

Create one Docker container per conversation/thread. This is the safest default because each thread gets a clean filesystem and process space.

```python
from deepagents import create_deep_agent
from langchain_dockersandbox import DockerSandbox

thread_id = "thread-123"

with DockerSandbox(image="python:3.11-slim") as sandbox:
    agent = create_deep_agent(
        model=model,
        backend=sandbox,
        system_prompt="Use the sandbox for shell commands and generated files.",
    )
    agent.invoke(
        {"messages": [{"role": "user", "content": "Create a package and run pytest"}]},
        config={"configurable": {"thread_id": thread_id}},
    )


## Resuing Containers

Reuse a container across many threads for a given assistant. This is useful when a coding assistant should keep a cloned repository, installed dependencies, or generated files.

```python
from langchain_dockersandbox import DockerSandbox

sandbox = DockerSandbox.from_container("assistant-workspace") # or container ID
sandbox.open()
try:
    result = sandbox.execute("git status --short")
finally:
    sandbox.close()  # Detaches; it does not stop an existing container.
```

## Direct Backend Usage

You can call the backend directly from application code:

```python
from langchain_dockersandbox import DockerSandbox

with DockerSandbox(image="python:3.11-slim") as sandbox:
    result = sandbox.execute("python - <<'PY'\nprint('hello from docker')\nPY")
    print(result.output)
    print(result.exit_code)
```

## File Transfer APIs

Deep Agents filesystem tools operate through `execute()`. Your application should use `upload_files()` and `download_files()` to move files across the host/container boundary.

```python
from langchain_dockersandbox import DockerSandbox

with DockerSandbox(image="python:3.11-slim") as sandbox:
    sandbox.upload_files([
        ("/tmp/lc-sandbox/app.py", b"print('uploaded')\n"),
        ("/tmp/lc-sandbox/data.txt", b"hello\n"),
    ])

    sandbox.execute("python /tmp/lc-sandbox/app.py > /tmp/lc-sandbox/output.txt")

    results = sandbox.download_files(["/tmp/lc-sandbox/output.txt"])
    for result in results:
        if result.content is not None:
            print(result.content.decode())
        else:
            print(result.error)
```

## Security Model

For production isolation, configure Docker to use [gVisor](https://gvisor.dev/) at the daemon level after installing `runsc`. 
With Docker `default-runtime` set to `runsc`, every container created by this library runs through gVisor without any library-managed isolation layer.

```json
{
  "default-runtime": "runsc",
  "runtimes": {
    "runsc": {
      "path": "/usr/bin/runsc",
      "runtimeArgs": [
        "--platform=systrap",
        "--overlay2=root:self",
        "--directfs=true",
        "--dcache=5000",
        "--network=sandbox",
        "--log-packets=false",
        "--rootless=false"
      ]
    }
  },
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}
```

Restart Docker after changing `daemon.json`

Quick verify (before/after gVisor):

```bash
# Before enabling gVisor default runtime
docker run --rm python:3.11-slim id
6.8.0-51-generic
# After enabling gVisor
docker run --rm --runtime=runsc python:3.11-slim id
4.4.0-gvisor
```


## Deepagents Sandbox Integration 

```bash
uv add langchain-tests
```
```python
"""LangChain standard sandbox integration tests.

"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest
from langchain_tests.integration_tests import SandboxIntegrationTests

from langchain_dockersandbox import DockerSandbox

if TYPE_CHECKING:
    from collections.abc import Iterator

    from deepagents.backends.protocol import SandboxBackendProtocol

TEST_IMAGE = "python:3.11-slim"



class TestDockerSandboxStandard(SandboxIntegrationTests):
    """Standard LangChain sandbox integration tests for DockerSandbox."""

    @pytest.fixture(scope="class")
    def sandbox(self) -> Iterator[SandboxBackendProtocol]:
        backend = DockerSandbox(
            image=TEST_IMAGE,
            verbose=False,
        )
        backend.open()
        try:
            yield backend
        finally:
            backend.close()

```
```bash
uv run --group test pytest tests/integration_tests/test_integration.py
```

## Notes

- Default image: `python:3.11-slim`
- gVisor supported (`gvisor=True` or daemon `default-runtime=runsc`)
- On some `runsc` hosts, Docker archive API fails -> package falls back to CLI copy/exec paths

