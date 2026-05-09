"""LangChain Docker sandbox integration for Deep Agents.

Provides a language-agnostic ``DockerSandbox`` that implements the
``BaseSandbox`` protocol for use with ``create_deep_agent(backend=...)``.

Basic usage::

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
"""

from langchain_dockersandbox.exceptions import (
    GVisorError,
    LangChainSandboxError,
    SandboxContainerError,
    SandboxFileError,
    SandboxNotOpenError,
    SandboxTimeoutError,
)
from langchain_dockersandbox.gvisor import (
    configure_gvisor_runtime,
    is_gvisor_docker_runtime_available,
    is_runsc_installed,
    validate_gvisor_container,
)
from langchain_dockersandbox.sandbox import DockerSandbox

__all__ = [
    "DockerSandbox",
    "GVisorError",
    "LangChainSandboxError",
    "SandboxContainerError",
    "SandboxFileError",
    "SandboxNotOpenError",
    "SandboxTimeoutError",
    "configure_gvisor_runtime",
    "is_gvisor_docker_runtime_available",
    "is_runsc_installed",
    "validate_gvisor_container",
]
