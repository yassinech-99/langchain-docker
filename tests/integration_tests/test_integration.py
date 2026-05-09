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
