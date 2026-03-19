from abc import abstractmethod
from pathlib import Path
from typing import Iterator, Protocol

from milvus_lib import ComponentEntry


class ProjectExists(Protocol):
    """Protocol for types that represent a project with a root directory."""

    @abstractmethod
    def project_root(self) -> Path:
        """Return the path to the project root directory."""
        raise NotImplementedError

    @abstractmethod
    def project_exists(self) -> bool:
        """Return True if the project root exists on disk."""
        raise NotImplementedError


class ExtractComponents(ProjectExists, Protocol):
    """Protocol for extractors that yield design system components from a project."""

    @abstractmethod
    def component_count(self) -> int:
        """Return the number of components available in the project."""
        raise NotImplementedError

    @abstractmethod
    def extract_components(self) -> Iterator[ComponentEntry]:
        """Yield ComponentEntry instances from the project."""
        raise NotImplementedError
