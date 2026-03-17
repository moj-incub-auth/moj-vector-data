from abc import abstractmethod
from pathlib import Path
from typing import Iterator, Protocol

from milvus_lib import ComponentEntry


class ProjectExists(Protocol):
    @abstractmethod
    def project_root(self) -> Path:
        raise NotImplementedError

    @abstractmethod
    def project_exists(self) -> bool:
        raise NotImplementedError


class ExtractComponents(ProjectExists, Protocol):
    @abstractmethod
    def component_count(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def extract_components(self) -> Iterator[ComponentEntry]:
        raise NotImplementedError
