from abc import abstractmethod
from pathlib import Path
from typing import Iterator, Protocol
import re
import logging


from milvus_lib import ComponentEntry

logger = logging.getLogger(f"uvicorn.{__name__}")


# Check for research headings (case-insensitive)
research_heading_pattern = re.compile(
    r"##\s*(?:Research|Research findings)\s*$", re.IGNORECASE | re.MULTILINE
    )


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
    
    @staticmethod
    def _check_has_research(content: str) -> bool:
        """
        Check if the document contains research based on specific criteria.

        Returns True if the document contains a heading "Research" OR "Research findings"
        AND one or more of the key terms:
        - "research showed"
        - "users understood"
        - "we found"
        - "testing showed"
        - "we observed"

        Args:
            content: The document content to check

        Returns:
            bool: True if research criteria are met
        """
        has_research_heading = bool(research_heading_pattern.search(content))
        logger.info(f"Has Research Heading: {has_research_heading}")
        if not has_research_heading:
            return False

        # Check for key research terms (case-insensitive)
        research_terms = [
            r"research showed",
            r"users understood",
            r"we found",
            r"testing showed",
            r"we observed",
            r"usability tested"
        ]

        for term in research_terms:
            if re.search(term, content, re.IGNORECASE):
                logger.info(f"Also Has Research")
                return True

        return False    
