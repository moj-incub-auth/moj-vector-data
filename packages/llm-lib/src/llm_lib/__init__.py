# Standard library imports
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

# Third party imports
from openai import OpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ComponentData(BaseModel):
    """Component data model matching the required JSON format."""

    title: str
    url: str
    description: str
    parent: str
    accessibility: str = Field(alias="accessability")  # Note: matching the typo in spec
    created_at: str
    updated_at: str
    has_research: bool
    views: int = 0

    class Config:
        populate_by_name = True


class LLMResponse(BaseModel):
    """LLM response model."""

    message: str
    components: list[ComponentData]


class ComponentExtractor:
    """Extract structured component data from design system files using LLM."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = "qwen3-14b-llm",
    ):
        """
        Initialize the ComponentExtractor.

        Args:
            base_url: Base URL for the OpenAI-compatible API endpoint
            api_key: API key for authentication
            model: Model name to use for extraction (default: qwen3-14b-llm)
        """

        self.model = model
        self.client = OpenAI(
            base_url=base_url
            or os.getenv(
                "OPENAI_BASE_URL",
                "http://127.0.0.1:8090/v1", #oc port-forward qwen3-14b-llm-predictor-67c95fbd5-r4vs5 8090:808
            ),            
            api_key=api_key or os.getenv("OPENAI_API_KEY", "not-needed"),
        )

    def _check_has_research(self, content: str) -> bool:
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
        # Check for research headings (case-insensitive)
        research_heading_pattern = re.compile(
            r"<h[1-6][^>]*>\s*(?:Research|Research findings)\s*</h[1-6]>", re.IGNORECASE
        )
        has_research_heading = bool(research_heading_pattern.search(content))
        print(f"Has Research Heading:{has_research_heading}")
        if not has_research_heading:
            return False

        # Check for key research terms (case-insensitive)
        research_terms = [
            r"research showed",
            r"users understood",
            r"we found",
            r"testing showed",
            r"we observed",
            r"We tested"
        ]

        for term in research_terms:
            if re.search(term, content, re.IGNORECASE):
                return True

        return False

    def _build_extraction_prompt(
        self, file_content: str, component_url: str, parent: str = "NHS Design System"
    ) -> str:
        """
        Build the prompt for extracting component data from file content.

        Args:
            file_content: The content of the component file
            component_url: The URL where the component documentation is hosted
            parent: The parent design system name

        Returns:
            str: The formatted prompt
        """
        prompt = f"""You are an expert at extracting structured information from design system documentation.

Analyze the following component documentation file and extract the key information into a structured JSON format.

Component URL: {component_url}
Parent Design System: {parent}

IMPORTANT INSTRUCTIONS:
1. Extract the title from the pageTitle variable or main heading
2. Extract the description from the pageDescription variable or first paragraph
3. Determine the accessibility level (AA or AAA) from any WCAG mentions in the content
4. Extract the date from the dateUpdated variable, converting it to "YYYY-MM-DD HH:MM:SS" format (use 00:00:00 for time if not specified)
5. Use the same date for both created_at and updated_at
6. Set views to 0
7. Return a valid JSON object with this exact structure:

{{
  "message": "Successfully extracted component data",
  "components": [
    {{
      "title": "Component Title",
      "url": "{component_url}",
      "description": "Component description",
      "parent": "{parent}",
      "accessability": "AA",
      "created_at": "2026-03-04 10:55:00",
      "updated_at": "2026-03-04 10:55:00",
      "has_research": false,
      "views": 0
    }}
  ]
}}

COMPONENT FILE CONTENT:
{file_content}

Return ONLY the JSON object, no additional text or explanation."""
        return prompt

    def extract_from_file(
        self,
        file_path: Path,
        component_url: str,
        parent: str = "NHS Design System",
    ) -> LLMResponse:
        """
        Extract component data from a file using LLM.

        Args:
            file_path: Path to the component file
            component_url: URL where the component is hosted
            parent: Name of the parent design system

        Returns:
            LLMResponse: Extracted component data

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If LLM response cannot be parsed
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Read file content
        file_content = file_path.read_text()

        print("------------------ FILE CONTENT --------------------")
        print (file_content)
        print("------------------ HAS RESEARCH ? --------------------")

        # Check for research using rule-based method
        has_research = self._check_has_research(file_content)
        print (has_research)
        print("------------------ PROMPT --------------------")        

        # Build prompt and query LLM
        prompt = self._build_extraction_prompt(file_content, component_url, parent)

        print (prompt)

        print("------------------ RESPONSE --------------------")        


        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that extracts structured data from documentation. Always respond with valid JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,  # Low temperature for consistent extraction
                max_tokens=1000,
            )

            # Extract JSON from response
            llm_output = response.choices[0].message.content.strip()

            # Try to find JSON in the response (in case LLM adds extra text)
            json_match = re.search(r"\{.*\}", llm_output, re.DOTALL)
            if json_match:
                llm_output = json_match.group(0)

            # Parse JSON response
            data = json.loads(llm_output)

            # Override has_research with our rule-based check
            if data.get("components") and len(data["components"]) > 0:
                data["components"][0]["has_research"] = has_research

            # Validate and return
            return LLMResponse(**data)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"LLM response was: {llm_output}")
            raise ValueError(f"LLM response was not valid JSON: {e}")

        except Exception as e:
            logger.error(f"Error querying LLM: {e}")
            raise

    def extract_from_content(
        self,
        content: str,
        component_url: str,
        parent: str = "NHS Design System",
    ) -> LLMResponse:
        """
        Extract component data from content string using LLM.

        Args:
            content: The component file content as string
            component_url: URL where the component is hosted
            parent: Name of the parent design system

        Returns:
            LLMResponse: Extracted component data

        Raises:
            ValueError: If LLM response cannot be parsed
        """
        # Check for research using rule-based method
        has_research = self._check_has_research(content)

        # Build prompt and query LLM
        prompt = self._build_extraction_prompt(content, component_url, parent)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that extracts structured data from documentation. Always respond with valid JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,  # Low temperature for consistent extraction
                max_tokens=1000,
            )

            # Extract JSON from response
            llm_output = response.choices[0].message.content.strip()

            # Try to find JSON in the response (in case LLM adds extra text)
            json_match = re.search(r"\{.*\}", llm_output, re.DOTALL)
            if json_match:
                llm_output = json_match.group(0)

            # Parse JSON response
            data = json.loads(llm_output)

            # Override has_research with our rule-based check
            if data.get("components") and len(data["components"]) > 0:
                data["components"][0]["has_research"] = has_research

            # Validate and return
            return LLMResponse(**data)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"LLM response was: {llm_output}")
            raise ValueError(f"LLM response was not valid JSON: {e}")

        except Exception as e:
            logger.error(f"Error querying LLM: {e}")
            raise


__all__ = ["ComponentExtractor", "ComponentData", "LLMResponse"]
