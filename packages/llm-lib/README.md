# llm-lib

LLM integration library for extracting structured component data from design system documentation.

## Features

- **LLM-powered extraction**: Uses qwen3-14b-llm to extract structured data from component files
- **Rule-based research detection**: Automatically detects research sections with specific criteria
- **Pydantic models**: Type-safe data structures for component information
- **Flexible input**: Extract from files or content strings
- **JSON output**: Returns structured JSON matching the required format

## Installation

```bash
uv pip install -e packages/llm-lib
```

## Usage

### Basic Example

```python
from pathlib import Path
from llm_lib import ComponentExtractor

# Initialize the extractor
extractor = ComponentExtractor(
    model="qwen3-14b-llm",
    base_url="your-llm-endpoint",  # Optional
    api_key="your-api-key"  # Optional
)

# Extract from a file
result = extractor.extract_from_file(
    file_path=Path("path/to/component/index.njk"),
    component_url="https://service-manual.nhs.uk/design-system/components/action-link",
    parent="NHS Design System"
)

# Access the data
for component in result.components:
    print(f"Title: {component.title}")
    print(f"Has Research: {component.has_research}")

# Get as JSON
json_output = result.model_dump_json(indent=2)
```

### Extract from Content String

```python
content = """
{% set pageTitle = "Action link" %}
{% set pageDescription = "Use action links..." %}
{% set dateUpdated = "November 2025" %}

<h2>Research</h2>
<p>Testing showed users found this helpful.</p>
"""

result = extractor.extract_from_content(
    content=content,
    component_url="https://example.com/component",
    parent="NHS Design System"
)
```

## Research Detection

The `has_research` field is determined using rule-based logic:

**Returns `True` if:**
1. Document contains a heading "Research" OR "Research findings" (case-insensitive)
2. **AND** contains one or more of these key terms:
   - "research showed"
   - "users understood"
   - "we found"
   - "testing showed"
   - "we observed"

## Output Format

```json
{
  "message": "Successfully extracted component data",
  "components": [
    {
      "title": "Action link",
      "url": "https://service-manual.nhs.uk/design-system/components/action-link",
      "description": "Use action links to help users get to the next stage of a journey quickly by signposting the start of a digital service.",
      "parent": "NHS Design System",
      "accessability": "AA",
      "created_at": "2025-11-01 00:00:00",
      "updated_at": "2025-11-01 00:00:00",
      "has_research": true,
      "views": 0
    }
  ]
}
```

## Environment Variables

- `OPENAI_BASE_URL`: Base URL for the LLM API endpoint
- `OPENAI_API_KEY`: API key for authentication

## Models

### ComponentData

Fields:
- `title` (str): Component title
- `url` (str): Component documentation URL
- `description` (str): Component description
- `parent` (str): Parent design system name
- `accessibility` (str): Accessibility level (AA/AAA)
- `created_at` (str): Creation date in "YYYY-MM-DD HH:MM:SS" format
- `updated_at` (str): Update date in "YYYY-MM-DD HH:MM:SS" format
- `has_research` (bool): Whether component has research
- `views` (int): View count (default: 0)

### LLMResponse

Fields:
- `message` (str): Response message
- `components` (list[ComponentData]): List of extracted components
