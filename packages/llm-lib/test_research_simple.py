"""
Simple test of the research detection logic without requiring dependencies.
"""
import re


def check_has_research(content: str) -> bool:
    """
    Check if the document contains research based on specific criteria.

    Returns True if the document contains a heading "Research" OR "Research findings"
    AND one or more of the key terms:
    - "research showed"
    - "users understood"
    - "we found"
    - "testing showed"
    - "we observed"
    """
    # Check for research headings (case-insensitive)
    research_heading_pattern = re.compile(
        r"<h[1-6][^>]*>\s*(?:Research|Research findings)\s*</h[1-6]>", re.IGNORECASE
    )
    has_research_heading = bool(research_heading_pattern.search(content))

    if not has_research_heading:
        return False

    # Check for key research terms (case-insensitive)
    research_terms = [
        r"research showed",
        r"users understood",
        r"we found",
        r"testing showed",
        r"we observed",
    ]

    for term in research_terms:
        if re.search(term, content, re.IGNORECASE):
            return True

    return False


def test_research_detection():
    """Test various scenarios for research detection."""

    # Test Case 1: Has Research heading + key term "research showed"
    content1 = """
    <h2 id="research">Research</h2>
    <p>Our research showed that users preferred the green color.</p>
    """
    result1 = check_has_research(content1)
    print(f"✓ Test 1 - Research heading + 'research showed': {result1} (expected: True)")
    assert result1 == True

    # Test Case 2: Has Research heading + key term "users understood"
    content2 = """
    <h2>Research findings</h2>
    <p>In testing, users understood the purpose of the component.</p>
    """
    result2 = check_has_research(content2)
    print(f"✓ Test 2 - Research findings + 'users understood': {result2} (expected: True)")
    assert result2 == True

    # Test Case 3: Has Research heading + key term "we found"
    content3 = """
    <h3>Research</h3>
    <p>We found that the larger size was more visible.</p>
    """
    result3 = check_has_research(content3)
    print(f"✓ Test 3 - Research heading + 'we found': {result3} (expected: True)")
    assert result3 == True

    # Test Case 4: Has Research heading + key term "testing showed"
    content4 = """
    <h2>Research</h2>
    <p>Testing showed significant improvements in user satisfaction.</p>
    """
    result4 = check_has_research(content4)
    print(f"✓ Test 4 - Research heading + 'testing showed': {result4} (expected: True)")
    assert result4 == True

    # Test Case 5: Has Research heading + key term "we observed"
    content5 = """
    <h2>Research</h2>
    <p>During our tests, we observed that users completed tasks faster.</p>
    """
    result5 = check_has_research(content5)
    print(f"✓ Test 5 - Research heading + 'we observed': {result5} (expected: True)")
    assert result5 == True

    # Test Case 6: Has Research heading but NO key terms
    content6 = """
    <h2>Research</h2>
    <p>More research is needed on this component.</p>
    """
    result6 = check_has_research(content6)
    print(f"✓ Test 6 - Research heading but NO key terms: {result6} (expected: False)")
    assert result6 == False

    # Test Case 7: Has key terms but NO Research heading
    content7 = """
    <h2>Overview</h2>
    <p>Research showed this was effective, but we need more data.</p>
    """
    result7 = check_has_research(content7)
    print(f"✓ Test 7 - Key terms but NO Research heading: {result7} (expected: False)")
    assert result7 == False

    # Test Case 8: Case insensitive matching
    content8 = """
    <h2>RESEARCH</h2>
    <p>TESTING SHOWED that USERS UNDERSTOOD the design.</p>
    """
    result8 = check_has_research(content8)
    print(f"✓ Test 8 - Case insensitive: {result8} (expected: True)")
    assert result8 == True

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_research_detection()
