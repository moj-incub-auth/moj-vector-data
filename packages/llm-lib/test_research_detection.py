"""
Test the research detection logic.

This script demonstrates how the has_research detection works.
"""
from llm_lib import ComponentExtractor


def test_research_detection():
    """Test various scenarios for research detection."""
    extractor = ComponentExtractor()

    # Test Case 1: Has Research heading + key term "research showed"
    content1 = """
    <h2 id="research">Research</h2>
    <p>Our research showed that users preferred the green color.</p>
    """
    result1 = extractor._check_has_research(content1)
    print(f"Test 1 - Research heading + 'research showed': {result1}")
    assert result1 == True, "Should be True"

    # Test Case 2: Has Research heading + key term "users understood"
    content2 = """
    <h2>Research findings</h2>
    <p>In testing, users understood the purpose of the component.</p>
    """
    result2 = extractor._check_has_research(content2)
    print(f"Test 2 - Research findings + 'users understood': {result2}")
    assert result2 == True, "Should be True"

    # Test Case 3: Has Research heading + key term "we found"
    content3 = """
    <h3>Research</h3>
    <p>We found that the larger size was more visible.</p>
    """
    result3 = extractor._check_has_research(content3)
    print(f"Test 3 - Research heading + 'we found': {result3}")
    assert result3 == True, "Should be True"

    # Test Case 4: Has Research heading + key term "testing showed"
    content4 = """
    <h2>Research</h2>
    <p>Testing showed significant improvements in user satisfaction.</p>
    """
    result4 = extractor._check_has_research(content4)
    print(f"Test 4 - Research heading + 'testing showed': {result4}")
    assert result4 == True, "Should be True"

    # Test Case 5: Has Research heading + key term "we observed"
    content5 = """
    <h2>Research</h2>
    <p>During our tests, we observed that users completed tasks faster.</p>
    """
    result5 = extractor._check_has_research(content5)
    print(f"Test 5 - Research heading + 'we observed': {result5}")
    assert result5 == True, "Should be True"

    # Test Case 6: Has Research heading but NO key terms
    content6 = """
    <h2>Research</h2>
    <p>More research is needed on this component.</p>
    """
    result6 = extractor._check_has_research(content6)
    print(f"Test 6 - Research heading but NO key terms: {result6}")
    assert result6 == False, "Should be False"

    # Test Case 7: Has key terms but NO Research heading
    content7 = """
    <h2>Overview</h2>
    <p>Research showed this was effective, but we need more data.</p>
    """
    result7 = extractor._check_has_research(content7)
    print(f"Test 7 - Key terms but NO Research heading: {result7}")
    assert result7 == False, "Should be False"

    # Test Case 8: No Research heading and no key terms
    content8 = """
    <h2>How to use</h2>
    <p>Use this component to display information.</p>
    """
    result8 = extractor._check_has_research(content8)
    print(f"Test 8 - No Research heading and no key terms: {result8}")
    assert result8 == False, "Should be False"

    # Test Case 9: Case insensitive matching
    content9 = """
    <h2>RESEARCH</h2>
    <p>TESTING SHOWED that USERS UNDERSTOOD the design.</p>
    """
    result9 = extractor._check_has_research(content9)
    print(f"Test 9 - Case insensitive: {result9}")
    assert result9 == True, "Should be True"

    # Test Case 10: Real example from NHS file
    content10 = """
    <h2 id="research">Research</h2>
    <p>We tested the action links on health information pages with lots of content, callout boxes and multi-page navigation.</p>
    <p>Users didn't notice early versions, so we made the size of the text larger than body text size.</p>
    <p>We used NHS blue first but users didn't notice it. So we changed the arrow colour to green (our "action" colour). Users seemed to see the green better.</p>
    <p>In follow-up tests on busy content pages, users pointed out the action links and said they found them useful.</p>
    """
    result10 = extractor._check_has_research(content10)
    print(f"Test 10 - Real NHS example: {result10}")
    # Note: This should be False because it doesn't have the specific key terms
    assert result10 == False, "Should be False (no specific key terms)"

    # Test Case 11: Real example with key term
    content11 = """
    <h2 id="research">Research</h2>
    <p>We tested the action links and research showed they were effective.</p>
    """
    result11 = extractor._check_has_research(content11)
    print(f"Test 11 - NHS example with key term: {result11}")
    assert result11 == True, "Should be True"

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_research_detection()
