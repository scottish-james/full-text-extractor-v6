"""
Tests for the hyperlink extractor module.
"""

import unittest
from unittest.mock import patch, MagicMock
from src.converters.hyperlink_extractor import fix_url, format_hyperlinks_section


class TestHyperlinkExtractor(unittest.TestCase):
    """Test cases for hyperlink extractor functionality."""

    def test_fix_url_email(self):
        """Test fixing email URLs."""
        email = "test@example.com"
        fixed = fix_url(email)
        self.assertEqual(fixed, "mailto:test@example.com")

    def test_fix_url_web(self):
        """Test fixing web URLs."""
        # Without scheme
        web_url = "www.example.com"
        fixed = fix_url(web_url)
        self.assertEqual(fixed, "https://www.example.com")

        # With domain
        domain_url = "example.com"
        fixed = fix_url(domain_url)
        self.assertEqual(fixed, "https://example.com")

        # With scheme already
        https_url = "https://example.com"
        fixed = fix_url(https_url)
        self.assertEqual(fixed, "https://example.com")

    def test_fix_url_empty(self):
        """Test fixing empty URLs."""
        empty_url = ""
        fixed = fix_url(empty_url)
        self.assertEqual(fixed, "")

        none_url = None
        fixed = fix_url(none_url)
        self.assertEqual(fixed, None)

    def test_format_hyperlinks_section_empty(self):
        """Test formatting empty hyperlinks."""
        empty_links = []
        result = format_hyperlinks_section(empty_links)
        self.assertEqual(result, "")

    def test_format_hyperlinks_section_basic(self):
        """Test basic hyperlink formatting."""
        links = [
            {"text": "Example Link", "url": "https://example.com", "page": 1},
            {"text": "Another Link", "url": "https://example.org", "page": 1},
            {"text": "Page 2 Link", "url": "https://test.com", "page": 2}
        ]

        result = format_hyperlinks_section(links, "Document")

        # Check section header
        self.assertIn("## Hyperlinks in Document", result)

        # Check page headers
        self.assertIn("### Page 1", result)
        self.assertIn("### Page 2", result)

        # Check link formatting
        self.assertIn("* [Example Link](https://example.com)", result)
        self.assertIn("* [Another Link](https://example.org)", result)
        self.assertIn("* [Page 2 Link](https://test.com)", result)

    def test_format_hyperlinks_section_presentation(self):
        """Test presentation hyperlink formatting."""
        links = [
            {"text": "Slide 1 Link", "url": "https://example.com", "slide": 1},
            {"text": "Slide 2 Link", "url": "https://test.com", "slide": 2}
        ]

        result = format_hyperlinks_section(links, "Presentation")

        # Check section header
        self.assertIn("## Hyperlinks in Presentation", result)

        # Check slide headers instead of page headers
        self.assertIn("### Slide 1", result)
        self.assertIn("### Slide 2", result)

        # Check link formatting
        self.assertIn("* [Slide 1 Link](https://example.com)", result)
        self.assertIn("* [Slide 2 Link](https://test.com)", result)

    def test_format_hyperlinks_section_duplicates(self):
        """Test handling of duplicate links."""
        # Same URL on same page, different text
        links = [
            {"text": "Short Text", "url": "https://example.com", "page": 1},
            {"text": "Longer and better description", "url": "https://example.com", "page": 1},
            {"text": "Page 2 Link", "url": "https://example.com", "page": 2}
        ]

        result = format_hyperlinks_section(links)

        # Should keep the longer text for page 1
        self.assertIn("* [Longer and better description](https://example.com)", result)
        self.assertNotIn("* [Short Text](https://example.com)", result)

        # Should keep the link on page 2
        self.assertIn("### Page 2", result)
        self.assertIn("* [Page 2 Link](https://example.com)", result)

    def test_format_hyperlinks_missing_fields(self):
        """Test handling of links with missing fields."""
        links = [
            {"text": "Valid Link", "url": "https://example.com", "page": 1},
            {"text": "", "url": "https://empty-text.com", "page": 1},
            {"text": "No URL", "url": "", "page": 1},
            {"text": "No Page", "url": "https://no-page.com"},
            {}  # Completely empty
        ]

        result = format_hyperlinks_section(links)

        # Should only include the valid link
        self.assertIn("* [Valid Link](https://example.com)", result)
        self.assertNotIn("empty-text.com", result)
        self.assertNotIn("No URL", result)
        self.assertNotIn("no-page.com", result)


if __name__ == '__main__':
    unittest.main()