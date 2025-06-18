"""
Tests for the file converter module.
"""

import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from src.converters.file_converter import convert_file_to_markdown, enhance_markdown_formatting


class TestFileConverter(unittest.TestCase):
    """Test cases for file converter functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary dummy file for testing
        self.test_content = b"Test document content"
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
        self.temp_file.write(self.test_content)
        self.temp_file.close()

    def tearDown(self):
        """Clean up after tests."""
        # Remove the temporary file
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)

    @patch('src.converters.file_converter.MarkItDown')
    def test_convert_file_to_markdown_basic(self, mock_markitdown):
        """Test basic file conversion without enhancement."""
        # Mock MarkItDown instance
        mock_instance = MagicMock()
        mock_markitdown.return_value = mock_instance

        # Mock convert result
        mock_result = MagicMock()
        mock_result.text_content = "# Converted Markdown"
        mock_instance.convert.return_value = mock_result

        # Test the conversion
        markdown, error = convert_file_to_markdown(
            self.test_content,
            "test.txt",
            enhance=False
        )

        # Assertions
        self.assertIsNone(error)
        self.assertEqual(markdown, "# Converted Markdown")
        mock_instance.convert.assert_called_once()

    @patch('src.converters.file_converter.enhance_markdown_formatting')
    @patch('src.converters.file_converter.MarkItDown')
    def test_convert_file_to_markdown_with_enhancement(self, mock_markitdown, mock_enhance):
        """Test file conversion with enhancement."""
        # Mock MarkItDown instance
        mock_instance = MagicMock()
        mock_markitdown.return_value = mock_instance

        # Mock convert result
        mock_result = MagicMock()
        mock_result.text_content = "# Basic Markdown"
        mock_instance.convert.return_value = mock_result

        # Mock enhancement
        mock_enhance.return_value = ("# Enhanced Markdown", None)

        # Test the conversion
        markdown, error = convert_file_to_markdown(
            self.test_content,
            "test.txt",
            enhance=True,
            api_key="test_api_key"
        )

        # Assertions
        self.assertIsNone(error)
        self.assertEqual(markdown, "# Enhanced Markdown")
        mock_enhance.assert_called_once_with("# Basic Markdown", "test_api_key")

    @patch('src.converters.file_converter.OpenAI')
    def test_enhance_markdown_formatting(self, mock_openai):
        """Test markdown enhancement with OpenAI."""
        # Mock OpenAI client and response
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_response = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "# Enhanced Markdown"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        mock_client.chat.completions.create.return_value = mock_response

        # Test enhancement
        enhanced, error = enhance_markdown_formatting("# Basic Markdown", "test_api_key")

        # Assertions
        self.assertIsNone(error)
        self.assertEqual(enhanced, "# Enhanced Markdown")
        mock_openai.assert_called_once_with(api_key="test_api_key")
        mock_client.chat.completions.create.assert_called_once()

    def test_enhance_markdown_no_api_key(self):
        """Test enhancement with no API key."""
        # Test enhancement without API key
        enhanced, error = enhance_markdown_formatting("# Basic Markdown", None)

        # Assertions
        self.assertEqual(enhanced, "# Basic Markdown")
        self.assertEqual(error, "No OpenAI API key provided")

    @patch('os.unlink')
    @patch('src.converters.file_converter.MarkItDown')
    def test_convert_file_error_handling(self, mock_markitdown, mock_unlink):
        """Test error handling during conversion."""
        # Mock MarkItDown to raise an exception
        mock_instance = MagicMock()
        mock_markitdown.return_value = mock_instance
        mock_instance.convert.side_effect = Exception("Conversion error")

        # Test conversion with an error
        markdown, error = convert_file_to_markdown(
            self.test_content,
            "test.txt"
        )

        # Assertions
        self.assertEqual(markdown, "")
        self.assertEqual(error, "Conversion error")
        mock_unlink.assert_called_once()


if __name__ == '__main__':
    unittest.main()