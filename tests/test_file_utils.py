"""
Tests for the file utilities module.
"""

import unittest
import os
import tempfile
from unittest.mock import patch, MagicMock
from src.utils.file_utils import (
    get_file_extension,
    get_supported_formats,
    is_supported_extension,
    get_all_supported_extensions,
    ensure_directory_exists,
    safe_filename
)


class TestFileUtils(unittest.TestCase):
    """Test cases for file utility functions."""

    def test_get_file_extension(self):
        """Test extracting file extensions."""
        # Basic extension
        self.assertEqual(get_file_extension("test.txt"), "txt")

        # Multiple dots
        self.assertEqual(get_file_extension("test.doc.pdf"), "pdf")

        # No extension
        self.assertEqual(get_file_extension("noextension"), "")

        # Multiple extensions
        self.assertEqual(get_file_extension("archive.tar.gz"), "gz")

        # Uppercase extension
        self.assertEqual(get_file_extension("test.PDF"), "pdf")

        # Path with extension
        self.assertEqual(get_file_extension("/path/to/file.docx"), "docx")

    def test_get_supported_formats(self):
        """Test getting supported formats."""
        formats = get_supported_formats()

        # Check that the structure is correct
        self.assertIsInstance(formats, dict)

        # Check for expected categories
        self.assertIn("📝 Documents", formats)
        self.assertIn("📊 Spreadsheets", formats)
        self.assertIn("📊 Presentations", formats)

        # Check structure of a category
        category = formats["📝 Documents"]
        self.assertIn("formats", category)
        self.assertIn("extensions", category)

        # Check for specific extensions
        self.assertIn("docx", category["extensions"])
        self.assertIn("pdf", category["extensions"])

    def test_is_supported_extension(self):
        """Test checking if extensions are supported."""
        # Supported extensions
        self.assertTrue(is_supported_extension("test.docx"))
        self.assertTrue(is_supported_extension("data.xlsx"))
        self.assertTrue(is_supported_extension("slides.pptx"))
        self.assertTrue(is_supported_extension("page.html"))

        # Unsupported extensions
        self.assertFalse(is_supported_extension("test.unsupported"))
        self.assertFalse(is_supported_extension("noextension"))

        # Case insensitivity
        self.assertTrue(is_supported_extension("test.DOCX"))
        self.assertTrue(is_supported_extension("test.Pdf"))

    def test_get_all_supported_extensions(self):
        """Test getting all supported extensions."""
        extensions = get_all_supported_extensions()

        # Check type
        self.assertIsInstance(extensions, list)

        # Check for specific extensions
        self.assertIn("docx", extensions)
        self.assertIn("xlsx", extensions)
        self.assertIn("pdf", extensions)
        self.assertIn("pptx", extensions)
        self.assertIn("html", extensions)

        # Check that all are lowercase
        for ext in extensions:
            self.assertEqual(ext, ext.lower())

    def test_ensure_directory_exists(self):
        """Test ensuring a directory exists."""
        # Test with temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test creating a subdirectory
            test_dir = os.path.join(temp_dir, "testdir")

            # Should return True when creating the directory
            self.assertTrue(ensure_directory_exists(test_dir))

            # Directory should now exist
            self.assertTrue(os.path.isdir(test_dir))

            # Should return True for existing directory
            self.assertTrue(ensure_directory_exists(test_dir))

    def test_safe_filename(self):
        """Test creating safe filenames."""
        # Replace spaces
        self.assertEqual(safe_filename("test file.txt"), "test_file.txt")

        # Replace invalid characters
        self.assertEqual(safe_filename("test<>:\"/\\|?*.txt"), "test_________.txt")

        # Test length limitation
        long_name = "a" * 300 + ".txt"
        safe_name = safe_filename(long_name)

        # Should be trimmed
        self.assertLessEqual(len(safe_name), 255)

        # Should keep extension
        self.assertTrue(safe_name.endswith(".txt"))


if __name__ == '__main__':
    unittest.main()