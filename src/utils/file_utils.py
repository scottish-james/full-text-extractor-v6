"""
File Utilities Module

This module provides utility functions for file operations.
"""

import os
from config import FILE_FORMATS


def get_file_extension(filename):
    """
    Extract the file extension from a filename.

    Args:
        filename (str): The filename to extract the extension from

    Returns:
        str: The file extension (without the dot) or empty string if no extension
    """
    return filename.rsplit(".", 1)[1].lower() if "." in filename else ""


def get_supported_formats():
    """
    Get a dictionary of supported file formats categorized by type.

    Returns:
        dict: Dictionary of supported formats
    """
    return FILE_FORMATS


def is_supported_extension(filename):
    """
    Check if a file has a supported extension.

    Args:
        filename (str): The filename to check

    Returns:
        bool: True if the file extension is supported, False otherwise
    """
    ext = get_file_extension(filename)

    # Check if the extension is in any of the supported categories
    for category, info in FILE_FORMATS.items():
        if ext in info["extensions"]:
            return True

    return False


def get_all_supported_extensions():
    """
    Get a list of all supported file extensions.

    Returns:
        list: List of all supported file extensions
    """
    all_extensions = []
    for category, info in FILE_FORMATS.items():
        all_extensions.extend(info["extensions"])
    return all_extensions


def ensure_directory_exists(directory_path):
    """
    Ensure that a directory exists, creating it if necessary.

    Args:
        directory_path (str): Path to the directory to check/create

    Returns:
        bool: True if the directory exists or was created, False otherwise
    """
    try:
        os.makedirs(directory_path, exist_ok=True)
        return True
    except Exception:
        return False


def safe_filename(filename):
    """
    Convert a string to a safe filename by removing/replacing invalid characters.

    Args:
        filename (str): The filename to sanitize

    Returns:
        str: A safe filename
    """
    # Replace spaces with underscores
    safe_name = filename.replace(' ', '_')

    # Replace invalid characters with underscores
    invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
    for char in invalid_chars:
        safe_name = safe_name.replace(char, '_')

    # Limit length
    if len(safe_name) > 255:
        name_part, ext_part = os.path.splitext(safe_name)
        # Limit name part to 250 chars to leave room for extension
        safe_name = name_part[:250] + ext_part

    return safe_name