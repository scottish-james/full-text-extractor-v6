"""
Folder Processor Module - Optimized for Claude Sonnet 4

This module handles batch processing of files in a folder with priority given to PowerPoint files.
"""

import os
import glob
from src.converters.file_converter import convert_file_to_markdown
from src.utils.file_utils import get_supported_formats, get_all_supported_extensions, ensure_directory_exists
from config import DEFAULT_MARKDOWN_SUBFOLDER, PROCESSING_PRIORITIES


def process_folder(folder_path, output_folder=None, enhance=True, api_key=None):
    """
    Process all compatible files in a folder and convert them to markdown using Claude Sonnet 4.
    PowerPoint files are prioritized for processing.

    Args:
        folder_path (str): Path to folder containing files to convert
        output_folder (str): Path to save markdown files (defaults to subfolder in input folder)
        enhance (bool): Whether to enhance markdown with Claude
        api_key (str): Anthropic API key for Claude enhancement

    Yields:
        Various progress updates and final results
    """
    # Setup output folder
    if not output_folder:
        output_folder = os.path.join(folder_path, DEFAULT_MARKDOWN_SUBFOLDER)

    # Create output folder if it doesn't exist
    ensure_directory_exists(output_folder)

    # Get all supported file extensions
    extensions = get_all_supported_extensions()

    # Track results
    success_count = 0
    error_count = 0
    errors = {}

    # Get all compatible files
    files_to_process = []
    for ext in extensions:
        files_to_process.extend(glob.glob(os.path.join(folder_path, f"*.{ext}")))

    # Sort files by processing priority (PowerPoint first)
    files_to_process.sort(key=lambda f: get_processing_priority(f))

    total_files = len(files_to_process)

    # Early exit if no files found
    if total_files == 0:
        yield 1.0, "No compatible files found in folder"
        yield 0, 0, {}
        return

    # Process each file
    for i, file_path in enumerate(files_to_process):
        file_name = os.path.basename(file_path)
        file_ext = get_file_extension(file_name)

        try:
            # Update progress with file type indication
            progress = (i + 1) / total_files
            file_type = "PowerPoint" if file_ext.lower() in ["pptx", "ppt"] else "Document"
            yield progress, f"Processing {file_type}: {file_name} ({i + 1}/{total_files})"

            # Read file content
            with open(file_path, 'rb') as file:
                file_data = file.read()

            # Convert to markdown using Claude
            markdown_content, error = convert_file_to_markdown(
                file_data,
                file_name,
                enhance=enhance,
                api_key=api_key
            )

            if error:
                error_count += 1
                errors[file_name] = error
                continue

            # Save markdown content
            output_file = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}.md")
            with open(output_file, 'w', encoding='utf-8') as md_file:
                md_file.write(markdown_content)

            success_count += 1

        except Exception as e:
            error_count += 1
            errors[file_name] = str(e)

    # Return final results
    yield success_count, error_count, errors


def get_processing_priority(file_path):
    """
    Get the processing priority for a file based on its extension.
    Lower numbers = higher priority (PowerPoint files first).

    Args:
        file_path (str): Path to the file

    Returns:
        int: Priority number (lower = higher priority)
    """
    file_ext = get_file_extension(os.path.basename(file_path))
    return PROCESSING_PRIORITIES.get(file_ext.lower(), 999)


def get_file_extension(filename):
    """Extract file extension from filename."""
    return filename.rsplit(".", 1)[1].lower() if "." in filename else ""


def find_compatible_files(directory_path):
    """
    Find all compatible files in a directory, organized by category with PowerPoint prioritized.

    Args:
        directory_path (str): The directory to search

    Returns:
        dict: Dictionary of files by category
    """
    formats = get_supported_formats()
    result = {}

    # Initialize result categories
    for category in formats.keys():
        result[category] = []

    # Only process if directory exists
    if not os.path.isdir(directory_path):
        return result

    # Find all files by extension
    for category, info in formats.items():
        for ext in info["extensions"]:
            files = glob.glob(os.path.join(directory_path, f"*.{ext}"))
            for file_path in files:
                file_info = {
                    "path": file_path,
                    "name": os.path.basename(file_path),
                    "extension": ext,
                    "priority": get_processing_priority(file_path),
                    "optimized": ext.lower() in ["pptx", "ppt"]  # Mark PowerPoint as optimized
                }
                result[category].append(file_info)

    # Sort each category by priority (PowerPoint first)
    for category in result:
        result[category].sort(key=lambda x: x["priority"])

    return result


def get_folder_statistics(directory_path):
    """
    Get statistics about compatible files in a folder.

    Args:
        directory_path (str): The directory to analyze

    Returns:
        dict: Statistics about the folder contents
    """
    if not os.path.isdir(directory_path):
        return {"error": "Directory does not exist"}

    files_by_category = find_compatible_files(directory_path)

    stats = {
        "total_files": 0,
        "powerpoint_files": 0,
        "other_files": 0,
        "categories": {},
        "estimated_processing_time": 0
    }

    for category, files in files_by_category.items():
        file_count = len(files)
        stats["categories"][category] = file_count
        stats["total_files"] += file_count

        # Count PowerPoint files separately
        for file_info in files:
            if file_info["optimized"]:
                stats["powerpoint_files"] += 1
            else:
                stats["other_files"] += 1

    # Estimate processing time (rough approximation)
    # PowerPoint files: 30 seconds each, Others: 15 seconds each
    stats["estimated_processing_time"] = (
            stats["powerpoint_files"] * 30 +
            stats["other_files"] * 15
    )

    return stats