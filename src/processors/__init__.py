"""
Document Processors Module
Refactored modular architecture with PowerPoint subfolder
"""

# PowerPoint processing (from subfolder)
from .powerpoint import (
    PowerPointProcessor,
    convert_pptx_to_markdown_enhanced,
    process_powerpoint_file,
    AccessibilityOrderExtractor,
    ContentExtractor,
    TextProcessor,
    DiagramAnalyzer,
    MarkdownConverter,
    MetadataExtractor
)

# Folder processing (existing)
from .folder_processor import process_folder, find_compatible_files

# Screenshot processing (existing)
from .diagram_screenshot_processor import DiagramScreenshotProcessor

__all__ = [
    # PowerPoint processing
    'PowerPointProcessor',
    'convert_pptx_to_markdown_enhanced',
    'process_powerpoint_file',

    # PowerPoint components (for advanced usage)
    'AccessibilityOrderExtractor',
    'ContentExtractor',
    'TextProcessor',
    'DiagramAnalyzer',
    'MarkdownConverter',
    'MetadataExtractor',

    # Folder processing
    'process_folder',
    'find_compatible_files',

    # Screenshot processing
    'DiagramScreenshotProcessor'
]