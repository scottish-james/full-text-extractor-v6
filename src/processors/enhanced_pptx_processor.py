"""
PowerPoint Processor - Simplified Facade (v2.0)
XML-first approach: Use sophisticated functions when XML available, MarkItDown when not.

This file maintains backward compatibility while using the new simplified architecture.
"""

# Import from PowerPoint module with new architecture
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

# Re-export everything for backward compatibility
__all__ = [
    # Main processor class
    'PowerPointProcessor',

    # Convenience functions (maintain exact same API)
    'convert_pptx_to_markdown_enhanced',
    'process_powerpoint_file',

    # Component classes (for advanced usage)
    'AccessibilityOrderExtractor',
    'ContentExtractor',
    'TextProcessor',
    'DiagramAnalyzer',
    'MarkdownConverter',
    'MetadataExtractor'
]


# Convenience function for debugging (new feature)
def debug_powerpoint_reading_order(file_path, slide_number=1):
    """
    Debug reading order extraction for a specific slide.

    Args:
        file_path (str): Path to PowerPoint file
        slide_number (int): Slide number to debug (1-based)
    """
    processor = PowerPointProcessor()
    processor.debug_accessibility_order(file_path, slide_number)


# Convenience function for processing summary (new feature)
def get_powerpoint_processing_summary(file_path):
    """
    Get a summary of what will be processed without full conversion.

    Args:
        file_path (str): Path to PowerPoint file

    Returns:
        dict: Processing summary information
    """
    processor = PowerPointProcessor()
    return processor.get_processing_summary(file_path)


# Check if file will use XML or MarkItDown processing
def check_processing_method(file_path):
    """
    Check which processing method will be used for a file.

    Args:
        file_path (str): Path to PowerPoint file

    Returns:
        dict: Processing method information
    """
    processor = PowerPointProcessor()
    has_xml = processor._has_xml_access(file_path)

    return {
        "file_path": file_path,
        "has_xml_access": has_xml,
        "will_use": "sophisticated_xml_processing" if has_xml else "markitdown_fallback",
        "benefits": {
            "sophisticated": [
                "Accessibility-based reading order",
                "Advanced text formatting and bullet detection",
                "Diagram analysis and scoring",
                "Comprehensive metadata extraction"
            ] if has_xml else [],
            "fallback": [
                "Simple, reliable conversion",
                "Handles most PowerPoint features",
                "No complex dependencies"
            ] if not has_xml else []
        }
    }


# Add new functions to exports
__all__.extend([
    'debug_powerpoint_reading_order',
    'get_powerpoint_processing_summary',
    'check_processing_method'
])

# Version information
__version__ = "2.0.0"
__architecture__ = "xml_first_with_markitdown_fallback"

# Module docstring for users
__doc__ = """
PowerPoint to Markdown Processor (XML-First Architecture v2.0)

This module provides PowerPoint to Markdown conversion with intelligent processing:

**XML Available (Sophisticated Processing):**
- Accessibility-based reading order extraction
- Advanced text formatting and bullet detection  
- Diagram analysis and scoring
- Comprehensive metadata extraction
- Clean, testable modular architecture

**XML Not Available (MarkItDown Fallback):**
- Simple, reliable MarkItDown conversion
- Handles most PowerPoint features correctly
- No complex processing dependencies

Basic Usage:
    from src.processors.enhanced_pptx_processor import convert_pptx_to_markdown_enhanced

    markdown = convert_pptx_to_markdown_enhanced("presentation.pptx")

Check Processing Method:
    from src.processors.enhanced_pptx_processor import check_processing_method

    method_info = check_processing_method("presentation.pptx")
    print(f"Will use: {method_info['will_use']}")

Advanced Usage:
    from src.processors.enhanced_pptx_processor import PowerPointProcessor

    processor = PowerPointProcessor(use_accessibility_order=True)
    summary = processor.get_processing_summary("presentation.pptx")
    markdown = processor.convert_pptx_to_markdown_enhanced("presentation.pptx")

New in v2.0:
- XML-first architecture with MarkItDown fallback
- Simplified detection logic (XML-driven instead of pattern matching)
- Enhanced reliability and maintainability
- Same functionality, cleaner implementation
- Processing method transparency
"""