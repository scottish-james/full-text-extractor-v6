"""
PowerPoint Processing Components - Module Initialization
Modular architecture for PowerPoint to Markdown conversion

ARCHITECTURE OVERVIEW:
This module implements a component-based architecture where each processor
handles a specific aspect of PowerPoint conversion. The design follows
separation of concerns with clear interfaces between components.

COMPONENT RESPONSIBILITIES:
- PowerPointProcessor: Main orchestrator, decides XML vs MarkItDown processing
- AccessibilityOrderExtractor: Handles reading order using XML analysis
- ContentExtractor: Extracts content from different PowerPoint shape types
- TextProcessor: Handles advanced text formatting with XML-driven detection
- DiagramAnalyzer: Identifies potential diagrams using v19 scoring system
- MarkdownConverter: Converts structured data to clean markdown format
- MetadataExtractor: Extracts comprehensive PowerPoint metadata

PROCESSING PIPELINE:
1. PowerPointProcessor determines XML availability
2. If XML available: Use sophisticated components pipeline
3. If XML unavailable: Fall back to MarkItDown simple conversion
4. AccessibilityOrderExtractor determines shape reading order
5. ContentExtractor processes each shape with TextProcessor
6. DiagramAnalyzer scores slides for diagram potential
7. MarkdownConverter generates final output
8. MetadataExtractor adds document context

DEPENDENCY CHAIN:
PowerPointProcessor → AccessibilityOrderExtractor → ContentExtractor → TextProcessor
                  ↓
MarkdownConverter ← DiagramAnalyzer ← MetadataExtractor

DESIGN PATTERNS:
- Strategy Pattern: XML vs MarkItDown processing strategies
- Component Pattern: Modular processors with single responsibilities
- Factory Pattern: Content extraction based on shape types
- Template Method: Common extraction pipeline with specialized steps
"""

# Import all component classes with clear separation of concerns
from .powerpoint_processor import PowerPointProcessor
from .accessibility_extractor import AccessibilityOrderExtractor
from .content_extractor import ContentExtractor
from .text_processor import TextProcessor
from .diagram_analyzer import DiagramAnalyzer
from .markdown_converter import MarkdownConverter
from .metadata_extractor import MetadataExtractor

# Import convenience functions for backward compatibility and simple usage
from .powerpoint_processor import (
    convert_pptx_to_markdown_enhanced,
    process_powerpoint_file
)

# Export everything for easy importing - maintains clean public API
__all__ = [
    # Main processor class - primary entry point for most use cases
    'PowerPointProcessor',

    # Component classes - for advanced usage requiring fine-grained control
    'AccessibilityOrderExtractor',  # Reading order extraction
    'ContentExtractor',             # Shape content processing
    'TextProcessor',                # Text formatting and XML analysis
    'DiagramAnalyzer',             # Diagram detection and scoring
    'MarkdownConverter',           # Structured data to markdown conversion
    'MetadataExtractor',           # PowerPoint metadata extraction

    # Convenience functions - for simple integration and backward compatibility
    'convert_pptx_to_markdown_enhanced',  # Simple file conversion
    'process_powerpoint_file'             # Advanced file processing with options
]

# Version and architecture metadata for debugging and compatibility
__version__ = "2.0.0"
__architecture__ = "modular"

# USAGE PATTERNS:
#
# Simple Usage (Recommended):
# from powerpoint_processor import convert_pptx_to_markdown_enhanced
# markdown = convert_pptx_to_markdown_enhanced("presentation.pptx")
#
# Advanced Usage:
# from powerpoint_processor import PowerPointProcessor
# processor = PowerPointProcessor(use_accessibility_order=True)
# markdown = processor.convert_pptx_to_markdown_enhanced("presentation.pptx")
#
# Component-Level Control:
# from powerpoint_processor import AccessibilityOrderExtractor, ContentExtractor
# extractor = AccessibilityOrderExtractor(use_accessibility_order=False)
# content = ContentExtractor()
# # Custom processing logic...
#
# CONFIGURATION OPTIONS:
# - use_accessibility_order: Use XML-based reading order vs positional order
# - convert_slide_titles: Convert bullet points to H1 headings
# - Processing automatically falls back to MarkItDown when XML unavailable
#
# ERROR HANDLING:
# All components use defensive programming with try/catch blocks around
# XML access and PowerPoint object manipulation. Failures gracefully
# degrade functionality rather than crashing the entire conversion.
#
# PERFORMANCE CONSIDERATIONS:
# - XML parsing can be memory-intensive for large presentations
# - MarkItDown fallback is faster but less accurate
# - Component architecture allows selective processing optimization
# - Lazy loading of heavy dependencies where possible