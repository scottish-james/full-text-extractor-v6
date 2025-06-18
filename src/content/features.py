"""
Feature descriptions and benefits for the application.
"""

def get_main_features():
    """Return the main feature list."""
    return {
        "multi_format": {
            "title": "Multi-Format Support",
            "description": "Convert various file types to markdown",
            "icon": "✔️"
        },
        "high_quality": {
            "title": "High-Quality Output",
            "description": "Optimised output for customers and training LLMs",
            "icon": "✔️"
        },
        "fast_conversion": {
            "title": "Fast Conversion",
            "description": "Quickly transform your files with just a few clicks",
            "icon": "✔️"
        }
    }

def get_supported_formats():
    """Return detailed supported formats information."""
    return {
        "presentations": {
            "name": "Presentations",
            "formats": ["PowerPoint (.pptx, .ppt)"],
            "note": "Optimised for reading order and structure"
        },
        "documents": {
            "name": "Documents",
            "formats": ["Word (.docx, .doc)", "PDF (.pdf)"],
            "note": "Standard text extraction"
        },
        "spreadsheets": {
            "name": "Spreadsheets",
            "formats": ["Excel (.xlsx, .xls)"],
            "note": "Table structure preservation"
        },
        "web": {
            "name": "Web Files",
            "formats": ["HTML (.html, .htm)"],
            "note": "Clean text extraction"
        },
        "data": {
            "name": "Data Files",
            "formats": ["CSV (.csv)", "JSON (.json)", "XML (.xml)"],
            "note": "Structured data conversion"
        }
    }

def get_feature_tagline():
    """Return the main tagline."""
    return "Turn your documents into markdown in minutes"

def get_tool_description():
    """Return the tool description."""
    return "This tool will help you turn your documents into markdown in minutes."
