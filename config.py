"""
PowerPoint to Markdown - Configuration Settings

Optimized for Claude Sonnet 4 processing with focus on PowerPoint presentations.
"""

# App Settings
APP_TITLE = "Document to Markdown"
APP_ICON = ""
APP_LAYOUT = "centered"

# UI Settings
UI_THEME_COLOR = "#4e54c8"
UI_BACKGROUND_COLOR = "#f5f7f9"

# File Settings
DEFAULT_MARKDOWN_SUBFOLDER = "markdown"

# Claude API Configuration
CLAUDE_MODEL = "claude-sonnet-4-20250514"
CLAUDE_MAX_TOKENS = 4096
CLAUDE_TEMPERATURE = 0.1  # Low temperature for consistent formatting

# API Settings
DEFAULT_TIMEOUT = 30  # seconds
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

# Content Processing Settings
MAX_CONTENT_LENGTH = 100000  # Maximum characters to send to Claude
CHUNK_SIZE = 50000  # If content is too large, process in chunks

# Hyperlink Extraction Settings
HYPERLINK_CONTEXT_SIZE = 10  # Words to extract around a hyperlink for context

# File Format Categories - Organized with PowerPoint as primary focus
FILE_FORMATS = {
    "🎯 PowerPoint (Optimized)": {
        "formats": ["PowerPoint (.pptx, .ppt) with advanced formatting preservation"],
        "extensions": ["pptx", "ppt"],
    },
    "📝 Documents": {
        "formats": ["Word (.docx, .doc)", "PDF (with hyperlink extraction)", "EPub"],
        "extensions": ["docx", "doc", "pdf", "epub"],
    },
    "📊 Spreadsheets": {
        "formats": ["Excel (.xlsx, .xls)"],
        "extensions": ["xlsx", "xls"],
    },
    "🌐 Web": {
        "formats": ["HTML"],
        "extensions": ["html", "htm"]
    },
    "📁 Others": {
        "formats": ["CSV", "JSON", "XML", "ZIP (iterates over contents)"],
        "extensions": ["csv", "json", "xml", "zip"],
    },
}

# User Agent for web requests (if needed for HTML processing)
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

# Claude Enhancement Settings
CLAUDE_ENHANCEMENT_ENABLED = True
USE_CLAUDE_FOR_ENHANCEMENT = True

# Environment variable for API key
ANTHROPIC_API_KEY_ENV_VAR = "ANTHROPIC_API_KEY"

# PowerPoint Optimization Settings
POWERPOINT_OPTIMIZED = True
PREFER_POWERPOINT_PROCESSING = True
EXTRACT_HYPERLINKS = True
PRESERVE_FORMATTING = True
INTELLIGENT_HIERARCHY = True

# Processing Priorities (for folder processing)
PROCESSING_PRIORITIES = {
    "pptx": 1,  # Highest priority
    "ppt": 1,
    "docx": 2,
    "doc": 2,
    "pdf": 3,
    "xlsx": 4,
    "xls": 4,
    "html": 5,
    "htm": 5,
    "csv": 6,
    "json": 6,
    "xml": 6,
    "zip": 7
}