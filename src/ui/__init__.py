"""
UI Components Module
This module contains UI components and utilities for the Streamlit interface.
"""

# Core UI components
from src.ui.components import setup_page_config, display_supported_formats
from src.ui.sidebar import setup_enhanced_sidebar

# Content display components
from src.ui.main_content import render_welcome_section, render_main_features, render_app_header
from src.ui.output_display import (
    display_output_section,
    display_enhanced_output_section,
    display_content_statistics,
    display_download_options,
    set_output_content,
    get_output_content,
    has_output_content
)

# File processing components
from src.ui.file_upload import (
    render_file_upload_section,
    render_enhanced_file_upload,
    create_file_uploader,
    handle_file_conversion
)

# Folder processing components
from src.ui.folder_processing import (
    render_folder_processing_section,
    render_enhanced_folder_processing,
    display_folder_preview,
    execute_folder_processing
)

# Results display components
from src.ui.folder_results import (
    display_folder_results,
    display_processing_summary,
    set_folder_results,
    get_folder_results,
    clear_folder_results
)

# Additional components
from src.ui.about_tab import render_about_tab, render_compact_about
from src.ui.folder_picker import (
    show_folder_picker,
    show_output_folder_picker,
    browse_folder_contents,
    validate_folder_path
)

__all__ = [
    # Core components
    'setup_page_config',
    'setup_enhanced_sidebar',
    'display_supported_formats',

    # Main content
    'render_welcome_section',
    'render_main_features',
    'render_app_header',

    # Output display
    'display_output_section',
    'display_enhanced_output_section',
    'display_content_statistics',
    'display_download_options',
    'set_output_content',
    'get_output_content',
    'has_output_content',

    # File upload
    'render_file_upload_section',
    'render_enhanced_file_upload',
    'create_file_uploader',
    'handle_file_conversion',

    # Folder processing
    'render_folder_processing_section',
    'render_enhanced_folder_processing',
    'display_folder_preview',
    'execute_folder_processing',

    # Results
    'display_folder_results',
    'display_processing_summary',
    'set_folder_results',
    'get_folder_results',
    'clear_folder_results',

    # Additional
    'render_about_tab',
    'render_compact_about',
    'show_folder_picker',
    'show_output_folder_picker',
    'browse_folder_contents',
    'validate_folder_path'
]

