"""
UI Components Module

This module provides UI components and utilities for the Streamlit interface.
"""

import streamlit as st
from src.utils.file_utils import get_supported_formats
from config import APP_TITLE, APP_ICON, APP_LAYOUT, UI_THEME_COLOR, UI_BACKGROUND_COLOR


def setup_page_config():
    """
    Set up Streamlit page configuration.
    """
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout=APP_LAYOUT,
    )

    # Apply custom styling
    st.markdown(f"""
    <style>
        .main {{
            background-color: {UI_BACKGROUND_COLOR};
        }}
        .stButton button {{
            background-color: {UI_THEME_COLOR};
            color: white;
            font-weight: bold;
        }}
    </style>
    """, unsafe_allow_html=True)


def display_supported_formats():
    """
    Display the supported file formats in the sidebar.
    """
    formats = get_supported_formats()
    for category, info in formats.items():
        st.markdown(f"**{category}**")
        for format_name in info["formats"]:
            st.markdown(f"- {format_name}")


def create_progress_bar(total_steps=100):
    """
    Create a progress bar and status text display.

    Args:
        total_steps (int): The total number of steps

    Returns:
        tuple: (progress_bar, status_text, update_progress_func)
    """
    progress_bar = st.progress(0)
    status_text = st.empty()

    def update_progress(step, message="Processing..."):
        """Update the progress bar and status text."""
        progress = min(1.0, step / total_steps)
        progress_bar.progress(progress)
        status_text.text(message)

    return progress_bar, status_text, update_progress


def display_success_message(message):
    """
    Display a success message with consistent styling.

    Args:
        message (str): The success message to display
    """
    st.success(message)


def display_error_message(message):
    """
    Display an error message with consistent styling.

    Args:
        message (str): The error message to display
    """
    st.error(message)


def display_warning_message(message):
    """
    Display a warning message with consistent styling.

    Args:
        message (str): The warning message to display
    """
    st.warning(message)


def display_info_message(message):
    """
    Display an info message with consistent styling.

    Args:
        message (str): The info message to display
    """
    st.info(message)


def create_file_uploader(file_types, help_text="Choose a file to convert"):
    """
    Create a standardized file uploader.

    Args:
        file_types (list): List of file extensions to accept
        help_text (str): Help text for the uploader

    Returns:
        uploaded_file: The uploaded file object
    """
    return st.file_uploader(
        "Select a file to convert - if possible do not use PDF",
        type=file_types,
        help=help_text
    )


def create_download_button(content, filename, label="Download File", mime_type="text/plain"):
    """
    Create a standardized download button.

    Args:
        content (str): The content to download
        filename (str): The filename for the downloaded file
        label (str): The button label
        mime_type (str): The MIME type of the content
    """
    st.download_button(
        label=label,
        data=content,
        file_name=filename,
        mime=mime_type
    )