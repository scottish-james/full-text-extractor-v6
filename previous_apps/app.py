"""
Document to Markdown Converter - Main Application
Refactored for better maintainability and separation of concerns.
"""

import streamlit as st

# UI Components
from src.ui.components import setup_page_config
from src.ui.sidebar import setup_enhanced_sidebar
from src.ui.main_content import render_welcome_section
from src.ui.file_upload import render_enhanced_file_upload
from src.ui.folder_processing import render_enhanced_folder_processing
from src.ui.folder_results import display_folder_results
from src.ui.output_display import display_enhanced_output_section
from src.ui.about_tab import render_about_tab


def initialize_session_state():
    """Initialize session state variables."""
    if "markdown_content" not in st.session_state:
        st.session_state.markdown_content = ""
    if "file_name" not in st.session_state:
        st.session_state.file_name = ""
    if "folder_processing_results" not in st.session_state:
        st.session_state.folder_processing_results = None


def main():
    """Main application function."""
    # Set up page configuration
    setup_page_config()

    # Initialize session state
    initialize_session_state()

    # App header and main features
    render_welcome_section()

    # Set up the sidebar and get configuration - UPDATED to handle OpenAI key
    enhance_markdown, api_key_claude, api_key_openai = setup_enhanced_sidebar()

    # Store OpenAI key in session state for use by other components
    if api_key_openai:
        st.session_state.openai_api_key = api_key_openai
    elif 'openai_api_key' in st.session_state:
        # Clear if no key provided
        del st.session_state.openai_api_key

    # Main content area - Tabs
    tab1, tab2, tab3 = st.tabs(["🗂️ Upload File", "🗃️ Process Folder", "📘 About"])

    with tab1:
        render_enhanced_file_upload(enhance_markdown, api_key_claude)

    with tab2:
        render_enhanced_folder_processing(enhance_markdown, api_key_claude)

    with tab3:
        render_about_tab()

    # Display results sections
    display_results_sections()


def display_results_sections():
    """Display output and folder results sections."""
    # Display file conversion output
    display_enhanced_output_section()

    from src.ui.diagram_screenshot import render_diagram_screenshot_section
    render_diagram_screenshot_section()

    # Display folder processing results
    display_folder_results()


if __name__ == "__main__":
    main()