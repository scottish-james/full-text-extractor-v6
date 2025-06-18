"""
File upload UI components for the Document to Markdown Converter.
FIXED: Properly stores PowerPoint file data for screenshot functionality.
"""

import streamlit as st
from src.converters.file_converter import convert_file_to_markdown
from src.ui.components import get_supported_formats
from src.ui.output_display import set_output_content


def render_file_upload_section(enhance_markdown=True, api_key=None):
    """Render the complete file upload section."""
    st.header("📄 File Upload")

    # File uploader
    uploaded_file = create_file_uploader()

    # Upload instructions
    display_upload_instructions()

    # Convert button and processing
    if st.button("Convert File to Markdown", key="convert_file", type="primary"):
        handle_file_conversion(uploaded_file, enhance_markdown, api_key)


def create_file_uploader():
    """Create the file uploader component."""
    # Get supported file extensions
    all_extensions = []
    formats = get_supported_formats()
    for category, info in formats.items():
        all_extensions.extend(info["extensions"])

    uploaded_file = st.file_uploader(
        "Select a file to convert",
        type=all_extensions,
        help="Upload any supported document format for conversion"
    )

    return uploaded_file


def display_upload_instructions():
    """Display helpful instructions for file upload."""
    with st.expander("📋 Upload Instructions", expanded=False):
        st.markdown("""
        **Supported File Types:**
        - **PowerPoint**: .pptx, .ppt (optimised processing)
        - **Documents**: .docx, .doc, .pdf, .epub
        - **Spreadsheets**: .xlsx, .xls
        - **Web**: .html, .htm
        - **Data**: .csv, .json, .xml

        **Tips for Best Results:**
        - Use original files rather than scanned PDFs when possible
        - PowerPoint files get special optimised processing
        - Enable Claude enhancement for improved formatting
        - Larger files may take longer to process
        """)


def handle_file_conversion(uploaded_file, enhance_markdown=True, api_key=None):
    """Handle the file conversion process."""
    if not uploaded_file:
        st.error("Please upload a file to convert")
        return

    # FIXED: Store file data BEFORE processing (when we still have the uploaded_file object)
    _store_powerpoint_file_data(uploaded_file)

    with st.spinner("Converting to Markdown..."):
        # Show progress bar
        progress_bar = st.progress(0)

        # Simulate progress for user feedback
        for percent_complete in range(0, 90, 10):
            progress_bar.progress(percent_complete)

        # Convert uploaded file
        markdown_content, error = convert_file_to_markdown(
            uploaded_file.getbuffer(),
            uploaded_file.name,
            enhance=enhance_markdown,
            api_key=api_key
        )

        # Complete progress
        progress_bar.progress(100)

        if error:
            st.error(f"Error during conversion: {error}")
        else:
            # Store results
            set_output_content(markdown_content, uploaded_file.name)

            # Show success message
            if enhance_markdown and api_key:
                st.success("✨ Conversion completed successfully with Claude Sonnet 4 enhancement!")
            else:
                st.success("✅ Conversion completed successfully!")


def _store_powerpoint_file_data(uploaded_file):
    """
    Store PowerPoint file data in session state for later screenshot use.
    Called during file upload process when we have access to the file.
    """
    if not uploaded_file:
        return

    file_ext = uploaded_file.name.split('.')[-1].lower() if '.' in uploaded_file.name else ''

    if file_ext in ['pptx', 'ppt']:
        # Store the file data for later screenshot use
        st.session_state.uploaded_file_data = uploaded_file.getbuffer()
        st.session_state.uploaded_file_name = uploaded_file.name

        # Debug info
        file_size_mb = len(uploaded_file.getbuffer()) / (1024 * 1024)
        st.success(f"📁 Stored PowerPoint file data for screenshots: {uploaded_file.name} ({file_size_mb:.1f} MB)")
    else:
        # Clear any existing PowerPoint data for non-PowerPoint files
        if 'uploaded_file_data' in st.session_state:
            del st.session_state.uploaded_file_data
        if 'uploaded_file_name' in st.session_state:
            del st.session_state.uploaded_file_name


def render_enhanced_file_upload(enhance_markdown=True, api_key=None):
    """Render an enhanced file upload section with better UX."""
    st.header("Document Upload & Conversion")

    # File uploader
    uploaded_file = create_file_uploader()

    # Show file info if file is uploaded
    if uploaded_file:

        # Conversion options
        with st.expander("⚙️ Conversion Options", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                if enhance_markdown and api_key:
                    st.success("✅ Claude Sonnet 4 Enhancement: Enabled")
                elif enhance_markdown:
                    st.warning("⚠️ Claude Enhancement: API key required")
                else:
                    st.info("ℹ️ Claude Enhancement: Disabled")

            with col2:
                file_ext = uploaded_file.name.split('.')[-1].lower() if '.' in uploaded_file.name else ''
                if file_ext in ['pptx', 'ppt']:
                    st.success("🎯 PowerPoint Optimised Processing")
                else:
                    st.info("📄 Standard Processing")

        # Convert button
        if st.button("🚀 Convert to Markdown", key="convert_enhanced", type="primary", use_container_width=True):
            handle_file_conversion(uploaded_file, enhance_markdown, api_key)

    else:
        # Show instructions when no file is uploaded
        display_upload_instructions()


def validate_file_upload(uploaded_file):
    """Validate the uploaded file."""
    if not uploaded_file:
        return False, "No file uploaded"

    # Check file size (100MB limit)
    max_size_mb = 100
    file_size_mb = len(uploaded_file.getbuffer()) / (1024 * 1024)

    if file_size_mb > max_size_mb:
        return False, f"File too large. Maximum size is {max_size_mb}MB, got {file_size_mb:.1f}MB"

    # Check file type
    supported_extensions = []
    formats = get_supported_formats()
    for category, info in formats.items():
        supported_extensions.extend(info["extensions"])

    file_ext = uploaded_file.name.split('.')[-1].lower() if '.' in uploaded_file.name else ''

    if file_ext not in supported_extensions:
        return False, f"Unsupported file type: .{file_ext}"

    return True, "File is valid"


def show_conversion_tips():
    """Show tips for better conversion results."""
    st.info("""
    💡 **Tips for Better Results:**
    - Use original files instead of scanned copies
    - PowerPoint files get optimised processing with better text extraction
    - Enable Claude enhancement for improved formatting and structure
    - Review the output for accuracy, especially for complex layouts
    """)