"""
Output display UI components for markdown content.
"""

import streamlit as st


def display_output_section():
    """Display the markdown output section."""
    if "markdown_content" in st.session_state and st.session_state.markdown_content:
        st.subheader("📝 Converted Markdown")

        # Add some statistics about the content
        content = st.session_state.markdown_content
        word_count = len(content.split())
        char_count = len(content)
        line_count = len(content.split('\n'))

        # Display stats in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Words", word_count)
        with col2:
            st.metric("📝 Characters", char_count)
        with col3:
            st.metric("📄 Lines", line_count)

        # Text area with the content
        st.text_area(
            "Markdown Content",
            value=content,
            height=400,
            help="Your converted markdown content"
        )

        # Download button
        filename = st.session_state.file_name.rsplit(".", 1)[
                       0] + ".md" if "." in st.session_state.file_name else st.session_state.file_name + ".md"

        st.download_button(
            label="📥 Download Markdown File",
            data=content,
            file_name=filename,
            mime="text/markdown",
            help="Download your converted markdown file"
        )


def display_content_preview(content, max_lines=10):
    """Display a preview of the markdown content."""
    if not content:
        return

    lines = content.split('\n')
    preview_lines = lines[:max_lines]
    preview_content = '\n'.join(preview_lines)

    if len(lines) > max_lines:
        preview_content += f"\n\n... ({len(lines) - max_lines} more lines)"

    st.text_area(
        "Content Preview",
        value=preview_content,
        height=200,
        help="Preview of your converted content"
    )


def display_content_statistics(content):
    """Display detailed statistics about the content."""
    if not content:
        st.info("No content to analyze")
        return

    # Calculate statistics
    word_count = len(content.split())
    char_count = len(content)
    char_count_no_spaces = len(content.replace(' ', '').replace('\n', '').replace('\t', ''))
    line_count = len(content.split('\n'))
    paragraph_count = len([p for p in content.split('\n\n') if p.strip()])

    # Count markdown elements
    header_count = content.count('#')
    link_count = content.count('[')
    bullet_count = content.count('- ')

    st.subheader("📊 Content Analysis")

    # Basic stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📝 Words", word_count)
    with col2:
        st.metric("🔤 Characters", char_count)
    with col3:
        st.metric("📄 Lines", line_count)
    with col4:
        st.metric("📋 Paragraphs", paragraph_count)

    # Markdown elements
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🔗 Links", link_count)
    with col2:
        st.metric("📌 Headers", header_count)
    with col3:
        st.metric("• Bullets", bullet_count)


def display_download_options(content, base_filename):
    """Display various download options for the content."""
    if not content or not base_filename:
        return

    st.subheader("📥 Download Options")

    # Generate filename without extension
    filename_base = base_filename.rsplit(".", 1)[0] if "." in base_filename else base_filename

    col1, col2 = st.columns(2)

    with col1:
        # Markdown download
        st.download_button(
            label="📝 Download as Markdown (.md)",
            data=content,
            file_name=f"{filename_base}.md",
            mime="text/markdown",
            help="Download your converted markdown file"
        )

    with col2:
        # Text download
        st.download_button(
            label="📄 Download as Text (.txt)",
            data=content,
            file_name=f"{filename_base}.txt",
            mime="text/plain",
            help="Download as plain text file"
        )


def clear_output_content():
    """Clear output content from session state."""
    if "markdown_content" in st.session_state:
        del st.session_state.markdown_content
    if "file_name" in st.session_state:
        del st.session_state.file_name


def set_output_content(markdown_content, file_name):
    """Set output content in session state."""
    st.session_state.markdown_content = markdown_content
    st.session_state.file_name = file_name


def get_output_content():
    """Get output content from session state."""
    return (
        st.session_state.get("markdown_content", ""),
        st.session_state.get("file_name", "")
    )


def has_output_content():
    """Check if there's output content to display."""
    return (
            "markdown_content" in st.session_state and
            st.session_state.markdown_content and
            st.session_state.markdown_content.strip()
    )


def display_enhanced_output_section():
    """Display an enhanced output section with more features."""
    if not has_output_content():
        return

    content, filename = get_output_content()

    st.subheader("📝 Conversion Results")

    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📄 Content", "📊 Statistics", "📥 Download"])

    with tab1:
        st.text_area(
            "Converted Markdown",
            value=content,
            height=400,
            help="Your converted markdown content"
        )

    with tab2:
        display_content_statistics(content)

    with tab3:
        display_download_options(content, filename)