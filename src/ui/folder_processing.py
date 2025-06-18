"""
Folder processing UI components for batch document conversion.
"""

import streamlit as st
import os
from src.processors.folder_processor import process_folder
from src.ui.folder_picker import show_folder_picker, show_output_folder_picker
from src.ui.folder_results import set_folder_results


def render_folder_processing_section(enhance_markdown=True, api_key=None):
    """Render the complete folder processing section."""
    st.header("📁 Batch Folder Processing")

    # Introduction
    display_folder_processing_intro()

    # Input folder selection
    input_folder = render_input_folder_selection()

    # Output folder selection
    output_folder = render_output_folder_selection()

    # Process button and handling
    if input_folder:
        handle_folder_processing_button(input_folder, output_folder, enhance_markdown, api_key)


def display_folder_processing_intro():
    """Display introduction and benefits of folder processing."""
    st.info(
        "🚀 **Batch Processing:** Convert all supported files in a folder to markdown format. "
        "Perfect for processing multiple documents at once."
    )

    with st.expander("📋 Batch Processing Benefits", expanded=False):
        st.markdown("""
        **Why Use Batch Processing:**
        - Convert multiple documents simultaneously
        - Maintains folder structure and organisation
        - PowerPoint files are prioritised for optimised processing
        - Consistent formatting across all converted files
        - Progress tracking for large batches
        - Detailed error reporting for failed conversions

        **Supported File Types:** All the same formats as single file upload
        """)


def render_input_folder_selection():
    """Render input folder selection interface."""
    st.subheader("📂 Select Input Folder")
    input_folder = show_folder_picker("input")

    if input_folder:
        display_folder_preview(input_folder)

    return input_folder


def render_output_folder_selection():
    """Render output folder selection interface."""
    st.subheader("📁 Choose Output Location")
    output_folder = show_output_folder_picker("output")
    return output_folder


def display_folder_preview(folder_path):
    """Display a preview of the selected folder contents."""
    try:
        from src.processors.folder_processor import find_compatible_files

        compatible_files = find_compatible_files(folder_path)
        total_files = sum(len(files) for files in compatible_files.values())

        if total_files > 0:
            st.success(f"✅ Found {total_files} compatible files for conversion")

            # Show breakdown by category
            with st.expander("📊 File Breakdown", expanded=False):
                for category, files in compatible_files.items():
                    if files:
                        st.markdown(f"**{category}:** {len(files)} files")
                        for file_info in files[:3]:  # Show first 3 files
                            priority_text = " (Optimised)" if file_info.get("optimized") else ""
                            st.markdown(f"  • {file_info['name']}{priority_text}")
                        if len(files) > 3:
                            st.markdown(f"  • ... and {len(files) - 3} more files")
        else:
            st.warning("⚠️ No compatible files found in the selected folder")

    except Exception as e:
        st.error(f"Error analyzing folder: {str(e)}")


def handle_folder_processing_button(input_folder, output_folder, enhance_markdown, api_key):
    """Handle the folder processing button and execution."""
    processing_enabled = input_folder and os.path.isdir(input_folder)

    button_text = "🚀 Process Folder" if processing_enabled else "❌ Select Valid Folder First"
    button_disabled = not processing_enabled

    if st.button(button_text, key="process_folder", type="primary", disabled=button_disabled):
        execute_folder_processing(input_folder, output_folder, enhance_markdown, api_key)


def execute_folder_processing(input_folder, output_folder, enhance_markdown, api_key):
    """Execute the folder processing workflow."""
    # Create progress containers
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Create folder processor
        folder_processor = process_folder(
            input_folder,
            output_folder,
            enhance=enhance_markdown,
            api_key=api_key
        )

        # Process files with progress updates
        for progress, status in folder_processor:
            progress_bar.progress(min(1.0, progress))
            status_text.text(status)

        # Get final results
        success_count, error_count, errors = next(folder_processor)

        # Save results to session state
        final_output_folder = output_folder if output_folder else os.path.join(input_folder, "markdown")
        set_folder_results(
            success_count=success_count,
            error_count=error_count,
            errors=errors,
            output_folder=final_output_folder,
            enhanced=enhance_markdown and api_key
        )

        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()

        # Show completion message
        display_completion_message(success_count, error_count, enhance_markdown and api_key)

    except Exception as e:
        st.error(f"Error processing folder: {str(e)}")
        progress_bar.empty()
        status_text.empty()


def display_completion_message(success_count, error_count, enhanced):
    """Display completion message based on results."""
    if success_count > 0:
        if enhanced:
            st.success(f"✨ Successfully converted {success_count} files with Claude Sonnet 4 enhancement!")
        else:
            st.success(f"✅ Successfully converted {success_count} files!")

    if error_count > 0:
        st.warning(f"⚠️ Failed to convert {error_count} files. See details below.")

    if success_count == 0 and error_count == 0:
        st.info("ℹ️ No files were processed. Check that the folder contains supported file types.")


def render_processing_options(enhance_markdown, api_key):
    """Render processing options and settings."""
    with st.expander("⚙️ Processing Options", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**AI Enhancement:**")
            if enhance_markdown and api_key:
                st.success("✅ Claude Sonnet 4 enabled")
                st.markdown("All files will be enhanced with AI formatting")
            elif enhance_markdown:
                st.warning("⚠️ API key required")
                st.markdown("Enable Claude enhancement in the sidebar")
            else:
                st.info("ℹ️ Standard processing")
                st.markdown("Files will be converted without AI enhancement")

        with col2:
            st.markdown("**Processing Priority:**")
            st.markdown("1. PowerPoint files (optimised)")
            st.markdown("2. Word documents")
            st.markdown("3. PDF files")
            st.markdown("4. Other supported formats")


def estimate_processing_time(folder_path):
    """Estimate processing time for the folder."""
    try:
        from src.processors.folder_processor import get_folder_statistics

        stats = get_folder_statistics(folder_path)
        estimated_seconds = stats.get("estimated_processing_time", 0)

        if estimated_seconds > 0:
            minutes = estimated_seconds // 60
            seconds = estimated_seconds % 60

            if minutes > 0:
                time_str = f"~{minutes}m {seconds}s"
            else:
                time_str = f"~{seconds}s"

            st.info(f"⏱️ Estimated processing time: {time_str}")

    except Exception:
        # Silently fail if estimation doesn't work
        pass


def render_enhanced_folder_processing(enhance_markdown=True, api_key=None):
    """Render an enhanced folder processing interface."""
    st.header("📁 Batch Document Processing")

    # Show processing options first
    render_processing_options(enhance_markdown, api_key)

    # Two-column layout for folder selection
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📂 Input Folder")
        input_folder = show_folder_picker("enhanced_input")

        if input_folder:
            estimate_processing_time(input_folder)

    with col2:
        st.subheader("📁 Output Location")
        output_folder = show_output_folder_picker("enhanced_output")

    # Show folder preview
    if input_folder:
        st.subheader("📊 Folder Analysis")
        display_folder_preview(input_folder)

        # Processing button
        st.markdown("---")
        handle_folder_processing_button(input_folder, output_folder, enhance_markdown, api_key)
    else:
        st.info("👆 Select an input folder to begin batch processing")


def display_batch_tips():
    """Display tips for batch processing."""
    st.info("""
    💡 **Batch Processing Tips:**
    - Larger folders will take longer to process
    - PowerPoint files are processed with optimised algorithms
    - Enable Claude enhancement for consistent formatting across all files
    - The tool will skip files it cannot process and report errors
    - Processing continues even if some files fail
    """)