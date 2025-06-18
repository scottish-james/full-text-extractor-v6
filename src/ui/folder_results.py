"""
Folder processing results UI components.
"""
import streamlit as st
import os
import subprocess


def display_folder_results():
    """Display folder processing results."""
    if "folder_processing_results" in st.session_state and st.session_state.folder_processing_results:
        results = st.session_state.folder_processing_results

        st.subheader("📊 Processing Results")

        # Create columns for better layout
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("✅ Successfully Converted", results['success_count'])
        with col2:
            st.metric("❌ Failed Conversions", results['error_count'])
        with col3:
            st.metric("🤖 AI Provider", "Claude Sonnet 4" if results.get('enhanced') else "Standard")

        st.markdown(f"**📁 Output Location:** {results['output_folder']}")

        # Show errors if any
        if results['error_count'] > 0:
            with st.expander("🔍 View Conversion Errors"):
                for file_name, error in results['errors'].items():
                    st.markdown(f"**{file_name}**: {error}")

        # Option to open output folder
        if st.button("📂 Open Output Folder"):
            _open_output_folder(results['output_folder'])


def display_processing_summary(success_count, error_count, enhanced=False):
    """Display a summary of processing results."""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("✅ Converted", success_count)
    with col2:
        st.metric("❌ Failed", error_count)
    with col3:
        provider = "Claude Sonnet 4" if enhanced else "Standard"
        st.metric("🤖 Provider", provider)


def display_error_details(errors):
    """Display detailed error information."""
    if not errors:
        return

    with st.expander("🔍 View Conversion Errors", expanded=False):
        for file_name, error in errors.items():
            st.error(f"**{file_name}**: {error}")


def display_success_message(success_count, enhanced=False):
    """Display appropriate success message based on results."""
    if success_count > 0:
        if enhanced:
            st.success(f"✨ Successfully converted {success_count} files with Claude Sonnet 4 enhancement!")
        else:
            st.success(f"✅ Successfully converted {success_count} files!")


def display_processing_metrics(results):
    """Display comprehensive processing metrics."""
    st.subheader("📊 Processing Summary")

    # Main metrics
    total_files = results['success_count'] + results['error_count']
    success_count = results['success_count']
    error_count = results['error_count']
    success_rate = (success_count / total_files * 100) if total_files > 0 else 0

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📄 Total Files", total_files)
    with col2:
        st.metric("✅ Successful", success_count)
    with col3:
        st.metric("❌ Failed", error_count)
    with col4:
        st.metric("📈 Success Rate", f"{success_rate:.1f}%")


def _open_output_folder(output_folder):
    """Helper function to open output folder in file explorer."""
    try:
        if os.path.exists(output_folder):
            if os.name == 'nt':  # Windows
                os.startfile(output_folder)
            elif os.name == 'posix':  # macOS, Linux
                if os.path.exists('/usr/bin/open'):  # macOS
                    subprocess.call(['open', output_folder])
                else:  # Linux
                    subprocess.call(['xdg-open', output_folder])
            st.success(f"📂 Opened folder: {output_folder}")
        else:
            st.error("Output folder not found")
    except Exception as e:
        st.error(f"Failed to open folder: {str(e)}")


def clear_folder_results():
    """Clear folder processing results from session state."""
    if "folder_processing_results" in st.session_state:
        del st.session_state.folder_processing_results


def get_folder_results():
    """Get folder processing results from session state."""
    return st.session_state.get("folder_processing_results")


def set_folder_results(success_count, error_count, errors, output_folder, enhanced=False):
    """Set folder processing results in session state."""
    st.session_state.folder_processing_results = {
        "success_count": success_count,
        "error_count": error_count,
        "errors": errors,
        "output_folder": output_folder,
        "enhanced": enhanced
    }
