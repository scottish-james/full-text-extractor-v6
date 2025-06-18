"""
Folder picker components for the Document to Markdown Converter.
Provides cross-platform folder selection functionality.
"""

import streamlit as st
import os
from pathlib import Path


def show_folder_picker(key_suffix="", default_path=None):
    """
    Show a folder picker interface using Streamlit components.

    Args:
        key_suffix (str): Suffix for Streamlit component keys to ensure uniqueness
        default_path (str): Default path to display (optional)

    Returns:
        str: Selected folder path or None if not selected
    """
    st.markdown("**Select a folder containing your documents:**")

    # Initialize session state for folder path if not exists
    session_key = f"folder_path_{key_suffix}"
    if session_key not in st.session_state:
        if default_path is None:
            st.session_state[session_key] = str(Path.home())
        else:
            st.session_state[session_key] = default_path

    # Quick access buttons for common directories
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("📁 Home", key=f"home_btn_{key_suffix}"):
            st.session_state[session_key] = str(Path.home())
            st.rerun()

    with col2:
        if st.button("🖥️ Desktop", key=f"desktop_btn_{key_suffix}"):
            desktop_path = str(Path.home() / "Desktop")
            if os.path.exists(desktop_path):
                st.session_state[session_key] = desktop_path
                st.rerun()

    with col3:
        if st.button("📂 Documents", key=f"docs_btn_{key_suffix}"):
            docs_path = str(Path.home() / "Documents")
            if os.path.exists(docs_path):
                st.session_state[session_key] = docs_path
                st.rerun()

    with col4:
        if st.button("📥 Downloads", key=f"downloads_btn_{key_suffix}"):
            downloads_path = str(Path.home() / "Downloads")
            if os.path.exists(downloads_path):
                st.session_state[session_key] = downloads_path
                st.rerun()

    # Text input for manual path entry (uses session state as value)
    folder_path = st.text_input(
        "Folder path:",
        value=st.session_state[session_key],
        key=f"folder_input_{key_suffix}",
        help="Enter the full path to the folder containing your documents"
    )

    # Update session state if user manually changes the path (with automatic cleaning)
    if folder_path != st.session_state[session_key]:
        # Clean the path: strip whitespace and normalize
        cleaned_path = folder_path.strip()
        if cleaned_path:
            cleaned_path = os.path.normpath(cleaned_path)
        st.session_state[session_key] = cleaned_path

    # Validate the folder path
    current_path = st.session_state[session_key]
    if current_path and os.path.isdir(current_path):
        st.success(f"✅ Selected folder: `{current_path}`")

        # Show folder contents preview
        try:
            items = os.listdir(current_path)
            file_count = len([item for item in items if os.path.isfile(os.path.join(current_path, item))])
            folder_count = len([item for item in items if os.path.isdir(os.path.join(current_path, item))])

            st.info(f"📊 Folder contains: {file_count} files, {folder_count} subfolders")

            # Show a few example files
            if file_count > 0:
                example_files = [item for item in items if os.path.isfile(os.path.join(current_path, item))][:5]
                st.markdown("**Example files:**")
                for file in example_files:
                    st.markdown(f"• {file}")
                if file_count > 5:
                    st.markdown(f"• ... and {file_count - 5} more files")

        except PermissionError:
            st.warning("⚠️ Permission denied - cannot read folder contents")
        except Exception as e:
            st.warning(f"⚠️ Error reading folder: {str(e)}")

        return current_path

    elif current_path:
        st.error(f"❌ Invalid folder path: `{current_path}`")
        return None

    return None


def show_output_folder_picker(key_suffix="", default_name="markdown"):
    """
    Show an output folder picker interface.

    Args:
        key_suffix (str): Suffix for Streamlit component keys to ensure uniqueness
        default_name (str): Default subfolder name

    Returns:
        str: Selected output folder path or None for default behavior
    """
    st.markdown("**Choose where to save the converted markdown files:**")

    # Option selection
    output_option = st.radio(
        "Output location:",
        ["Create subfolder in input folder", "Choose custom location"],
        key=f"output_option_{key_suffix}",
        help="Select where you want the converted files to be saved"
    )

    if output_option == "Create subfolder in input folder":
        subfolder_name = st.text_input(
            "Subfolder name:",
            value=default_name,
            key=f"subfolder_name_{key_suffix}",
            help="Name of the subfolder to create for markdown files"
        )

        st.info(f"📁 Markdown files will be saved in a '{subfolder_name}' subfolder within your input folder")
        return None  # This signals to use the default behavior

    else:  # Custom location
        # Initialize session state for custom output path
        output_session_key = f"custom_output_path_{key_suffix}"
        if output_session_key not in st.session_state:
            st.session_state[output_session_key] = str(Path.home())

        # Quick access buttons for output location
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📁 Home", key=f"output_home_btn_{key_suffix}"):
                st.session_state[output_session_key] = str(Path.home())
                st.rerun()

        with col2:
            if st.button("🖥️ Desktop", key=f"output_desktop_btn_{key_suffix}"):
                desktop_path = str(Path.home() / "Desktop")
                if os.path.exists(desktop_path):
                    st.session_state[output_session_key] = desktop_path
                    st.rerun()

        with col3:
            if st.button("📂 Documents", key=f"output_docs_btn_{key_suffix}"):
                docs_path = str(Path.home() / "Documents")
                if os.path.exists(docs_path):
                    st.session_state[output_session_key] = docs_path
                    st.rerun()

        # Text input using session state
        custom_path = st.text_input(
            "Custom output folder:",
            value=st.session_state[output_session_key],
            key=f"custom_output_{key_suffix}",
            help="Enter the full path where you want to save the markdown files"
        )

        # Update session state if user manually changes the path (with automatic cleaning)
        if custom_path != st.session_state[output_session_key]:
            # Clean the path: strip whitespace and normalize
            cleaned_path = custom_path.strip()
            if cleaned_path:
                cleaned_path = os.path.normpath(cleaned_path)
            st.session_state[output_session_key] = cleaned_path

        # Validate custom path
        current_custom_path = st.session_state[output_session_key]
        if current_custom_path:
            if os.path.isdir(current_custom_path):
                st.success(f"✅ Output folder: `{current_custom_path}`")
                return current_custom_path
            else:
                # Check if parent directory exists (we can create the folder)
                parent_dir = os.path.dirname(current_custom_path)
                if os.path.isdir(parent_dir):
                    st.info(f"📁 Will create folder: `{current_custom_path}`")
                    return current_custom_path
                else:
                    st.error(f"❌ Invalid path - parent directory doesn't exist: `{parent_dir}`")
                    return None

        return None


def browse_folder_contents(folder_path, supported_extensions=None):
    """
    Browse and display folder contents with filtering by supported extensions.

    Args:
        folder_path (str): Path to the folder to browse
        supported_extensions (list): List of supported file extensions (optional)

    Returns:
        dict: Information about the folder contents
    """
    if not folder_path or not os.path.isdir(folder_path):
        return {"error": "Invalid folder path"}

    try:
        all_items = os.listdir(folder_path)

        files = []
        folders = []
        supported_files = []

        for item in all_items:
            item_path = os.path.join(folder_path, item)

            if os.path.isfile(item_path):
                files.append(item)

                # Check if file is supported
                if supported_extensions:
                    file_ext = item.rsplit('.', 1)[-1].lower() if '.' in item else ''
                    if file_ext in supported_extensions:
                        supported_files.append(item)

            elif os.path.isdir(item_path):
                folders.append(item)

        return {
            "total_items": len(all_items),
            "files": files,
            "folders": folders,
            "supported_files": supported_files,
            "file_count": len(files),
            "folder_count": len(folders),
            "supported_count": len(supported_files)
        }

    except PermissionError:
        return {"error": "Permission denied"}
    except Exception as e:
        return {"error": str(e)}


def validate_folder_path(folder_path):
    """
    Validate a folder path and return status information.

    Args:
        folder_path (str): Path to validate

    Returns:
        dict: Validation results
    """
    if not folder_path:
        return {"valid": False, "error": "No path provided"}

    if not os.path.exists(folder_path):
        return {"valid": False, "error": "Path does not exist"}

    if not os.path.isdir(folder_path):
        return {"valid": False, "error": "Path is not a directory"}

    try:
        # Test if we can read the directory
        os.listdir(folder_path)
        return {"valid": True, "readable": True}
    except PermissionError:
        return {"valid": True, "readable": False, "error": "Permission denied"}
    except Exception as e:
        return {"valid": False, "error": str(e)}


def create_folder_if_not_exists(folder_path):
    """
    Create a folder if it doesn't exist.

    Args:
        folder_path (str): Path to the folder to create

    Returns:
        dict: Creation result
    """
    try:
        if os.path.exists(folder_path):
            if os.path.isdir(folder_path):
                return {"success": True, "existed": True, "message": "Folder already exists"}
            else:
                return {"success": False, "error": "Path exists but is not a directory"}

        os.makedirs(folder_path, exist_ok=True)
        return {"success": True, "existed": False, "message": "Folder created successfully"}

    except PermissionError:
        return {"success": False, "error": "Permission denied"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_folder_size_info(folder_path):
    """
    Get basic size information about a folder.

    Args:
        folder_path (str): Path to analyze

    Returns:
        dict: Size information
    """
    if not os.path.isdir(folder_path):
        return {"error": "Invalid directory"}

    try:
        total_size = 0
        file_count = 0

        for dirpath, dirnames, filenames in os.walk(folder_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                    file_count += 1
                except OSError:
                    # Skip files we can't access
                    pass

        # Convert to human readable format
        def format_size(size_bytes):
            if size_bytes < 1024:
                return f"{size_bytes} B"
            elif size_bytes < 1024 ** 2:
                return f"{size_bytes / 1024:.1f} KB"
            elif size_bytes < 1024 ** 3:
                return f"{size_bytes / (1024 ** 2):.1f} MB"
            else:
                return f"{size_bytes / (1024 ** 3):.1f} GB"

        return {
            "total_size_bytes": total_size,
            "total_size_formatted": format_size(total_size),
            "file_count": file_count
        }

    except Exception as e:
        return {"error": str(e)}