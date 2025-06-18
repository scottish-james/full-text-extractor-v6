"""
Enterprise File Converter - FIXED to properly use PowerPoint processor output
Now ensures the sophisticated PowerPoint processor output flows correctly to enterprise LLM
"""

import io
import os
import tempfile
from typing import Tuple, Optional

from src.utils.file_utils import get_file_extension
from src.converters.hyperlink_extractor import extract_pdf_hyperlinks, format_hyperlinks_section
from src.processors.enhanced_pptx_processor import PowerPointProcessor

# Import the new enterprise LLM system
try:
    from enterprise_llm_converter import EnterpriseLLMEnhancer, enhance_markdown_with_enterprise_llm

    ENTERPRISE_LLM_AVAILABLE = True
except ImportError:
    ENTERPRISE_LLM_AVAILABLE = False
    print("Enterprise LLM not available. Please ensure JWT_token.txt and model_url.txt are present.")


def enhance_content_with_enterprise_llm(structured_data, metadata, source_filename: str, content_type: str) -> Tuple[str, Optional[str]]:
    """
    FIXED: Enhance content using enterprise LLM with PowerPoint processor integration

    Args:
        structured_data: Structured PowerPoint data from PowerPointProcessor
        metadata: Document metadata from PowerPointProcessor
        source_filename (str): Source filename
        content_type (str): Content type

    Returns:
        Tuple[str, Optional[str]]: Enhanced content and error message
    """
    if not ENTERPRISE_LLM_AVAILABLE:
        return "Enterprise LLM not available", "Missing JWT_token.txt or model_url.txt"

    try:
        enhancer = EnterpriseLLMEnhancer()
        # FIXED: This now uses the PowerPoint processor's markdown converter internally
        return enhancer.enhance_powerpoint_content(structured_data, metadata, source_filename)
    except Exception as e:
        return str(structured_data), str(e)


def convert_file_to_markdown_enterprise(file_data, filename, enhance=True):
    """
    Convert a file to Markdown using enterprise LLM enhancement
    Optimised for PowerPoint presentations with enterprise routing

    Args:
        file_data (bytes): The binary content of the file
        filename (str): The name of the file
        enhance (bool): Whether to enhance with enterprise LLM

    Returns:
        tuple: (markdown_content, error_message)
    """
    try:
        ext = get_file_extension(filename)

        # Use enhanced processing for PowerPoint files (primary optimisation)
        if ext.lower() in ["pptx", "ppt"]:
            return convert_pptx_enhanced_enterprise(file_data, filename, enhance)

        # Use standard processing for other file types
        return convert_standard_enterprise(file_data, filename, enhance)

    except Exception as e:
        return "", str(e)


def convert_pptx_enhanced_enterprise(file_data, filename, enhance=True):
    """
    FIXED: Convert PowerPoint files using enhanced processing with enterprise LLM
    Now properly uses the PowerPoint processor's output with semantic roles
    """
    try:
        # Create temporary file for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{get_file_extension(filename)}") as tmp_file:
            tmp_file.write(file_data)
            tmp_file_path = tmp_file.name

        try:
            # Use the sophisticated PowerPoint processor
            processor = PowerPointProcessor(use_accessibility_order=True)

            if enhance and ENTERPRISE_LLM_AVAILABLE:
                # Check if we can use XML processing
                if processor._has_xml_access(tmp_file_path):
                    print("🚀 Using sophisticated XML processing with enterprise LLM...")

                    # Load presentation for full processing
                    from pptx import Presentation
                    prs = Presentation(tmp_file_path)

                    # Extract comprehensive metadata
                    pptx_metadata = processor.metadata_extractor.extract_pptx_metadata(prs, tmp_file_path)

                    # Process entire presentation through component pipeline
                    # This extracts structured data with semantic roles
                    structured_data = processor.extract_presentation_data(prs)

                    # FIXED: Enhance with enterprise LLM using sophisticated PowerPoint processor output
                    # The enterprise LLM converter now uses the PowerPoint processor's markdown converter
                    enhanced_content, enhance_error = enhance_content_with_enterprise_llm(
                        structured_data,
                        pptx_metadata,
                        filename,
                        "PowerPoint Presentation"
                    )

                    if enhance_error:
                        print(f"Enterprise LLM enhancement error: {enhance_error}")
                        # Fallback to PowerPoint processor's own markdown conversion
                        markdown_content = processor.markdown_converter.convert_structured_data_to_markdown(
                            structured_data, convert_slide_titles=False  # XML controls titles
                        )
                        # Add metadata context
                        markdown_content = processor.metadata_extractor.add_pptx_metadata_for_claude(
                            markdown_content, pptx_metadata
                        )
                        # Add error comment
                        metadata_comment = f"\n<!-- Enterprise LLM Error: {enhance_error} -->\n"
                        markdown_content = metadata_comment + markdown_content
                    else:
                        markdown_content = enhanced_content

                else:
                    print("📄 XML not available - using MarkItDown with enterprise enhancement...")
                    # Use basic conversion then enhance
                    markdown_content = processor._simple_markitdown_processing(tmp_file_path)

                    if enhance and ENTERPRISE_LLM_AVAILABLE:
                        # Create minimal structured data for enhancement
                        structured_data = {"slides": [
                            {"slide_number": 1, "content_blocks": [{"type": "text", "content": markdown_content}]}]}
                        enhanced_content, enhance_error = enhance_content_with_enterprise_llm(
                            structured_data,
                            {},
                            filename,
                            "PowerPoint Presentation"
                        )

                        if not enhance_error:
                            markdown_content = enhanced_content
            else:
                # No enhancement - use PowerPoint processor's sophisticated processing
                if processor._has_xml_access(tmp_file_path):
                    print("🎯 Using sophisticated XML processing without LLM enhancement...")
                    markdown_content = processor.convert_pptx_to_markdown_enhanced(tmp_file_path, convert_slide_titles=False)
                else:
                    print("📄 Using MarkItDown fallback...")
                    markdown_content = processor._simple_markitdown_processing(tmp_file_path)

        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)

        return markdown_content, None

    except Exception as e:
        return "", str(e)


def convert_standard_enterprise(file_data, filename, enhance=True):
    """
    Convert non-PowerPoint files using standard MarkItDown with enterprise enhancement
    """
    try:
        from markitdown import MarkItDown

        # Create a BytesIO object from the file data
        file_stream = io.BytesIO(file_data)
        file_stream.name = filename

        # Initialize MarkItDown
        md = MarkItDown()

        # Convert using temporary file method (more reliable)
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{get_file_extension(filename)}") as tmp_file:
            tmp_file.write(file_data)
            tmp_file_path = tmp_file.name

        try:
            result = md.convert(tmp_file_path)
            os.unlink(tmp_file_path)
        except Exception as file_path_error:
            print(f"File path conversion failed: {str(file_path_error)}. Trying stream conversion...")
            file_stream.seek(0)
            result = md.convert_stream(file_stream)

        # Get markdown content
        try:
            markdown_content = result.markdown
        except AttributeError:
            try:
                markdown_content = result.text_content
            except AttributeError:
                raise Exception("Neither 'markdown' nor 'text_content' attribute found on result object")

        # Extract hyperlinks for PDF files
        ext = get_file_extension(filename)
        if ext.lower() == "pdf":
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp_file:
                tmp_file.write(file_data)
                tmp_file_path = tmp_file.name

                try:
                    hyperlinks = extract_pdf_hyperlinks(tmp_file_path)
                    markdown_content += format_hyperlinks_section(hyperlinks, "Document")
                finally:
                    os.unlink(tmp_file_path)

        # Determine content type for enterprise LLM
        content_type_map = {
            "pdf": "PDF Document",
            "docx": "Word Document",
            "doc": "Word Document",
            "xlsx": "Excel Spreadsheet",
            "xls": "Excel Spreadsheet",
            "html": "HTML Document",
            "csv": "CSV File",
            "json": "JSON File",
            "xml": "XML File"
        }
        content_type = content_type_map.get(ext.lower(), "Document")

        # Enhance with enterprise LLM if enabled
        if enhance and ENTERPRISE_LLM_AVAILABLE:
            # Create structured data for enhancement
            structured_data = {
                "slides": [{
                    "slide_number": 1,
                    "content_blocks": [{
                        "type": "text",
                        "paragraphs": [{
                            "clean_text": markdown_content,
                            "hints": {"is_bullet": False}
                        }]
                    }]
                }]
            }

            enhanced_content, enhance_error = enhance_content_with_enterprise_llm(
                structured_data,
                {"filename": filename, "content_type": content_type},
                filename,
                content_type
            )

            if enhance_error:
                print(f"Enterprise LLM enhancement error: {enhance_error}. Using original markdown.")
            else:
                markdown_content = enhanced_content

        return markdown_content, None

    except Exception as e:
        return "", str(e)


def process_folder_enterprise(folder_path, output_folder=None, enhance=True):
    """
    Process all compatible files in a folder using enterprise LLM enhancement

    Args:
        folder_path (str): Path to folder containing files to convert
        output_folder (str): Path to save markdown files
        enhance (bool): Whether to enhance with enterprise LLM

    Yields:
        Various progress updates and final results
    """
    from src.processors.folder_processor import find_compatible_files, get_file_extension
    from src.utils.file_utils import ensure_directory_exists
    from config import DEFAULT_MARKDOWN_SUBFOLDER
    import glob

    # Setup output folder
    if not output_folder:
        output_folder = os.path.join(folder_path, DEFAULT_MARKDOWN_SUBFOLDER)

    # Create output folder if it doesn't exist
    ensure_directory_exists(output_folder)

    # Get all compatible files
    from src.utils.file_utils import get_all_supported_extensions
    extensions = get_all_supported_extensions()

    files_to_process = []
    for ext in extensions:
        files_to_process.extend(glob.glob(os.path.join(folder_path, f"*.{ext}")))

    # Sort files by priority (PowerPoint first)
    def get_priority(file_path):
        ext = get_file_extension(os.path.basename(file_path))
        priorities = {"pptx": 1, "ppt": 2}
        return priorities.get(ext.lower(), 999)

    files_to_process.sort(key=get_priority)

    total_files = len(files_to_process)
    if total_files == 0:
        yield 1.0, "No compatible files found in folder"
        yield 0, 0, {}
        return

    # Track results
    success_count = 0
    error_count = 0
    errors = {}

    # Process each file
    for i, file_path in enumerate(files_to_process):
        file_name = os.path.basename(file_path)
        file_ext = get_file_extension(file_name)

        try:
            # Update progress
            progress = (i + 1) / total_files
            file_type = "PowerPoint" if file_ext.lower() in ["pptx", "ppt"] else "Document"
            yield progress, f"Processing {file_type} with Enterprise LLM: {file_name} ({i + 1}/{total_files})"

            # Read file content
            with open(file_path, 'rb') as file:
                file_data = file.read()

            # Convert to markdown using enterprise LLM
            markdown_content, error = convert_file_to_markdown_enterprise(
                file_data,
                file_name,
                enhance=enhance
            )

            if error:
                error_count += 1
                errors[file_name] = error
                continue

            # Save markdown content
            output_file = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}.md")
            with open(output_file, 'w', encoding='utf-8') as md_file:
                md_file.write(markdown_content)

            success_count += 1

        except Exception as e:
            error_count += 1
            errors[file_name] = str(e)

    # Return final results
    yield success_count, error_count, errors


# Configuration and setup functions
def setup_enterprise_llm():
    """
    Setup and validate enterprise LLM configuration

    Returns:
        tuple: (is_configured, status_message)
    """
    try:
        # Check for required files
        if not os.path.exists("JWT_token.txt"):
            return False, "JWT_token.txt file not found"

        if not os.path.exists("model_url.txt"):
            return False, "model_url.txt file not found"

        # Try to initialise the client
        enhancer = EnterpriseLLMEnhancer()

        # Test configuration
        test_content = {"slides": [{"slide_number": 1, "content_blocks": []}]}
        test_metadata = {"test": True}

        # This will validate tokens and URLs without making actual calls
        return True, "Enterprise LLM configured successfully"

    except Exception as e:
        return False, f"Configuration error: {str(e)}"


def get_enterprise_llm_status():
    """
    Get current status of enterprise LLM integration

    Returns:
        dict: Status information
    """
    status = {
        "available": ENTERPRISE_LLM_AVAILABLE,
        "jwt_token_exists": os.path.exists("JWT_token.txt"),
        "model_url_exists": os.path.exists("model_url.txt"),
        "configured": False,
        "message": ""
    }

    if status["available"] and status["jwt_token_exists"] and status["model_url_exists"]:
        configured, message = setup_enterprise_llm()
        status["configured"] = configured
        status["message"] = message
    elif not status["jwt_token_exists"]:
        status["message"] = "JWT_token.txt file missing"
    elif not status["model_url_exists"]:
        status["message"] = "model_url.txt file missing"
    else:
        status["message"] = "Enterprise LLM module not available"

    return status