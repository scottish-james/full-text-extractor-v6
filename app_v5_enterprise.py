"""
DocFlow Vision - Enterprise Edition
Document to Markdown Converter with Smart Hyperlink Integration
Created by James Taylor
Version 6.0 - Clean Rewrite with Enhanced Error Handling
"""

import streamlit as st
import os
import re
import pandas as pd
import tempfile
import base64
from pathlib import Path
from io import BytesIO
from typing import List, Dict, Optional, Tuple
import logging
import requests
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import your existing converters
try:
    from src.converters.file_converter import convert_file_to_markdown

    CONVERTER_AVAILABLE = True
except ImportError:
    CONVERTER_AVAILABLE = False
    logger.warning("File converter not available")

# Import DocVision components
try:
    from PIL import Image
    from pdf2image import convert_from_path

    DOCVISION_AVAILABLE = True
except ImportError:
    DOCVISION_AVAILABLE = False
    logger.warning("DocVision dependencies not available")


class EnterpriseLLMClient:
    """Client for connecting to enterprise LLM endpoints with comprehensive error handling."""

    def __init__(self, jwt_token: str, model_url: str):
        """Initialize the Enterprise LLM client."""
        self.jwt_token = jwt_token
        self.model_url = model_url
        self.headers = {
            "Authorization": f"Bearer {self.jwt_token}",
            "Content-Type": "application/json"
        }

    def validate_configuration(self) -> Tuple[bool, List[str]]:
        """
        Validate the configuration before attempting connection.
        Returns: (is_valid, list_of_errors)
        """
        errors = []

        if not self.jwt_token:
            errors.append("JWT token is missing")
        elif len(self.jwt_token) < 10:
            errors.append("JWT token appears to be too short")

        if not self.model_url:
            errors.append("Model URL is missing")
        elif not self.model_url.startswith(('http://', 'https://')):
            errors.append(f"Model URL must start with http:// or https:// (current: {self.model_url})")

        return len(errors) == 0, errors

    def test_connection_detailed(self) -> Dict:
        """
        Comprehensive connection test with multiple fallback methods.
        Returns detailed diagnostic information.
        """
        result = {
            "success": False,
            "method": None,
            "status_code": None,
            "error": None,
            "error_type": None,
            "suggestions": [],
            "timestamp": datetime.now().isoformat(),
            "url": self.model_url,
            "headers_sent": {k: v[:20] + "..." if len(v) > 20 else v for k, v in self.headers.items()}
        }

        # First validate configuration
        is_valid, config_errors = self.validate_configuration()
        if not is_valid:
            result["error"] = "Configuration validation failed"
            result["error_type"] = "CONFIG_ERROR"
            result["suggestions"] = config_errors
            return result

        # Test methods in order of preference
        test_methods = [
            ("POST", self._test_post_request),
            ("GET", self._test_get_request),
            ("HEAD", self._test_head_request),
            ("OPTIONS", self._test_options_request)
        ]

        for method_name, test_func in test_methods:
            try:
                success, status_code, error_msg = test_func()
                if success:
                    result["success"] = True
                    result["method"] = method_name
                    result["status_code"] = status_code
                    return result
                else:
                    # Store the best error we've seen
                    if status_code:
                        result["status_code"] = status_code
                        result["method"] = method_name
                        result["error"] = error_msg
                        result["error_type"] = self._classify_error(status_code)
                        result["suggestions"] = self._get_suggestions(status_code, error_msg)

                        # If we got a definitive error (4xx), stop trying
                        if 400 <= status_code < 500:
                            return result
            except Exception as e:
                logger.debug(f"Method {method_name} failed: {str(e)}")
                continue

        # If no method worked and we don't have a specific error
        if not result["error"]:
            result["error"] = "Could not connect to the Enterprise LLM endpoint"
            result["error_type"] = "CONNECTION_ERROR"
            result["suggestions"] = [
                "Check if the URL is accessible from your network",
                "Verify firewall or proxy settings",
                "Ensure the server is running and accepting connections",
                "Try accessing the URL directly in a browser to test connectivity"
            ]

        return result

    def _test_post_request(self) -> Tuple[bool, Optional[int], Optional[str]]:
        """Test with a minimal POST request."""
        try:
            test_payload = {
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 1,
                "temperature": 0
            }
            response = requests.post(
                self.model_url,
                headers=self.headers,
                json=test_payload,
                timeout=15
            )
            return response.status_code < 400, response.status_code, response.text[:500] if response.text else None
        except requests.exceptions.Timeout:
            return False, None, "Request timed out"
        except requests.exceptions.ConnectionError as e:
            return False, None, f"Connection error: {str(e)}"
        except Exception as e:
            return False, None, str(e)

    def _test_get_request(self) -> Tuple[bool, Optional[int], Optional[str]]:
        """Test with a GET request."""
        try:
            response = requests.get(
                self.model_url,
                headers=self.headers,
                timeout=10
            )
            return response.status_code < 400, response.status_code, response.text[:500] if response.text else None
        except Exception as e:
            return False, None, str(e)

    def _test_head_request(self) -> Tuple[bool, Optional[int], Optional[str]]:
        """Test with a HEAD request."""
        try:
            response = requests.head(
                self.model_url,
                headers=self.headers,
                timeout=10
            )
            return response.status_code < 400, response.status_code, None
        except Exception as e:
            return False, None, str(e)

    def _test_options_request(self) -> Tuple[bool, Optional[int], Optional[str]]:
        """Test with an OPTIONS request."""
        try:
            response = requests.options(
                self.model_url,
                headers=self.headers,
                timeout=10
            )
            return response.status_code < 400, response.status_code, None
        except Exception as e:
            return False, None, str(e)

    def _classify_error(self, status_code: int) -> str:
        """Classify the error based on status code."""
        if status_code == 401:
            return "AUTHENTICATION_ERROR"
        elif status_code == 403:
            return "PERMISSION_ERROR"
        elif status_code == 404:
            return "ENDPOINT_NOT_FOUND"
        elif status_code == 405:
            return "METHOD_NOT_ALLOWED"
        elif status_code >= 500:
            return "SERVER_ERROR"
        elif status_code >= 400:
            return "CLIENT_ERROR"
        else:
            return "UNKNOWN_ERROR"

    def _get_suggestions(self, status_code: int, error_msg: str) -> List[str]:
        """Get specific suggestions based on the error."""
        suggestions = []

        if status_code == 401:
            suggestions.extend([
                "Check if your JWT token is valid and not expired",
                "Verify the token format (some APIs need 'Bearer ' prefix, others don't)",
                "Try regenerating your authentication token",
                "Ensure the token has the necessary scopes/permissions"
            ])
        elif status_code == 403:
            suggestions.extend([
                "Your token lacks the required permissions",
                "Contact your administrator to grant necessary access",
                "Check if your account has access to this specific endpoint"
            ])
        elif status_code == 404:
            suggestions.extend([
                "Verify the endpoint URL is correct",
                "Common endpoints: /v1/chat/completions, /v1/completions, /chat/completions",
                "Check API documentation for the correct path",
                "Ensure you're not missing a required path segment"
            ])
        elif status_code == 405:
            suggestions.extend([
                "The endpoint exists but doesn't accept this HTTP method",
                "This often means the URL is correct but needs POST instead of GET",
                "Check API documentation for required HTTP methods"
            ])
        elif status_code >= 500:
            suggestions.extend([
                "The server is experiencing issues",
                "Try again in a few moments",
                "Contact your system administrator if the problem persists",
                "Check service status page if available"
            ])

        return suggestions

    def extract_text_from_image(self, image: Image.Image, prompt: str) -> Optional[str]:
        """Extract text from image using enterprise LLM."""
        try:
            # Prepare image
            buffer = BytesIO()
            image.save(buffer, format='PNG', optimize=True, quality=85)
            image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            # Prepare payload
            payload = {
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": "Convert this image to Markdown:\n[Image attached]"}
                ],
                "max_tokens": 4000,
                "temperature": 0.1,
                "image_data": image_b64
            }

            # Make request
            response = requests.post(
                self.model_url,
                headers=self.headers,
                json=payload,
                timeout=120
            )

            if response.status_code == 200:
                result = response.json()

                # Try different response formats
                if "choices" in result and result["choices"]:
                    return result["choices"][0]["message"]["content"]
                elif "generated_text" in result:
                    return result["generated_text"]
                elif "content" in result:
                    return result["content"]
                elif "response" in result:
                    return result["response"]
                else:
                    logger.warning(f"Unexpected response format: {result.keys()}")
                    return str(result)
            else:
                logger.error(f"LLM request failed with status {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Error in extract_text_from_image: {str(e)}")
            return None


class EnhancedDocVision:
    """Document vision processor with hyperlink awareness."""

    def __init__(self, jwt_token: str, model_url: str):
        """Initialize with Enterprise LLM credentials."""
        self.client = EnterpriseLLMClient(jwt_token, model_url)
        self.base_prompt = self._get_base_prompt()

    def _get_base_prompt(self) -> str:
        """Get the base prompt for slide conversion."""
        return """You are a PowerPoint slide to Markdown converter. Convert the image to clean markdown.

REQUIREMENTS:
1. Extract ALL visible text from the slide
2. Use # for the main title (only ONE per slide)
3. Use ### for subtitles/section headers (not ##)
4. Convert bullet points to markdown lists with proper indentation
5. Preserve table structures using markdown table syntax
6. Maintain the slide's logical structure
7. Include any visible links, captions, or annotations
8. For diagrams with text boxes AND connecting lines/arrows, create mermaid diagrams
9. If only text boxes without connections, just extract the text

OUTPUT FORMAT:
- Clean markdown without code block markers
- Ready for direct use in .md files
- No ```markdown tags needed"""

    def process_with_hyperlinks(self, file_path: Path, hyperlinks_df: pd.DataFrame) -> str:
        """Process document with hyperlink preservation."""
        try:
            # Convert to images
            if file_path.suffix.lower() == '.pdf':
                images = self._pdf_to_images(file_path)
            elif file_path.suffix.lower() in ['.pptx', '.ppt']:
                images = self._powerpoint_to_images(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_path.suffix}")

            if not images:
                return "# Error\n\nCould not convert document to images."

            # Process each slide
            markdown_sections = []
            progress_bar = st.progress(0)

            for i, image in enumerate(images, 1):
                progress_bar.progress(i / len(images))

                # Check for hyperlinks on this slide
                slide_links = hyperlinks_df[
                    hyperlinks_df['Slide Number'] == i] if not hyperlinks_df.empty else pd.DataFrame()

                # Build prompt
                prompt = self.base_prompt
                if not slide_links.empty:
                    link_text = "\n".join([f"[{row['Link Text']}]({row['URL']})" for _, row in slide_links.iterrows()])
                    prompt += f"\n\nIMPORTANT: Preserve these hyperlinks:\n{link_text}"
                    st.info(f"📎 Slide {i}: Processing with {len(slide_links)} hyperlinks")
                else:
                    st.info(f"📄 Slide {i}: Standard processing")

                # Process image
                markdown_text = self._extract_text_from_image(image, prompt)

                if markdown_text:
                    markdown_sections.append(f"<!-- Slide {i} -->\n\n{markdown_text}")
                else:
                    markdown_sections.append(f"<!-- Slide {i} -->\n\n*[Could not extract text from this slide]*")

            progress_bar.empty()

            # Combine sections
            header = f"# {file_path.name}\n\n*AI-enhanced conversion with preserved hyperlinks*\n\n"
            return header + "\n\n---\n\n".join(markdown_sections)

        except Exception as e:
            logger.error(f"Error in process_with_hyperlinks: {str(e)}")
            return f"# Error\n\nFailed to process document: {str(e)}"

    def _pdf_to_images(self, pdf_path: Path) -> List[Image.Image]:
        """Convert PDF to images."""
        try:
            return convert_from_path(str(pdf_path), dpi=200, fmt='PNG')
        except Exception as e:
            logger.error(f"PDF conversion error: {str(e)}")
            st.error(f"Failed to convert PDF: {str(e)}")
            return []

    def _powerpoint_to_images(self, ppt_path: Path) -> List[Image.Image]:
        """Convert PowerPoint to images."""
        import subprocess
        import shutil

        # Find LibreOffice
        libreoffice_path = None
        for cmd in ["/Applications/LibreOffice.app/Contents/MacOS/soffice", "soffice", "libreoffice"]:
            if cmd.startswith("/"):
                if Path(cmd).exists():
                    libreoffice_path = cmd
                    break
            else:
                libreoffice_path = shutil.which(cmd)
                if libreoffice_path:
                    break

        if not libreoffice_path:
            st.error("LibreOffice not found. Please install it first.")
            st.info("Install with: brew install --cask libreoffice (macOS) or apt-get install libreoffice (Linux)")
            return []

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            try:
                # Convert to PDF first
                st.info("Converting PowerPoint to PDF...")
                cmd = [
                    libreoffice_path,
                    "--headless",
                    "--convert-to", "pdf",
                    "--outdir", str(temp_path),
                    str(ppt_path)
                ]

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

                if result.returncode != 0:
                    st.error(f"Conversion failed: {result.stderr}")
                    return []

                # Find generated PDF
                pdf_files = list(temp_path.glob("*.pdf"))
                if not pdf_files:
                    st.error("PDF not generated")
                    return []

                # Convert PDF to images
                st.info("Converting PDF to images...")
                return convert_from_path(str(pdf_files[0]), dpi=200, fmt='PNG')

            except Exception as e:
                st.error(f"PowerPoint conversion error: {str(e)}")
                return []

    def _extract_text_from_image(self, image: Image.Image, prompt: str) -> str:
        """Extract text from image using Enterprise LLM."""
        try:
            # Resize if needed
            if max(image.size) > 2048:
                ratio = 2048 / max(image.size)
                new_size = tuple(int(d * ratio) for d in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)

            # Convert to RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')

            return self.client.extract_text_from_image(image, prompt)

        except Exception as e:
            logger.error(f"Image extraction error: {str(e)}")
            return None


def setup_page():
    """Configure the Streamlit page."""
    st.set_page_config(
        page_title="DocFlow Vision Enterprise",
        page_icon="🔬",
        layout="centered",
        initial_sidebar_state="collapsed"
    )

    # Custom CSS
    st.markdown("""
    <style>
    .main {max-width: 900px; margin: 0 auto;}
    .stButton>button {width: 100%; background-color: #4CAF50; color: white;}
    .stButton>button:hover {background-color: #45a049;}
    h1 {text-align: center; color: #2c3e50;}
    </style>
    """, unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    defaults = {
        "markdown_content": "",
        "file_name": "",
        "extracted_hyperlinks": [],
        "vision_enhanced": False,
        "connection_tested": False,
        "connection_success": False
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def load_configuration() -> Tuple[str, str, Dict]:
    """Load Enterprise LLM configuration."""
    jwt_token = ""
    model_url = ""
    status = {}

    # Try files first
    if os.path.exists("JWT_token.txt"):
        try:
            with open("JWT_token.txt", "r") as f:
                jwt_token = f.read().strip()
                status["JWT Token"] = "✅ Loaded from file"
        except Exception as e:
            status["JWT Token"] = f"❌ Error reading file: {e}"
    else:
        status["JWT Token"] = "❌ JWT_token.txt not found"

    if os.path.exists("model_url.txt"):
        try:
            with open("model_url.txt", "r") as f:
                model_url = f.read().strip()
                status["Model URL"] = "✅ Loaded from file"
        except Exception as e:
            status["Model URL"] = f"❌ Error reading file: {e}"
    else:
        status["Model URL"] = "❌ model_url.txt not found"

    # Try environment variables as fallback
    if not jwt_token:
        jwt_token = os.getenv("ENTERPRISE_JWT_TOKEN", "")
        if jwt_token:
            status["JWT Token"] = "✅ Loaded from environment"

    if not model_url:
        model_url = os.getenv("ENTERPRISE_MODEL_URL", "")
        if model_url:
            status["Model URL"] = "✅ Loaded from environment"

    return jwt_token, model_url, status


def extract_hyperlinks(markdown_content: str) -> List[Dict]:
    """Extract hyperlinks from markdown content."""
    if not markdown_content:
        return []

    hyperlinks = []
    current_slide = 1

    for line in markdown_content.split('\n'):
        # Track slide numbers
        if match := re.search(r'<!--\s*Slide\s*(\d+)\s*-->', line):
            current_slide = int(match.group(1))
        elif match := re.search(r'^#+\s*Slide\s*(\d+)', line):
            current_slide = int(match.group(1))

        # Find markdown links
        for match in re.finditer(r'\[([^\]]+)\]\(([^)]+)\)', line):
            text = re.sub(r'\*{1,3}', '', match.group(1)).strip()
            url = match.group(2)

            if not text.startswith('!') and url != "image":
                hyperlinks.append({
                    'Link Text': text,
                    'URL': url,
                    'Slide Number': current_slide
                })

    return hyperlinks


def process_document(uploaded_file, use_vision: bool, jwt_token: str, model_url: str):
    """Main document processing function."""
    if not CONVERTER_AVAILABLE:
        st.error("❌ Document converter not available. Check installation.")
        return

    try:
        file_data = uploaded_file.getbuffer()

        # Step 1: XML Extraction
        with st.spinner("📊 Extracting document structure..."):
            xml_markdown, error = convert_file_to_markdown(
                file_data,
                uploaded_file.name,
                enhance=False,
                api_key=None
            )

            if error:
                st.error(f"Extraction failed: {error}")
                return

            st.session_state.xml_markdown = xml_markdown

        # Step 2: Extract hyperlinks
        hyperlinks = extract_hyperlinks(xml_markdown)
        st.session_state.extracted_hyperlinks = hyperlinks

        if hyperlinks:
            hyperlinks_df = pd.DataFrame(hyperlinks)
            st.success(f"✅ Found {len(hyperlinks)} hyperlinks")
        else:
            hyperlinks_df = pd.DataFrame()
            st.info("No hyperlinks found")

        # Step 3: Vision processing if requested
        if use_vision and jwt_token and model_url:
            if not DOCVISION_AVAILABLE:
                st.error("❌ Vision dependencies not installed")
                st.info("Install: pip install pdf2image pillow")
                st.session_state.markdown_content = xml_markdown
                st.session_state.vision_enhanced = False
            else:
                # Test connection first
                client = EnterpriseLLMClient(jwt_token, model_url)
                result = client.test_connection_detailed()

                if not result["success"]:
                    st.error("❌ **Enterprise LLM Connection Failed**")

                    with st.expander("🔍 Connection Diagnostics", expanded=True):
                        st.json(result)

                    st.warning("⚠️ Using XML extraction only (connection failed)")
                    st.session_state.markdown_content = xml_markdown
                    st.session_state.vision_enhanced = False
                else:
                    st.success(f"✅ Connected via {result['method']}")

                    # Process with vision
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                        tmp.write(file_data)
                        tmp_path = Path(tmp.name)

                    try:
                        processor = EnhancedDocVision(jwt_token, model_url)
                        vision_markdown = processor.process_with_hyperlinks(tmp_path, hyperlinks_df)

                        st.session_state.markdown_content = vision_markdown
                        st.session_state.vision_enhanced = True
                        st.success("✨ Vision processing complete!")
                    finally:
                        if tmp_path.exists():
                            tmp_path.unlink()
        else:
            # Use XML extraction only
            st.session_state.markdown_content = xml_markdown
            st.session_state.vision_enhanced = False
            st.success("✅ Document converted successfully!")

        st.session_state.file_name = uploaded_file.name

    except Exception as e:
        st.error(f"Processing error: {str(e)}")
        logger.exception("Processing error")


def main():
    """Main application entry point."""
    setup_page()
    initialize_session_state()

    # Header
    st.title("🔬 DocFlow Vision Enterprise")
    st.markdown("*AI-powered document conversion with hyperlink preservation*")
    st.markdown("---")

    # Configuration
    jwt_token, model_url, config_status = load_configuration()

    with st.expander("⚙️ Configuration", expanded=not st.session_state.connection_tested):
        for key, status in config_status.items():
            st.markdown(f"{status}")

        if jwt_token and model_url:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔍 Test Connection"):
                    with st.spinner("Testing..."):
                        client = EnterpriseLLMClient(jwt_token, model_url)
                        result = client.test_connection_detailed()
                        st.session_state.connection_tested = True
                        st.session_state.connection_success = result["success"]

                        if result["success"]:
                            st.success(f"✅ Connected via {result['method']}")
                        else:
                            st.error("❌ Connection failed")
                            with st.expander("Details"):
                                st.json(result)

            with col2:
                use_vision = st.checkbox(
                    "🤖 Use Vision Processing",
                    value=st.session_state.connection_success,
                    disabled=not st.session_state.connection_success
                )
        else:
            st.error("❌ Configuration incomplete")
            st.markdown("""
            **Required files:**
            - `JWT_token.txt` - Your JWT token
            - `model_url.txt` - Enterprise LLM endpoint URL
            """)
            use_vision = False

    st.markdown("---")

    # File upload
    uploaded_file = st.file_uploader(
        "Upload Document",
        type=['pptx', 'ppt', 'pdf', 'docx', 'doc'],
        help="Upload a PowerPoint, PDF, or Word document"
    )

    if uploaded_file:
        st.info(f"📄 {uploaded_file.name} ({uploaded_file.size:,} bytes)")

        if st.button("🚀 Convert to Markdown", type="primary"):
            process_document(uploaded_file, use_vision, jwt_token, model_url)

    # Display results
    if st.session_state.markdown_content:
        st.markdown("---")
        st.markdown("### 📝 Results")

        if st.session_state.vision_enhanced:
            st.success("Enhanced with Enterprise LLM Vision")
        else:
            st.info("Processed with XML extraction")

        # Download button
        st.download_button(
            "📥 Download Markdown",
            data=st.session_state.markdown_content,
            file_name=f"{Path(st.session_state.file_name).stem}.md",
            mime="text/markdown"
        )

        # Preview
        with st.expander("Preview Markdown"):
            st.text_area(
                "Content",
                value=st.session_state.markdown_content,
                height=400,
                label_visibility="collapsed"
            )

        # Hyperlinks table
        if st.session_state.extracted_hyperlinks:
            st.markdown("### 🔗 Extracted Hyperlinks")
            df = pd.DataFrame(st.session_state.extracted_hyperlinks)

            csv = df.to_csv(index=False)
            st.download_button(
                "💾 Download Hyperlinks CSV",
                data=csv,
                file_name=f"{Path(st.session_state.file_name).stem}_hyperlinks.csv",
                mime="text/csv"
            )

            st.dataframe(df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()