"""
PowerPoint to PDF to OpenAI Vision Converter
Converts PowerPoint to PDF, then uses OpenAI's GPT-4 Vision for text extraction
Enhanced with better poppler detection and error handling
"""

import os
import json
import logging
import base64
import tempfile
import subprocess
import platform
import shutil
from io import BytesIO
from typing import Tuple, Optional, List
from PIL import Image
from openai import OpenAI
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import pdf2image - provide helpful error if not available
try:
    from pdf2image import convert_from_path
    from pdf2image.exceptions import PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError

    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    logger.warning("pdf2image not installed. Run: pip install pdf2image")

# Prompt for PDF/PowerPoint image to markdown conversion
POWERPOINT_IMAGE_PROMPT = """
You are a PowerPoint slide to Markdown converter. You receive an image of a PowerPoint slide and must convert all visible text and structure into clean, professional markdown.

Your job:
1. Extract ALL text content from the slide
2. Identify the main title and format as # header (only ONE per slide)
3. Identify subtitles or section headers and format as ### headers
4. Convert bullet points to proper markdown lists with correct indentation (use 2 spaces for nested bullets)
5. Preserve table structures using markdown table syntax
6. Extract and format any numbered lists
7. Maintain slide hierarchy and structure
8. Include any visible links, captions, or annotations
9. Format code blocks or technical content appropriately
10. If there are diagrams, describe them briefly in italics

Key Rules:
- Extract ALL visible text - don't miss anything
- Each slide should have only ONE main # heading
- Use ### for subheadings (not ##)
- Use proper markdown syntax throughout
- Maintain the original slide's logical structure
- If text is unclear, make your best reasonable interpretation
- Don't add content that isn't visible in the image
- Format tables properly with | separators
- Preserve bullet point hierarchies with proper indentation

Output clean, readable markdown that captures everything visible on the slide.
"""


class PowerPointToMarkdownConverter:
    """Converts PowerPoint to PDF then to markdown via OpenAI Vision."""

    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize with OpenAI API key."""
        logger.info("🔧 Initializing PowerPoint to Markdown Converter...")

        # Get API key from parameter or environment
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "❌ OpenAI API key required. Set OPENAI_API_KEY environment variable or pass api_key parameter")

        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)

        # Check for poppler
        self.poppler_available = self._check_poppler()
        if not self.poppler_available:
            logger.warning("⚠️ Poppler not found. PDF to image conversion may fail.")
            logger.info("📝 Install poppler:")
            logger.info("   Ubuntu/WSL: sudo apt-get install poppler-utils")
            logger.info("   macOS: brew install poppler")
            logger.info("   Windows: Download from GitHub and add to PATH")

        # Detect LibreOffice for PDF conversion
        self.libreoffice_path = self._detect_libreoffice()

        if not self.libreoffice_path:
            raise ValueError("❌ LibreOffice not found. Please install LibreOffice for PowerPoint to PDF conversion")

        logger.info("✅ PowerPoint converter ready")
        if self.poppler_available:
            logger.info("✅ Poppler detected")

    def _check_poppler(self) -> bool:
        """Check if poppler (pdftoppm) is available."""
        # Add Homebrew paths for macOS
        commands_to_try = [
            "pdftoppm",
            "/opt/homebrew/bin/pdftoppm",  # ARM Mac homebrew location
            "/usr/local/bin/pdftoppm",  # Intel Mac homebrew location
            "/usr/bin/pdftoppm",
        ]

        # Also try using full shell to find it
        try:
            result = subprocess.run(
                ["bash", "-c", "which pdftoppm"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                logger.info(f"✅ Found poppler via bash: {result.stdout.strip()}")
                return True
        except:
            pass

        for cmd in commands_to_try:
            try:
                if os.path.exists(cmd):
                    result = subprocess.run(
                        [cmd, "-h"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if result.returncode == 0 or "Usage:" in result.stdout or "Usage:" in result.stderr:
                        logger.info(f"✅ Found poppler command: {cmd}")
                        return True
            except (subprocess.SubprocessError, FileNotFoundError):
                continue

        return False

    def _detect_libreoffice(self) -> Optional[str]:
        """Detect LibreOffice installation across platforms."""
        system = platform.system()

        possible_paths = []

        if system == "Darwin":  # macOS
            possible_paths = [
                "/Applications/LibreOffice.app/Contents/MacOS/soffice",
                "/opt/homebrew/bin/soffice",  # ARM Mac homebrew
                "/usr/local/bin/soffice"  # Intel Mac homebrew
            ]
        elif system == "Windows":
            possible_paths = [
                "C:\\Program Files\\LibreOffice\\program\\soffice.exe",
                "C:\\Program Files (x86)\\LibreOffice\\program\\soffice.exe",
            ]
        else:  # Linux/WSL
            possible_paths = [
                "/usr/bin/soffice",
                "/usr/local/bin/soffice",
                "/snap/bin/libreoffice",
                "/usr/bin/libreoffice"
            ]

        # Check each possible path
        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"✅ Found LibreOffice at: {path}")
                return path

        # Try using bash to find it
        try:
            result = subprocess.run(
                ["bash", "-c", "which soffice"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                path = result.stdout.strip()
                logger.info(f"✅ Found LibreOffice via bash: {path}")
                return path
        except:
            pass

        return None

    def convert_powerpoint_to_markdown(
            self,
            pptx_path: str,
            dpi: int = 200,
            max_workers: int = 5
    ) -> Tuple[str, Optional[str]]:
        """
        Convert PowerPoint to markdown via PDF and OpenAI Vision.

        Args:
            pptx_path: Path to PowerPoint file
            dpi: Resolution for image conversion (200-300 recommended)
            max_workers: Maximum parallel API calls to OpenAI

        Returns:
            Tuple of (markdown_content, error_message)
        """
        logger.info("📊 Converting PowerPoint: %s", pptx_path)

        # Check if file exists
        if not os.path.exists(pptx_path):
            return "", f"File not found: {pptx_path}"

        try:
            # Step 1: Convert PowerPoint to PDF
            with tempfile.TemporaryDirectory() as temp_dir:
                pdf_path = self._powerpoint_to_pdf(pptx_path, temp_dir)

                if not pdf_path:
                    return "", "Failed to convert PowerPoint to PDF. Check LibreOffice installation."

                # Step 2: Convert PDF to images
                try:
                    images = self._pdf_to_images(pdf_path, dpi)
                    logger.info("🖼️ Converted %d slides to images", len(images))
                except Exception as e:
                    error_msg = f"Failed to convert PDF to images: {str(e)}"
                    logger.error("❌ %s", error_msg)

                    # Provide specific help for poppler issues
                    if "poppler" in str(e).lower() or "PDFInfoNotInstalledError" in str(type(e)):
                        error_msg += "\n\nPoppler is not installed. Please install it:\n"
                        error_msg += "• Ubuntu/WSL: sudo apt-get install poppler-utils\n"
                        error_msg += "• macOS: brew install poppler\n"
                        error_msg += "• Windows: Download from GitHub and add to PATH"

                    return "", error_msg

                # Step 3: Process each image with OpenAI Vision
                markdown_slides = []

                for i, image in enumerate(images, 1):
                    logger.info("🚀 Processing slide %d/%d...", i, len(images))
                    slide_markdown, error = self._image_to_markdown_openai(image, i)

                    if error:
                        logger.warning("⚠️ Slide %d failed: %s", i, error)
                        markdown_slides.append(
                            f"<!-- Slide {i} - Conversion failed: {error} -->"
                        )
                    else:
                        slide_content = f"<!-- Slide {i} -->\n\n{slide_markdown}"
                        markdown_slides.append(slide_content)
                        logger.info("✅ Slide %d converted successfully", i)

                # Combine all slides
                final_markdown = "\n\n---\n\n".join(markdown_slides)

                # Add header
                filename = os.path.basename(pptx_path)
                header = (
                    f"# {filename}\n\n"
                    "*Converted from PowerPoint using OpenAI GPT-4 Vision*\n\n"
                )
                final_markdown = header + final_markdown

                logger.info(
                    "✅ PowerPoint conversion complete: %d characters",
                    len(final_markdown)
                )
                return final_markdown, None

        except Exception as e:
            error_msg = f"PowerPoint conversion failed: {str(e)}"
            logger.error("❌ %s", error_msg)
            return "", error_msg

    def _powerpoint_to_pdf(self, pptx_path: str, output_dir: str) -> Optional[str]:
        """Convert PowerPoint to PDF using LibreOffice."""
        logger.info("📄 Converting PowerPoint to PDF...")

        cmd = [
            self.libreoffice_path,
            "--headless",
            "--convert-to", "pdf",
            "--outdir", output_dir,
            pptx_path
        ]

        try:
            logger.debug("Running command: %s", " ".join(cmd))

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout
            )

            if result.returncode != 0:
                logger.error("❌ LibreOffice conversion failed")
                logger.error("STDOUT: %s", result.stdout)
                logger.error("STDERR: %s", result.stderr)
                return None

            # Find the generated PDF
            pdf_files = [f for f in os.listdir(output_dir) if f.endswith('.pdf')]

            if not pdf_files:
                logger.error("❌ No PDF file was generated")
                logger.error("Directory contents: %s", os.listdir(output_dir))
                return None

            pdf_path = os.path.join(output_dir, pdf_files[0])

            if not os.path.exists(pdf_path):
                logger.error("❌ PDF file not found at expected path: %s", pdf_path)
                return None

            pdf_size = os.path.getsize(pdf_path) / 1024
            logger.info("✅ PDF created: %s (%.1f KB)", pdf_files[0], pdf_size)

            return pdf_path

        except subprocess.TimeoutExpired:
            logger.error("❌ PDF conversion timed out")
            return None
        except Exception as e:
            logger.error("❌ Error during PDF conversion: %s", str(e))
            return None

    def _pdf_to_images(self, pdf_path: str, dpi: int) -> List[Image.Image]:
        """Convert PDF pages to PIL Images using poppler."""
        if not PDF2IMAGE_AVAILABLE:
            raise Exception("pdf2image is not installed. Run: pip install pdf2image")

        logger.info("Converting PDF to images at %d DPI...", dpi)

        try:
            # For macOS with Homebrew
            poppler_path = None
            if platform.system() == "Darwin":
                if os.path.exists("/opt/homebrew/bin"):
                    poppler_path = "/opt/homebrew/bin"
                elif os.path.exists("/usr/local/bin"):
                    poppler_path = "/usr/local/bin"

            # Convert with explicit poppler path
            images = convert_from_path(
                pdf_path,
                dpi=dpi,
                fmt='PNG',
                thread_count=4,
                poppler_path=poppler_path  # This tells pdf2image where to find pdftoppm
            )

            logger.info("Successfully converted %d pages", len(images))
            return images
        except Exception as e:
            logger.error("PDF to image conversion failed: %s", str(e))
            raise

    def _image_to_markdown_openai(
            self,
            image: Image.Image,
            slide_num: int
    ) -> Tuple[str, Optional[str]]:
        """Convert single image to markdown via OpenAI Vision API."""
        try:
            # Convert image to base64
            base64_image = self._image_to_base64(image)

            # Call OpenAI Vision API
            response = self.client.chat.completions.create(
                model="gpt-4o",  # or "gpt-4-vision-preview" if you prefer
                messages=[
                    {
                        "role": "system",
                        "content": POWERPOINT_IMAGE_PROMPT
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Please convert this PowerPoint slide (slide {slide_num}) to markdown:"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}",
                                    "detail": "high"  # Use high detail for better text extraction
                                }
                            }
                        ]
                    }
                ],
                max_tokens=4000,
                temperature=0.1
            )

            # Extract content from response
            content = response.choices[0].message.content
            return content, None

        except Exception as e:
            error_msg = f"OpenAI API error: {str(e)}"
            logger.error("❌ %s", error_msg)
            return "", error_msg

    def _image_to_base64(self, image: Image.Image, quality: int = 85) -> str:
        """Convert PIL Image to base64 string."""
        # Optimize image size for API (OpenAI has limits)
        max_size = 2048  # OpenAI recommended max dimension
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            logger.debug("Resized image to %s for API limits", new_size)

        # Convert to PNG for better quality (OpenAI prefers PNG)
        buffer = BytesIO()
        image.save(buffer, format='PNG', optimize=True)
        image_bytes = buffer.getvalue()

        return base64.b64encode(image_bytes).decode('utf-8')


def test_dependencies():
    """Test if all dependencies are properly installed."""
    print("🔍 Checking dependencies...\n")

    # Check Python packages
    packages = {
        "pdf2image": False,
        "PIL": False,
        "openai": False
    }

    try:
        import pdf2image
        packages["pdf2image"] = True
        print("✅ pdf2image installed")
    except ImportError:
        print("❌ pdf2image not installed - Run: pip install pdf2image")

    try:
        from PIL import Image
        packages["PIL"] = True
        print("✅ Pillow (PIL) installed")
    except ImportError:
        print("❌ Pillow not installed - Run: pip install pillow")

    try:
        import openai
        packages["openai"] = True
        print("✅ OpenAI library installed")
    except ImportError:
        print("❌ OpenAI not installed - Run: pip install openai")

    print("\n🔍 Checking system dependencies...\n")

    # Check poppler - updated for macOS paths
    poppler_found = False
    poppler_paths = [
        "/opt/homebrew/bin/pdftoppm",  # ARM Mac
        "/usr/local/bin/pdftoppm",  # Intel Mac
        "pdftoppm"
    ]

    for cmd in poppler_paths:
        try:
            if os.path.exists(cmd):
                result = subprocess.run([cmd, "-h"], capture_output=True, timeout=5)
                if result.returncode == 0 or "Usage" in str(result.stdout) + str(result.stderr):
                    poppler_found = True
                    print(f"✅ Poppler found: {cmd}")
                    break
        except:
            continue

    # Also try with bash
    if not poppler_found:
        try:
            result = subprocess.run(
                ["bash", "-c", "which pdftoppm"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                poppler_found = True
                print(f"✅ Poppler found via bash: {result.stdout.strip()}")
        except:
            pass

    if not poppler_found:
        print("❌ Poppler not found")
        print("   Install instructions:")
        print("   • macOS: brew install poppler")

    # Check LibreOffice - updated for macOS paths
    libreoffice_found = False
    libreoffice_paths = [
        "/Applications/LibreOffice.app/Contents/MacOS/soffice",
        "/opt/homebrew/bin/soffice",  # ARM Mac
        "/usr/local/bin/soffice"  # Intel Mac
    ]

    for path in libreoffice_paths:
        if os.path.exists(path):
            libreoffice_found = True
            print(f"✅ LibreOffice found: {path}")
            break

    # Also try with bash
    if not libreoffice_found:
        try:
            result = subprocess.run(
                ["bash", "-c", "which soffice"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                libreoffice_found = True
                print(f"✅ LibreOffice found via bash: {result.stdout.strip()}")
        except:
            pass

    if not libreoffice_found:
        print("❌ LibreOffice not found")
        print("   Install instructions:")
        print("   • macOS: brew install --cask libreoffice")

    # Summary
    print("\n" + "=" * 50)
    all_good = all(packages.values()) and poppler_found and libreoffice_found

    if all_good:
        print("✅ All dependencies are installed!")
    else:
        print("⚠️ Some dependencies are missing. Please install them.")

    return all_good


def convert_powerpoint_to_markdown(
        pptx_path: str,
        output_path: Optional[str] = None,
        api_key: Optional[str] = None,
        dpi: int = 200
) -> Tuple[str, Optional[str]]:
    """
    Simple function to convert PowerPoint to markdown.

    Args:
        pptx_path: Path to input PowerPoint file
        output_path: Optional path to save markdown file
        api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        dpi: Image resolution (200-300 recommended)

    Returns:
        Tuple of (markdown_content, error_message)
    """
    try:
        converter = PowerPointToMarkdownConverter(openai_api_key=api_key)
        markdown_content, error = converter.convert_powerpoint_to_markdown(pptx_path, dpi)

        if not error and output_path:
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)
                logger.info("✅ Markdown saved to: %s", output_path)
            except Exception as e:
                logger.warning("⚠️ Failed to save file: %s", e)

        return markdown_content, error

    except Exception as e:
        error_msg = f"Initialization failed: {str(e)}"
        logger.error("❌ %s", error_msg)
        return "", error_msg


# Integration with existing app.py or app_v3.py
def add_openai_conversion_to_app(uploaded_file, api_key: str):
    """
    Add this to your Streamlit app for OpenAI Vision conversion.

    Usage in app_v3.py:

    if st.button("🎯 Convert with OpenAI Vision"):
        markdown, error = add_openai_conversion_to_app(uploaded_file, openai_api_key)
        if not error:
            st.session_state.markdown_content = markdown
            st.success("✅ Converted with OpenAI Vision!")
    """
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pptx") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name

    try:
        markdown, error = convert_powerpoint_to_markdown(
            pptx_path=tmp_path,
            api_key=api_key,
            dpi=250  # Higher quality for better OCR
        )
        return markdown, error
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# Example usage
if __name__ == "__main__":
    # First, test dependencies
    print("=" * 50)
    print("DEPENDENCY CHECK")
    print("=" * 50)
    test_dependencies()

    print("\n" + "=" * 50)
    print("CONVERSION TEST")
    print("=" * 50)

    # Example: Convert a PowerPoint file
    pptx_file = "test_powerpoint.pptx"  # Replace with your PowerPoint path
    output_file = "presentation_openai.md"  # Optional output file

    # Set your OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")  # Or set directly: "sk-..."

    if not api_key:
        print("❌ Please set OPENAI_API_KEY environment variable")
    else:
        try:
            markdown, error = convert_powerpoint_to_markdown(
                pptx_path=pptx_file,
                output_path=output_file,
                api_key=api_key,
                dpi=250  # Higher quality for better text extraction
            )

            if error:
                print(f"❌ Conversion failed: {error}")
            else:
                print(f"✅ Success! Generated {len(markdown)} characters of markdown")
                print(f"📄 Saved to: {output_file}")

        except Exception as e:
            print(f"❌ Error: {e}")