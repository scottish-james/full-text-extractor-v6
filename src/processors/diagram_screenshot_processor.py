"""
PDF-Based Screenshot Processor - Much more reliable than LibreOffice direct export
Replace your entire src/processors/diagram_screenshot_processor.py with this content
"""

import os
import subprocess
import platform
import shutil
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class DiagramScreenshotProcessor:
    """
    Handles PowerPoint screenshot generation using PDF conversion method.
    Much more reliable than direct LibreOffice slide export.
    """

    def __init__(self):
        self.libreoffice_path = self._detect_libreoffice()
        self.poppler_available = self._check_poppler()

    def _detect_libreoffice(self) -> Optional[str]:
        """Detect LibreOffice installation across platforms."""
        system = platform.system()

        possible_paths = []

        if system == "Darwin":  # macOS
            possible_paths = [
                "/Applications/LibreOffice.app/Contents/MacOS/soffice",
                "/opt/homebrew/bin/soffice",
                "/usr/local/bin/soffice"
            ]
        elif system == "Windows":
            possible_paths = [
                "C:\\Program Files\\LibreOffice\\program\\soffice.exe",
                "C:\\Program Files (x86)\\LibreOffice\\program\\soffice.exe",
            ]
        else:  # Linux
            possible_paths = [
                "/usr/bin/soffice",
                "/usr/local/bin/soffice"
            ]

        # Check each possible path
        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"Found LibreOffice at: {path}")
                return path

        # Try to find in PATH
        try:
            result = subprocess.run(
                ["which", "soffice"] if system != "Windows" else ["where", "soffice"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                path = result.stdout.strip().split('\n')[0]
                logger.info(f"Found LibreOffice in PATH: {path}")
                return path
        except:
            pass

        return None

    def _check_poppler(self) -> bool:
        """Check if poppler-utils (pdftoppm) is available."""
        try:
            result = subprocess.run(
                ["pdftoppm", "-h"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except:
            # Try alternative locations
            try:
                result = subprocess.run(
                    ["/usr/local/bin/pdftoppm", "-h"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                return result.returncode == 0
            except:
                return False

    def _get_pdftoppm_command(self) -> str:
        """Get the pdftoppm command path."""
        # Try common locations
        possible_paths = ["pdftoppm", "/usr/local/bin/pdftoppm", "/opt/homebrew/bin/pdftoppm"]

        for path in possible_paths:
            try:
                result = subprocess.run([path, "-h"], capture_output=True, timeout=5)
                if result.returncode == 0:
                    return path
            except:
                continue

        return "pdftoppm"  # Default fallback

    def is_available(self) -> bool:
        """Check if screenshot capability is available."""
        return self.libreoffice_path is not None

    def screenshot_slides_with_all_methods(self,
                                           pptx_path: str,
                                           slide_numbers: List[int],
                                           output_dir: str,
                                           base_filename: str,
                                           debug_mode: bool = True) -> Dict[int, str]:
        """
        Try all available methods to screenshot slides.

        Methods in order of preference:
        1. PDF method (most reliable)
        2. Direct LibreOffice export (original method)

        Args:
            pptx_path (str): Path to PowerPoint file
            slide_numbers (List[int]): Slide numbers to screenshot (1-based)
            output_dir (str): Directory to save screenshots
            base_filename (str): Base filename for screenshots
            debug_mode (bool): Show debug information

        Returns:
            Dict[int, str]: Mapping of slide numbers to screenshot file paths
        """
        if debug_mode:
            print(f"🎯 Attempting to screenshot slides: {slide_numbers}")
            print(f"📊 Available methods:")
            print(f"  • PDF method: {'✅' if self.is_available() else '❌'}")
            print(f"  • Poppler available: {'✅' if self.poppler_available else '❌'}")

        # Method 1: PDF conversion approach (most reliable)
        if debug_mode:
            print(f"\n🔥 Trying PDF method (most reliable)...")

        try:
            results = self.screenshot_slides_pdf_method(
                pptx_path, slide_numbers, output_dir, base_filename, debug_mode
            )

            if len(results) == len(slide_numbers):
                if debug_mode:
                    print(f"🎉 PDF method succeeded for all slides!")
                return results
            elif len(results) > 0:
                if debug_mode:
                    print(f"⚠️  PDF method partially succeeded ({len(results)}/{len(slide_numbers)})")
                return results
            else:
                if debug_mode:
                    print(f"❌ PDF method failed, trying fallback...")
        except Exception as e:
            if debug_mode:
                print(f"❌ PDF method error: {str(e)}")

        # Method 2: Original LibreOffice direct method (fallback)
        if debug_mode:
            print(f"\n🔄 Trying original LibreOffice method (fallback)...")

        try:
            # Use the original method as absolute fallback
            results = self._original_libreoffice_method(
                pptx_path, slide_numbers, output_dir, base_filename, debug_mode
            )
            return results
        except Exception as e:
            if debug_mode:
                print(f"❌ All methods failed: {str(e)}")
            return {}

    def screenshot_slides_pdf_method(self,
                                     pptx_path: str,
                                     slide_numbers: List[int],
                                     output_dir: str,
                                     base_filename: str,
                                     debug_mode: bool = True) -> Dict[int, str]:
        """
        Screenshot specific slides using PDF conversion method.
        Much more reliable than direct LibreOffice slide export.

        Steps:
        1. Convert PowerPoint to PDF using LibreOffice
        2. Extract specific pages from PDF as PNG images
        3. Clean up temporary files

        Args:
            pptx_path (str): Path to PowerPoint file
            slide_numbers (List[int]): Slide numbers to screenshot (1-based)
            output_dir (str): Directory to save screenshots
            base_filename (str): Base filename for screenshots
            debug_mode (bool): Show debug information

        Returns:
            Dict[int, str]: Mapping of slide numbers to screenshot file paths
        """
        if not self.is_available():
            raise RuntimeError("LibreOffice not available")

        os.makedirs(output_dir, exist_ok=True)
        results = {}

        with tempfile.TemporaryDirectory() as temp_dir:
            if debug_mode:
                print(f"🔧 Using temp directory: {temp_dir}")
                print(f"🎯 Requested slides: {slide_numbers}")
                print(f"📄 Poppler available: {self.poppler_available}")

            try:
                # Step 1: Convert PowerPoint to PDF
                pdf_path = self._convert_pptx_to_pdf(pptx_path, temp_dir, debug_mode)

                if not pdf_path or not os.path.exists(pdf_path):
                    if debug_mode:
                        print("❌ Failed to convert PowerPoint to PDF")
                    return {}

                # Step 2: Extract specific slides from PDF
                if self.poppler_available:
                    results = self._extract_slides_with_poppler(
                        pdf_path, slide_numbers, output_dir, base_filename, debug_mode
                    )
                else:
                    # Fallback: Use LibreOffice to convert PDF pages
                    results = self._extract_slides_with_libreoffice(
                        pdf_path, slide_numbers, output_dir, base_filename, debug_mode
                    )

            except Exception as e:
                if debug_mode:
                    print(f"❌ Error in PDF method: {str(e)}")

        if debug_mode:
            print(f"\n📊 Final PDF method results: {len(results)} of {len(slide_numbers)} slides captured")
            for slide_num in slide_numbers:
                status = "✅" if slide_num in results else "❌"
                print(f"  {status} Slide {slide_num}")

        return results

    def _convert_pptx_to_pdf(self, pptx_path: str, temp_dir: str, debug_mode: bool) -> Optional[str]:
        """Convert PowerPoint to PDF using LibreOffice."""
        if debug_mode:
            print(f"📄 Converting PowerPoint to PDF...")

        # LibreOffice command to convert to PDF
        cmd = [
            self.libreoffice_path,
            "--headless",
            "--convert-to", "pdf",
            "--outdir", temp_dir,
            pptx_path
        ]

        if debug_mode:
            print(f"🚀 Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout for PDF conversion
            )

            if result.returncode != 0:
                if debug_mode:
                    print(f"❌ LibreOffice PDF conversion failed: {result.stderr}")
                return None

            # Find the generated PDF
            pdf_files = [f for f in os.listdir(temp_dir) if f.endswith('.pdf')]

            if not pdf_files:
                if debug_mode:
                    print("❌ No PDF file was generated")
                return None

            pdf_path = os.path.join(temp_dir, pdf_files[0])

            if debug_mode:
                pdf_size = os.path.getsize(pdf_path) / 1024
                print(f"✅ PDF created: {pdf_files[0]} ({pdf_size:.1f} KB)")

            return pdf_path

        except subprocess.TimeoutExpired:
            if debug_mode:
                print("❌ PDF conversion timed out")
            return None
        except Exception as e:
            if debug_mode:
                print(f"❌ Error during PDF conversion: {str(e)}")
            return None

    # Replace these 3 functions in your existing DiagramScreenshotProcessor class
    # Keep the same function names - just enhanced with better quality

    def _extract_slides_with_poppler(self,
                                     pdf_path: str,
                                     slide_numbers: List[int],
                                     output_dir: str,
                                     base_filename: str,
                                     debug_mode: bool = True) -> Dict[int, str]:
        """Extract specific slides using poppler-utils with HIGH QUALITY settings."""
        if debug_mode:
            print(f"🖼️  Extracting slides using poppler-utils (HIGH QUALITY)...")

        results = {}
        pdftoppm_cmd = self._get_pdftoppm_command()

        for slide_num in slide_numbers:
            if debug_mode:
                print(f"\n🎯 Extracting slide {slide_num} with poppler (high quality)...")

            # Output filename for this slide
            output_filename = f"{base_filename}_slide_{slide_num:02d}_diagram.png"
            output_path = os.path.join(output_dir, output_filename)

            # HIGH QUALITY pdftoppm command with better settings
            cmd = [
                pdftoppm_cmd,
                "-png",  # PNG format (good for diagrams)
                "-r", "300",  # 300 DPI resolution (high quality)
                "-scale-to", "2048",  # Scale to 2048px width (good for OpenAI)
                "-aa", "yes",  # Enable anti-aliasing
                "-aaVector", "yes",  # Anti-aliasing for vector graphics
                "-f", str(slide_num),  # First page
                "-l", str(slide_num),  # Last page
                "-singlefile",  # Single file output
                pdf_path,
                os.path.join(output_dir, f"{base_filename}_slide_{slide_num:02d}_diagram")
            ]

            if debug_mode:
                print(f"🚀 Running HIGH QUALITY: {' '.join(cmd)}")

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60  # Longer timeout for high-res processing
                )

                if result.returncode != 0:
                    if debug_mode:
                        print(f"❌ pdftoppm failed for slide {slide_num}: {result.stderr}")
                    continue

                # Check if the file was created
                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path) / 1024
                    results[slide_num] = output_path

                    if debug_mode:
                        print(f"✅ HIGH QUALITY Slide {slide_num}: {output_filename} ({file_size:.1f} KB)")
                else:
                    if debug_mode:
                        print(f"❌ Output file not found for slide {slide_num}: {output_path}")

            except subprocess.TimeoutExpired:
                if debug_mode:
                    print(f"❌ pdftoppm timed out for slide {slide_num}")
                continue
            except Exception as e:
                if debug_mode:
                    print(f"❌ Error extracting slide {slide_num}: {str(e)}")
                continue

        return results

    def _convert_pptx_to_pdf(self, pptx_path: str, temp_dir: str, debug_mode: bool) -> Optional[str]:
        """Convert PowerPoint to PDF using LibreOffice with BETTER QUALITY settings."""
        if debug_mode:
            print(f"📄 Converting PowerPoint to HIGH QUALITY PDF...")

        # Enhanced LibreOffice command with quality settings
        cmd = [
            self.libreoffice_path,
            "--headless",
            "--convert-to", "pdf",
            "--outdir", temp_dir,
            # Quality improvements for PDF conversion
            "-env:UserInstallation=file:///tmp/lo_temp_profile_" + str(os.getpid()),
            pptx_path
        ]

        if debug_mode:
            print(f"🚀 Running HIGH QUALITY PDF: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=180  # Longer timeout for better quality
            )

            if result.returncode != 0:
                if debug_mode:
                    print(f"❌ LibreOffice PDF conversion failed: {result.stderr}")
                return None

            # Find the generated PDF
            pdf_files = [f for f in os.listdir(temp_dir) if f.endswith('.pdf')]

            if not pdf_files:
                if debug_mode:
                    print("❌ No PDF file was generated")
                return None

            pdf_path = os.path.join(temp_dir, pdf_files[0])

            if debug_mode:
                pdf_size = os.path.getsize(pdf_path) / 1024
                print(f"✅ HIGH QUALITY PDF created: {pdf_files[0]} ({pdf_size:.1f} KB)")

            return pdf_path

        except subprocess.TimeoutExpired:
            if debug_mode:
                print("❌ PDF conversion timed out")
            return None
        except Exception as e:
            if debug_mode:
                print(f"❌ Error during PDF conversion: {str(e)}")
            return None

    def _original_libreoffice_method(self,
                                     pptx_path: str,
                                     slide_numbers: List[int],
                                     output_dir: str,
                                     base_filename: str,
                                     debug_mode: bool) -> Dict[int, str]:
        """Original LibreOffice method as last resort with ENHANCED QUALITY."""
        if debug_mode:
            print("🔄 Using original LibreOffice method (ENHANCED QUALITY)...")

        with tempfile.TemporaryDirectory() as temp_dir:
            # Export all slides as high-quality images with better settings
            cmd = [
                self.libreoffice_path,
                "--headless",
                "--convert-to", "png",
                "--outdir", temp_dir,
                # Try to improve quality with environment settings
                pptx_path
            ]

            if debug_mode:
                print(f"🚀 Enhanced original method: {' '.join(cmd)}")

            try:
                # Set environment variables for better quality
                env = os.environ.copy()
                env['SAL_USE_VCLPLUGIN'] = 'gen'  # Use generic VCL plugin for better rendering

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    env=env
                )

                if result.returncode != 0:
                    if debug_mode:
                        print(f"❌ Enhanced original method failed: {result.stderr}")
                    return {}

                # Find exported files
                exported_files = sorted([f for f in os.listdir(temp_dir) if f.endswith('.png')])

                if debug_mode:
                    print(f"📁 Enhanced original method exported {len(exported_files)} files")

                # Map slides to files
                results = {}
                for slide_num in slide_numbers:
                    if slide_num <= len(exported_files):
                        source_file = os.path.join(temp_dir, exported_files[slide_num - 1])
                        output_filename = f"{base_filename}_slide_{slide_num:02d}_diagram.png"
                        output_path = os.path.join(output_dir, output_filename)

                        shutil.copy2(source_file, output_path)
                        results[slide_num] = output_path

                        if debug_mode:
                            file_size = os.path.getsize(output_path) / 1024
                            print(f"✅ Enhanced Slide {slide_num}: {output_filename} ({file_size:.1f} KB)")
                    else:
                        if debug_mode:
                            print(f"❌ Slide {slide_num} not available (only {len(exported_files)} pages)")

                return results

            except subprocess.TimeoutExpired:
                if debug_mode:
                    print("❌ Enhanced original method timed out")
                return {}
            except Exception as e:
                if debug_mode:
                    print(f"❌ Error in enhanced original method: {str(e)}")
                return {}


def test_diagram_screenshot_capability() -> Tuple[bool, str]:
    """Test if diagram screenshot capability is available."""
    processor = DiagramScreenshotProcessor()

    if not processor.is_available():
        return False, "LibreOffice not found - required for diagram screenshots"

    # Test LibreOffice version
    try:
        result = subprocess.run(
            [processor.libreoffice_path, "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            version_info = result.stdout.strip()
            poppler_status = "with Poppler support" if processor.poppler_available else "without Poppler (fallback mode)"
            return True, f"LibreOffice available: {version_info} ({poppler_status})"
        else:
            return False, f"LibreOffice found but not working: {result.stderr}"

    except subprocess.TimeoutExpired:
        return False, "LibreOffice version check timed out"
    except Exception as e:
        return False, f"Error testing LibreOffice: {str(e)}"

def install_poppler_instructions():
    """Return platform-specific instructions for installing poppler-utils."""
    system = platform.system()

    if system == "Darwin":  # macOS
        return """Install poppler-utils on macOS:

Using Homebrew:
brew install poppler

Using MacPorts:
sudo port install poppler"""
    elif system == "Linux":
        return """Install poppler-utils on Linux:

Ubuntu/Debian:
sudo apt-get install poppler-utils

CentOS/RHEL:
sudo yum install poppler-utils

Fedora:
sudo dnf install poppler-utils"""
    elif system == "Windows":
        return """Install poppler-utils on Windows:

1. Download poppler for Windows from: 
   https://github.com/oschwartz10612/poppler-windows/releases
2. Extract to a folder (e.g., C:\\poppler)
3. Add C:\\poppler\\bin to your PATH environment variable

Or use chocolatey:
choco install poppler"""
    else:
        return "Install poppler-utils for your operating system to enable improved PDF processing."
