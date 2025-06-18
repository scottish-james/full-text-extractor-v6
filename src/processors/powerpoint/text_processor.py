"""
Text Processor - Handles advanced text formatting with XML-driven detection
Simplified to rely on XML data instead of text pattern matching
Fixed: Bold detection for consistently formatted text

ARCHITECTURE OVERVIEW:
This module processes python-pptx TextFrame objects to extract structured text data
with formatting metadata. The key insight is using PowerPoint's internal XML
representations rather than text pattern matching for reliable bullet/numbering detection.

KEY DEPENDENCIES:
- python-pptx library for PowerPoint parsing
- Access to internal _p.xml attributes for XML inspection
- re module for text cleaning operations

DESIGN DECISIONS:
- XML-first approach: Trust PowerPoint's internal XML over visual text patterns
- Defensive programming: All XML access wrapped in try/catch blocks
- Structured output: Consistent dict format for downstream processing
- Run-level formatting: Preserve individual text run formatting within paragraphs
"""

import re


class TextProcessor:
    """
    Main class for extracting structured text data from PowerPoint shapes.

    PROCESSING PIPELINE:
    1. extract_text_frame() - Main entry point, processes all paragraphs
    2. process_paragraph() - Individual paragraph processing with XML analysis
    3. _extract_runs_with_formatting() - Run-level formatting extraction
    4. Various helper methods for URL fixing, text cleaning, etc.

    OUTPUT FORMAT:
    Returns nested dict structure:
    {
        "type": "text",
        "paragraphs": [
            {
                "raw_text": str,        # Original text from PowerPoint
                "clean_text": str,      # Processed text (bullets removed, etc.)
                "formatted_runs": [     # Individual text runs with formatting
                    {
                        "text": str,
                        "bold": bool,
                        "italic": bool,
                        "hyperlink": str|None
                    }
                ],
                "hints": {              # Metadata for downstream processing
                    "bullet_level": int,
                    "is_bullet": bool,
                    "is_numbered": bool,
                    # ... additional flags
                }
            }
        ],
        "shape_hyperlink": str|None    # Shape-level hyperlink if present
    }
    """

    def extract_text_frame(self, text_frame, shape):
        """
        Primary extraction method for TextFrame objects.

        IMPLEMENTATION NOTES:
        - Skips empty paragraphs to avoid noise in output
        - Returns None if no meaningful content found
        - Extracts shape-level hyperlinks separately from text-level ones

        Args:
            text_frame: python-pptx TextFrame object
            shape: parent Shape object (needed for shape-level hyperlinks)

        Returns:
            dict|None: Structured text data or None if empty
        """
        if not text_frame.paragraphs:
            return None

        block = {
            "type": "text",
            "paragraphs": [],
            "shape_hyperlink": self._extract_shape_hyperlink(shape)
        }

        # Process each paragraph, filtering out empty ones
        for para_idx, para in enumerate(text_frame.paragraphs):
            if not para.text.strip():
                continue

            para_data = self.process_paragraph(para)
            if para_data:
                block["paragraphs"].append(para_data)

        return block if block["paragraphs"] else None

    def extract_plain_text(self, shape):
        """
        Fallback method for shapes without full TextFrame structure.

        USE CASE: Some PowerPoint shapes only expose .text attribute without
        full paragraph/run structure. This provides basic extraction.

        LIMITATIONS:
        - No run-level formatting detection
        - No bullet/numbering analysis
        - Basic hint analysis only

        Args:
            shape: python-pptx Shape object with .text attribute

        Returns:
            dict|None: Basic text structure or None if no text
        """
        if not hasattr(shape, 'text') or not shape.text:
            return None

        return {
            "type": "text",
            "paragraphs": [{
                "raw_text": shape.text,
                "clean_text": shape.text.strip(),
                "formatted_runs": [{"text": shape.text, "bold": False, "italic": False, "hyperlink": None}],
                "hints": self._analyze_plain_text_hints(shape.text)
            }],
            "shape_hyperlink": self._extract_shape_hyperlink(shape)
        }

    def process_paragraph(self, para):
        """
        Core paragraph processing with XML-based formatting detection.

        ALGORITHM:
        1. Extract PowerPoint level (.level attribute)
        2. Analyse internal XML for bullet formatting indicators
        3. Determine final bullet level using hierarchy of evidence
        4. Clean text based on bullet detection results
        5. Extract individual runs with formatting
        6. Generate metadata hints for downstream processing

        XML INDICATORS DETECTED:
        - buChar: Custom bullet character
        - buAutoNum: Automatic numbering
        - buFont: Bullet font specification
        - lvl="n": Explicit level specification

        Args:
            para: python-pptx Paragraph object

        Returns:
            dict|None: Processed paragraph data or None if empty
        """
        raw_text = para.text
        if not raw_text.strip():
            return None

        # Extract formatting indicators from multiple sources
        ppt_level = getattr(para, 'level', None)
        is_ppt_bullet, xml_level = self._check_xml_bullet_formatting(para)

        # Resolve bullet level from available evidence
        bullet_level = self._determine_bullet_level(is_ppt_bullet, xml_level, ppt_level)

        # Clean text based on bullet detection
        clean_text = raw_text.strip()
        if bullet_level >= 0:
            clean_text = self._remove_bullet_char(clean_text)

        # Extract formatted runs (preserves individual formatting within paragraph)
        formatted_runs = self._extract_runs_with_formatting(para.runs, clean_text, bullet_level >= 0)

        # Build comprehensive metadata for downstream systems
        para_data = {
            "raw_text": raw_text,
            "clean_text": clean_text,
            "formatted_runs": formatted_runs,
            "hints": {
                "has_powerpoint_level": ppt_level is not None,
                "powerpoint_level": ppt_level,
                "bullet_level": bullet_level,
                "is_bullet": bullet_level >= 0,
                "is_numbered": self._is_numbered_from_xml(para),
                "short_text": len(clean_text) < 100,
                "all_caps": clean_text.isupper() if clean_text else False,
            }
        }

        return para_data

    def _check_xml_bullet_formatting(self, para):
        """
        XML analysis for bullet formatting detection.

        RELIABILITY: XML indicators are more reliable than text pattern matching
        because they represent PowerPoint's internal formatting decisions.

        ERROR HANDLING: All XML access wrapped in try/catch to handle:
        - Missing _p attribute
        - XML parsing failures
        - Malformed level attributes

        REGEX PATTERN: r'lvl="(\d+)"' matches level specifications in XML

        Args:
            para: python-pptx Paragraph object

        Returns:
            tuple: (is_bullet_according_to_xml, xml_level_if_specified)
        """
        is_ppt_bullet = False
        xml_level = None

        try:
            if hasattr(para, '_p') and para._p is not None:
                xml_str = str(para._p.xml)

                # Check for any bullet formatting indicators
                if any(indicator in xml_str for indicator in ['buChar', 'buAutoNum', 'buFont']):
                    is_ppt_bullet = True

                    # Extract explicit level if specified
                    level_match = re.search(r'lvl="(\d+)"', xml_str)
                    if level_match:
                        xml_level = int(level_match.group(1))
        except:
            # Defensive: XML access can fail in various ways
            pass

        return is_ppt_bullet, xml_level

    def _is_numbered_from_xml(self, para):
        """
        Detects numbered lists vs bullet points from XML.

        DISTINCTION: 'buAutoNum' indicates automatic numbering (1, 2, 3...)
        vs 'buChar' which indicates bullet characters (•, ◦, etc.)

        Args:
            para: python-pptx Paragraph object

        Returns:
            bool: True if numbered list, False otherwise
        """
        try:
            if hasattr(para, '_p') and para._p is not None:
                xml_str = str(para._p.xml)
                return 'buAutoNum' in xml_str
        except:
            return False

    def _determine_bullet_level(self, is_ppt_bullet, xml_level, ppt_level):
        """
        Bullet level resolution using hierarchy of evidence.

        PRECEDENCE ORDER:
        1. XML bullet indicator + XML level
        2. XML bullet indicator + PowerPoint level
        3. XML bullet indicator + default level 0
        4. PowerPoint level (even without XML bullet indicator)
        5. No bullet (-1)

        RATIONALE: XML indicators are most reliable, but level information
        can come from multiple sources with varying reliability.

        Args:
            is_ppt_bullet (bool): XML indicates bullet formatting
            xml_level (int|None): Level from XML if specified
            ppt_level (int|None): Level from paragraph.level attribute

        Returns:
            int: Final bullet level (-1 indicates non-bullet)
        """
        bullet_level = -1

        if is_ppt_bullet:
            # XML says it's a bullet, determine level from available sources
            bullet_level = xml_level if xml_level is not None else (ppt_level if ppt_level is not None else 0)
        elif ppt_level is not None:
            # No XML bullet indicator, but PowerPoint has level info
            bullet_level = ppt_level

        return bullet_level

    def _extract_runs_with_formatting(self, runs, clean_text, has_prefix_removed):
        """
        Extract individual text runs while preserving formatting.

        COMPLEXITY: When bullet characters are removed from clean_text,
        we need to map the cleaned text back to the original runs to
        preserve individual formatting of words/phrases.

        ALGORITHM:
        - If prefix removed: Calculate offset and trim runs accordingly
        - If no prefix: Process runs directly
        - Each run maintains its individual bold/italic/hyperlink state

        EDGE CASES:
        - Empty runs are skipped
        - Runs that fall entirely within removed prefix are discarded
        - Runs spanning prefix boundary are trimmed

        Args:
            runs: List of python-pptx Run objects
            clean_text (str): Text with prefixes removed
            has_prefix_removed (bool): Whether bullet chars were removed

        Returns:
            list: Formatted run data preserving individual styling
        """
        if not runs:
            # Fallback for missing runs
            return [{"text": clean_text, "bold": False, "italic": False, "hyperlink": None}]

        formatted_runs = []

        if has_prefix_removed:
            # Complex case: map clean text back to original runs
            full_text = "".join(run.text for run in runs)
            start_pos = self._find_clean_text_start_position(full_text, clean_text)

            char_count = 0
            for run in runs:
                run_text = run.text
                run_start = char_count
                run_end = char_count + len(run_text)

                # Skip runs entirely within the removed prefix
                if run_end <= start_pos:
                    char_count += len(run_text)
                    continue

                # Trim runs that span the prefix boundary
                if run_start < start_pos < run_end:
                    run_text = run_text[start_pos - run_start:]

                if run_text:
                    formatted_runs.append(self._extract_run_formatting(run, run_text))

                char_count += len(run.text)
        else:
            # Simple case: process runs directly
            for run in runs:
                if run.text:
                    formatted_runs.append(self._extract_run_formatting(run, run.text))

        return formatted_runs

    def _find_clean_text_start_position(self, full_text, clean_text):
        """
        Locate clean text position within original text.

        ALGORITHM: Linear search for clean text match after stripping
        whitespace. Handles cases where bullet removal affects spacing.

        FALLBACK: Returns 0 if exact match not found (graceful degradation).

        Args:
            full_text (str): Original text with bullet chars
            clean_text (str): Cleaned text

        Returns:
            int: Character position where clean text begins
        """
        for i in range(len(full_text)):
            remaining = full_text[i:].strip()
            if remaining == clean_text:
                return i
        return 0  # Graceful fallback

    def _extract_run_formatting(self, run, text_override=None):
        """
        Extract formatting attributes from individual text run.

        FORMATTING ATTRIBUTES:
        - Bold: run.font.bold (boolean or None)
        - Italic: run.font.italic (boolean or None)
        - Hyperlink: run.hyperlink.address (string or None)

        ERROR HANDLING: Font attributes can be None, missing, or raise
        exceptions depending on PowerPoint version and file format.

        URL PROCESSING: Hyperlink addresses are passed through _fix_url()
        to handle common formatting issues.

        Args:
            run: python-pptx Run object
            text_override (str|None): Use this text instead of run.text

        Returns:
            dict: Run data with formatting flags
        """
        run_data = {
            "text": text_override if text_override is not None else run.text,
            "bold": False,
            "italic": False,
            "hyperlink": None
        }

        # Extract font formatting with defensive checks
        try:
            font = run.font
            if hasattr(font, 'bold') and font.bold:
                run_data["bold"] = True
            if hasattr(font, 'italic') and font.italic:
                run_data["italic"] = True
        except:
            # Font access can fail in various ways - continue with defaults
            pass

        # Extract hyperlink with URL cleaning
        try:
            if hasattr(run, 'hyperlink') and run.hyperlink and run.hyperlink.address:
                run_data["hyperlink"] = self._fix_url(run.hyperlink.address)
        except:
            # Hyperlink extraction can be unreliable
            pass

        return run_data

    def _remove_bullet_char(self, text):
        """
        Remove bullet characters from text start.

        REGEX PATTERN: Matches common bullet chars followed by optional whitespace
        CHARACTER SET: Covers Unicode bullet variations commonly used by PowerPoint

        MAINTENANCE NOTE: Add new bullet characters to regex as needed.
        Current set covers: • ◦ ▪ ▫ ‣ · ○ ■ □ → ► ✓ ✗ - * + ※ ◆ ◇

        Args:
            text (str): Text potentially starting with bullet chars

        Returns:
            str: Text with leading bullet chars removed
        """
        if not text:
            return text
        return re.sub(r'^[•◦▪▫‣·○■□→►✓✗\-\*\+※◆◇]\s*', '', text)

    def _analyze_plain_text_hints(self, text):
        """
        Basic text analysis for fallback cases.

        PURPOSE: Provides minimal metadata when full XML analysis unavailable.
        Used by extract_plain_text() for shapes without TextFrame structure.

        LIMITATIONS: Cannot detect bullets/numbering reliably from text alone,
        so most formatting flags are set to safe defaults.

        Args:
            text (str): Plain text to analyze

        Returns:
            dict: Basic text characteristics
        """
        if not text:
            return {}

        stripped = text.strip()

        return {
            "has_powerpoint_level": False,
            "powerpoint_level": None,
            "bullet_level": -1,
            "is_bullet": False,
            "is_numbered": False,
            "starts_with_bullet": False,
            "starts_with_number": False,
            "short_text": len(stripped) < 100,
            "all_caps": stripped.isupper() if stripped else False,
            "likely_heading": len(stripped) < 80 and len(stripped) > 0
        }

    def _extract_shape_hyperlink(self, shape):
        """
        Extract shape-level hyperlinks (entire shape is clickable).

        OBJECT HIERARCHY: shape.click_action.hyperlink.address
        Each level can be None, requiring defensive navigation.

        USE CASE: PowerPoint allows making entire text boxes/shapes clickable,
        separate from text-level hyperlinks within the content.

        Args:
            shape: python-pptx Shape object

        Returns:
            str|None: Cleaned URL or None if no shape hyperlink
        """
        try:
            if hasattr(shape, 'click_action') and shape.click_action:
                if hasattr(shape.click_action, 'hyperlink') and shape.click_action.hyperlink:
                    if shape.click_action.hyperlink.address:
                        return self._fix_url(shape.click_action.hyperlink.address)
        except:
            # Shape hyperlink access can fail
            pass
        return None

    def _fix_url(self, url):
        """
        Normalize and fix common URL formatting issues.

        COMMON ISSUES FIXED:
        - Missing mailto: prefix for email addresses
        - Missing http/https scheme for web URLs
        - Incomplete www. URLs

        SCHEME DETECTION: Uses common TLD patterns to identify web URLs
        FALLBACK: Defaults to HTTPS for security

        PRESERVATION: Returns original URL if already properly formatted
        or if format is unrecognized.

        Args:
            url (str): Potentially malformed URL

        Returns:
            str: Normalized URL with appropriate scheme
        """
        if not url:
            return url

        # Fix email addresses missing mailto: scheme
        if '@' in url and not url.startswith('mailto:'):
            return f"mailto:{url}"

        # Fix web URLs missing scheme
        if not url.startswith(('http://', 'https://', 'mailto:', 'tel:', 'ftp://', '#')):
            # Detect web URLs by common patterns
            if url.startswith('www.') or any(
                    domain in url.lower() for domain in ['.com', '.org', '.net', '.edu', '.gov', '.io']):
                return f"https://{url}"  # Default to HTTPS

        return url