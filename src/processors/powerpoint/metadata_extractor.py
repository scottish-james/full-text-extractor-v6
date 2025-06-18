"""
Metadata Extractor - Handles PowerPoint metadata extraction and formatting
Extracts comprehensive document metadata and formats it for Claude enhancement

ARCHITECTURE OVERVIEW:
This component extracts metadata from PowerPoint files using python-pptx's
core properties interface. It provides comprehensive document information
that enhances content understanding and provides context for AI processing.

METADATA CATEGORIES:
1. Document properties: Title, author, subject, keywords, comments
2. Date properties: Created, modified, last printed timestamps
3. File properties: Size, filename, application information
4. Presentation properties: Slide count, layout types, masters
5. Application properties: Creating software, company, security settings

CLAUDE INTEGRATION:
Formats metadata as HTML comments that Claude can read for context but
don't interfere with the main content. This provides AI systems with
document context for better content enhancement.

ERROR HANDLING PHILOSOPHY:
- Graceful degradation: Missing metadata doesn't stop processing
- Defensive access: All property access wrapped in error handling
- Sensible defaults: Provide reasonable fallbacks for missing data
- Validation: Check metadata completeness and quality
"""

import os
from datetime import datetime


class MetadataExtractor:
    """
    Extracts and formats PowerPoint metadata for document enhancement.

    COMPONENT RESPONSIBILITIES:
    - Extract comprehensive PowerPoint document metadata
    - Format metadata for Claude AI processing
    - Validate metadata completeness and quality
    - Provide human-readable metadata summaries
    - Handle version differences in PowerPoint metadata access

    METADATA ACCESS PATTERNS:
    Different PowerPoint versions and file formats expose metadata
    differently. This component handles multiple access patterns:
    - Core properties: Standard OOXML document properties
    - App properties: Application-specific metadata
    - Custom properties: User-defined metadata fields

    OUTPUT FORMATS:
    - Raw metadata: Structured dictionary format
    - Claude comments: HTML comment format for AI processing
    - Summary: Human-readable overview
    - Validation: Quality assessment and recommendations
    """

    def extract_pptx_metadata(self, presentation, file_path):
        """
        Extract comprehensive metadata from PowerPoint file.

        EXTRACTION STRATEGY:
        1. Access core properties through python-pptx interface
        2. Extract different property categories systematically
        3. Handle missing or inaccessible properties gracefully
        4. Combine all metadata into structured dictionary
        5. Add computed fields (file size, etc.)

        PROPERTY CATEGORIES:
        Each category handled by specialized extraction method:
        - Document: Core document properties (title, author, etc.)
        - Date: Temporal information (created, modified, etc.)
        - Revision: Version control and tracking information
        - Presentation: PowerPoint-specific properties
        - Application: Software and creation environment

        ERROR RESILIENCE:
        Individual property extraction failures don't stop overall
        processing. Missing properties result in empty/None values
        rather than exceptions.

        Args:
            presentation: python-pptx Presentation object
            file_path (str): Path to the PowerPoint file for file-level metadata

        Returns:
            dict: Comprehensive metadata dictionary with all available properties
        """
        metadata = {}

        try:
            # Access PowerPoint's core properties object
            core_props = presentation.core_properties

            # Extract basic file information (not from PowerPoint properties)
            metadata['filename'] = os.path.basename(file_path)
            metadata['file_size'] = os.path.getsize(file_path) if os.path.exists(file_path) else None

            # Extract different categories of document properties
            metadata.update(self._extract_document_properties(core_props))
            metadata.update(self._extract_date_properties(core_props))
            metadata.update(self._extract_revision_properties(core_props))
            metadata.update(self._extract_presentation_properties(presentation))
            metadata.update(self._extract_application_properties(presentation))

        except Exception as e:
            print(f"Warning: Could not extract some metadata: {e}")

        return metadata

    def _extract_document_properties(self, core_props):
        """
        Extract basic document properties with defensive access.

        CORE DOCUMENT PROPERTIES:
        - title: Document title (may be different from filename)
        - author: Primary author/creator
        - subject: Document subject/topic
        - keywords: Search keywords and tags
        - comments: Document comments/description
        - category: Document category classification
        - content_status: Document status (draft, final, etc.)
        - language: Document language code
        - version: Document version information

        DEFENSIVE ACCESS PATTERN:
        Uses getattr() with empty string default to handle:
        - Missing attributes in different PowerPoint versions
        - None values from PowerPoint when properties unset
        - AttributeError exceptions from property access

        NORMALIZATION:
        All values normalized to strings with empty string for missing/None
        to ensure consistent data types for downstream processing.

        Args:
            core_props: PowerPoint core properties object

        Returns:
            dict: Document properties with consistent string values
        """
        return {
            'title': getattr(core_props, 'title', '') or '',
            'author': getattr(core_props, 'author', '') or '',
            'subject': getattr(core_props, 'subject', '') or '',
            'keywords': getattr(core_props, 'keywords', '') or '',
            'comments': getattr(core_props, 'comments', '') or '',
            'category': getattr(core_props, 'category', '') or '',
            'content_status': getattr(core_props, 'content_status', '') or '',
            'language': getattr(core_props, 'language', '') or '',
            'version': getattr(core_props, 'version', '') or '',
        }

    def _extract_date_properties(self, core_props):
        """
        Extract date-related properties with timezone awareness.

        DATE PROPERTIES:
        - created: Document creation timestamp
        - modified: Last modification timestamp
        - last_modified_by: User who last modified document
        - last_printed: Last print timestamp

        DATETIME HANDLING:
        PowerPoint stores dates as datetime objects or None.
        - Preserves original datetime objects for timezone info
        - None values preserved for missing dates
        - No date format conversion to maintain precision

        TIMEZONE CONSIDERATIONS:
        PowerPoint dates may include timezone information depending
        on creation environment. Preserving original objects allows
        downstream code to handle timezone conversion as needed.

        USER INFORMATION:
        last_modified_by provides audit trail information about
        document editing history and collaboration.

        Args:
            core_props: PowerPoint core properties object

        Returns:
            dict: Date properties with datetime objects or None values
        """
        return {
            'created': getattr(core_props, 'created', None),
            'modified': getattr(core_props, 'modified', None),
            'last_modified_by': getattr(core_props, 'last_modified_by', '') or '',
            'last_printed': getattr(core_props, 'last_printed', None),
        }

    def _extract_revision_properties(self, core_props):
        """
        Extract revision and identifier properties for version control.

        REVISION PROPERTIES:
        - revision: Document revision number (numeric or string)
        - identifier: Unique document identifier/GUID

        VERSION CONTROL USAGE:
        - revision: Tracks document version changes
        - identifier: Provides unique document identification
        - Useful for document management systems
        - Helps track document lineage and relationships

        DATA TYPE FLEXIBILITY:
        revision can be numeric or string depending on PowerPoint
        version and how versioning is implemented.

        Args:
            core_props: PowerPoint core properties object

        Returns:
            dict: Revision properties with original data types preserved
        """
        return {
            'revision': getattr(core_props, 'revision', None),
            'identifier': getattr(core_props, 'identifier', '') or '',
        }

    def _extract_presentation_properties(self, presentation):
        """
        Extract PowerPoint-specific presentation properties.

        PRESENTATION STRUCTURE:
        - slide_count: Total number of slides
        - slide_master_count: Number of slide master templates
        - layout_types: Available slide layout names

        SLIDE MASTER ANALYSIS:
        Slide masters define the overall design and layout options.
        - Multiple masters indicate complex design themes
        - Layout types show available content structures
        - Useful for understanding presentation sophistication

        LAYOUT ENUMERATION:
        Iterates through all slide masters and their layouts to
        build comprehensive list of available layout types.
        Deduplicates layout names across multiple masters.

        ERROR HANDLING:
        Slide master access can fail with corrupted presentations.
        Graceful fallback to basic slide count information.

        Args:
            presentation: python-pptx Presentation object

        Returns:
            dict: Presentation structure properties
        """
        metadata = {
            'slide_count': len(presentation.slides)
        }

        # Extract slide master and layout information with error handling
        try:
            slide_masters = presentation.slide_masters
            if slide_masters:
                metadata['slide_master_count'] = len(slide_masters)

                # Get comprehensive layout information
                layout_names = []
                for master in slide_masters:
                    for layout in master.slide_layouts:
                        if hasattr(layout, 'name') and layout.name:
                            layout_names.append(layout.name)

                # Deduplicate and format layout types
                metadata['layout_types'] = ', '.join(set(layout_names)) if layout_names else ''
            else:
                metadata['slide_master_count'] = 0
                metadata['layout_types'] = ''
        except Exception:
            # Slide master access failed - use defaults
            metadata['slide_master_count'] = 0
            metadata['layout_types'] = ''

        return metadata

    def _extract_application_properties(self, presentation):
        """
        Extract application-related properties about creation environment.

        APPLICATION PROPERTIES:
        - application: Software used to create presentation
        - app_version: Version of creating application
        - company: Company information from application settings
        - doc_security: Document security level/restrictions

        CREATION ENVIRONMENT:
        Provides insight into:
        - What software created the presentation
        - Version compatibility considerations
        - Corporate environment indicators
        - Security settings and restrictions

        ACCESS CHALLENGES:
        App properties are less standardized than core properties:
        - May not exist in all PowerPoint versions
        - Different property names across versions
        - Inconsistent data types and formats

        DEFENSIVE PROGRAMMING:
        Multiple layers of error handling due to inconsistent
        app property implementation across PowerPoint versions.

        Args:
            presentation: python-pptx Presentation object

        Returns:
            dict: Application environment properties with safe defaults
        """
        metadata = {
            'application': '',
            'app_version': '',
            'company': '',
            'doc_security': None
        }

        try:
            # Access application properties if available
            app_props = presentation.app_properties if hasattr(presentation, 'app_properties') else None
            if app_props:
                metadata['application'] = getattr(app_props, 'application', '') or ''
                metadata['app_version'] = getattr(app_props, 'app_version', '') or ''
                metadata['company'] = getattr(app_props, 'company', '') or ''
                metadata['doc_security'] = getattr(app_props, 'doc_security', None)
        except Exception:
            # App properties access failed - use defaults
            pass

        return metadata

    def add_pptx_metadata_for_claude(self, markdown_content, metadata):
        """
        Add PowerPoint metadata as HTML comments for Claude AI processing.

        CLAUDE INTEGRATION STRATEGY:
        Embeds metadata as HTML comments that:
        - Claude can read for document context
        - Don't interfere with human readability
        - Provide structured information for AI enhancement
        - Enable context-aware content processing

        COMMENT FORMAT:
        Uses <!-- POWERPOINT METADATA FOR CLAUDE: ... --> structure
        that clearly identifies the content purpose while remaining
        valid HTML/markdown.

        METADATA ORGANIZATION:
        Groups related metadata for easy AI parsing:
        - Document information (title, author, etc.)
        - Date information (created, modified, etc.)
        - File information (size, application, etc.)
        - Presentation information (slides, layouts, etc.)

        PLACEMENT STRATEGY:
        Metadata placed at beginning of document to provide context
        before Claude processes the main content.

        Args:
            markdown_content (str): Original markdown content
            metadata (dict): Extracted metadata dictionary

        Returns:
            str: Markdown content with embedded metadata comments
        """
        # Build comprehensive metadata comment for Claude
        metadata_comments = "\n<!-- POWERPOINT METADATA FOR CLAUDE:\n"

        # Add organized metadata sections
        metadata_comments += self._format_document_metadata(metadata)
        metadata_comments += self._format_date_metadata(metadata)
        metadata_comments += self._format_file_metadata(metadata)
        metadata_comments += self._format_presentation_metadata(metadata)

        metadata_comments += "-->\n"

        # Prepend metadata to content for Claude context
        return metadata_comments + markdown_content

    def _format_document_metadata(self, metadata):
        """
        Format document-related metadata for Claude with conditional inclusion.

        FORMATTING STRATEGY:
        Only includes non-empty metadata fields to avoid cluttering
        Claude's context with empty/missing information.

        FIELD PRIORITIZATION:
        Orders fields by importance for document understanding:
        1. Title: Most important for content context
        2. Author: Creator information
        3. Subject: Topic/theme information
        4. Keywords: Search/classification terms
        5. Category, Comments: Additional context
        6. Status, Language, Version: Technical details

        CONDITIONAL FORMATTING:
        Uses if statements to only include fields with actual content,
        making the metadata more concise and relevant.

        Args:
            metadata (dict): Metadata dictionary

        Returns:
            str: Formatted document metadata section
        """
        formatted = ""

        if metadata.get('title'):
            formatted += f"Document Title: {metadata['title']}\n"
        if metadata.get('author'):
            formatted += f"Author: {metadata['author']}\n"
        if metadata.get('subject'):
            formatted += f"Subject: {metadata['subject']}\n"
        if metadata.get('keywords'):
            formatted += f"Keywords: {metadata['keywords']}\n"
        if metadata.get('category'):
            formatted += f"Category: {metadata['category']}\n"
        if metadata.get('comments'):
            formatted += f"Document Comments: {metadata['comments']}\n"
        if metadata.get('content_status'):
            formatted += f"Content Status: {metadata['content_status']}\n"
        if metadata.get('language'):
            formatted += f"Language: {metadata['language']}\n"
        if metadata.get('version'):
            formatted += f"Version: {metadata['version']}\n"

        return formatted

    def _format_date_metadata(self, metadata):
        """
        Format date-related metadata with human-readable timestamps.

        DATE FORMATTING:
        Preserves original datetime objects from PowerPoint which
        may include timezone information. Let Claude handle date
        interpretation based on its capabilities.

        AUDIT TRAIL:
        Date information provides important context about:
        - Document age and recency
        - Editing history and collaboration
        - Print/distribution history

        CONDITIONAL INCLUSION:
        Only includes dates that are actually set to avoid
        "None" or empty timestamp clutter.

        Args:
            metadata (dict): Metadata dictionary

        Returns:
            str: Formatted date metadata section
        """
        formatted = ""

        if metadata.get('created'):
            formatted += f"Created Date: {metadata['created']}\n"
        if metadata.get('modified'):
            formatted += f"Last Modified: {metadata['modified']}\n"
        if metadata.get('last_modified_by'):
            formatted += f"Last Modified By: {metadata['last_modified_by']}\n"
        if metadata.get('last_printed'):
            formatted += f"Last Printed: {metadata['last_printed']}\n"

        return formatted

    def _format_file_metadata(self, metadata):
        """
        Format file-related metadata including computed information.

        FILE INFORMATION:
        - filename: Original file name for reference
        - file_size: Computed file size in human-readable format
        - application: Creating application information
        - company: Corporate context if available

        SIZE CALCULATION:
        Converts bytes to MB for human readability with 2 decimal
        precision. File size provides context about presentation
        complexity and content richness.

        CREATION CONTEXT:
        Application and company information helps understand:
        - Compatibility considerations
        - Corporate vs personal creation
        - Software capability constraints

        Args:
            metadata (dict): Metadata dictionary

        Returns:
            str: Formatted file metadata section
        """
        formatted = ""

        # Always include filename for reference
        formatted += f"Filename: {metadata.get('filename', 'unknown')}\n"

        # Include file size with human-readable format
        if metadata.get('file_size'):
            file_size_mb = metadata['file_size'] / (1024 * 1024)
            formatted += f"File Size: {file_size_mb:.2f} MB\n"

        # Include application information if available
        if metadata.get('application'):
            formatted += f"Created With: {metadata['application']}\n"
        if metadata.get('company'):
            formatted += f"Company: {metadata['company']}\n"

        return formatted

    def _format_presentation_metadata(self, metadata):
        """
        Format PowerPoint-specific metadata about presentation structure.

        PRESENTATION STRUCTURE:
        - slide_count: Essential for understanding scope
        - slide_masters: Indicates design complexity
        - layout_types: Shows content structure variety

        COMPLEXITY INDICATORS:
        - Multiple masters suggest sophisticated design
        - Variety of layouts indicates diverse content types
        - Large slide counts suggest comprehensive presentations

        CLAUDE CONTEXT:
        This information helps Claude understand:
        - Presentation scope and complexity
        - Content organization patterns
        - Design sophistication level

        Args:
            metadata (dict): Metadata dictionary

        Returns:
            str: Formatted presentation metadata section
        """
        formatted = ""

        # Always include slide count as fundamental metric
        formatted += f"Slide Count: {metadata.get('slide_count', 0)}\n"

        # Include design complexity information if available
        if metadata.get('slide_master_count'):
            formatted += f"Slide Masters: {metadata['slide_master_count']}\n"
        if metadata.get('layout_types'):
            formatted += f"Layout Types: {metadata['layout_types']}\n"

        return formatted

    def get_metadata_summary(self, metadata):
        """
        Generate human-readable summary of metadata for quick assessment.

        SUMMARY PURPOSE:
        Provides quick overview of metadata completeness and key
        information without full detail dump.

        KEY METRICS:
        - Presence indicators for important fields
        - Quantitative measures (slide count, file size)
        - Temporal information (dates)
        - Technical details (application)

        COMPUTED FIELDS:
        - file_size_mb: Human-readable file size
        - has_* flags: Boolean indicators for presence checks

        USAGE SCENARIOS:
        - Quick metadata assessment
        - Logging and monitoring
        - Quality checks
        - User interfaces

        Args:
            metadata (dict): Metadata dictionary

        Returns:
            dict: Summary information with key indicators and metrics
        """
        summary = {
            'has_title': bool(metadata.get('title')),
            'has_author': bool(metadata.get('author')),
            'slide_count': metadata.get('slide_count', 0),
            'file_size_mb': None,
            'creation_date': metadata.get('created'),
            'last_modified': metadata.get('modified'),
            'has_keywords': bool(metadata.get('keywords')),
            'application': metadata.get('application', 'Unknown'),
        }

        # Compute human-readable file size
        if metadata.get('file_size'):
            summary['file_size_mb'] = round(metadata['file_size'] / (1024 * 1024), 2)

        return summary

    def validate_metadata(self, metadata):
        """
        Validate metadata completeness and provide quality assessment.

        VALIDATION PURPOSE:
        - Assess metadata completeness for quality scoring
        - Identify missing important information
        - Provide actionable recommendations for improvement

        COMPLETENESS SCORING:
        Calculates percentage based on presence of essential fields:
        - title: Critical for document identification
        - author: Important for provenance
        - slide_count: Basic structural information

        ISSUE DETECTION:
        Identifies common metadata problems:
        - Missing titles (poor document management)
        - Missing authors (audit trail gaps)
        - No slides (file corruption or processing errors)

        RECOMMENDATIONS:
        Provides specific, actionable advice for improving
        metadata quality and document management practices.

        Args:
            metadata (dict): Metadata dictionary

        Returns:
            dict: Validation results with score, issues, and recommendations
        """
        validation = {
            'completeness_score': 0,
            'issues': [],
            'recommendations': []
        }

        # Check for essential fields and calculate completeness score
        essential_fields = ['title', 'author', 'slide_count']
        present_fields = sum(1 for field in essential_fields if metadata.get(field))
        validation['completeness_score'] = (present_fields / len(essential_fields)) * 100

        # Identify specific issues and provide recommendations
        if not metadata.get('title'):
            validation['issues'].append("No document title")
            validation['recommendations'].append("Add a descriptive title to the presentation")

        if not metadata.get('author'):
            validation['issues'].append("No author information")
            validation['recommendations'].append("Set author information in document properties")

        if metadata.get('slide_count', 0) == 0:
            validation['issues'].append("No slides detected")

        if not metadata.get('keywords'):
            validation['recommendations'].append("Add keywords to improve searchability")

        return validation

