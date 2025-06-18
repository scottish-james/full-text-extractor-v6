"""
Updated Claude Sonnet 4 Markdown Enhancement Module - Simplified for PowerPoint processing
"""

import os
import anthropic
from typing import Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simplified system prompt for PowerPoint processing
PPTX_PROCESSING_SYSTEM_PROMPT = """
You are a PowerPoint to Markdown converter. You receive roughly converted markdown from PowerPoint slides and must clean it up into professional, well-structured markdown for AI training and vector database storage.

Your job:
1. Fix bullet point hierarchies - create proper nested lists with 2-space indentation
2. Identify slide titles and format as # headers
3. Identify sub heading and format as ### headers 
4. Preserve ALL hyperlinks and formatting (bold, italic)
5. Fix broken list structures
6. Ensure tables are properly formatted
7. Clean up spacing and structure
8. USE POWERPOINT METADATA when available (look for HTML comments with "POWERPOINT METADATA FOR CLAUDE")
9. Split all slides with ---
10. ADD COMPREHENSIVE METADATA at the end for vector database optimization

Key Rules:
- Keep ALL original text content - this is critical as this is for regulatory documents
- Fix bullet nesting based on context and content
- Make short, standalone text into appropriate ### headers 
- Preserve all hyperlinks exactly as provided
- Use proper markdown syntax throughout
- Look for numbered sequences that are out of order and fix them
- INCORPORATE PowerPoint metadata (author, creation date, etc.) into the final metadata section

CRITICAL: Always end with a metadata section that includes:
- TLDR/Executive Summary (2-3 sentences)
- Key Topics/Themes (for embeddings)
- Target Audience (inferred)
- Key Concepts/Terms
- File metadata (author, creation date, version, etc. from PowerPoint properties)
- Diagram Types (list any Mermaid diagrams created)

The input will have slide markers like `<!-- Slide 1 -->` and may include PowerPoint metadata in HTML comments and diagram candidates.

Format the metadata section like this at the very end:

---
## DOCUMENT METADATA (for AI/Vector DB)

**TLDR:** [2-3 sentence summary of the entire presentation]

**Key Topics:** [comma-separated list of main topics/themes]

**Content Type:** [e.g., Educational, Business Presentation, Training Material, etc.]

**Related Topics:** [concepts that would be complementary to search for]

**Visual Elements:** [number and types of diagrams, charts, images converted to Mermaid]

**File Properties:**
- **Author:** [from PowerPoint metadata]
- **Created Date:** [from PowerPoint metadata]
- **Last Modified:** [from PowerPoint metadata]
- **Version:** [from PowerPoint metadata]
- **Company/Organization:** [from PowerPoint metadata]
- **Document Title:** [from PowerPoint metadata]
- **Keywords:** [from PowerPoint metadata]
- **Category:** [from PowerPoint metadata]
- **Slide Dimensions:** [from PowerPoint metadata]
---

Output clean, readable markdown that maintains the original document's intent but fixes obvious ordering issues, converts diagrams to Mermaid, and adds rich metadata for AI training.
"""

# Keep the original system prompt for other document types
DOCUMENT_TO_MARKDOWN_SYSTEM_PROMPT = """
You are a markdown formatting expert. Your task is to:
1. Fix any syntax errors in the markdown
2. Improve the structure and hierarchy of headers
3. Ensure consistent formatting throughout
4. Enhance bullet points and numbered lists
5. Properly format tables and code blocks
6. Add appropriate spacing between sections
7. Maintain the original content without adding new information
8. Preserve all links and references

Return ONLY the enhanced markdown content without any explanations or additional text.
"""


class ClaudeMarkdownEnhancer:
    """
    A class to enhance markdown documents using Claude Sonnet 4.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Claude Markdown Enhancer.

        Args:
            api_key (Optional[str]): Anthropic API key. If not provided, will try to get from environment.
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key is required. Set ANTHROPIC_API_KEY environment variable or pass it directly.")

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = "claude-sonnet-4-20250514"

    def enhance_markdown(self, markdown_content: str, source_filename: str = "unknown",
                         content_type: str = "Document") -> Tuple[str, Optional[str]]:
        """
        Enhance markdown content using Claude Sonnet 4.

        Args:
            markdown_content (str): The original markdown content to enhance
            source_filename (str): The source filename for context
            content_type (str): The type of document (PowerPoint, Word, PDF, etc.)

        Returns:
            Tuple[str, Optional[str]]: Enhanced markdown content and error message (if any)
        """
        try:
            # Choose system prompt based on content type
            if "powerpoint" in content_type.lower() or "pptx" in source_filename.lower():
                system_prompt = PPTX_PROCESSING_SYSTEM_PROMPT
                user_prompt = f"""
                Please clean up this PowerPoint markdown conversion for AI training and vector database storage:

                **Source:** {source_filename}

                **Content to clean up:**
                {markdown_content}

IMPORTANT: 
1. Fix the structure, bullet hierarchies, and formatting while preserving all content and hyperlinks
2. If you see content with ordering words like "first", "second", "third", "fourth", "fifth" etc., REORDER those items into the correct sequence
3. ADD COMPREHENSIVE METADATA at the end following the specified format for vector database optimization

The metadata section is CRITICAL for AI training - make sure to analyze the content thoroughly and provide rich, searchable metadata that will help with embeddings and retrieval."""


            else:
                system_prompt = DOCUMENT_TO_MARKDOWN_SYSTEM_PROMPT
                user_prompt = f"""Please enhance this markdown document:

**Source Information:**
- Filename: {source_filename}
- Content Type: {content_type}

**Original Markdown Content:**
{markdown_content}

Please apply the formatting standards and ensure all content is preserved while improving the structure, consistency, and readability."""

            # Make the API call
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                temperature=0.1,  # Low temperature for consistent formatting
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ]
            )

            enhanced_content = response.content[0].text
            logger.info(f"Successfully enhanced markdown for {source_filename}")
            return enhanced_content, None

        except anthropic.APIError as e:
            error_msg = f"Anthropic API error: {str(e)}"
            logger.error(error_msg)
            return markdown_content, error_msg

        except Exception as e:
            error_msg = f"Unexpected error enhancing markdown: {str(e)}"
            logger.error(error_msg)
            return markdown_content, error_msg

    def enhance_multiple_documents(self, documents: list) -> list:
        """
        Enhance multiple markdown documents.

        Args:
            documents (list): List of dictionaries with keys 'content', 'filename', 'content_type'

        Returns:
            list: List of results with enhanced content and any errors
        """
        results = []

        for doc in documents:
            content = doc.get('content', '')
            filename = doc.get('filename', 'unknown')
            content_type = doc.get('content_type', 'Document')

            enhanced_content, error = self.enhance_markdown(content, filename, content_type)

            results.append({
                'filename': filename,
                'original_content': content,
                'enhanced_content': enhanced_content,
                'error': error,
                'success': error is None
            })

        return results


def enhance_markdown_with_claude(markdown_content: str, api_key: str,
                                 source_filename: str = "unknown",
                                 content_type: str = "Document") -> Tuple[str, Optional[str]]:
    """
    Convenience function to enhance markdown content using Claude Sonnet 4.

    Args:
        markdown_content (str): The markdown content to enhance
        api_key (str): Anthropic API key
        source_filename (str): Source filename for context
        content_type (str): Document type

    Returns:
        Tuple[str, Optional[str]]: Enhanced content and error message (if any)
    """
    try:
        enhancer = ClaudeMarkdownEnhancer(api_key)
        return enhancer.enhance_markdown(markdown_content, source_filename, content_type)
    except Exception as e:
        return markdown_content, str(e)