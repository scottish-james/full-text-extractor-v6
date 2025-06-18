"""
Debug Markdown Converter - Enhanced with debugging to track duplication
"""

import re


class MarkdownConverter:
    """
    Converts structured presentation data to clean markdown format using XML semantic roles.
    ENHANCED: Added debugging to track where duplication might occur.
    """

    def convert_structured_data_to_markdown(self, data, convert_slide_titles=True):
        """
        Main conversion entry point - processes entire presentation.
        ENHANCED: Added debugging to track slide processing and detect duplication.
        """
        print(f"\nDEBUG: MarkdownConverter starting with {len(data['slides'])} slides")

        markdown_parts = []

        for slide_idx, slide in enumerate(data["slides"]):
            print(
                f"\nDEBUG: Converting slide {slide['slide_number']} with {len(slide['content_blocks'])} content blocks")

            # Add slide marker for debugging
            slide_marker = f"\n<!-- Slide {slide['slide_number']} -->\n"
            markdown_parts.append(slide_marker)
            print(f"DEBUG: Added slide marker")

            # Process each content block with semantic role awareness
            block_count = 0
            for block_idx, block in enumerate(slide["content_blocks"]):
                print(
                    f"DEBUG: Processing block {block_idx + 1}: type={block['type']}, semantic_role={block.get('semantic_role', 'unknown')}")

                block_markdown = None
                if block["type"] == "text":
                    # Check semantic role for direct title conversion
                    block_markdown = self._convert_text_block_to_markdown(block)
                elif block["type"] == "table":
                    block_markdown = self._convert_table_to_markdown(block)
                elif block["type"] == "image":
                    block_markdown = self._convert_image_to_markdown(block)
                elif block["type"] == "chart":
                    block_markdown = self._convert_chart_to_markdown(block)
                elif block["type"] == "group":
                    block_markdown = self._convert_group_to_markdown(block)

                if block_markdown:
                    markdown_parts.append(block_markdown)
                    block_count += 1
                    print(f"DEBUG: Added markdown for block {block_idx + 1}")
                    # Show preview of what was added
                    preview = block_markdown.replace('\n', ' ')[:100] + "..." if len(
                        block_markdown) > 100 else block_markdown.replace('\n', ' ')
                    print(f"DEBUG: Content preview: {preview}")
                else:
                    print(f"DEBUG: Block {block_idx + 1} produced no markdown")

            print(f"DEBUG: Slide {slide['slide_number']} produced {block_count} markdown blocks")

        # Combine all parts with proper spacing
        markdown_content = "\n\n".join(filter(None, markdown_parts))

        print(f"\nDEBUG: Final markdown has {len(markdown_parts)} parts")
        print(f"DEBUG: Final markdown length: {len(markdown_content)} characters")

        return markdown_content

    def _convert_text_block_to_markdown(self, block):
        """
        Convert text content blocks to markdown using semantic role information.
        ENHANCED: Added debugging to track semantic role processing.
        """
        lines = []
        semantic_role = block.get("semantic_role", "other")

        print(
            f"DEBUG: Converting text block with semantic_role='{semantic_role}', {len(block.get('paragraphs', []))} paragraphs")

        # Handle semantic roles directly without pattern matching
        if semantic_role == "title":
            # Titles become H1 headings directly
            for para_idx, para in enumerate(block["paragraphs"]):
                if para.get("clean_text"):
                    formatted_text = self._build_formatted_text_from_runs(
                        para["formatted_runs"], para["clean_text"]
                    )
                    title_line = f"# {formatted_text}"
                    lines.append(title_line)
                    print(f"DEBUG: Added title: {title_line}")
        elif semantic_role == "subtitle":
            # Subtitles become H2 headings directly
            for para_idx, para in enumerate(block["paragraphs"]):
                if para.get("clean_text"):
                    formatted_text = self._build_formatted_text_from_runs(
                        para["formatted_runs"], para["clean_text"]
                    )
                    subtitle_line = f"## {formatted_text}"
                    lines.append(subtitle_line)
                    print(f"DEBUG: Added subtitle: {subtitle_line}")
        else:
            # Process all other content normally
            for para_idx, para in enumerate(block["paragraphs"]):
                line = self._convert_paragraph_to_markdown(para)
                if line:
                    lines.append(line)
                    print(f"DEBUG: Added content line: {line[:50]}{'...' if len(line) > 50 else ''}")

        # Combine paragraphs with newline separation
        result = "\n".join(lines)

        # Apply shape-level hyperlink if present
        if block.get("shape_hyperlink") and result:
            result = f"[{result}]({block['shape_hyperlink']})"

        return result

    def _convert_paragraph_to_markdown(self, para):
        """Convert individual paragraphs to markdown with structure and formatting."""
        if not para.get("clean_text"):
            return ""

        # Build formatted text from individual runs
        formatted_text = self._build_formatted_text_from_runs(
            para["formatted_runs"], para["clean_text"]
        )

        # Apply structural formatting based on paragraph hints
        hints = para.get("hints", {})

        # Handle bullet points with proper indentation
        if hints.get("is_bullet", False):
            level = hints.get("bullet_level", 0)
            if level < 0:
                level = 0  # Ensure non-negative indentation
            indent = "  " * level  # 2 spaces per indentation level
            return f"{indent}- {formatted_text}"

        # Handle numbered lists
        elif hints.get("is_numbered", False):
            return f"1. {formatted_text}"

        # Handle remaining headings (should be rare since titles are handled semantically)
        elif hints.get("likely_heading", False):
            if hints.get("all_caps") or len(para["clean_text"]) < 30:
                return f"## {formatted_text}"  # H2 for short/caps headings
            else:
                return f"### {formatted_text}"  # H3 for longer headings

        # Regular paragraph
        else:
            return formatted_text

    def _convert_group_to_markdown(self, block):
        """Convert grouped shapes to markdown by processing extracted content with semantic awareness."""
        extracted_blocks = block.get("extracted_blocks", [])

        print(f"DEBUG: Converting group with {len(extracted_blocks)} extracted blocks")

        if not extracted_blocks:
            return ""

        # Process each extracted block with semantic role handling
        content_parts = []

        for block_idx, extracted_block in enumerate(extracted_blocks):
            print(f"DEBUG: Processing group block {block_idx + 1}: type={extracted_block['type']}")

            content = None

            if extracted_block["type"] == "text":
                # Handle semantic roles in group content too
                content = self._convert_text_block_to_markdown(extracted_block)
            elif extracted_block["type"] == "image":
                content = self._convert_image_to_markdown(extracted_block)
            elif extracted_block["type"] == "table":
                content = self._convert_table_to_markdown(extracted_block)
            elif extracted_block["type"] == "chart":
                content = self._convert_chart_to_markdown(extracted_block)
            elif extracted_block["type"] == "line":
                # Lines are tracked for diagram analysis but don't produce visible content
                pass
            elif extracted_block["type"] == "arrow":
                # Arrows are tracked for diagram analysis but don't produce visible content
                pass
            elif extracted_block["type"] == "shape":
                # Generic shapes get basic representation
                content = f"[Shape: {extracted_block.get('shape_subtype', 'unknown')}]"

            if content:
                content_parts.append(content)
                print(f"DEBUG: Added group content: {content[:50]}{'...' if len(content) > 50 else ''}")

        # Combine all content with proper spacing
        group_md = "\n\n".join(content_parts) if content_parts else ""

        # Apply shape-level hyperlink if present
        if block.get("hyperlink") and group_md:
            group_md = f"[{group_md}]({block['hyperlink']})"

        return group_md

    def _build_formatted_text_from_runs(self, runs, clean_text):
        """Build formatted text from runs with consistent formatting handling."""
        if not runs:
            return clean_text

        # Filter out empty runs for formatting analysis
        text_runs = [run for run in runs if run.get("text")]

        if not text_runs:
            return clean_text

        # Analyze formatting consistency across all runs
        all_bold = all(run.get("bold", False) for run in text_runs)
        all_italic = all(run.get("italic", False) for run in text_runs)
        all_have_hyperlinks = all(run.get("hyperlink") for run in text_runs)

        # Check if all hyperlinks are the same
        if all_have_hyperlinks:
            unique_hyperlinks = set(run.get("hyperlink") for run in text_runs)
            all_same_hyperlink = len(unique_hyperlinks) == 1
        else:
            all_same_hyperlink = False

        # Apply consistent formatting to entire text
        if all_bold and all_italic and not all_same_hyperlink:
            return f"***{clean_text}***"  # Bold + italic
        elif all_bold and not all_same_hyperlink:
            return f"**{clean_text}**"  # Bold
        elif all_italic and not all_same_hyperlink:
            return f"*{clean_text}*"  # Italic
        elif all_same_hyperlink:
            # All runs have same hyperlink - apply to entire text
            hyperlink = text_runs[0]["hyperlink"]
            if all_bold and all_italic:
                return f"[***{clean_text}***]({hyperlink})"
            elif all_bold:
                return f"[**{clean_text}**]({hyperlink})"
            elif all_italic:
                return f"[*{clean_text}*]({hyperlink})"
            else:
                return f"[{clean_text}]({hyperlink})"

        # Mixed formatting - use per-run logic
        formatted_parts = []

        for run in runs:
            text = run["text"]
            if not text:
                continue

            # Apply text formatting per run
            if run.get("bold") and run.get("italic"):
                text = f"***{text}***"
            elif run.get("bold"):
                text = f"**{text}**"
            elif run.get("italic"):
                text = f"*{text}*"

            # Apply hyperlink per run
            if run.get("hyperlink"):
                text = f"[{text}]({run['hyperlink']})"

            formatted_parts.append(text)

        return "".join(formatted_parts)

    def _convert_table_to_markdown(self, block):
        """Convert table data to standard markdown table format."""
        if not block["data"]:
            return ""

        markdown = ""
        for i, row in enumerate(block["data"]):
            # Escape pipes in cell content to prevent table structure breaks
            escaped_row = [cell.replace("|", "\\|") for cell in row]
            markdown += "| " + " | ".join(escaped_row) + " |\n"

            # Add separator after header row (first row)
            if i == 0:
                markdown += "| " + " | ".join("---" for _ in row) + " |\n"

        return markdown

    def _convert_image_to_markdown(self, block):
        """Convert image blocks to markdown image syntax."""
        # Basic image markdown syntax
        image_md = f"![{block['alt_text']}](image)"

        # Wrap in hyperlink if present
        if block.get("hyperlink"):
            image_md = f"[{image_md}]({block['hyperlink']})"

        return image_md

    def _convert_chart_to_markdown(self, block):
        """Convert chart blocks to markdown with diagram candidate annotation."""
        # Basic chart information
        chart_md = f"**Chart: {block.get('title', 'Untitled Chart')}**\n"
        chart_md += f"*Chart Type: {block.get('chart_type', 'unknown')}*\n\n"

        # Add data summary if available
        if block.get('categories') and block.get('series'):
            chart_md += "Data:\n"
            for series in block['series']:
                if series.get('name'):
                    chart_md += f"- {series['name']}: "
                    if series.get('values'):
                        # Show first 5 values with truncation indicator
                        chart_md += ", ".join(map(str, series['values'][:5]))
                        if len(series['values']) > 5:
                            chart_md += "..."
                    chart_md += "\n"

        # Add diagram candidate annotation for downstream processing
        chart_md += f"\n<!-- DIAGRAM_CANDIDATE: chart, type={block.get('chart_type', 'unknown')} -->\n"

        # Apply hyperlink if present
        if block.get("hyperlink"):
            chart_md = f"[{chart_md}]({block['hyperlink']})"

        return chart_md
