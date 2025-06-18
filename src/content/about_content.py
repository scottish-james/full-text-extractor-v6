"""
About content for the Document to Markdown Converter application.
"""


def get_about_content():
    """Return the main about content for the application."""
    return """
    **Why We Built This Tool**

    Doing a flashy demo with AI? Easy. Building an AI tool that's robust enough for production? That's the real challenge. When it comes to delivering high-quality AI solutions that meet your organisation's standards, the old saying holds true: rubbish in, rubbish out. Even "okay in, okay out" doesn't cut it when you're facing governance teams who want real confidence before anything goes live.

    That's why we built this tool. It helps you create high-quality inputs that drive high-quality outputs. The result is you get to production faster, with fewer headaches.

    A lot of company knowledge lives in PowerPoint decks, but slides aren't structured like Word docs. They're visually scattered, with no guaranteed top-to-bottom, left-to-right reading order. That makes them tricky for LLMs to interpret. This tool extracts text based on the actual reading order. And just in case that order's off, the LLM checks and adjusts it automatically.

    We've seen huge improvements in RAG performance using this tool. The embedded information, or the numerical representation of text that AI uses, becomes much more accurate. That leads to better understanding and sharper results.

    Chunking, which is deciding how to break up your documents, is another key piece. Our testing shows that section-based chunking works far better than arbitrary splits. That's why our output follows consistent rules and formatting to make your documents easier to process. Your engineers will thank you, and you'll build faster, with higher quality.

    Finally, we built this in Python, the go-to language for data engineers. That means your team can easily plug it into automated pipelines, convert documents, and send them wherever they need to go for the next stage of your project.
    """


def get_technical_benefits():
    """Return technical benefits section."""
    return {
        "RAG Performance": "Significantly improved embedding accuracy through proper text extraction and reading order",
        "Section-Based Chunking": "Consistent formatting rules that outperform arbitrary document splits",
        "Reading Order Intelligence": "AI-powered verification and adjustment of text extraction sequence",
        "Pipeline Integration": "Python-based architecture for seamless automation and deployment"
    }


def get_problem_solution_pairs():
    """Return problem/solution pairs for structured display."""
    return [
        {
            "problem": "PowerPoint slides have scattered, unstructured text layout",
            "solution": "Extract text based on actual reading order with AI verification"
        },
        {
            "problem": "Poor document chunking leads to degraded RAG performance",
            "solution": "Section-based chunking with consistent formatting rules"
        },
        {
            "problem": "Manual document conversion slows down AI project delivery",
            "solution": "Automated Python pipeline integration for seamless workflows"
        },
        {
            "problem": "Governance teams need confidence in AI solution quality",
            "solution": "High-quality inputs ensure high-quality outputs and faster approval"
        }
    ]
