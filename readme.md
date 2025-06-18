# Document to Markdown Converter 📄

A Streamlit web application that converts documents to clean, structured Markdown with AI enhancement using Claude Sonnet 4. Optimised for PowerPoint presentations with intelligent reading order detection.

## 🎯 Why We Built This

Company knowledge often lives in PowerPoint decks, but slides aren't structured like documents. They're visually scattered with no guaranteed reading order, making them tricky for AI systems to interpret properly. This tool extracts text based on actual reading order and uses Claude to verify and enhance the structure.

We've seen significant improvements in RAG performance using this approach. Better document structure leads to more accurate embeddings and better AI results.

## ✨ Features

### 🎯 PowerPoint Processing (Optimised)
- **Smart Reading Order**: Extracts text based on spatial positioning with AI verification
- **Format Preservation**: Maintains bold, italic, bullet points, and tables
- **Hyperlink Extraction**: Preserves clickable links within text
- **Diagram Recognition**: Identifies potential diagrams *(screenshot functionality in development)*

### 📝 Document Support
- **Presentations**: PowerPoint (.pptx, .ppt) 
- **Documents**: Word (.docx, .doc), PDF, EPub
- **Spreadsheets**: Excel (.xlsx, .xls)
- **Web & Data**: HTML, CSV, JSON, XML

### 🤖 Claude Sonnet 4 Enhancement
- **Structure Optimisation**: Improves formatting and hierarchy
- **Content Reordering**: Fixes misordered sequences (e.g., "first", "second", "third")
- **Metadata Generation**: Adds comprehensive metadata for AI systems

### 📊 Batch Processing
- **Folder Processing**: Convert multiple files with PowerPoint prioritisation
- **Pipeline Integration**: Python-based for automation

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   streamlit run app.py
   ```

3. **Get Claude API key:**
   - Visit [console.anthropic.com](https://console.anthropic.com/)
   - Generate API key
   - Enter in the application

## 📖 Usage

### Single File
1. Upload document in "File Upload" tab
2. Enable Claude enhancement and add API key
3. Convert and download result

### Batch Processing  
1. Select "Folder Processing" tab
2. Choose input folder
3. Process all compatible files at once

## 🗺️ Roadmap

### In Development
- **Screenshot Functionality**: Automatic diagram capture for complex visuals
- **Docling Integration**: Enhanced PDF processing capabilities

### Planned
- API endpoint for programmatic access
- Docker containerisation
- Additional document formats

## 🔧 Technical Details

Built with Streamlit and powered by:
- **Claude Sonnet 4**: AI enhancement and structure optimisation
- **python-pptx**: Advanced PowerPoint processing
- **MarkItDown**: Base document conversion
- **PyMuPDF**: PDF hyperlink extraction

## 💡 Best Practices

- Use original PowerPoint files rather than PDF exports when possible
- Review converted content for accuracy
- Use batch processing for multiple files
- Well-organised source documents produce better results

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## 📄 Licence

MIT Licence - see [LICENSE](LICENSE) file for details.

---

**Developed by James Taylor**

Turn your documents into AI-ready markdown in minutes. 🚀
