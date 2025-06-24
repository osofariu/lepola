# AI Legal & Policy Research Assistant

A comprehensive AI-powered research assistant designed to help users understand, summarize, and assess the impact of legislative texts, court rulings, and regulatory documents. Built with Python, FastAPI, and LangChain.

## Progress

This project is in progress of being implemented.  Much of the code was produced with the help of Claude 4, and that has been a fun and interesting experience.

Not all the features below exist yet, but here's what works:

- the server exposes all the endpoints for injestion, pipeline, and chat
- a sqlite3 database stores documents and metadata
- injestion works with PDF documents and text
- a simple analysis pipeline is able to extract entities and summary from text
- accounted for _some_ variation in output format for entity extraction by openAI's gpt-4-turbo-preview, Meta's llama3.1 and Google's gemma3:27b models.
- added ability to create embeddings from a document, but not using it yet
- created a cli for interacting with the server from the command line.

There's a lot more to to, but the focus of this project is learning not building a full-fledge production application.

## 🎯 Features

### 1. Document Ingestion

- **Multi-format Support**: PDF, plain text, HTML, and web URLs
- **Intelligent Parsing**: Extracts structured text and metadata
- **Content Validation**: File size limits and type checking
- **Checksum Verification**: Ensures document integrity

### 2. AI-Powered Analysis

- **Entity Extraction**: Identifies laws, agencies, affected groups, and legal concepts
- **Risk Assessment**: Evaluates civil rights, privacy, and constitutional implications
- **Impact Analysis**: Summarizes potential effects on different populations
- **Confidence Scoring**: Provides transparency in AI predictions

### 3. Interactive Querying

- **Natural Language Questions**: Ask questions like "Who does this bill affect?"
- **Contextual Answers**: Responses backed by source citations
- **Multi-document Search**: Query across multiple analyzed documents
- **Follow-up Suggestions**: Intelligent question recommendations

### 4. Output Generation

- **Multiple Formats**: Markdown, HTML, JSON, and PDF reports
- **Customizable Templates**: Tailored outputs for different audiences
- **Source Citations**: Complete traceability to original documents
- **Export Options**: Ready for newsletters, reports, or fact sheets

### 5. Responsible AI Features

- **Transparency**: Full audit trails and decision logging
- **Confidence Indicators**: Clear uncertainty communication
- **Human Review Flags**: Automatic flagging of low-confidence results
- **Privacy Protection**: Local processing with optional cloud deployment

## 🛠️ Technology Stack

- **Backend**: Python 3.13+ (3.11+ supported), FastAPI, Uvicorn
- **AI/ML**: LangChain, OpenAI/Anthropic/Google APIs, works with ollama
- **Document Processing**: pypdf, BeautifulSoup4
- **Vector Database**: FAISS, ChromaDB
- **Testing**: pytest, pytest-asyncio
- **Code Quality**: Ruff (formatting/linting), mypy (type checking)
- **Logging**: structlog for structured logging
- **Configuration**: Pydantic Settings with environment variables

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.13 or higher (3.11+ supported)
- Poetry (for dependency management)
- OpenAI or Anthropic API key or local ollama

**Note for Python 3.13 users**: The project includes automatic compatibility settings for PyO3-based packages (like tiktoken). If you encounter installation issues, ensure `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` is set in your environment (it's included in the `.env.example`).

### 2. Installation

```bash
# Clone the repository
git clone <repository-url>
cd lepola

# Install dependencies
poetry install

# Activate virtual environment
poetry shell

# Set up pre-commit hooks (optional)
poetry run pre-commit install
```

### 3. Configuration

Copy the example environment file and configure your settings:

```bash
# Create configuration file
cp .env.example .env

# Edit configuration (add your API keys)
# Required: OPENAI_API_KEY or ANTHROPIC_API_KEY
# Optional: Adjust other settings as needed
```

### 4. Run the Application

```bash
# Development mode (recommended for development)
poetry run dev
# Features: auto-reload, debug logging, external access, colored output

# Production mode (recommended for production-like testing)
poetry run start
# Features: uses settings from .env, controlled reload, production logging

# Alternative commands:
poetry run lepola  # Same as start

# Manual options:
poetry run python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
poetry run python -m src.main
```

The API will be available at:

- **API**: <http://localhost:8000>
- **Documentation**: <http://localhost:8000/docs>
- **Alternative Docs**: <http://localhost:8000/redoc>

### 5. Test the Setup

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest tests/ -v --cov=src --cov-report=html

# Check code quality
poetry run ruff check
poetry run ruff format
poetry run mypy src/
```

## 🖥️ Command Line Interface (CLI)

The Lepola CLI provides easy access to all server functionality from the command line. It's designed to be intuitive and user-friendly for both interactive use and automation.

### Installation

The CLI is included with the project installation. After installing dependencies with Poetry, you can use it directly:

```bash
# Make sure you're in the project directory and have activated the virtual environment
poetry shell

# The CLI is available as a Python script
python scripts/lepola_cli.py --help
```

### Basic Usage

```bash
# Check server health
python scripts/lepola_cli.py health

# Use a different server URL
python scripts/lepola_cli.py --server http://localhost:8001 health
```

### Document Management

#### Upload Documents

```bash
# Upload a local file
python scripts/lepola_cli.py documents upload examples/sample_bill.txt

# Upload with metadata
python scripts/lepola_cli.py documents upload examples/sample_bill.txt \
  --metadata '{"source": "congress.gov", "category": "privacy"}'

# Upload without automatic embedding (for large files)
python scripts/lepola_cli.py documents upload large_document.pdf --no-embedding
```

#### Ingest from URLs

```bash
# Ingest content from a URL
python scripts/lepola_cli.py documents ingest-url https://example.com/bill.pdf

# Ingest with metadata
python scripts/lepola_cli.py documents ingest-url https://example.com/bill.pdf \
  --metadata '{"source": "government", "priority": "high"}'
```

#### List and View Documents

```bash
# List all documents
python scripts/lepola_cli.py documents list

# List with pagination
python scripts/lepola_cli.py documents list --limit 10 --offset 20

# Filter by file type
python scripts/lepola_cli.py documents list --file-type pdf

# Filter by status
python scripts/lepola_cli.py documents list --status processed

# Get detailed information about a specific document
python scripts/lepola_cli.py documents get <document-id>
```

### Interactive Querying

#### Ask Questions

```bash
# Ask a general question about all documents
python scripts/lepola_cli.py query ask "What are the privacy implications of this legislation?"

# Ask about specific documents
python scripts/lepola_cli.py query ask "Who does this bill affect?" \
  --document-ids <doc-id-1> <doc-id-2>

# Ask follow-up questions
python scripts/lepola_cli.py query ask "What are the enforcement mechanisms?"
```

### AI Pipeline Operations

#### Document Analysis

```bash
# Start analysis of a document
python scripts/lepola_cli.py pipeline analyze <document-id>

# Check analysis results
python scripts/lepola_cli.py pipeline results <analysis-id>

# List all analyses
python scripts/lepola_cli.py pipeline list

# List analyses with filters
python scripts/lepola_cli.py pipeline list --limit 5 --status completed
```

### Output Generation

#### Generate Reports

```bash
# Generate markdown report (default)
python scripts/lepola_cli.py outputs generate <analysis-id>

# Generate HTML report
python scripts/lepola_cli.py outputs generate <analysis-id> --format html

# Generate JSON output
python scripts/lepola_cli.py outputs generate <analysis-id> --format json

# Generate PDF report
python scripts/lepola_cli.py outputs generate <analysis-id> --format pdf
```

### Service Status Monitoring

#### Check Service Health

```bash
# Check overall server health
python scripts/lepola_cli.py health

# Check ingestion service status
python scripts/lepola_cli.py status ingestion

# Check query service status
python scripts/lepola_cli.py status query

# Check pipeline service status
python scripts/lepola_cli.py status pipeline

# Check outputs service status
python scripts/lepola_cli.py status outputs

# Check embeddings service status
python scripts/lepola_cli.py status embeddings
```

### Advanced Usage Examples

#### Complete Workflow

```bash
# 1. Upload a document
python scripts/lepola_cli.py documents upload examples/sample_bill.txt

# 2. Start analysis (use the document ID from step 1)
python scripts/lepola_cli.py pipeline analyze <document-id>

# 3. Wait for analysis to complete, then check results
python scripts/lepola_cli.py pipeline results <analysis-id>

# 4. Generate a report
python scripts/lepola_cli.py outputs generate <analysis-id> --format markdown

# 5. Ask questions about the document
python scripts/lepola_cli.py query ask "What are the key provisions?" --document-ids <document-id>
```

#### Batch Operations

```bash
# Upload multiple documents
for file in documents/*.pdf; do
  python scripts/lepola_cli.py documents upload "$file"
done

# Analyze all documents
python scripts/lepola_cli.py documents list --limit 100 | grep "ID:" | cut -d' ' -f2 | \
  while read doc_id; do
    python scripts/lepola_cli.py pipeline analyze "$doc_id"
  done
```

#### Automation Scripts

```bash
#!/bin/bash
# Example automation script

# Check server health first
if ! python scripts/lepola_cli.py health > /dev/null 2>&1; then
    echo "Server is not running. Please start the server first."
    exit 1
fi

# Upload and analyze a document
echo "Uploading document..."
DOC_ID=$(python scripts/lepola_cli.py documents upload "$1" | grep "Document ID:" | cut -d' ' -f3)

echo "Starting analysis..."
ANALYSIS_ID=$(python scripts/lepola_cli.py pipeline analyze "$DOC_ID" | grep "Analysis ID:" | cut -d' ' -f3)

echo "Waiting for analysis to complete..."
sleep 30

echo "Generating report..."
python scripts/lepola_cli.py outputs generate "$ANALYSIS_ID" --format markdown
```

### CLI Features

- **Rich Output**: Beautiful tables and formatted output using Rich library
- **Progress Indicators**: Visual progress bars for long-running operations
- **Error Handling**: Clear error messages and helpful suggestions
- **Async Operations**: Non-blocking operations for better user experience
- **Configuration**: Easy server URL configuration
- **Help System**: Comprehensive help for all commands

### Troubleshooting

#### Common Issues

```bash
# Server not running
python scripts/lepola_cli.py health
# Error: Connection error: Cannot connect to host localhost:8000

# Solution: Start the server first
poetry run dev

# Invalid document ID
python scripts/lepola_cli.py documents get invalid-id
# Error: HTTP 404: Document not found

# Solution: Use a valid document ID from the list command
python scripts/lepola_cli.py documents list

# File not found
python scripts/lepola_cli.py documents upload nonexistent.pdf
# Error: File not found: nonexistent.pdf

# Solution: Check the file path and ensure the file exists
```

#### Getting Help

```bash
# General help
python scripts/lepola_cli.py --help

# Command-specific help
python scripts/lepola_cli.py documents --help
python scripts/lepola_cli.py documents upload --help
python scripts/lepola_cli.py query ask --help
```

## 📖 Usage Examples

### Document Upload

```bash
# Upload a document
curl -X POST "http://localhost:8000/api/v1/ingestion/upload" \
  -F "file=@examples/sample_bill.txt"

# Ingest from URL
curl -X POST "http://localhost:8000/api/v1/ingestion/url" \
  -F "url=https://example.com/bill.pdf"
```

### Document Analysis

```bash
# Start analysis
curl -X POST "http://localhost:8000/api/v1/pipeline/analyze/{document_id}"

# Check results
curl "http://localhost:8000/api/v1/pipeline/analysis/{analysis_id}"
```

### Interactive Querying

```bash
# Ask a question
curl -X POST "http://localhost:8000/api/v1/query/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the privacy implications of this bill?",
    "max_results": 5
  }'
```

### Generate Reports

```bash
# Generate markdown report
curl -X POST "http://localhost:8000/api/v1/outputs/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "analysis_id": "uuid-here",
    "format": "markdown",
    "include_sources": true
  }'
```

## 🏗️ Project Structure

``` text
lepola/
├── src/                     # Source code
│   ├── core/               # Core functionality
│   │   ├── config.py       # Configuration management
│   │   ├── logging.py      # Logging setup
│   │   └── models.py       # Pydantic models
│   ├── ingestion/          # Document ingestion
│   │   ├── service.py      # Ingestion logic
│   │   └── router.py       # API endpoints
│   ├── pipeline/           # AI analysis pipeline
│   │   ├── service.py      # Analysis logic
│   │   └── router.py       # API endpoints
│   ├── querying/           # Interactive querying
│   │   └── router.py       # API endpoints
│   ├── outputs/            # Report generation
│   │   └── router.py       # API endpoints
│   └── main.py             # FastAPI application
├── tests/                  # Test suite
│   ├── test_config.py      # Configuration tests
│   ├── test_ingestion.py   # Ingestion tests
│   └── test_main.py        # Application tests
├── examples/               # Sample documents
├── outputs/                # Generated reports
├── data/                   # Data storage
├── pyproject.toml          # Project configuration & dependencies
├── poetry.lock             # Dependency lock file
├── pytest.ini             # Test configuration
├── ruff.toml              # Code quality config
└── README.md              # This file
```

## 🧪 Development

### Code Quality

This project follows strict code quality standards:

```bash
# Format code
poetry run ruff format

# Lint code
poetry run ruff check

# Type check
poetry run mypy src/

# Run all quality checks
poetry run ruff format && poetry run ruff check && poetry run mypy src/
```

### Testing

```bash
# Run all tests
poetry run pytest

# Run specific test file
poetry run pytest tests/test_ingestion.py -v

# Run with coverage
poetry run pytest --cov=src --cov-report=html

# Run only fast tests (exclude slow/integration)
poetry run pytest -m "not slow and not integration"
```

### Adding New Features

1. **Create the service logic** in the appropriate module
2. **Add API endpoints** in the corresponding router
3. **Write comprehensive tests** with >90% coverage
4. **Update documentation** including docstrings
5. **Add type annotations** for all functions
6. **Follow the project's coding standards**

## 🔒 Security & Privacy

- **API Keys**: Never commit API keys to version control
- **Data Processing**: Documents are processed locally by default
- **Audit Logging**: All AI operations are logged for transparency
- **Rate Limiting**: Built-in protection against abuse
- **Input Validation**: Comprehensive validation of all inputs

## 🌍 Deployment

### Local Development

The application is configured for local development by default.

### Cloud Deployment (AWS)

The application supports deployment to AWS using:

- **ECS/Fargate**: For containerized deployment
- **Lambda**: For serverless deployment
- **S3**: For document storage
- **CloudWatch**: For monitoring and logging

See the configuration section for AWS-specific environment variables.

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/new-feature`
3. **Write tests** for your changes
4. **Ensure code quality**: Run formatting, linting, and type checking
5. **Submit a pull request** with a clear description

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

## ⚠️ Disclaimer

This tool is designed to assist with legal and policy research but should not be considered a substitute for professional legal advice. All AI-generated content should be reviewed by qualified professionals before being used for official purposes.

## 🆘 Support

- **Documentation**: Check the `/docs` endpoint when running the application
- **Issues**: Report bugs and feature requests via GitHub issues
- **Discussions**: Use GitHub discussions for questions and community support

---

Built with ❤️ for legal researchers, policy advocates, and civil society organizations.
