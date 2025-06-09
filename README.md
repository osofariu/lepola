# AI Legal & Policy Research Assistant

A comprehensive AI-powered research assistant designed to help users understand, summarize, and assess the impact of legislative texts, court rulings, and regulatory documents. Built with Python, FastAPI, and LangChain.

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

- **Backend**: Python 3.11+, FastAPI, Uvicorn
- **AI/ML**: LangChain, OpenAI/Anthropic APIs
- **Document Processing**: pypdf, BeautifulSoup4
- **Vector Database**: FAISS, ChromaDB
- **Testing**: pytest, pytest-asyncio
- **Code Quality**: Ruff (formatting/linting), mypy (type checking)
- **Logging**: structlog for structured logging
- **Configuration**: Pydantic Settings with environment variables

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.11 or higher
- Poetry (for dependency management)
- OpenAI or Anthropic API key

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
# Start the development server
poetry run start

# Or using Poetry scripts:
poetry run dev

# Or manually:
poetry run python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

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

```
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