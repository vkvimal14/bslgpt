# BSL GPT - AI-Powered Document Query System

A modern Flask-based web application that processes PDF documents and provides intelligent question-answering capabilities using Google's Gemini AI, specifically designed for Bokaro Steel Plant (BSL) documentation.

## Features

- 🤖 **AI-Powered Q&A**: Utilizes Google's Gemini 2.0 Flash for intelligent document analysis
- 📄 **PDF Processing**: Automatic extraction and indexing of PDF content
- 🔍 **Multi-Document Search**: Query across multiple documents simultaneously
- 🌐 **Modern Web Interface**: Responsive chat-like interface
- 🚀 **Production Ready**: Docker support, proper logging, and error handling
- 🔐 **Secure**: Environment-based configuration and input validation
- 📊 **Monitoring**: Health checks and metrics endpoints

## Quick Start

### Prerequisites

- Python 3.8+
- Google AI API key (Gemini)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/vkvimal14/bslgpt.git
cd bslgpt
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Initialize the application:
```bash
python app.py init
```

5. Run the application:
```bash
python app.py
```

Visit `http://localhost:5000` to access the web interface.

## Configuration

The application uses environment variables for configuration:

- `GEMINI_API_KEY`: Your Google AI API key
- `PDF_FOLDER`: Directory containing PDF files (default: ./pdf)
- `DATABASE_URL`: Database connection string
- `REDIS_URL`: Redis connection string for caching
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)

## API Endpoints

- `GET /`: Web interface
- `POST /api/init`: Initialize PDF contexts
- `POST /api/query`: Submit questions
- `GET /api/health`: Health check
- `GET /api/docs`: API documentation

## Docker Support

```bash
docker build -t bslgpt .
docker run -p 5000:5000 --env-file .env bslgpt
```

## Development

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black .
flake8 .
mypy .
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.