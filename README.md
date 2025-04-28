# Playground Project

A collection of interactive applications including a RAG system and a coding game.

## Features

### Chat Module
- RAG (Retrieval-Augmented Generation) system using DeepSeek R1 & Ollama
- Support for PDF, TXT, and DOCX documents
- Streaming responses
- Chat history management
- Response download functionality

### Game Module
- Interactive coding game
- Real-time code validation
- Score tracking
- User-friendly interface

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd playground
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
or
uv sync
```

4. Set up environment variables:
Create a `.env` file in the root directory with the following content:
```env
API_KEY=your_api_key_here
MODEL_NAME=gpt-4o-mini
DEBUG=false
LOG_LEVEL=INFO
```

## Usage

### Chat Module
To run the RAG system:
```bash
streamlit run src/chat/deepseek.py
```

### Game Module
To run the coding game:
```bash
python src/game/space.py
```

## Project Structure

```
src/
├── chat/
│   ├── deepseek.py      # RAG system implementation
│   ├── gpt.py           # GPT integration
│   └── settings.py      # Configuration settings
├── game/
│   └── space.py         # Coding game implementation
└── logging_config.py    # Logging configuration
```

## Development

### Code Style
The project uses:
- Black for code formatting
- isort for import sorting
- mypy for type checking
- pylint for code quality

Run the following commands to maintain code quality:
```bash
black .
isort .
mypy .
pylint src/
```

### Testing
Run tests using pytest:
```bash
pytest
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
