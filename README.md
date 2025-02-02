📄 RAG System with DeepSeek R1 & Ollama

This is a Streamlit-based Retrieval-Augmented Generation (RAG) System that allows users to upload PDF documents, extract content, and ask questions based on the document context using the DeepSeek R1 model via Ollama.

🚀 Features

PDF Upload: Supports multiple PDF files.

Text Processing: Extracts and splits document content using Semantic Chunking.

DeepSeek R1 via Ollama: Uses the LLM for answering document-related questions.

Streaming Responses: Real-time response generation.

Download Responses: Save responses as .txt files.

Streaming Toggle: Enable/Disable response streaming via UI.

Auto-Close Notifications: Success banners disappear automatically after 5 seconds.

🛠️ Installation

Ensure you have Python 3.11+ installed.

# Clone the repository
git clone https://github.com/tyagiankush/playground.git
cd playground

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
or
uv sync

🏃‍♂️ Running the App

1️⃣ Start Ollama with DeepSeek R1

Make sure you have Ollama installed and running DeepSeek R1:

ollama run deepseek-r1

2️⃣ Run the Streamlit App

streamlit run  streamlit run ./playground/chat/deepseek.py

🖼️ UI Overview

Upload PDFs → Select one or multiple PDFs.

Toggle Streaming → Enable or disable response streaming.

Ask a Question → Enter a query based on uploaded documents.

View or Download Response → Read or save the AI-generated answer.

📌 Example Usage

Upload a PDF file.

Ask a question like: "What is the main topic of this document?"

View or download the AI-generated response.

⚡ Future Enhancements

Better Chunking Strategies to improve retrieval.

Multi-Document Querying for cross-document answers.

Different Model Support (e.g., Mistral, GPT-4, Llama2).

🤝 Contributing

Feel free to open an issue or submit a PR!

📜 License

MIT License © 2025 Your Name/Company
