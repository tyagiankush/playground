import logging
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import (
	Docx2txtLoader,
	PDFPlumberLoader,
	TextLoader,
	UnstructuredHTMLLoader,
	UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from streamlit.runtime.uploaded_file_manager import UploadedFile

from src.chat.settings import settings

# Configure logging
logger = logging.getLogger(__name__)

__MODEL = "deepseek-r1"

# Supported file types and their loaders
SUPPORTED_FILE_TYPES = {
	"pdf": PDFPlumberLoader,
	"txt": TextLoader,
	"docx": Docx2txtLoader,
	"html": UnstructuredHTMLLoader,
	"md": UnstructuredMarkdownLoader,
}


def extract_metadata(file_path: Path) -> dict[str, Any]:
	"""Extract metadata from a file."""
	try:
		metadata = {
			"filename": file_path.name,
			"extension": file_path.suffix[1:],
			"size": file_path.stat().st_size,
			"created": datetime.fromtimestamp(file_path.stat().st_ctime),
			"modified": datetime.fromtimestamp(file_path.stat().st_mtime),
		}
		return metadata
	except Exception as e:
		logger.error(f"Error extracting metadata: {str(e)}")
		return {}


@contextmanager
def temporary_file(file_content: bytes, extension: str) -> Generator[Path, None, None]:
	"""Context manager for handling temporary files."""
	temp_path = Path(f"./target/temp_{uuid.uuid4()}.{extension}")
	try:
		temp_path.write_bytes(file_content)
		yield temp_path
	finally:
		if temp_path.exists():
			temp_path.unlink()


def prepare_chain(model_name: str = None) -> tuple[Runnable[dict[str, Any], Any], Any]:
	"""Prepare the LLM chain with proper error handling."""
	try:
		# Get model configuration
		model_name = model_name or settings.default_model
		model_config = settings.models[model_name]

		# Initialize LLM
		llm = OllamaLLM(
			model=model_config.name,
			temperature=model_config.temperature,
		)

		# Prepare prompt
		prompt = """
		Use the following context to answer the question.
		Context: {context}
		Question: {question}
		Answer:"""
		qa_prompt = ChatPromptTemplate.from_template(prompt)

		# Create base chain
		chain = create_stuff_documents_chain(llm, qa_prompt)

		return chain, llm
	except Exception as e:
		logger.error(f"Failed to prepare chain: {str(e)}")
		raise


def create_hybrid_retriever(documents: list[Document]) -> Any:
	"""Create a hybrid retriever combining semantic and keyword search."""
	try:
		# Create embeddings
		embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

		# Create vector store
		vectorstore = FAISS.from_documents(documents, embeddings)

		# Create base retriever
		base_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

		# Create multi-query retriever
		multi_retriever = MultiQueryRetriever.from_llm(
			retriever=base_retriever, llm=OllamaLLM(model=settings.default_model)
		)

		# Create contextual compression
		compressor = LLMChainExtractor.from_llm(llm=OllamaLLM(model=settings.default_model))

		# Create hybrid retriever
		retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=multi_retriever)

		return retriever
	except Exception as e:
		logger.error(f"Error creating hybrid retriever: {str(e)}")
		raise


def summarize_document(documents: list[Document]) -> str:
	"""Generate a summary of the document."""
	try:
		llm = OllamaLLM(model=__MODEL, temperature=0.3)
		prompt = """
		Please provide a concise summary of the following document content.
		Focus on the main points and key information.
		
		Content: {content}
		
		Summary:"""

		summary_prompt = ChatPromptTemplate.from_template(prompt)
		chain = create_stuff_documents_chain(llm, summary_prompt)

		# Combine all document content
		content = "\n\n".join(doc.page_content for doc in documents)
		summary = chain.invoke({"content": content})

		return summary
	except Exception as e:
		logger.error(f"Error generating summary: {str(e)}")
		return "Unable to generate summary."


def track_citations(response: str, documents: list[Document]) -> list[dict[str, Any]]:
	"""Track which documents contributed to the response."""
	try:
		citations = []
		for doc in documents:
			# Simple content matching for now - could be improved with semantic similarity
			if any(phrase.lower() in response.lower() for phrase in doc.page_content.split()[:10]):
				citations.append(
					{
						"filename": doc.metadata.get("filename", "Unknown"),
						"page": doc.metadata.get("page", 1),
						"snippet": doc.page_content[:200] + "...",
					}
				)
		return citations
	except Exception as e:
		logger.error(f"Error tracking citations: {str(e)}")
		return []


def write_response(documents: list[Document], streaming: bool) -> None:
	"""Write response with proper error handling and logging."""
	logger.info(f"Write response was called at {datetime.now()}")

	try:
		# Create hybrid retriever if enabled
		retriever = None
		if settings.use_hybrid_search:
			retriever = create_hybrid_retriever(documents)

		# Prepare chain
		chain, llm = prepare_chain()

		col1, col2 = st.columns([0.8, 0.2], vertical_alignment="center", gap="medium")

		with col2:
			# Model selection
			model_name = st.selectbox(  # noqa: F841
				"Select Model",
				options=list(settings.models.keys()),
				index=list(settings.models.keys()).index(settings.default_model),
			)

			if st.button(label="Clear Chats", icon="🗑", type="secondary", use_container_width=True):
				st.session_state.user_input = ""
				st.session_state.chat_history = []

		if "chat_history" not in st.session_state:
			st.session_state.chat_history = []

		if "my_text" not in st.session_state:
			st.session_state.my_text = ""

		def submit():
			st.session_state.my_text = st.session_state.widget
			st.session_state.widget = ""

		with col1:
			st.text_input(
				label="Ask a question about your documents:",
				key="widget",
				on_change=submit,
				max_chars=500,
			)

		user_input = st.session_state.my_text
		if user_input:
			st.session_state.user_input = user_input
			st.session_state.chat_history.append({"question": user_input, "answer": ""})
			response_container = st.empty()
			response = ""

			try:
				# Get relevant documents using hybrid search if enabled
				relevant_docs = documents
				if retriever:
					retriever: ContextualCompressionRetriever = retriever
					relevant_docs = retriever.invoke(input=user_input)

				if streaming:
					with st.spinner("streaming..."):
						for chunk in chain.stream({"context": relevant_docs, "question": user_input}):
							response += chunk
							response_container.markdown(f"### **Response:**\n\n{response}")
				else:
					with st.spinner("generating..."):
						response = chain.invoke({"context": relevant_docs, "question": user_input})
						response_container.markdown(f"### **Response:**\n\n{response}")

				st.session_state.chat_history[-1]["answer"] = response

				# Display document summary
				with st.expander("Document Summary"):
					summary = summarize_document(documents)
					st.markdown(summary)

				# Display citations
				citations = track_citations(response, documents)
				if citations:
					with st.expander("Sources"):
						for citation in citations:
							st.markdown(f"**{citation['filename']}**")
							st.markdown(f"*{citation['snippet']}*")
							st.markdown("---")

				for chat in st.session_state.chat_history:
					st.markdown(f"**Q:** {chat['question']}")
					st.markdown(f"**A:** {chat['answer']}")
					st.markdown("---")

				if response:
					st.download_button(data=response, label="Download Response", icon="📥", file_name="response.txt")
				else:
					st.warning(body="No response generated. Try rephrasing your question.", icon="⚠️")
			except Exception as e:
				logger.error(f"Error generating response: {str(e)}")
				st.error("An error occurred while generating the response. Please try again.")
	except Exception as e:
		logger.error(f"Error generating response: {str(e)}")
		st.error("An error occurred while generating the response. Please try again.")


@st.cache_data
def load_document(uploaded_file: UploadedFile) -> list[Document]:
	"""Load and process document with proper error handling."""
	logger.info(f"Loading document: {uploaded_file.name}")
	ext = uploaded_file.name.split(".")[-1].lower()

	try:
		with temporary_file(uploaded_file.getvalue(), ext) as temp_path:
			if ext not in SUPPORTED_FILE_TYPES:
				st.error(f"Unsupported file format: {ext}")
				logger.error(f"Unsupported file format: {ext}")
				return []

			# Extract metadata
			metadata = extract_metadata(temp_path)

			# Load document
			loader = SUPPORTED_FILE_TYPES[ext](str(temp_path))
			docs = loader.load()

			# Add metadata to each document
			for doc in docs:
				doc.metadata.update(metadata)

			# Split documents
			text_splitter = SemanticChunker(HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))
			documents = text_splitter.split_documents(docs)

			logger.info(f"Successfully processed document: {uploaded_file.name}")
			return documents
	except Exception as e:
		logger.error(f"Error processing document {uploaded_file.name}: {str(e)}")
		st.error(f"Error processing document: {str(e)}")
		return []


def generate(uploaded_files: list[UploadedFile], streaming: bool) -> None:
	"""Generate responses for uploaded files."""
	documents = []
	for uploaded_file in uploaded_files:
		docs = load_document(uploaded_file)
		if docs:
			documents.extend(docs)

	if not documents:
		st.error("No valid documents were processed. Please check your files and try again.")
		return

	write_response(documents, streaming)


def main() -> None:
	"""Main function to run the Streamlit app."""
	try:
		col1, col2 = st.columns([0.8, 0.2])

		with col1:
			st.title("📄 RAG System with DeepSeek R1 & Ollama")

		with col2:
			streaming = st.toggle(label="Streaming", value=True, key="streaming")

		uploaded_files = st.file_uploader(
			"Upload PDFs", type=["pdf", "txt", "docx", "html", "md"], accept_multiple_files=True
		)

		if uploaded_files:
			generate(uploaded_files, streaming)
	except Exception as e:
		logger.error(f"Application error: {str(e)}")
		st.error("An unexpected error occurred. Please try again later.")


if __name__ == "__main__":
	main()
