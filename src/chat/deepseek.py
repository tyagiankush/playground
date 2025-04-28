import logging
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator, List, Optional

import streamlit as st
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_community.document_loaders import Docx2txtLoader, PDFPlumberLoader, TextLoader
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


def prepare_chain() -> Runnable[dict[str, Any], Any]:
	"""Prepare the LLM chain with proper error handling."""
	try:
		llm = OllamaLLM(
			model=__MODEL,
			temperature=settings.temperature,
		)
		prompt = """
    Use the following context to answer the question.
    Context: {context}
    Question: {question}
    Answer:"""
		qa_prompt = ChatPromptTemplate.from_template(prompt)
		return create_stuff_documents_chain(llm, qa_prompt)
	except Exception as e:
		logger.error(f"Failed to prepare chain: {str(e)}")
		raise


def write_response(documents: List[Document], streaming: bool) -> None:
	"""Write response with proper error handling and logging."""
	logger.info(f"Write response was called at {datetime.now()}")
	
	try:
		chain = prepare_chain()
	except Exception as e:
		st.error("Failed to initialize the language model. Please try again later.")
		logger.error(f"Chain preparation failed: {str(e)}")
		return

	col1, col2 = st.columns(
		[0.8, 0.2],
		vertical_alignment="center",
		gap="medium",
	)

	with col2:
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
			if streaming:
				with st.spinner("streaming..."):
					for chunk in chain.stream({"context": documents, "question": user_input}):
						response += chunk
						response_container.markdown(f"### **Response:**\n\n{response}")
			else:
				with st.spinner("generating..."):
					response = chain.invoke({"context": documents, "question": user_input})
					response_container.markdown(f"### **Response:**\n\n{response}")

			st.session_state.chat_history[-1]["answer"] = response

			for chat in st.session_state.chat_history:
				st.markdown(f"**Q:** {chat['question']}")
				st.markdown(f"**A:** {chat['answer']}")
				st.markdown("---")

			if response:
				st.download_button(
					data=response,
					label="Download Response",
					icon="📥",
					file_name="response.txt"
				)
			else:
				st.warning(
					body="No response generated. Try rephrasing your question.",
					icon="⚠️"
				)
		except Exception as e:
			logger.error(f"Error generating response: {str(e)}")
			st.error("An error occurred while generating the response. Please try again.")


@st.cache_data
def load_document(uploaded_file: UploadedFile) -> List[Document]:
	"""Load and process document with proper error handling."""
	logger.info(f"Loading document: {uploaded_file.name}")
	ext = uploaded_file.name.split(".")[-1].lower()

	try:
		with temporary_file(uploaded_file.getvalue(), ext) as temp_path:
			if ext == "pdf":
				loader = PDFPlumberLoader(str(temp_path))
			elif ext == "txt":
				loader = TextLoader(str(temp_path))
			elif ext == "docx":
				loader = Docx2txtLoader(str(temp_path))
			else:
				st.error("Unsupported file format!")
				logger.error(f"Unsupported file format: {ext}")
				return []

			docs = loader.load()
			text_splitter = SemanticChunker(
				HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
			)
			documents = text_splitter.split_documents(docs)
			logger.info(f"Successfully processed document: {uploaded_file.name}")
			return documents
	except Exception as e:
		logger.error(f"Error processing document {uploaded_file.name}: {str(e)}")
		st.error(f"Error processing document: {str(e)}")
		return []


def generate(uploaded_files: List[UploadedFile], streaming: bool) -> None:
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
			"Upload PDFs",
			type=["pdf", "txt", "docx"],
			accept_multiple_files=True
		)

		if uploaded_files:
			generate(uploaded_files, streaming)
	except Exception as e:
		logger.error(f"Application error: {str(e)}")
		st.error("An unexpected error occurred. Please try again later.")


if __name__ == "__main__":
	main()
