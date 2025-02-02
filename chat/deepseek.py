import time
import uuid
from pathlib import Path
from typing import Any

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

__MODEL = "deepseek-r1"


def clear_banner(banner, time_in_secs: int = 3) -> None:
	time.sleep(time_in_secs)
	banner.empty()


def prepare_chain() -> Runnable[dict[str, Any], Any]:
	llm = OllamaLLM(
		model=__MODEL,
		temperature=0.5,
	)
	prompt = """
    Use the following context to answer the question.
    Context: {context}
    Question: {question}
    Answer:"""
	qa_prompt = ChatPromptTemplate.from_template(prompt)

	# prepare chain from context and question
	return create_stuff_documents_chain(llm, qa_prompt)


def write_response(documents: list[Document], streaming: bool) -> None:
	# setup the llm and prompt
	chain = prepare_chain()

	col1, col2 = st.columns([0.8, 0.2])
	with col2:
		if st.button(label="Clear History", icon="🗑️", type="secondary", use_container_width=True):
			st.session_state.chat_history = []
			st.rerun()

	# Custom CSS to pin the text input at the bottom
	st.markdown(
		"""
	    <style>
	    .bottom-container {
	        position: fixed;
	        bottom: 20px;
	        left: 0;
	        right: 0;
	        background-color: white;
	        padding: 10px;
	        z-index: 999;
	    }
	    </style>
	    """,  # noqa: E101
		unsafe_allow_html=True,
	)

	if "chat_history" not in st.session_state:
		st.session_state.chat_history = []

	# Check if 'user_input' is not in session_state, and initialize it
	if "my_text" not in st.session_state:
		st.session_state.my_text = ""

	# Function to clear the input after Enter is pressed
	def submit():
		st.session_state.my_text = st.session_state.widget
		st.session_state.widget = ""

	# Display the text input field with the current value from session_state
	with col1, st.container():
		# st.markdown('<div class="bottom-container">', unsafe_allow_html=True)
		st.text_input(
			label="Ask a question about your documents:",
			key="widget",
			on_change=submit,
			max_chars=500,
		)
		st.markdown("</div>", unsafe_allow_html=True)

	user_input = st.session_state.my_text
	# Check if the user presses Enter (when input field is not empty)
	if user_input:
		st.session_state.user_input = user_input  # Store input in session state

	if user_input:
		st.session_state.chat_history.append({"question": user_input, "answer": ""})
		response_container = st.empty()
		response = ""

		if streaming:
			with st.spinner("streaming..."):
				for chunk in chain.stream({"context": documents, "question": user_input}):
					response += chunk
					response_container.markdown(f"### **Response:**\n\n{response}")
		else:
			with st.spinner("generating..."):
				response = chain.invoke({"context": documents, "question": user_input})
				response_container.markdown(f"### **Response:**\n\n{response}")

		# Update history
		st.session_state.chat_history[-1]["answer"] = response

		# Show history
		for chat in st.session_state.chat_history:
			st.markdown(f"**Q:** {chat['question']}")
			st.markdown(f"**A:** {chat['answer']}")
			st.markdown("---")

		if response:
			st.download_button("📥 Download Response", response, file_name="response.txt")
		else:
			st.warning("⚠️ No response generated. Try rephrasing your question.")


@st.cache_data
def load_document(uploaded_file: UploadedFile) -> list[Document]:
	# clear_banner(st.info("📄 Loading and processing document..."))
	ext = uploaded_file.name.split(".")[-1].lower()

	if ext == "pdf":
		temp_path = Path(f"temp_{uuid.uuid4()}.pdf")
		temp_path.write_bytes(uploaded_file.getvalue())
		loader = PDFPlumberLoader(str(temp_path))
	elif ext == "txt":
		temp_path = Path(f"temp_{uuid.uuid4()}.txt")
		temp_path.write_bytes(uploaded_file.getvalue())
		loader = TextLoader(str(temp_path))
	elif ext == "docx":
		temp_path = Path(f"temp_{uuid.uuid4()}.docx")
		temp_path.write_bytes(uploaded_file.getvalue())
		loader = Docx2txtLoader(str(temp_path))
	else:
		st.error("Unsupported file format!")
		return []

	docs = loader.load()
	temp_path.unlink()

	text_splitter = SemanticChunker(HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))
	documents = text_splitter.split_documents(docs)

	# clear_banner(st.success(body="Document successfully processed!", icon="✅"))
	return documents


def generate(uploaded_files: list[UploadedFile], streaming: bool) -> None:
	# read pdf and load the document
	documents = []
	for uploaded_file in uploaded_files:
		documents.extend(load_document(uploaded_file))

	# write the response
	write_response(documents, streaming)


def main() -> None:
	col1, col2 = st.columns([0.8, 0.2])

	with col1:
		st.title("📄 RAG System with DeepSeek R1 & Ollama")

	with col2:
		streaming = st.toggle(label="Streaming", value=True, key="streaming")

	uploaded_files = st.file_uploader("Upload PDFs", type=["pdf", "txt", "docx"], accept_multiple_files=True)

	if uploaded_files:
		generate(uploaded_files, streaming)


if __name__ == "__main__":
	main()
