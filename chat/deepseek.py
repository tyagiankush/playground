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

__MODEL = 'deepseek-r1'


def clear_banner(banner, time_in_secs: int = 5) -> None:
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

	if st.button('🗑️ Clear Chat'):
		st.session_state.chat_history = []
		st.rerun()

	if 'chat_history' not in st.session_state:
		st.session_state.chat_history = []

	user_input = st.text_input('Ask a question about your documents:', max_chars=500)

	if user_input:
		st.session_state.chat_history.append({'question': user_input, 'answer': ''})
		response_container = st.empty()
		response = ''

		if streaming:
			with st.spinner('streaming...'):
				for chunk in chain.stream({'context': documents, 'question': user_input}):
					response += chunk
					response_container.markdown(f'### **Response:**\n\n{response}')
		else:
			with st.spinner('generating...'):
				response = chain.invoke({'context': documents, 'question': user_input})
				response_container.markdown(f'### **Response:**\n\n{response}')

		# Update history
		st.session_state.chat_history[-1]['answer'] = response

		# Show history
		for chat in st.session_state.chat_history:
			st.markdown(f'**Q:** {chat["question"]}')
			st.markdown(f'**A:** {chat["answer"]}')
			st.markdown('---')

		if response:
			st.download_button('📥 Download Response', response, file_name='response.txt')
		else:
			st.warning('⚠️ No response generated. Try rephrasing your question.')


@st.cache_data
def load_document(uploaded_file: UploadedFile) -> list[Document]:
	ext = uploaded_file.name.split('.')[-1].lower()

	if ext == 'pdf':
		temp_path = Path(f'temp_{uuid.uuid4()}.pdf')
		temp_path.write_bytes(uploaded_file.getvalue())
		loader = PDFPlumberLoader(str(temp_path))
	elif ext == 'txt':
		temp_path = Path(f'temp_{uuid.uuid4()}.txt')
		temp_path.write_bytes(uploaded_file.getvalue())
		loader = TextLoader(str(temp_path))
	elif ext == 'docx':
		temp_path = Path(f'temp_{uuid.uuid4()}.docx')
		temp_path.write_bytes(uploaded_file.getvalue())
		loader = Docx2txtLoader(str(temp_path))
	else:
		st.error('Unsupported file format!')
		return []

	docs = loader.load()
	temp_path.unlink()

	text_splitter = SemanticChunker(HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))
	return text_splitter.split_documents(docs)


def generate(uploaded_files: list[UploadedFile], streaming: bool) -> None:
	# read pdf and load the document
	clear_banner(st.info('📄 Loading and processing document...'))
	documents = []
	for uploaded_file in uploaded_files:
		documents.extend(load_document(uploaded_file))
	clear_banner(st.success(body='Document successfully processed!', icon='✅'))

	# write the response
	write_response(documents, streaming)


def main() -> None:
	col1, col2 = st.columns([0.8, 0.2])

	with col1:
		st.title('📄 RAG System with DeepSeek R1 & Ollama')

	with col2:
		streaming = st.toggle(label='Streaming', value=True, key='streaming')

	uploaded_files = st.file_uploader('Upload PDFs', type=['pdf', 'txt', 'docx'], accept_multiple_files=True)

	if uploaded_files:
		generate(uploaded_files, streaming)


if __name__ == '__main__':
	main()
