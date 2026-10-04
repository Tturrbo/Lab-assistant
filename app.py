import io
import pdfplumber
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from ai_helper import answer_question, create_embedding_model, load_api_key, search_query


st.set_page_config(page_title="Healthcare PDF Chat", page_icon="🩺")

st.markdown(
    """
    <style>
        .block-container { padding-top: 1rem; }
        div[data-testid="stFileUploader"] { margin-top: 0.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def extract_pdf_text(file_bytes: bytes, file_name: str) -> str:
    try:
        pdf_stream = io.BytesIO(file_bytes)
        with pdfplumber.open(pdf_stream) as pdf:
            pages = []
            for page in pdf.pages:
                text = page.extract_text() or ""
                if text:
                    pages.append(text)
            return "\n\n".join(pages)
    except Exception as exc:
        st.error(f"Не удалось прочитать файл {file_name}: {exc}")
        return ""


@st.cache_data
def build_pdf_documents(uploaded_files: list[str]) -> list[Document]:
    documents: list[Document] = []

    for file_name in uploaded_files:
        pass
    return documents


def create_pdf_vector_store(uploaded_files) -> FAISS | None:
    docs: list[Document] = []
    for uploaded_file in uploaded_files:
        text = extract_pdf_text(uploaded_file.getvalue(), uploaded_file.name)
        if not text.strip():
            continue

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_text(text)
        docs.extend(Document(page_content=chunk) for chunk in chunks)

    if not docs:
        st.warning("Не удалось извлечь текст из PDF-файлов.")
        return None

    embedding_model = create_embedding_model()
    vector_store = FAISS.from_documents(documents=docs, embedding=embedding_model)
    return vector_store


if "messages" not in st.session_state:
    st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "Привет! Загрузи PDF-файл и задай вопрос по содержимому документа.",
                }
    ]

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None


header_col, clear_col = st.columns([5, 1])
with header_col:
    st.title("ИИ-ассистент")

with st.container():
    uploaded_files = st.file_uploader(
        "Загрузите PDF-файл(ы)",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button("Извлечь текст из PDF", use_container_width=True):
            with st.spinner("Обрабатываю PDF и создаю поиск по документам..."):
                vector_store = create_pdf_vector_store(uploaded_files)
                if vector_store is not None:
                    st.session_state.vector_store = vector_store
                    st.success("PDF-файлы загружены и готовы к поиску.")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Введите ваш вопрос...")

if prompt:
    if st.session_state.vector_store is None:
        st.session_state.messages.append({"role": "assistant", "content": "Сначала загрузите PDF-файл и создайте индекс."})
        with st.chat_message("assistant"):
            st.markdown("Сначала загрузите PDF-файл и создайте индекс.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Ищу релевантные фрагменты..."):
            relevant_docs = search_query(prompt, st.session_state.vector_store, k=3)
            context = "\n\n".join(doc.page_content for doc in relevant_docs)

        api_key = load_api_key()
        with st.spinner("Генерирую ответ..."):
            response = answer_question(
                question=prompt,
                api_key=api_key,
                vector_store=st.session_state.vector_store,
            )

        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

        if relevant_docs:
            with st.expander("Показать найденный контекст"):
                st.text_area(
                    "Контекст из PDF",
                    value=context,
                    height=220,
                    label_visibility="collapsed",
                )
