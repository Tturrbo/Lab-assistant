import os
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "healthcare_dataset.csv"
API_KEY_PATH = PROJECT_ROOT / "api_key.txt"
FAISS_DB_PATH = PROJECT_ROOT / "faiss_langchain_db"


def load_api_key(path: str | os.PathLike[str] = API_KEY_PATH) -> str:
    with open(path, "r", encoding="utf-8") as file:
        return file.read().strip()


def load_data(path: str | os.PathLike[str] = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8")


def build_documents(data: pd.DataFrame) -> list[Document]:
    docs: list[Document] = []
    content_cols = list(data.columns)

    for _, row in data.iterrows():
        page_content = "\n".join([f"{col}: {row[col]}" for col in content_cols])
        docs.append(Document(page_content=page_content))

    return docs


def create_embedding_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vector_store(
    docs: list[Document],
    embedding_model: HuggingFaceEmbeddings,
    persist_directory: str | os.PathLike[str] = FAISS_DB_PATH,
) -> FAISS:
    persist_directory = str(persist_directory)

    if os.path.exists(persist_directory):
        return FAISS.load_local(
            persist_directory,
            embedding_model,
            allow_dangerous_deserialization=True,
        )

    vector_store = FAISS.from_documents(documents=docs, embedding=embedding_model)
    vector_store.save_local(persist_directory)
    return vector_store


def search_query(
    query: str,
    vector_store: FAISS,
    k: int = 5,
) -> list[Document]:
    return vector_store.similarity_search(query, k=k)


def answer_question(
    question: str,
    api_key: str | None = None,
    vector_store: FAISS | None = None,
    model_name: str = "qwen/qwen3.8-27b",
    temperature: float = 0.2,
    max_tokens: int = 1000,
) -> str:
    if api_key is None:
        api_key = load_api_key()

    if vector_store is None:
        data = load_data()
        docs = build_documents(data)
        embedding_model = create_embedding_model()
        vector_store = get_vector_store(docs[:10], embedding_model)

    results = search_query(question, vector_store, k=5)
    context = results[0].page_content

    llm = ChatGroq(
        api_key=api_key,
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    messages = [
        (
            "system",
            f"You are a helpful assistant. Use the following context to answer the user question. Context: {context}",
        ),
        ("human", question),
    ]

    ai_msg = llm.invoke(messages)
    return ai_msg.content


def main() -> None:
    data = load_data()
    docs = build_documents(data)
    embedding_model = create_embedding_model()
    vector_store = get_vector_store(docs[:10], embedding_model)

    user_query = "is the patient child, teen, or adult? and what's her bloodtype?"
    result = search_query(user_query, vector_store, k=5)[0].page_content
    print("\n--- Result from user query ---")
    print(result)

    api_key = load_api_key()
    answer = answer_question(
        user_query,
        api_key=api_key,
        vector_store=vector_store,
    )
    print("\n--- Answer ---")
    print(answer)


if __name__ == "__main__":
    main()
