import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


load_dotenv()
# === CONFIG ===
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")


# === Load Vector Store ===
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

# === LLM: Groq Cloud (Llama3.1 8B) ===
llm = ChatGroq(
    model="llama-3.1-8b-instant",   
    temperature=0,
    groq_api_key=GROQ_API_KEY
)

# === RAG Chain ===
template = """You are an expert on developer documentation. Answer the question using only the provided context. Be concise and accurate.

Context:
{context}

Question: {question}

Answer:"""

prompt = PromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(f"Source: {doc.metadata.get('source_file')}\n{doc.page_content}" for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# === Interactive Chat ===
print("RAG Chat ready (Groq cloud LLM)! Type 'exit' to quit.\n")
while True:
    query = input("Your question: ")
    if query.lower() in ["exit", "quit"]:
        break
    response = chain.invoke(query)
    print(f"\nAnswer: {response}\n")