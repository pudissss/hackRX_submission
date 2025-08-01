import asyncio
import os
import tempfile
from operator import itemgetter
from typing import List

import requests
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from pydantic import BaseModel, HttpUrl
from rank_bm25 import BM25Okapi

# Load environment variables from .env file
load_dotenv()

# --- Configuration & Security ---
app = FastAPI(
    title="Bajaj RX Hackathon 6.0 Submission",
    description="An API that processes a document from a URL and answers questions about it.",
)
security = HTTPBearer()
EXPECTED_TOKEN = "9e0d05561dcac37f88a694cb07f2b08f55b6487433f612155d30b883cbeb6008"


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.scheme != "Bearer" or credentials.credentials != EXPECTED_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authentication token",
        )
    return credentials


# --- Pydantic Schemas for API ---
class RunRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]


class RunResponse(BaseModel):
    answers: List[str]


# --- Core RAG Components (loaded once) ---
embeddings = HuggingFaceEndpointEmbeddings(
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    model="sentence-transformers/all-MiniLM-L6-v2",
)

llm = ChatGroq(
    temperature=0, model_name="gemma2-9b-it", api_key=os.getenv("GROQ_API_KEY")
)

prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful Q&A assistant. Your task is to answer the user's question based *only* on the provided context.

    Instructions:
    1.  Provide a direct, concise answer.
    2.  Synthesize the information into a single, clean paragraph. Do not use bullet points or lists.
    3.  Do NOT add any conversational phrases or introductory text like "Based on the context...".
    4.  If the information is not available in the context, respond with the exact phrase: "This information is not available in the provided policy document."

    CONTEXT:
    {context}

    QUESTION:
    {question}

    CONCISE ANSWER:
    """
)


# --- API Endpoint ---
@app.post(
    "/hackrx/run", response_model=RunResponse, dependencies=[Depends(verify_token)]
)
async def run_submission(request: RunRequest):
    tmp_file_path = None
    try:
        # 1. Download and process the document
        response = requests.get(str(request.documents))
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(response.content)
            tmp_file_path = tmp_file.name

        loader = PyPDFLoader(file_path=tmp_file_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100
        )
        splits = text_splitter.split_documents(docs)

        # 2. Create both Semantic and Keyword retrievers
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        tokenized_corpus = [doc.page_content.split(" ") for doc in splits]
        bm25 = BM25Okapi(tokenized_corpus)

        def keyword_retriever(query):
            tokenized_query = query.split(" ")
            bm25_scores = bm25.get_scores(tokenized_query)
            top_n_indices = sorted(
                range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
            )[:3]
            return [splits[i] for i in top_n_indices]

        # 3. Define the RAG chain for a single question using Hybrid Search
        async def hybrid_retrieve(question):
            semantic_docs = await semantic_retriever.ainvoke(question)
            keyword_docs = keyword_retriever(question)
            combined_docs = {
                doc.page_content: doc for doc in semantic_docs + keyword_docs
            }
            return list(combined_docs.values())

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {
                "context": itemgetter("question")
                | RunnableLambda(hybrid_retrieve)
                | RunnableLambda(format_docs),
                "question": itemgetter("question"),
            }
            | prompt
            | llm
            | StrOutputParser()
            | (
                lambda x: x.strip()
            )  # ADDED: Clean up whitespace and newlines from the final output
        )

        # 4. Run all tasks concurrently
        tasks = [rag_chain.ainvoke({"question": q}) for q in request.questions]
        answers = await asyncio.gather(*tasks)

        return RunResponse(answers=answers)

    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to download document: {e}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal error occurred: {e}",
        )
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
