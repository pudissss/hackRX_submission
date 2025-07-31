import asyncio
import os
import tempfile
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
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from pydantic import BaseModel, HttpUrl

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
    """Dependency function to verify the Bearer token."""
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
# Using Google's embedding model to match the LLM
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY")
)

# UPDATED: Using Gemini 2.0 Flash-Lite for the highest request limit
llm = ChatGoogleGenerativeAI(
    temperature=0,
    model="gemini-2.0-flash-lite",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

# A more direct prompt for concise, user-friendly answers
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
            chunk_size=1500, chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        # 2. Define the RAG chain for a single question
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        # 3. Run all tasks concurrently for maximum speed
        tasks = [rag_chain.ainvoke(question) for question in request.questions]
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
        # Clean up the temporary file after the request is done
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
