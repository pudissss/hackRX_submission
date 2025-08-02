import os
import io
import requests
import tempfile
import asyncio
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from pydantic import BaseModel, HttpUrl

# Load environment variables from .env file
load_dotenv()

# --- Configuration & Security ---
app = FastAPI(
    title="Bajaj RX Hackathon 6.0 Submission",
    description="An API that processes a document from a URL and answers questions about it."
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
embeddings = HuggingFaceEndpointEmbeddings(
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    model="sentence-transformers/all-MiniLM-L6-v2"
)

llm = ChatGroq(
    temperature=0,
    model_name="llama3-8b-8192",
    api_key=os.getenv("GROQ_API_KEY")
)

prompt = ChatPromptTemplate.from_template(
    """
    You are an expert policy analyst. Your goal is to provide direct, factual, and user-friendly answers based *only* on the provided text.
    Instructions:
    1.  Carefully analyze the user's question and the provided context.
    2.  If the question can be answered with a "Yes" or "No", start your answer with "Yes," or "No,".
    3.  After the initial "Yes/No", concisely explain the conditions, definitions, or calculations based on the context.
    4.  If the question asks for a specific piece of information, provide it directly without a "Yes/No" prefix.
    5.  **Do not** quote the policy document. Rephrase the information in plain, easy-to-understand language.
    6.  **Do not** use introductory phrases like "According to the policy...".
    7.  If the information is not available in the context, respond with the exact phrase: "This information is not available in the provided policy document."

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """
)

# --- API Endpoint ---
@app.post("/hackrx/run", response_model=RunResponse, dependencies=[Depends(verify_token)])
async def run_submission(request: RunRequest):
    tmp_file_path = None
    try:
        response = requests.get(str(request.documents))
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(response.content)
            tmp_file_path = tmp_file.name
        
        loader = PyPDFLoader(file_path=tmp_file_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 7})

        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        tasks = [rag_chain.ainvoke(question) for question in request.questions]
        answers = await asyncio.gather(*tasks)
            
        return RunResponse(answers=answers)

    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to download document: {e}",
        )
    except Exception as e:
        print(f"An unexpected error occurred: {e}") # Debugging line
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal error occurred: {e}",
        )
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
