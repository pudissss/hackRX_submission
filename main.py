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
from langchain_community.embeddings import HuggingFaceEmbeddings
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
# Using a fast and effective embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
llm = ChatGroq(
    temperature=0, model_name="llama3-8b-8192", api_key=os.getenv("GROQ_API_KEY")
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
        # Retrieving a balanced number of documents for speed and context
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        # 2. Define the RAG chain for a single question
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        # 3. OPTIMIZATION: Run tasks in smaller concurrent batches to respect rate limits
        answers = []
        batch_size = 3  # Process 3 requests at a time
        for i in range(0, len(request.questions), batch_size):
            # Get the current batch of questions
            batch_questions = request.questions[i : i + batch_size]
            # Create async tasks for the current batch
            tasks = [rag_chain.ainvoke(question) for question in batch_questions]
            # Run the batch and get the results
            batch_answers = await asyncio.gather(*tasks)
            # Add the results to our final list
            answers.extend(batch_answers)

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
