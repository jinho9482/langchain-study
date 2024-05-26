import os
import logging
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from fastapi import FastAPI, HTTPException, Request, Body
from fastapi.responses import JSONResponse
import asyncio
from langserve import add_routes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get the Google API key from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# print(GOOGLE_API_KEY)
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")

# Initialize the ChatGoogleGenerativeAI instance


async def initialize_model():
    model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)   
    if model.async_client is None:
        raise ValueError("ChatGoogleGenerativeAI async_client is not initialized")
    return model
model = asyncio.run(initialize_model())

# Define the prompt template and parser
prompt_template = PromptTemplate.from_template("Translate the {text} into {language}:")
parser = StrOutputParser()

# Create the chain with the prompt template, LLM, and parser
chain = prompt_template | model | parser

# Initialize the FastAPI app
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using LangChain's Runnable interfaces",
)

# add_routes(
#     app,
#     chain,
#     path="/chain",
# )


# # Add the routes to the FastAPI app
@app.post("/chain")
async def stream_log(data: dict = Body(...)):
    try:
        language = data.get("language")
        text = data.get("text")
        if not language or not text:
            raise HTTPException(status_code=400, detail="Missing 'language' or 'text' in request body")
        
        result = chain.invoke({"language": language, "text": text})
        return {"result": result}
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# # Define an endpoint to test the setup
# @app.get("/healthcheck")
# async def healthcheck():
#     return {"status": "ok"}

# # Enhanced error handling for POST requests
# @app.exception_handler(Exception)
# async def validation_exception_handler(request: Request, exc: Exception):
#     logger.error(f"Unexpected error: {exc}")
#     return JSONResponse(
#         status_code=500,
#         content={"message": "Internal Server Error"},
#     )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
