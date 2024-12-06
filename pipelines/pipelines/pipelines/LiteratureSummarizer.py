"""
Title: Advanced Automatic Literature Summarizer (Multiple Formats)
Author: MCPTechTeam
Author URL: https://genai.mcp.org
Version: 2.0
License: MIT
Required OpenWebUI Version: 0.3.32
"""

# System imports
import asyncio
import logging
import os
from typing import Optional, List, Dict, Callable, Any, Awaitable

# Libraries available to OpenWebUI
from pydantic import BaseModel as PydanticBaseModel, Field

# Third-party libraries for file handling
import fitz  # PyMuPDF
import docx2txt
from bs4 import BeautifulSoup

# OpenWebUI imports
from open_webui.apps.retrieval.vector.connector import VECTOR_DB_CLIENT
from open_webui.apps.retrieval.main import app as rag_app
from open_webui.utils.misc import get_last_user_message
from open_webui.main import generate_chat_completions

from open_webui.apps.webui.models.models import Models

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ChromaDB client (assuming it's available without additional dependencies)
CHROMA_CLIENT = VECTOR_DB_CLIENT.client

# Embedding function (placeholder)
EMBEDDING_FUNCTION = rag_app.state.EMBEDDING_FUNCTION

# Summary Types
SUMMARY_TYPES: Dict[str, Dict[str, Any]] = {
    "brief_abstract": {
        "description": "A brief abstract (approx. 50-100 words).",
        "max_tokens": 150,
    },
    "detailed_summary": {
        "description": "A detailed summary with bullet-point key findings and methodology overview (approx. 200-300 words).",
        "max_tokens": 300,
    },
    "key_takeaways": {
        "description": "Key takeaways for quick insight into the document's main contributions.",
        "max_tokens": 200,
    },
}

# Prompt templates
PROMPT_TEMPLATES: Dict[str, str] = {
    "brief_abstract": "Summarize the following text into a concise abstract of approximately 100 words.",
    "detailed_summary": "Provide a detailed summary of the following text, including key findings and an overview of the methodology. Use bullet points where appropriate.",
    "key_takeaways": "List the key takeaways from the following text for quick insight into the main contributions.",
}

# Supported Languages (currently English)
SUPPORTED_LANGUAGES: List[str] = ["en"]

# Utility functions for file handling
def extract_text_from_txt(file_path: str) -> str:
    """Extract text from a plain text file.

    Args:
        file_path (str): Path to the text file.

    Returns:
        str: Extracted text from the file.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading text file: {e}")
        return ""

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file using PyMuPDF.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        str: Extracted text from the PDF.
    """
    try:
        text = ""
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()
        return text
    except Exception as e:
        logger.error(f"Error reading PDF file: {e}")
        return ""

def extract_text_from_docx(file_path: str) -> str:
    """Extract text from a DOC/DOCX file using docx2txt.

    Args:
        file_path (str): Path to the DOC/DOCX file.

    Returns:
        str: Extracted text from the document.
    """
    try:
        text = docx2txt.process(file_path)
        return text
    except Exception as e:
        logger.error(f"Error reading DOC/DOCX file: {e}")
        return ""

def extract_text_from_html(file_path: str) -> str:
    """Extract text from an HTML file using BeautifulSoup.

    Args:
        file_path (str): Path to the HTML file.

    Returns:
        str: Extracted text from the HTML.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
            text = soup.get_text(separator="\n")
        return text
    except Exception as e:
        logger.error(f"Error reading HTML file: {e}")
        return ""

def extract_text_from_file(file_path: str) -> str:
    """Extract text from a file based on its extension.

    Args:
        file_path (str): Path to the file.

    Returns:
        str: Extracted text from the file.
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == ".txt":
        return extract_text_from_txt(file_path)
    elif file_extension == ".pdf":
        return extract_text_from_pdf(file_path)
    elif file_extension in [".doc", ".docx"]:
        return extract_text_from_docx(file_path)
    elif file_extension in [".html", ".htm"]:
        return extract_text_from_html(file_path)
    else:
        logger.error(f"Unsupported file format: {file_extension}")
        return ""

class BaseModel(PydanticBaseModel):
    """Base model class with arbitrary types allowed."""

    class Config:
        arbitrary_types_allowed = True

class SummarizerResponse(BaseModel):
    """Response model for the summarizer.

    Attributes:
        summary (str): The generated summary text.
    """

    summary: str

class LiteratureSummarizer(BaseModel):
    """Summarizes a given document text using a language model.

    Attributes:
        document_text (str): The text to summarize.
        model (str): The identifier of the language model to use.
        user (Any): The user object from OpenWebUI.
        summary_type (str): The type of summary to generate.
        language (str): The language of the document.
    """

    document_text: str
    model: str
    user: Any
    summary_type: str = "brief_abstract"
    language: str = "en"

    async def summarize(self) -> Optional[SummarizerResponse]:
        """Generates a summary of the document text.

        Returns:
            Optional[SummarizerResponse]: The summarization result or None if failed.
        """
        if self.language not in SUPPORTED_LANGUAGES:
            logger.error(f"Language '{self.language}' not supported.")
            return None

        prompt_template: str = PROMPT_TEMPLATES.get(
            self.summary_type, PROMPT_TEMPLATES["brief_abstract"]
        )
        prompt: str = f"{prompt_template}\n\n{self.document_text}"

        messages: List[Dict[str, str]] = [{"role": "user", "content": prompt}]
        request: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "max_tokens": SUMMARY_TYPES[self.summary_type]["max_tokens"],
            "keep_alive": "10s",
        }

        logger.debug(f"Summarization request: {request}")

        try:
            resp = await generate_chat_completions(request, user=self.user)
            logger.debug(f"Language model response: {resp}")

            if resp and "choices" in resp and len(resp["choices"]) > 0:
                content: str = resp["choices"][0]["message"]["content"]
                return SummarizerResponse(summary=content.strip())
            else:
                logger.error("No choices returned from the language model.")
                return None
        except Exception as e:
            logger.error(f"Error during language model generation: {e}")
            return None

class LiteratureSummaryTool(BaseModel):
    """Manages the summarization process.

    Attributes:
        model (str): The identifier of the language model to use.
        user (Any): The user object from OpenWebUI.
    """

    model: str
    user: Any

    async def process_document(
        self, file_path: str, summary_type: str
    ) -> Optional[SummarizerResponse]:
        """Processes a document and generates a summary.

        Args:
            file_path (str): The path to the document file.
            summary_type (str): The type of summary to generate.

        Returns:
            Optional[SummarizerResponse]: The summarization result or None if failed.
        """
        logger.debug(f"Processing document: {file_path}")
        document_text: str = extract_text_from_file(file_path)

        if not document_text:
            logger.error("Failed to extract text from the document.")
            return None

        logger.debug(
            f"Document text extracted. Length: {len(document_text)} characters."
        )

        try:
            summarizer = LiteratureSummarizer(
                document_text=document_text,
                model=self.model,
                user=self.user,
                summary_type=summary_type,
            )

            response = await summarizer.summarize()
            return response
        except Exception as e:
            logger.error(f"Error during summarization: {e}")
            return None

#########################
# OpenWebUI Integration
#########################

class Tools:
    """Integration class for the summarizer tool with OpenWebUI.

    Attributes:
        valves (Valves): Configuration options for the tool.
    """

    class Valves(BaseModel):
        """Configuration options for the tool."""

        pass

    class UserValves(BaseModel):
        """User-specific configuration options."""

        pass

    def __init__(self):
        self.valves = self.Valves()
        pass

    async def inlet(
        self,
        body: dict,
        __user__: Optional[dict],
        __model__: Optional[dict],
        __event_emitter__: Callable[[Any], Awaitable[None]],
    ) -> dict:
        """Processes incoming messages to intercept summarization commands.

        Args:
            body (dict): The request body containing messages.
            __user__ (Optional[dict]): The user object from OpenWebUI.
            __model__ (Optional[dict]): The model object from OpenWebUI.
            __event_emitter__ (Callable): The event emitter function.

        Returns:
            dict: The modified request body.
        """
        # Extract user message to check for summarization command
        user_message: Optional[str] = get_last_user_message(body["messages"])
        if not user_message:
            return body

        # Check if the user is requesting a summarization
        if user_message.startswith("!summarize"):
            # Parse command arguments
            args: List[str] = user_message.split(maxsplit=2)
            if len(args) < 2:
                # Inform the user about incorrect usage
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": (
                                "Usage: !summarize <file_path> [summary_type]\n"
                                "Supported formats: TXT, PDF, DOC, DOCX, HTML"
                            ),
                            "done": True,
                        },
                    }
                )
                # Remove the last message to prevent default response
                if body["messages"]:
                    body["messages"] = body["messages"][:-1]
                return body

            file_path: str = args[1]
            summary_type: str = args[2] if len(args) > 2 else "brief_abstract"

            # Validate summary type
            if summary_type not in SUMMARY_TYPES:
                summary_type = "brief_abstract"

            # Initialize the summarization tool
            summarization_tool = LiteratureSummaryTool(
                model=__model__["id"],
                user=__user__,
            )

            # Perform summarization
            try:
                response = await summarization_tool.process_document(
                    file_path, summary_type
                )
                if response:
                    # Return the summary to the user
                    body["messages"].append(
                        {
                            "role": "assistant",
                            "content": response.summary,
                        }
                    )
                    # Prevent the assistant from generating a default response
                    if body["messages"]:
                        body["messages"] = body["messages"][:-1]
                else:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": "Failed to generate summary. The document may be unsupported or empty.",
                                "done": True,
                            },
                        }
                    )
                    # Prevent the assistant from generating a default response
                    if body["messages"]:
                        body["messages"] = body["messages"][:-1]
            except Exception as e:
                logger.error(f"Error during summarization: {e}")
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Error: {e}",
                            "done": True,
                        },
                    }
                )
                # Prevent the assistant from generating a default response
                if body["messages"]:
                    body["messages"] = body["messages"][:-1]

        return body

    async def outlet(
        self,
        body: dict,
        __user__: Optional[dict],
        __model__: Optional[dict],
        __event_emitter__: Callable[[Any], Awaitable[None]],
    ) -> dict:
        """Processes outgoing messages (no changes made in this tool).

        Args:
            body (dict): The response body containing messages.
            __user__ (Optional[dict]): The user object from OpenWebUI.
            __model__ (Optional[dict]): The model object from OpenWebUI.
            __event_emitter__ (Callable): The event emitter function.

        Returns:
            dict: The unmodified response body.
        """
        # No modifications needed on the outlet
        return body
