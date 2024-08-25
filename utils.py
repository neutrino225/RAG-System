import os
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma


class DocumentProcessor:
    def __init__(self, directory_path: str):
        self.directory_path = directory_path

    def load_pdf_directory(self) -> List:
        """Load a directory of PDF files."""
        loader = PyPDFDirectoryLoader(self.directory_path)
        return loader.load()

    def split_text(
        self, documents: List, chunk_size: int = 1000, chunk_overlap: int = 200
    ) -> List:
        """Split a list of Document objects into chunks of text."""
        text_splitter = RecursiveCharacterTextSplitter(
            is_separator_regex=False,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return text_splitter.split_documents(documents)

    def process_documents(self) -> List:
        """Load the documents, split them into chunks, and return the chunks."""
        documents = self.load_pdf_directory()
        return self.split_text(documents)


class VectorStore:
    def __init__(self, documents, embeddings_model: callable):
        self.vector_store = Chroma.from_documents(
            documents=documents, embedding=embeddings_model
        )

    def similarity_search_by_vector(self, embedding: List[float], k=5) -> List:
        """Perform a similarity search using the provided embedding."""
        return self.vector_store.similarity_search_by_vector(embedding=embedding, k=k)

    def similarity_search_by_vector_with_relevance_scores(
        self, embedding: List[float], k=5
    ):
        """Perform a similarity search with relevance scores using the provided embedding."""
        return self.vector_store.similarity_search_by_vector_with_relevance_scores(
            embedding=embedding, k=k
        )


class LLM:
    def __init__(self, model="gemini-1.5-flash", google_api_key=None):
        if not google_api_key:
            raise ValueError("Google API key must be provided.")

        self.model = model
        self.llm = GoogleGenerativeAI(model=model, google_api_key=google_api_key)

    def answer(self, prompt, stream=False, **kwargs):
        """
        Generate an answer based on the provided prompt.

        Args:
            prompt (str): The prompt or question to be answered.
            stream (bool): Whether to stream the output. Defaults to False.
            **kwargs: Additional parameters to pass to the LLM.

        Returns:
            str or generator: The generated answer or a generator if streaming is enabled.
        """
        if not prompt:
            raise ValueError("Prompt must be provided.")

        if stream:
            return self.llm.stream(prompt, **kwargs)
        return self.llm.invoke(prompt, **kwargs)


def generate_prompt(document, question):
    """Generate a refined prompt for the LLM using an enhanced template."""
    template = """
        You are an expert question-answering assistant with access to detailed documents. 
        Given the following document, answer the question based on the most relevant information found within it.
        
        Document:
        {document}
        
        Question:
        {question}
        
        Your task:
        - Provide a concise and clear answer to the question.
        - Highlight the specific text from the document that led to your answer.
        - If possible, include the page number or section where the information was found.
        - If you are unsure or the document does not contain relevant information, respond with "I don't know".
        - Format the answer in a structured manner: 
            Answer: <your answer>

            Supporting Text: <text from document>
            
            Source: <source or page number if available>
    """
    prompt_template = PromptTemplate.from_template(template)
    return prompt_template.format(document=document, question=question)


def num_docs():
    """Get the number of documents in the specified directory."""
    directory = "./docs"
    return len(os.listdir(directory))


def clear_screen():
    os.system("clear" if os.name != "nt" else "cls")
    return
