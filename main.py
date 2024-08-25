import os
from dotenv import load_dotenv
from utils import (
    DocumentProcessor,
    VectorStore,
    LLM,
    generate_prompt,
    num_docs,
    clear_screen,
)
from colorist import Color
from langchain_huggingface import HuggingFaceEmbeddings
import time

load_dotenv()  # Load environment variables from .env file


def main():
    start_time = time.time()
    api_key = os.getenv("API_KEY")

    ## Embedding Model
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    clear_screen()

    print(f"{Color.CYAN}Embedding model has been initialized!{Color.OFF}")
    ## Document Processing
    document_processor = DocumentProcessor("./docs")
    documents = document_processor.process_documents()
    print(f"    ► Processed {num_docs()} documents.\n")

    ## Vector Store
    vector_store = VectorStore(documents, embedding_model)
    print(f"{Color.CYAN}Vector store has been initialized!{Color.OFF}")
    end_time = time.time()
    print(
        f"  ► Time taken to initialize the vector store: {end_time - start_time:.2f} seconds\n"
    )
    input(f"{Color.GREEN}Press Enter to continue...{Color.OFF}")

    clear_screen()

    ## LLM
    llm = LLM("gemini-1.5-flash", api_key)

    while True:
        try:
            prompt = input(
                f"{Color.YELLOW}Ask me something (type 'exit' to quit): {Color.CYAN}"
            )

            ## Turn of color after prompt input
            print(Color.OFF, end="")

            embed_query = embedding_model.embed_query(prompt)
            results = vector_store.similarity_search_by_vector(embed_query, k=5)
            prompt = generate_prompt(prompt, results)
            answer = llm.answer(prompt)
            print(answer)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"{Color.RED}An error occurred: {str(e)}{Color.OFF}")

    print(f"{Color.CYAN}Thank you for using the application!{Color.OFF}")


if __name__ == "__main__":
    main()
