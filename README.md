<!-- @format -->

# Retrieval-Augmented Generation (RAG) Project

This project is a Retrieval-Augmented Generation (RAG) system that utilizes LangChain for managing documents, Chroma for vector-based document retrieval, and Google Gemini Flash for the Large Language Model (LLM). The system is designed to process PDF documents, split them into manageable chunks, embed them into a vector store, and then use these embeddings to perform efficient similarity searches. It finally generates answers to user queries based on the most relevant information retrieved from the document collection.

## Features

- **Document Processing:** Efficiently loads and splits PDF documents into smaller chunks for better retrieval and analysis.
- **Vector Store:** Utilizes Chroma for storing document embeddings and performing similarity searches.
- **LLM Integration:** Leverages Google Gemini Flash to generate high-quality responses based on retrieved documents.
- **Interactive Query System:** Allows users to input queries, retrieve relevant document sections, and generate answers.

## Prerequisites

- **Python 3.8+**
- **Google API Key:** Required for accessing the Google Gemini Flash model. You can obtain an API key from the Google Cloud Console.

## Setup Instructions

### 1. Clone the Repository

```bash!
git clone https://github.com/neutrino225/rag-project.git
cd rag-project
```

### 2. Create and Activate a Virtual Environment

```bash!
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

Make sure to install the required packages from requirements.txt:

```bash!
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a .env file in the project root and add your Google API key:

```
API_KEY=your_google_api_key_here
```

### 5. Prepare Your Document Directory

Place your PDF documents in a directory named docs in the project root.

### 6. Run the Application

```bash!
python main.py
```

Follow the on-screen instructions to interact with the system.

## Project Structure

- main.py: The main entry point of the application.
- utils.py: Contains utility classes for document processing, vector store management, and LLM interaction.
- requirements.txt: List of dependencies required to run the project.

## Notes

- The application is designed to be interactive, allowing you to input queries and retrieve answers dynamically.
- The API key is essential for using the LLM; without it, the application will not function.

## License

This project is licensed under the MIT License.
