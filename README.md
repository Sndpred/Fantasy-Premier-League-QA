**Fantasy Premier League (FPL) Player Data Q&A System**

This project is a Retrieval-Augmented Generation (RAG) system that allows you to ask questions about Fantasy Premier League player data from the 2019/20 season and get answers based on a specific PDF document. It demonstrates practical experience with key machine learning and AI frameworks, including OpenAI and LangChain.

The system is built as a web application using Streamlit, making it easy to interact with via a simple user interface.

  **Key Features:**

  **PDF Data Ingestion:** Loads and processes a PDF document containing FPL player data.

  **Vector Embeddings:** Converts document text into numerical vector representations using OpenAI's embedding models.

  **Semantic Search:** Finds the most relevant information from the document chunks based on the meaning of your question, not just keywords.

  **AI-powered Answers:** Uses a Large Language Model (LLM) from OpenAI to generate a human-readable answer based on the retrieved context.

  **Context-Aware:** The system is specifically trained on data from the 2019/20 season and includes a custom prompt to handle out-of-scope questions, such as those about more recent seasons.

  **User-Friendly Interface:** The Streamlit application offers a straightforward and intuitive way to ask questions and view answers.

  **Technologies Used**

  **Python:** The core programming language for the project.

  **LangChain:** The framework that orchestrates the entire RAG workflow.

  **OpenAI:** Provides the LLM (GPT-3.5 Turbo or GPT-4) and embedding models (text-embedding-3-small).

  **Streamlit:** A powerful tool for creating and deploying data applications.

  **FAISS:** An efficient library for similarity search, used as our in-memory vector store.

  **PyPDF:** A Python library for working with PDF files.

  **Installation & Setup**

  **Clone the repository:**

    git clone <your-repo-url>
    cd <your-repo-folder>

  **Set up a Python Virtual Environment:**

    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate

  **Install dependencies:**
    This project uses a requirements.txt file to manage dependencies.

    pip install -r requirements.txt

  **Set up your OpenAI API Key:**
    Create a new file named' .env' in the root directory of the project. Add API key in the following format:

    OPENAI_API_KEY="your-openai-api-key-here"

    Note: Keep this file private and do not commit it to your repository.

  **How to Run the Application**

    Ensure you are in the virtual environment and have completed the setup steps.

    Run the Streamlit application:

    streamlit run app.py

    The application will automatically open in your web browser. You can then interact with the Q&A system to ask questions about the FPL player data.

  **Project Structure**

  **app.py:** The main Streamlit application file that handles the user interface.

  **document_processor.py:** Contains the logic for loading, splitting, and embedding documents.

  **qa_system.py:** The core of the RAG system, which orchestrates the document retrieval and question-answering process.

  **documents/:** A folder containing the PDF file (2019-20-FPL-Player-prices-by-club-070819.pdf) used as the knowledge base.

  **requirements.txt:** Lists all Python package dependencies.

  **.env:** (Not committed to Git) Your private file for storing API keys.

  **.gitignore:** Specifies files and directories to be ignored by Git.

  **Important Note on OpenAI Usage**

This project makes a single API call to OpenAI for embeddings when the document is first loaded. This is done to create a local, in-memory vector store that can be queried multiple times without additional embedding costs. Subsequent questions will incur a small cost for the LLM call to generate the answer.
