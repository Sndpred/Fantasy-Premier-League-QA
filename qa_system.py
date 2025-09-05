import os
import re
from dotenv import load_dotenv
from document_processor import DocumentProcessor
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

class QASystem:
    def __init__(self, config=None):
        """Initialize the Q&A system with configuration dictionary"""
        
        # Default settings
        default_config = {
            "show_validation": True,
            "verbose": True,
            "llm_model": "gpt-3.5-turbo",
            "temperature": 0.1,
            "retrieval_k": 3,
            "data_year": "2019-20"
        }
        
        # Merge defaults with user-provided config
        self.config = {**default_config, **(config or {})}

        # Assign frequently used configs as attributes
        self.show_validation = self.config["show_validation"]
        self.verbose = self.config["verbose"]
        self.data_year = self.config["data_year"]

        # Core components
        self.processor = DocumentProcessor()
        self.llm = ChatOpenAI(
            model=self.config["llm_model"],
            temperature=self.config["temperature"],
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.vectorstore = None
        self.qa_chain = None
    
    def setup_document(self, pdf_path):
        """Load and process document, create vector store"""
        if self.verbose:
            print("Setting up document processing...")
        
        # Load and chunk document
        documents = self.processor.load_document(pdf_path)
        chunks = self.processor.split_documents(documents)
        
        # Create embeddings
        if self.verbose:
            print(" Creating embeddings (this will use OpenAI credits)...")
        self.vectorstore = self.processor.create_embeddings(chunks)
        
        # Custom prompt
        prompt_template = f"""You are a Fantasy Premier League (FPL) expert assistant. 
Use the following pieces of context to answer the question about player data, teams, positions, costs, and points.

IMPORTANT: The data you have is from the {self.data_year} FPL season only. 
If asked about other seasons or current data, clearly state that your information is limited to the {self.data_year} season.

Context: {{context}}

Question: {{question}}

Please provide a helpful and accurate answer based on the FPL data provided. 
If the question asks about data outside the {self.data_year} season, clearly state this limitation. 
Include specific player names, costs, and points when relevant.

Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"]
        )
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": self.config["retrieval_k"]}
            ),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        if self.verbose:
            print("✅ Q&A system ready!")
    
    def validate_question(self, question):
        """Check if question might be outside our data scope"""
        warnings = []
        
        # Look for years beyond dataset
        years = re.findall(r'\b(20\d{2})\b', question)
        for year in years:
            if int(year) > int(self.data_year.split("-")[-1]):
                warnings.append(f"Question mentions {year}, but our data is from {self.data_year}")
        
        # Look for "recent season" terms
        recent_patterns = [r'202[3-9]', r'2024[/-]202[5-9]', r'current', r'\bnow\b', 
                           r'\btoday\b', r'\blatest\b', r'\brecent\b', r'\bthis season\b']
        for pattern in recent_patterns:
            if re.search(pattern, question.lower()):
                warnings.append(f"Question asks about current/recent data, but our dataset is from {self.data_year}")
                break
        
        # Players not in 2019-20 season (examples)
        modern_players = ['haaland', 'nunez', 'antony', 'casemiro', 'tchouameni']
        for player in modern_players:
            if player in question.lower():
                warnings.append(f"Question asks about {player.title()}, who likely wasn't in the {self.data_year} season")
        
        return warnings
    
    def ask_question(self, question):
        """Ask a question and get an answer"""
        if not self.qa_chain:
            return "Please setup a document first!"
        
        # Show validation warnings
        if self.show_validation:
            warnings = self.validate_question(question)
            if warnings:
                print("\n DATA SCOPE WARNINGS:")
                for warning in warnings:
                    print(f"   - {warning}")
                print(f"    Our data covers: {self.data_year} FPL season only\n")
        
        if self.verbose:
            print(f"\n Question: {question}")
            print("Searching for relevant information...")
        
        # Get answer
        result = self.qa_chain.invoke({"query": question})
        answer = result['result']
        
        print(f"Answer: {answer}")
        
        # Reminder if question is out of scope
        if self.show_validation and self.validate_question(question):
            print(f"\nREMINDER: This answer is based on {self.data_year} data only!")
        
        if self.verbose:
            print(f"\nBased on {len(result['source_documents'])} document chunks")
        
        return answer


def test_qa_system():
    """Test the complete Q&A system"""
    
    # Example custom configuration
    config = {
        "show_validation": True,
        "verbose": True,
        "llm_model": "gpt-4",   # Switch easily
        "temperature": 0.2,
        "retrieval_k": 5,
        "data_year": "2019-20"
    }
    
    qa_system = QASystem(config=config)
    
    # Find PDF file
    documents_folder = "documents"
    pdf_files = [f for f in os.listdir(documents_folder) if f.endswith('.pdf')]
    
    if not pdf_files:
        print("No PDF files found in documents folder")
        return
    
    pdf_path = os.path.join(documents_folder, pdf_files[0])
    
    # Setup the system
    qa_system.setup_document(pdf_path)
    
    # Test questions
    test_questions = [
        "What is the most expensive player and their cost?",
        "List some Arsenal players and their positions",
        "What is Haaland's price in 2025/2026 season?",
        "Who is the best Manchester United player in 2024/2025 season?",
        "Who are the goalkeepers in the data?"
    ]
    
    print("\n" + "="*60)
    print("TESTING Q&A SYSTEM")
    print("="*60)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\nTEST {i}/5:")
        print("-" * 60)
        qa_system.ask_question(question)
        print("-" * 60)


if __name__ == "__main__":
    test_qa_system()
