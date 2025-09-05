import streamlit as st
import os
from qa_system import QASystem

def main():
    st.title("FPL(2019/20) Player Data Q&A System")
    st.markdown("Ask questions about Fantasy Premier League player data from 2019/20 season below!")
    
    # Initialize session state
    if 'qa_system' not in st.session_state:
        st.session_state.qa_system = None
        st.session_state.document_loaded = False
        st.session_state.current_question = ""
    
    # Sidebar for document loading
    with st.sidebar:
        st.header("Document Setup")
        
        if st.button("Load FPL Document", type="primary"):
            with st.spinner("Loading document and creating embeddings..."):
                try:
                    # Use config-driven QASystem
                    qa_system = QASystem(config={
                        "show_validation": True,   # Show validation warnings in Streamlit
                        "verbose": False,          # Avoid console spam
                        "model": "gpt-3.5-turbo",
                        "temperature": 0.1,
                        "retrieval_k": 3,
                        "data_year": "2019-20"
                    })
                    
                    # Find PDF file
                    documents_folder = "documents"
                    pdf_files = [f for f in os.listdir(documents_folder) if f.endswith('.pdf')]
                    
                    if pdf_files:
                        pdf_path = os.path.join(documents_folder, pdf_files[0])
                        qa_system.setup_document(pdf_path)
                        
                        st.session_state.qa_system = qa_system
                        st.session_state.document_loaded = True
                        st.success(f"Loaded: {pdf_files[0]}")
                    else:
                        st.error("No PDF files found in documents folder")
                        
                except Exception as e:
                    st.error(f"Error loading document: {str(e)}")
        
        if st.session_state.document_loaded:
            st.success("✅ Document ready for questions!")
            st.info("Each question costs ~$0.01-0.02 in OpenAI credits")
    
    # Main interface
    if st.session_state.document_loaded:
        st.header("Ask Your Questions")
        
        # Suggested questions
        st.subheader("Try these questions:")
        suggestions = [
            "Who is the most expensive midfielder?",
            "List all Manchester United players with price.",
            "Which goalkeeper scored the most points?",
            "Who are the cheapest forwards?",
            "Which Chelsea players cost more than 8.0?",
            "Compare Salah and Rooney.",
            "What Arsenal defenders are available?"
        ]
        
        cols = st.columns(2)
        for i, suggestion in enumerate(suggestions):
            col = cols[i % 2]
            if col.button(suggestion, key=f"suggestion_{i}"):
                st.session_state.current_question = suggestion
        
        # Question input
        question = st.text_input(
            "Your question:", 
            value=st.session_state.get('current_question', ''),
            placeholder="Type your question about FPL players..."
        )
        
        if st.button("Get Answer", type="primary"):
            if question:
                with st.spinner("Analyzing document and generating answer..."):
                    try:
                        qa_system = st.session_state.qa_system

                        # Capture validation warnings
                        if qa_system.show_validation:
                            warnings = qa_system.validate_question(question)
                            if warnings:
                                st.warning("⚠️ **Data Scope Warnings:**")
                                for warning in warnings:
                                    st.write(f"• {warning}")
                                st.write(f" **Our data covers:** {qa_system.data_year} FPL season only")
                        
                        # Ask question using the QA system (config handles validation now)
                        answer = qa_system.ask_question(question)
                
                        st.subheader("Answer:")
                        st.write(answer)
                        
                        if qa_system.show_validation and warnings:
                            st.info(f" **Reminder:** This answer is based on {qa_system.data_year} data only!")
                
                        st.caption(" Estimated cost: ~$0.01-0.02")
                
                    except Exception as e:
                        st.error(f"Error generating answer: {str(e)}")
            else:
                st.warning("Please enter a question!")
    
    else:
        st.info("Click 'Load FPL Document' in the sidebar to get started!")
        
        # Show what the system does
        st.header(" How This Works")
        st.markdown("""
        **This Q&A system demonstrates:**
        
        1. **Document Processing** - Loads and chunks PDF into smaller, searchable pieces.
        2. **OpenAI Embeddings** - Creates vector representations of text for semantic search.
        3. **Semantic Search** - Finds relevant content based on meaning, not just keywords.
        4. **LangChain Integration** - Orchestrates the AI workflow for retrieval and answer generation.
        5. **Natural Language Generation** - Generates human-readable answers using LLM.
        
        **Technologies Used:**
        - OpenAI GPT-4 for answer generation (Can be switched easily in config).
        - OpenAI text-embedding-3-small for document search
        - LangChain for workflow orchestration
        - Streamlit for the web interface
        - PyPDF for document processing
        """)

if __name__ == "__main__":
    main()
