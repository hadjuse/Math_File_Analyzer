import streamlit as st
import fitz  # PyMuPDF
from pdf2image import convert_from_path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
import chromadb
from langsmith import Client
import tempfile
import pandas as pd
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = __import__('pysqlite3')
# Début du CSS personnalisé
st.markdown("""
<style>
/* Style personnalisé pour les équations */
.katex {
    font-size: 1.3em !important;
    padding: 0.8em !important;
    margin: 1em 0 !important;
    background-color: #f8f9fa !important;
    border-radius: 8px !important;
    border: 1px solid #dee2e6 !important;
    overflow-x: auto !important;
}

/* Amélioration de l'espacement des formules */
.katex-display {
    margin: 1.5em 0 !important;
    padding: 1em !important;
}

/* Style cohérent pour le texte Markdown */
.stMarkdown {
    font-size: 1.1em !important;
    line-height: 1.6 !important;
}

/* Conteneur pour les blocs de code */
.stCodeBlock {
    margin: 1.5em 0 !important;
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)
def render_markdown_table(content):
    """Render Markdown tables from content."""
    if re.search(r'^(\|.+\|)(\n\|[- :|]+\|)(\n\|.+\|)*$', content.strip(), flags=re.MULTILINE):
        st.markdown(content.strip())
        return True
    return False

def render_latex(content):
    """Render LaTeX content."""
    latex_content = re.sub(r'^\$+|\\[()\[\]]', '', content.strip())
    try:
        st.latex(latex_content)
    except:
        st.code(content)

def render_text(content):
    """Render text content."""
    st.markdown(content)

def render_math(content):
    """Intelligent math rendering function."""
    parts = re.split(r'((?:^\|.*\|$\n)+)', content, flags=re.MULTILINE)
    
    for part in parts:
        if not part:
            continue
        
        if render_markdown_table(part):
            continue
        
        subparts = re.split(r'(\\\(.*?\\\)|\\\[.*?\\\]|\$\$.*?\$\$|\$.*?\$|\\begin\{.*?\}.*?\\end\{.*?\})', 
                            part, 
                            flags=re.DOTALL)
        
        for subpart in subparts:
            if not subpart:
                continue
            
            if re.match(r'(\\\(.*?\\\)|\\\[.*?\\\]|\$\$.*?\$\$|\$.*?\$|\\begin\{.*?\}.*?\\end\{.*?\})', 
                        subpart, 
                        flags=re.DOTALL):
                render_latex(subpart)
            else:
                render_text(subpart)
# Configuration
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Math PDF QA with Mistral"
client = Client()
MISTRAL_API_KEY = st.secrets["MISTRAL_API_KEY"]

def process_pdf(pdf_path):
    """Process PDF and return both text content and rendered images"""
    # Convert PDF to images
    images = convert_from_path(pdf_path, dpi=300)
    
    # Traditional text processing for QA
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(pages)
    
    embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=MISTRAL_API_KEY)
    vector_store = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    return vector_store, images

from langchain.chains import LLMChain
from langchain.prompts import PipelinePromptTemplate

def initialize_qa_chain(vector_store):
    llm = ChatMistralAI(
        model="mistral-large-latest",
        mistral_api_key=MISTRAL_API_KEY,
        temperature=0.2
    )

    # First prompt: Generate initial answer
    main_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are a math expert. Generate an answer using:
        - LaTeX ($...$) for equations
        - Markdown tables
        - Statistical notation
        
        Context: {context}
        Question: {question}
        
        Initial Answer:
        """
    )

    # Second prompt: Refine the answer
    refiner_prompt = PromptTemplate(
        input_variables=["context", "question", "initial_answer"],
        template="""
        Refine this answer maintaining format:
        - Keep LaTeX equations
        - Maintain Markdown tables
        - Preserve statistical notation
        - Add explanations where needed
        
        Context: {context}
        Question: {question}
        Initial Answer: {initial_answer}
        
        Final Answer:
        """
    )

    # Create a pipeline of prompts
    final_prompt = PipelinePromptTemplate(
        final_prompt=refiner_prompt,
        pipeline_prompts=[
            ("initial_answer", main_prompt)
        ]
    )

    # Create the QA chain
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={
            "prompt": final_prompt,
            "document_prompt": PromptTemplate(
                input_variables=["page_content"],
                template="{page_content}"
            )
        },
        return_source_documents=True
    )

# Initialize session state
if 'qa_chain' not in st.session_state:
    st.session_state.update({
        'qa_chain': None,
        'pdf_images': None
    })



# Streamlit UI
st.title("📚 Analyseur de PDF Mathématiques")

# Section Téléchargement PDF
with st.container(border=True):
    uploaded_file = st.file_uploader("Télécharger un PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    if not st.session_state.qa_chain:
        with st.spinner("Traitement du PDF en cours..."):
            try:
                vector_store, images = process_pdf(tmp_path)
                st.session_state.qa_chain = initialize_qa_chain(vector_store)
                st.session_state.pdf_images = images
                st.success("PDF traité avec succès!")
            except Exception as e:
                st.error(f"Erreur de traitement : {str(e)}")

# Affichage du PDF
if st.session_state.pdf_images:
    with st.container(border=True):
        st.subheader("📄 Aperçu du Document")
        
        # Navigation entre pages
        cols = st.columns([1, 4, 1])
        with cols[1]:
            page_number = st.number_input(
                "Page", 
                min_value=1, 
                max_value=len(st.session_state.pdf_images),
                value=1,
                label_visibility="collapsed"
            )
        
        # Affichage de la page
        col1, col2 = st.columns(2)
        with col1:
            st.image(
                st.session_state.pdf_images[page_number-1],
                use_container_width=True,
                caption=f"Page {page_number}"
            )
import re
# Section Analyse
with st.container(border=True):
    st.subheader("🔍 Analyse du Contenu")
    question = st.text_input("Posez votre question ici :", key="question_input")
    
    if question and st.session_state.qa_chain:
        with st.spinner("Analyse en cours..."):
            try:
                result = st.session_state.qa_chain.invoke({"query": question})
                answer = result["result"]

                # Modification ici
                with st.expander("### 📝 Réponse Complète", expanded=True):
                    render_math(answer)  # Appel à la nouvelle fonction

                # Modification ici
                with st.expander("### 📚 Sources de Référence"):
                    for doc in result["source_documents"]:
                        source_page = doc.metadata['page'] + 1
                        content = doc.page_content
                        
                        with st.container(border=True):
                            st.markdown(f"**Page {source_page}**")
                            render_math(content)  # Appel à la nouvelle fonction
                
            except Exception as e:
                st.error(f"Erreur d'analyse : {str(e)}")

