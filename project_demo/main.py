import streamlit as st
import fitz  # PyMuPDF
from pdf2image import convert_from_path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import tempfile
import re as regex
from datetime import datetime
import os
from sympy import *  # Pour générer du LaTeX depuis des expressions Sympy
import requests  # Nécessaire pour interroger l'API QuickLaTeX

# Configuration de SQLite via pysqlite3
__import__('pysqlite3')
import pysqlite3
import sys
sys.modules['sqlite3'] = sys.modules["pysqlite3"]

import chromadb
from langchain_community.vectorstores import Chroma

# --- CSS personnalisé ---
st.markdown("""
<style>
body {
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
}
.chat-message {
    margin: 1em 0;
    padding: 1.5em;
    border-radius: 15px;
    animation: slideIn 0.3s ease-out;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.user-message {
    background: #e3f2fd;
    margin-left: 20%;
    border-radius: 15px 15px 0 15px;
}
.assistant-message {
    background: #ffffff;
    margin-right: 20%;
    border-radius: 15px 15px 15px 0;
    border: 1px solid #e0e0e0;
}
.timestamp {
    font-size: 0.75em;
    color: #666;
    margin-top: 0.8em;
}
@keyframes slideIn {
    from { transform: translateY(20px); opacity: 0; }
    to   { transform: translateY(0); opacity: 1; }
}
</style>
""", unsafe_allow_html=True)

# --- Fonction pour obtenir l'URL d'une image LaTeX via QuickLaTeX ---
def get_quicklatex_image_url(latex_code: str) -> str:
    """
    Envoie le code LaTeX à QuickLaTeX et renvoie l'URL de l'image générée.
    
    Le service QuickLaTeX est interrogé via une requête GET.
    """
    params = {
        'formula': latex_code,
        'fsize': 12,
        'fcolor': '000000',
        'mode': 0,
        'out': 1,
        'errors': 1
    }
    try:
        response = requests.get("http://quicklatex.com/latex3.f", params=params)
        lines = response.text.split('\n')
        status = lines[0].strip()
        if status == '0':
            image_url = lines[1].strip()
            return image_url
    except Exception as e:
        st.error(f"Erreur lors de la génération de l'image LaTeX : {str(e)}")
    return None

# --- Fonctions de rendu ---

def render_latex(content: str):
    st.latex(content)

def render_text(content: str):
    text_stripped = content.strip()
    if text_stripped:
        st.markdown(text_stripped)

def render_math(content: str):
    """
    Détecte et affiche des blocs LaTeX dans le texte en les convertissant en images via QuickLaTeX.
    
    La fonction cherche les formats suivants :
      - $...$ pour le mode inline,
      - \(...\) pour le mode inline,
      - \[...\] pour le mode display,
      - \begin{...}...\end{...} pour les environnements LaTeX,
      - [ ... ] si le contenu contient une commande LaTeX (ex: \text, \frac, etc.) ou un '='.
      
    Pour un rendu correct, on retire les délimiteurs superflus avant d'envoyer le code à QuickLaTeX.
    """
    pattern = (
        r'('
        r'\$.*?\$'                                # inline math avec $
        r'|\\\(\s*.*?\s*\\\)'                      # inline math avec \(...\)
        r'|\\\[\s*.*?\s*\\\]'                      # display math avec \[...\]
        r'|\\begin\{.*?\}.*?\\end\{.*?\}'           # environnements LaTeX
        r'|\[\s*(?=.*(?:\\[a-zA-Z]+|=)).*?\s*\]'     # bloc entre crochets contenant une commande ou '='
        r')'
    )
    
    # Récupération des blocs LaTeX dans le contenu (y compris sur plusieurs lignes)
    latex_blocks = regex.findall(pattern, content, regex.DOTALL)
    
    for block in latex_blocks:
        code = block
        # Retirer les délimiteurs selon le format détecté
        if code.startswith('$') and code.endswith('$'):
            code = code[1:-1].strip()
        elif code.startswith('\\(') and code.endswith('\\)'):
            code = code[2:-2].strip()
        elif code.startswith('\\[') and code.endswith('\\]'):
            code = code[2:-2].strip()
        elif code.startswith('[') and code.endswith(']'):
            code = code[1:-1].strip()
        # Vous pouvez conserver intact le code pour les environnements \begin... \end...
        
        # Obtenir l'URL de l'image via QuickLaTeX
        img_url = get_quicklatex_image_url(code)
        if img_url:
            st.image(img_url)
        else:
            # Fallback : afficher le code LaTeX dans un bloc de code
            st.markdown(f"```latex\n{code}\n```")
    
    # Afficher le reste du texte (sans les blocs LaTeX)
    text_content = regex.sub(pattern, '', content)
    if text_content.strip():
        st.markdown(text_content)

# --- Traitement du PDF ---
def process_pdf_images(pdf_path: str, max_pages: int = 15, dpi: int = 300):
    images = convert_from_path(pdf_path, dpi=dpi)
    return images[:max_pages]

def process_pdf_text(pdf_path: str, max_pages: int = 15, chunk_size: int = 1000, chunk_overlap: int = 200):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()[:max_pages]
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(pages)

def create_vector_store(texts):
    embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=st.secrets["MISTRAL_API_KEY"])
    return Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")

def process_pdf(pdf_path: str):
    images = process_pdf_images(pdf_path)
    texts = process_pdf_text(pdf_path)
    vector_store = create_vector_store(texts)
    return vector_store, images

def get_pdf_page_count(pdf_path: str) -> int:
    doc = fitz.open(pdf_path)
    return doc.page_count

# --- Chaîne QA ---
def initialize_qa_chain(vector_store):
    llm = ChatMistralAI(
        model="mistral-large-latest",
        mistral_api_key=st.secrets["MISTRAL_API_KEY"],
        temperature=0
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
[Rôle] Expert mathématique
[Règles]
1. Utilisez LaTeX ($...$) pour les équations.
2. Utilisez des tableaux en Markdown si nécessaire.
3. Respectez la notation statistique.
4. Pas de mélange texte/équation sur la même ligne.
5. Répondez de manière claire et concise.
6. Indiquez vos sources si nécessaire.
7. Évitez les réponses trop longues.
8. Ne donnez pas de réponses triviales.
9. Répondez de manière professionnelle.
10. Utilisez des phrases complètes.
11. Évitez les réponses trop techniques.
12. Répondez en français.
[Contexte]
{context}

[Question]
{question}

[Réponse]
"""
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

# --- État de session ---
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'pdf_images' not in st.session_state:
    st.session_state.pdf_images = []

def update_chat_history(role: str, content: str):
    entry = {
        "role": role,
        "content": content,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    }
    st.session_state.chat_history.append(entry)
    if len(st.session_state.chat_history) > 10:
        st.session_state.chat_history.pop(0)

def display_chat_message(role: str, content: str, timestamp: str):
    css_class = "user-message" if role == "user" else "assistant-message"
    st.markdown(f"""
    <div class="chat-message {css_class}">
        <div>{content}</div>
        <div class="timestamp">{timestamp}</div>
    </div>
    """, unsafe_allow_html=True)

# --- Interface principale ---
st.title("📚 Assistant d'Analyse Mathématique")

# 1. Chargement du PDF
with st.container():
    uploaded_file = st.file_uploader("Télécharger un PDF (max 15 pages)", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    if get_pdf_page_count(tmp_path) > 15:
        st.error("Document trop volumineux. Maximum 15 pages autorisées.")
        st.stop()

    if not st.session_state.qa_chain:
        with st.spinner("Analyse du document en cours..."):
            try:
                vector_store, images = process_pdf(tmp_path)
                st.session_state.qa_chain = initialize_qa_chain(vector_store)
                st.session_state.pdf_images = images
                st.session_state.chat_history = []  # Réinitialisation de l'historique
                st.success("Document prêt pour analyse !")
            except Exception as e:
                st.error(f"Erreur de traitement : {str(e)}")

# 2. Aperçu du PDF
if st.session_state.pdf_images and uploaded_file:
    with st.container():
        st.subheader("📄 Aperçu du Document")
        page_number = st.number_input(
            "Page",
            min_value=1,
            max_value=len(st.session_state.pdf_images),
            value=1,
            label_visibility="collapsed"
        )
        st.image(st.session_state.pdf_images[page_number - 1], use_container_width=True, caption=f"Page {page_number}")

# 3. Affichage dynamique de l'historique de la conversation
chat_container = st.empty()

def render_chat_history():
    with chat_container.container():
        st.subheader("🗨️ Historique de la Conversation")
        for chat in st.session_state.chat_history:
            display_chat_message(chat["role"], chat["content"], chat["timestamp"])

if st.session_state.chat_history:
    render_chat_history()

# 4. Zone pour poser une nouvelle question via un formulaire
with st.form(key="chat_form"):
    question = st.text_input("Votre question :", key="question_input", placeholder="Entrez votre question ici")
    submit_button = st.form_submit_button("Envoyer")
    
    if submit_button and question and st.session_state.qa_chain:
        with st.spinner("Analyse en cours..."):
            update_chat_history("user", question)
            result = st.session_state.qa_chain.invoke({"query": question})
            answer = result["result"]
            update_chat_history("assistant", answer)
            # Rendu du LaTeX de la réponse sous forme d'image via QuickLaTeX :
            render_math(answer)
            render_chat_history()

# --- Exemple d'utilisation de Sympy pour générer et afficher du LaTeX ---
if st.button("Afficher un exemple Sympy"):
    x, y = symbols('x y')
    eq = Eq(x + y, 2)
    st.markdown("Exemple d'équation générée avec Sympy :")
    st.latex(latex(eq))
    content = """
    Pour calculer la fréquence théorique, nous utilisons la formule :
    [\text{Fréquence théorique} = \frac{\text{Total de la ligne} \times \text{Total de la colonne}}{\text{Effectif total}}]
    """
    render_math(content)
