import streamlit as st
from openai import OpenAI
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from dotenv import load_dotenv
from gtts import gTTS
import tempfile

# Load environment variables (optional locally)
load_dotenv()

# Debug: Check if API key is loaded
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("GROQ_API_KEY not found in environment variables. Please check your .env file.")
    st.stop()

# Initialize Groq client
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=api_key
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "doc_chunks" not in st.session_state:
    st.session_state.doc_chunks = []
if "embedder" not in st.session_state:
    st.session_state.embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Functions
def speak(text):
    """Convert text to speech and return audio path"""
    try:
        tts = gTTS(text)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_file.name)
        return temp_file.name
    except Exception as e:
        st.error(f"TTS Error: {e}")
        return None

def process_pdf(uploaded_file):
    """Process PDF and create FAISS index"""
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    chunks = [c for c in text.split("\n") if c.strip() != ""]
    st.session_state.doc_chunks = chunks

    embeddings = st.session_state.embedder.encode(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings, dtype='float32'))
    st.session_state.faiss_index = index

    return len(chunks)

def get_relevant_chunks(query, k=3):
    if not st.session_state.faiss_index or not st.session_state.doc_chunks:
        return []

    query_vec = st.session_state.embedder.encode([query])[0]
    D, I = st.session_state.faiss_index.search(np.array([query_vec], dtype='float32'), k=k)
    return [st.session_state.doc_chunks[i] for i in I[0]]

# UI
st.title("🤖 AI Chatbot with PDF Intelligence & Voice Playback")

# Sidebar for PDF upload
with st.sidebar:
    st.header("📄 Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file and st.button("Process Document"):
        with st.spinner("Processing PDF..."):
            num_chunks = process_pdf(uploaded_file)
            st.success(f"✅ PDF processed! ({num_chunks} chunks indexed)")

st.markdown("---")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "audio_file" in message:
            st.audio(message["audio_file"], format="audio/mp3")

# Chat input area
prompt = None
col1, col2 = st.columns([4, 1])

with col1:
    prompt_text = st.chat_input("Ask about your document or chat...")
    if prompt_text:
        prompt = prompt_text

with col2:
    audio_upload = st.file_uploader("🎤 Upload voice (mp3/wav)", type=["mp3", "wav"])
    if audio_upload:
        # For now, we just acknowledge voice input
        prompt = "User uploaded an audio message"

# Process input
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Context from document
            context = ""
            if st.session_state.doc_chunks:
                relevant_chunks = get_relevant_chunks(prompt)
                if relevant_chunks:
                    context = " ".join(relevant_chunks)

            full_prompt = f"Use this document info to answer:\n{context}\nQuestion: {prompt}" if context else prompt

            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": full_prompt}]
            )
            bot_reply = response.choices[0].message.content

            audio_file = speak(bot_reply)
            st.markdown(bot_reply)
            if audio_file:
                st.audio(audio_file, format="audio/mp3")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": bot_reply,
                    "audio_file": audio_file
                })
            else:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": bot_reply
                })
