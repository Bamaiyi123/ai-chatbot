import streamlit as st
from openai import OpenAI
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from dotenv import load_dotenv
from gtts import gTTS
import speech_recognition as sr
import tempfile
import time

# Load environment variables
load_dotenv()

# Initialize Groq client
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
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
    """Convert text to speech and play it"""
    try:
        tts = gTTS(text)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_file.name)
        return temp_file.name
    except Exception as e:
        st.error(f"TTS Error: {e}")
        return None

def record_audio():
    """Record audio from microphone"""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Listening... Speak now!")
        try:
            audio = r.listen(source, timeout=5)
            text = r.recognize_google(audio)
            return text
        except sr.WaitTimeoutError:
            st.error("No speech detected")
            return None
        except sr.UnknownValueError:
            st.error("Could not understand audio")
            return None
        except sr.RequestError as e:
            st.error(f"Could not request results; {e}")
            return None

def process_pdf(uploaded_file):
    """Process uploaded PDF and create FAISS index"""
    reader = PdfReader(uploaded_file)
    text = ""

    for page in reader.pages:
        text += page.extract_text()

    # Split into chunks
    chunks = [c for c in text.split("\n") if c.strip() != ""]
    st.session_state.doc_chunks = chunks

    # Generate embeddings
    embeddings = st.session_state.embedder.encode(chunks)

    # Create FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings, dtype='float32'))
    st.session_state.faiss_index = index

    return len(chunks)

def get_relevant_chunks(query, k=3):
    """Get most relevant document chunks for a query"""
    if not st.session_state.faiss_index or not st.session_state.doc_chunks:
        return []

    query_vec = st.session_state.embedder.encode([query])[0]
    D, I = st.session_state.faiss_index.search(np.array([query_vec], dtype='float32'), k=k)
    return [st.session_state.doc_chunks[i] for i in I[0]]

# UI
st.title("🤖 AI Chatbot with PDF Intelligence and voice recognition")

# Sidebar for PDF upload
with st.sidebar:
    st.header("📄 Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file and st.button("Process Document"):
        with st.spinner("Processing PDF..."):
            num_chunks = process_pdf(uploaded_file)
            st.success(f"✅ PDF processed! ({num_chunks} chunks indexed)")

# Main chat interface
st.markdown("---")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "audio_file" in message:
            st.audio(message["audio_file"], format="audio/mp3")

# Chat input area with voice button
col1, col2 = st.columns([4, 1])

with col1:
    prompt = st.chat_input("Ask about your document or chat...")

with col2:
    if st.button("🎤 Voice", use_container_width=True):
        voice_text = record_audio()
        if voice_text:
            prompt = voice_text
            st.rerun()

# Process input
if prompt:
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Get relevant context if document is loaded
            context = ""
            if st.session_state.doc_chunks:
                relevant_chunks = get_relevant_chunks(prompt)
                if relevant_chunks:
                    context = " ".join(relevant_chunks)

            # Create prompt
            if context:
                full_prompt = f"""
                Use this document information to answer the question:

                {context}

                Question: {prompt}

                If the question cannot be answered from the document, say so politely.
                """
            else:
                full_prompt = prompt

            # Get AI response
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": full_prompt}]
            )

            bot_reply = response.choices[0].message.content

            # Generate audio
            audio_file = speak(bot_reply)

            # Display response
            st.markdown(bot_reply)

            if audio_file:
                st.audio(audio_file, format="audio/mp3")

                # Add to message history
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

    st.rerun()
