from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")
AUTH_TOKEN = os.getenv("SERVER_AUTH_TOKEN", "dev-token")

if not API_KEY:
    raise RuntimeError("GROQ_API_KEY missing in environment")

client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=API_KEY)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

app = Flask(__name__, static_url_path='', static_folder='frontend')
CORS(app)

memory = {
    "doc_chunks": [],
    "faiss_index": None
}


def require_auth():
    auth = request.headers.get("Authorization", "")
    if auth != f"Bearer {AUTH_TOKEN}":
        return jsonify({"error": "Unauthorized"}), 401
    return None


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/upload", methods=["POST"])
def upload_pdf():
    auth_err = require_auth()
    if auth_err:
        return auth_err

    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No PDF provided"}), 400

    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text += page_text

    chunks = [c for c in text.split("\n") if c.strip()]
    memory["doc_chunks"] = chunks

    embeddings = embedder.encode(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings, dtype='float32'))
    memory["faiss_index"] = index

    return jsonify({"status": "indexed", "chunks": len(chunks)})


@app.route("/chat", methods=["POST"])
def chat():
    auth_err = require_auth()
    if auth_err:
        return auth_err

    data = request.get_json(force=True)
    prompt = data.get("prompt", "").strip()
    use_context = data.get("use_context", True)

    if not prompt:
        return jsonify({"error": "prompt required"}), 400

    context = ""
    if use_context and memory["faiss_index"] and memory["doc_chunks"]:
        qv = embedder.encode([prompt])[0]
        D, I = memory["faiss_index"].search(np.array([qv], dtype='float32'), k=3)
        relevant = [memory["doc_chunks"][i] for i in I[0]]
        context = "\n".join(relevant)

    full_prompt = f"Use this document info to answer:\n{context}\nQuestion: {prompt}" if context else prompt

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": full_prompt}]
    )

    answer = response.choices[0].message.content
    return jsonify({"answer": answer})


@app.route("/")
def index():
    return app.send_static_file('index.html')


if __name__ == "__main__":
    app.run()
