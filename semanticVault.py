import os
import json
import pickle
import hashlib
import requests
from typing import Optional, Tuple, List, Dict
from dotenv import load_dotenv

# Optional vendors left intact if you want to switch later
from openai import OpenAI
import google.generativeai as genai

# Try to import sentence-transformers and numpy for semantic search
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    SEMANTIC_SEARCH_AVAILABLE = True
except ImportError:
    SEMANTIC_SEARCH_AVAILABLE = False
    SentenceTransformer = None
    np = None

# -------------------------
# Config
# -------------------------
load_dotenv()

# Vault config
# change to your Obsidian vault root
VAULT_PATH = "/path/to/your/obsidian/vault"
# limit how many files to stuff into the prompt (only used if SEMANTIC_SEARCH=False)
MAX_FILES = 50
MAX_CHARS_PER_NOTE = 8000  # truncate very large notes to keep prompt manageable

# Semantic search config
USE_SEMANTIC_SEARCH = True  # Set to True to use semantic search instead of file limit
# Number of most relevant notes to include when using semantic search
TOP_K_NOTES = 10
EMBEDDINGS_CACHE_DIR = os.path.join(
    os.path.dirname(__file__), ".embeddings_cache")
# CPU-only embedding model - no GPU required, runs efficiently on CPU
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Model selector: "gemini", "openai", "ollama"
# USE_MODEL = "ollama"
# USE_MODEL = "openai"
USE_MODEL = "gemini"

# OpenAI
OPENAI_MODEL = "gpt-4o-mini"
openai_api_key = os.getenv("OPENAI_API_KEY")
client_openai = OpenAI(api_key=openai_api_key) if openai_api_key else None

# Gemini
GEMINI_MODEL = "gemini-2.0-flash"
gemini_api_key = os.getenv("GEMINI_API_KEY")
if gemini_api_key:
    genai.configure(api_key=gemini_api_key)

# Ollama
OLLAMA_HOST = "10.0.1.101"
OLLAMA_URL = f"http://{OLLAMA_HOST}:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_OPTIONS = {
    "num_gpu": 0,          # force CPU only
    "num_ctx": 2048,       # lower memory to be safe
    "keep_alive": "0s"     # unload between calls
}
REQUEST_TIMEOUT = 120

# Ignore folders inside the vault
IGNORED_DIRS = {
    ".obsidian", ".trash", "attachments",         # common Obsidian clutter
    ".git", ".github", ".venv", "__pycache__",    # dev clutter
}

# Supported file extensions for notes
SUPPORTED_EXTENSIONS = {".md", ".txt"}  # Both Markdown and plain text files

# -------------------------
# IO
# -------------------------


def load_all_notes(vault_path: str, limit: Optional[int] = None) -> List[Dict[str, str]]:
    """
    Load all notes from the vault (or up to limit if specified).
    Supports both .md and .txt files.
    Removed MAX_FILES limit when semantic search is enabled.
    """
    notes = []
    for root, dirs, files in os.walk(vault_path):
        # drop hidden directories and known ignored ones
        dirs[:] = [d for d in dirs if not d.startswith(
            ".") and d not in IGNORED_DIRS]
        for file in files:
            # Check if file has a supported extension
            file_lower = file.lower()
            if not any(file_lower.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                continue
            full_path = os.path.join(root, file)
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except (IOError, OSError, UnicodeDecodeError, PermissionError) as e:
                print(f"Skipped {full_path} due to read error: {e}")
                continue
            if MAX_CHARS_PER_NOTE and len(content) > MAX_CHARS_PER_NOTE:
                content = content[:MAX_CHARS_PER_NOTE] + "\n...[truncated]..."
            notes.append({"content": content, "path": full_path})
            if limit and len(notes) >= limit:
                return notes
    return notes


def load_notes(vault_path: str) -> List[Dict[str, str]]:
    """Legacy function - now delegates to load_all_notes with appropriate limit."""
    if USE_SEMANTIC_SEARCH:
        # No limit when using semantic search
        return load_all_notes(vault_path)
    else:
        # Use MAX_FILES limit for backward compatibility
        return load_all_notes(vault_path, limit=MAX_FILES)

# -------------------------
# Semantic Search (Embeddings)
# -------------------------


def get_cache_path(vault_path: str) -> str:
    """Generate cache file path based on vault path."""
    os.makedirs(EMBEDDINGS_CACHE_DIR, exist_ok=True)
    vault_hash = hashlib.md5(vault_path.encode()).hexdigest()[:8]
    return os.path.join(EMBEDDINGS_CACHE_DIR, f"embeddings_{vault_hash}.pkl")


def generate_embeddings(notes: List[Dict[str, str]], force_rebuild: bool = False) -> Tuple[List, object]:
    """
    Generate embeddings for all notes and cache them.
    Returns tuple of (embeddings_list, model).
    """
    if not SEMANTIC_SEARCH_AVAILABLE:
        raise RuntimeError(
            "Semantic search requires sentence-transformers. Install with: pip install sentence-transformers"
        )

    cache_path = get_cache_path(VAULT_PATH)

    # Try to load cached embeddings
    if not force_rebuild and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cached_data = pickle.load(f)
                if cached_data.get("note_paths") == [n["path"] for n in notes]:
                    print("📦 Using cached embeddings...")
                    # Explicitly use CPU - no GPU needed
                    model = SentenceTransformer(EMBEDDING_MODEL, device='cpu')
                    return cached_data["embeddings"], model
        except Exception as e:
            print(f"⚠️  Cache load failed, regenerating: {e}")

    # Generate new embeddings
    print(f"🔄 Generating embeddings for {len(notes)} notes (CPU mode)...")
    # Explicitly use CPU - the model is lightweight and designed for CPU
    model = SentenceTransformer(EMBEDDING_MODEL, device='cpu')

    # Create text representations for embedding (path + content)
    texts = []
    for note in notes:
        text = f"{note['path']}\n{note['content']}"
        texts.append(text)

    embeddings = model.encode(texts, show_progress_bar=True)

    # Cache embeddings
    try:
        with open(cache_path, "wb") as f:
            pickle.dump({
                "embeddings": embeddings,
                "note_paths": [n["path"] for n in notes]
            }, f)
        print("✅ Embeddings cached successfully")
    except Exception as e:
        print(f"⚠️  Failed to cache embeddings: {e}")

    return embeddings, model


def find_relevant_notes(question: str, all_notes: List[Dict[str, str]],
                        embeddings: List, model: object, top_k: int = TOP_K_NOTES) -> List[Dict[str, str]]:
    """
    Find the most relevant notes for a question using semantic similarity.
    """
    if not SEMANTIC_SEARCH_AVAILABLE or np is None:
        return all_notes[:top_k]  # Fallback to first N notes

    # Generate embedding for the question
    question_embedding = model.encode([question])

    # Calculate cosine similarities
    similarities = np.dot(embeddings, question_embedding.T).flatten()

    # Get top K most similar notes
    top_indices = np.argsort(similarities)[::-1][:top_k]

    relevant_notes = [all_notes[i] for i in top_indices]

    return relevant_notes


# -------------------------
# Prompting
# -------------------------


def build_prompt(notes: List[Dict[str, str]], question: str) -> str:
    header = (
        "You are my knowledge assistant. Use only the provided notes. "
        "If the answer is not present in the notes, say you do not have enough information. "
        "Return a concise answer.\n\n"
    )
    body = []
    for i, note in enumerate(notes, start=1):
        body.append(f"Note {i} (path: {note['path']}):\n{note['content']}\n")
    q = f"\nQuestion: {question}\nAnswer:"
    return header + "\n".join(body) + q

# -------------------------
# Providers
# -------------------------


def ask_openai(prompt: str) -> str:
    if not client_openai:
        raise RuntimeError("OpenAI API key missing")
    response = client_openai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant for searching personal notes."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=2000,
    )
    return response.choices[0].message.content


def ask_gemini(prompt: str) -> str:
    if not gemini_api_key:
        raise RuntimeError("Gemini API key missing")
    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(prompt)
    return response.text


def ask_ollama(prompt: str) -> Optional[str]:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": OLLAMA_OPTIONS,
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as e:
        print(f"❌ Ollama network error: {e}")
        return None

    if not resp.ok:
        print(
            f"❌ Ollama Error: HTTP {resp.status_code} body={resp.text[:300]}")
        return None

    try:
        data = resp.json()
    except json.JSONDecodeError as e:
        print(f"❌ Ollama JSON parse error: {e}")
        return None

    return data.get("response", "").strip()

# -------------------------
# Unified ask
# -------------------------


def ask_question(question: str, return_answer: bool = False) -> Optional[str]:
    """
    Ask a question to the knowledge base.

    Args:
        question: The question to ask
        return_answer: If True, returns the answer string. If False, prints it (CLI mode).

    Returns:
        The answer string if return_answer=True, None otherwise.
    """
    # Input validation
    if not question or not question.strip():
        error_msg = "❌ Please provide a non-empty question."
        if return_answer:
            return error_msg
        print(error_msg)
        return None

    # Load all notes (no limit when using semantic search)
    all_notes = load_all_notes(VAULT_PATH)
    if not all_notes:
        error_msg = "No notes found."
        if return_answer:
            return error_msg
        print(error_msg)
        return None

    # Use semantic search if enabled
    if USE_SEMANTIC_SEARCH and SEMANTIC_SEARCH_AVAILABLE:
        try:
            embeddings, model = generate_embeddings(all_notes)
            notes = find_relevant_notes(
                question, all_notes, embeddings, model, TOP_K_NOTES)
            if not return_answer:
                print(
                    f"🔍 Found {len(notes)} most relevant notes out of {len(all_notes)} total notes")
        except Exception as e:
            print(
                f"⚠️  Semantic search failed: {e}. Falling back to first {MAX_FILES} notes.")
            notes = all_notes[:MAX_FILES]
    else:
        # Fallback to limited notes
        notes = all_notes[:MAX_FILES] if not USE_SEMANTIC_SEARCH else all_notes

    prompt = build_prompt(notes, question)

    try:
        if USE_MODEL == "ollama":
            answer = ask_ollama(prompt)
            if answer is None:
                error_msg = "❌ Failed to get answer from Ollama."
                if return_answer:
                    return error_msg
                print(error_msg)
                return None
        elif USE_MODEL == "openai":
            answer = ask_openai(prompt)
        elif USE_MODEL == "gemini":
            answer = ask_gemini(prompt)
        else:
            error_msg = f"❌ Unknown model: {USE_MODEL}"
            if return_answer:
                return error_msg
            print(error_msg)
            return None

        if return_answer:
            return answer

        print("\n📌 Answer:")
        print(answer)
        return None
    except RuntimeError as e:
        error_msg = f"❌ Error: {e}"
        if return_answer:
            return error_msg
        print(error_msg)
        return None


# -------------------------
# Web Interface (Flask)
# -------------------------

def create_app():
    from flask import Flask, render_template, request, jsonify

    app = Flask(__name__)

    @app.route('/')
    def index():
        return render_template('index.html', model=USE_MODEL)

    @app.route('/api/ask', methods=['POST'])
    def api_ask():
        data = request.get_json()
        question = data.get('question', '').strip()

        if not question:
            return jsonify({'error': 'Please provide a non-empty question.'}), 400

        answer = ask_question(question, return_answer=True)
        return jsonify({'answer': answer})

    @app.route('/api/stats', methods=['GET'])
    def api_stats():
        all_notes = load_all_notes(VAULT_PATH)
        return jsonify({
            'note_count': len(all_notes),
            'model': USE_MODEL,
            'vault_path': VAULT_PATH,
            'semantic_search': USE_SEMANTIC_SEARCH and SEMANTIC_SEARCH_AVAILABLE,
            'top_k_notes': TOP_K_NOTES if USE_SEMANTIC_SEARCH else MAX_FILES
        })

    return app


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    import sys

    # Check if running in web mode
    if len(sys.argv) > 1 and sys.argv[1] == '--web':
        app = create_app()
        print("🚀 Starting Semantic Vault Web Interface...")
        print(f"📊 Model: {USE_MODEL}")
        print(f"📁 Vault: {VAULT_PATH}")
        print("🌐 Open your browser to http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        # CLI mode
        user_question = input("Ask your question: ").strip()
        ask_question(user_question)
