import os
import json
import requests
from typing import Optional
from dotenv import load_dotenv

# Optional vendors left intact if you want to switch later
from openai import OpenAI
import google.generativeai as genai

# -------------------------
# Config
# -------------------------
load_dotenv()

# Vault config
VAULT_PATH = "/tmp/vault"  # change to your Obsidian vault root
MAX_FILES = 50             # limit how many files to stuff into the prompt
MAX_CHARS_PER_NOTE = 8000  # truncate very large notes to keep prompt manageable

# Model selector: "gemini", "openai", "ollama"
USE_MODEL = "openai"

# OpenAI
OPENAI_MODEL = "gpt-4o-mini"
openai_api_key = os.getenv("OPENAI_API_KEY")
client_openai = OpenAI(api_key=openai_api_key) if openai_api_key else None

# Gemini
GEMINI_MODEL = "gemini-1.5-flash"
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

# -------------------------
# IO
# -------------------------


def load_notes(vault_path: str) -> list[dict[str, str]]:
    notes = []
    for root, dirs, files in os.walk(vault_path):
        # drop hidden directories and known ignored ones
        dirs[:] = [d for d in dirs if not d.startswith(
            ".") and d not in IGNORED_DIRS]
        for file in files:
            if not file.endswith(".md"):
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
            if len(notes) >= MAX_FILES:
                return notes
    return notes

# -------------------------
# Prompting
# -------------------------


def build_prompt(notes: list[dict[str, str]], question: str) -> str:
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

    notes = load_notes(VAULT_PATH)
    if not notes:
        error_msg = "No notes found."
        if return_answer:
            return error_msg
        print(error_msg)
        return None

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
        notes = load_notes(VAULT_PATH)
        return jsonify({
            'note_count': len(notes),
            'model': USE_MODEL,
            'vault_path': VAULT_PATH
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
