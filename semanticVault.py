import os
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai as genai
import requests


# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

# CONFIGURATION
VAULT_PATH = "/path/to/your/obsidian/vault"
MAX_FILES = 50  # Max number of .md files to load (adjust if needed)

# Options: "gemini", "openai", "ollama"
# USE_MODEL = "ollama"
USE_MODEL = "openai"
# USE_MODEL = "gemini"

# OpenAI
OPENAI_MODEL = "gpt-4o-mini"
# OPENAI_MODEL = "gpt-4o"
# OPENAI_MODEL = "gpt-3.5-turbo"

# Gemini
GEMINI_MODEL = "gemini-1.5-flash"

# Ollama Local LLM
OLLAMA_URL = "http://localhost:11434/api/generate"


# Common Ollama Models and Recommendations:
# - "mistral"  -> Fast, low resource (~4-8GB RAM), good general answers
# - "llama3"   -> More advanced reasoning, needs ~8-12GB RAM
# - "gemma:2b" -> Very lightweight, minimal resource, basic tasks
# - "gemma:7b" -> Better quality, ~8GB RAM, still efficient


OLLAMA_MODEL = "deepseek-r1:14b"  # Change here for local model choice

# Initialize OpenAI
client_openai = OpenAI(api_key=openai_api_key) if openai_api_key else None

# Initialize Gemini
if gemini_api_key:
    genai.configure(api_key=gemini_api_key)

# Load Markdown Files with Paths


def load_notes(vault_path):
    notes = []
    ignored_dirs = {".obsidian", ".trash", "attachments"}

    for root, dirs, files in os.walk(vault_path):
        # Remove ignored directories from traversal
        dirs[:] = [d for d in dirs if d not in ignored_dirs]

        for file in files:
            if file.endswith(".md"):
                full_path = os.path.join(root, file)
                with open(full_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    notes.append({"content": content, "path": full_path})
                    if len(notes) >= MAX_FILES:
                        return notes
    return notes
# Build Prompt with File Context


def build_prompt(notes, question):
    prompt = "You are my knowledge assistant. Based on the following notes, answer the question:\n\n"
    for i, note in enumerate(notes):
        prompt += f"Note {i+1}:\n{note}\n\n"
    prompt += f"Question: {question}\nAnswer:"
    return prompt

# Ask OpenAI


def ask_openai(prompt):
    response = client_openai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant for searching personal notes."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2000
    )
    return response.choices[0].message.content

# Ask Gemini


def ask_gemini(prompt):
    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(prompt)
    return response.text

# Ask Ollama Local LLM


def ask_ollama(prompt):
    data = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(OLLAMA_URL, json=data)
    if response.ok:
        result = response.json()
        return result.get("response", "").strip()
    else:
        print(f"❌ Ollama Error: {response.status_code}")
        return None

# Unified Ask Function with Fallback


def ask_question(question):
    notes = load_notes(VAULT_PATH)
    if not notes:
        print("No notes found.")
        return
    prompt = build_prompt(notes, question)

    answer = None

    if USE_MODEL == "gemini":
        if not gemini_api_key:
            print("❌ Gemini API key missing.")
            return
        try:
            answer = ask_gemini(prompt)
        except Exception as e:
            print(f"⚠️ Gemini failed ({e}). Switching to OpenAI...")
            if not openai_api_key:
                print("❌ OpenAI API key missing. Cannot fallback.")
                return
            answer = ask_openai(prompt)

    elif USE_MODEL == "openai":
        if not openai_api_key:
            print("❌ OpenAI API key missing.")
            return
        answer = ask_openai(prompt)

    elif USE_MODEL == "ollama":
        answer = ask_ollama(prompt)

    else:
        print(f"❌ Unknown model selected: {USE_MODEL}")
        return

    print("\n📌 Answer:")
    print(answer)

    # print("\n🔍 Notes included in this search:")
    # for note in notes:
    #     print(f"- {note['path']}")


# Main
if __name__ == "__main__":
    user_question = input("Ask your question: ")
    ask_question(user_question)
