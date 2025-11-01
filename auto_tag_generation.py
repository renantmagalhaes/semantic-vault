import os
import re
import requests
import argparse
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai as genai


# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Vault Location
VAULT_PATH = "/tmp/vault"

# Model Config
# USE_MODEL = "ollama"
USE_MODEL = "openai"
# USE_MODEL = "gemini"

# OpenAI
# OPENAI_MODEL = "gpt-4o"
OPENAI_MODEL = "gpt-4o-mini"


# Gemini
GEMINI_MODEL = "gemini-1.5-flash"

# Ollama Local LLM
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "deepseek-r1:14b"  # Change as needed

# Initialize APIs
client_openai = OpenAI(api_key=openai_api_key) if openai_api_key else None
if gemini_api_key:
    genai.configure(api_key=gemini_api_key)


# Generate Tags from Note
def generate_tags(note_content, max_new_tags):
    prompt = (
        f"Based on the following note, suggest up to {max_new_tags} relevant, short keywords as tags.\n"
        "Only return the tags as a comma-separated list, no hashtags or explanations.\n\n"
        f"{note_content}\n\n"
        "Tags:"
    )

    if USE_MODEL == "ollama":
        data = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
        response = requests.post(OLLAMA_URL, json=data)
        if response.ok:
            tags_raw = response.json().get("response", "").strip()
            return clean_tag_list(tags_raw, max_new_tags)

    elif USE_MODEL == "openai":
        response = client_openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You suggest short, relevant tags for notes.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=100,
        )
        tags_raw = response.choices[0].message.content.strip()
        return clean_tag_list(tags_raw, max_new_tags)

    elif USE_MODEL == "gemini":
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        return clean_tag_list(response.text.strip(), max_new_tags)

    print(f"❌ Failed to generate tags with model: {USE_MODEL}")
    return []


# Clean tag string to list, max tags
def clean_tag_list(tags_raw, max_tags):
    tags = [tag.strip() for tag in re.split(r",|\n", tags_raw) if tag.strip()]
    return tags[:max_tags]


# Apply Tags to Notes
def apply_tags_to_vault(vault_path, dry_run=False, force_overwrite=False):
    ignored_dirs = {".obsidian", ".trash", "attachments"}

    total_files = 0
    updated_files = 0
    skipped_files = 0

    for root, dirs, files in os.walk(vault_path):
        dirs[:] = [d for d in dirs if d not in ignored_dirs]

        for file in files:
            if file.endswith(".md"):
                total_files += 1
                full_path = os.path.join(root, file)

                with open(full_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Extract YAML and existing tags
                existing_tags = []
                yaml_match = re.match(
                    r"^---\n(.*?)\n---\n", content, re.DOTALL)

                if yaml_match:
                    yaml_block = yaml_match.group(1)
                    tag_match = re.search(r"tags:\s*\[([^\]]*)\]", yaml_block)
                    if tag_match:
                        existing_tags = [
                            tag.strip()
                            for tag in tag_match.group(1).split(",")
                            if tag.strip()
                        ]

                if len(existing_tags) >= 5 and not force_overwrite:
                    skipped_files += 1
                    continue

                max_new_tags = 5 if force_overwrite else 5 - len(existing_tags)
                generated_tags = generate_tags(content, max_new_tags)

                if not generated_tags and not force_overwrite:
                    skipped_files += 1
                    continue

                final_tags = (
                    generated_tags
                    if force_overwrite
                    else list(dict.fromkeys(existing_tags + generated_tags))[:5]
                )
                tag_line = f"tags: [{', '.join(final_tags)}]"

                # Update YAML or prepend
                if yaml_match:
                    new_yaml = re.sub(
                        r"tags:\s*\[[^\]]*\]", tag_line, yaml_block)
                    if "tags:" not in yaml_block:
                        new_yaml += f"\n{tag_line}"
                    updated_content = re.sub(
                        r"^---\n(.*?)\n---\n",
                        f"---\n{new_yaml}\n---\n",
                        content,
                        flags=re.DOTALL,
                    )
                else:
                    updated_content = f"---\n{tag_line}\n---\n\n{content}"

                print(
                    f"✅ {'[Dry Run] ' if dry_run else ''}Tags for {full_path} → {final_tags}"
                )

                if not dry_run and content != updated_content:
                    with open(full_path, "w", encoding="utf-8") as f:
                        f.write(updated_content)
                    updated_files += 1
                elif dry_run:
                    updated_files += 1  # Counted for preview

    print("\n📊 Tagging Summary:")
    print(f"Total Markdown files scanned: {total_files}")
    print(f"Files updated: {updated_files}")
    print(f"Files skipped (max tags reached): {skipped_files}")


# CLI Entry Point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Obsidian Vault Tag Generator")
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview changes without modifying files"
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite all existing tags with new tags"
    )

    args = parser.parse_args()

    print(f"\n🗂 Scanning vault at: {VAULT_PATH} with model: {USE_MODEL}")
    apply_tags_to_vault(VAULT_PATH, dry_run=args.dry_run,
                        force_overwrite=args.force)
    print("\n🎉 Tag generation complete.")
