import os
from redis_ai_client import RedisAIClient
import markdown
from bs4 import BeautifulSoup
from tiktoken import get_encoding

def md_to_text(md_path):
    with open(md_path, "r", encoding="utf-8") as f:
        html = markdown.markdown(f.read())
        text = BeautifulSoup(html, features="html.parser").get_text()
    return text

def chunk_text(text, max_tokens=3000, overlap=100, encoding_name='cl100k_base'):
    enc = get_encoding(encoding_name)
    tokens = enc.encode(text)
    
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk = enc.decode(tokens[start:end])
        chunks.append(chunk)
        start += max_tokens - overlap
    return chunks

# Use an absolute path for the commands directory
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# COMMANDS_DIR = os.path.join(REPO_ROOT, 'redis-inference-optimization', 'docs')
COMMANDS_DIR = os.path.join(REPO_ROOT, 'redis-doc', 'commands')

client = RedisAIClient()
client.create_vector_index()

total = 0
success = 0
fail = 0

for filename in os.listdir(COMMANDS_DIR):
    if filename.endswith('.md'):
        print(filename)
        total += 1
        file_path = os.path.join(COMMANDS_DIR, filename)
        print(f"[INFO] Processing {file_path}")
        try:
            md_text = md_to_text(file_path)
            doc_id = filename.replace('.md', '')
            chunks = chunk_text(md_text)
            for i, chunk in enumerate(chunks):
                embedding = client.embed_text(chunk)
                client.store_document_with_embedding(i, chunk, embedding)
            print(f"[SUCCESS] Ingested {doc_id}")
            success += 1
        except Exception as e:
            print(f"[ERROR] Failed to ingest {filename}: {e}")
            fail += 1

print(f"\n[SUMMARY] Processed: {total}, Success: {success}, Failed: {fail}")