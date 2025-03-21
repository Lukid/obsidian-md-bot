import gradio as gr
import requests
from bs4 import BeautifulSoup
import os
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Dict, Optional
import openai
from anthropic import Anthropic
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, util

# Load environment variables
load_dotenv()

# Configure LLM providers
DEFAULT_PROVIDER = os.getenv('DEFAULT_LLM_PROVIDER', 'openai')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Initialize LLM clients
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
if ANTHROPIC_API_KEY:
    # Disabilita temporaneamente il supporto Anthropic per problemi di compatibilità
    pass
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Initialize sentence transformer for semantic similarity
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load Obsidian vault path
DEFAULT_VAULT_PATH = os.getenv('OBSIDIAN_VAULT_PATH')
GRAPH_SIMILARITY_THRESHOLD = float(os.getenv('GRAPH_SIMILARITY_THRESHOLD', '0.85').split('#')[0].strip())

def find_similar_notes(content: str, vault_path: str) -> List[Dict]:
    """Find semantically similar notes in the vault."""
    # Get all markdown files in the vault
    similar_notes = []
    
    # Rimuovi eventuali metadati YAML dal contenuto di input
    if content.startswith('---'):
        content_parts = content.split('---', 2)
        if len(content_parts) >= 3:
            content = content_parts[2].strip()
    
    # Rimuovi eventuali titoli H1 dal contenuto
    content_lines = content.split('\n')
    content_without_h1 = [line for line in content_lines if not line.strip().startswith('# ')]
    content = '\n'.join(content_without_h1)
    
    content_embedding = embedding_model.encode(content)
    
    for root, _, files in os.walk(vault_path):
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    note_content = f.read()
                    # Rimuovi metadati YAML dalla nota esistente
                    if note_content.startswith('---'):
                        content_parts = note_content.split('---', 2)
                        if len(content_parts) >= 3:
                            note_content = content_parts[2].strip()
                    note_embedding = embedding_model.encode(note_content)
                    similarity = util.pytorch_cos_sim(content_embedding, note_embedding)
                    
                    if similarity > GRAPH_SIMILARITY_THRESHOLD:
                        similar_notes.append({
                            'title': file,
                            'path': os.path.relpath(file_path, vault_path),
                            'similarity': float(similarity)
                        })
    
    return sorted(similar_notes, key=lambda x: x['similarity'], reverse=True)

def get_llm_summary(content: str, provider: str = DEFAULT_PROVIDER) -> str:
    """Generate summary using the configured LLM provider."""
    prompt = f"""Analizza il seguente contenuto e crea un riassunto dettagliato in italiano.
Anche se il contenuto originale è in un'altra lingua, il riassunto deve essere completamente in italiano.
Estrai i concetti chiave, le idee principali e le informazioni più rilevanti.
Il riassunto deve essere conciso ma completo, mantenendo tutti i punti importanti.

Contenuto da riassumere:
{content}"""
    
    if provider == 'openai' and OPENAI_API_KEY:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=os.getenv('OPENAI_MODEL', 'gpt-4-turbo-preview'),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=int(os.getenv('MAX_TOKENS', '2000')),
            temperature=float(os.getenv('TEMPERATURE', '0.7'))
        )
        return response.choices[0].message.content
    
    elif provider == 'anthropic' and ANTHROPIC_API_KEY:
        response = anthropic.messages.create(
            model=os.getenv('ANTHROPIC_MODEL', 'claude-3-opus'),
            max_tokens=int(os.getenv('MAX_TOKENS', '2000')),
            temperature=float(os.getenv('TEMPERATURE', '0.7')),
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    
    elif provider == 'gemini' and GEMINI_API_KEY:
        model = genai.GenerativeModel(os.getenv('GEMINI_MODEL', 'gemini-pro'))
        response = model.generate_content(prompt)
        return response.text
    
    else:
        raise ValueError(f"LLM provider {provider} not configured or invalid")

def scrape_and_summarize(url: str, vault_path: str = DEFAULT_VAULT_PATH, include_metadata: bool = True, llm_provider: str = DEFAULT_PROVIDER):
    try:
        # Verifica che il vault path esista
        if not os.path.exists(vault_path):
            raise ValueError(f"Il percorso del vault Obsidian non esiste: {vault_path}")

        # Effettua il webscraping con timeout
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Estrai il contenuto principale
        main_content = soup.find('main') or soup.find('article') or soup.find('body')
        content = main_content.get_text(separator='\n', strip=True) if main_content else soup.get_text(separator='\n', strip=True)
        
        # Generate summary using LLM
        summary = get_llm_summary(content, llm_provider)
        
        # Find similar notes
        similar_notes = find_similar_notes(content, vault_path)
        
        # Crea il contenuto della nota Markdown
        title = soup.title.string.strip() if soup.title else "Untitled"
        metadata = {
            "url": url,
            "created": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "tags": ["web-clip", "auto-generated"],
            "llm_provider": llm_provider
        }
        
        if similar_notes:
            metadata["similar_notes"] = similar_notes
        
        # Formatta la nota con o senza metadati YAML
        if include_metadata:
            note_content = "---\n"
            for key, value in metadata.items():
                if isinstance(value, list):
                    note_content += f"{key}:\n"
                    for item in value:
                        note_content += f"  - {item}\n"
                else:
                    note_content += f"{key}: {value}\n"
            note_content += "---\n\n"
        else:
            note_content = ""

        note_content += f"# {title}\n\n## Riassunto\n{summary}\n\n## Fonte\n{url}\n\n## Note Simili\n"
        
        if similar_notes:
            for note in similar_notes:
                note_content += f"- [[{note['path']}]] (Somiglianza: {note['similarity']:.2f})\n"
        else:
            note_content += "_Nessuna nota simile trovata_\n"
        
        note_content += f"\n_Creato il: {metadata['created']}_\n_Generato da: {llm_provider}_"
        
        # Crea il nome del file sanitizzato
        safe_title = ''.join(c if c.isalnum() or c in '-_ ' else '-' for c in title)
        filename = f"{datetime.now().strftime('%Y%m%d')}-{safe_title[:30]}.md".strip('-')
        filepath = os.path.join(vault_path, filename)
        
        # Salva il file nel vault di Obsidian
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(note_content)
        
        return f"✅ Nota creata con successo: {filename}"
    
    except requests.Timeout:
        return "❌ Errore: Timeout durante il recupero della pagina web"
    except requests.RequestException as e:
        return f"❌ Errore di rete: {str(e)}"
    except ValueError as e:
        return f"❌ Errore di validazione: {str(e)}"
    except Exception as e:
        return f"❌ Errore imprevisto: {str(e)}"

def main():
    # Create Gradio interface
    iface = gr.Interface(
        fn=scrape_and_summarize,
        inputs=[
            gr.Textbox(label="URL del sito web", placeholder="https://example.com"),
            gr.Textbox(label="Percorso del vault Obsidian", value=DEFAULT_VAULT_PATH or "", placeholder="/percorso/al/tuo/vault"),
            gr.Checkbox(label="Includi metadati YAML", value=True),
            gr.Dropdown(
                label="LLM Provider",
                choices=["openai", "anthropic", "gemini"],
                value=DEFAULT_PROVIDER
            )
        ],
        outputs=gr.Textbox(label="Stato"),
        title="Obsidian Web Scraper con LLM",
        description="Crea note Markdown nel tuo vault Obsidian con sommari generati da LLM e collegamenti semantici",
        theme=gr.themes.Soft()
    )
    
    iface.launch(share=False)

if __name__ == "__main__":
    main()