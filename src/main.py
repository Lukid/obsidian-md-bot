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
import logging
import re
from urllib.parse import urlparse
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure LLM providers
DEFAULT_PROVIDER = os.getenv('DEFAULT_LLM_PROVIDER', 'openai')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Validate configuration
if not any([OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY]):
    raise ValueError("Nessun provider LLM configurato. Configurare almeno un provider nelle variabili d'ambiente.")

# Initialize LLM clients
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Initialize sentence transformer for semantic similarity
embedding_model = None  # Lazy loading

# Load Obsidian vault path
DEFAULT_VAULT_PATH = os.getenv('OBSIDIAN_VAULT_PATH')
GRAPH_SIMILARITY_THRESHOLD = float(os.getenv('GRAPH_SIMILARITY_THRESHOLD', '0.85').split('#')[0].strip())
MAX_CONTENT_SIZE = 10 * 1024 * 1024  # 10MB

def validate_url(url: str) -> bool:
    """Validate URL format and scheme."""
    try:
        result = urlparse(url)
        return all([result.scheme in ['http', 'https'], result.netloc])
    except:
        return False

def sanitize_filename(title: str) -> str:
    """Sanitize filename removing invalid characters and limiting length."""
    # Remove invalid characters
    safe_title = re.sub(r'[<>:"/\\|?*]', '-', title)
    # Limit length while preserving file extension
    return safe_title[:100]

def get_unique_filename(base_path: str, filename: str) -> str:
    """Generate unique filename if file already exists."""
    name, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename
    while os.path.exists(os.path.join(base_path, new_filename)):
        new_filename = f"{name}_{counter}{ext}"
        counter += 1
    return new_filename

def lazy_load_embedding_model():
    """Lazy load the embedding model when needed."""
    global embedding_model
    if embedding_model is None:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return embedding_model

def find_similar_notes(content: str, vault_path: str) -> List[Dict]:
    """Find semantically similar notes in the vault."""
    try:
        # Ensure the embedding model is loaded
        model = lazy_load_embedding_model()
        
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
        
        if not content.strip():
            logger.warning("Contenuto vuoto dopo la pulizia, impossibile cercare note simili")
            return []
        
        content_embedding = model.encode(content)
        
        for root, _, files in os.walk(vault_path):
            for file in files:
                if file.endswith('.md'):
                    try:
                        file_path = os.path.join(root, file)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            note_content = f.read()
                            # Rimuovi metadati YAML dalla nota esistente
                            if note_content.startswith('---'):
                                content_parts = note_content.split('---', 2)
                                if len(content_parts) >= 3:
                                    note_content = content_parts[2].strip()
                            
                            if note_content.strip():
                                note_embedding = model.encode(note_content)
                                similarity = util.pytorch_cos_sim(content_embedding, note_embedding)
                                
                                if similarity > GRAPH_SIMILARITY_THRESHOLD:
                                    similar_notes.append({
                                        'title': file,
                                        'path': os.path.relpath(file_path, vault_path),
                                        'similarity': float(similarity)
                                    })
                    except Exception as e:
                        logger.error(f"Errore nell'elaborazione del file {file}: {str(e)}")
                        continue
        
        return sorted(similar_notes, key=lambda x: x['similarity'], reverse=True)
    except Exception as e:
        logger.error(f"Errore nella ricerca di note simili: {str(e)}")
        return []

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
        try:
            # Inizializza il client qui, quando serve
            anthropic = Anthropic(api_key=ANTHROPIC_API_KEY)
            message = anthropic.messages.create(
                model=os.getenv('ANTHROPIC_MODEL', 'claude-3-opus-20240229'),
                max_tokens=int(os.getenv('MAX_TOKENS', '2000')),
                temperature=float(os.getenv('TEMPERATURE', '0.7')),
                system="Sei un assistente esperto nella creazione di riassunti in italiano...",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            return message.content[0].text
        except Exception as e:
            logger.error(f"Errore nell'utilizzo di Anthropic: {str(e)}")
            raise ValueError(f"Errore nella generazione del riassunto con Anthropic: {str(e)}")
    
    elif provider == 'gemini' and GEMINI_API_KEY:
        model = genai.GenerativeModel(os.getenv('GEMINI_MODEL', 'gemini-pro'))
        response = model.generate_content(prompt)
        return response.text
    
    else:
        raise ValueError(f"LLM provider {provider} not configured or invalid")

def scrape_and_summarize(url: str, vault_path: str = DEFAULT_VAULT_PATH, include_metadata: bool = True, llm_provider: str = DEFAULT_PROVIDER):
    try:
        # Validate inputs
        if not validate_url(url):
            raise ValueError("URL non valido")
        
        if not os.path.isdir(vault_path):
            raise ValueError(f"Il percorso del vault non esiste o non è accessibile: {vault_path}")
        
        if not os.access(vault_path, os.W_OK):
            raise ValueError(f"Permessi insufficienti per scrivere nel vault: {vault_path}")

        # Effettua il webscraping con timeout e limiti
        response = requests.get(
            url, 
            timeout=10,
            headers={'User-Agent': 'Obsidian-MD-Bot/1.0'},
            stream=True
        )
        response.raise_for_status()
        
        # Check content length
        content_length = int(response.headers.get('content-length', 0))
        if content_length > MAX_CONTENT_SIZE:
            raise ValueError(f"Contenuto troppo grande: {content_length} bytes")
        
        content = response.text
        soup = BeautifulSoup(content, 'html.parser')
        
        # Estrai il contenuto principale
        main_content = soup.find('main') or soup.find('article') or soup.find('body')
        content = main_content.get_text(separator='\n', strip=True) if main_content else soup.get_text(separator='\n', strip=True)
        
        if not content.strip():
            raise ValueError("Nessun contenuto estratto dalla pagina")

        # Generate summary using LLM
        try:
            summary = get_llm_summary(content, llm_provider)
        except Exception as e:
            logger.error(f"Errore nella generazione del sommario: {str(e)}")
            raise ValueError(f"Errore nella generazione del sommario con {llm_provider}")

        # Find similar notes
        try:
            similar_notes = find_similar_notes(content, vault_path)
        except Exception as e:
            logger.error(f"Errore nella ricerca di note simili: {str(e)}")
            similar_notes = []

        # Crea il contenuto della nota Markdown
        title = soup.title.string.strip() if soup.title else "Untitled"
        safe_title = sanitize_filename(title)
        
        # Generate unique filename
        filename = f"{datetime.now().strftime('%Y%m%d')}-{safe_title}.md"
        filename = get_unique_filename(vault_path, filename)
        filepath = os.path.join(vault_path, filename)
        
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
                value=DEFAULT_PROVIDER,
                allow_custom_value=True
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