# 🤖 Obsidian MD Bot

Un'applicazione web basata su Gradio per il webscraping e la creazione automatica di note Obsidian utilizzando modelli di linguaggio avanzati.

## 🌟 Caratteristiche

- 🔍 Webscraping di contenuti web
- 📝 Generazione automatica di note in formato Markdown in italiano
- 🧠 Integrazione con diversi modelli di linguaggio (OpenAI, Anthropic, Google Gemini)
- 🔗 Rilevamento di note semanticamente simili nel vault Obsidian
- 🌐 Interfaccia web intuitiva con Gradio

## 🛠️ Requisiti

- Python 3.9 o superiore
- Un vault Obsidian esistente
- Chiavi API per almeno uno dei servizi di AI supportati

## 📦 Installazione

### Con Poetry (Consigliato)

```bash
# Clona il repository
git clone https://github.com/lucabaroncini/obsidian-md-bot.git
cd obsidian-md-bot

# Installa Poetry se non è già installato
pip install poetry

# Installa le dipendenze con Poetry
poetry install

# Attiva l'ambiente virtuale
poetry shell

# Verifica l'installazione
python -c "import gradio, openai, anthropic, google.generativeai"
```

### Con pip e venv

```bash
# Clona il repository
git clone https://github.com/lucabaroncini/obsidian-md-bot.git
cd obsidian-md-bot

# Crea un nuovo ambiente virtuale
python -m venv venv

# Attiva l'ambiente virtuale
source venv/bin/activate  # Linux/macOS
.\venv\Scripts\activate  # Windows

# Aggiorna pip all'ultima versione
pip install --upgrade pip

# Installa le dipendenze
pip install -r requirements.txt

# Verifica l'installazione
python -c "import gradio, openai, anthropic, google.generativeai"
```

### Risoluzione dei problemi

- Se riscontri errori durante l'installazione delle dipendenze, prova ad aggiornare pip:
  ```bash
  pip install --upgrade pip
  ```

- Per problemi con le librerie di machine learning:
  ```bash
  pip install --upgrade torch sentence-transformers
  ```

- Se l'ambiente virtuale non si attiva correttamente:
  - Linux/macOS: Verifica i permessi della cartella venv
  - Windows: Esegui PowerShell come amministratore


## ⚙️ Configurazione

1. Copia il file `.env-default` in `.env`:
```bash
cp .env-default .env
```

2. Modifica il file `.env` con le tue configurazioni:
```env
OBSIDIAN_VAULT_PATH=/percorso/al/tuo/vault/obsidian
OPENAI_API_KEY=la-tua-chiave-api  # opzionale
ANTHROPIC_API_KEY=la-tua-chiave-api  # opzionale
GOOGLE_API_KEY=la-tua-chiave-api  # opzionale
```

## 🚀 Avvio

### Con Poetry

```bash
poetry run python src/main.py
```

### Con pip e venv

```bash
source venv/bin/activate && python src/main.py
```

L'interfaccia web sarà disponibile all'indirizzo http://127.0.0.1:7860

## 📚 Utilizzo

1. Apri l'interfaccia web nel tuo browser
2. Incolla l'URL della pagina web da cui vuoi estrarre contenuti
3. Seleziona il modello di linguaggio da utilizzare
4. Clicca su "Genera" per creare la nota Markdown
5. La nota verrà salvata automaticamente nel tuo vault Obsidian

## 🤝 Contribuire

Sei il benvenuto a contribuire al progetto! Puoi farlo attraverso:
- Segnalazione di bug
- Suggerimenti di nuove funzionalità
- Pull request

## 📄 Licenza

Questo progetto è distribuito sotto licenza MIT. Vedi il file `LICENSE` per maggiori dettagli.
