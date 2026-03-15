import os
import sys
import subprocess
import asyncio
import json
import getpass
from pathlib import Path

try:
    import pandas as pd
    import requests
    from dotenv import load_dotenv
except ImportError:
    print("Missing dependencies. Installing pandas, openpyxl, requests, and python-dotenv...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "openpyxl", "requests", "python-dotenv"])
    import pandas as pd
    import requests
    from dotenv import load_dotenv

# Import pipeline functions
from pipeline import ingest, query

# Load configuration from .env
load_dotenv()

API_KEY = os.getenv("LUCIO_API_KEY")
TEAM_ID = os.getenv("LUCIO_TEAM_ID")
SUBMISSION_URL = "https://luciohackathon.purplewater-eec0a096.centralindia.azurecontainerapps.io/teams/submissions"

if not API_KEY or not TEAM_ID:
    print("⚠️  Warning: LUCIO_API_KEY or LUCIO_TEAM_ID not directly found in environment. Please check your .env file.")

def check_ollama_status():
    """Ensure Ollama is running and responding."""
    try:
        resp = requests.get("http://127.0.0.1:11434/", timeout=3)
        if resp.status_code == 200:
            print("✅ Ollama is up and running.")
            return True
    except requests.exceptions.RequestException:
        pass
    print("⚠️  Ollama backend does not appear to be running on localhost:11434. Please ensure it is started.")
    return False

async def main():
    print("🚀 Initializing Decryption and Ingestion Pipeline")
    
    zip_path = input("Enter the path to the encrypted ZIP file: ").strip()
    if not os.path.exists(zip_path):
        print(f"❌ Error: ZIP file not found at {zip_path}")
        return

    excel_path = input("Enter the path to the questions Excel file (e.g., questions.xlsx): ").strip()
    if not os.path.exists(excel_path):
        print(f"❌ Error: Excel file not found at {excel_path}")
        return

    # 1. Read Excel File
    print("📊 Loading questions from Excel...")
    try:
        df = pd.read_excel(excel_path)
        # Ensure 'id' and 'questions'/'question' columns exist
        col_names = [c.lower() for c in df.columns]
        
        id_col = next((c for c in df.columns if c.lower() == 'id'), None)
        q_col = next((c for c in df.columns if c.lower() in ['questions', 'question']), None)

        if not id_col or not q_col:
            raise ValueError(f"Could not find 'id' and 'questions' columns. Found: {list(df.columns)}")

        questions_list = df[q_col].tolist()
        ids_list = df[id_col].tolist()
        print(f"✅ Loaded {len(questions_list)} questions mapped and ready.")
    except Exception as e:
        print(f"❌ Failed to parse Excel document: {e}")
        return

    # 2. Start / Validate Backend
    check_ollama_status()

    # 3. Wait for Decryption Key
    decryption_key = getpass.getpass("🔑 Enter decryption key for the ZIP file: ")

    # 4. Decrypt and Extract
    extract_dir = "data/extracted_pdfs"
    os.makedirs(extract_dir, exist_ok=True)
    
    print(f"🔓 Extracting {zip_path} to {extract_dir}...")
    # Use subprocess unzip because the Python standard library zipfile module does 
    # not natively support AES decryption properly without 3rd party packages
    result = subprocess.run(
        ["unzip", "-o", "-P", decryption_key, zip_path, "-d", extract_dir],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print("❌ Failed to unzip. Decryption key might be incorrect or file is corrupted.")
        print("Logs:\n", result.stderr)
        return
        
    print("✅ Extraction successful.")
    
    # Force Env vars for Fast Ingestion and FAISS
    os.environ["FAST_INGEST"] = "1"
    os.environ["VECTOR_BACKEND"] = "faiss"

    # 5. Run Ingestion on Extracted PDFs
    print("⚙️ Running Ingestion Pipeline on extracted documents...")
    await ingest(extract_dir)
    print("✅ Ingestion complete.")

    # 6. Run Query Pipeline
    print("🤖 Running Query Pipeline with loaded questions...")
    query_results = await query(questions_list)
    
    # Remap the question_ids to the actual IDs defined in the user's Excel sheet
    final_answers = []
    # Using enumerate here in case zipped answers are fewer than questions (failsafe)
    for i, answer_obj in enumerate(query_results["answers"]):
        try:
            actual_id = int(ids_list[i])
        except (ValueError, IndexError):
            actual_id = answer_obj["question_id"] # Fallback

        answer_obj["question_id"] = actual_id
        final_answers.append(answer_obj)

    # Prepare payload using exactly the structure required:
    # { "team_id": "...", "answers": [ {"question_id": X, "answer": "...", "citations": [...] } ] }
    payload = {
        "team_id": TEAM_ID,
        "answers": final_answers
    }

    # 7. Make HTTP POST Request
    print(f"📤 Submitting answers to endpoint: {SUBMISSION_URL}")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        resp = requests.post(SUBMISSION_URL, headers=headers, json=payload, timeout=30)
        print("\n📬 HTTP Submission Result:")
        print(f"Status Code: {resp.status_code}")
        try:
            print("Response:", json.dumps(resp.json(), indent=2))
        except ValueError:
            print("Response:", resp.text)
            
        if resp.status_code in [200, 201]:
            print("🎉 Submission Successful!")
        else:
            print("⚠️ Submission returned a non-successful status code.")
    except Exception as e:
        print(f"❌ Failed to submit data over HTTP. Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
