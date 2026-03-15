import os
import sys
import subprocess
import asyncio
import json
import getpass
from pathlib import Path

# Query-only mode defaults to enabled so existing indexes can be reused.
SKIP_INGEST = os.getenv("SKIP_INGEST", "1") == "1"

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
SUBMISSION_URL = "https://luciohackathon.purplewater-eec0a096.centralindia.azurecontainerapps.io/submissions"

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

    zip_path = "hackathon_data.zip"
    if not os.path.exists(zip_path):
        print(f"❌ Error: ZIP file not found at {zip_path}")
        return

    questions_path = "Hackathon Final Questions.csv"
    if not os.path.exists(questions_path):
        print(f"❌ Error: file not found at {questions_path}")
        return

    # 1. Read File
    print("📊 Loading questions...")
    try:
        if questions_path.lower().endswith('.csv'):
            df = pd.read_csv(questions_path)
        else:
            df = pd.read_excel(questions_path)
        # Ensure 'id' and 'questions'/'question' columns exist
        col_names = [str(c).lower().strip() for c in df.columns]

        id_col = next((c for c in df.columns if str(c).lower().strip() == 'id'), None)
        q_col = next((c for c in df.columns if str(c).lower().strip() in ['questions', 'question', 'query']), None)

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

    if SKIP_INGEST:
        print("⏭️  SKIP_INGEST=1, skipping decrypt and ingestion. Reusing existing indexes.")
    else:
        # 3. Wait for Decryption Key
        print("🔑 Fetching decryption key from API...")
        decryption_key = None
        while not decryption_key:
            try:
                pw_resp = requests.get(
                    "https://luciohackathon.purplewater-eec0a096.centralindia.azurecontainerapps.io/password",
                    headers={"X-API-Key": API_KEY},
                    timeout=10
                )
                if pw_resp.status_code == 200:
                    try:
                        data = pw_resp.json()
                        decryption_key = data.get("password", data) if isinstance(data, dict) else data
                    except Exception:
                        decryption_key = pw_resp.text.strip().strip('"')
                    print("✅ Successfully retrieved decryption key.")
                    break
                else:
                    print(f"⏳ Waiting for password (Status {pw_resp.status_code}). Retrying in 5 seconds...")
                    await asyncio.sleep(5)
            except Exception as e:
                print(f"❌ Error fetching password: {e}. Retrying in 5 seconds...")
                await asyncio.sleep(5)

        # 4. Decrypt and Extract
        extract_dir = "data/extracted_pdfs"
        os.makedirs(extract_dir, exist_ok=True)

        print(f"🔓 Extracting {zip_path} to {extract_dir}...")
        # Use 7z to extract the archive
        # e: extract, -o: output directory, -p: password, -y: assume yes
        result = subprocess.run(
            ["7z", "x", f"-p{decryption_key}", f"-o{extract_dir}", "-y", zip_path],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print("❌ Failed to extract using 7z. Decryption key might be incorrect, file is corrupted, or 7z is not installed.")
            print("Logs:\n", result.stderr)
            print("Stdout:\n", result.stdout)
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
