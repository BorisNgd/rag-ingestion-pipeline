"""
Bulk folder ingestion script.
Scans a directory (recursively) and submits every supported file to the pipeline API.

Usage:
    python ingest_folder.py                          # uses ./ressources/ by default
    python ingest_folder.py ./ressources/Administration
    python ingest_folder.py /path/to/docs --watch    # re-scan every 60s for new files
"""
from __future__ import annotations

import argparse
import mimetypes
import sys
import time
from pathlib import Path

import httpx

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_URL = "http://localhost:8000"

# Read API key from .env automatically
def _load_api_key() -> str:
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("API_KEY="):
                return line.split("=", 1)[1].strip()
    return ""

API_KEY = _load_api_key()

# Supported extensions → MIME type
SUPPORTED = {
    # PDF
    ".pdf": "application/pdf",
    # Word
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc":  "application/msword",
    # Excel
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xls":  "application/vnd.ms-excel",
    ".csv":  "text/csv",
    # Web
    ".html": "text/html",
    ".htm":  "text/html",
    # Markdown
    ".md":       "text/markdown",
    ".markdown": "text/markdown",
    # Images (OCR)
    ".png":  "image/png",
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".tiff": "image/tiff",
    ".bmp":  "image/bmp",
    # Audio (transcription)
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    # Code
    ".py":   "text/x-python",
    ".js":   "text/javascript",
    ".ts":   "text/typescript",
    ".java": "text/x-java",
    ".go":   "text/x-go",
    ".rs":   "text/x-rust",
    ".cpp":  "text/x-c++",
    ".c":    "text/x-c",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_files(directory: Path) -> list[Path]:
    """Return all supported files in directory (recursive)."""
    found = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            found.append(path)
    return found


def ingest_file(client: httpx.Client, file_path: Path, tags: list[str]) -> dict | None:
    """POST a single file to /api/v1/ingest. Returns response dict or None on error."""
    suffix = file_path.suffix.lower()
    mime = SUPPORTED.get(suffix, "application/octet-stream")

    try:
        with file_path.open("rb") as f:
            response = client.post(
                f"{API_URL}/api/v1/ingest",
                headers={"x-api-key": API_KEY},
                files={"file": (file_path.name, f, mime)},
                data={"tags": ",".join(tags)},
                timeout=120,
            )
        if response.status_code == 202:
            return response.json()
        else:
            print(f"  [ERREUR {response.status_code}] {response.text[:200]}")
            return None
    except httpx.RequestError as e:
        print(f"  [CONNEXION ECHOUEE] {e}")
        return None


def poll_job(client: httpx.Client, job_id: str, max_wait: int = 300) -> str:
    """Poll job status until terminal state. Returns final status string."""
    terminal = {"completed", "failed", "dlq"}
    for _ in range(max_wait // 5):
        try:
            r = client.get(
                f"{API_URL}/api/v1/jobs/{job_id}",
                headers={"x-api-key": API_KEY},
                timeout=120,
            )
            if r.status_code == 200:
                status = r.json().get("status", "unknown")
                if status.lower() in terminal:
                    return status
        except httpx.RequestError:
            pass
        time.sleep(5)
    return "timeout"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(directory: Path, tags: list[str], watch: bool, poll: bool) -> None:
    already_submitted: set[Path] = set()

    with httpx.Client() as client:
        while True:
            files = find_files(directory)
            new_files = [f for f in files if f not in already_submitted]

            if not new_files:
                if not watch:
                    print("Aucun nouveau fichier supporté trouvé.")
                    break
            else:
                print(f"\n{len(new_files)} fichier(s) à ingérer dans '{directory}'")
                print("-" * 60)

                for i, file_path in enumerate(new_files, 1):
                    rel = file_path.relative_to(directory)
                    print(f"[{i}/{len(new_files)}] {rel} ({file_path.stat().st_size // 1024} KB)", end=" ... ")

                    result = ingest_file(client, file_path, tags)
                    already_submitted.add(file_path)

                    if result:
                        job_id = result["job_id"]
                        doc_id = result["document_id"]
                        print(f"OK → job_id={job_id[:8]}... doc_id={doc_id[:8]}...")

                        if poll:
                            print(f"     Attente du résultat", end="", flush=True)
                            final = poll_job(client, job_id)
                            print(f" → {final.upper()}")
                    else:
                        print("ECHEC")

            if not watch:
                break

            print(f"\nMode surveillance actif — prochaine vérification dans 60s (Ctrl+C pour quitter)")
            time.sleep(60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingestion en masse depuis un répertoire")
    parser.add_argument(
        "directory",
        nargs="?",
        default="ressources",
        help="Répertoire à scanner (défaut: ./ressources)",
    )
    parser.add_argument(
        "--tags", "-t",
        default="",
        help="Tags séparés par des virgules (ex: urgence,administration)",
    )
    parser.add_argument(
        "--watch", "-w",
        action="store_true",
        help="Surveiller le répertoire en continu pour les nouveaux fichiers",
    )
    parser.add_argument(
        "--poll", "-p",
        action="store_true",
        help="Attendre la fin du traitement pour chaque fichier avant de continuer",
    )
    args = parser.parse_args()

    directory = Path(args.directory)
    if not directory.exists():
        print(f"Répertoire introuvable: {directory}")
        sys.exit(1)

    tags = [t.strip() for t in args.tags.split(",") if t.strip()]

    print(f"Répertoire  : {directory.resolve()}")
    print(f"API         : {API_URL}")
    print(f"Tags        : {tags or '(aucun)'}")
    print(f"Surveillance: {'oui' if args.watch else 'non'}")
    print(f"Attente     : {'oui' if args.poll else 'non'}")

    run(directory, tags, watch=args.watch, poll=args.poll)


if __name__ == "__main__":
    main()
