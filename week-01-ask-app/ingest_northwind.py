# ingest_northwind.py - send every document in a zip file to the /ingest endpoint.
#
# Usage (with the backend running):
#   python ingest_northwind.py ~/Downloads/northwind-sample-docs.zip
#   python ingest_northwind.py ~/Downloads/northwind-sample-docs.zip --api-url https://your-app.onrender.com

import argparse
import sys
import zipfile
from pathlib import PurePosixPath

import requests

TEXT_EXTENSIONS = {".txt", ".md"}


def is_document(name):
    """Skip folders, macOS metadata and anything that isn't a text file."""
    path = PurePosixPath(name)
    return (
        not name.endswith("/")
        and "__MACOSX" not in path.parts
        and not path.name.startswith(".")
        and path.suffix.lower() in TEXT_EXTENSIONS
    )


def main():
    parser = argparse.ArgumentParser(description="Ingest every text file in a zip into the Ask App.")
    parser.add_argument("zip_path", help="Path to the zip file, e.g. northwind-sample-docs.zip")
    parser.add_argument("--api-url", default="http://localhost:8000", help="Backend URL (default: %(default)s)")
    args = parser.parse_args()
    api_url = args.api_url.rstrip("/")

    failures = 0
    with zipfile.ZipFile(args.zip_path) as archive:
        for name in sorted(filter(is_document, archive.namelist())):
            # The file name without its extension becomes the document_id, e.g. "doc2_expenses".
            document_id = PurePosixPath(name).stem
            text = archive.read(name).decode("utf-8")
            # Generous timeout: a Render free service can take a minute to wake up.
            response = requests.post(
                f"{api_url}/ingest",
                json={"document_id": document_id, "text": text},
                timeout=120,
            )
            if response.ok:
                print(f"OK      {document_id}: {response.json()['chunks']} chunks")
            else:
                failures += 1
                print(f"FAILED  {document_id}: HTTP {response.status_code} {response.text}")

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
