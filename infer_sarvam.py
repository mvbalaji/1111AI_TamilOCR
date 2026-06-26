"""
infer_sarvam.py — inference wrapper for Sarvam Vision Document Intelligence API.

Sarvam uses an async job-based workflow:
  1. POST /doc-digitization/job/v1          → job_id
  2. POST /doc-digitization/job/v1/{id}/urls → upload URLs (Azure Blob)
  3. PUT  <upload_url>                       → upload image
  4. POST /doc-digitization/job/v1/{id}/start
  5. GET  /doc-digitization/job/v1/{id}      → poll until Completed
  6. GET  /doc-digitization/job/v1/{id}/download-urls → result URLs
  7. GET  <download_url>                     → fetch OCR text (markdown)

Requirements:
  pip install requests

Setup:
  export SARVAM_API_KEY=<your key from https://dashboard.sarvam.ai>

Usage:
  python infer_sarvam.py data/manifests/gate.jsonl
  python infer_sarvam.py data/manifests/gate.jsonl --max_samples 50
  python infer_sarvam.py data/manifests/gate.jsonl --language ta-IN

Outputs: results/sarvam/<manifest_stem>.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RESULTS_DIR = Path("results") / "sarvam"
MODEL_ID    = "sarvam-vision"
BASE_URL    = "https://api.sarvam.ai/doc-digitization/job/v1"
POLL_INTERVAL = 3   # seconds between status checks
MAX_POLL      = 60  # max attempts (~3 min per image)


def _headers(api_key: str) -> dict:
    return {"api-subscription-key": api_key, "Content-Type": "application/json"}


def create_job(api_key: str, language: str = "ta-IN") -> str:
    import requests
    payload = {"job_parameters": {"language": language, "output_format": "md"}}
    resp = requests.post(BASE_URL, json=payload, headers=_headers(api_key), timeout=30)
    if not resp.ok:
        raise RuntimeError(f"create_job failed {resp.status_code}: {resp.text[:300]}")
    return resp.json()["job_id"]


def get_upload_url(api_key: str, job_id: str, filename: str) -> tuple[str, str]:
    """Returns (upload_url, file_id)."""
    import requests
    url  = f"{BASE_URL}/{job_id}/urls"
    payload = {"files": [{"file_name": filename}]}
    resp = requests.post(url, json=payload, headers=_headers(api_key), timeout=30)
    if not resp.ok:
        raise RuntimeError(f"get_upload_url failed {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    # Response: {"upload_urls": [{"url": ..., "file_id": ...}]}
    entry = data["upload_urls"][0]
    return entry["url"], entry.get("file_id", filename)


def upload_file(upload_url: str, image_path: str) -> None:
    import requests
    with open(image_path, "rb") as f:
        content = f.read()
    resp = requests.put(
        upload_url, data=content,
        headers={"Content-Type": "image/png", "x-ms-blob-type": "BlockBlob"},
        timeout=60,
    )
    if not resp.ok:
        raise RuntimeError(f"upload failed {resp.status_code}: {resp.text[:200]}")


def start_job(api_key: str, job_id: str) -> None:
    import requests
    url  = f"{BASE_URL}/{job_id}/start"
    resp = requests.post(url, json={}, headers=_headers(api_key), timeout=30)
    if not resp.ok:
        raise RuntimeError(f"start_job failed {resp.status_code}: {resp.text[:300]}")


def poll_job(api_key: str, job_id: str) -> str:
    """Poll until job state is Completed or Failed. Returns final state."""
    import requests
    url = f"{BASE_URL}/{job_id}"
    for _ in range(MAX_POLL):
        resp = requests.get(url, headers=_headers(api_key), timeout=30)
        if not resp.ok:
            raise RuntimeError(f"poll failed {resp.status_code}: {resp.text[:300]}")
        state = resp.json().get("job_state", "")
        if state in ("Completed", "Failed", "Error"):
            return state
        time.sleep(POLL_INTERVAL)
    raise TimeoutError(f"Job {job_id} did not complete in {MAX_POLL * POLL_INTERVAL}s")


def get_result(api_key: str, job_id: str) -> str:
    """Fetch download URL and retrieve OCR text."""
    import requests
    url  = f"{BASE_URL}/{job_id}/download-urls"
    resp = requests.get(url, headers=_headers(api_key), timeout=30)
    if not resp.ok:
        raise RuntimeError(f"download_urls failed {resp.status_code}: {resp.text[:300]}")
    download_urls = resp.json().get("download_urls", [])
    if not download_urls:
        raise RuntimeError("No download URLs returned")
    # Fetch first result file (markdown text)
    dl_resp = requests.get(download_urls[0]["url"], timeout=30)
    dl_resp.raise_for_status()
    return dl_resp.text.strip()


def infer_one(api_key: str, image_path: str, language: str = "ta-IN") -> str:
    filename = Path(image_path).name
    job_id   = create_job(api_key, language)
    upload_url, _ = get_upload_url(api_key, job_id, filename)
    upload_file(upload_url, image_path)
    start_job(api_key, job_id)
    state = poll_job(api_key, job_id)
    if state != "Completed":
        raise RuntimeError(f"Job {job_id} ended with state={state}")
    return get_result(api_key, job_id)


def run(manifest_path: str, max_samples: int | None = None,
        language: str = "ta-IN") -> None:
    api_key = os.environ.get("SARVAM_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "SARVAM_API_KEY not set.\n"
            "Get your key from https://dashboard.sarvam.ai\n"
            "Then: export SARVAM_API_KEY=your_key"
        )

    records = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line.strip())
            if r:
                records.append(r)

    if max_samples:
        records = records[:max_samples]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stem     = Path(manifest_path).stem
    out_path = RESULTS_DIR / f"{stem}.jsonl"

    print(f"Sarvam Vision — {len(records)} records → {out_path}", flush=True)

    with open(out_path, "w", encoding="utf-8") as out:
        for i, rec in enumerate(records):
            t0 = time.time()
            try:
                pred  = infer_one(api_key, rec["image_path"], language)
                error = None
            except Exception as exc:
                pred  = ""
                error = str(exc)
                print(f"  ERROR [{rec['id']}]: {exc}")

            row = {
                "id":           rec["id"],
                "script":       rec.get("script", ""),
                "mode":         rec.get("mode", ""),
                "ground_truth": rec["ground_truth"],
                "prediction":   pred,
                "model":        MODEL_ID,
                "elapsed_s":    round(time.time() - t0, 3),
            }
            if error:
                row["error"] = error
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(records)}", flush=True)

    print(f"Done. Results → {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Sarvam Vision Document Intelligence OCR")
    ap.add_argument("manifest",       help="JSONL manifest from datagen.py")
    ap.add_argument("--max_samples",  type=int, default=None)
    ap.add_argument("--language",     default="ta-IN",
                    help="BCP-47 language code (default: ta-IN for Tamil)")
    args = ap.parse_args()
    run(args.manifest, args.max_samples, args.language)


if __name__ == "__main__":
    main()
