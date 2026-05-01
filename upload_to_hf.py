"""Upload AI emotions experiment data to HuggingFace.

Uploads all data folders with batching for large directories to avoid 504 timeouts.
Splits directories with >9000 files into subdirectories to respect HF's 10k file limit.
"""

import os
import shutil
import tempfile
import time
from pathlib import Path

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

from huggingface_hub import HfApi, create_repo

TOKEN = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
REPO_ID = "LindaP/ai-emotions-v2"
BASE = str(Path(__file__).resolve().parent)
DATA = os.path.join(BASE, "data")

BATCH_SIZE = 500
MAX_FILES_PER_DIR = 9000  # HF limit is 10k, leave margin
MAX_RETRIES = 5
RETRY_DELAY = 30  # seconds

api = HfApi(token=TOKEN)


def upload_with_retry(func, *args, **kwargs):
    """Retry an upload function on 504 or connection errors."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            err = str(e)
            if any(code in err for code in ["504", "502", "503", "Connection", "timeout", "Timeout"]):
                if attempt < MAX_RETRIES:
                    wait = RETRY_DELAY * attempt
                    print(f"    Retry {attempt}/{MAX_RETRIES} after {wait}s: {err[:100]}")
                    time.sleep(wait)
                else:
                    print(f"    Failed after {MAX_RETRIES} attempts: {err[:200]}")
                    raise
            else:
                raise


def upload_flat_dir(folder_path, path_in_repo, desc, file_list=None):
    """Upload a flat directory (no subdirs), with batching for large file lists."""
    if file_list is None:
        file_list = sorted(f for f in os.listdir(folder_path)
                           if os.path.isfile(os.path.join(folder_path, f)))
    total = len(file_list)

    if total == 0:
        print(f"  SKIP (empty): {desc}")
        return

    print(f"  {desc}: {total} files")

    if total <= BATCH_SIZE:
        upload_with_retry(
            api.upload_folder,
            folder_path=folder_path,
            path_in_repo=path_in_repo,
            repo_id=REPO_ID,
            repo_type="dataset",
            allow_patterns=file_list if file_list else None,
        )
        print(f"  Done: {desc}")
        return

    # Batch upload
    batches = [file_list[i:i + BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]
    print(f"  Splitting into {len(batches)} batches of ~{BATCH_SIZE}")

    for i, batch in enumerate(batches):
        print(f"  Batch {i+1}/{len(batches)}: {len(batch)} files ({batch[0]} ... {batch[-1]})")
        upload_with_retry(
            api.upload_folder,
            folder_path=folder_path,
            path_in_repo=path_in_repo,
            repo_id=REPO_ID,
            repo_type="dataset",
            allow_patterns=batch,
        )
        print(f"  Batch {i+1}/{len(batches)} committed.")

    print(f"  Done: {desc}")


def upload_large_dir_split(folder_path, path_in_repo, desc):
    """Upload a directory with >MAX_FILES_PER_DIR files by splitting into repo subdirs.

    Creates temporary directories with symlinks, uploading as part_0/, part_1/, etc.
    """
    all_files = sorted(f for f in os.listdir(folder_path)
                       if os.path.isfile(os.path.join(folder_path, f)))
    total = len(all_files)
    n_parts = (total + MAX_FILES_PER_DIR - 1) // MAX_FILES_PER_DIR
    print(f"  {desc}: {total} files -> splitting into {n_parts} parts of ~{MAX_FILES_PER_DIR}")

    for part_idx in range(n_parts):
        start = part_idx * MAX_FILES_PER_DIR
        end = min(start + MAX_FILES_PER_DIR, total)
        part_files = all_files[start:end]
        part_repo_path = f"{path_in_repo}/part_{part_idx}"
        part_desc = f"{desc}/part_{part_idx}"

        # Create a temp dir with symlinks to the original files
        tmpdir = tempfile.mkdtemp(prefix=f"hf_upload_part{part_idx}_")
        try:
            for f in part_files:
                os.symlink(os.path.join(folder_path, f), os.path.join(tmpdir, f))
            upload_flat_dir(tmpdir, part_repo_path, part_desc, file_list=part_files)
        finally:
            shutil.rmtree(tmpdir)


def upload_folder_batched(folder_path, path_in_repo, desc):
    """Upload a folder, handling subdirs, large dirs, and batching."""
    if not os.path.exists(folder_path):
        print(f"  SKIP (not found): {folder_path}")
        return

    # Get all files (flat, in this directory only)
    all_files = sorted(f for f in os.listdir(folder_path)
                       if os.path.isfile(os.path.join(folder_path, f)))

    # Check for subdirectories (skip __pycache__)
    subdirs = [d for d in os.listdir(folder_path)
               if os.path.isdir(os.path.join(folder_path, d)) and d != "__pycache__"]

    if subdirs:
        # Has subdirectories -- upload each separately
        print(f"  {desc}: {len(subdirs)} subdirectories")
        for subdir in sorted(subdirs):
            subdir_path = os.path.join(folder_path, subdir)
            sub_repo_path = f"{path_in_repo}/{subdir}"
            upload_folder_batched(subdir_path, sub_repo_path, f"{desc}/{subdir}")
        # Also upload any top-level files (excluding .py and __pycache__)
        top_files = [f for f in all_files if not f.endswith('.py')]
        if top_files:
            print(f"  {desc}: {len(top_files)} top-level files")
            upload_flat_dir(folder_path, path_in_repo, f"{desc} (top-level)", file_list=top_files)
        return

    total = len(all_files)

    if total > MAX_FILES_PER_DIR:
        upload_large_dir_split(folder_path, path_in_repo, desc)
    else:
        upload_flat_dir(folder_path, path_in_repo, desc, file_list=all_files)


# --- Main ---

# 1. Create repo
print(f"Creating repo {REPO_ID}...")
try:
    create_repo(REPO_ID, repo_type="dataset", private=True, token=TOKEN)
    print("Repo created.")
except Exception as e:
    if "already" in str(e).lower() or "409" in str(e):
        print("Repo already exists, continuing.")
    else:
        raise

# 2. Upload README
readme_path = os.path.join(BASE, "dataset_readme.md")
print("\nUploading README.md...")
upload_with_retry(
    api.upload_file,
    path_or_fileobj=readme_path,
    path_in_repo="README.md",
    repo_id=REPO_ID,
    repo_type="dataset",
)
print("README uploaded.")

# 3. Upload data folders in order of size (smallest first)
uploads = [
    ("figures", "figures", "Figures"),
    ("steering", "steering", "Steering results"),
    ("stories", "stories", "Stories"),
    ("shutdown", "shutdown", "Shutdown trials"),
    ("vectors", "vectors", "Vectors"),
    ("activations", "activations", "Activations"),
]

for folder_name, repo_path, desc in uploads:
    folder_path = os.path.join(DATA, folder_name)
    print(f"\n{'='*60}")
    print(f"Uploading {desc}: {folder_path} -> {repo_path}")
    print(f"{'='*60}")
    upload_folder_batched(folder_path, repo_path, desc)

print(f"\n{'='*60}")
print("All uploads complete!")
print(f"Dataset URL: https://huggingface.co/datasets/{REPO_ID}")
print(f"{'='*60}")
