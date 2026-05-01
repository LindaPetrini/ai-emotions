"""Upload remaining activation data to HuggingFace.

The initial upload hit the 10k files/directory limit. This script handles:
1. llama-8b-base: 11,521 files, 10k already uploaded flat -> upload remaining 1,521 to overflow dir
2. llama-8b-inst: 16,930 files, not started -> split into part_0, part_1
3. qwen-7b-base: 10,081 files, not started -> split into part_0, part_1
4. qwen-7b-inst: 10,081 files, not started -> split into part_0, part_1

Also removes __pycache__ from shutdown and the .py file.
"""

import os
import shutil
import tempfile
import time
from pathlib import Path

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

from huggingface_hub import HfApi

TOKEN = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
REPO_ID = "LindaP/ai-emotions-v2"
BASE = str(Path(__file__).resolve().parent)
DATA = os.path.join(BASE, "data")

BATCH_SIZE = 200
MAX_FILES_PER_DIR = 9500  # HF limit is 10k, leave margin
MAX_RETRIES = 10
RETRY_DELAY = 30
RATE_LIMIT_DELAY = 600  # 10 minutes for 429 rate limits

api = HfApi(token=TOKEN)


def upload_with_retry(func, *args, **kwargs):
    """Retry on transient errors including 429 rate limits."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            err = str(e)
            if "429" in err:
                wait = RATE_LIMIT_DELAY * attempt  # exponential: 600, 1200, 1800, ...
                print(f"    Rate limited (429). Waiting {wait}s before retry {attempt}/{MAX_RETRIES}: {err[:100]}")
                time.sleep(wait)
                if attempt == MAX_RETRIES:
                    raise
            elif any(code in err for code in ["504", "502", "503", "Connection", "timeout", "Timeout"]):
                if attempt < MAX_RETRIES:
                    wait = RETRY_DELAY * attempt
                    print(f"    Retry {attempt}/{MAX_RETRIES} after {wait}s: {err[:100]}")
                    time.sleep(wait)
                else:
                    raise
            else:
                raise


def upload_batch(folder_path, path_in_repo, file_list, desc):
    """Upload a batch of files from a folder."""
    total = len(file_list)
    if total == 0:
        return

    batches = [file_list[i:i + BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]
    print(f"  {desc}: {total} files in {len(batches)} batch(es)")

    for i, batch in enumerate(batches):
        print(f"    Batch {i+1}/{len(batches)}: {len(batch)} files ({batch[0]} ... {batch[-1]})")
        upload_with_retry(
            api.upload_folder,
            folder_path=folder_path,
            path_in_repo=path_in_repo,
            repo_id=REPO_ID,
            repo_type="dataset",
            allow_patterns=batch,
        )
        print(f"    Batch {i+1}/{len(batches)} done.")
        if i < len(batches) - 1:
            print(f"    Cooling down 30s before next batch...")
            time.sleep(30)


def upload_via_symlink_dir(source_dir, file_list, path_in_repo, desc):
    """Create a temp dir with symlinks and upload as a virtual subdirectory.
    Skips files already uploaded to the destination path."""
    already = get_already_uploaded(path_in_repo)
    remaining = sorted(set(file_list) - already)
    if not remaining:
        print(f"  {desc}: all {len(file_list)} files already uploaded, skipping.")
        return
    if already:
        print(f"  {desc}: {len(already)} already uploaded, {len(remaining)} remaining")

    tmpdir = tempfile.mkdtemp(prefix="hf_upload_")
    try:
        for f in remaining:
            os.symlink(os.path.join(source_dir, f), os.path.join(tmpdir, f))
        upload_batch(tmpdir, path_in_repo, remaining, desc)
    finally:
        shutil.rmtree(tmpdir)


def get_already_uploaded(path_in_repo):
    """Get set of filenames already in a repo directory."""
    try:
        items = list(api.list_repo_tree(REPO_ID, path_in_repo=path_in_repo,
                                         repo_type="dataset", recursive=False))
        return {os.path.basename(item.path) for item in items
                if hasattr(item, 'size')}  # files only (have size attr)
    except Exception:
        return set()


# ============================================================
# 1. llama-8b-base: 10k already uploaded, ~1.5k remaining
# ============================================================
print("\n" + "="*60)
print("Step 1: llama-8b-base remaining activations")
print("="*60)

llama_base_dir = os.path.join(DATA, "activations/llama-8b-base")
all_llama_base = sorted(f for f in os.listdir(llama_base_dir)
                        if os.path.isfile(os.path.join(llama_base_dir, f)))
print(f"Total files on disk: {len(all_llama_base)}")

already = get_already_uploaded("activations/llama-8b-base")
print(f"Already uploaded: {len(already)}")

remaining = sorted(set(all_llama_base) - already)
print(f"Remaining: {len(remaining)}")

if remaining:
    # Upload remaining to an overflow directory since main dir is at 10k
    # upload_via_symlink_dir will itself check what's already in the overflow dir
    upload_via_symlink_dir(llama_base_dir, remaining,
                           "activations/llama-8b-base-overflow",
                           "llama-8b-base overflow")
else:
    print("  All files already uploaded.")


# ============================================================
# 2. qwen-7b-base: 10,081 files -> split into parts
# ============================================================
print("\n" + "="*60)
print("Step 2: qwen-7b-base activations (split)")
print("="*60)

qwen_base_dir = os.path.join(DATA, "activations/qwen-7b-base")
all_qwen_base = sorted(f for f in os.listdir(qwen_base_dir)
                        if os.path.isfile(os.path.join(qwen_base_dir, f)))
total = len(all_qwen_base)
print(f"Total files: {total}")

if total <= MAX_FILES_PER_DIR:
    upload_batch(qwen_base_dir, "activations/qwen-7b-base", all_qwen_base, "qwen-7b-base")
else:
    n_parts = (total + MAX_FILES_PER_DIR - 1) // MAX_FILES_PER_DIR
    for p in range(n_parts):
        start = p * MAX_FILES_PER_DIR
        end = min(start + MAX_FILES_PER_DIR, total)
        part_files = all_qwen_base[start:end]
        upload_via_symlink_dir(qwen_base_dir, part_files,
                               f"activations/qwen-7b-base/part_{p}",
                               f"qwen-7b-base part_{p}")


# ============================================================
# 3. qwen-7b-inst: 10,081 files -> split into parts
# ============================================================
print("\n" + "="*60)
print("Step 3: qwen-7b-inst activations (split)")
print("="*60)

qwen_inst_dir = os.path.join(DATA, "activations/qwen-7b-inst")
all_qwen_inst = sorted(f for f in os.listdir(qwen_inst_dir)
                        if os.path.isfile(os.path.join(qwen_inst_dir, f)))
total = len(all_qwen_inst)
print(f"Total files: {total}")

if total <= MAX_FILES_PER_DIR:
    upload_batch(qwen_inst_dir, "activations/qwen-7b-inst", all_qwen_inst, "qwen-7b-inst")
else:
    n_parts = (total + MAX_FILES_PER_DIR - 1) // MAX_FILES_PER_DIR
    for p in range(n_parts):
        start = p * MAX_FILES_PER_DIR
        end = min(start + MAX_FILES_PER_DIR, total)
        part_files = all_qwen_inst[start:end]
        upload_via_symlink_dir(qwen_inst_dir, part_files,
                               f"activations/qwen-7b-inst/part_{p}",
                               f"qwen-7b-inst part_{p}")


# ============================================================
# 4. llama-8b-inst: 16,930 files -> split into parts
# ============================================================
print("\n" + "="*60)
print("Step 4: llama-8b-inst activations (split)")
print("="*60)

llama_inst_dir = os.path.join(DATA, "activations/llama-8b-inst")
all_llama_inst = sorted(f for f in os.listdir(llama_inst_dir)
                         if os.path.isfile(os.path.join(llama_inst_dir, f)))
total = len(all_llama_inst)
print(f"Total files: {total}")

n_parts = (total + MAX_FILES_PER_DIR - 1) // MAX_FILES_PER_DIR
for p in range(n_parts):
    start = p * MAX_FILES_PER_DIR
    end = min(start + MAX_FILES_PER_DIR, total)
    part_files = all_llama_inst[start:end]
    upload_via_symlink_dir(llama_inst_dir, part_files,
                           f"activations/llama-8b-inst/part_{p}",
                           f"llama-8b-inst part_{p}")


print("\n" + "="*60)
print("All remaining uploads complete!")
print(f"Dataset URL: https://huggingface.co/datasets/{REPO_ID}")
print("="*60)
