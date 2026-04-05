#!/usr/bin/env bash
# Deploy code to Hyperstack VM and optionally run experiments.
# Usage: bash scripts/deploy.sh [IP] [--run-stream1|--run-stream2|--run-stream3]

set -euo pipefail

SSH_KEY="$HOME/.ssh/id_rsa_hyperstack"
REMOTE_USER="ubuntu"
PROJECT_DIR="ai-emotions-v2"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <VM_IP> [--run-stream1|--run-stream2|--run-stream3|--setup]"
    exit 1
fi

VM_IP="$1"
ACTION="${2:-}"
SSH_CMD="ssh -i $SSH_KEY -o StrictHostKeyChecking=no $REMOTE_USER@$VM_IP"

echo "=== Deploying to $VM_IP ==="

# Rsync code (exclude data/ and .git/)
rsync -avz --progress \
    --exclude 'data/' \
    --exclude '.git/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '.venv/' \
    -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no" \
    "$LOCAL_DIR/" "$REMOTE_USER@$VM_IP:~/$PROJECT_DIR/"

echo "=== Code synced ==="

if [ "$ACTION" = "--setup" ]; then
    echo "=== Setting up environment ==="
    $SSH_CMD << 'SETUP_EOF'
cd ~/ai-emotions-v2
pip install --upgrade pip
pip install -e ".[dev]" 2>/dev/null || pip install torch transformers numpy scipy scikit-learn matplotlib seaborn umap-learn tqdm google-generativeai
# Login to HuggingFace for Llama access
echo "Remember to: huggingface-cli login"
SETUP_EOF
fi

if [ "$ACTION" = "--run-stream1" ]; then
    echo "=== Running Stream 1 on GPU ==="
    $SSH_CMD "cd ~/ai-emotions-v2 && nohup python -m scripts.run_stream1 --stage extract --stage vectors > stream1.log 2>&1 &"
    echo "Stream 1 started in background. Check with: ssh -i $SSH_KEY $REMOTE_USER@$VM_IP 'tail -f ~/ai-emotions-v2/stream1.log'"
fi

if [ "$ACTION" = "--run-stream2" ]; then
    echo "=== Running Stream 2 on GPU ==="
    $SSH_CMD "cd ~/ai-emotions-v2 && nohup python -m scripts.run_stream2 --stage extract --stage vectors > stream2.log 2>&1 &"
    echo "Stream 2 started in background."
fi

if [ "$ACTION" = "--run-stream3" ]; then
    echo "=== Running Stream 3 on GPU ==="
    $SSH_CMD "cd ~/ai-emotions-v2 && nohup python -m scripts.run_stream3 > stream3.log 2>&1 &"
    echo "Stream 3 started in background."
fi

# Sync results back
echo ""
echo "To sync results back:"
echo "  rsync -avz -e \"ssh -i $SSH_KEY\" $REMOTE_USER@$VM_IP:~/$PROJECT_DIR/data/ $LOCAL_DIR/data/"
