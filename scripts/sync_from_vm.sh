#!/bin/bash
# Sync activation and vector data from GPU VM back to local
VM_IP="69.19.137.99"
SSH_KEY="$HOME/.ssh/id_rsa_hyperstack"
KNOWN_HOSTS_FILE="$HOME/.ssh/known_hosts_ai_emotions"
SSH_OPTS="-i $SSH_KEY -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$KNOWN_HOSTS_FILE"
LOCAL_ROOT="$(dirname "$(realpath "$0")")/.."
REMOTE_PROJECT_DIR="${REMOTE_PROJECT_DIR:-$(basename "$LOCAL_ROOT")}"
REMOTE_DIR="/home/ubuntu/$REMOTE_PROJECT_DIR/data"
LOCAL_DIR="$(dirname "$(realpath "$0")")/../data"

echo "Syncing activations and vectors from $VM_IP..."

# Sync activations (large files)
rsync -avz --progress \
    -e "ssh $SSH_OPTS" \
    "ubuntu@$VM_IP:$REMOTE_DIR/activations/" \
    "$LOCAL_DIR/activations/"

# Sync vectors (small files)
rsync -avz --progress \
    -e "ssh $SSH_OPTS" \
    "ubuntu@$VM_IP:$REMOTE_DIR/vectors/" \
    "$LOCAL_DIR/vectors/"

echo "Sync complete."
echo "Activation files:"
for d in "$LOCAL_DIR/activations"/*/; do
    echo "  $(basename $d): $(ls $d/*.npy 2>/dev/null | wc -l) files"
done
echo "Vector files:"
for d in "$LOCAL_DIR/vectors"/*/; do
    echo "  $(basename $d): $(ls $d/* 2>/dev/null | wc -l) files"
done
