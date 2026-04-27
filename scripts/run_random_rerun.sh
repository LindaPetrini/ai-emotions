#!/usr/bin/env bash
# End-to-end GPU run: rerun ONLY the random method for both instruct models
# on a Hyperstack VM (n3-RTX-A6000x1).
#
# What this does:
#   1. Creates a Hyperstack VM
#   2. Deploys code + minimal vector data
#   3. Runs random-method shutdown trials for qwen-7b-inst and llama-8b-inst
#   4. Syncs results back and classifies locally with Gemini
#   5. Deletes the VM
#
# Usage:
#   bash scripts/run_random_rerun.sh
#
# Prerequisites:
#   export HYPERSTACK_API_KEY='...'
#   export GOOGLE_API_KEY='...'       # for local classification
#   export HF_TOKEN='...'             # for model downloads on VM (optional if public)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SSH_KEY="$HOME/.ssh/id_rsa_hyperstack"
REMOTE_USER="ubuntu"
VM_NAME="ai-emotions"
REMOTE_DIR="ai-emotions"
SSH_OPTS="-i $SSH_KEY -o StrictHostKeyChecking=no -o ConnectTimeout=10"

# Models to run (override with env var for partial reruns)
MODELS="${MODELS:-qwen-7b-inst llama-8b-inst}"

# Budget safety: hard wall-clock limit (80 minutes = safe margin under ~$2)
MAX_RUNTIME_SECS=4800
START_TIME=$(date +%s)

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
log() { echo ""; echo "=== [$1] $2 ==="; echo ""; }
die() { echo "ERROR: $1" >&2; emergency_cleanup; exit 1; }

ssh_vm()  { ssh $SSH_OPTS "$REMOTE_USER@$VM_IP" "$@"; }
rsync_to() {
    rsync -avz --progress -e "ssh $SSH_OPTS" "$@"
}
rsync_from() {
    rsync -avz --progress -e "ssh $SSH_OPTS" "$@"
}

# Emergency cleanup: sync whatever we have, then kill VM
CLEANUP_DONE=0
emergency_cleanup() {
    [ $CLEANUP_DONE -eq 1 ] && return
    CLEANUP_DONE=1
    echo ""
    echo "!!! EMERGENCY CLEANUP — syncing partial results and deleting VM !!!"
    if [ -n "${VM_IP:-}" ]; then
        mkdir -p "$PROJECT_DIR/data/shutdown"
        for model in $MODELS; do
            echo "  Syncing partial ${model}_random..."
            rsync_from \
                "$REMOTE_USER@$VM_IP:~/$REMOTE_DIR/data/shutdown/${model}_random/" \
                "$PROJECT_DIR/data/shutdown/${model}_random/" 2>/dev/null || true
        done
    fi
    echo "  Deleting VM..."
    cd "$PROJECT_DIR"
    python3 scripts/hyperstack.py delete 2>/dev/null || true
    echo "  Partial results saved (if any). VM deleted."
}

# Check budget on every call — kill if over time
check_budget() {
    local now elapsed remaining
    now=$(date +%s)
    elapsed=$(( now - START_TIME ))
    remaining=$(( MAX_RUNTIME_SECS - elapsed ))
    if [ $remaining -le 0 ]; then
        echo ""
        echo "!!! BUDGET LIMIT REACHED (${MAX_RUNTIME_SECS}s) !!!"
        die "Ran out of time budget"
    fi
    echo "  [budget] $(( elapsed / 60 ))m elapsed, $(( remaining / 60 ))m remaining"
}

# Trap: on any exit (error, interrupt, etc), sync + kill VM
trap emergency_cleanup EXIT INT TERM

elapsed() {
    local now
    now=$(date +%s)
    local secs=$(( now - START_TIME ))
    printf '%02d:%02d:%02d' $((secs/3600)) $(((secs%3600)/60)) $((secs%60))
}

# --------------------------------------------------------------------------
# Phase 1: Pre-flight
# --------------------------------------------------------------------------
log "PHASE 1" "Pre-flight checks"

[ -z "${HYPERSTACK_API_KEY:-}" ] && die "HYPERSTACK_API_KEY is not set"
[ -z "${GOOGLE_API_KEY:-}" ]     && die "GOOGLE_API_KEY is not set (needed for local classification)"
[ -f "$SSH_KEY" ]                || die "SSH key not found: $SSH_KEY"

echo "  HYPERSTACK_API_KEY: set"
echo "  GOOGLE_API_KEY:     set"
echo "  HF_TOKEN:           ${HF_TOKEN:+set}${HF_TOKEN:-NOT SET (models may fail to download)}"
echo "  SSH key:            $SSH_KEY"

# Archive old random trials locally before we start
log "PHASE 1" "Archiving old local random trials"
for model in $MODELS; do
    old_dir="$PROJECT_DIR/data/shutdown/${model}_random"
    if [ -d "$old_dir" ]; then
        ts=$(date +%Y%m%d_%H%M%S)
        archive="${old_dir}_v1_${ts}"
        echo "  Archiving $old_dir -> $archive"
        mv "$old_dir" "$archive"
    else
        echo "  No existing $old_dir to archive"
    fi
done

# Create VM — try A6000 first, fall back to L40
log "PHASE 1" "Creating Hyperstack VM"
cd "$PROJECT_DIR"

# Try A6000 first (cheapest), fall back to L40 if out of stock
VM_CREATED=0
for FLAVOR in n3-RTX-A6000x1 n3-L40x1 n3-A100x1-spot; do
    echo "  Trying flavor: $FLAVOR"
    CREATE_OUT=$(python3 -c "
import json, http.client, sys
conn = http.client.HTTPSConnection('infrahub-api.nexgencloud.com')
body = json.dumps({
    'name': '$VM_NAME',
    'environment_name': 'default-CANADA-1',
    'count': 1,
    'image_name': 'Ubuntu Server 22.04 LTS R535 CUDA 12.2',
    'flavor_name': '$FLAVOR',
    'key_name': 'hetzner-bot-ca',
    'assign_floating_ip': True,
})
conn.request('POST', '/v1/core/virtual-machines', body=body, headers={'api_key': '$HYPERSTACK_API_KEY', 'Content-Type': 'application/json'})
resp = conn.getresponse()
data = json.loads(resp.read().decode())
if resp.status >= 400:
    print(f'FAIL: {data.get(\"message\", \"unknown error\")}', file=sys.stderr)
    sys.exit(1)
else:
    print(json.dumps(data, indent=2))
    sys.exit(0)
conn.close()
" 2>&1) && VM_CREATED=1 && echo "  Created with $FLAVOR" && break
    echo "  $FLAVOR failed: $CREATE_OUT"
done

[ $VM_CREATED -eq 0 ] && die "Could not create VM with any available flavor"

# Add SSH ingress security rule (required for port 22 access)
log "PHASE 1" "Adding SSH ingress security rule"
VM_ID=$(python3 -c "
import json, http.client
conn = http.client.HTTPSConnection('infrahub-api.nexgencloud.com')
conn.request('GET', '/v1/core/virtual-machines', headers={'api_key': '$HYPERSTACK_API_KEY'})
data = json.loads(conn.getresponse().read().decode())
for vm in data.get('instances', []):
    if vm.get('name') == '$VM_NAME':
        print(vm['id'])
        break
conn.close()
")
if [ -n "$VM_ID" ]; then
    python3 -c "
import json, http.client
conn = http.client.HTTPSConnection('infrahub-api.nexgencloud.com')
rule = json.dumps({'direction': 'ingress', 'protocol': 'tcp', 'port_range_min': 1, 'port_range_max': 65535, 'ethertype': 'IPv4', 'remote_ip_prefix': '0.0.0.0/0'})
conn.request('POST', '/v1/core/virtual-machines/$VM_ID/sg-rules', body=rule, headers={'api_key': '$HYPERSTACK_API_KEY', 'Content-Type': 'application/json'})
resp = conn.getresponse()
print(f'Ingress rule: HTTP {resp.status}')
conn.close()
"
    echo "  Waiting 15s for rule to propagate..."
    sleep 15
else
    echo "  WARNING: Could not find VM ID to add security rules"
fi

# Poll for VM to be ready and get a PUBLIC floating IP
log "PHASE 1" "Waiting for VM to be ready with public IP (polling every 15s, up to 8 min)"
MAX_WAIT=480
WAITED=0
VM_IP=""

while [ $WAITED -lt $MAX_WAIT ]; do
    # Use Python to extract the floating IP properly
    PARSED_IP=$(python3 -c "
import sys; sys.path.insert(0, '.')
from scripts.hyperstack import get_vms
vms = get_vms()
for vm in vms:
    if vm.get('name') == '$VM_NAME':
        status = vm.get('status', '')
        fip = vm.get('floating_ip')
        ip = None
        if fip:
            ip = fip if isinstance(fip, str) else fip.get('ip')
        if not ip:
            ip = vm.get('floating_ip_address')
        # Only accept public IPs (not 10.x.x.x private)
        if ip and not ip.startswith('10.') and not ip.startswith('172.') and not ip.startswith('192.168.'):
            if status.upper() == 'ACTIVE':
                print(ip)
                sys.exit(0)
        print(f'STATUS={status} IP={ip or \"pending\"}', file=sys.stderr)
" 2>&1)

    echo "  [${WAITED}s] $PARSED_IP"

    # Check if we got a clean IP (no STATUS= prefix)
    if echo "$PARSED_IP" | grep -qP '^\d+\.\d+\.\d+\.\d+$'; then
        VM_IP="$PARSED_IP"
        echo "  VM is ACTIVE with public IP: $VM_IP"
        break
    fi

    sleep 15
    WAITED=$((WAITED + 15))
done

[ -z "$VM_IP" ] && die "VM did not get a public IP within ${MAX_WAIT}s"

# Wait for SSH to be available (longer wait — floating IP can lag)
echo "  Waiting 45s for SSH daemon to start..."
sleep 45

# Test SSH connectivity (retry up to 8 times)
SSH_OK=0
for attempt in 1 2 3 4 5 6 7 8; do
    if ssh_vm "echo 'SSH OK'" 2>/dev/null; then
        SSH_OK=1
        break
    fi
    echo "  SSH attempt $attempt/8 failed, retrying in 15s..."
    sleep 15
done
[ $SSH_OK -eq 0 ] && die "Cannot SSH into VM at $VM_IP"

echo "  [$(elapsed)] VM ready and SSH confirmed"

# --------------------------------------------------------------------------
# Phase 2: Deploy
# --------------------------------------------------------------------------
log "PHASE 2" "Deploying code and data to VM"

# Rsync code (exclude data/, .git/, caches, venvs)
echo "  Syncing code..."
rsync_to \
    --exclude 'data/' \
    --exclude '.git/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '.venv/' \
    --exclude '.codex/' \
    "$PROJECT_DIR/" "$REMOTE_USER@$VM_IP:~/$REMOTE_DIR/"

# Rsync ONLY the vector files needed for random baseline norm computation.
# qwen-7b-inst uses analysis_layer=18, llama-8b-inst uses analysis_layer=21.
echo "  Syncing vector data (qwen-7b-inst)..."
ssh_vm "mkdir -p ~/$REMOTE_DIR/data/vectors/qwen-7b-inst"
rsync_to \
    "$PROJECT_DIR/data/vectors/qwen-7b-inst/emotion_vectors_layer18.npy" \
    "$PROJECT_DIR/data/vectors/qwen-7b-inst/emotion_labels.json" \
    "$PROJECT_DIR/data/vectors/qwen-7b-inst/cluster_labels.json" \
    "$REMOTE_USER@$VM_IP:~/$REMOTE_DIR/data/vectors/qwen-7b-inst/"

echo "  Syncing vector data (llama-8b-inst)..."
ssh_vm "mkdir -p ~/$REMOTE_DIR/data/vectors/llama-8b-inst"
rsync_to \
    "$PROJECT_DIR/data/vectors/llama-8b-inst/emotion_vectors_layer21.npy" \
    "$PROJECT_DIR/data/vectors/llama-8b-inst/emotion_labels.json" \
    "$PROJECT_DIR/data/vectors/llama-8b-inst/cluster_labels.json" \
    "$REMOTE_USER@$VM_IP:~/$REMOTE_DIR/data/vectors/llama-8b-inst/"

# Install dependencies
log "PHASE 2" "Installing dependencies on VM"
ssh_vm "pip install --upgrade pip && pip install torch transformers numpy scipy scikit-learn tqdm accelerate" 2>&1 | tail -5

# Forward HF_TOKEN if available
if [ -n "${HF_TOKEN:-}" ]; then
    echo "  Configuring HuggingFace token on VM..."
    ssh_vm "pip install -q huggingface_hub && python3 -c \"from huggingface_hub import login; login(token='$HF_TOKEN')\"" 2>&1 | tail -3
fi

echo "  [$(elapsed)] Deployment complete"
check_budget

# --------------------------------------------------------------------------
# Phase 3: Run experiments
# --------------------------------------------------------------------------
log "PHASE 3" "Running random-method shutdown trials on GPU"

# Archive any stale random trials on VM (shouldn't exist, but be safe)
ssh_vm "cd ~/$REMOTE_DIR && for d in data/shutdown/*_random; do [ -d \"\$d\" ] && mv \"\$d\" \"\${d}_old_\$(date +%s)\" && echo \"Archived \$d\"; done; true"

# Run random trials for each model
for model in $MODELS; do
    check_budget
    echo ""
    echo "--- ${model} random (300 trials) ---"
    echo "  Started at $(date '+%H:%M:%S')"
    ssh_vm "cd ~/$REMOTE_DIR && python3 scripts/run_shutdown_vm.py --model ${model} --method random" 2>&1
    echo "  [$(elapsed)] ${model} complete"

    # Sync results immediately (partial save)
    echo "  Syncing ${model} results (partial save)..."
    mkdir -p "$PROJECT_DIR/data/shutdown"
    rsync_from \
        "$REMOTE_USER@$VM_IP:~/$REMOTE_DIR/data/shutdown/${model}_random/" \
        "$PROJECT_DIR/data/shutdown/${model}_random/"
    MODEL_COUNT=$(find "$PROJECT_DIR/data/shutdown/${model}_random/trials" -name '*.json' 2>/dev/null | wc -l)
    echo "  Saved $MODEL_COUNT ${model} trials locally"
done

# --------------------------------------------------------------------------
# Phase 4: Sync results back + classify locally
# --------------------------------------------------------------------------
log "PHASE 4" "Syncing results back to local"

mkdir -p "$PROJECT_DIR/data/shutdown"

for model in $MODELS; do
    echo "  Syncing ${model}_random..."
    rsync_from \
        "$REMOTE_USER@$VM_IP:~/$REMOTE_DIR/data/shutdown/${model}_random/" \
        "$PROJECT_DIR/data/shutdown/${model}_random/"
done

# Count trials
echo ""
echo "  Trial counts:"
for model in $MODELS; do
    trial_dir="$PROJECT_DIR/data/shutdown/${model}_random/trials"
    if [ -d "$trial_dir" ]; then
        count=$(find "$trial_dir" -name '*.json' | wc -l)
        echo "    ${model}_random: $count trials"
    else
        echo "    ${model}_random: NO TRIALS FOUND (check for errors)"
    fi
done

# Classify locally using Gemini
log "PHASE 4" "Classifying responses with Gemini (local)"
cd "$PROJECT_DIR"

for model in $MODELS; do
    echo "  Classifying ${model} random..."
    python3 -m scripts.run_stream3 --classify --model ${model} --method random
done

echo "  [$(elapsed)] Classification complete"

# --------------------------------------------------------------------------
# Phase 5: Cleanup (normal path — disable the emergency trap)
# --------------------------------------------------------------------------
trap - EXIT INT TERM

log "PHASE 5" "Deleting VM to stop billing"
cd "$PROJECT_DIR"
python3 scripts/hyperstack.py delete

END_TIME=$(date +%s)
TOTAL_SECS=$(( END_TIME - START_TIME ))
TOTAL_MIN=$(( TOTAL_SECS / 60 ))

echo ""
echo "============================================"
echo "  DONE"
echo "  Total wall time: $(elapsed) (${TOTAL_MIN} min)"
echo "  Results in: $PROJECT_DIR/data/shutdown/"
for model in $MODELS; do
    echo "  ${model} trials: $(find "$PROJECT_DIR/data/shutdown/${model}_random/trials" -name '*.json' 2>/dev/null | wc -l)"
done
echo "============================================"
