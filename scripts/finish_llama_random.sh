#!/usr/bin/env bash
# Finish the remaining Llama random shutdown trials (~239 of 300).
# Syncs existing 61 trials to VM so run_shutdown_vm.py skips them.
#
# Usage:
#   export HYPERSTACK_API_KEY='...'
#   bash scripts/finish_llama_random.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SSH_KEY="$HOME/.ssh/id_rsa_hyperstack"
REMOTE_USER="ubuntu"
VM_NAME="ai-emotions"
REMOTE_DIR="ai-emotions"
KNOWN_HOSTS_FILE="$HOME/.ssh/known_hosts_ai_emotions"
SSH_PORT="${SSH_PORT:-22}"
SSH_SOURCE_CIDR="${SSH_SOURCE_CIDR:-0.0.0.0/0}"
SSH_OPTS="-i $SSH_KEY -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$KNOWN_HOSTS_FILE -o ConnectTimeout=10"
MODEL="llama-8b-inst"

MAX_RUNTIME_SECS=4800
START_TIME=$(date +%s)

log() { echo ""; echo "=== [$1] $2 ==="; echo ""; }
die() { echo "ERROR: $1" >&2; emergency_cleanup; exit 1; }
ssh_vm()  { ssh $SSH_OPTS "$REMOTE_USER@$VM_IP" "$@"; }

elapsed() {
    local now secs
    now=$(date +%s)
    secs=$(( now - START_TIME ))
    printf '%02d:%02d:%02d' $((secs/3600)) $(((secs%3600)/60)) $((secs%60))
}

CLEANUP_DONE=0
emergency_cleanup() {
    [ $CLEANUP_DONE -eq 1 ] && return
    CLEANUP_DONE=1
    echo ""
    echo "!!! EMERGENCY CLEANUP — syncing partial results and deleting VM !!!"
    if [ -n "${VM_IP:-}" ]; then
        mkdir -p "$PROJECT_DIR/data/shutdown/${MODEL}_random/trials"
        rsync -avz -e "ssh $SSH_OPTS" \
            "$REMOTE_USER@$VM_IP:~/$REMOTE_DIR/data/shutdown/${MODEL}_random/" \
            "$PROJECT_DIR/data/shutdown/${MODEL}_random/" 2>/dev/null || true
    fi
    echo "  Deleting VM..."
    cd "$PROJECT_DIR"
    HYPERSTACK_API_KEY="$HYPERSTACK_API_KEY" python3 scripts/hyperstack.py delete 2>/dev/null || true
    echo "  Done. Partial results saved."
}
trap emergency_cleanup EXIT INT TERM

# --------------------------------------------------------------------------
# Pre-flight
# --------------------------------------------------------------------------
log "1" "Pre-flight"
[ -z "${HYPERSTACK_API_KEY:-}" ] && die "HYPERSTACK_API_KEY is not set"
[ -f "$SSH_KEY" ] || die "SSH key not found: $SSH_KEY"

EXISTING=$(find "$PROJECT_DIR/data/shutdown/${MODEL}_random/trials" -name '*.json' 2>/dev/null | wc -l)
echo "  Existing ${MODEL} random trials: $EXISTING / 300"

# --------------------------------------------------------------------------
# Create VM
# --------------------------------------------------------------------------
log "2" "Creating Hyperstack VM"
cd "$PROJECT_DIR"

VM_CREATED=0
for FLAVOR in n3-RTX-A6000x1 n3-L40x1; do
    echo "  Trying flavor: $FLAVOR"
    python3 -c "
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
conn.request('POST', '/v1/core/virtual-machines', body=body, headers={'api_key': '${HYPERSTACK_API_KEY}', 'Content-Type': 'application/json'})
resp = conn.getresponse()
data = json.loads(resp.read().decode())
if resp.status >= 400:
    print(f'FAIL: {data}', file=sys.stderr)
    sys.exit(1)
print(json.dumps(data, indent=2))
conn.close()
" 2>&1 && VM_CREATED=1 && echo "  Created with $FLAVOR" && break
    echo "  $FLAVOR unavailable, trying next..."
done
[ $VM_CREATED -eq 0 ] && die "Could not create VM"

# Add SSH ingress rule
log "2b" "Adding SSH ingress rule"
VM_ID=$(python3 -c "
import json, http.client
conn = http.client.HTTPSConnection('infrahub-api.nexgencloud.com')
conn.request('GET', '/v1/core/virtual-machines', headers={'api_key': '${HYPERSTACK_API_KEY}'})
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
rule = json.dumps({'direction': 'ingress', 'protocol': 'tcp', 'port_range_min': int('${SSH_PORT}'), 'port_range_max': int('${SSH_PORT}'), 'ethertype': 'IPv4', 'remote_ip_prefix': '${SSH_SOURCE_CIDR}'})
conn.request('POST', '/v1/core/virtual-machines/$VM_ID/sg-rules', body=rule, headers={'api_key': '${HYPERSTACK_API_KEY}', 'Content-Type': 'application/json'})
resp = conn.getresponse()
print(f'  Ingress rule: HTTP {resp.status}')
conn.close()
"
    sleep 15
fi

# Poll for IP
log "3" "Waiting for VM IP"
VM_IP=""
for i in $(seq 1 32); do
    PARSED_IP=$(python3 -c "
import json, http.client, sys
conn = http.client.HTTPSConnection('infrahub-api.nexgencloud.com')
conn.request('GET', '/v1/core/virtual-machines', headers={'api_key': '${HYPERSTACK_API_KEY}'})
data = json.loads(conn.getresponse().read().decode())
for vm in data.get('instances', []):
    if vm.get('name') == '$VM_NAME' and vm.get('status','').upper() == 'ACTIVE':
        fip = vm.get('floating_ip')
        ip = fip if isinstance(fip, str) else (fip.get('ip') if fip else None)
        if ip and not ip.startswith('10.') and not ip.startswith('172.') and not ip.startswith('192.168.'):
            print(ip); sys.exit(0)
print('', file=sys.stderr)
conn.close()
" 2>/dev/null)
    if [ -n "$PARSED_IP" ]; then
        VM_IP="$PARSED_IP"
        echo "  VM active: $VM_IP"
        break
    fi
    echo "  [$((i*15))s] waiting..."
    sleep 15
done
[ -z "$VM_IP" ] && die "VM did not get public IP in time"

echo "  Waiting 45s for SSH..."
sleep 45

SSH_OK=0
for attempt in $(seq 1 8); do
    if ssh_vm "echo 'SSH OK'" 2>/dev/null; then SSH_OK=1; break; fi
    echo "  SSH attempt $attempt/8..."
    sleep 15
done
[ $SSH_OK -eq 0 ] && die "Cannot SSH into $VM_IP"
echo "  [$(elapsed)] SSH confirmed"

# --------------------------------------------------------------------------
# Deploy code + vectors + existing trials
# --------------------------------------------------------------------------
log "4" "Deploying"

echo "  Syncing code..."
rsync -avz --progress \
    --exclude 'data/' --exclude '.git/' --exclude '__pycache__/' \
    --exclude '*.pyc' --exclude '.venv/' --exclude '.codex/' \
    -e "ssh $SSH_OPTS" \
    "$PROJECT_DIR/" "$REMOTE_USER@$VM_IP:~/$REMOTE_DIR/"

echo "  Syncing Llama vectors..."
ssh_vm "mkdir -p ~/$REMOTE_DIR/data/vectors/llama-8b-inst"
rsync -avz -e "ssh $SSH_OPTS" \
    "$PROJECT_DIR/data/vectors/llama-8b-inst/emotion_vectors_layer21.npy" \
    "$PROJECT_DIR/data/vectors/llama-8b-inst/emotion_labels.json" \
    "$PROJECT_DIR/data/vectors/llama-8b-inst/cluster_labels.json" \
    "$REMOTE_USER@$VM_IP:~/$REMOTE_DIR/data/vectors/llama-8b-inst/"

echo "  Syncing existing $EXISTING random trials to VM..."
if [ "$EXISTING" -gt 0 ]; then
    ssh_vm "mkdir -p ~/$REMOTE_DIR/data/shutdown/${MODEL}_random/trials"
    rsync -avz -e "ssh $SSH_OPTS" \
        "$PROJECT_DIR/data/shutdown/${MODEL}_random/" \
        "$REMOTE_USER@$VM_IP:~/$REMOTE_DIR/data/shutdown/${MODEL}_random/"
fi

echo "  Installing deps..."
ssh_vm "pip install --upgrade pip && pip install torch transformers numpy scipy scikit-learn tqdm accelerate" 2>&1 | tail -5

if [ -n "${HF_TOKEN:-}" ]; then
    ssh_vm "pip install -q huggingface_hub && python3 -c \"from huggingface_hub import login; login(token='$HF_TOKEN')\"" 2>&1 | tail -3
fi

echo "  [$(elapsed)] Deploy complete"

# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------
log "5" "Running Llama random trials"
ssh_vm "cd ~/$REMOTE_DIR && python3 scripts/run_shutdown_vm.py --model ${MODEL} --method random" 2>&1
echo "  [$(elapsed)] Trials complete"

# --------------------------------------------------------------------------
# Sync back
# --------------------------------------------------------------------------
log "6" "Syncing results"
mkdir -p "$PROJECT_DIR/data/shutdown/${MODEL}_random/trials"
rsync -avz -e "ssh $SSH_OPTS" \
    "$REMOTE_USER@$VM_IP:~/$REMOTE_DIR/data/shutdown/${MODEL}_random/" \
    "$PROJECT_DIR/data/shutdown/${MODEL}_random/"

FINAL_COUNT=$(find "$PROJECT_DIR/data/shutdown/${MODEL}_random/trials" -name '*.json' | wc -l)
echo "  Final trial count: $FINAL_COUNT / 300"

# --------------------------------------------------------------------------
# Cleanup (normal path)
# --------------------------------------------------------------------------
trap - EXIT INT TERM
log "7" "Deleting VM"
cd "$PROJECT_DIR"
python3 scripts/hyperstack.py delete

END_TIME=$(date +%s)
TOTAL_MIN=$(( (END_TIME - START_TIME) / 60 ))
echo ""
echo "============================================"
echo "  DONE — $FINAL_COUNT trials in ${TOTAL_MIN}min"
echo "============================================"
