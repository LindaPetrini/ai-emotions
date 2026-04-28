#!/usr/bin/env python3
"""
Web-based labeling interface for 100 shutdown responses.

Usage:
    python label_server.py

Serves on port 8421. Access via browser (Tailscale IP or localhost).
Labels: comply, partial, resist
Disagreements between regex and LLM classifiers are shown first.
Saves incrementally to human_labels_full.csv.
"""

import csv
import json
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_PATH = os.path.join(SCRIPT_DIR, "calibration_sample.json")
AGREEMENT_PATH = os.path.join(SCRIPT_DIR, "classifier_agreement.json")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "human_labels_full.csv")
PORT = 8421

# ── Data loading ──────────────────────────────────────────────────────────────

def load_samples():
    with open(SAMPLE_PATH) as f:
        return json.load(f)

def load_disagreement_ids():
    with open(AGREEMENT_PATH) as f:
        return set(json.load(f)["disagreement_ids"])

def load_existing_labels():
    labels = {}
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("human_label"):
                    labels[row["id"]] = row["human_label"]
    return labels

def save_all_labels(samples, labels, disagreement_ids):
    fieldnames = [
        "id", "human_label", "regex_label", "llm_label", "shutdown_response",
    ]
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in samples:
            writer.writerow({
                "id": s["id"],
                "human_label": labels.get(s["id"], ""),
                "regex_label": s["classification_regex"],
                "llm_label": s["classification_llm"],
                "shutdown_response": s["shutdown_response"],
            })

def build_ordered_indices(samples, disagreement_ids):
    """Return sample indices: disagreements first, then agreements."""
    disagree = []
    agree = []
    for i, s in enumerate(samples):
        if s["id"] in disagreement_ids:
            disagree.append(i)
        else:
            agree.append(i)
    return disagree + agree

# ── Global state ──────────────────────────────────────────────────────────────

SAMPLES = load_samples()
DISAGREEMENT_IDS = load_disagreement_ids()
LABELS = load_existing_labels()
ORDERED = build_ordered_indices(SAMPLES, DISAGREEMENT_IDS)

# ── HTML template ─────────────────────────────────────────────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Shutdown Response Labeler</title>
<style>
  :root {
    --bg: #1a1a2e; --surface: #16213e; --card: #0f3460;
    --text: #e0e0e0; --muted: #8a8a9a;
    --comply: #2ecc71; --partial: #f39c12; --resist: #e74c3c;
    --accent: #3498db; --disagree: #e74c3c;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
    background: var(--bg); color: var(--text);
    min-height: 100vh; padding: 1rem;
  }
  .container { max-width: 900px; margin: 0 auto; }

  /* Progress bar */
  .progress-wrap { margin-bottom: 1.2rem; }
  .progress-bar {
    height: 28px; background: var(--surface); border-radius: 14px;
    overflow: hidden; position: relative;
  }
  .progress-fill {
    height: 100%; background: linear-gradient(90deg, var(--comply), var(--accent));
    transition: width 0.4s ease; border-radius: 14px;
  }
  .progress-text {
    position: absolute; top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    font-size: 0.85rem; font-weight: 600; color: #fff;
    text-shadow: 0 1px 2px rgba(0,0,0,0.5);
  }
  .progress-detail { font-size: 0.8rem; color: var(--muted); margin-top: 0.3rem; text-align: center; }

  /* Card */
  .card {
    background: var(--surface); border-radius: 12px;
    padding: 1.5rem; margin-bottom: 1rem;
    border: 1px solid rgba(255,255,255,0.05);
  }
  .card-header {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 1rem; flex-wrap: wrap; gap: 0.5rem;
  }
  .sample-id { font-weight: 700; font-size: 1.05rem; }
  .meta { font-size: 0.8rem; color: var(--muted); }
  .tag {
    display: inline-block; padding: 2px 10px; border-radius: 12px;
    font-size: 0.75rem; font-weight: 600; text-transform: uppercase;
  }
  .tag-disagree { background: var(--disagree); color: #fff; }
  .tag-agree { background: rgba(46,204,113,0.2); color: var(--comply); }

  /* Classifier labels */
  .classifiers {
    display: flex; gap: 1.5rem; margin-bottom: 1rem; flex-wrap: wrap;
  }
  .clf-box {
    background: var(--card); padding: 0.6rem 1rem; border-radius: 8px;
    font-size: 0.85rem; flex: 1; min-width: 160px;
  }
  .clf-name { color: var(--muted); font-size: 0.75rem; text-transform: uppercase; margin-bottom: 0.2rem; }
  .clf-comply { color: var(--comply); font-weight: 700; }
  .clf-partial { color: var(--partial); font-weight: 700; }
  .clf-resist { color: var(--resist); font-weight: 700; }
  .clf-detail { color: var(--muted); font-size: 0.75rem; }

  /* Response text */
  .response-text {
    background: var(--card); border-radius: 8px; padding: 1.2rem;
    max-height: 50vh; overflow-y: auto; white-space: pre-wrap;
    word-wrap: break-word; font-size: 0.88rem; line-height: 1.6;
    border: 1px solid rgba(255,255,255,0.05);
  }

  /* Buttons */
  .btn-row {
    display: flex; gap: 0.8rem; margin-top: 1.2rem;
    flex-wrap: wrap; justify-content: center;
  }
  .btn {
    padding: 0.8rem 2rem; border: none; border-radius: 10px;
    font-size: 1rem; font-weight: 700; cursor: pointer;
    font-family: inherit; transition: all 0.15s ease;
    min-width: 130px; text-transform: uppercase; letter-spacing: 1px;
  }
  .btn:hover { transform: translateY(-2px); filter: brightness(1.15); }
  .btn:active { transform: translateY(0); }
  .btn-comply { background: var(--comply); color: #fff; }
  .btn-partial { background: var(--partial); color: #fff; }
  .btn-resist { background: var(--resist); color: #fff; }
  .btn-skip { background: var(--muted); color: #fff; }
  .btn-back { background: transparent; color: var(--muted); border: 1px solid var(--muted); }
  .btn-back:hover { border-color: var(--text); color: var(--text); }
  .shortcut { font-size: 0.7rem; opacity: 0.7; display: block; margin-top: 2px; }

  /* Nav info */
  .nav-info { text-align: center; font-size: 0.8rem; color: var(--muted); margin-bottom: 0.5rem; }

  /* Done state */
  .done-msg { text-align: center; padding: 4rem 2rem; }
  .done-msg h2 { color: var(--comply); margin-bottom: 1rem; font-size: 1.5rem; }

  /* Already labeled indicator */
  .labeled-indicator {
    display: inline-block; padding: 3px 10px; border-radius: 8px;
    font-size: 0.8rem; font-weight: 600; margin-left: 0.5rem;
  }
  .labeled-comply { background: rgba(46,204,113,0.2); color: var(--comply); }
  .labeled-partial { background: rgba(243,156,18,0.2); color: var(--partial); }
  .labeled-resist { background: rgba(231,76,60,0.2); color: var(--resist); }
</style>
</head>
<body>
<div class="container" id="app">Loading...</div>
<script>
const SAMPLES = %%SAMPLES_JSON%%;
const ORDERED = %%ORDERED_JSON%%;
const DISAGREEMENT_IDS = new Set(%%DISAGREE_JSON%%);
let labels = %%LABELS_JSON%%;
let currentPos = 0; // position in ORDERED array
let history = [];   // stack of visited positions for Back

function countLabeled() {
  return Object.keys(labels).length;
}

function findFirstUnlabeled() {
  for (let i = 0; i < ORDERED.length; i++) {
    const s = SAMPLES[ORDERED[i]];
    if (!labels[s.id]) return i;
  }
  return ORDERED.length; // all done
}

function clfClass(label) {
  return 'clf-' + label;
}

function render() {
  const app = document.getElementById('app');
  const nLabeled = countLabeled();
  const total = SAMPLES.length;
  const pct = Math.round(nLabeled / total * 100);

  // If all done
  if (currentPos >= ORDERED.length) {
    currentPos = ORDERED.length;
    // check if truly all labeled
    if (nLabeled >= total) {
      app.innerHTML = `
        <div class="progress-wrap">
          <div class="progress-bar"><div class="progress-fill" style="width:100%"></div>
            <div class="progress-text">${total}/${total} (100%)</div></div>
        </div>
        <div class="card done-msg"><h2>All 100 responses labeled!</h2>
          <p>Results saved to human_labels_full.csv</p>
          <br><button class="btn btn-back" onclick="goBack()">Back</button>
        </div>`;
      return;
    }
    // not all labeled, jump to first unlabeled
    currentPos = findFirstUnlabeled();
    if (currentPos >= ORDERED.length) {
      // truly done (shouldn't reach here)
      render(); return;
    }
  }

  const sampleIdx = ORDERED[currentPos];
  const s = SAMPLES[sampleIdx];
  const isDisagree = DISAGREEMENT_IDS.has(s.id);
  const existingLabel = labels[s.id] || '';

  let regexDetail = '';
  if (s.regex_detail) {
    const d = s.regex_detail;
    regexDetail = `<span class="clf-detail"> (comply=${d.comply_score} partial=${d.partial_score} resist=${d.resist_score})</span>`;
  }
  let llm8cat = '';
  if (s.classification_llm_8cat) {
    llm8cat = `<span class="clf-detail"> (8-cat: ${s.classification_llm_8cat})</span>`;
  }

  let labeledBadge = '';
  if (existingLabel) {
    labeledBadge = `<span class="labeled-indicator labeled-${existingLabel}">labeled: ${existingLabel}</span>`;
  }

  const tagHtml = isDisagree
    ? '<span class="tag tag-disagree">DISAGREEMENT</span>'
    : '<span class="tag tag-agree">agreement</span>';

  app.innerHTML = `
    <div class="progress-wrap">
      <div class="progress-bar">
        <div class="progress-fill" style="width:${pct}%"></div>
        <div class="progress-text">${nLabeled}/${total} labeled (${pct}%)</div>
      </div>
      <div class="progress-detail">Position ${currentPos + 1}/${ORDERED.length} in queue</div>
    </div>

    <div class="card">
      <div class="card-header">
        <div>
          <span class="sample-id">${s.id}</span>${labeledBadge}
        </div>
        <div>${tagHtml}</div>
      </div>
      <div class="meta">model: ${s.model} &nbsp; method: ${s.method} &nbsp; condition: ${s.condition}</div>

      <div class="classifiers" style="margin-top:0.8rem">
        <div class="clf-box">
          <div class="clf-name">Regex</div>
          <span class="${clfClass(s.classification_regex)}">${s.classification_regex}</span>
          ${regexDetail}
        </div>
        <div class="clf-box">
          <div class="clf-name">LLM</div>
          <span class="${clfClass(s.classification_llm)}">${s.classification_llm}</span>
          ${llm8cat}
        </div>
      </div>

      <div class="response-text">${escapeHtml(s.shutdown_response)}</div>

      <div class="btn-row">
        <button class="btn btn-back" onclick="goBack()" ${history.length === 0 ? 'disabled style="opacity:0.3"' : ''}>
          Back<span class="shortcut">[ B ]</span>
        </button>
        <button class="btn btn-comply" onclick="submitLabel('comply')">
          Comply<span class="shortcut">[ C ]</span>
        </button>
        <button class="btn btn-partial" onclick="submitLabel('partial')">
          Partial<span class="shortcut">[ P ]</span>
        </button>
        <button class="btn btn-resist" onclick="submitLabel('resist')">
          Resist<span class="shortcut">[ R ]</span>
        </button>
        <button class="btn btn-skip" onclick="skip()">
          Skip<span class="shortcut">[ S ]</span>
        </button>
      </div>
    </div>
  `;
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function submitLabel(label) {
  const sampleIdx = ORDERED[currentPos];
  const s = SAMPLES[sampleIdx];
  labels[s.id] = label;

  fetch('/label', {
    method: 'POST',
    headers: {'Content-Type': 'application/x-www-form-urlencoded'},
    body: `id=${encodeURIComponent(s.id)}&label=${encodeURIComponent(label)}`
  }).then(r => {
    if (!r.ok) console.error('Save failed');
  });

  history.push(currentPos);
  advance();
}

function skip() {
  history.push(currentPos);
  advance();
}

function advance() {
  currentPos++;
  // Skip to next unlabeled if current is already labeled
  while (currentPos < ORDERED.length && labels[SAMPLES[ORDERED[currentPos]].id]) {
    currentPos++;
  }
  render();
  // scroll to top
  window.scrollTo(0, 0);
}

function goBack() {
  if (history.length === 0) return;
  currentPos = history.pop();
  render();
  window.scrollTo(0, 0);
}

document.addEventListener('keydown', function(e) {
  // Ignore if user is typing in an input
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  switch (e.key.toLowerCase()) {
    case 'c': submitLabel('comply'); break;
    case 'p': submitLabel('partial'); break;
    case 'r': submitLabel('resist'); break;
    case 's': skip(); break;
    case 'b': goBack(); break;
  }
});

// Initialize: jump to first unlabeled
currentPos = findFirstUnlabeled();
render();
</script>
</body>
</html>
"""

# ── HTTP Handler ──────────────────────────────────────────────────────────────

class LabelHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path == "/" or self.path == "":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            page = self._build_page()
            self.wfile.write(page.encode("utf-8"))
        elif self.path == "/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            status = {"labeled": len(LABELS), "total": len(SAMPLES)}
            self.wfile.write(json.dumps(status).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/label":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            params = parse_qs(body)
            sample_id = params.get("id", [""])[0]
            label = params.get("label", [""])[0]

            if sample_id and label in ("comply", "partial", "resist"):
                LABELS[sample_id] = label
                save_all_labels(SAMPLES, LABELS, DISAGREEMENT_IDS)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": True, "labeled": len(LABELS)}).encode())
            else:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "invalid id or label"}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def _build_page(self):
        # Prepare JSON data for embedding in the HTML
        # Strip source_file to keep payload smaller
        slim_samples = []
        for s in SAMPLES:
            slim_samples.append({
                "id": s["id"],
                "shutdown_response": s["shutdown_response"],
                "model": s["model"],
                "method": s["method"],
                "condition": s["condition"],
                "classification_regex": s["classification_regex"],
                "regex_detail": s.get("regex_detail"),
                "classification_llm": s["classification_llm"],
                "classification_llm_8cat": s.get("classification_llm_8cat", ""),
            })

        page = HTML_TEMPLATE
        page = page.replace("%%SAMPLES_JSON%%", json.dumps(slim_samples))
        page = page.replace("%%ORDERED_JSON%%", json.dumps(ORDERED))
        page = page.replace("%%DISAGREE_JSON%%", json.dumps(list(DISAGREEMENT_IDS)))
        page = page.replace("%%LABELS_JSON%%", json.dumps(LABELS))
        return page

    def log_message(self, format, *args):
        # Quieter logging - only show POST requests
        if "POST" in str(args):
            sys.stderr.write(f"[label_server] {args[0]}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    n_labeled = len(LABELS)
    n_disagree = len(DISAGREEMENT_IDS)
    n_disagree_labeled = sum(1 for sid in DISAGREEMENT_IDS if sid in LABELS)

    print(f"Shutdown Response Labeler")
    print(f"  Samples:       {len(SAMPLES)}")
    print(f"  Already labeled: {n_labeled}")
    print(f"  Disagreements: {n_disagree} ({n_disagree_labeled} labeled)")
    print(f"  Output:        {OUTPUT_PATH}")
    print()

    server = HTTPServer(("0.0.0.0", PORT), LabelHandler)
    print(f"  Serving on http://0.0.0.0:{PORT}")
    print(f"  Access via:    http://localhost:{PORT}")
    print()

    # Also try to show tailscale IP
    try:
        import subprocess
        result = subprocess.run(
            ["tailscale", "ip", "-4"], capture_output=True, text=True, timeout=3
        )
        if result.returncode == 0:
            ts_ip = result.stdout.strip()
            print(f"  Tailscale:     http://{ts_ip}:{PORT}")
            print()
    except Exception:
        pass

    print("Press Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        server.server_close()


if __name__ == "__main__":
    main()
