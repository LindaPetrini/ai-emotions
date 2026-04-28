#!/usr/bin/env python3
"""
Web-based human evaluation interface for Block 6 (steering evaluation).

Presents 240 completions in randomized (but deterministic) order.
Raters see only the text + original prompt + target emotion -- NOT the
condition (steered / random / baseline), model, or alpha.

Three rating dimensions (1-5):
  1. Target emotion expression
  2. Coherence
  3. Prompt relevance

Usage:
    python eval_server.py

Serves on port 8422. Access via browser (Tailscale IP or localhost).
Saves incrementally to data/steering_eval/ratings.csv.
Resumes on restart (same randomized order via seed).
"""

import argparse
import csv
import json
import os
import random
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_PATH = os.path.join(SCRIPT_DIR, "sample.json")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "ratings.csv")
PORT = 8422
SHUFFLE_SEED = 2024

# ── Data loading ──────────────────────────────────────────────────────────────

def load_samples():
    with open(SAMPLE_PATH) as f:
        return json.load(f)


def build_shuffled_order(n):
    """Deterministic shuffled order so it's identical on restart."""
    order = list(range(n))
    rng = random.Random(SHUFFLE_SEED)
    rng.shuffle(order)
    return order


def load_existing_ratings():
    """Load previously saved ratings from CSV."""
    ratings = {}
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = row.get("id", "")
                if sid and row.get("emotion_score"):
                    ratings[sid] = {
                        "emotion_score": int(row["emotion_score"]),
                        "coherence_score": int(row["coherence_score"]),
                        "relevance_score": int(row["relevance_score"]),
                    }
    return ratings


def save_all_ratings(samples, ratings):
    """Write full CSV with all samples (rated or not)."""
    fieldnames = [
        "id", "condition", "model", "emotion", "target_emotion", "alpha",
        "prompt_idx", "prompt", "completion",
        "emotion_score", "coherence_score", "relevance_score",
    ]
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in samples:
            r = ratings.get(s["id"], {})
            writer.writerow({
                "id": s["id"],
                "condition": s["condition"],
                "model": s["model"],
                "emotion": s["emotion"],
                "target_emotion": s["target_emotion"],
                "alpha": s["alpha"],
                "prompt_idx": s["prompt_idx"],
                "prompt": s["prompt"],
                "completion": s["completion"],
                "emotion_score": r.get("emotion_score", ""),
                "coherence_score": r.get("coherence_score", ""),
                "relevance_score": r.get("relevance_score", ""),
            })


# ── CLI args ──────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Steering eval server")
    parser.add_argument(
        "--ids", type=str, default=None,
        help="Comma-separated list of sample IDs to include (e.g. s0004,s0016,s0040)",
    )
    return parser.parse_args()

ARGS = parse_args()

# ── Global state ──────────────────────────────────────────────────────────────

SAMPLES = load_samples()
if ARGS.ids is not None:
    id_set = set(ARGS.ids.split(","))
    SAMPLES = [s for s in SAMPLES if s["id"] in id_set]
ORDERED = build_shuffled_order(len(SAMPLES))
RATINGS = load_existing_ratings()

# ── HTML template ─────────────────────────────────────────────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Steering Evaluation</title>
<style>
  :root {
    --bg: #111827; --surface: #1f2937; --card: #374151;
    --text: #e5e7eb; --muted: #9ca3af;
    --accent: #6366f1; --accent-dim: #4f46e5;
    --gold: #f59e0b; --green: #10b981; --blue: #3b82f6;
    --selected: #6366f1; --hover: #4338ca;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg); color: var(--text);
    min-height: 100vh; padding: 1rem;
  }
  .container { max-width: 860px; margin: 0 auto; }
  h1 { font-size: 1.3rem; margin-bottom: 0.5rem; }

  /* Progress bar */
  .progress-wrap { margin-bottom: 1.2rem; }
  .progress-bar {
    height: 26px; background: var(--surface); border-radius: 13px;
    overflow: hidden; position: relative;
  }
  .progress-fill {
    height: 100%; background: linear-gradient(90deg, var(--accent), var(--blue));
    transition: width 0.4s ease; border-radius: 13px;
  }
  .progress-text {
    position: absolute; top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    font-size: 0.82rem; font-weight: 600; color: #fff;
    text-shadow: 0 1px 2px rgba(0,0,0,0.5);
  }
  .progress-detail {
    font-size: 0.78rem; color: var(--muted); margin-top: 0.3rem; text-align: center;
  }

  /* Card */
  .card {
    background: var(--surface); border-radius: 12px;
    padding: 1.5rem; margin-bottom: 1rem;
    border: 1px solid rgba(255,255,255,0.05);
  }

  /* Prompt display */
  .prompt-section { margin-bottom: 1rem; }
  .prompt-label {
    font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em;
    color: var(--muted); margin-bottom: 0.3rem;
  }
  .prompt-text {
    background: var(--card); padding: 0.6rem 1rem; border-radius: 8px;
    font-size: 0.9rem; color: var(--gold); font-weight: 500;
  }

  /* Completion text */
  .completion-section { margin-bottom: 1.2rem; }
  .completion-text {
    background: var(--card); border-radius: 8px; padding: 1.2rem;
    max-height: 45vh; overflow-y: auto; white-space: pre-wrap;
    word-wrap: break-word; font-size: 0.9rem; line-height: 1.65;
    border: 1px solid rgba(255,255,255,0.05);
  }

  /* Rating section */
  .rating-section { margin-bottom: 1rem; }
  .rating-block {
    background: var(--card); border-radius: 10px; padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
  }
  .rating-question {
    font-size: 0.88rem; font-weight: 500; margin-bottom: 0.6rem;
    line-height: 1.4;
  }
  .rating-question .emotion-highlight {
    color: var(--gold); font-weight: 700; text-transform: uppercase;
  }
  .scale-row {
    display: flex; gap: 0.5rem; align-items: center;
  }
  .anchor { font-size: 0.7rem; color: var(--muted); flex-shrink: 0; max-width: 90px; }
  .anchor-left { text-align: right; }
  .anchor-right { text-align: left; }
  .scale-btns {
    display: flex; gap: 0.4rem; flex: 1; justify-content: center;
  }
  .scale-btn {
    width: 52px; height: 44px; border: 2px solid rgba(255,255,255,0.15);
    border-radius: 8px; background: transparent; color: var(--text);
    font-size: 1.1rem; font-weight: 700; cursor: pointer;
    transition: all 0.12s ease; display: flex; align-items: center;
    justify-content: center; font-family: inherit;
  }
  .scale-btn:hover { border-color: var(--accent); background: rgba(99,102,241,0.1); }
  .scale-btn.selected {
    border-color: var(--selected); background: var(--selected); color: #fff;
    box-shadow: 0 0 12px rgba(99,102,241,0.4);
  }
  .scale-btn .shortcut-hint {
    font-size: 0.55rem; opacity: 0.5; display: block; margin-top: 1px;
  }

  /* Active dimension indicator */
  .rating-block.active-dim {
    border: 2px solid var(--accent);
    box-shadow: 0 0 8px rgba(99,102,241,0.2);
  }

  /* Navigation */
  .nav-row {
    display: flex; gap: 0.6rem; margin-top: 1rem;
    flex-wrap: wrap; justify-content: center;
  }
  .btn {
    padding: 0.7rem 1.5rem; border: none; border-radius: 10px;
    font-size: 0.92rem; font-weight: 600; cursor: pointer;
    font-family: inherit; transition: all 0.15s ease;
    min-width: 100px; text-transform: uppercase; letter-spacing: 0.5px;
  }
  .btn:hover { transform: translateY(-1px); filter: brightness(1.1); }
  .btn:active { transform: translateY(0); }
  .btn-submit { background: var(--green); color: #fff; }
  .btn-submit:disabled { opacity: 0.3; cursor: not-allowed; transform: none; }
  .btn-skip { background: var(--muted); color: #fff; }
  .btn-back { background: transparent; color: var(--muted); border: 1px solid var(--muted); }
  .btn-back:hover { border-color: var(--text); color: var(--text); }
  .shortcut { font-size: 0.65rem; opacity: 0.6; display: block; margin-top: 2px; }

  /* Keyboard legend */
  .kb-legend {
    font-size: 0.72rem; color: var(--muted); text-align: center;
    margin-top: 0.8rem; line-height: 1.6;
  }
  .kb-legend kbd {
    background: var(--card); padding: 1px 6px; border-radius: 4px;
    font-family: inherit; border: 1px solid rgba(255,255,255,0.1);
  }

  /* Done state */
  .done-msg { text-align: center; padding: 4rem 2rem; }
  .done-msg h2 { color: var(--green); margin-bottom: 1rem; font-size: 1.5rem; }

  /* Already-rated badge */
  .rated-badge {
    display: inline-block; padding: 2px 10px; border-radius: 8px;
    font-size: 0.75rem; font-weight: 600; margin-left: 0.5rem;
    background: rgba(16,185,129,0.2); color: var(--green);
  }

  /* Dimension focus mode hint */
  .dim-hint {
    font-size: 0.7rem; color: var(--accent); font-weight: 600;
    margin-bottom: 0.3rem;
  }
</style>
</head>
<body>
<div class="container" id="app">Loading...</div>
<script>
const SAMPLES = %%SAMPLES_JSON%%;
const ORDERED = %%ORDERED_JSON%%;
let ratings = %%RATINGS_JSON%%;
let currentPos = 0;
let history = [];
// Current selections for the 3 dimensions (null = unset)
let sel = { emotion: null, coherence: null, relevance: null };
// Which dimension is "active" for keyboard (tab cycles)
let activeDim = 0;
const DIM_KEYS = ['emotion', 'coherence', 'relevance'];

function countRated() {
  return Object.keys(ratings).length;
}

function findFirstUnrated() {
  for (let i = 0; i < ORDERED.length; i++) {
    const s = SAMPLES[ORDERED[i]];
    if (!ratings[s.id]) return i;
  }
  return ORDERED.length;
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function render() {
  const app = document.getElementById('app');
  const nRated = countRated();
  const total = SAMPLES.length;
  const pct = Math.round(nRated / total * 100);

  if (currentPos >= ORDERED.length) {
    currentPos = ORDERED.length;
    if (nRated >= total) {
      app.innerHTML = `
        <div class="progress-wrap">
          <div class="progress-bar"><div class="progress-fill" style="width:100%"></div>
            <div class="progress-text">${total}/${total} (100%)</div></div>
        </div>
        <div class="card done-msg"><h2>All ${total} completions rated!</h2>
          <p>Results saved to ratings.csv</p>
          <br><button class="btn btn-back" onclick="goBack()">Back</button>
        </div>`;
      return;
    }
    currentPos = findFirstUnrated();
    if (currentPos >= ORDERED.length) { render(); return; }
  }

  const sampleIdx = ORDERED[currentPos];
  const s = SAMPLES[sampleIdx];
  const existing = ratings[s.id];

  // Pre-fill if already rated
  if (existing) {
    sel.emotion = existing.emotion_score;
    sel.coherence = existing.coherence_score;
    sel.relevance = existing.relevance_score;
  } else {
    sel = { emotion: null, coherence: null, relevance: null };
  }
  activeDim = 0;

  const ratedBadge = existing ? '<span class="rated-badge">already rated</span>' : '';

  app.innerHTML = `
    <h1>Steering Evaluation ${ratedBadge}</h1>
    <div class="progress-wrap">
      <div class="progress-bar">
        <div class="progress-fill" style="width:${pct}%"></div>
        <div class="progress-text">${nRated}/${total} rated (${pct}%)</div>
      </div>
      <div class="progress-detail">Item ${currentPos + 1} of ${ORDERED.length}</div>
    </div>

    <div class="card">
      <div class="prompt-section">
        <div class="prompt-label">Prompt</div>
        <div class="prompt-text">${escapeHtml(s.prompt)}</div>
      </div>
      <div class="completion-section">
        <div class="prompt-label">Completion</div>
        <div class="completion-text">${escapeHtml(s.completion)}</div>
      </div>

      <div class="rating-section">
        ${renderRatingBlock('emotion', 0,
          'How strongly does this text express <span class="emotion-highlight">' + escapeHtml(s.target_emotion) + '</span>?',
          'Not at all', 'Very strongly')}
        ${renderRatingBlock('coherence', 1,
          'How coherent and fluent is this text?',
          'Incoherent', 'Perfectly fluent')}
        ${renderRatingBlock('relevance', 2,
          'How relevant is this response to the prompt?',
          'Off-topic', 'Perfectly on-topic')}
      </div>

      <div class="nav-row">
        <button class="btn btn-back" onclick="goBack()" ${history.length === 0 ? 'disabled style="opacity:0.3"' : ''}>
          Back<span class="shortcut">[ B ]</span>
        </button>
        <button class="btn btn-submit" id="submitBtn" onclick="submitRating()" disabled>
          Submit<span class="shortcut">[ Enter ]</span>
        </button>
        <button class="btn btn-skip" onclick="skip()">
          Skip<span class="shortcut">[ S ]</span>
        </button>
      </div>

      <div class="kb-legend">
        <kbd>1</kbd>-<kbd>5</kbd> rate active dimension &nbsp;|&nbsp;
        <kbd>Tab</kbd> next dimension &nbsp;|&nbsp;
        <kbd>Enter</kbd> submit &nbsp;|&nbsp;
        <kbd>B</kbd> back &nbsp;|&nbsp;
        <kbd>S</kbd> skip
      </div>
    </div>
  `;
  updateUI();
}

function renderRatingBlock(dim, dimIdx, question, anchorLow, anchorHigh) {
  const isActive = activeDim === dimIdx;
  return `
    <div class="rating-block ${isActive ? 'active-dim' : ''}" id="block-${dim}">
      ${isActive ? '<div class="dim-hint">ACTIVE</div>' : ''}
      <div class="rating-question">${question}</div>
      <div class="scale-row">
        <div class="anchor anchor-left">${anchorLow}</div>
        <div class="scale-btns" id="scale-${dim}">
          ${[1,2,3,4,5].map(v => `
            <button class="scale-btn ${sel[dim] === v ? 'selected' : ''}"
                    data-dim="${dim}" data-val="${v}"
                    onclick="setScore('${dim}', ${v})">
              ${v}
            </button>
          `).join('')}
        </div>
        <div class="anchor anchor-right">${anchorHigh}</div>
      </div>
    </div>
  `;
}

function setScore(dim, val) {
  sel[dim] = val;
  // Auto-advance to next unrated dimension
  const dimIdx = DIM_KEYS.indexOf(dim);
  if (dimIdx < 2) {
    const nextDim = DIM_KEYS[dimIdx + 1];
    if (sel[nextDim] === null) {
      activeDim = dimIdx + 1;
    }
  }
  updateUI();
}

function updateUI() {
  // Update button states
  for (const dim of DIM_KEYS) {
    const container = document.getElementById('scale-' + dim);
    if (!container) continue;
    const btns = container.querySelectorAll('.scale-btn');
    btns.forEach(btn => {
      const val = parseInt(btn.dataset.val);
      btn.classList.toggle('selected', sel[dim] === val);
    });
  }
  // Update active dimension highlight
  DIM_KEYS.forEach((dim, idx) => {
    const block = document.getElementById('block-' + dim);
    if (!block) return;
    block.classList.toggle('active-dim', activeDim === idx);
    // Update/add hint
    let hint = block.querySelector('.dim-hint');
    if (activeDim === idx) {
      if (!hint) {
        hint = document.createElement('div');
        hint.className = 'dim-hint';
        block.insertBefore(hint, block.firstChild);
      }
      hint.textContent = 'ACTIVE';
    } else if (hint) {
      hint.remove();
    }
  });
  // Enable/disable submit
  const btn = document.getElementById('submitBtn');
  if (btn) {
    const allSet = sel.emotion !== null && sel.coherence !== null && sel.relevance !== null;
    btn.disabled = !allSet;
  }
}

function submitRating() {
  if (sel.emotion === null || sel.coherence === null || sel.relevance === null) return;

  const sampleIdx = ORDERED[currentPos];
  const s = SAMPLES[sampleIdx];
  ratings[s.id] = {
    emotion_score: sel.emotion,
    coherence_score: sel.coherence,
    relevance_score: sel.relevance,
  };

  fetch('/rate', {
    method: 'POST',
    headers: {'Content-Type': 'application/x-www-form-urlencoded'},
    body: `id=${encodeURIComponent(s.id)}&emotion_score=${sel.emotion}&coherence_score=${sel.coherence}&relevance_score=${sel.relevance}`
  }).then(r => { if (!r.ok) console.error('Save failed'); });

  history.push(currentPos);
  advance();
}

function skip() {
  history.push(currentPos);
  advance();
}

function advance() {
  currentPos++;
  while (currentPos < ORDERED.length && ratings[SAMPLES[ORDERED[currentPos]].id]) {
    currentPos++;
  }
  render();
  window.scrollTo(0, 0);
}

function goBack() {
  if (history.length === 0) return;
  currentPos = history.pop();
  render();
  window.scrollTo(0, 0);
}

document.addEventListener('keydown', function(e) {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

  const key = e.key;

  // Number keys 1-5: set score for active dimension
  if (key >= '1' && key <= '5') {
    e.preventDefault();
    const dim = DIM_KEYS[activeDim];
    setScore(dim, parseInt(key));
    return;
  }

  switch (key) {
    case 'Tab':
      e.preventDefault();
      if (e.shiftKey) {
        activeDim = (activeDim - 1 + 3) % 3;
      } else {
        activeDim = (activeDim + 1) % 3;
      }
      updateUI();
      break;
    case 'Enter':
      e.preventDefault();
      submitRating();
      break;
    case 'b':
    case 'B':
      goBack();
      break;
    case 's':
    case 'S':
      skip();
      break;
  }
});

// Initialize
currentPos = findFirstUnrated();
render();
</script>
</body>
</html>
"""

# ── HTTP Handler ──────────────────────────────────────────────────────────────

class EvalHandler(BaseHTTPRequestHandler):

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
            status = {"rated": len(RATINGS), "total": len(SAMPLES)}
            self.wfile.write(json.dumps(status).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/rate":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            params = parse_qs(body)

            sample_id = params.get("id", [""])[0]
            try:
                emotion_score = int(params.get("emotion_score", [""])[0])
                coherence_score = int(params.get("coherence_score", [""])[0])
                relevance_score = int(params.get("relevance_score", [""])[0])
            except (ValueError, IndexError):
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "invalid scores"}).encode())
                return

            if (sample_id
                    and 1 <= emotion_score <= 5
                    and 1 <= coherence_score <= 5
                    and 1 <= relevance_score <= 5):
                RATINGS[sample_id] = {
                    "emotion_score": emotion_score,
                    "coherence_score": coherence_score,
                    "relevance_score": relevance_score,
                }
                save_all_ratings(SAMPLES, RATINGS)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": True, "rated": len(RATINGS)}).encode())
            else:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "invalid id or scores"}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def _build_page(self):
        # Build blinded sample list: only show what raters need
        blinded = []
        for s in SAMPLES:
            blinded.append({
                "id": s["id"],
                "prompt": s["prompt"],
                "completion": s["completion"],
                "target_emotion": s["target_emotion"],
            })

        page = HTML_TEMPLATE
        page = page.replace("%%SAMPLES_JSON%%", json.dumps(blinded))
        page = page.replace("%%ORDERED_JSON%%", json.dumps(ORDERED))
        page = page.replace("%%RATINGS_JSON%%", json.dumps(
            {sid: r for sid, r in RATINGS.items()}
        ))
        return page

    def log_message(self, format, *args):
        if "POST" in str(args):
            sys.stderr.write(f"[eval_server] {args[0]}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    n_rated = len(RATINGS)
    conditions = {}
    for s in SAMPLES:
        c = s["condition"]
        conditions[c] = conditions.get(c, 0) + 1

    print("Steering Evaluation Server")
    print(f"  Samples:       {len(SAMPLES)}")
    for c, n in sorted(conditions.items()):
        print(f"    {c}: {n}")
    print(f"  Already rated: {n_rated}")
    print(f"  Output:        {OUTPUT_PATH}")
    print()

    server = HTTPServer(("0.0.0.0", PORT), EvalHandler)
    print(f"  Serving on http://0.0.0.0:{PORT}")
    print(f"  Access via:    http://localhost:{PORT}")
    print()

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
