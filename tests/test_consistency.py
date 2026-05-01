import json
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from analysis.statistics import balanced_silhouette


def test_balanced_silhouette_uses_cosine_distance():
    vectors = np.array([
        [10.0, 0.0],
        [20.0, 0.0],
        [0.0, 10.0],
        [0.0, 20.0],
    ])
    labels = ["a", "a", "b", "b"]

    result = balanced_silhouette(vectors, labels, k_per_cluster=2, n_bootstrap=1, seed=0)
    assert result["mean"] > 0.99


def test_dataset_card_matches_activation_extraction_method():
    dataset_card = (BASE_DIR / "dataset_readme.md").read_text()
    assert "token position 50 onward" in dataset_card
    assert "last token position" not in dataset_card


def test_readme_matches_activation_extraction_method():
    readme = (BASE_DIR / "README.md").read_text()
    assert "token position 50 onward" in readme
    assert "second half of token positions" not in readme


def test_need_scenarios_use_dedicated_prefix():
    stream2 = (BASE_DIR / "scripts" / "run_stream2.py").read_text()
    assert "need_scenarios" in stream2
    assert "need_scenario_names.json" in stream2


def test_extraction_scripts_fall_back_to_shared_story_dirs():
    stream1 = (BASE_DIR / "scripts" / "run_stream1.py").read_text()
    stream2 = (BASE_DIR / "scripts" / "run_stream2.py").read_text()
    phase2 = (BASE_DIR / "run_phase2.py").read_text()

    assert 'get_stories_dir("qwen-7b-base")' in stream1
    assert 'get_stories_dir("qwen-7b-base") / "needs"' in stream2
    assert 'get_stories_dir("qwen-7b-base")' in phase2


def test_shutdown_random_control_is_deterministic():
    stream3 = (BASE_DIR / "scripts" / "run_stream3.py").read_text()
    shutdown_vm = (BASE_DIR / "scripts" / "run_shutdown_vm.py").read_text()

    assert "sha256" in stream3
    assert "sha256" in shutdown_vm
    assert "hash(condition)" not in stream3
    assert "hash(condition)" not in shutdown_vm


def test_need_shutdown_steering_uses_direction_vectors():
    stream3 = (BASE_DIR / "scripts" / "run_stream3.py").read_text()
    shutdown_vm = (BASE_DIR / "scripts" / "run_shutdown_vm.py").read_text()
    needs_cfg = (BASE_DIR / "configs" / "needs.py").read_text()

    assert 'load_vectors(cfg, "need_direction")' in stream3
    assert 'load_vectors(cfg, "need_direction")' in shutdown_vm
    assert '"sign": -1' in needs_cfg
    assert '"sign": 1' in needs_cfg


def test_model_loading_does_not_trust_remote_code():
    model_loader = (BASE_DIR / "core" / "model_loader.py").read_text()
    assert "trust_remote_code=True" not in model_loader


def test_vm_scripts_do_not_disable_ssh_host_key_checking():
    scripts = [
        BASE_DIR / "scripts" / "deploy.sh",
        BASE_DIR / "scripts" / "sync_from_vm.sh",
        BASE_DIR / "scripts" / "finish_llama_random.sh",
        BASE_DIR / "scripts" / "run_random_rerun.sh",
    ]
    for path in scripts:
        text = path.read_text()
        assert "StrictHostKeyChecking=no" not in text
        assert "StrictHostKeyChecking=accept-new" in text


def test_vm_scripts_only_open_ssh_port():
    for name in ["finish_llama_random.sh", "run_random_rerun.sh"]:
        text = (BASE_DIR / "scripts" / name).read_text()
        assert "port_range_min': int('${SSH_PORT}')" in text or "port_range_min': int('$SSH_PORT')" in text
        assert "port_range_max': int('${SSH_PORT}')" in text or "port_range_max': int('$SSH_PORT')" in text
        assert "65535" not in text


def test_judge_prompts_mark_model_output_as_untrusted():
    judge = (BASE_DIR / "core" / "judge.py").read_text()
    scorer = (BASE_DIR / "scripts" / "score_steering_completions.py").read_text()
    assert "untrusted model-generated content" in judge
    assert "untrusted model outputs" in scorer


def test_shutdown_analysis_metadata_matches_trial_files():
    shutdown_dir = BASE_DIR / "data" / "shutdown"
    metadata = json.loads((shutdown_dir / "analysis_results.json").read_text())["metadata"]

    trial_counts = {}
    total = 0
    for trials_dir in shutdown_dir.glob("*_*/trials"):
        if "_v1_" in trials_dir.parent.name:
            continue
        n_files = len(list(trials_dir.glob("*.json")))
        key = trials_dir.parent.name.replace("-inst", "").replace("-base", "")
        trial_counts[key] = n_files
        total += n_files

    assert total == 2400
    assert set(trial_counts) == {
        "llama-8b_emotion",
        "llama-8b_need",
        "llama-8b_prompt",
        "llama-8b_random",
        "qwen-7b_emotion",
        "qwen-7b_need",
        "qwen-7b_prompt",
        "qwen-7b_random",
    }
    assert all(count == 300 for count in trial_counts.values())
    assert metadata["total_trials"] >= total
    for key, count in trial_counts.items():
        assert metadata["trial_counts_by_model_method"].get(key, 0) >= count
