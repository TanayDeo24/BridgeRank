import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

# scripts/ is not an installed package. ProcessPoolExecutor (spawn start
# method, the macOS/Windows default) pickles worker functions by
# module-qualified name and re-imports that module by name in each fresh
# child interpreter, using the *parent's* sys.path snapshot taken at pool
# creation time -- so scripts/ must be a real, importable module on
# sys.path (a dynamically exec'd module registered only in the parent's
# sys.modules is invisible to the spawned children and raises
# ModuleNotFoundError there).
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import parallel_runner


def _write_config(tmp_path: Path, config_id: str, **overrides) -> Path:
    body = {
        "id": config_id,
        "embedding_model": "hashing",
        "index_type": "flat",
        "candidate_k": 20,
        "rerank_depth": 0,
        "top_k": 3,
        "retrieval_features": {"bm25": 0.5, "dense": 0.5},
        "fusion": "weighted_sum",
        "normalization": "minmax",
        "split": "dev",
    }
    body.update(overrides)
    path = tmp_path / f"{config_id}.json"
    path.write_text(json.dumps(body))
    return path


def test_run_one_succeeds_on_a_valid_config(tmp_path):
    config_path = _write_config(tmp_path, "ok-config")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    outcome = parallel_runner._run_one(str(config_path), "dev", str(out_dir))
    assert outcome.success is True
    assert outcome.summary is not None
    assert (out_dir / "ok-config.json").exists()


def test_run_one_captures_failure_from_a_missing_config_file(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    missing_path = tmp_path / "does_not_exist.json"
    outcome = parallel_runner._run_one(str(missing_path), "dev", str(out_dir))
    assert outcome.success is False
    assert outcome.error is not None
    assert outcome.config_id == "does_not_exist"


def test_run_one_captures_failure_from_unknown_fusion_error_path(tmp_path):
    # a config with a learned_linear fusion but no model path raises inside
    # score_query on the very first query; evaluate_config's own per-query
    # try/except absorbs that into error_rate=1.0 rather than a hard crash --
    # verify that behavior explicitly here since it looks like a failure but
    # is actually reported as a (successful run, 100% error rate) row.
    config_path = _write_config(tmp_path, "bad-fusion", fusion="learned_linear", learned_model_path=None)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    outcome = parallel_runner._run_one(str(config_path), "dev", str(out_dir))
    assert outcome.success is True
    assert outcome.summary["error_rate"] == 1.0


def test_sequential_and_parallel_runs_produce_identical_relevance_metrics(tmp_path):
    config_paths = [_write_config(tmp_path, f"cfg{i}", top_k=3) for i in range(3)]
    seq_outcomes, _ = parallel_runner.run_manifest(config_paths, "dev", workers=1, out_dir=tmp_path / "seq")
    par_outcomes, _ = parallel_runner.run_manifest(config_paths, "dev", workers=2, out_dir=tmp_path / "par")

    seq_by_id = {o.config_id: o.summary["ndcg_at_k"] for o in seq_outcomes}
    par_by_id = {o.config_id: o.summary["ndcg_at_k"] for o in par_outcomes}
    assert seq_by_id == par_by_id
    assert all(o.success for o in seq_outcomes)
    assert all(o.success for o in par_outcomes)


def test_write_comparison_produces_json_and_csv_with_one_row_per_config(tmp_path):
    config_paths = [_write_config(tmp_path, f"cfg{i}") for i in range(2)]
    outcomes, wall = parallel_runner.run_manifest(config_paths, "dev", workers=1, out_dir=tmp_path / "run")
    report = parallel_runner.write_comparison(outcomes, workers=1, wall_clock=wall, out_dir=tmp_path)
    assert report["config_count"] == 2
    assert (tmp_path / "comparison_workers_1.json").exists()
    assert (tmp_path / "comparison_workers_1.csv").exists()
