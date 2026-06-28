#!/usr/bin/env python3
"""Unit tests for the current StageManager API."""

import logging
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.core.pipeline.stage_manager import PipelineStage, StageManager, StageStatus


def make_manager(tmp_path=None):
    config = {"stage_settings": {"flag": True}}
    return StageManager(project="demo", config=config, logger=logging.getLogger("test_stage_manager"))


def test_pipeline_stage_defaults():
    def execute(**kwargs):
        return True

    stage = PipelineStage(name="sample", description="Sample", execute_fn=execute)

    assert stage.name == "sample"
    assert stage.dependencies == []
    assert stage.enabled is True
    assert stage.optional is False
    assert stage.status == StageStatus.PENDING
    assert stage.result_metadata == {}


def test_compute_execution_order_linear_chain():
    manager = make_manager()

    manager.register_stage(PipelineStage("extract", "Extract", lambda **_: True))
    manager.register_stage(PipelineStage("detect", "Detect", lambda **_: True, dependencies=["extract"]))
    manager.register_stage(PipelineStage("segment", "Segment", lambda **_: True, dependencies=["detect"]))

    assert manager.compute_execution_order() == ["extract", "detect", "segment"]


def test_compute_execution_order_diamond():
    manager = make_manager()

    manager.register_stage(PipelineStage("A", "A", lambda **_: True))
    manager.register_stage(PipelineStage("B", "B", lambda **_: True, dependencies=["A"]))
    manager.register_stage(PipelineStage("C", "C", lambda **_: True, dependencies=["A"]))
    manager.register_stage(PipelineStage("D", "D", lambda **_: True, dependencies=["B", "C"]))

    order = manager.compute_execution_order()

    assert order[0] == "A"
    assert order[-1] == "D"
    assert set(order[1:3]) == {"B", "C"}


def test_unknown_dependency_raises():
    manager = make_manager()
    manager.register_stage(PipelineStage("broken", "Broken", lambda **_: True, dependencies=["missing"]))

    try:
        manager.compute_execution_order()
    except ValueError as exc:
        assert "depends on unknown stage" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown dependency")


def test_circular_dependency_raises():
    manager = make_manager()
    manager.register_stage(PipelineStage("A", "A", lambda **_: True, dependencies=["B"]))
    manager.register_stage(PipelineStage("B", "B", lambda **_: True, dependencies=["A"]))

    try:
        manager.compute_execution_order()
    except ValueError as exc:
        assert "Circular dependency" in str(exc)
    else:
        raise AssertionError("Expected circular dependency error")


def test_execute_stage_success_records_metadata():
    manager = make_manager()

    def execute(project, config, stage_config, logger):
        assert project == "demo"
        assert config["stage_settings"]["flag"] is True
        assert stage_config == {"flag": True}
        return {"success": True, "items_processed": 3}

    manager.register_stage(
        PipelineStage(
            name="stage",
            description="Stage",
            execute_fn=execute,
            config_key="stage_settings",
        )
    )

    assert manager.execute_stage("stage") is True
    stage = manager.stages["stage"]
    assert stage.status == StageStatus.COMPLETED
    assert stage.result_metadata["items_processed"] == 3
    assert stage.start_time is not None
    assert stage.end_time is not None


def test_execute_stage_failure_sets_status():
    manager = make_manager()

    def execute(**kwargs):
        return {"success": False, "error": "boom"}

    manager.register_stage(PipelineStage("stage", "Stage", execute))

    assert manager.execute_stage("stage") is False
    assert manager.stages["stage"].status == StageStatus.FAILED
    assert "boom" in manager.stages["stage"].error_message


def test_execute_all_skips_failed_dependencies_when_requested():
    manager = make_manager()

    manager.register_stage(PipelineStage("first", "First", lambda **_: {"success": False, "error": "failed"}))
    manager.register_stage(PipelineStage("second", "Second", lambda **_: True, dependencies=["first"]))
    manager.compute_execution_order()

    assert manager.execute_all(skip_failed_dependencies=True) is False
    assert manager.stages["first"].status == StageStatus.FAILED
    assert manager.stages["second"].status == StageStatus.SKIPPED


def test_optional_stage_with_missing_input_is_skipped(tmp_path):
    manager = make_manager()
    missing = tmp_path / "missing.txt"

    manager.register_stage(
        PipelineStage(
            name="optional",
            description="Optional",
            execute_fn=lambda **_: True,
            required_inputs=[missing],
            optional=True,
        )
    )

    assert manager.execute_stage("optional") is True
    assert manager.stages["optional"].status == StageStatus.SKIPPED


def test_summary_and_reset(tmp_path):
    manager = make_manager()
    manager.register_stage(PipelineStage("stage", "Stage", lambda **_: True))
    manager.execute_stage("stage")

    summary = manager.get_pipeline_summary()
    assert summary["project"] == "demo"
    assert summary["status_counts"]["completed"] == 1

    output_path = tmp_path / "summary.json"
    manager.save_summary(output_path)
    assert output_path.exists()

    manager.reset_stages()
    assert manager.stages["stage"].status == StageStatus.PENDING
