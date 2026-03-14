"""Tests para orchestration.pipelines."""

from unittest.mock import patch

import pytest

from orchestration.pipelines import (
    Pipeline,
    PipelineResult,
    StepResult,
    pipeline_step,
)


class TestStepResult:
    """Tests para StepResult."""

    def test_success_true_when_no_error(self):
        result = StepResult(output="ok", duration_seconds=1.0)
        assert result.success is True

    def test_success_false_when_error(self):
        result = StepResult(output=None, duration_seconds=1.0, error="boom")
        assert result.success is False

    def test_to_dict_has_all_keys(self):
        result = StepResult(output="data", duration_seconds=0.5, tokens_used=10)
        d = result.to_dict()
        expected_keys = {
            "output",
            "duration_seconds",
            "tokens_used",
            "cost_usd",
            "error",
            "metadata",
            "success",
        }
        assert set(d.keys()) == expected_keys
        assert d["success"] is True
        assert d["output"] == "data"


class TestPipelineStep:
    """Tests para el decorador pipeline_step."""

    def test_returns_step_result(self):
        @pipeline_step(name="test_step")
        def my_step(x):
            return x * 2

        result = my_step(5)
        assert isinstance(result, StepResult)
        assert result.output == 10
        assert result.success is True

    def test_step_name_attribute(self):
        @pipeline_step(name="named_step")
        def my_step(x):
            return x

        assert my_step.step_name == "named_step"

    @patch("orchestration.pipelines.time.sleep")
    def test_retries_on_failure(self, mock_sleep):
        call_count = 0

        @pipeline_step(name="flaky", max_retries=2, timeout_seconds=5)
        def flaky_step(x):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("not yet")
            return "ok"

        result = flaky_step("input")
        assert result.success is True
        assert result.output == "ok"
        assert call_count == 3
        assert mock_sleep.call_count == 2

    @patch("orchestration.pipelines.time.sleep")
    def test_retries_exhausted_returns_error(self, mock_sleep):
        @pipeline_step(name="always_fail", max_retries=1, timeout_seconds=5)
        def failing_step(x):
            raise RuntimeError("always fails")

        result = failing_step("input")
        assert result.success is False
        assert "always fails" in result.error


class TestPipeline:
    """Tests para Pipeline."""

    def test_sequential_execution(self):
        @pipeline_step(name="add_one", max_retries=0)
        def add_one(x):
            return x + 1

        @pipeline_step(name="double", max_retries=0)
        def double(x):
            return x * 2

        @pipeline_step(name="to_string", max_retries=0)
        def to_string(x):
            return f"result={x}"

        pipe = Pipeline(name="math", steps=[add_one, double, to_string])
        result = pipe.run(5)

        assert result.success is True
        assert result.final_output == "result=12"
        assert len(result.steps) == 3

    @patch("orchestration.pipelines.time.sleep")
    def test_failure_stops_execution(self, mock_sleep):
        calls = []

        @pipeline_step(name="step1", max_retries=0)
        def step1(x):
            calls.append("step1")
            return x

        @pipeline_step(name="step2", max_retries=0)
        def step2(x):
            calls.append("step2")
            raise RuntimeError("step2 fails")

        @pipeline_step(name="step3", max_retries=0)
        def step3(x):
            calls.append("step3")
            return x

        pipe = Pipeline(name="test", steps=[step1, step2, step3])
        result = pipe.run("input")

        assert result.success is False
        assert result.final_output is None
        assert "step1" in calls
        assert "step2" in calls
        assert "step3" not in calls

    def test_summary_contains_info(self):
        @pipeline_step(name="simple", max_retries=0)
        def simple(x):
            return x

        pipe = Pipeline(name="demo", steps=[simple])
        result = pipe.run("data")
        summary = result.summary()

        assert "SUCCESS" in summary
        assert "Steps: 1" in summary
        assert "Step Breakdown:" in summary

    def test_run_from_starts_at_index(self):
        @pipeline_step(name="a", max_retries=0)
        def step_a(x):
            return x + "_a"

        @pipeline_step(name="b", max_retries=0)
        def step_b(x):
            return x + "_b"

        @pipeline_step(name="c", max_retries=0)
        def step_c(x):
            return x + "_c"

        pipe = Pipeline(name="abc", steps=[step_a, step_b, step_c])
        result = pipe.run_from(1, "start")

        assert result.success is True
        assert result.final_output == "start_b_c"
        assert len(result.steps) == 2

    def test_run_from_invalid_index_raises(self):
        @pipeline_step(name="only", max_retries=0)
        def only_step(x):
            return x

        pipe = Pipeline(name="small", steps=[only_step])
        with pytest.raises(ValueError, match="out of range"):
            pipe.run_from(5, "data")

    def test_step_names_property(self):
        @pipeline_step(name="alpha", max_retries=0)
        def step_a(x):
            return x

        @pipeline_step(name="beta", max_retries=0)
        def step_b(x):
            return x

        pipe = Pipeline(name="test", steps=[step_a, step_b])
        assert pipe.step_names == ["alpha", "beta"]
