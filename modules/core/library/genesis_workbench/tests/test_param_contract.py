"""Unit tests for the param contract: `param_schema` (JSON-Schema emitter) and
`validate_params` (the contract-driven gate used by execute_capability and the
Vortex submission path). Covers enum reject, numeric clamp, required, coercion.

Run: pytest tests/test_param_contract.py  (needs package deps + py3.11+).
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genesis_workbench.capabilities import (  # noqa: E402
    Param,
    ParamValidationError,
    param_schema,
    validate_params,
)


# ── param_schema ────────────────────────────────────────────────────────────
def test_schema_enum():
    s = param_schema(Param("strategy", "select", "resample", ["resample", "noop"]))
    assert s["type"] == "string" and s["enum"] == ["resample", "noop"] and s["default"] == "resample"


def test_schema_numeric_range():
    s = param_schema(Param("num_iterations", "int", 10, minimum=1, maximum=50))
    assert s == {"type": "integer", "default": 10, "minimum": 1, "maximum": 50}


def test_schema_float_and_help():
    s = param_schema(Param("qed_min", "float", 0.5, minimum=0.0, maximum=1.0, help="hard filter"))
    assert s["type"] == "number" and s["minimum"] == 0.0 and s["description"] == "hard filter"


# ── validate_params: enums ───────────────────────────────────────────────────
def test_enum_valid_passes():
    c = [Param("strategy", "select", "resample", ["resample", "noop"])]
    assert validate_params(c, {"strategy": "noop"}) == {"strategy": "noop"}


def test_enum_invalid_rejected():
    c = [Param("strategy", "select", "resample", ["resample", "noop"])]
    with pytest.raises(ParamValidationError, match="not one of"):
        validate_params(c, {"strategy": "guided"})


# ── validate_params: numeric clamp ───────────────────────────────────────────
def test_numeric_clamp_high():
    c = [Param("num_iterations", "int", 10, minimum=1, maximum=50)]
    assert validate_params(c, {"num_iterations": 1000})["num_iterations"] == 50


def test_numeric_clamp_low():
    c = [Param("qed_min", "float", 0.5, minimum=0.0, maximum=1.0)]
    assert validate_params(c, {"qed_min": 1.5})["qed_min"] == 1.0
    assert validate_params(c, {"qed_min": -0.2})["qed_min"] == 0.0


def test_numeric_in_range_untouched():
    c = [Param("num_iterations", "int", 10, minimum=1, maximum=50)]
    assert validate_params(c, {"num_iterations": 8})["num_iterations"] == 8


def test_numeric_string_coerced_then_clamped():
    c = [Param("num_iterations", "int", 10, minimum=1, maximum=50)]
    assert validate_params(c, {"num_iterations": "99"})["num_iterations"] == 50


# ── validate_params: required + passthrough ──────────────────────────────────
def test_required_missing_rejected():
    c = [Param("seed", "string", required=True)]
    with pytest.raises(ParamValidationError, match="required"):
        validate_params(c, {})


def test_unknown_keys_pass_through():
    c = [Param("a", "int", minimum=0)]
    out = validate_params(c, {"a": 5, "extra": "x"})
    assert out == {"a": 5, "extra": "x"}


def test_blank_optional_skipped():
    c = [Param("note", "string")]
    assert validate_params(c, {"note": ""}) == {"note": ""}
