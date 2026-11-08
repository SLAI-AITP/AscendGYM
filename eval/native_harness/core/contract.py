"""The same JSON schemas are exposed to agents and enforced by the harness."""
import math


def obj(properties, required=None):
    return {"type": "object", "properties": properties,
            "required": list(properties) if required is None else required,
            "additionalProperties": False}


STR = {"type": "string", "minLength": 1}
NULL_STR = {"type": ["string", "null"]}
EDIT = obj({"path": STR, "content": {"type": "string"}})
ARGS = {
    "list_files": obj({"snapshot_id": STR}),
    "read_file": obj({"snapshot_id": STR, "path": STR}),
    "read_files": obj({"snapshot_id": STR, "paths": {"type": "array", "minItems": 1, "items": STR}}),
    "read_spec": obj({}),
    "evaluate_baseline": obj({}),
    "create_candidate": obj({
        "parent_snapshot_id": STR, "repair_from_snapshot_id": NULL_STR,
        "hypothesis": STR, "expected_profile_signal": STR,
        "edits": {"type": "array", "minItems": 1, "items": EDIT},
    }),
    "evaluate_candidate": obj({"candidate_snapshot_id": STR}),
    "checkout_snapshot": obj({"snapshot_id": STR}),
    "record_event": obj({"note": STR}),
    "finish": obj({"summary": STR}),
}
REQUEST_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "oneOf": [obj({
        "schema_version": {"const": "ascend-gym-tool-v2"},
        "request_id": STR, "run_id": STR,
        "tool": {"const": name}, "arguments": args,
    }) for name, args in ARGS.items()],
}
RESPONSE_SCHEMA = obj({
    "schema_version": {"const": "ascend-gym-tool-v2"}, "request_id": STR,
    "run_id": STR, "tool": STR, "status": {"enum": ["ok", "error"]},
    "data": {"type": "object"}, "error": {"type": ["object", "null"]},
    "server_timestamp_utc": STR,
})
EVALUATION_SCHEMA = obj({
    "status": {"enum": ["passed", "compile_failed", "test_failed", "test_mismatch",
                          "profiling_failed", "sandbox_error", "timeout"]},
    "correctness_passed": {"type": ["boolean", "null"]},
    "duration_ms": {"type": ["number", "null"]},
    "profiling_report": {"type": "string"},
    "hardware_metrics": {"type": "object"},
    "diagnostic": {"type": "string"},
    "raw": {"type": "object"},
})


def validate(value, schema, path="$", depth=0):
    """Validate the JSON Schema subset used here; reject unknown object fields."""
    if depth > 64:
        raise ValueError("JSON nesting exceeds 64")
    if "oneOf" in schema:
        matches = 0
        for alternative in schema["oneOf"]:
            try:
                validate(value, alternative, path, depth + 1)
                matches += 1
            except ValueError:
                pass
        if matches != 1:
            raise ValueError(f"{path}: request must match exactly one tool schema")
        return
    if "const" in schema and value != schema["const"]:
        raise ValueError(f"{path}: wrong constant")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path}: invalid enum")
    types = schema.get("type", [])
    if isinstance(types, str):
        types = [types]
    checks = {"object": isinstance(value, dict), "array": isinstance(value, list),
              "string": isinstance(value, str), "boolean": isinstance(value, bool),
              "null": value is None,
              "number": type(value) in (int, float) and math.isfinite(value),
              "integer": type(value) is int}
    if types and not any(checks.get(t, False) for t in types):
        raise ValueError(f"{path}: expected {types}")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        if not set(schema.get("required", [])).issubset(value):
            raise ValueError(f"{path}: missing required fields")
        if schema.get("additionalProperties") is False and set(value) - set(props):
            raise ValueError(f"{path}: unknown fields {sorted(set(value) - set(props))}")
        for key, item in value.items():
            if key in props:
                validate(item, props[key], f"{path}.{key}", depth + 1)
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            raise ValueError(f"{path}: too few entries")
        for i, item in enumerate(value):
            validate(item, schema.get("items", {}), f"{path}[{i}]", depth + 1)
    if isinstance(value, str) and len(value) < schema.get("minLength", 0):
        raise ValueError(f"{path}: empty string")


def checked_evaluation(result):
    validate(result, EVALUATION_SCHEMA)
    runtime = result["duration_ms"]
    if result["status"] == "passed":
        if result["correctness_passed"] is not True or runtime is None or runtime <= 0:
            raise ValueError("passed evaluation requires correctness and positive finite runtime")
        if not result["profiling_report"].strip():
            raise ValueError("passed evaluation requires a profiling report")
    elif runtime is not None:
        raise ValueError("failed evaluation cannot contain a valid runtime")
    return result
